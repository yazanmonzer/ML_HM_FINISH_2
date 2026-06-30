import itertools
import json
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import joblib
import mlflow
import numpy as np
import pandas as pd

from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier

from sklearn.base import clone
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder


RANDOM_STATE = 42
N_SPLITS = 5
ID_COL = "ID"
TARGET_COL = "target"
AVERAGE = "macro"

HERE = Path(__file__).resolve().parent
DATA_DIR = HERE if (HERE / "train_data.csv").exists() else HERE.parent
ARTIFACT_DIR = HERE / "artifacts"
SUBMISSION_DIR = HERE / "submissions"
MLFLOW_DB = HERE / "mlflow.db"
MLFLOW_ARTIFACT_DIR = HERE / "mlartifacts"
EXPERIMENT_NAME = "fite_boosting_best_solution"


ARTIFACT_DIR.mkdir(exist_ok=True)
SUBMISSION_DIR.mkdir(exist_ok=True)
MLFLOW_ARTIFACT_DIR.mkdir(exist_ok=True)


def evaluate_model(name, estimator, X, y, X_test, folds):
    n_classes = len(np.unique(y))
    oof_proba = np.zeros((len(X), n_classes), dtype=float)
    test_probas = []
    fold_scores = []
    fold_rows = []

    for fold, (tr_idx, va_idx) in enumerate(folds, start=1):
        model = clone(estimator)
        model.fit(X.iloc[tr_idx], y[tr_idx])
        va_proba = model.predict_proba(X.iloc[va_idx])
        test_proba = model.predict_proba(X_test)

        va_pred = va_proba.argmax(axis=1)
        oof_proba[va_idx] = va_proba
        test_probas.append(test_proba)
        score = f1_score(y[va_idx], va_pred, average=AVERAGE)
        acc = accuracy_score(y[va_idx], va_pred)
        fold_scores.append(score)
        fold_rows.append({
            "model": name,
            "fold": fold,
            "macro_f1": score,
            "accuracy": acc,
        })
        print(f"{name} fold {fold}: macro_f1={score:.6f}, accuracy={acc:.6f}")

    oof_pred = oof_proba.argmax(axis=1)
    return {
        "name": name,
        "oof_score": f1_score(y, oof_pred, average=AVERAGE),
        "oof_accuracy": accuracy_score(y, oof_pred),
        "fold_mean": float(np.mean(fold_scores)),
        "fold_std": float(np.std(fold_scores, ddof=1)),
        "oof_proba": oof_proba,
        "test_proba": np.mean(test_probas, axis=0),
        "fold_rows": fold_rows,
    }


def best_class_multipliers(y, oof_proba):
    grid = [0.65, 0.75, 0.85, 0.95, 1.0, 1.05, 1.15, 1.3, 1.5, 1.8, 2.2]
    best_score = f1_score(y, oof_proba.argmax(axis=1), average=AVERAGE)
    best_mult = np.ones(oof_proba.shape[1])

    for mult in itertools.product(grid, repeat=oof_proba.shape[1]):
        mult = np.asarray(mult, dtype=float)
        pred = (oof_proba * mult).argmax(axis=1)
        score = f1_score(y, pred, average=AVERAGE)
        if score > best_score:
            best_score = score
            best_mult = mult

    return best_score, best_mult


def compact_params(estimator):
    model = estimator.named_steps["model"]
    keys = [
        "n_estimators",
        "min_samples_leaf",
        "max_features",
        "class_weight",
        "objective",
        "eval_metric",
        "max_depth",
        "learning_rate",
        "min_child_weight",
        "subsample",
        "colsample_bytree",
        "reg_lambda",
        "reg_alpha",
        "num_leaves",
        "min_child_samples",
        "iterations",
        "depth",
        "l2_leaf_reg",
        "auto_class_weights",
    ]
    params = model.get_params()
    return {k: params[k] for k in keys if k in params}


def log_mlflow_run(run_name, output, y, class_names, test_ids, labels, feature_cols,
                   multipliers, fold_rows, estimator_names, model_bundle_path=None):
    oof_pred = (output["oof_proba"] * multipliers).argmax(axis=1)
    metrics = {
        "accuracy": accuracy_score(y, oof_pred),
        "macro_f1": f1_score(y, oof_pred, average="macro"),
        "weighted_f1": f1_score(y, oof_pred, average="weighted"),
        "micro_f1": f1_score(y, oof_pred, average="micro"),
        "raw_oof_macro_f1": output["oof_score"],
        "fold_mean_macro_f1": output.get("fold_mean", np.nan),
        "fold_std_macro_f1": output.get("fold_std", np.nan),
    }

    safe_name = run_name[:120].replace("+", "_").replace(" ", "_")
    submission_path = SUBMISSION_DIR / f"{safe_name}_submission.csv"
    fold_path = ARTIFACT_DIR / f"{safe_name}_fold_metrics.csv"
    pd.DataFrame({ID_COL: test_ids, TARGET_COL: labels}).to_csv(submission_path, index=False)
    pd.DataFrame(fold_rows).to_csv(fold_path, index=False)

    with mlflow.start_run(run_name=run_name):
        mlflow.log_param("run_name", run_name)
        mlflow.log_param("random_state", RANDOM_STATE)
        mlflow.log_param("n_splits", N_SPLITS)
        mlflow.log_param("feature_count", len(feature_cols))
        mlflow.log_param("f1_average", AVERAGE)
        mlflow.log_param("class_names", class_names)
        mlflow.log_param("class_multipliers", multipliers.tolist())
        mlflow.log_param("estimator_names", estimator_names)
        mlflow.log_metrics(metrics)
        mlflow.log_artifact(submission_path, artifact_path="submissions")
        mlflow.log_artifact(fold_path, artifact_path="cv")
        if model_bundle_path is not None:
            mlflow.log_artifact(model_bundle_path, artifact_path="model_bundle")

    return {
        "run_name": run_name,
        **metrics,
        "class_multipliers": multipliers.tolist(),
        "submission_path": str(submission_path),
        "model_bundle_path": str(model_bundle_path) if model_bundle_path else "",
    }


def build_full_model_bundle(model_names, models, X, y, feature_cols, class_names, multipliers):
    fitted = {}
    for name in model_names:
        fitted_model = clone(models[name])
        fitted_model.fit(X, y)
        fitted[name] = fitted_model

    return {
        "feature_cols": feature_cols,
        "class_names": class_names,
        "class_multipliers": multipliers.tolist(),
        "estimators": fitted,
        "preprocessing": "Each estimator is an sklearn Pipeline with SimpleImputer(strategy='median') and the trained model.",
        "prediction_rule": "Average predict_proba outputs, multiply by class_multipliers, then choose argmax.",
    }


def main():
    mlflow.set_tracking_uri(f"sqlite:///{MLFLOW_DB.as_posix()}")
    mlflow.set_experiment(EXPERIMENT_NAME)

    train = pd.read_csv(DATA_DIR / "train_data.csv")
    test = pd.read_csv(DATA_DIR / "test_data.csv")

    feature_cols = [c for c in train.columns if c not in [ID_COL, TARGET_COL]]
    X = train[feature_cols].copy()
    X_test = test[feature_cols].copy()

    encoder = LabelEncoder()
    y = encoder.fit_transform(train[TARGET_COL].astype(str))
    class_names = list(encoder.classes_)

    dataset_info = {
        "train_rows": int(len(train)),
        "test_rows": int(len(test)),
        "feature_count": int(len(feature_cols)),
        "class_distribution": train[TARGET_COL].value_counts().to_dict(),
        "data_dir": str(DATA_DIR.resolve()),
    }
    dataset_info_path = ARTIFACT_DIR / "dataset_info.json"
    dataset_info_path.write_text(json.dumps(dataset_info, ensure_ascii=False, indent=2), encoding="utf-8")

    folds = list(StratifiedKFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE,
    ).split(X, y))

    models = {
        "rf_600_leaf1_balanced": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", RandomForestClassifier(
                n_estimators=600,
                min_samples_leaf=1,
                max_features="sqrt",
                class_weight="balanced_subsample",
                random_state=RANDOM_STATE,
                n_jobs=-1,
            )),
        ]),
        "extra_800_leaf1_balanced": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", ExtraTreesClassifier(
                n_estimators=800,
                min_samples_leaf=1,
                max_features="sqrt",
                class_weight="balanced",
                random_state=RANDOM_STATE,
                n_jobs=-1,
            )),
        ]),
        "xgb_balanced": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", XGBClassifier(
                objective="multi:softprob",
                eval_metric="mlogloss",
                num_class=len(class_names),
                n_estimators=450,
                max_depth=3,
                learning_rate=0.035,
                min_child_weight=2,
                subsample=0.85,
                colsample_bytree=0.85,
                reg_lambda=4.0,
                reg_alpha=0.2,
                random_state=RANDOM_STATE,
                n_jobs=-1,
            )),
        ]),
        "lgbm_balanced": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", LGBMClassifier(
                objective="multiclass",
                n_estimators=500,
                learning_rate=0.035,
                num_leaves=15,
                min_child_samples=12,
                subsample=0.85,
                colsample_bytree=0.85,
                reg_lambda=4.0,
                class_weight="balanced",
                random_state=RANDOM_STATE,
                n_jobs=-1,
                verbose=-1,
            )),
        ]),
        "cat_balanced": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", CatBoostClassifier(
                loss_function="MultiClass",
                iterations=500,
                depth=4,
                learning_rate=0.04,
                l2_leaf_reg=8.0,
                auto_class_weights="Balanced",
                random_seed=RANDOM_STATE,
                verbose=False,
                allow_writing_files=False,
            )),
        ]),
    }

    outputs = []
    mlflow_rows = []
    result_rows = []

    for name, model in models.items():
        print(f"\nRunning {name}")
        out = evaluate_model(name, model, X, y, X_test, folds)
        tuned_score, mult = best_class_multipliers(y, out["oof_proba"])
        out["tuned_score"] = tuned_score
        out["multipliers"] = mult
        outputs.append(out)

        pred = (out["test_proba"] * mult).argmax(axis=1)
        labels = encoder.inverse_transform(pred)
        mlflow_rows.append(log_mlflow_run(
            run_name=name,
            output=out,
            y=y,
            class_names=class_names,
            test_ids=test[ID_COL],
            labels=labels,
            feature_cols=feature_cols,
            multipliers=mult,
            fold_rows=out["fold_rows"],
            estimator_names=[name],
        ))
        result_rows.append({
            "model": name,
            "oof_macro_f1": out["oof_score"],
            "tuned_oof_macro_f1": tuned_score,
            "fold_mean": out["fold_mean"],
            "fold_std": out["fold_std"],
            "multipliers": mult.tolist(),
        })

    base_outputs = list(outputs)
    for combo_size in [2, 3, 4, 5]:
        for combo in itertools.combinations(base_outputs, combo_size):
            names = "+".join(o["name"] for o in combo)
            oof = np.mean([o["oof_proba"] for o in combo], axis=0)
            test_proba = np.mean([o["test_proba"] for o in combo], axis=0)
            score = f1_score(y, oof.argmax(axis=1), average=AVERAGE)
            tuned_score, mult = best_class_multipliers(y, oof)
            outputs.append({
                "name": f"ens_{names}",
                "oof_score": score,
                "tuned_score": tuned_score,
                "multipliers": mult,
                "oof_proba": oof,
                "test_proba": test_proba,
                "fold_mean": np.nan,
                "fold_std": np.nan,
                "fold_rows": [],
                "component_names": [o["name"] for o in combo],
            })
            result_rows.append({
                "model": f"ens_{names}",
                "oof_macro_f1": score,
                "tuned_oof_macro_f1": tuned_score,
                "fold_mean": np.nan,
                "fold_std": np.nan,
                "multipliers": mult.tolist(),
            })

    results = pd.DataFrame(result_rows).sort_values("tuned_oof_macro_f1", ascending=False)
    results_path = ARTIFACT_DIR / "boosting_experiment_results.csv"
    results.to_csv(results_path, index=False)

    print("\nTop results")
    print(results.head(15).to_string(index=False))

    best_name = results.iloc[0]["model"]
    best = next(o for o in outputs if o["name"] == best_name)
    multipliers = np.asarray(results.iloc[0]["multipliers"], dtype=float)
    pred = (best["test_proba"] * multipliers).argmax(axis=1)
    labels = encoder.inverse_transform(pred)

    if best_name.startswith("ens_"):
        component_names = best.get("component_names", [])
    else:
        component_names = [best_name]

    model_bundle = build_full_model_bundle(component_names, models, X, y, feature_cols, class_names, multipliers)
    bundle_path = ARTIFACT_DIR / "best_model_bundle.joblib"
    joblib.dump(model_bundle, bundle_path)

    best_summary_row = log_mlflow_run(
        run_name=f"best_{best_name[:180]}",
        output=best,
        y=y,
        class_names=class_names,
        test_ids=test[ID_COL],
        labels=labels,
        feature_cols=feature_cols,
        multipliers=multipliers,
        fold_rows=best.get("fold_rows", []),
        estimator_names=component_names,
        model_bundle_path=bundle_path,
    )
    mlflow_rows.append(best_summary_row)

    submission_path = SUBMISSION_DIR / f"submission_{best_name[:80].replace('+', '_')}_tuned_macro_f1.csv"
    pd.DataFrame({ID_COL: test[ID_COL], TARGET_COL: labels}).to_csv(submission_path, index=False)

    metadata = {
        "best_model": best_name,
        "best_tuned_oof_macro_f1": float(results.iloc[0]["tuned_oof_macro_f1"]),
        "class_names": class_names,
        "feature_cols": feature_cols,
        "submission_path": str(submission_path),
        "mlflow_tracking_uri": f"sqlite:///{MLFLOW_DB.as_posix()}",
        "mlflow_experiment_name": EXPERIMENT_NAME,
    }
    metadata_path = ARTIFACT_DIR / "boosting_experiment_metadata.json"
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    with mlflow.start_run(run_name="experiment_summary"):
        mlflow.log_artifact(results_path, artifact_path="summary")
        mlflow.log_artifact(metadata_path, artifact_path="summary")
        mlflow.log_artifact(dataset_info_path, artifact_path="data")

    mlflow_summary = pd.DataFrame(mlflow_rows).sort_values("macro_f1", ascending=False)
    summary_path = HERE / "mlflow_experiment_summary.csv"
    mlflow_summary.to_csv(summary_path, index=False)

    print(f"\nBest: {best_name}")
    print(f"Saved: {submission_path}")
    print(f"MLflow summary: {summary_path}")
    print("Start MLflow UI with:")
    print(f"python -m mlflow ui --backend-store-uri sqlite:///{MLFLOW_DB.as_posix()} --host 127.0.0.1 --port 5000")


if __name__ == "__main__":
    main()
