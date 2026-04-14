#!/usr/bin/env python3
"""Validate LT-02 workflow/task visibility scorecards against support contract v1."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List

try:  # pragma: no cover
    from jsonschema import Draft202012Validator, FormatChecker
except Exception:  # pragma: no cover
    Draft202012Validator = None
    FormatChecker = None

EXPECTED_DIMENSIONS = (
    "task_freshness",
    "worker_health",
    "blockage_severity",
    "evidence_quality",
    "execution_path_stability",
)


@dataclass
class Issue:
    code: str
    path: str
    message: str

    def as_dict(self) -> Dict[str, str]:
        return {"code": self.code, "path": self.path, "message": self.message}


def json_ptr(parts: Iterable[Any]) -> str:
    seq = list(parts or [])
    if not seq:
        return "$"
    return "$/" + "/".join(str(p) for p in seq)


def load_json_file(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _rating_for(score: float, healthy_min: float, degraded_min: float) -> str:
    if score >= healthy_min:
        return "healthy"
    if score >= degraded_min:
        return "degraded"
    return "critical"


def semantic_checks(scorecard: Dict[str, Any]) -> List[Issue]:
    issues: List[Issue] = []

    system = scorecard.get("system_scorecard", {}) if isinstance(scorecard.get("system_scorecard"), dict) else {}
    thresholds = system.get("thresholds", {}) if isinstance(system.get("thresholds"), dict) else {}
    healthy_min = float(thresholds.get("healthy_min", 85))
    degraded_min = float(thresholds.get("degraded_min", 60))

    if degraded_min > healthy_min:
        issues.append(
            Issue(
                code="threshold_order_invalid",
                path="$/system_scorecard/thresholds",
                message="degraded_min must be <= healthy_min",
            )
        )

    weights = system.get("weights", {}) if isinstance(system.get("weights"), dict) else {}
    if set(weights.keys()) != set(EXPECTED_DIMENSIONS):
        issues.append(
            Issue(
                code="weights_dimension_mismatch",
                path="$/system_scorecard/weights",
                message="weights keys must match expected LT-02 visibility dimensions",
            )
        )
    else:
        weight_sum = float(sum(float(weights[k]) for k in EXPECTED_DIMENSIONS))
        if abs(weight_sum - 1.0) > 1e-6:
            issues.append(
                Issue(
                    code="weight_sum_out_of_bounds",
                    path="$/system_scorecard/weights",
                    message=f"weights must sum to 1.0 (observed={weight_sum:.6f})",
                )
            )

    system_dimensions = system.get("dimensions", {}) if isinstance(system.get("dimensions"), dict) else {}
    if set(system_dimensions.keys()) != set(EXPECTED_DIMENSIONS):
        issues.append(
            Issue(
                code="system_dimensions_mismatch",
                path="$/system_scorecard/dimensions",
                message="system_scorecard.dimensions keys must match expected LT-02 visibility dimensions",
            )
        )

    task_cards = scorecard.get("task_scorecards", []) if isinstance(scorecard.get("task_scorecards"), list) else []
    task_count = int(system.get("task_count", 0))
    if task_count != len(task_cards):
        issues.append(
            Issue(
                code="task_count_mismatch",
                path="$/system_scorecard/task_count",
                message=f"task_count={task_count} but task_scorecards has {len(task_cards)} entries",
            )
        )

    suppressed_count = int(system.get("suppressed_task_count", 0))
    visible_count = int(system.get("visible_task_count", 0))
    if visible_count + suppressed_count != task_count:
        issues.append(
            Issue(
                code="visible_suppressed_count_mismatch",
                path="$/system_scorecard",
                message="visible_task_count + suppressed_task_count must equal task_count",
            )
        )

    computed_rating_counts = {"healthy": 0, "degraded": 0, "critical": 0}
    computed_dimension_sums = {k: 0.0 for k in EXPECTED_DIMENSIONS}

    for idx, task in enumerate(task_cards):
        if not isinstance(task, dict):
            continue

        task_dimensions = task.get("dimensions", {}) if isinstance(task.get("dimensions"), dict) else {}
        if set(task_dimensions.keys()) != set(EXPECTED_DIMENSIONS):
            issues.append(
                Issue(
                    code="task_dimensions_mismatch",
                    path=f"$/task_scorecards/{idx}/dimensions",
                    message="task dimensions keys must match expected LT-02 visibility dimensions",
                )
            )
            continue

        for key in EXPECTED_DIMENSIONS:
            computed_dimension_sums[key] += float(task_dimensions[key])

        task_score = float(task.get("score", 0.0))
        task_rating = str(task.get("rating", "")).strip().lower()
        expected_task_rating = _rating_for(task_score, healthy_min, degraded_min)
        if task_rating != expected_task_rating:
            issues.append(
                Issue(
                    code="task_rating_threshold_mismatch",
                    path=f"$/task_scorecards/{idx}/rating",
                    message=(
                        f"task rating '{task_rating}' inconsistent with score={task_score} "
                        f"and thresholds healthy_min={healthy_min}, degraded_min={degraded_min}"
                    ),
                )
            )

        if task_rating in computed_rating_counts:
            computed_rating_counts[task_rating] += 1

        if set(weights.keys()) == set(EXPECTED_DIMENSIONS):
            weighted = sum(float(task_dimensions[k]) * float(weights[k]) for k in EXPECTED_DIMENSIONS)
            if abs(weighted - task_score) > 0.05:
                issues.append(
                    Issue(
                        code="task_weighted_score_mismatch",
                        path=f"$/task_scorecards/{idx}/score",
                        message=f"task score={task_score} but weighted dimensions compute to {weighted:.2f}",
                    )
                )

    declared_rating_counts = (
        system.get("task_rating_counts", {}) if isinstance(system.get("task_rating_counts"), dict) else {}
    )
    for rating_key in ("healthy", "degraded", "critical"):
        declared = int(declared_rating_counts.get(rating_key, 0))
        if declared != computed_rating_counts[rating_key]:
            issues.append(
                Issue(
                    code="task_rating_counts_mismatch",
                    path=f"$/system_scorecard/task_rating_counts/{rating_key}",
                    message=(
                        f"declared {rating_key}={declared} but computed {computed_rating_counts[rating_key]} "
                        "from task_scorecards"
                    ),
                )
            )

    if task_cards and set(system_dimensions.keys()) == set(EXPECTED_DIMENSIONS):
        task_len = float(len(task_cards))
        for key in EXPECTED_DIMENSIONS:
            observed_avg = float(system_dimensions[key])
            computed_avg = computed_dimension_sums[key] / task_len
            if abs(observed_avg - computed_avg) > 0.05:
                issues.append(
                    Issue(
                        code="system_dimension_average_mismatch",
                        path=f"$/system_scorecard/dimensions/{key}",
                        message=f"declared average={observed_avg} but computed from tasks={computed_avg:.2f}",
                    )
                )

    if set(weights.keys()) == set(EXPECTED_DIMENSIONS) and set(system_dimensions.keys()) == set(EXPECTED_DIMENSIONS):
        computed_system_score = sum(float(system_dimensions[k]) * float(weights[k]) for k in EXPECTED_DIMENSIONS)
        declared_system_score = float(system.get("score", 0.0))
        if abs(computed_system_score - declared_system_score) > 0.05:
            issues.append(
                Issue(
                    code="system_weighted_score_mismatch",
                    path="$/system_scorecard/score",
                    message=f"declared score={declared_system_score} but weighted dimensions compute to {computed_system_score:.2f}",
                )
            )

    declared_system_rating = str(system.get("rating", "")).strip().lower()
    expected_system_rating = _rating_for(float(system.get("score", 0.0)), healthy_min, degraded_min)
    if declared_system_rating != expected_system_rating:
        issues.append(
            Issue(
                code="system_rating_threshold_mismatch",
                path="$/system_scorecard/rating",
                message=(
                    f"system rating '{declared_system_rating}' inconsistent with score={system.get('score')} "
                    f"and thresholds healthy_min={healthy_min}, degraded_min={degraded_min}"
                ),
            )
        )

    return issues


def validate_scorecard(path: Path, validator: Draft202012Validator) -> Dict[str, Any]:
    try:
        payload = load_json_file(path)
    except Exception as exc:
        return {
            "scorecard_path": str(path),
            "ok": False,
            "error": "scorecard_unreadable",
            "detail": str(exc),
        }

    errors = sorted(
        validator.iter_errors(payload),
        key=lambda err: (list(err.absolute_path), list(err.absolute_schema_path), str(err.message)),
    )
    if errors:
        err = errors[0]
        return {
            "scorecard_path": str(path),
            "ok": False,
            "scorecard_id": payload.get("scorecard_id"),
            "error": "schema_validation_failed",
            "data_path": json_ptr(err.absolute_path),
            "schema_path": json_ptr(err.absolute_schema_path),
            "message": str(err.message),
        }

    issues = semantic_checks(payload)
    if issues:
        return {
            "scorecard_path": str(path),
            "ok": False,
            "scorecard_id": payload.get("scorecard_id"),
            "error": "semantic_validation_failed",
            "issues": [issue.as_dict() for issue in issues],
        }

    return {
        "scorecard_path": str(path),
        "ok": True,
        "scorecard_id": payload.get("scorecard_id"),
    }


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Validate LT-02 workflow/task visibility scorecards")
    parser.add_argument(
        "--schema",
        default=str(repo_root / "docs/ops/schemas/workflow_task_visibility_scorecard.v1.schema.json"),
        help="Path to scorecard schema JSON",
    )
    parser.add_argument(
        "--scorecard",
        action="append",
        required=True,
        help="Path to scorecard JSON (repeat --scorecard for multiple files)",
    )
    parser.add_argument("--pretty", action="store_true", help="Pretty-print JSON output")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if Draft202012Validator is None or FormatChecker is None:
        payload = {"ok": False, "error": "jsonschema_validator_unavailable"}
        print(json.dumps(payload, indent=2 if args.pretty else None))
        return 2

    schema_path = Path(args.schema).expanduser().resolve()
    try:
        schema_doc = load_json_file(schema_path)
    except Exception as exc:
        payload = {
            "ok": False,
            "error": "schema_unreadable",
            "schema_path": str(schema_path),
            "detail": str(exc),
        }
        print(json.dumps(payload, indent=2 if args.pretty else None))
        return 2

    validator = Draft202012Validator(schema_doc, format_checker=FormatChecker())
    scorecards = [Path(p).expanduser().resolve() for p in args.scorecard]
    results = [validate_scorecard(path, validator) for path in scorecards]

    failed = [r for r in results if not r.get("ok")]
    payload = {
        "ok": len(failed) == 0,
        "schema_path": str(schema_path),
        "checked": len(results),
        "failed": len(failed),
        "results": results,
    }
    print(json.dumps(payload, indent=2 if args.pretty else None))
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
