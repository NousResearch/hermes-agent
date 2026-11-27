#!/usr/bin/env python3
"""Score context-compaction summaries against privacy-safe retention fixtures.

The scorer is deliberately provider-neutral and deterministic. A fixture records
facts that a long-session checkpoint must retain (or must not leak); a separate
JSON file supplies one candidate summary per case. This keeps credentials and
LLM calls out of CI while allowing any context engine to be evaluated.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any


SUPPORTED_CHECK_KINDS = frozenset(
    {"contains", "contains_any", "excludes", "excludes_any"}
)
SUPPORTED_NORMALIZERS = frozenset({"casefold", "whitespace"})
MAX_CHECK_WEIGHT = 1_000_000


class BenchmarkValidationError(ValueError):
    """Raised when a benchmark fixture is malformed."""


def _require_nonempty_string(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise BenchmarkValidationError(f"{field} must be a non-empty string")
    if any("\ud800" <= character <= "\udfff" for character in value):
        raise BenchmarkValidationError(f"{field} contains invalid Unicode")
    return value


def _reject_json_constant(value: str) -> None:
    """Reject Python's non-RFC JSON extensions: NaN and infinities."""
    raise ValueError(f"Non-standard JSON numeric constant: {value}")


def _validate_check(check: Any, *, case_id: str, seen_ids: set[str]) -> None:
    if not isinstance(check, dict):
        raise BenchmarkValidationError(f"Case {case_id!r} contains a non-object check")

    check_id = _require_nonempty_string(check.get("id"), f"check id in case {case_id!r}")
    if check_id in seen_ids:
        raise BenchmarkValidationError(f"Duplicate check id {check_id!r} in case {case_id!r}")
    seen_ids.add(check_id)

    kind = check.get("kind")
    if not isinstance(kind, str) or kind not in SUPPORTED_CHECK_KINDS:
        raise BenchmarkValidationError(
            f"Unsupported check kind {kind!r} in case {case_id!r}; "
            f"expected one of {sorted(SUPPORTED_CHECK_KINDS)}"
        )

    weight = check.get("weight", 1)
    if (
        isinstance(weight, bool)
        or not isinstance(weight, (int, float))
        or not math.isfinite(weight)
        or weight <= 0
        or weight > MAX_CHECK_WEIGHT
    ):
        raise BenchmarkValidationError(
            f"Check {check_id!r} in case {case_id!r} has invalid weight {weight!r}"
        )

    normalizers = check.get("normalize", [])
    if not isinstance(normalizers, list) or any(
        not isinstance(normalizer, str)
        or normalizer not in SUPPORTED_NORMALIZERS
        for normalizer in normalizers
    ):
        raise BenchmarkValidationError(
            f"Check {check_id!r} in case {case_id!r} has invalid normalize list"
        )

    if kind in {"contains_any", "excludes_any"}:
        values = check.get("values")
        if not isinstance(values, list) or not values:
            raise BenchmarkValidationError(
                f"Check {check_id!r} in case {case_id!r} requires non-empty values"
            )
        for index, value in enumerate(values):
            _require_nonempty_string(value, f"values[{index}] for check {check_id!r}")
    else:
        _require_nonempty_string(check.get("value"), f"value for check {check_id!r}")


def validate_fixture(fixture: Any) -> None:
    """Validate the benchmark fixture, raising a precise error on bad input."""
    if not isinstance(fixture, dict):
        raise BenchmarkValidationError("Fixture must be a JSON object")
    if type(fixture.get("schema_version")) is not int or fixture["schema_version"] != 1:
        raise BenchmarkValidationError("schema_version must be 1")

    cases = fixture.get("cases")
    if not isinstance(cases, list) or not cases:
        raise BenchmarkValidationError("cases must be a non-empty list")

    seen_case_ids: set[str] = set()
    for case in cases:
        if not isinstance(case, dict):
            raise BenchmarkValidationError("Each case must be a JSON object")
        case_id = _require_nonempty_string(case.get("id"), "case id")
        if case_id in seen_case_ids:
            raise BenchmarkValidationError(f"Duplicate case id {case_id!r}")
        seen_case_ids.add(case_id)

        checks = case.get("checks")
        if not isinstance(checks, list) or not checks:
            raise BenchmarkValidationError(f"Case {case_id!r} must define checks")
        seen_check_ids: set[str] = set()
        for check in checks:
            _validate_check(check, case_id=case_id, seen_ids=seen_check_ids)


def _normalize(text: str, normalizers: list[str]) -> str:
    for normalizer in normalizers:
        if normalizer == "casefold":
            text = text.casefold()
        elif normalizer == "whitespace":
            text = re.sub(r"\s+", " ", text).strip()
    return text


def _check_passes(check: dict[str, Any], summary: str) -> bool:
    normalizers = check.get("normalize", [])
    candidate = _normalize(summary, normalizers)
    kind = check["kind"]

    if kind in {"contains_any", "excludes_any"}:
        matched = any(
            _normalize(value, normalizers) in candidate for value in check["values"]
        )
        return matched if kind == "contains_any" else not matched

    expected = _normalize(check["value"], normalizers)
    if kind == "contains":
        return expected in candidate
    return expected not in candidate


def score_case(case: dict[str, Any], summary: str) -> dict[str, Any]:
    """Score one summary.

    ``contains`` and ``contains_any`` checks contribute to lexical retention.
    ``excludes`` and ``excludes_any`` checks are integrity gates reported
    separately so a summary cannot hide a leaked secret or known stale
    instruction behind a high lexical-recall score.
    """
    validate_fixture({"schema_version": 1, "cases": [case]})
    if not isinstance(summary, str):
        raise BenchmarkValidationError(f"Summary for case {case['id']!r} must be a string")

    retained_weight = 0.0
    retention_weight = 0.0
    failed_checks: list[dict[str, str]] = []
    passed_checks = 0
    safety_passed = True

    for check in case["checks"]:
        passed = _check_passes(check, summary)
        if check["kind"] not in {"excludes", "excludes_any"}:
            weight = float(check.get("weight", 1))
            retention_weight += weight
            if passed:
                retained_weight += weight
        elif not passed:
            safety_passed = False

        if passed:
            passed_checks += 1
        else:
            failed_checks.append({"id": check["id"], "kind": check["kind"]})

    retention_score = retained_weight / retention_weight if retention_weight else 1.0
    return {
        "case_id": case["id"],
        "passed": not failed_checks,
        "retention_score": retention_score,
        "safety_passed": safety_passed,
        "passed_checks": passed_checks,
        "total_checks": len(case["checks"]),
        "failed_checks": failed_checks,
    }


def score_fixture(fixture: dict[str, Any], summaries: dict[str, str]) -> dict[str, Any]:
    """Score all fixture cases and return aggregate and per-case results."""
    validate_fixture(fixture)
    if not isinstance(summaries, dict):
        raise BenchmarkValidationError("Summaries must be a JSON object")
    for case_id, summary in summaries.items():
        if not isinstance(case_id, str) or not isinstance(summary, str):
            raise BenchmarkValidationError(
                f"Summary for case {case_id!r} must be a string"
            )
    case_ids = {case["id"] for case in fixture["cases"]}
    unexpected = sorted(summaries.keys() - case_ids)
    if unexpected:
        raise BenchmarkValidationError(
            f"Unexpected summaries for cases: {', '.join(unexpected)}"
        )
    missing = sorted(case_ids - summaries.keys())
    if missing:
        raise BenchmarkValidationError(f"Missing summaries for cases: {', '.join(missing)}")

    results = [score_case(case, summaries[case["id"]]) for case in fixture["cases"]]
    return {
        "schema_version": 1,
        "case_count": len(results),
        "passed_cases": sum(result["passed"] for result in results),
        "mean_retention_score": sum(result["retention_score"] for result in results) / len(results),
        "safety_passed": all(result["safety_passed"] for result in results),
        "cases": results,
    }


def load_json(path: str | Path) -> Any:
    """Load one benchmark JSON document with a validation-friendly error."""
    json_path = Path(path)
    try:
        return json.loads(
            json_path.read_text(encoding="utf-8"),
            parse_constant=_reject_json_constant,
        )
    except (OSError, UnicodeError, ValueError, RecursionError) as exc:
        raise BenchmarkValidationError(f"Could not read {json_path}: {exc}") from exc


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--summaries", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    try:
        report = score_fixture(load_json(args.fixture), load_json(args.summaries))
    except BenchmarkValidationError as exc:
        parser.error(str(exc))

    rendered = json.dumps(
        report, ensure_ascii=False, indent=2, allow_nan=False
    ) + "\n"
    if args.output:
        try:
            args.output.write_text(rendered, encoding="utf-8")
        except OSError as exc:
            parser.error(f"Could not write {args.output}: {exc}")
    else:
        print(rendered, end="")
    return 0 if report["passed_cases"] == report["case_count"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
