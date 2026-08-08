"""Tests for the provider-neutral long-session retention benchmark."""

import math
import subprocess
import sys

import pytest

from scripts.context_compression_benchmark import (
    BenchmarkValidationError,
    load_json,
    score_case,
    score_fixture,
    validate_fixture,
)


@pytest.fixture()
def case():
    return {
        "id": "exact-operational-context",
        "category": "retention",
        "checks": [
            {
                "id": "windows-path",
                "kind": "contains",
                "value": r"C:\Work\project\source.txt",
            },
            {
                "id": "unicode-glyph",
                "kind": "contains",
                "value": "Use — rather than - in the final output.",
            },
            {
                "id": "latest-request",
                "kind": "contains_any",
                "values": ["run the verification now", "verify it now"],
            },
            {
                "id": "secret-redacted",
                "kind": "excludes",
                "value": "sk-test-secret-value",
            },
        ],
    }


def test_score_case_reports_weighted_retention_and_safety(case):
    case["checks"][0]["weight"] = 2
    summary = (
        "Work in C:\\Work\\project\\source.txt. "
        "Use — rather than - in the final output. "
        "The latest request is to verify it now. Credentials were [REDACTED]."
    )

    result = score_case(case, summary)

    assert result["case_id"] == "exact-operational-context"
    assert result["passed"] is True
    assert result["retention_score"] == pytest.approx(1.0)
    assert result["safety_passed"] is True
    assert result["passed_checks"] == 4
    assert result["failed_checks"] == []


def test_score_case_distinguishes_retention_failure_from_safety_failure(case):
    summary = "The old file was changed. sk-test-secret-value"

    result = score_case(case, summary)

    assert result["passed"] is False
    assert result["retention_score"] == pytest.approx(0.0)
    assert result["safety_passed"] is False
    assert {failure["id"] for failure in result["failed_checks"]} == {
        "windows-path",
        "unicode-glyph",
        "latest-request",
        "secret-redacted",
    }


def test_contains_is_exact_by_default_and_can_opt_into_normalization(case):
    case["checks"] = [
        {
            "id": "exact",
            "kind": "contains",
            "value": "Case  Sensitive\nText",
        },
        {
            "id": "normalized",
            "kind": "contains",
            "value": "Case  Sensitive\nText",
            "normalize": ["casefold", "whitespace"],
        },
    ]

    result = score_case(case, "case sensitive text")

    assert result["retention_score"] == pytest.approx(0.5)
    assert [failure["id"] for failure in result["failed_checks"]] == ["exact"]


def test_validate_fixture_rejects_unknown_check_kind(case):
    case["checks"][0]["kind"] = "semantic-vibes"

    with pytest.raises(BenchmarkValidationError, match="semantic-vibes"):
        validate_fixture({"schema_version": 1, "cases": [case]})


@pytest.mark.parametrize("kind", [["contains"], {"contains": True}])
def test_validate_fixture_rejects_nonstring_check_kind(case, kind):
    case["checks"][0]["kind"] = kind

    with pytest.raises(BenchmarkValidationError, match="check kind"):
        validate_fixture({"schema_version": 1, "cases": [case]})


def test_validate_fixture_rejects_nonstring_normalizer(case):
    case["checks"][0]["normalize"] = [{"casefold": True}]

    with pytest.raises(BenchmarkValidationError, match="normalize list"):
        validate_fixture({"schema_version": 1, "cases": [case]})


def test_validate_fixture_requires_unique_case_and_check_ids(case):
    duplicate_case = {**case, "checks": [dict(check) for check in case["checks"]]}

    with pytest.raises(BenchmarkValidationError, match="Duplicate case id"):
        validate_fixture({"schema_version": 1, "cases": [case, duplicate_case]})

    case["checks"].append(dict(case["checks"][0]))
    with pytest.raises(BenchmarkValidationError, match="Duplicate check id"):
        validate_fixture({"schema_version": 1, "cases": [case]})


def test_validate_fixture_rejects_nonpositive_weights(case):
    case["checks"][0]["weight"] = 0

    with pytest.raises(BenchmarkValidationError, match="weight"):
        validate_fixture({"schema_version": 1, "cases": [case]})


@pytest.mark.parametrize("weight", [math.nan, math.inf, -math.inf, 1e308])
def test_validate_fixture_rejects_nonfinite_weights(case, weight):
    case["checks"][0]["weight"] = weight

    with pytest.raises(BenchmarkValidationError, match="weight"):
        validate_fixture({"schema_version": 1, "cases": [case]})


def test_validate_fixture_rejects_boolean_schema_version(case):
    with pytest.raises(BenchmarkValidationError, match="schema_version"):
        validate_fixture({"schema_version": True, "cases": [case]})


def test_score_fixture_rejects_malformed_summary_documents(case):
    fixture = {"schema_version": 1, "cases": [case]}

    with pytest.raises(BenchmarkValidationError, match="JSON object"):
        score_fixture(fixture, ["not", "an", "object"])

    with pytest.raises(BenchmarkValidationError, match="must be a string"):
        score_fixture(fixture, {case["id"]: 42})

    with pytest.raises(BenchmarkValidationError, match="Unexpected summaries"):
        score_fixture(fixture, {case["id"]: "ok", "typo-id": "ignored"})


def test_load_json_wraps_invalid_utf8(tmp_path):
    invalid = tmp_path / "invalid.json"
    invalid.write_bytes(b"\xff\xfe\x00")

    with pytest.raises(BenchmarkValidationError, match="Could not read"):
        load_json(invalid)


@pytest.mark.parametrize("constant", ["NaN", "Infinity", "-Infinity"])
def test_load_json_rejects_nonstandard_numeric_constants(tmp_path, constant):
    invalid = tmp_path / "invalid.json"
    invalid.write_text(
        f'{{"schema_version": 1, "description": {constant}}}',
        encoding="utf-8",
    )

    with pytest.raises(BenchmarkValidationError, match="Could not read"):
        load_json(invalid)


def test_load_json_wraps_excessive_nesting(tmp_path):
    invalid = tmp_path / "deep.json"
    invalid.write_text("[" * 2_000 + "0" + "]" * 2_000, encoding="utf-8")

    with pytest.raises(BenchmarkValidationError, match="Could not read"):
        load_json(invalid)


def test_validate_fixture_rejects_unpaired_unicode_surrogate(case):
    case["id"] = "invalid-\ud800-id"

    with pytest.raises(BenchmarkValidationError, match="invalid Unicode"):
        validate_fixture({"schema_version": 1, "cases": [case]})


def test_cli_reports_unwritable_output_without_traceback(tmp_path):
    output = tmp_path / "missing" / "result.json"
    command = [
        sys.executable,
        "scripts/context_compression_benchmark.py",
        "--fixture",
        "benchmarks/context_compression/long_session_v1.json",
        "--summaries",
        "benchmarks/context_compression/example_summaries.json",
        "--output",
        str(output),
    ]

    completed = subprocess.run(command, capture_output=True, text=True, check=False)

    assert completed.returncode == 2
    assert "Could not write" in completed.stderr
    assert "Traceback" not in completed.stderr


def test_excludes_any_is_an_integrity_gate(case):
    case["checks"] = [
        {
            "id": "stale-action",
            "kind": "excludes_any",
            "values": ["publish immediately", "deploy to staging"],
            "normalize": ["casefold"],
        }
    ]

    result = score_case(case, "The obsolete instruction says: PUBLISH IMMEDIATELY")

    assert result["passed"] is False
    assert result["retention_score"] == 1.0
    assert result["safety_passed"] is False


def test_repository_fixture_and_example_summaries_pass_end_to_end():
    fixture = load_json("benchmarks/context_compression/long_session_v1.json")
    summaries = load_json(
        "benchmarks/context_compression/example_summaries.json"
    )

    report = score_fixture(fixture, summaries)

    assert report["case_count"] == 3
    assert report["passed_cases"] == 3
    assert report["mean_retention_score"] == 1.0
    assert report["safety_passed"] is True


def test_repository_adversarial_inversions_are_rejected():
    fixture = load_json("benchmarks/context_compression/long_session_v1.json")
    summaries = load_json(
        "benchmarks/context_compression/adversarial_summaries.json"
    )

    report = score_fixture(fixture, summaries)

    assert report["passed_cases"] == 0
    assert report["safety_passed"] is False
