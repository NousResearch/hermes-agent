from collections.abc import Mapping


REQUIRED = {"baseline_sha", "fixes_sha", "model", "concurrency", "metrics", "status"}


def validate_toolperf_report(report: Mapping[str, object]) -> list[str]:
    errors = [f"missing:{key}" for key in sorted(REQUIRED - report.keys())]
    if report.get("status") not in {"pass", "fail", "partial", "unavailable"}:
        errors.append("invalid_status")
    if not isinstance(report.get("metrics"), Mapping):
        errors.append("metrics_must_be_mapping")
    if not isinstance(report.get("concurrency"), int) or report.get("concurrency", 0) < 1:
        errors.append("concurrency_must_be_positive_integer")
    if report.get("baseline_sha") == report.get("fixes_sha"):
        errors.append("arms_must_use_distinct_shas")
    return errors


def test_toolperf_report_requires_distinct_provenance() -> None:
    value = {"baseline_sha": "a", "fixes_sha": "b", "model": "test-model", "concurrency": 1, "metrics": {"tool_calls": 4}, "status": "pass"}
    assert validate_toolperf_report(value) == []
    value["fixes_sha"] = "a"
    assert "arms_must_use_distinct_shas" in validate_toolperf_report(value)
