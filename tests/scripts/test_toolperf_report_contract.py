from scripts.toolperf_abeval.report_contract import validate_toolperf_report


def test_toolperf_report_requires_distinct_provenance() -> None:
    value = {"baseline_sha": "a", "fixes_sha": "b", "model": "test-model", "concurrency": 1, "metrics": {"tool_calls": 4}, "status": "pass"}
    assert validate_toolperf_report(value) == []
    value["fixes_sha"] = "a"
    assert "arms_must_use_distinct_shas" in validate_toolperf_report(value)
