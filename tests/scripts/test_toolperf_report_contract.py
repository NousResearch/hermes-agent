from scripts.toolperf_abeval.report_contract import validate_toolperf_report


def test_toolperf_report_requires_distinct_provenance() -> None:
    value = {"baseline_sha": "a" * 40, "fixes_sha": "b" * 40, "model": "test-model", "concurrency": 1, "metrics": {"tool_calls": 4}, "status": "pass", "model_provenance": {"model": "test-model", "provider": "test-provider", "config_digest": "c" * 64}}
    assert validate_toolperf_report(value) == []
    value["fixes_sha"] = "a" * 40
    assert "arms_must_use_distinct_shas" in validate_toolperf_report(value)


def test_toolperf_report_requires_model_configuration_provenance() -> None:
    value = {"baseline_sha": "a" * 40, "fixes_sha": "b" * 40, "model": "test-model", "concurrency": 1, "metrics": {}, "status": "pass"}
    assert "model_provenance_must_be_mapping" in validate_toolperf_report(value)
