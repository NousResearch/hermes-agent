from scripts.compression_eval.report_contract import validate_report


def report() -> dict[str, object]:
    return {"schema_version": 1, "source_sha": "a" * 40, "fixture_digest": "b" * 64, "compressed_tokens": 100, "baseline_tokens": 200, "probe_scores": {"accuracy": 5}, "artifact_trail_preserved": True, "continuity_preserved": True, "status": "pass"}


def test_valid_report_passes() -> None:
    assert validate_report(report()) == []


def test_missing_and_invalid_fields_fail_closed() -> None:
    value = report()
    value.pop("fixture_digest")
    value["status"] = "green"
    errors = validate_report(value)
    assert "missing:fixture_digest" in errors
    assert "invalid_status" in errors


def test_secret_or_local_path_is_rejected() -> None:
    value = report()
    value["probe_scores"] = {"detail": "OPENAI_API_KEY=secret"}
    assert any(error.startswith("forbidden_marker:") for error in validate_report(value))
