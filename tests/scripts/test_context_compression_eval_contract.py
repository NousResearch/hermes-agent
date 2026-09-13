from scripts.compression_eval.report_contract import validate_report


def report() -> dict[str, object]:
    return {
        "schema_version": 1,
        "evaluator_digest": "c" * 64,
        "battery_digest": "d" * 64,
        "source_sha": "a" * 40,
        "fixture_digest": "b" * 64,
        "compressed_tokens": 100,
        "baseline_tokens": 200,
        "probe_manifest": ["accuracy"],
        "probe_scores": {"accuracy": 5},
        "artifact_trail_preserved": True,
        "continuity_preserved": True,
        "model_provenance": {
            "compression_model": "compression-model",
            "evaluator_model": "evaluator-model",
            "provider": "test-provider",
            "model_config": "config-digest",
        },
        "status": "pass",
    }


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
    assert "forbidden_credential_assignment" in validate_report(value)

    value["probe_scores"] = {"detail": "OPENROUTER_API_KEY=secret"}
    assert "forbidden_credential_assignment" in validate_report(value)

    value["probe_scores"] = {"detail": "Authorization: Bearer sk-secret"}
    assert "forbidden_credential_assignment" in validate_report(value)


def test_pass_requires_preservation_and_rejects_nested_credentials() -> None:
    value = report()
    value["artifact_trail_preserved"] = False
    value["probe_scores"] = {"nested": {"OPENAI_API_KEY": "secret"}}
    errors = validate_report(value)
    assert "artifact_trail_preserved_must_be_true_for_pass" in errors
    assert "forbidden_key:report.probe_scores.nested.OPENAI_API_KEY" in errors


def test_report_rejects_impossible_counts_and_empty_provenance() -> None:
    value = report()
    value["compressed_tokens"] = -1
    value["baseline_tokens"] = 0
    value["model_provenance"]["compression_model"] = ""
    errors = validate_report(value)
    assert "compressed_tokens_must_be_nonnegative" in errors
    assert "baseline_tokens_must_be_positive" in errors
    assert "model_provenance.compression_model_must_be_nonempty_string" in errors


def test_report_rejects_invalid_fixture_scores_and_home_paths() -> None:
    value = report()
    value["fixture_digest"] = "not-a-digest"
    value["probe_scores"] = {"accuracy": float("nan")}
    value["model_provenance"]["provider"] = r"C:\Users\alice\provider"
    errors = validate_report(value)
    assert "fixture_digest_must_be_sha256" in errors
    assert "nonfinite_number:probe_scores.accuracy" in errors
    assert "forbidden_home_path" in errors


def test_pass_requires_every_declared_probe() -> None:
    value = report()
    value["probe_manifest"] = ["accuracy", "continuity"]
    errors = validate_report(value)
    assert "missing_probe_scores:continuity" in errors


def test_compound_secret_keys_and_nonnumeric_scores_are_rejected():
    value = report()
    value["metadata"] = {"credentials": [{"AWS_SECRET_ACCESS_KEY": "fake"}]}
    assert any("AWS_SECRET_ACCESS_KEY" in error for error in validate_report(value))
    for score in (None, "1", {}, True):
        value = report()
        value["probe_scores"]["accuracy"] = score
        assert "probe_scores.accuracy_must_be_numeric" in validate_report(value)


def test_serialized_config_credentials_are_rejected():
    import json
    value = report()
    value["model_provenance"]["model_config"] = json.dumps({"OPENAI_API_KEY": "fake-secret"})
    assert any("forbidden_key" in error for error in validate_report(value))
