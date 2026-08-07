from __future__ import annotations

import pytest

from hermes_cli.reliability_doctor import (
    DiagnosticResult,
    SmokeValidationError,
    validate_smoke_spec,
)


def test_validate_smoke_spec_canonicalises_supported_static_probes():
    source = {
        "version": 1,
        "probes": [
            {"type": "command-exists", "name": "gh"},
            {"type": "env-present", "name": "GH_TOKEN"},
            {"type": "file-exists", "root": "hermes_home", "path": "scripts/a.py"},
            {"type": "directory-exists", "root": "workdir", "path": "fixtures"},
            {"type": "python-import", "module": "yaml"},
            {"type": "mcp-configured", "server": "example"},
        ],
    }

    validated = validate_smoke_spec(source)

    assert validated == {
        "version": 1,
        "probes": [
            {"type": "command-exists", "name": "gh"},
            {"type": "env-present", "name": "GH_TOKEN"},
            {"type": "file-exists", "root": "hermes_home", "path": "scripts/a.py"},
            {"type": "directory-exists", "root": "workdir", "path": "fixtures"},
            {"type": "python-import", "module": "yaml"},
            {"type": "mcp-configured", "server": "example"},
        ],
    }


def test_validate_smoke_spec_rejects_unknown_top_level_field():
    with pytest.raises(SmokeValidationError, match="unknown_field"):
        validate_smoke_spec({"version": 1, "probes": [], "extra": True})


def test_validate_smoke_spec_rejects_unknown_probe_field():
    with pytest.raises(SmokeValidationError, match="unknown_probe_field"):
        validate_smoke_spec({
            "version": 1,
            "probes": [{"type": "env-present", "name": "X", "value": "secret"}],
        })


def test_validate_smoke_spec_rejects_command_probe_with_stable_reason():
    with pytest.raises(SmokeValidationError, match="unknown_probe_type"):
        validate_smoke_spec({
            "version": 1,
            "probes": [
                {"type": "command", "argv": ["true"], "expected_exit_codes": [0]}
            ],
        })


def test_validate_smoke_spec_rejects_more_than_32_probes():
    probes = [{"type": "env-present", "name": f"X_{idx}"} for idx in range(33)]
    with pytest.raises(SmokeValidationError, match="too_many_probes"):
        validate_smoke_spec({"version": 1, "probes": probes})


@pytest.mark.parametrize(
    "path", ["/etc/passwd", "../secret", "a/../../secret", "C:/temp/x"]
)
def test_validate_smoke_spec_rejects_absolute_and_parent_paths(path):
    with pytest.raises(SmokeValidationError, match="invalid_path"):
        validate_smoke_spec({
            "version": 1,
            "probes": [{"type": "file-exists", "root": "workdir", "path": path}],
        })


@pytest.mark.parametrize("value", ["BAD\nNAME", "BAD\u200bNAME", "BAD\x00NAME"])
def test_validate_smoke_spec_rejects_control_and_invisible_characters(value):
    with pytest.raises(SmokeValidationError, match="invalid_text"):
        validate_smoke_spec({
            "version": 1,
            "probes": [{"type": "env-present", "name": value}],
        })


@pytest.mark.parametrize(
    ("probe", "reason"),
    [
        (
            {"type": "file-exists", "root": "workdir", "path": ["secret"]},
            "invalid_path",
        ),
        ({"type": "env-present", "name": {"bad": "shape"}}, "invalid_name"),
        ({"type": "python-import", "module": "bad-name"}, "invalid_module"),
        ({"type": "mcp-configured", "server": "bad/name"}, "invalid_server"),
    ],
)
def test_validate_smoke_spec_malformed_shapes_raise_stable_reasons(probe, reason):
    with pytest.raises(SmokeValidationError, match=reason):
        validate_smoke_spec({"version": 1, "probes": [probe]})


@pytest.mark.parametrize(
    ("spec", "reason"),
    [
        ({"version": 1, "probes": [{"type": []}]}, "unknown_probe_type"),
        (
            {
                "version": 1,
                "probes": [{"type": "env-present", "name": "TOKEN", 1: "x"}],
            },
            "invalid_probe_field",
        ),
        ({"version": 1, "probes": [], 1: "x"}, "invalid_smoke_field"),
    ],
)
def test_validate_smoke_spec_non_string_schema_shapes_raise_stable_reasons(
    spec, reason
):
    with pytest.raises(SmokeValidationError, match=reason):
        validate_smoke_spec(spec)


def test_diagnostic_result_to_dict_has_no_free_form_output_field():
    result = DiagnosticResult(
        subject_type="cron",
        subject="daily",
        probe_type="env-present",
        target="GH_TOKEN",
        status="pass",
        reason="env_present",
    )

    assert result.to_dict() == {
        "subject_type": "cron",
        "subject": "daily",
        "probe_type": "env-present",
        "target": "GH_TOKEN",
        "status": "pass",
        "reason": "env_present",
    }
    forbidden = {
        "stdout",
        "stderr",
        "output",
        "detail",
        "exception",
        "environment",
        "exit_code",
        "duration_ms",
    }
    assert forbidden.isdisjoint(result.to_dict())
