from __future__ import annotations

import inspect
import json

import hermes_cli.reliability_doctor as reliability_doctor
from hermes_cli.reliability_doctor import (
    evaluate_static_probes,
    render_diagnostic_json,
    render_diagnostic_text,
    validate_smoke_spec,
)


def test_reliability_doctor_exposes_no_command_execution_path():
    source = inspect.getsource(reliability_doctor)

    assert not hasattr(reliability_doctor, "execute_command_probe")
    assert "subprocess" not in source
    assert "Popen" not in source
    assert "psutil" not in source
    assert "build_subprocess_env" not in source


def test_command_probe_is_rejected_before_static_evaluation():
    try:
        validate_smoke_spec({
            "version": 1,
            "probes": [
                {"type": "command", "argv": ["true"], "expected_exit_codes": [0]}
            ],
        })
    except Exception as exc:
        assert str(exc) == "unknown_probe_type"
    else:  # pragma: no cover - should be rejected
        raise AssertionError("command smoke probes must not validate")


def test_static_diagnostics_do_not_disclose_secret_values(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "PLANTED_PROVIDER_SECRET")
    smoke = validate_smoke_spec({
        "version": 1,
        "probes": [{"type": "env-present", "name": "OPENAI_API_KEY"}],
    })

    [result] = evaluate_static_probes(
        smoke, subject_type="cron", subject="job", roots={}
    )

    assert result.status == "pass"
    assert "PLANTED_PROVIDER_SECRET" not in repr(result)
    assert "PLANTED_PROVIDER_SECRET" not in json.dumps(result.to_dict())
    assert "PLANTED_PROVIDER_SECRET" not in render_diagnostic_text([result])
    assert "PLANTED_PROVIDER_SECRET" not in render_diagnostic_json([result])


def test_hand_edited_job_fields_cannot_inject_or_expand_diagnostic_output(monkeypatch):
    malicious_schedule = "every 1h\nFAIL forged " + ("x" * 2_000)
    monkeypatch.setattr(
        reliability_doctor,
        "resolve_job_ref",
        lambda ref: {
            "id": "job\nFAIL forged-subject",
            "prompt": "Run",
            "schedule": {"kind": "interval"},
            "schedule_display": malicious_schedule,
        },
    )

    results = reliability_doctor.diagnose_cron("daily")
    text = render_diagnostic_text(results)
    payload = render_diagnostic_json(results)

    assert "\nFAIL forged" not in text
    assert "\nFAIL forged" not in payload
    assert len(text) < 1_200
