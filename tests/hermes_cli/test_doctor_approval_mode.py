"""Focused security diagnostics for the effective approval mode."""

from __future__ import annotations

import pytest

from hermes_cli import doctor, doctor_config
from tools import approval_context


def _run_approval_check(monkeypatch: pytest.MonkeyPatch, config: object, capsys):
    monkeypatch.setattr(approval_context, "_get_approval_config", lambda: config)
    finding = doctor_config._check_approval_mode(False)
    return finding, capsys.readouterr().out


def test_doctor_registers_the_approval_prompt_safety_check():
    assert ("Approval Prompt Safety", doctor_config._check_approval_mode) in doctor.DOCTOR_CHECKS


def test_doctor_warns_when_effective_approval_mode_is_off(monkeypatch, capsys):
    finding, output = _run_approval_check(monkeypatch, {"mode": "off"}, capsys)

    assert "Approval mode is off" in output
    assert "/approvals manual" in output
    assert "/approvals smart" in output
    assert finding.manual_issues == [
        "Approval prompts are disabled for dangerous commands. Run '/approvals manual' or '/approvals smart' to restore them."
    ]


@pytest.mark.parametrize("mode", ["manual", "smart"])
def test_doctor_accepts_approval_modes_that_keep_prompts_enabled(monkeypatch, capsys, mode):
    finding, output = _run_approval_check(monkeypatch, {"mode": mode}, capsys)

    assert f"Approval mode: {mode}" in output
    assert finding.manual_issues == []


def test_doctor_normalizes_yaml_boolean_false_as_approval_mode_off(monkeypatch, capsys):
    finding, output = _run_approval_check(monkeypatch, {"mode": False}, capsys)

    assert "Approval mode is off" in output
    assert finding.manual_issues


@pytest.mark.parametrize("config", [{}, {"mode": "unsupported"}, []])
def test_doctor_fails_open_for_absent_or_malformed_approval_config(monkeypatch, capsys, config):
    finding, output = _run_approval_check(monkeypatch, config, capsys)

    assert "Approval mode is off" not in output
    assert finding.manual_issues == []
