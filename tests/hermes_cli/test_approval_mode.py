"""Tests for hermes_cli/approval_mode.py — pure functions and dataclass invariants."""

import pytest
from unittest.mock import patch


def test_valid_approval_modes_exact():
    from hermes_cli.approval_mode import VALID_APPROVAL_MODES
    assert set(VALID_APPROVAL_MODES) == {"manual", "smart", "off"}


def test_approval_mode_result_is_frozen():
    from hermes_cli.approval_mode import ApprovalModeResult
    result = ApprovalModeResult(ok=True, mode="manual", changed=False, message="ok")
    assert result.ok is True
    assert result.mode == "manual"
    assert result.changed is False
    assert result.message == "ok"
    with pytest.raises(Exception):
        result.ok = False  # frozen dataclass


def test_nonpersisted_empty_request_inspects_current():
    from hermes_cli.approval_mode import run_approval_mode_command
    with patch("hermes_cli.approval_mode._effective_mode", return_value="manual"):
        result = run_approval_mode_command(None)
    assert result.ok is True
    assert result.mode == "manual"
    assert result.changed is False


def test_nonpersisted_blank_request_inspects_current():
    from hermes_cli.approval_mode import run_approval_mode_command
    with patch("hermes_cli.approval_mode._effective_mode", return_value="smart"):
        result = run_approval_mode_command("  ")
    assert result.ok is True
    assert result.mode == "smart"
    assert result.changed is False


def test_invalid_mode_rejected():
    from hermes_cli.approval_mode import run_approval_mode_command
    with patch("hermes_cli.approval_mode._effective_mode", return_value="off"):
        result = run_approval_mode_command("invalid")
    assert result.ok is False
    assert "Usage" in result.message


@pytest.mark.parametrize("mode", ["manual", "smart", "off"])
def test_valid_mode_persists(mode):
    from hermes_cli.approval_mode import run_approval_mode_command
    with (
        patch("hermes_cli.approval_mode._effective_mode", side_effect=["off", mode]),
        patch("hermes_cli.approval_mode.set_config_value") as mock_set,
    ):
        result = run_approval_mode_command(mode)
    assert result.ok is True
    assert result.mode == mode
    mock_set.assert_called_once_with("approvals.mode", mode)


def test_persist_divergent_effective_reports_failure():
    from hermes_cli.approval_mode import run_approval_mode_command
    with (
        patch("hermes_cli.approval_mode._effective_mode", side_effect=["off", "off"]),
        patch("hermes_cli.approval_mode.set_config_value"),
    ):
        result = run_approval_mode_command("smart")
    assert result.ok is False
    assert result.mode == "off"


def test_systemexit_during_set_config_reported():
    from hermes_cli.approval_mode import run_approval_mode_command
    with (
        patch("hermes_cli.approval_mode._effective_mode", return_value="smart"),
        patch("hermes_cli.approval_mode.set_config_value", side_effect=SystemExit),
    ):
        result = run_approval_mode_command("manual")
    assert result.ok is False
    assert "managed" in result.message.lower() or "cannot" in result.message.lower()


def test_exception_during_set_config_reported():
    from hermes_cli.approval_mode import run_approval_mode_command
    with (
        patch("hermes_cli.approval_mode._effective_mode", return_value="smart"),
        patch("hermes_cli.approval_mode.set_config_value", side_effect=RuntimeError("boom")),
    ):
        result = run_approval_mode_command("off")
    assert result.ok is False
    assert "boom" in result.message
