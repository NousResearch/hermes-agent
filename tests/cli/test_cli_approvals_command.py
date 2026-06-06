"""Tests for the CLI ``/approvals`` slash command.

The ``/approvals`` command provides a unified interface for viewing and
setting the approval mode (manual/smart/off) from within a chat session.
It extends ``/yolo`` (which only toggles manual↔off) with:

1. Display of the current approval mode when called without arguments.
2. Three-way mode selection: manual, smart, off.
3. Persistence to ``~/.hermes/config.yaml`` via ``save_config_value``.
4. Session-yolo state synchronization (``/approvals off`` enables session
   yolo; switching away from ``off`` disables it).

Test strategy mirrors ``test_cli_yolo_toggle.py``: we use a minimal
``SimpleNamespace`` stand-in instead of constructing a full ``HermesCLI``,
and mock ``_cprint`` to suppress output.

NOTE: The handler receives the full command string (e.g. "approvals manual",
not just "manual"), matching how ``cmd_original`` is passed from
``process_command``.
"""

from types import SimpleNamespace
from unittest.mock import patch, MagicMock

import pytest

import tools.approval as approval_module
from cli import HermesCLI


SESSION_KEY = "test-cli-approvals-session"


@pytest.fixture(autouse=True)
def _clear_approval_state(monkeypatch):
    """Clear session-yolo state around every test."""
    approval_module.clear_session(SESSION_KEY)
    approval_module.clear_session("default")
    yield
    approval_module.clear_session(SESSION_KEY)
    approval_module.clear_session("default")


def _make_stand_in(session_id: str = SESSION_KEY) -> SimpleNamespace:
    """Minimal stand-in exposing only ``session_id``."""
    return SimpleNamespace(session_id=session_id)


class TestApprovalsShowCurrentMode:
    """/approvals without arguments displays the current approval mode."""

    @patch("cli.save_config_value")
    @patch("tools.approval._get_approval_mode", return_value="manual")
    def test_show_manual_mode(self, mock_get_mode, mock_save):
        stand_in = _make_stand_in()
        with patch("cli._cprint") as mock_print:
            HermesCLI._handle_approvals_command(stand_in, "approvals")

        assert mock_print.called
        calls = [str(c) for c in mock_print.call_args_list]
        assert any("manual" in c for c in calls)

    @patch("cli.save_config_value")
    @patch("tools.approval._get_approval_mode", return_value="smart")
    def test_show_smart_mode(self, mock_get_mode, mock_save):
        stand_in = _make_stand_in()
        with patch("cli._cprint") as mock_print:
            HermesCLI._handle_approvals_command(stand_in, "approvals")

        calls = [str(c) for c in mock_print.call_args_list]
        assert any("smart" in c for c in calls)

    @patch("cli.save_config_value")
    @patch("tools.approval._get_approval_mode", return_value="off")
    def test_show_off_mode(self, mock_get_mode, mock_save):
        stand_in = _make_stand_in()
        with patch("cli._cprint") as mock_print:
            HermesCLI._handle_approvals_command(stand_in, "approvals")

        calls = [str(c) for c in mock_print.call_args_list]
        assert any("off" in c for c in calls)


class TestApprovalsSetMode:
    """/approvals <mode> persists the mode and synchronizes session-yolo."""

    @patch("cli.save_config_value", return_value=True)
    def test_set_manual_mode(self, mock_save):
        stand_in = _make_stand_in()
        with patch("cli._cprint"):
            HermesCLI._handle_approvals_command(stand_in, "approvals manual")

        mock_save.assert_called_once_with("approvals.mode", "manual")

    @patch("cli.save_config_value", return_value=True)
    def test_set_smart_mode(self, mock_save):
        stand_in = _make_stand_in()
        with patch("cli._cprint"):
            HermesCLI._handle_approvals_command(stand_in, "approvals smart")

        mock_save.assert_called_once_with("approvals.mode", "smart")

    @patch("cli.save_config_value", return_value=True)
    def test_set_off_enables_session_yolo(self, mock_save):
        stand_in = _make_stand_in()
        assert approval_module.is_session_yolo_enabled(SESSION_KEY) is False

        with patch("cli._cprint"):
            HermesCLI._handle_approvals_command(stand_in, "approvals off")

        assert approval_module.is_session_yolo_enabled(SESSION_KEY) is True

    @patch("cli.save_config_value", return_value=True)
    def test_set_manual_disables_session_yolo(self, mock_save):
        stand_in = _make_stand_in()
        approval_module.enable_session_yolo(SESSION_KEY)
        assert approval_module.is_session_yolo_enabled(SESSION_KEY) is True

        with patch("cli._cprint"):
            HermesCLI._handle_approvals_command(stand_in, "approvals manual")

        assert approval_module.is_session_yolo_enabled(SESSION_KEY) is False

    @patch("cli.save_config_value", return_value=True)
    def test_set_smart_disables_session_yolo(self, mock_save):
        stand_in = _make_stand_in()
        approval_module.enable_session_yolo(SESSION_KEY)
        assert approval_module.is_session_yolo_enabled(SESSION_KEY) is True

        with patch("cli._cprint"):
            HermesCLI._handle_approvals_command(stand_in, "approvals smart")

        assert approval_module.is_session_yolo_enabled(SESSION_KEY) is False


class TestApprovalsInvalidInput:
    """/approvals with an invalid mode shows an error."""

    @patch("cli.save_config_value")
    def test_invalid_mode_shows_error(self, mock_save):
        stand_in = _make_stand_in()
        with patch("cli._cprint") as mock_print:
            HermesCLI._handle_approvals_command(stand_in, "approvals invalid")

        mock_save.assert_not_called()
        calls = [str(c) for c in mock_print.call_args_list]
        assert any("Unknown mode" in c for c in calls)

    @patch("cli.save_config_value")
    def test_invalid_mode_does_not_mutate_session_yolo(self, mock_save):
        stand_in = _make_stand_in()
        assert approval_module.is_session_yolo_enabled(SESSION_KEY) is False

        with patch("cli._cprint"):
            HermesCLI._handle_approvals_command(stand_in, "approvals bogus")

        assert approval_module.is_session_yolo_enabled(SESSION_KEY) is False


class TestApprovalsSaveFailure:
    """/approvals handles config save failures gracefully."""

    @patch("cli.save_config_value", return_value=False)
    def test_save_failure_shows_error(self, mock_save):
        stand_in = _make_stand_in()
        with patch("cli._cprint") as mock_print:
            HermesCLI._handle_approvals_command(stand_in, "approvals smart")

        calls = [str(c) for c in mock_print.call_args_list]
        assert any("Failed" in c for c in calls)

    @patch("cli.save_config_value", return_value=False)
    def test_save_failure_does_not_mutate_session_yolo(self, mock_save):
        stand_in = _make_stand_in()
        assert approval_module.is_session_yolo_enabled(SESSION_KEY) is False

        with patch("cli._cprint"):
            HermesCLI._handle_approvals_command(stand_in, "approvals off")

        assert approval_module.is_session_yolo_enabled(SESSION_KEY) is False


class TestApprovalsSessionIsolation:
    """/approvals in one session must not affect another session."""

    @patch("cli.save_config_value", return_value=True)
    def test_session_isolation(self, mock_save):
        cli_a = _make_stand_in(session_id="session-approvals-a")
        cli_b = _make_stand_in(session_id="session-approvals-b")

        try:
            with patch("cli._cprint"):
                HermesCLI._handle_approvals_command(cli_a, "approvals off")

            assert approval_module.is_session_yolo_enabled("session-approvals-a") is True
            assert approval_module.is_session_yolo_enabled("session-approvals-b") is False
        finally:
            approval_module.clear_session("session-approvals-a")
            approval_module.clear_session("session-approvals-b")

    @patch("cli.save_config_value", return_value=True)
    def test_default_session_fallback(self, mock_save):
        """When session_id is empty, falls back to 'default' key."""
        stand_in = _make_stand_in(session_id="")
        with patch("cli._cprint"):
            HermesCLI._handle_approvals_command(stand_in, "approvals off")

        assert approval_module.is_session_yolo_enabled("default") is True


class TestApprovalsEndToEnd:
    """End-to-end: /approvals off must bypass dangerous command checks."""

    @patch("cli.save_config_value", return_value=True)
    def test_approvals_off_bypasses_dangerous_command(self, mock_save):
        stand_in = _make_stand_in()

        token = approval_module.set_current_session_key(SESSION_KEY)
        try:
            with patch("cli._cprint"):
                HermesCLI._handle_approvals_command(stand_in, "approvals off")

            result = approval_module.check_all_command_guards(
                "rm -rf /tmp/scratch-xyzzy", "local",
            )
            assert result["approved"] is True, (
                f"/approvals off should auto-approve dangerous commands, got: {result}"
            )
        finally:
            approval_module.reset_current_session_key(token)

    @patch("cli.save_config_value", return_value=True)
    def test_approvals_manual_restores_prompts(self, mock_save):
        stand_in = _make_stand_in()

        token = approval_module.set_current_session_key(SESSION_KEY)
        try:
            with patch("cli._cprint"):
                HermesCLI._handle_approvals_command(stand_in, "approvals off")

            with patch("cli._cprint"):
                HermesCLI._handle_approvals_command(stand_in, "approvals manual")

            assert approval_module.is_session_yolo_enabled(SESSION_KEY) is False
        finally:
            approval_module.reset_current_session_key(token)
