"""Tests for CLI /sessions command behavior."""

from unittest.mock import MagicMock, patch

from cli import HermesCLI
from hermes_cli.commands import resolve_command


def _make_cli():
    cli_obj = HermesCLI.__new__(HermesCLI)
    cli_obj.config = {}
    cli_obj.console = MagicMock()
    cli_obj.agent = None
    cli_obj.conversation_history = []
    cli_obj.session_id = "session-123"
    cli_obj._pending_input = MagicMock()
    cli_obj._agent_running = False
    cli_obj._session_db = MagicMock()
    cli_obj._session_db.list_sessions_rich.return_value = []
    return cli_obj


def test_sessions_command_is_available_in_cli_registry():
    cmd = resolve_command("sessions")
    assert cmd is not None
    assert cmd.name == "sessions"


def test_handle_sessions_command_no_args_shows_recent():
    cli_obj = _make_cli()

    with patch.object(cli_obj, "_show_recent_sessions", return_value=True) as mock_show:
        cli_obj._handle_sessions_command("/sessions")

    mock_show.assert_called_once_with(reason="sessions", limit=20)


def test_handle_sessions_command_with_id_delegates_to_resume():
    cli_obj = _make_cli()

    with patch.object(cli_obj, "_handle_resume_command") as mock_resume:
        cli_obj._handle_sessions_command("/sessions abc-123")

    mock_resume.assert_called_once_with("/resume abc-123")


def test_handle_sessions_command_with_title_delegates_to_resume():
    cli_obj = _make_cli()

    with patch.object(cli_obj, "_handle_resume_command") as mock_resume:
        cli_obj._handle_sessions_command("/sessions my cool session")

    mock_resume.assert_called_once_with("/resume my cool session")


def test_handle_sessions_command_no_db_shows_no_sessions():
    cli_obj = _make_cli()
    cli_obj._session_db = None

    with patch("cli._cprint") as mock_print:
        cli_obj._handle_sessions_command("/sessions")

    calls = " ".join(str(c) for c in mock_print.call_args_list)
    assert "No previous sessions" in calls


def test_process_command_dispatches_to_handler():
    cli_obj = _make_cli()

    with patch.object(cli_obj, "_handle_sessions_command") as mock_handler:
        cli_obj.process_command("/sessions")

    mock_handler.assert_called_once_with("/sessions")
