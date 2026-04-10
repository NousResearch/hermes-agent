"""Tests for CLI /status command behavior."""
from datetime import datetime
from types import SimpleNamespace
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
    cli_obj._status_bar_visible = True
    cli_obj.model = "openai/gpt-5.4"
    cli_obj.provider = "openai"
    cli_obj.session_start = datetime(2026, 4, 9, 19, 24)
    cli_obj._agent_running = False
    cli_obj._session_db = MagicMock()
    cli_obj._session_db.get_session.return_value = None
    return cli_obj


def test_status_command_is_available_in_cli_registry():
    cmd = resolve_command("status")
    assert cmd is not None
    assert cmd.gateway_only is False


def test_process_command_status_dispatches_without_toggling_status_bar():
    cli_obj = _make_cli()

    with patch.object(cli_obj, "_show_session_status", create=True) as mock_status:
        assert cli_obj.process_command("/status") is True

    mock_status.assert_called_once_with()
    assert cli_obj._status_bar_visible is True


def test_statusbar_still_toggles_visibility():
    cli_obj = _make_cli()

    assert cli_obj.process_command("/statusbar") is True
    assert cli_obj._status_bar_visible is False


def test_status_prefix_prefers_status_command_over_statusbar_toggle():
    cli_obj = _make_cli()

    with patch.object(cli_obj, "_show_session_status") as mock_status:
        assert cli_obj.process_command("/sta") is True

    mock_status.assert_called_once_with()
    assert cli_obj._status_bar_visible is True


def test_show_session_status_prints_gateway_style_summary():
    cli_obj = _make_cli()
    cli_obj.agent = SimpleNamespace(
        session_total_tokens=321,
        session_api_calls=4,
    )
    cli_obj._session_db.get_session.return_value = {
        "title": "My titled session",
        "started_at": 1775791440,
    }

    with patch("cli.display_hermes_home", return_value="~/.hermes"):
        cli_obj._show_session_status()

    printed = "\n".join(str(call.args[0]) for call in cli_obj.console.print.call_args_list)
    assert "Hermes CLI Status" in printed
    assert "Session ID: session-123" in printed
    assert "Path: ~/.hermes" in printed
    assert "Title: My titled session" in printed
    assert "Model: openai/gpt-5.4 (openai)" in printed
    assert "Tokens: 321" in printed
    assert "Agent Running: No" in printed
    _, kwargs = cli_obj.console.print.call_args
    assert kwargs.get("highlight") is False
    assert kwargs.get("markup") is False
