"""Tests for CLI `/evolve-cc` registration and dispatch."""

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
    return cli_obj


def test_evolve_cc_command_is_available_in_cli_registry():
    cmd = resolve_command("evolve-cc")
    assert cmd is not None
    assert cmd.cli_only is True
    assert cmd.gateway_only is False


def test_process_command_dispatches_to_evolve_cc_handler():
    cli_obj = _make_cli()

    with patch.object(cli_obj, "_handle_evolve_cc_command") as mock_handler:
        assert cli_obj.process_command("/evolve-cc --scope repo") is True

    mock_handler.assert_called_once_with("/evolve-cc --scope repo")
