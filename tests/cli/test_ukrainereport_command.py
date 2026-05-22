"""Tests for the built-in /ukrainereport skill command wiring."""

from unittest.mock import MagicMock, patch

import pytest

from cli import HermesCLI
from hermes_cli.commands import GATEWAY_KNOWN_COMMANDS, resolve_command


def _make_cli():
    cli_obj = HermesCLI.__new__(HermesCLI)
    cli_obj.config = {}
    cli_obj.console = MagicMock()
    cli_obj.agent = None
    cli_obj.conversation_history = []
    cli_obj.session_id = "session-ukraine"
    cli_obj._pending_input = MagicMock()
    return cli_obj


def test_ukrainereport_command_is_available_in_cli_registry():
    cmd = resolve_command("ukrainereport")

    assert cmd is not None
    assert cmd.category == "Tools & Skills"
    assert cmd.args_hint == ""
    assert cmd.subcommands == ()
    assert cmd.cli_only is True
    assert "ukrainereport" not in GATEWAY_KNOWN_COMMANDS


@pytest.mark.parametrize(
    "command",
    [
        "/ukrainereport",
        "/ukrainereport FAST",
        "/ukrainereport STANDARD",
        "/ukrainereport DEEP",
    ],
)
def test_ukrainereport_queues_skill_invocation_without_mode(command):
    cli_obj = _make_cli()
    queued_message = "loaded ukrainereport"

    with patch("cli.get_skill_commands", return_value={"/ukrainereport": {"name": "ukrainereport"}}), \
         patch("cli.scan_skill_commands") as mock_scan, \
         patch("cli.build_skill_invocation_message", return_value=queued_message) as mock_build:
        assert cli_obj.process_command(command) is True

    mock_scan.assert_not_called()
    mock_build.assert_called_once_with(
        "/ukrainereport",
        "",
        task_id="session-ukraine",
    )
    cli_obj._pending_input.put.assert_called_once_with(queued_message)


def test_ukrainereport_rescans_skills_when_command_cache_is_missing():
    cli_obj = _make_cli()

    with patch("cli.get_skill_commands", return_value={}), \
         patch("cli.scan_skill_commands") as mock_scan, \
         patch("cli.build_skill_invocation_message", return_value="loaded") as mock_build:
        assert cli_obj.process_command("/ukrainereport") is True

    mock_scan.assert_called_once_with()
    mock_build.assert_called_once_with(
        "/ukrainereport",
        "",
        task_id="session-ukraine",
    )
    cli_obj._pending_input.put.assert_called_once_with("loaded")


def test_ukrainereport_missing_skill_does_not_queue_input():
    cli_obj = _make_cli()

    with patch("cli.get_skill_commands", return_value={}), \
         patch("cli.scan_skill_commands"), \
         patch("cli.build_skill_invocation_message", return_value=None):
        assert cli_obj.process_command("/ukrainereport") is True

    cli_obj._pending_input.put.assert_not_called()
