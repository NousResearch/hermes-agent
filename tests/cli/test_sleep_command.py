"""Tests for CLI /sleep command registration and dispatch."""

from unittest.mock import MagicMock, patch

from cli import HermesCLI
from hermes_cli.commands import COMMAND_REGISTRY, resolve_command


def _make_cli():
    cli_obj = HermesCLI.__new__(HermesCLI)
    cli_obj.config = {"quick_commands": {}}
    cli_obj.console = MagicMock()
    cli_obj.agent = None
    cli_obj._app = None
    return cli_obj


def test_sleep_command_is_registered():
    names = [cmd.name for cmd in COMMAND_REGISTRY]
    assert "sleep" in names


def test_sleep_alias_resolves():
    cmd = resolve_command("休眠")
    assert cmd is not None
    assert cmd.name == "sleep"


def test_dream_command_is_not_registered():
    assert resolve_command("dream") is None
    assert resolve_command("做梦") is None


def test_process_command_dispatches_sleep_handler():
    cli_obj = _make_cli()

    with patch.object(cli_obj, "_handle_sleep_command") as mock_sleep:
        assert cli_obj.process_command("/sleep deep --apply") is True

    mock_sleep.assert_called_once_with("/sleep deep --apply")
