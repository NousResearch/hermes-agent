"""Tests for the /busy CLI command and busy-input-mode config handling."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch


def _import_cli():
    import hermes_cli.config as config_mod

    if not hasattr(config_mod, "save_env_value_secure"):
        config_mod.save_env_value_secure = lambda key, value: {
            "success": True,
            "stored_as": key,
            "validated": False,
        }

    import cli as cli_mod

    return cli_mod


class TestHandleBusyCommand(unittest.TestCase):
    def _make_cli(self, busy_input_mode="interrupt"):
        return SimpleNamespace(
            busy_input_mode=busy_input_mode,
            agent=None,
        )

    def test_no_args_shows_status(self):
        cli_mod = _import_cli()
        stub = self._make_cli("queue")
        with (
            patch.object(cli_mod, "_cprint") as mock_cprint,
            patch.object(cli_mod, "save_config_value") as mock_save,
        ):
            cli_mod.HermesCLI._handle_busy_command(stub, "/busy")

        mock_save.assert_not_called()
        printed = " ".join(str(c) for c in mock_cprint.call_args_list)
        self.assertIn("queue", printed)
        self.assertIn("interrupt", printed)

    def test_queue_argument_sets_queue_mode_and_saves(self):
        cli_mod = _import_cli()
        stub = self._make_cli("interrupt")
        with (
            patch.object(cli_mod, "_cprint"),
            patch.object(cli_mod, "save_config_value", return_value=True) as mock_save,
        ):
            cli_mod.HermesCLI._handle_busy_command(stub, "/busy queue")

        self.assertEqual(stub.busy_input_mode, "queue")
        mock_save.assert_called_once_with("display.busy_input_mode", "queue")

    def test_interrupt_argument_sets_interrupt_mode_and_saves(self):
        cli_mod = _import_cli()
        stub = self._make_cli("queue")
        with (
            patch.object(cli_mod, "_cprint"),
            patch.object(cli_mod, "save_config_value", return_value=True) as mock_save,
        ):
            cli_mod.HermesCLI._handle_busy_command(stub, "/busy interrupt")

        self.assertEqual(stub.busy_input_mode, "interrupt")
        mock_save.assert_called_once_with("display.busy_input_mode", "interrupt")

    def test_invalid_argument_prints_usage(self):
        cli_mod = _import_cli()
        stub = self._make_cli()
        with (
            patch.object(cli_mod, "_cprint") as mock_cprint,
            patch.object(cli_mod, "save_config_value") as mock_save,
        ):
            cli_mod.HermesCLI._handle_busy_command(stub, "/busy nonsense")

        mock_save.assert_not_called()
        printed = " ".join(str(c) for c in mock_cprint.call_args_list)
        self.assertIn("Usage: /busy", printed)


class TestBusyCommandRegistry(unittest.TestCase):
    def test_busy_in_registry(self):
        from hermes_cli.commands import COMMAND_REGISTRY

        names = [c.name for c in COMMAND_REGISTRY]
        assert "busy" in names

    def test_busy_subcommands_documented(self):
        from hermes_cli.commands import COMMAND_REGISTRY

        busy = next(c for c in COMMAND_REGISTRY if c.name == "busy")
        assert busy.args_hint == "[queue|interrupt|status]"
        assert busy.category == "Configuration"
