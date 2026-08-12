"""Tests for the /indicator CLI command and busy-indicator style config.

The /indicator command is registered in COMMAND_REGISTRY (and advertised by
/help, tab-completion and the tips system) but used to have no dispatch branch
in HermesCLI.process_command — so typing it printed "Unknown command:
/indicator". These tests lock in the dispatch wiring and the handler behavior.
"""

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from cli import HermesCLI


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


def _make_cli():
    cli_obj = HermesCLI.__new__(HermesCLI)
    cli_obj.config = {}
    cli_obj.console = MagicMock()
    cli_obj.agent = None
    cli_obj.conversation_history = []
    cli_obj.session_id = None
    cli_obj._pending_input = MagicMock()
    return cli_obj


class TestIndicatorDispatch(unittest.TestCase):
    """The command must route to its handler — not fall through to "Unknown"."""

    def test_indicator_dispatches_to_handler(self):
        cli_obj = _make_cli()
        with patch.object(cli_obj, "_handle_indicator_command") as mock_handler:
            result = cli_obj.process_command("/indicator emoji")

        mock_handler.assert_called_once_with("/indicator emoji")
        self.assertTrue(result)

    def test_indicator_is_not_unknown_command(self):
        cli_obj = _make_cli()
        with (
            patch("cli._cprint") as mock_cprint,
            patch("cli.save_config_value", return_value=True),
        ):
            result = cli_obj.process_command("/indicator emoji")

        printed = " ".join(str(c) for c in mock_cprint.call_args_list)
        self.assertNotIn("Unknown command", printed)
        self.assertTrue(result)


class TestHandleIndicatorCommand(unittest.TestCase):
    def _stub(self, current=None):
        config = {}
        if current is not None:
            config["display"] = {"tui_status_indicator": current}
        return SimpleNamespace(config=config)

    def test_no_args_shows_status(self):
        cli_mod = _import_cli()
        stub = self._stub("emoji")
        with (
            patch.object(cli_mod, "_cprint") as mock_cprint,
            patch.object(cli_mod, "save_config_value") as mock_save,
        ):
            cli_mod.HermesCLI._handle_indicator_command(stub, "/indicator")

        mock_save.assert_not_called()
        printed = " ".join(str(c) for c in mock_cprint.call_args_list)
        self.assertIn("emoji", printed)

    def test_status_argument_shows_status(self):
        cli_mod = _import_cli()
        stub = self._stub()  # no display config -> default kaomoji
        with (
            patch.object(cli_mod, "_cprint") as mock_cprint,
            patch.object(cli_mod, "save_config_value") as mock_save,
        ):
            cli_mod.HermesCLI._handle_indicator_command(stub, "/indicator status")

        mock_save.assert_not_called()
        printed = " ".join(str(c) for c in mock_cprint.call_args_list)
        self.assertIn("kaomoji", printed)

    def test_valid_style_saves_to_config_key(self):
        cli_mod = _import_cli()
        stub = self._stub("kaomoji")
        with (
            patch.object(cli_mod, "_cprint"),
            patch.object(cli_mod, "save_config_value", return_value=True) as mock_save,
        ):
            cli_mod.HermesCLI._handle_indicator_command(stub, "/indicator unicode")

        # Persists to the SAME key the TUI reads, and mirrors it in memory.
        mock_save.assert_called_once_with("display.tui_status_indicator", "unicode")
        self.assertEqual(stub.config["display"]["tui_status_indicator"], "unicode")

    def test_invalid_style_prints_usage_and_does_not_save(self):
        cli_mod = _import_cli()
        stub = self._stub("kaomoji")
        with (
            patch.object(cli_mod, "_cprint") as mock_cprint,
            patch.object(cli_mod, "save_config_value") as mock_save,
        ):
            cli_mod.HermesCLI._handle_indicator_command(stub, "/indicator rainbow")

        mock_save.assert_not_called()
        # The stored value must be untouched.
        self.assertEqual(stub.config["display"]["tui_status_indicator"], "kaomoji")
        printed = " ".join(str(c) for c in mock_cprint.call_args_list)
        self.assertIn("Usage: /indicator", printed)


class TestIndicatorRegistry(unittest.TestCase):
    def test_indicator_in_registry(self):
        from hermes_cli.commands import COMMAND_REGISTRY

        names = [c.name for c in COMMAND_REGISTRY]
        self.assertIn("indicator", names)

    def test_indicator_subcommands_match_handler(self):
        from hermes_cli.commands import COMMAND_REGISTRY
        from hermes_constants import INDICATOR_STYLES

        indicator = next(c for c in COMMAND_REGISTRY if c.name == "indicator")
        self.assertEqual(indicator.category, "Configuration")
        # The registered styles are what the handler accepts — single source of truth.
        self.assertEqual(
            set(indicator.subcommands), set(INDICATOR_STYLES)
        )


if __name__ == "__main__":
    unittest.main()
