"""Tests for the /indicator CLI command (classic CLI dispatch + handler).

Regression coverage for the documented ``/indicator`` slash command, which was
registered as a CLI command and surfaced in help/autocomplete but had no handler
in ``HermesCLI.process_command()`` — so it fell through to the "Unknown command"
fallback in the classic CLI even though the TUI and gateway already supported it.
"""

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch


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


class TestHandleIndicatorCommand(unittest.TestCase):
    def _make_cli(self, current="kaomoji"):
        return SimpleNamespace(
            config={"display": {"tui_status_indicator": current}},
        )

    def test_no_args_shows_status(self):
        cli_mod = _import_cli()
        stub = self._make_cli("emoji")
        with (
            patch.object(cli_mod, "_cprint") as mock_cprint,
            patch.object(cli_mod, "save_config_value") as mock_save,
        ):
            cli_mod.HermesCLI._handle_indicator_command(stub, "/indicator")

        mock_save.assert_not_called()
        printed = " ".join(str(c) for c in mock_cprint.call_args_list)
        self.assertIn("emoji", printed)
        self.assertIn("/indicator", printed)

    def test_status_argument_shows_current(self):
        cli_mod = _import_cli()
        stub = self._make_cli("unicode")
        with (
            patch.object(cli_mod, "_cprint") as mock_cprint,
            patch.object(cli_mod, "save_config_value") as mock_save,
        ):
            cli_mod.HermesCLI._handle_indicator_command(stub, "/indicator status")

        mock_save.assert_not_called()
        printed = " ".join(str(c) for c in mock_cprint.call_args_list)
        self.assertIn("unicode", printed)

    def test_status_falls_back_to_default_for_unknown_config(self):
        cli_mod = _import_cli()
        stub = self._make_cli("rainbow")  # not a valid style
        with (
            patch.object(cli_mod, "_cprint") as mock_cprint,
            patch.object(cli_mod, "save_config_value"),
        ):
            cli_mod.HermesCLI._handle_indicator_command(stub, "/indicator")

        printed = " ".join(str(c) for c in mock_cprint.call_args_list)
        # Normalizes to the default rather than echoing the bogus value.
        self.assertIn("kaomoji", printed)
        self.assertNotIn("rainbow", printed)

    def test_valid_argument_sets_style_and_saves(self):
        cli_mod = _import_cli()
        stub = self._make_cli("kaomoji")
        with (
            patch.object(cli_mod, "_cprint"),
            patch.object(cli_mod, "save_config_value", return_value=True) as mock_save,
        ):
            cli_mod.HermesCLI._handle_indicator_command(stub, "/indicator emoji")

        mock_save.assert_called_once_with("display.tui_status_indicator", "emoji")
        # In-memory config is updated so a follow-up status reflects the change.
        self.assertEqual(stub.config["display"]["tui_status_indicator"], "emoji")

    def test_argument_is_case_insensitive(self):
        cli_mod = _import_cli()
        stub = self._make_cli("kaomoji")
        with (
            patch.object(cli_mod, "_cprint"),
            patch.object(cli_mod, "save_config_value", return_value=True) as mock_save,
        ):
            cli_mod.HermesCLI._handle_indicator_command(stub, "/indicator  EMOJI ")

        mock_save.assert_called_once_with("display.tui_status_indicator", "emoji")

    def test_session_only_when_save_fails(self):
        cli_mod = _import_cli()
        stub = self._make_cli("kaomoji")
        with (
            patch.object(cli_mod, "_cprint") as mock_cprint,
            patch.object(cli_mod, "save_config_value", return_value=False),
        ):
            cli_mod.HermesCLI._handle_indicator_command(stub, "/indicator ascii")

        # Even when the disk write fails, the session value is still applied.
        self.assertEqual(stub.config["display"]["tui_status_indicator"], "ascii")
        printed = " ".join(str(c) for c in mock_cprint.call_args_list)
        self.assertIn("session only", printed)

    def test_invalid_argument_prints_usage_and_does_not_save(self):
        cli_mod = _import_cli()
        stub = self._make_cli("kaomoji")
        with (
            patch.object(cli_mod, "_cprint") as mock_cprint,
            patch.object(cli_mod, "save_config_value") as mock_save,
        ):
            cli_mod.HermesCLI._handle_indicator_command(stub, "/indicator nonsense")

        mock_save.assert_not_called()
        self.assertEqual(stub.config["display"]["tui_status_indicator"], "kaomoji")
        printed = " ".join(str(c) for c in mock_cprint.call_args_list)
        self.assertIn("Usage: /indicator", printed)


class TestIndicatorDispatch(unittest.TestCase):
    """End-to-end: process_command routes /indicator instead of "Unknown command"."""

    def _make_cli(self):
        from cli import HermesCLI

        cli_obj = HermesCLI.__new__(HermesCLI)
        cli_obj.config = {"display": {"tui_status_indicator": "kaomoji"}}
        cli_obj.console = MagicMock()
        cli_obj.agent = None
        cli_obj.conversation_history = []
        cli_obj.session_id = None
        cli_obj._pending_input = MagicMock()
        return cli_obj

    def test_indicator_dispatches_to_handler(self):
        cli_obj = self._make_cli()
        with patch.object(cli_obj, "_handle_indicator_command") as mock_handler:
            cli_obj.process_command("/indicator emoji")
        mock_handler.assert_called_once_with("/indicator emoji")

    def test_indicator_not_unknown_command(self):
        cli_obj = self._make_cli()
        with (
            patch("cli._cprint") as mock_cprint,
            patch("cli.save_config_value", return_value=True),
        ):
            cli_obj.process_command("/indicator emoji")
        printed = " ".join(str(c) for c in mock_cprint.call_args_list)
        self.assertNotIn("Unknown command", printed)
        # The change actually landed through the real handler.
        self.assertEqual(cli_obj.config["display"]["tui_status_indicator"], "emoji")


class TestIndicatorRegistry(unittest.TestCase):
    def test_indicator_in_registry(self):
        from hermes_cli.commands import COMMAND_REGISTRY

        names = [c.name for c in COMMAND_REGISTRY]
        assert "indicator" in names

    def test_indicator_subcommands_documented(self):
        from hermes_cli.commands import COMMAND_REGISTRY

        ind = next(c for c in COMMAND_REGISTRY if c.name == "indicator")
        assert ind.cli_only is True
        assert ind.category == "Configuration"
        assert set(ind.subcommands) == {"kaomoji", "emoji", "unicode", "ascii"}
