"""Tests for the /indicator CLI command."""

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


class TestHandleIndicatorCommand(unittest.TestCase):
    def test_no_args_shows_normalized_status(self):
        cli_mod = _import_cli()
        stub = SimpleNamespace()
        with (
            patch.object(cli_mod, "_cprint") as mock_cprint,
            patch("hermes_cli.config.load_config", return_value={"display": {"tui_status_indicator": " EMOJI "}}),
            patch.object(cli_mod, "save_config_value") as mock_save,
        ):
            cli_mod.HermesCLI._handle_indicator_command(stub, "/indicator")

        mock_save.assert_not_called()
        printed = " ".join(str(call) for call in mock_cprint.call_args_list)
        self.assertIn("emoji", printed.lower())
        self.assertIn("dashboard", printed.lower())

    def test_valid_argument_saves_lowercased_style(self):
        cli_mod = _import_cli()
        stub = SimpleNamespace()
        with (
            patch.object(cli_mod, "_cprint"),
            patch.object(cli_mod, "save_config_value", return_value=True) as mock_save,
        ):
            cli_mod.HermesCLI._handle_indicator_command(stub, "/indicator EMOJI")

        mock_save.assert_called_once_with("display.tui_status_indicator", "emoji")

    def test_invalid_argument_prints_usage_without_saving(self):
        cli_mod = _import_cli()
        stub = SimpleNamespace()
        with (
            patch.object(cli_mod, "_cprint") as mock_cprint,
            patch.object(cli_mod, "save_config_value") as mock_save,
        ):
            cli_mod.HermesCLI._handle_indicator_command(stub, "/indicator sparkle")

        mock_save.assert_not_called()
        printed = " ".join(str(call) for call in mock_cprint.call_args_list)
        self.assertIn("unknown indicator", printed.lower())
        self.assertIn("ascii|emoji|kaomoji|unicode", printed.lower())
