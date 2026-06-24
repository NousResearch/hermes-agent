"""Tests for the /whoami and /indicator classic-CLI commands.

Both commands were defined in ``COMMAND_REGISTRY`` and surfaced in the CLI's
``/help`` listing and autocomplete, but ``cli.py``'s ``process_command`` never
dispatched them — so typing either one printed "Unknown command". These tests
cover the handlers and the dispatch wiring that fixes that.
"""

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


class TestHandleWhoamiCommand(unittest.TestCase):
    def test_reports_unrestricted_local_access(self):
        cli_mod = _import_cli()
        stub = SimpleNamespace()
        with patch.object(cli_mod, "_cprint") as mock_cprint:
            cli_mod.HermesCLI._handle_whoami_command(stub)

        printed = " ".join(str(c) for c in mock_cprint.call_args_list)
        self.assertIn("unrestricted", printed.lower())
        self.assertIn("all available", printed.lower())


class TestHandleIndicatorCommand(unittest.TestCase):
    def _make_cli(self, current=None):
        display = {} if current is None else {"tui_status_indicator": current}
        return SimpleNamespace(config={"display": display})

    def test_no_args_shows_current_style(self):
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

    def test_no_args_falls_back_to_default_when_unset(self):
        cli_mod = _import_cli()
        stub = self._make_cli(None)
        with (
            patch.object(cli_mod, "_cprint") as mock_cprint,
            patch.object(cli_mod, "save_config_value"),
        ):
            cli_mod.HermesCLI._handle_indicator_command(stub, "/indicator")

        printed = " ".join(str(c) for c in mock_cprint.call_args_list)
        self.assertIn("kaomoji", printed)

    def test_valid_style_sets_and_saves(self):
        cli_mod = _import_cli()
        stub = self._make_cli("kaomoji")
        with (
            patch.object(cli_mod, "_cprint"),
            patch.object(cli_mod, "save_config_value", return_value=True) as mock_save,
        ):
            cli_mod.HermesCLI._handle_indicator_command(stub, "/indicator emoji")

        mock_save.assert_called_once_with("display.tui_status_indicator", "emoji")
        # In-memory config mirrors the new value for a follow-up read.
        self.assertEqual(stub.config["display"]["tui_status_indicator"], "emoji")

    def test_invalid_style_lists_choices_and_does_not_save(self):
        cli_mod = _import_cli()
        stub = self._make_cli("kaomoji")
        with (
            patch.object(cli_mod, "_cprint") as mock_cprint,
            patch.object(cli_mod, "save_config_value") as mock_save,
        ):
            cli_mod.HermesCLI._handle_indicator_command(stub, "/indicator sparkles")

        mock_save.assert_not_called()
        printed = " ".join(str(c) for c in mock_cprint.call_args_list)
        self.assertIn("Unknown indicator", printed)


class TestSlashDispatchRegression(unittest.TestCase):
    """The exact command token must route to its handler, never "Unknown command"."""

    def test_whoami_dispatches_to_handler(self):
        cli_mod = _import_cli()
        calls = []
        stub = SimpleNamespace(
            _pending_resume_sessions=None,
            _handle_whoami_command=lambda: calls.append("whoami"),
        )
        with patch.object(cli_mod, "_cprint") as mock_cprint:
            result = cli_mod.HermesCLI.process_command(stub, "/whoami")

        self.assertTrue(result)
        self.assertEqual(calls, ["whoami"])
        printed = " ".join(str(c) for c in mock_cprint.call_args_list)
        self.assertNotIn("Unknown command", printed)

    def test_indicator_dispatches_to_handler_with_args(self):
        cli_mod = _import_cli()
        calls = []
        stub = SimpleNamespace(
            _pending_resume_sessions=None,
            _handle_indicator_command=lambda c: calls.append(c),
        )
        with patch.object(cli_mod, "_cprint") as mock_cprint:
            result = cli_mod.HermesCLI.process_command(stub, "/indicator emoji")

        self.assertTrue(result)
        self.assertEqual(calls, ["/indicator emoji"])
        printed = " ".join(str(c) for c in mock_cprint.call_args_list)
        self.assertNotIn("Unknown command", printed)


class TestRegistryWiring(unittest.TestCase):
    def test_handlers_exist_on_cli(self):
        cli_mod = _import_cli()
        self.assertTrue(hasattr(cli_mod.HermesCLI, "_handle_whoami_command"))
        self.assertTrue(hasattr(cli_mod.HermesCLI, "_handle_indicator_command"))

    def test_commands_are_registered_on_cli_surface(self):
        from hermes_cli.commands import COMMANDS, resolve_command

        self.assertIsNotNone(resolve_command("whoami"))
        self.assertIsNotNone(resolve_command("indicator"))
        # Both are advertised on the CLI surface (not gateway_only), which is
        # exactly why the missing dispatch produced "Unknown command".
        self.assertIn("/whoami", COMMANDS)
        self.assertIn("/indicator", COMMANDS)


if __name__ == "__main__":
    unittest.main()
