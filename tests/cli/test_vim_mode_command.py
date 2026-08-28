"""Tests for the /vim CLI command and display.vim_mode config handling."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from prompt_toolkit.enums import EditingMode


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


class TestHandleVimCommand(unittest.TestCase):
    """/vim toggles vi editing mode, persists it, and applies it live."""

    def _make_cli(self, vim_mode=False, app=None):
        return SimpleNamespace(
            _vim_mode=vim_mode,
            _app=app,
            _console_print=lambda *a, **k: None,
        )

    def test_status_reports_without_saving(self):
        cli_mod = _import_cli()
        stub = self._make_cli(vim_mode=True)
        printed = []
        stub._console_print = lambda msg: printed.append(str(msg))

        with patch.object(cli_mod, "save_config_value") as mock_save:
            cli_mod.HermesCLI._handle_vim_command(stub, "/vim status")

        mock_save.assert_not_called()
        self.assertTrue(stub._vim_mode)
        self.assertIn("on", " ".join(printed))

    def test_bare_command_toggles_on_and_persists(self):
        cli_mod = _import_cli()
        stub = self._make_cli(vim_mode=False)

        with patch.object(cli_mod, "save_config_value") as mock_save:
            cli_mod.HermesCLI._handle_vim_command(stub, "/vim")

        self.assertTrue(stub._vim_mode)
        mock_save.assert_called_once_with("display.vim_mode", True)

    def test_off_argument_disables_and_persists(self):
        cli_mod = _import_cli()
        stub = self._make_cli(vim_mode=True)

        with patch.object(cli_mod, "save_config_value") as mock_save:
            cli_mod.HermesCLI._handle_vim_command(stub, "/vim off")

        self.assertFalse(stub._vim_mode)
        mock_save.assert_called_once_with("display.vim_mode", False)

    def test_applies_editing_mode_to_running_app(self):
        cli_mod = _import_cli()
        app = SimpleNamespace(editing_mode=EditingMode.EMACS)
        stub = self._make_cli(vim_mode=False, app=app)

        with patch.object(cli_mod, "save_config_value"):
            cli_mod.HermesCLI._handle_vim_command(stub, "/vim on")
        self.assertEqual(app.editing_mode, EditingMode.VI)

        with patch.object(cli_mod, "save_config_value"):
            cli_mod.HermesCLI._handle_vim_command(stub, "/vim off")
        self.assertEqual(app.editing_mode, EditingMode.EMACS)

    def test_no_running_app_is_not_fatal(self):
        cli_mod = _import_cli()
        stub = self._make_cli(vim_mode=False, app=None)

        with patch.object(cli_mod, "save_config_value"):
            cli_mod.HermesCLI._handle_vim_command(stub, "/vim on")

        self.assertTrue(stub._vim_mode)

    def test_invalid_argument_does_not_change_state(self):
        cli_mod = _import_cli()
        stub = self._make_cli(vim_mode=False)
        printed = []
        stub._console_print = lambda msg: printed.append(str(msg))

        with patch.object(cli_mod, "save_config_value") as mock_save:
            cli_mod.HermesCLI._handle_vim_command(stub, "/vim sideways")

        mock_save.assert_not_called()
        self.assertFalse(stub._vim_mode)
        self.assertIn("Usage", " ".join(printed))


class TestVimModeLabel(unittest.TestCase):
    """The status-bar label reflects the live vi input mode."""

    def test_empty_when_vim_mode_off(self):
        cli_mod = _import_cli()
        stub = SimpleNamespace(_vim_mode=False, _app=None)
        self.assertEqual(cli_mod.HermesCLI._vim_mode_label(stub), "")

    def test_empty_when_no_app_yet(self):
        cli_mod = _import_cli()
        stub = SimpleNamespace(_vim_mode=True, _app=None)
        self.assertEqual(cli_mod.HermesCLI._vim_mode_label(stub), "")

    def test_reports_insert_and_normal(self):
        cli_mod = _import_cli()
        from prompt_toolkit.key_binding.vi_state import InputMode

        app = SimpleNamespace(vi_state=SimpleNamespace(input_mode=InputMode.INSERT))
        stub = SimpleNamespace(_vim_mode=True, _app=app)
        self.assertEqual(cli_mod.HermesCLI._vim_mode_label(stub), "INSERT")

        app.vi_state.input_mode = InputMode.NAVIGATION
        self.assertEqual(cli_mod.HermesCLI._vim_mode_label(stub), "NORMAL")

        app.vi_state.input_mode = InputMode.REPLACE
        self.assertEqual(cli_mod.HermesCLI._vim_mode_label(stub), "REPLACE")


class TestVimModeConfigAndRegistration(unittest.TestCase):
    """The feature is opt-in and discoverable."""

    def test_default_is_off(self):
        from hermes_cli.config_defaults import DEFAULT_CONFIG

        self.assertIs(DEFAULT_CONFIG["display"]["vim_mode"], False)

    def test_command_is_registered_with_subcommands(self):
        from hermes_cli.commands import COMMANDS, SUBCOMMANDS

        self.assertIn("/vim", COMMANDS)
        self.assertEqual(SUBCOMMANDS["/vim"], ["on", "off", "status"])


if __name__ == "__main__":
    unittest.main()
