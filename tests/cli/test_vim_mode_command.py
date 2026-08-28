"""Tests for the /vim CLI command and display.vim_mode config handling."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from prompt_toolkit.enums import EditingMode


def _import_cli():
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

    def test_toggle_persists_and_applies_to_running_app(self):
        cli_mod = _import_cli()
        app = SimpleNamespace(editing_mode=EditingMode.EMACS)
        stub = self._make_cli(vim_mode=False, app=app)

        with patch.object(cli_mod, "save_config_value") as mock_save:
            cli_mod.HermesCLI._handle_vim_command(stub, "/vim")
        self.assertTrue(stub._vim_mode)
        self.assertEqual(app.editing_mode, EditingMode.VI)
        mock_save.assert_called_once_with("display.vim_mode", True)

        with patch.object(cli_mod, "save_config_value") as mock_save:
            cli_mod.HermesCLI._handle_vim_command(stub, "/vim off")
        self.assertFalse(stub._vim_mode)
        self.assertEqual(app.editing_mode, EditingMode.EMACS)
        mock_save.assert_called_once_with("display.vim_mode", False)

    def test_status_and_invalid_args_change_nothing(self):
        cli_mod = _import_cli()
        printed = []
        stub = self._make_cli(vim_mode=True)
        stub._console_print = lambda msg: printed.append(str(msg))

        with patch.object(cli_mod, "save_config_value") as mock_save:
            cli_mod.HermesCLI._handle_vim_command(stub, "/vim status")
            cli_mod.HermesCLI._handle_vim_command(stub, "/vim sideways")

        mock_save.assert_not_called()
        self.assertTrue(stub._vim_mode)
        self.assertIn("Usage", " ".join(printed))


class TestVimModeLabel(unittest.TestCase):
    """The status-bar label reflects the live vi input mode; off/no-app yields empty."""

    def test_label_tracks_input_mode(self):
        cli_mod = _import_cli()
        from prompt_toolkit.key_binding.vi_state import InputMode

        self.assertEqual(
            cli_mod.HermesCLI._vim_mode_label(SimpleNamespace(_vim_mode=False, _app=None)), "")
        self.assertEqual(
            cli_mod.HermesCLI._vim_mode_label(SimpleNamespace(_vim_mode=True, _app=None)), "")

        app = SimpleNamespace(vi_state=SimpleNamespace(input_mode=InputMode.INSERT))
        stub = SimpleNamespace(_vim_mode=True, _app=app)
        self.assertEqual(cli_mod.HermesCLI._vim_mode_label(stub), "INSERT")
        app.vi_state.input_mode = InputMode.NAVIGATION
        self.assertEqual(cli_mod.HermesCLI._vim_mode_label(stub), "NORMAL")
        app.vi_state.input_mode = InputMode.REPLACE
        self.assertEqual(cli_mod.HermesCLI._vim_mode_label(stub), "REPLACE")


if __name__ == "__main__":
    unittest.main()
