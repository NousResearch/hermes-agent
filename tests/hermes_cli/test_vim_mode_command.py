"""display.vim_mode: config-only vi keybindings for the CLI composer."""

import unittest
from types import SimpleNamespace

from prompt_toolkit.enums import EditingMode


def _import_cli():
    import cli as cli_mod

    return cli_mod


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


class TestNoSlashCommand(unittest.TestCase):
    """vim_mode is a config key only: no /vim command is registered."""

    def test_vim_not_in_command_registry(self):
        from hermes_cli.commands import COMMAND_REGISTRY

        self.assertNotIn("vim", {c.name for c in COMMAND_REGISTRY})


if __name__ == "__main__":
    unittest.main()
