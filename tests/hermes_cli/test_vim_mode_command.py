"""display.vim_mode: config-only vi keybindings for the CLI composer."""

from typing import Any, cast
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import cli as cli_mod
from prompt_toolkit.application import Application as RealApplication
from prompt_toolkit.cursor_shapes import CursorShape, ModalCursorShapeConfig
from prompt_toolkit.key_binding.vi_state import InputMode
from prompt_toolkit.layout import Layout
from prompt_toolkit.layout.containers import Window


def _import_cli():
    return cli_mod


class TestVimModeLabel(unittest.TestCase):
    """The status-bar label reflects the live vi input mode; off/no-app yields empty."""

    def test_label_tracks_input_mode(self):
        cli_mod = _import_cli()
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


class TestVimModeCursor(unittest.TestCase):
    """Cursor shape configuration in vim mode vs non-vim mode."""

    def test_modal_cursor_wiring_and_transitions(self):
        """CUR-1 & CUR-2: ModalCursorShapeConfig wired when _vim_mode=True and transitions live."""
        captured_kwargs = []

        def spy_application(*args, **kwargs):
            captured_kwargs.append(kwargs)
            return RealApplication(*args, **kwargs)

        with patch("hermes_cli.cli_tui_runtime_mixin.Application", side_effect=spy_application):
            stub = cast(Any, SimpleNamespace(_vim_mode=True))
            layout = Layout(Window())
            app = cli_mod.HermesCLI._tui_build_application(stub, layout, None, None)

        self.assertEqual(len(captured_kwargs), 1)
        cursor_config = captured_kwargs[0].get("cursor")
        self.assertIsInstance(
            cursor_config,
            ModalCursorShapeConfig,
            "Expected cursor kwarg to be an instance of ModalCursorShapeConfig",
        )

        # CUR-1: verify 5/5 mapping across vi input modes
        mode_mapping = [
            (InputMode.NAVIGATION, CursorShape.BLOCK),
            (InputMode.INSERT, CursorShape.BEAM),
            (InputMode.INSERT_MULTIPLE, CursorShape.BEAM),
            (InputMode.REPLACE, CursorShape.UNDERLINE),
            (InputMode.REPLACE_SINGLE, CursorShape.UNDERLINE),
        ]
        for mode, expected_shape in mode_mapping:
            app.vi_state.input_mode = mode
            self.assertEqual(
                app.cursor.get_cursor_shape(app),
                expected_shape,
                f"Mode {mode} did not resolve to {expected_shape}",
            )

        # CUR-2: reversible transitions on the SAME Application instance (ESC -> i -> R -> ESC)
        app.vi_state.input_mode = InputMode.NAVIGATION
        self.assertEqual(app.cursor.get_cursor_shape(app), CursorShape.BLOCK)
        app.vi_state.input_mode = InputMode.INSERT
        self.assertEqual(app.cursor.get_cursor_shape(app), CursorShape.BEAM)
        app.vi_state.input_mode = InputMode.REPLACE
        self.assertEqual(app.cursor.get_cursor_shape(app), CursorShape.UNDERLINE)
        app.vi_state.input_mode = InputMode.NAVIGATION
        self.assertEqual(app.cursor.get_cursor_shape(app), CursorShape.BLOCK)

    def test_non_vim_preserves_steady_block(self):
        """CUR-3: _vim_mode=False retains steady block cursor kwarg identity and shape."""
        captured_kwargs = []

        def spy_application(*args, **kwargs):
            captured_kwargs.append(kwargs)
            return RealApplication(*args, **kwargs)

        with patch("hermes_cli.cli_tui_runtime_mixin.Application", side_effect=spy_application):
            stub = cast(Any, SimpleNamespace(_vim_mode=False))
            layout = Layout(Window())
            app = cli_mod.HermesCLI._tui_build_application(stub, layout, None, None)

        self.assertEqual(len(captured_kwargs), 1)
        self.assertIn("cursor", captured_kwargs[0])
        self.assertIs(
            captured_kwargs[0]["cursor"],
            cli_mod._STEADY_CURSOR,
            "Expected cursor kwarg to be identical to cli._STEADY_CURSOR",
        )
        self.assertEqual(app.cursor.get_cursor_shape(app), CursorShape.BLOCK)

    def test_non_vim_omits_cursor_kwarg_when_steady_cursor_is_none(self):
        """CUR-3: when _STEADY_CURSOR is None, cursor kwarg is omitted entirely."""
        captured_kwargs = []

        def spy_application(*args, **kwargs):
            captured_kwargs.append(kwargs)
            return RealApplication(*args, **kwargs)

        with patch.object(cli_mod, "_STEADY_CURSOR", None):
            with patch("hermes_cli.cli_tui_runtime_mixin.Application", side_effect=spy_application):
                stub = cast(Any, SimpleNamespace(_vim_mode=False))
                layout = Layout(Window())
                _ = cli_mod.HermesCLI._tui_build_application(stub, layout, None, None)

        self.assertEqual(len(captured_kwargs), 1)
        self.assertNotIn(
            "cursor",
            captured_kwargs[0],
            "Expected 'cursor' kwarg to be omitted when _STEADY_CURSOR is None",
        )


if __name__ == "__main__":
    unittest.main()
