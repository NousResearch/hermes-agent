"""display.vim_mode: config-only vi keybindings for the CLI composer."""

import unittest
from types import SimpleNamespace
from typing import Any, cast

from prompt_toolkit.application import Application as RealApplication
from prompt_toolkit.key_binding.vi_state import InputMode
from prompt_toolkit.layout import Layout
from prompt_toolkit.layout.containers import Window



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
        app.vi_state.input_mode = InputMode.REPLACE_SINGLE
        self.assertEqual(cli_mod.HermesCLI._vim_mode_label(stub), "REPLACE")

    def test_replace_single_label(self):
        """AS-RSR-1-map & AS-RSR-3-stub: REPLACE_SINGLE maps to REPLACE; stub needs no current_buffer."""
        cli_mod = _import_cli()
        app = SimpleNamespace(vi_state=SimpleNamespace(input_mode=InputMode.REPLACE_SINGLE))
        stub = SimpleNamespace(_vim_mode=True, _app=app)
        self.assertEqual(cli_mod.HermesCLI._vim_mode_label(stub), "REPLACE")

    def test_replace_single_fallbacks(self):
        """AS-RSR-3-fallbacks: _vim_mode=False, _app=None, and exceptions degrade to empty string."""
        cli_mod = _import_cli()
        app = SimpleNamespace(vi_state=SimpleNamespace(input_mode=InputMode.REPLACE_SINGLE))
        self.assertEqual(cli_mod.HermesCLI._vim_mode_label(SimpleNamespace(_vim_mode=False, _app=app)), "")
        self.assertEqual(cli_mod.HermesCLI._vim_mode_label(SimpleNamespace(_vim_mode=True, _app=None)), "")
        bad_stub = SimpleNamespace(_vim_mode=True, _app=SimpleNamespace(vi_state=None))
        self.assertEqual(cli_mod.HermesCLI._vim_mode_label(bad_stub), "")

    def test_replace_single_transient_resolves(self):
        """AS-RSR-2-complete, AS-RSR-2-escape, AS-RSR-4-visual-precedence:
        Transient resolution via real key dispatch: character overwrite, escape abort,
        and visual selection precedence evaluated inside active Application context.
        """
        from prompt_toolkit.application import Application
        from prompt_toolkit.application.current import set_app
        from prompt_toolkit.buffer import Buffer
        from prompt_toolkit.enums import EditingMode
        from prompt_toolkit.input import DummyInput
        from prompt_toolkit.key_binding.key_processor import KeyPress
        from prompt_toolkit.key_binding.vi_state import InputMode
        from prompt_toolkit.keys import Keys
        from prompt_toolkit.layout import Layout
        from prompt_toolkit.layout.containers import Window
        from prompt_toolkit.layout.controls import BufferControl
        from prompt_toolkit.output import DummyOutput
        from prompt_toolkit.selection import SelectionState, SelectionType

        cli_mod = _import_cli()
        buf = Buffer()
        buf.text = "hello"
        buf.cursor_position = 0

        app = Application(
            layout=Layout(Window(BufferControl(buffer=buf))),
            editing_mode=EditingMode.VI,
            input=DummyInput(),
            output=DummyOutput(),
        )
        cast(Any, app).timeoutlen = None
        app.vi_state.input_mode = InputMode.NAVIGATION
        stub = cast(Any, SimpleNamespace(_vim_mode=True, _app=app))

        with set_app(app):
            # Initial state: NAVIGATION -> NORMAL
            self.assertEqual(app.vi_state.input_mode, InputMode.NAVIGATION)
            self.assertEqual(cli_mod.HermesCLI._vim_mode_label(stub), "NORMAL")

            # SC-RSR-2-1: real 'r' key -> REPLACE_SINGLE -> REPLACE
            app.key_processor.feed(KeyPress("r"))
            app.key_processor.process_keys()
            self.assertEqual(app.vi_state.input_mode, InputMode.REPLACE_SINGLE)
            self.assertEqual(cli_mod.HermesCLI._vim_mode_label(stub), "REPLACE")

            # real 'x' key -> overwrites one character and returns to NAVIGATION -> NORMAL
            app.key_processor.feed(KeyPress("x"))
            app.key_processor.process_keys()
            self.assertEqual(buf.text, "xello")
            self.assertEqual(buf.cursor_position, 0)
            self.assertEqual(app.vi_state.input_mode, InputMode.NAVIGATION)
            self.assertEqual(cli_mod.HermesCLI._vim_mode_label(stub), "NORMAL")

            # SC-RSR-2-2: real 'r' + Escape aborts without changes, returns to NAVIGATION -> NORMAL
            app.key_processor.feed(KeyPress("r"))
            app.key_processor.process_keys()
            self.assertEqual(app.vi_state.input_mode, InputMode.REPLACE_SINGLE)
            self.assertEqual(cli_mod.HermesCLI._vim_mode_label(stub), "REPLACE")

            app.key_processor.feed(KeyPress(Keys.Escape))
            app.key_processor.process_keys()
            self.assertEqual(buf.text, "xello")
            self.assertEqual(app.vi_state.input_mode, InputMode.NAVIGATION)
            self.assertIsNone(buf.selection_state)
            self.assertEqual(cli_mod.HermesCLI._vim_mode_label(stub), "NORMAL")

            # SC-RSR-4-1: live visual selection disables actual 'r' binding
            bindings = app.key_processor._bindings.get_bindings_for_keys(("r",))
            replace_binding = next(b for b in bindings if b.handler.__name__ == "_replace")
            self.assertTrue(replace_binding.filter())

            buf.selection_state = SelectionState(original_cursor_position=0, type=SelectionType.CHARACTERS)
            self.assertFalse(replace_binding.filter())
            app.key_processor.feed(KeyPress("r"))
            app.key_processor.process_keys()
            self.assertNotEqual(app.vi_state.input_mode, InputMode.REPLACE_SINGLE)


class TestVimModeVisual(unittest.TestCase):
    """Visual labels follow real prompt_toolkit key dispatch and focused selection state."""

    @staticmethod
    def _application(buffers=None, extra_window=None):
        from prompt_toolkit.buffer import Buffer
        from prompt_toolkit.enums import EditingMode
        from prompt_toolkit.input import DummyInput
        from prompt_toolkit.layout.containers import HSplit
        from prompt_toolkit.layout.controls import BufferControl
        from prompt_toolkit.output import DummyOutput

        buffers = buffers if buffers is not None else [Buffer()]
        for buf in buffers:
            buf.text = "hello\nworld"
            buf.cursor_position = 0
        windows = [Window(BufferControl(buffer=buf)) for buf in buffers]
        if extra_window is not None:
            windows.append(extra_window)
        app = RealApplication(
            layout=Layout(HSplit(windows)), editing_mode=EditingMode.VI,
            input=DummyInput(), output=DummyOutput(),
        )
        cast(Any, app).timeoutlen = None
        app.vi_state.input_mode = InputMode.NAVIGATION
        return app, SimpleNamespace(_vim_mode=True, _app=app)

    @staticmethod
    def _press(app, key):
        from prompt_toolkit.key_binding.key_processor import KeyPress

        app.key_processor.feed(KeyPress(key))
        app.key_processor.process_keys()

    def test_real_visual_entry_zero_length_read_only_and_toggle(self):
        cli_mod = _import_cli()
        """AS-VIS-1: v/V/Ctrl-V enter live, zero-length selections; label reads never mutate."""
        from copy import deepcopy
        from prompt_toolkit.application.current import set_app
        from prompt_toolkit.keys import Keys
        from prompt_toolkit.selection import SelectionType

        for key, selection_type, expected in (
            ("v", SelectionType.CHARACTERS, "VISUAL"),
            ("V", SelectionType.LINES, "VISUAL LINE"),
            (Keys.ControlV, SelectionType.BLOCK, "VISUAL BLOCK"),
        ):
            with self.subTest(key=key):
                app, stub = self._application()
                with set_app(app):
                    self._press(app, key)
                    buf = app.current_buffer
                    selection = buf.selection_state
                    self.assertIsNotNone(selection)
                    assert selection is not None
                    self.assertEqual(selection.type, selection_type)
                    self.assertEqual(selection.original_cursor_position, buf.cursor_position)
                    self.assertEqual(app.vi_state.input_mode, InputMode.NAVIGATION)
                    before = deepcopy((vars(selection), vars(app.vi_state), buf.text, buf.cursor_position))
                    for _ in range(2):
                        self.assertEqual(cast(Any, cli_mod.HermesCLI)._vim_mode_label(stub), expected)
                        self.assertIs(buf.selection_state, selection)
                        self.assertEqual(
                            (vars(selection), vars(app.vi_state), buf.text, buf.cursor_position), before,
                        )
                    self._press(app, key)
                    self.assertIsNone(buf.selection_state)
                    self.assertEqual(cast(Any, cli_mod.HermesCLI)._vim_mode_label(stub), "NORMAL")

    def test_real_visual_exit_matrix(self):
        cli_mod = _import_cli()
        """AS-VIS-2-exit-restore: all three submodes exit through toggle/Escape/delete/yank."""
        from prompt_toolkit.application.current import set_app
        from prompt_toolkit.keys import Keys

        for entry in ("v", "V", Keys.ControlV):
            for exit_key in (entry, Keys.Escape, "d", "y"):
                with self.subTest(entry=entry, exit=exit_key):
                    app, stub = self._application()
                    with set_app(app):
                        self._press(app, entry)
                        self.assertIsNotNone(app.current_buffer.selection_state)
                        self.assertTrue(cast(Any, cli_mod.HermesCLI)._vim_mode_label(stub).startswith("VISUAL"))
                        self._press(app, exit_key)
                        self.assertIsNone(app.current_buffer.selection_state)
                        self.assertEqual(app.vi_state.input_mode, InputMode.NAVIGATION)
                        self.assertEqual(cast(Any, cli_mod.HermesCLI)._vim_mode_label(stub), "NORMAL")

    def test_real_visual_type_switches(self):
        cli_mod = _import_cli()
        """AS-VIS-2-type-switch: changing selection type never reports NORMAL."""
        from prompt_toolkit.application.current import set_app
        from prompt_toolkit.keys import Keys
        from prompt_toolkit.selection import SelectionType

        app, stub = self._application()
        with set_app(app):
            for key, selection_type, expected in (
                ("v", SelectionType.CHARACTERS, "VISUAL"),
                ("V", SelectionType.LINES, "VISUAL LINE"),
                (Keys.ControlV, SelectionType.BLOCK, "VISUAL BLOCK"),
                ("v", SelectionType.CHARACTERS, "VISUAL"),
            ):
                self._press(app, key)
                selection = app.current_buffer.selection_state
                self.assertIsNotNone(selection)
                assert selection is not None
                self.assertEqual(selection.type, selection_type)
                self.assertEqual(cast(Any, cli_mod.HermesCLI)._vim_mode_label(stub), expected)

    def test_focused_buffer_and_dummy_buffer(self):
        cli_mod = _import_cli()
        """AS-VIS-3: selection on an unfocused buffer is isolated; dummy focus is NORMAL."""
        from prompt_toolkit.application.current import set_app
        from prompt_toolkit.buffer import Buffer
        from prompt_toolkit.layout.controls import FormattedTextControl

        first, second = Buffer(name="first"), Buffer(name="second")
        non_buffer = Window(FormattedTextControl("focus target", focusable=True))
        app, stub = self._application([first, second], extra_window=non_buffer)
        with set_app(app):
            app.layout.focus(first)
            self._press(app, "v")
            self.assertEqual(cast(Any, cli_mod.HermesCLI)._vim_mode_label(stub), "VISUAL")
            app.layout.focus(second)
            self.assertEqual(cast(Any, cli_mod.HermesCLI)._vim_mode_label(stub), "NORMAL")
            self.assertIsNotNone(first.selection_state)
            app.layout.focus(first)
            self.assertEqual(cast(Any, cli_mod.HermesCLI)._vim_mode_label(stub), "VISUAL")
            app.layout.focus(non_buffer)
            self.assertEqual(app.current_buffer.name, "dummy-buffer")
            self.assertIsNone(app.current_buffer.selection_state)
            self.assertEqual(cast(Any, cli_mod.HermesCLI)._vim_mode_label(stub), "NORMAL")

    def test_visual_stub_mapping_and_other_modes_take_precedence(self):
        cli_mod = _import_cli()
        """AS-VIS-1/3/4: default mapping, sentinel read, fallbacks, and predecessor labels."""
        from prompt_toolkit.selection import SelectionState, SelectionType

        app = SimpleNamespace(
            vi_state=SimpleNamespace(input_mode=InputMode.NAVIGATION),
            current_buffer=SimpleNamespace(selection_state=None),
        )
        stub = SimpleNamespace(_vim_mode=True, _app=app)
        for selection_type, expected in (
            (SelectionType.CHARACTERS, "VISUAL"),
            (SelectionType.LINES, "VISUAL LINE"),
            (SelectionType.BLOCK, "VISUAL BLOCK"),
            ("future-selection-type", "VISUAL"),
        ):
            app.current_buffer.selection_state = SimpleNamespace(type=selection_type)
            self.assertEqual(cast(Any, cli_mod.HermesCLI)._vim_mode_label(stub), expected)
        app.current_buffer.selection_state = SelectionState(type=SelectionType.CHARACTERS)
        for mode, expected in (
            (InputMode.INSERT, "INSERT"), (InputMode.INSERT_MULTIPLE, "INSERT"),
            (InputMode.REPLACE, "REPLACE"), (InputMode.REPLACE_SINGLE, "REPLACE"),
        ):
            app.vi_state.input_mode = mode
            self.assertEqual(cast(Any, cli_mod.HermesCLI)._vim_mode_label(stub), expected)
        self.assertEqual(cast(Any, cli_mod.HermesCLI)._vim_mode_label(SimpleNamespace(_vim_mode=False, _app=app)), "")
        self.assertEqual(cast(Any, cli_mod.HermesCLI)._vim_mode_label(SimpleNamespace(_vim_mode=True, _app=None)), "")
        app.vi_state.input_mode = InputMode.NAVIGATION
        del app.current_buffer
        self.assertEqual(cast(Any, cli_mod.HermesCLI)._vim_mode_label(stub), "NORMAL")

    def test_visual_keys_exist_raw_and_resolve_in_every_classic_catalog(self):
        """AS-VIS-5: raw keys cannot be masked by English fallback; no separate r-mode key."""
        from pathlib import Path
        import hermes_yaml as yaml
        from agent.i18n import t

        locales = Path(__file__).resolve().parents[2] / "locales"
        keys = ("vim_visual", "vim_visual_line", "vim_visual_block")
        for lang in ("en", "af", "ar", "de", "es", "fr", "ga", "hu", "it", "ja", "ko", "pt", "ru", "tr", "uk", "zh-hant", "zh"):
            with self.subTest(lang=lang):
                raw = yaml.safe_load((locales / f"{lang}.yaml").read_text(encoding="utf-8"))
                status = raw["cli"]["status_bar"]
                self.assertNotIn("vim_replace_single", status)
                for key in keys:
                    self.assertIsInstance(status[key], str)
                    self.assertTrue(status[key].strip())
                    dotted = f"cli.status_bar.{key}"
                    self.assertNotEqual(t(dotted, lang=lang), dotted)
                    self.assertEqual(t(dotted, lang=lang), status[key])




if __name__ == "__main__":
    unittest.main()
