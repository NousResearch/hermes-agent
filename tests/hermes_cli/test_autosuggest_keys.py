"""Classic CLI ghost-text key acceptance (regression for #151352)."""

import asyncio
from types import SimpleNamespace

import pytest
from prompt_toolkit.application import Application
from prompt_toolkit.application.current import set_app
from prompt_toolkit.auto_suggest import Suggestion
from prompt_toolkit.document import Document
from prompt_toolkit.enums import EditingMode
from prompt_toolkit.input import DummyInput
from prompt_toolkit.key_binding.key_processor import KeyPress
from prompt_toolkit.key_binding.vi_state import InputMode
from prompt_toolkit.keys import Keys
from prompt_toolkit.layout import Layout
from prompt_toolkit.output import DummyOutput
from prompt_toolkit.widgets import TextArea

from hermes_cli.cli_tui_mixin import CLITuiMixin


@pytest.fixture
def composer(monkeypatch):
    cli = CLITuiMixin()
    cli.config = {}
    for overlay in (
        "clarify", "approval", "slash_confirm", "sudo", "secret", "connection",
        "model_picker", "command_palette",
    ):
        setattr(cli, f"_{overlay}_state", None)
    cli._prompt_stash = SimpleNamespace(panel_open=False)
    monkeypatch.setattr(cli, "_tui_voice_record_key_sequence", lambda: ("c-b",))
    return cli


def _press(cli, keys, *, text="okay let", cursor=None, suggestion=" us continue setup",
           mode=EditingMode.EMACS, vi_input=InputMode.INSERT):
    async def dispatch():
        editor = TextArea(multiline=True)
        app = Application(
            layout=Layout(editor), key_bindings=cli._tui_build_key_bindings(),
            editing_mode=mode, input=DummyInput(), output=DummyOutput(),
        )
        editor.buffer.document = Document(text, len(text) if cursor is None else cursor)
        editor.buffer.suggestion = Suggestion(suggestion) if suggestion is not None else None
        app.vi_state.input_mode = vi_input
        with set_app(app):
            app.key_processor.feed_multiple([KeyPress(key) for key in keys])
            app.key_processor.process_keys()
        return editor.text, editor.buffer.cursor_position

    return asyncio.run(dispatch())


@pytest.mark.parametrize("mode", [EditingMode.EMACS, EditingMode.VI])
@pytest.mark.parametrize("key", [Keys.Right, Keys.ControlE, Keys.ControlF])
def test_accepts_whole_suggestion(composer, mode, key):
    expected = "okay let us continue setup"
    assert _press(composer, [key], mode=mode) == (expected, len(expected))


def test_alt_f_accepts_next_word_in_emacs(composer):
    expected = "okay let's "
    assert _press(composer, [Keys.Escape, "f"], suggestion="'s continue setup") == (
        expected, len(expected))


@pytest.mark.parametrize("suggestion", [None, " us continue setup"])
@pytest.mark.parametrize("keys,position", [
    ([Keys.Right], 2), ([Keys.ControlF], 2), ([Keys.ControlE], 8),
    ([Keys.Escape, "f"], 4),
])
def test_keys_keep_cursor_movement_away_from_end(composer, suggestion, keys, position):
    assert _press(composer, keys, cursor=1, suggestion=suggestion) == ("okay let", position)


@pytest.mark.parametrize("keys", [
    [Keys.Right], [Keys.ControlF], [Keys.ControlE], [Keys.Escape, "f"],
])
@pytest.mark.parametrize("suggestion", [None, ""])
def test_keys_without_suggestion_at_end_keep_buffer(composer, keys, suggestion):
    assert _press(composer, keys, suggestion=suggestion) == ("okay let", 8)


def test_connection_overlay_keeps_right(composer, monkeypatch):
    composer._connection_state = {"phase": "authorized", "selected": 0}
    monkeypatch.setattr(composer, "_paint_now", lambda: None, raising=False)
    assert _press(composer, [Keys.Right]) == ("okay let", 8)
    assert composer._connection_state["selected"] == 1


@pytest.mark.parametrize("prompt", ["sudo", "secret", "connection"])
@pytest.mark.parametrize("key", [Keys.Right, Keys.ControlE, Keys.ControlF])
def test_secret_prompts_do_not_accept_history(composer, prompt, key):
    state = {"phase": "form", "fields": [{"secret": True}], "field_index": 0}
    setattr(composer, f"_{prompt}_state", state)
    assert _press(composer, [key]) == ("okay let", 8)


def test_vi_navigation_right_keeps_cursor_movement(composer):
    assert _press(composer, [Keys.Right], cursor=1, mode=EditingMode.VI,
                  vi_input=InputMode.NAVIGATION) == ("okay let", 2)
