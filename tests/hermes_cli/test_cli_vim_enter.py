"""Vim Enter contracts through real CLI and PromptSession key processors."""

from types import SimpleNamespace

import pytest
from prompt_toolkit import PromptSession
from prompt_toolkit.application import Application
from prompt_toolkit.application.current import set_app
from prompt_toolkit.buffer import Buffer
from prompt_toolkit.document import Document
from prompt_toolkit.enums import EditingMode
from prompt_toolkit.filters import Never
from prompt_toolkit.input import DummyInput
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.key_binding.key_processor import KeyPress
from prompt_toolkit.key_binding.vi_state import InputMode
from prompt_toolkit.keys import Keys
from prompt_toolkit.layout import BufferControl, Layout, Window
from prompt_toolkit.output import DummyOutput

from hermes_cli.cli_tui_mixin import CLITuiMixin
from hermes_cli.cli_vim_enter import vim_composer_session_kwargs
from hermes_cli.config import load_config_readonly
from hermes_constants import get_hermes_home

ENTER = KeyPress(Keys.ControlM, "\r")


def press(app, *keys):
    with set_app(app):
        app.key_processor.feed_multiple([k if isinstance(k, KeyPress) else KeyPress(k) for k in keys])
        app.key_processor.process_keys()


def rig(surface, vim=True, modal=False):
    submitted = []
    if surface == "gateway":
        (get_hermes_home() / "config.yaml").write_text(f"display:\n  vim_mode: {str(vim).lower()}\n")
        kwargs = vim_composer_session_kwargs(load_config_readonly(), composer_active=lambda: not modal)
        session = PromptSession(input=DummyInput(), output=DummyOutput(), **kwargs)
        app, buf = session.app, session.default_buffer
        buf.validate_while_typing = Never()
        buf.complete_while_typing = Never()
        buf.auto_suggest = None
        buf.accept_handler = lambda b: submitted.append(b.text) or True
    else:
        def submit(_event, buf):
            submitted.append(buf.text)
        cli = SimpleNamespace(_tui_enter_overlay=lambda event: modal and not submitted.append("answer"),
                              _tui_submit_buffer=submit, _tui_multiline_shortcuts=True,
                              _tui_last_text_change=0.0)
        kb = KeyBindings()
        kb.add("enter")(lambda event: CLITuiMixin._tui_handle_enter(cli, event))
        kb.add("escape", "enter")(lambda event: event.current_buffer.insert_text("\n"))
        buf = Buffer(multiline=True)
        app = Application(layout=Layout(Window(BufferControl(buffer=buf))), key_bindings=kb,
                          editing_mode=EditingMode.VI if vim else EditingMode.EMACS,
                          input=DummyInput(), output=DummyOutput())
    app.timeoutlen = None
    return app, buf, submitted


@pytest.mark.parametrize("surface", ["gateway", "in_process"])
def test_insert_newline_at_caret_then_normal_submit(surface):
    app, buf, submitted = rig(surface)
    buf.document = Document("abcd", 2)
    press(app, ENTER)
    assert (buf.text, submitted) == ("ab\ncd", [])
    app.vi_state.input_mode = InputMode.NAVIGATION
    press(app, ENTER)
    assert submitted == ["ab\ncd"]


@pytest.mark.parametrize("surface", ["gateway", "in_process"])
@pytest.mark.parametrize("pending", ["count", "operator", "visual", "replace_single"])
def test_incomplete_vim_commands_never_submit(surface, pending):
    app, buf, submitted = rig(surface)
    buf.document = Document("first\nsecond", 0)
    app.vi_state.input_mode = InputMode.NAVIGATION
    if pending == "count":
        press(app, "3")
    elif pending == "operator":
        press(app, "d")
    elif pending == "visual":
        press(app, "v")
    else:
        press(app, "r")
    press(app, ENTER)
    assert submitted == []
    assert buf.text == "first\nsecond"
    assert app.vi_state.operator_func is None
    assert app.vi_state.input_mode == InputMode.NAVIGATION


@pytest.mark.parametrize("surface", ["gateway", "in_process"])
def test_modified_enter_paste_and_non_vim_keep_their_contract(surface):
    app, buf, submitted = rig(surface)
    buf.document = Document("draft", 5)
    press(app, Keys.Escape, ENTER)  # Alt+Enter must remain a newline, not a submit.
    assert (buf.text, submitted) == ("draft\n", [])
    press(app, KeyPress(Keys.BracketedPaste, "first\nsecond"))
    assert submitted == []
    assert "first\nsecond" in buf.text
    app, buf, submitted = rig(surface, vim=False)
    buf.text = "ordinary"
    press(app, ENTER)
    assert submitted == ["ordinary"]


@pytest.mark.parametrize("surface", ["gateway", "in_process"])
def test_modal_answers_keep_enter(surface):
    app, buf, submitted = rig(surface, modal=True)
    buf.text = "yes"
    press(app, ENTER)
    assert submitted == (["yes"] if surface == "gateway" else ["answer"])
