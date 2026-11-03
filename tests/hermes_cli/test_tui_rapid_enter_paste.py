"""Enter arriving mid-paste (no bracketed paste; each newline is its own key event) must not
submit / steer the partial line (#10994). Only the Enter-vs-recent-text-change timing is exercised;
the real prompt_toolkit Buffer carries the text so the newline lands where the user's did."""
from __future__ import annotations

import time
from types import SimpleNamespace
from unittest.mock import MagicMock

from prompt_toolkit.buffer import Buffer

from hermes_cli.cli_tui_mixin import _RAPID_INPUT_ENTER_WINDOW_S


def _shell():
    from cli import HermesCLI
    shell = object.__new__(HermesCLI)
    shell._tui_enter_overlay = lambda event: False
    shell._tui_multiline_shortcuts = False
    shell._attached_images = []
    shell._agent_running = True
    shell._tui_enter_while_busy = MagicMock()
    shell._tui_enter_inline_command = lambda *a, **k: False
    shell._inline_pastes = lambda buf: None
    shell._tui_paste_counter = 0
    shell.config = {}
    shell._tui_prev_text_len = shell._tui_prev_newline_count = 0
    shell._tui_paste_just_collapsed = shell._skip_paste_collapse = False
    return shell


def _event(buf: Buffer):
    return SimpleNamespace(app=SimpleNamespace(current_buffer=buf, invalidate=lambda: None, is_running=False))


def test_enter_right_after_text_arrives_is_a_newline_not_a_steer():
    """Line 1 of a paste followed by Enter within the window stays in the buffer; the same Enter
    after a human-scale pause submits the accumulated multi-line message once."""
    shell = _shell()
    buf = Buffer()
    buf.on_text_changed += shell._tui_on_text_changed
    buf.insert_text("line one")
    shell._tui_handle_enter(_event(buf))          # < 50 ms after the text change
    assert buf.text == "line one\n"
    shell._tui_enter_while_busy.assert_not_called()

    buf.insert_text("line two")
    time.sleep(_RAPID_INPUT_ENTER_WINDOW_S * 3)  # the paste has finished; a human presses Enter
    shell._tui_handle_enter(_event(buf))
    shell._tui_enter_while_busy.assert_called_once()
    assert shell._tui_enter_while_busy.call_args.args[0] == "line one\nline two"


def test_fast_backslash_continuation_still_consumes_the_backslash():
    """`\\` + Enter inside a paste (multiline shortcuts on) behaves like a typed one: the
    backslash is removed and a newline inserted, never a literal backslash left in the text."""
    shell = _shell()
    shell._tui_multiline_shortcuts = True
    buf = Buffer()
    buf.on_text_changed += shell._tui_on_text_changed
    buf.insert_text("alpha \\")
    shell._tui_handle_enter(_event(buf))          # < 50 ms after the text change
    assert buf.text == "alpha \n"
    shell._tui_enter_while_busy.assert_not_called()
