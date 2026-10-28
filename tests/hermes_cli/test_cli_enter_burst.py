"""Text + Enter arriving in ONE read (``tmux send-keys 'text' Enter``, a PTY write of ``text\\r``)
submits, while Enters inside an unbracketed multi-line paste stay newlines (#10994).

The real prompt_toolkit Buffer carries the text; the app exposes the two things the handler reads
from prompt_toolkit: the key processor's queue (keys of the same read still to be processed) and
the event loop, whose ``call_later`` is captured so the quiet-window callback runs on demand. The
burst clock is a controlled one, advanced explicitly, so a slow runner cannot change the verdict.
"""
from __future__ import annotations

import queue
from collections import deque
from types import SimpleNamespace

import pytest
from prompt_toolkit.buffer import Buffer
from prompt_toolkit.document import Document

from hermes_cli import cli_enter_burst
from hermes_cli.cli_enter_burst import BURST_END_QUIET_S, RAPID_INPUT_ENTER_WINDOW_S


class _Clock:
    def __init__(self):
        self.now = 1000.0

    def __call__(self):
        return self.now

    def advance(self, seconds):
        self.now += seconds


class _Loop:
    def __init__(self, clock):
        self.clock = clock
        self.scheduled = []

    def call_later(self, delay, callback):
        self.scheduled.append((delay, callback))

    def run_scheduled(self):
        """Let the quiet window elapse with no input, then fire what was scheduled."""
        pending, self.scheduled = self.scheduled, []
        for delay, callback in pending:
            self.clock.advance(delay)
            callback()


@pytest.fixture
def clock(monkeypatch):
    fake = _Clock()
    monkeypatch.setattr(cli_enter_burst, "monotonic", fake)
    return fake


def _shell():
    from cli import HermesCLI
    shell = object.__new__(HermesCLI)
    shell._tui_enter_overlay = lambda event: False
    shell._tui_multiline_shortcuts = False
    shell._attached_images = []
    shell._agent_running = False
    shell._pending_input = queue.Queue()
    shell._tui_enter_inline_command = lambda *a, **k: False
    shell._inline_pastes = lambda buffer: None
    shell._tui_paste_counter = 0
    shell.config = {}
    shell._tui_prev_text_len = shell._tui_prev_newline_count = 0
    shell._tui_paste_just_collapsed = shell._skip_paste_collapse = False
    return shell


def _rig(clock):
    shell, buf, loop = _shell(), Buffer(), _Loop(clock)
    buf.on_text_changed += shell._tui_on_text_changed
    app = SimpleNamespace(current_buffer=buf, invalidate=lambda: None, is_running=True,
                          loop=loop, key_processor=SimpleNamespace(input_queue=deque()))
    return shell, buf, loop, app


def _enter(shell, app, *, more_keys_in_read: bool):
    app.key_processor.input_queue = deque(["next-key"] if more_keys_in_read else [])
    shell._tui_handle_enter(SimpleNamespace(app=app))


def _submitted(shell):
    items = []
    while not shell._pending_input.empty():
        items.append(shell._pending_input.get_nowait())
    return items


def test_text_and_enter_in_one_write_submits_after_the_quiet_window(clock):
    shell, buf, loop, app = _rig(clock)
    buf.insert_text("/help")
    _enter(shell, app, more_keys_in_read=False)   # same read, same instant as the text
    # Not an ordinary submit: a provisional newline waits for the quiet window.
    assert _submitted(shell) == []
    assert buf.text == "/help\n"
    assert [delay for delay, _ in loop.scheduled] == [BURST_END_QUIET_S]
    loop.run_scheduled()                          # input stayed quiet
    assert _submitted(shell) == ["/help"]
    assert buf.text == ""


def test_scripted_edit_inside_a_draft_submits_the_whole_draft(clock):
    """A script moves the cursor into an existing draft and writes text + Enter: the provisional
    newline lands at the cursor, so settling must restore the pre-Enter draft, not drop its last
    character."""
    shell, buf, loop, app = _rig(clock)
    buf.document = Document("hello world", cursor_position=5)
    clock.advance(10)                             # the draft was typed long ago
    buf.insert_text(", big")                      # the scripted write: text...
    _enter(shell, app, more_keys_in_read=False)   # ...+ Enter in the same read
    assert buf.text == "hello, big\n world"
    loop.run_scheduled()
    assert _submitted(shell) == ["hello, big world"]


def test_enter_after_a_pause_is_an_ordinary_submit(clock):
    shell, buf, loop, app = _rig(clock)
    buf.insert_text("hi")
    clock.advance(RAPID_INPUT_ENTER_WINDOW_S * 2)  # a human-scale gap before Enter
    _enter(shell, app, more_keys_in_read=False)
    assert _submitted(shell) == ["hi"]
    assert loop.scheduled == []


def test_unbracketed_multiline_paste_keeps_newlines_and_never_submits(clock):
    """One read: line + Enter + line + Enter + line + Enter (tmux paste-buffer without -p, with
    a trailing newline), then a paste split across reads; nothing is submitted in either."""
    shell, buf, loop, app = _rig(clock)
    buf.insert_text("line one")
    _enter(shell, app, more_keys_in_read=True)
    buf.insert_text("line two")
    _enter(shell, app, more_keys_in_read=True)
    buf.insert_text("line three")
    _enter(shell, app, more_keys_in_read=False)   # trailing newline of the paste
    loop.run_scheduled()
    assert buf.text == "line one\nline two\nline three\n"
    assert _submitted(shell) == []

    shell, buf, loop, app = _rig(clock)
    buf.insert_text("chunk one")
    _enter(shell, app, more_keys_in_read=False)   # the read ended right after this Enter...
    clock.advance(BURST_END_QUIET_S / 5)
    buf.insert_text("chunk two")                  # ...and the next read arrived in the window
    loop.run_scheduled()
    assert buf.text == "chunk one\nchunk two"
    assert _submitted(shell) == []
