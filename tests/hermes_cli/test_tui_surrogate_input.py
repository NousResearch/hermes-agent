"""Regression coverage for Win32 emoji input reaching prompt history."""

from queue import Queue
from types import SimpleNamespace

from prompt_toolkit.buffer import Buffer
from prompt_toolkit.history import FileHistory

from cli import HermesCLI


def test_enter_repairs_utf16_surrogate_pair_before_history_write(tmp_path):
    history = FileHistory(str(tmp_path / "history"))
    buffer = Buffer(history=history)
    buffer.text = "brain works in parallel \ud83d\ude02"

    cli = HermesCLI.__new__(HermesCLI)
    cli._attached_images = []
    cli._agent_running = False
    cli._pending_input = Queue()
    cli._tui_multiline_shortcuts = False
    cli._tui_last_text_change = 0.0
    cli._tui_enter_overlay = lambda event: False
    cli._tui_enter_inline_command = lambda event, text, has_images: False
    cli._inline_pastes = lambda buffer: None

    app = SimpleNamespace(current_buffer=buffer, invalidate=lambda: None)
    cli._tui_handle_enter(SimpleNamespace(app=app))

    expected = "brain works in parallel 😂"
    assert cli._pending_input.get_nowait() == expected
    assert list(history.load_history_strings()) == [expected]
