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


def test_text_change_combines_surrogate_pair_before_the_prompt_renders():
    buffer = Buffer()
    cli = HermesCLI.__new__(HermesCLI)
    cli._tui_prev_text_len = 0
    cli._tui_prev_newline_count = 0
    cli._tui_paste_just_collapsed = False
    cli._skip_paste_collapse = False
    cli._tui_paste_over_threshold = lambda text, line_count, threshold_key: False
    cli._recover_terminal_input_modes = lambda **_kwargs: None
    buffer.on_text_changed += cli._tui_on_text_changed

    # Win32 can deliver an astral character as two separate UTF-16 input events.
    # Preserve the incomplete high surrogate, then combine it as soon as the low
    # surrogate arrives so prompt_toolkit renders the emoji before Enter.
    buffer.text = "brain works in parallel \ud83d"
    buffer.cursor_position = len(buffer.text)
    assert buffer.text.endswith("\ud83d")
    buffer.insert_text("\ude02")

    assert buffer.text == "brain works in parallel 😂"
    assert buffer.cursor_position == len(buffer.text)


def test_text_change_keeps_cursor_before_pair_when_between_code_units():
    buffer = Buffer()
    buffer.text = "A\ud83d\ude02B"
    buffer.cursor_position = 2  # Between the high and low UTF-16 code units.
    cli = HermesCLI.__new__(HermesCLI)
    cli._tui_prev_text_len = 0
    cli._tui_prev_newline_count = 0
    cli._tui_paste_just_collapsed = False
    cli._skip_paste_collapse = False
    cli._tui_paste_over_threshold = lambda text, line_count, threshold_key: False
    cli._recover_terminal_input_modes = lambda **_kwargs: None

    cli._tui_on_text_changed(buffer)

    assert buffer.text == "A😂B"
    assert buffer.cursor_position == 1


def test_live_surrogate_repair_keeps_fallback_paste_collapse():
    buffer = Buffer()
    cli = HermesCLI.__new__(HermesCLI)
    cli._tui_prev_text_len = 0
    cli._tui_prev_newline_count = 0
    cli._tui_paste_just_collapsed = False
    cli._skip_paste_collapse = False
    cli._tui_paste_over_threshold = lambda text, line_count, threshold_key: True
    cli._tui_collapse_paste = (
        lambda text, line_count, fallback: f"<collapsed fallback={fallback} len={len(text)}>"
    )
    cli._recover_terminal_input_modes = lambda **_kwargs: None
    buffer.on_text_changed += cli._tui_on_text_changed

    buffer.text = "x" * 100 + "\ud83d\ude02"

    assert buffer.text == "<collapsed fallback=True len=101>"
