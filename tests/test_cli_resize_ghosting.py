from pathlib import Path


def test_resize_handler_clears_screen_before_prompt_toolkit_redraw():
    """The custom resize handler must blank the terminal before PTK redraws.

    When the terminal shrinks, the emulator reflows previously full-width lines
    into multiple rows. If cli.py calls _original_on_resize() before
    erase_screen(), prompt_toolkit can redraw on top of those ghost rows and
    leave duplicated wrapped content behind.
    """

    source = Path("cli.py").read_text(encoding="utf-8")

    marker = "def _resize_clear_ghosts():"
    start = source.index(marker)
    end = source.index("app._on_resize = _resize_clear_ghosts", start)
    block = source[start:end]

    erase_idx = block.index("app.output.erase_screen()")
    cursor_idx = block.index("app.output.cursor_goto(0, 0)")
    flush_idx = block.index("app.output.flush()")
    original_idx = block.rindex("_original_on_resize()")
    invalidate_idx = block.index("app.invalidate()")

    assert erase_idx < original_idx, "resize handler must clear screen before PTK redraw"
    assert cursor_idx < original_idx, "cursor reset must happen before PTK redraw"
    assert flush_idx < original_idx, "screen clear must be flushed before PTK redraw"
    assert original_idx < invalidate_idx, "redraw invalidation should happen after PTK resize accounting"
