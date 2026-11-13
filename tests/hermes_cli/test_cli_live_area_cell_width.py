"""A VS16 glyph, and a zero-width character left in a cell of its own, reach the terminal at the
width prompt_toolkit counted.

Regression for #120588. prompt_toolkit measures a cell with ``get_cwidth``, which counts U+FE0F
(VS16) as zero, so ``🗜️`` is one cell to it, while a terminal that honours VS16 (Ghostty, kitty,
foot) draws it two cells wide. Every cell right of such a glyph then lands a column off: the row's
last column falls off the edge, and a diff repaint writes over the wrong cells (a timer reading
``4m 63s``, a vi-mode label reading ``INSERTL``). VS16 reaches the live area from Hermes's own
glyphs (the compressions count, tool emojis in the activity text) and from text it shows (what the
user types, session titles). A zero-width character that ends a line (VS16, a keycap's U+20E3)
keeps a cell of its own, which the renderer still counts as a column. Flags, skin-tone modifiers
and ZWJ sequences are out of scope: prompt_toolkit adds up their code points.

wcwidth's ``wcswidth`` applies the VS16 rule those terminals use, so it measures what they draw.
"""
import asyncio
import functools
import io

import pytest
from wcwidth import wcswidth


class _Tty(io.StringIO):
    encoding = "utf-8"

    def isatty(self):
        return True

    def fileno(self):
        return 1


class _Tap:
    """Passes each string the renderer writes on to the output it was given, and records it with
    what reached the terminal for it."""

    def __init__(self, output, terminal_log, pairs):
        self._output, self._terminal_log, self._pairs = output, terminal_log, pairs

    def write(self, data):
        start = len(self._terminal_log)
        self._output.write(data)
        self._pairs.append((data, "".join(self._terminal_log[start:])))

    def __getattr__(self, name):
        return getattr(self._output, name)


def _first_paint(monkeypatch, layout, style=None, *, fail_first_attempt=False) -> list[tuple[str, str]]:
    """(cell, written) for every string the first paint writes, cells and cursor moves alike: what
    prompt_toolkit's renderer wrote and what reached the terminal for it, painted through Hermes's
    screen-diff wrapper installed as ``HermesCLI.run()`` installs it. ``fail_first_attempt`` raises
    as a corrupt previous screen does (tmux re-attach), so the wrapper's retry paints."""
    import prompt_toolkit.renderer as pt_renderer
    from prompt_toolkit.application import Application
    from prompt_toolkit.data_structures import Size
    from prompt_toolkit.input import create_pipe_input
    from prompt_toolkit.output.vt100 import Vt100_Output
    import cli as cli_mod

    terminal_log: list[str] = []
    pairs: list[tuple[str, str]] = []
    attempts: list[int] = []
    real_diff = pt_renderer._output_screen_diff

    class _Terminal(Vt100_Output):
        def write(self, data):
            terminal_log.append(data)
            super().write(data)

    def diff(app, output, *rest):
        attempts.append(1)
        if fail_first_attempt and len(attempts) == 1:
            raise AttributeError("'cell' object has no attribute 'char'")
        return real_diff(app, _Tap(output, terminal_log, pairs), *rest)

    monkeypatch.setattr(pt_renderer, "_output_screen_diff", functools.partial(
        cli_mod._hermes_call_output_screen_diff, diff))
    output = _Terminal(_Tty(), lambda: Size(rows=30, columns=120), term="xterm-256color", enable_cpr=False)
    first: list[tuple[str, str]] = []

    async def run():
        with create_pipe_input() as pipe:
            app = Application(layout=layout, style=style, full_screen=False, input=pipe, output=output)
            painted = asyncio.Event()

            def record(_app):
                if not painted.is_set():
                    first.extend(pairs)
                    painted.set()

            app.after_render += record
            task = asyncio.create_task(app.run_async())
            await asyncio.wait_for(painted.wait(), 5)
            app.exit()
            await asyncio.wait_for(task, 3)

    asyncio.run(run())
    return first


def _assert_each_cell_keeps_its_width(pairs):
    from prompt_toolkit.utils import get_cwidth

    assert wcswidth("❤\ufe0f") == 2  # the oracle still applies the VS16 rule
    for cell, written in pairs:
        if not cell.strip("\r\n"):  # a cursor move (or an empty cell) passes through as it is
            assert written == cell, (cell, written)
        else:  # the renderer advances one column past a zero-width cell
            assert wcswidth(written) == (get_cwidth(cell) or 1), (cell, written)


@pytest.mark.parametrize("modern", [True, False], ids=["glyph-layout", "classic-layout"])
def test_every_live_area_cell_keeps_its_measured_width(monkeypatch, modern):
    from prompt_toolkit.key_binding import KeyBindings
    from prompt_toolkit.layout.controls import BufferControl
    from cli import HermesCLI

    real_snapshot = HermesCLI._get_status_bar_snapshot
    monkeypatch.setattr(HermesCLI, "_get_status_bar_snapshot", lambda self: {**real_snapshot(self), "compressions": 1})
    monkeypatch.setenv("HERMES_DEFER_AGENT_STARTUP", "1")
    cli = HermesCLI(model="fixture", provider="openai-compat", api_key="fixture", base_url="http://127.0.0.1:1/v1")
    cli._tui_init_run_state()
    cli._modern_layout = modern
    layout, style = cli._tui_build_layout(KeyBindings())
    cli._agent_running = True
    cli._spinner_text = "✍️ Writing calc.py"  # write_file's registered emoji carries VS16
    composer = next(window.content.buffer for window in layout.find_all_windows()
                    if isinstance(window.content, BufferControl))
    composer.text = "❤️ thanks"

    pairs = _first_paint(monkeypatch, layout, style)

    renderer_text = "".join(cell for cell, _ in pairs)
    terminal_text = "".join(written for _, written in pairs)
    for glyph in ("🗜", "✍", "❤"):  # the compressions count, the activity text, the composer
        assert glyph in terminal_text, glyph
    assert "\ufe0f" in renderer_text  # the renderer was handed VS16 to paint
    _assert_each_cell_keeps_its_width(pairs)


@pytest.mark.parametrize("fail_first_attempt", [False, True], ids=["first-paint", "retry"])
def test_a_line_ending_in_a_zero_width_character_keeps_its_last_column(monkeypatch, fail_first_attempt):
    """A line ending in ``❤️`` or a keycap leaves its selector or mark in a cell of its own; written
    with no width, the terminal's cursor ends a column short of where the renderer moves on from, and
    a later diff repaint draws text a column left (the next typed letter over the previous one)."""
    from prompt_toolkit.layout import Layout, Window
    from prompt_toolkit.layout.controls import FormattedTextControl
    from prompt_toolkit.utils import get_cwidth

    lines = ["thanks ❤️", "keycap 1️⃣", "a❤️b mid-line", "☕️ already wide"]
    layout = Layout(Window(FormattedTextControl("\n".join(lines))))

    pairs = _first_paint(monkeypatch, layout, fail_first_attempt=fail_first_attempt)

    zero_width = [cell for cell, _ in pairs if cell.strip("\r\n") and get_cwidth(cell) == 0]
    assert len(zero_width) == 2, pairs  # the selector and the mark left alone at their lines' ends
    assert ("1\ufe0f\u20e3", "1\u20e3") in pairs  # the keycap keeps its mark
    _assert_each_cell_keeps_its_width(pairs)
