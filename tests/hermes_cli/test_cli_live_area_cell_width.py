"""The classic CLI's live area never sends U+FE0F (VS16) to the terminal.

Regression for #120588. prompt_toolkit measures a cell with ``get_cwidth``, which counts VS16 as
zero, so ``🗜️`` is one cell to it, while a terminal that honours VS16 (Ghostty, kitty, foot,
WezTerm) draws it two cells wide. Every cell right of such a glyph then lands a column off: the
row's last column falls off the edge, and a diff repaint writes over the wrong cells (a timer
reading ``4m 63s``, a vi-mode label reading ``INSERTL``). VS16 reaches the live area from Hermes's
own glyphs (the compressions count, tool emojis in the activity text) and from text it shows
(what the user types, session titles).
"""
import asyncio
import functools
import io

import pytest

VS16 = "\ufe0f"


class _Tty(io.StringIO):
    encoding = "utf-8"

    def isatty(self):
        return True

    def fileno(self):
        return 1


def _first_frame(layout, style, rows=30, columns=120) -> str:
    """Everything the CLI's first paint writes, through the CPR-disabled vt100 output it ships."""
    from prompt_toolkit.application import Application
    from prompt_toolkit.data_structures import Size
    from prompt_toolkit.input import create_pipe_input
    from prompt_toolkit.output.vt100 import Vt100_Output

    tty = _Tty()
    output = Vt100_Output(tty, lambda: Size(rows=rows, columns=columns), term="xterm-256color", enable_cpr=False)
    frames: list[str] = []

    async def run():
        with create_pipe_input() as pipe:
            app = Application(layout=layout, style=style, full_screen=False, input=pipe, output=output)
            painted = asyncio.Event()

            def record(_app):
                frames.append(tty.getvalue())
                painted.set()

            app.after_render += record
            task = asyncio.create_task(app.run_async())
            await asyncio.wait_for(painted.wait(), 5)
            app.exit()
            await asyncio.wait_for(task, 3)

    asyncio.run(run())
    return frames[0]


@pytest.mark.parametrize("modern", [True, False], ids=["glyph-layout", "classic-layout"])
def test_the_live_area_writes_no_vs16(monkeypatch, modern):
    import prompt_toolkit.renderer as pt_renderer
    from prompt_toolkit.key_binding import KeyBindings
    from prompt_toolkit.layout.controls import BufferControl
    import cli as cli_mod
    from cli import HermesCLI

    # Hermes's screen-diff wrapper, installed as HermesCLI.run() installs it.
    monkeypatch.setattr(pt_renderer, "_output_screen_diff", functools.partial(
        cli_mod._hermes_call_output_screen_diff, pt_renderer._output_screen_diff))
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

    written = _first_frame(layout, style)

    for glyph in ("🗜", "✍", "❤"):  # the compressions count, the activity text, the composer
        assert glyph in written, glyph
    assert VS16 not in written
