"""The classic CLI footer never paints prompt_toolkit's "Window too small..." placeholder.

Regression for #57393: a footer row appearing between the renderer's measure pass and the paint
pass (the agent thread starting a turn) made the root ``HSplit`` give up on the whole footer, on
any terminal size.
"""
import asyncio

from prompt_toolkit.application import Application
from prompt_toolkit.data_structures import Size
from prompt_toolkit.input import create_pipe_input
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.layout import HSplit, Layout, Window
from prompt_toolkit.layout.dimension import Dimension
from prompt_toolkit.layout.screen import WritePosition
from prompt_toolkit.output.vt100 import Vt100_Output


class _Tty:
    encoding = "utf-8"

    def __init__(self):
        self.chunks = []

    def write(self, data):
        self.chunks.append(data)

    def flush(self):
        pass

    def isatty(self):
        return True

    def fileno(self):
        return 1

    @property
    def text(self):
        return "".join(self.chunks)


def _render_once(layout, kb, style, tty, rows=50, columns=200):
    """One non-full-screen frame through the same CPR-disabled vt100 output the CLI ships."""
    output = Vt100_Output(tty, lambda: Size(rows=rows, columns=columns), term="xterm-256color", enable_cpr=False)

    async def run():
        with create_pipe_input() as pipe:
            app = Application(layout=layout, key_bindings=kb, style=style, full_screen=False,
                              input=pipe, output=output)
            painted = asyncio.Event()
            app.after_render += lambda _: painted.set()
            task = asyncio.create_task(app.run_async())
            await asyncio.wait_for(painted.wait(), 5)
            app.exit()
            await asyncio.wait_for(task, 3)

    asyncio.run(run())


def test_footer_survives_a_row_appearing_between_measure_and_paint(monkeypatch):
    monkeypatch.setenv("HERMES_DEFER_AGENT_STARTUP", "1")
    from cli import HermesCLI

    cli = HermesCLI(model="fixture", provider="openai-compat", api_key="fixture", base_url="http://127.0.0.1:1/v1")
    cli._tui_init_run_state()
    kb = KeyBindings()
    layout, style = cli._tui_build_layout(kb)
    root = layout.container

    measured = HSplit.preferred_height

    def measure_then_agent_starts(self, width, max_available_height):
        dimension = measured(self, width, max_available_height)
        if self is root:
            # The agent thread flips these right after the renderer measured the footer.
            cli._agent_running = True
            cli._spinner_text = "⠋ Thinking..."
        return dimension

    monkeypatch.setattr(HSplit, "preferred_height", measure_then_agent_starts)
    tty = _Tty()
    _render_once(layout, kb, style, tty)

    assert "Window too small" not in tty.text
    assert "❯" in tty.text, "the composer prompt must survive the frame"


def test_footer_split_clips_from_the_top_when_minimums_overflow():
    from prompt_toolkit.application import DummyApplication, set_app

    from hermes_cli.cli_footer_split import FooterSplit

    top, middle, bottom = (Window(height=Dimension.exact(1)) for _ in range(3))
    split = FooterSplit([top, middle, bottom])
    with set_app(DummyApplication()):
        sizes = split._divide_heights(WritePosition(0, 0, 80, 2))

    # ``_all_children`` interleaves zero-height padding windows between the three children.
    assert sizes == [0, 0, 1, 0, 1], "composer-side rows keep their height; the late top row is clipped"
