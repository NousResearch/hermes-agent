"""``_cprint(text, raw=...)`` writes *raw* to the terminal verbatim (regression for #157632).

The CLI prints through the real ``prompt_toolkit.print_formatted_text``, which renders any plain
``list`` as its ``repr``: an inline image's transmit came out as tuple text and base64 instead of
drawing, whenever ``_cprint`` printed with no app running (``chat -q … --oneshot``) or on the
app's own loop.
"""

from __future__ import annotations

import asyncio
import io

import pytest
from prompt_toolkit.application import Application
from prompt_toolkit.application.current import create_app_session
from prompt_toolkit.data_structures import Size
from prompt_toolkit.input import create_pipe_input
from prompt_toolkit.output.vt100 import Vt100_Output

import cli
from tests.waiting import async_wait_until

TRANSMITS = [
    pytest.param("\x1b_Ga=T,f=100,q=2;iVBORw0KGgoAAAANSUhEUg==\x1b\\", id="kitty-apc"),
    pytest.param("\x1b7\x1b[3A\x1b[2C\x1b]1337;File=inline=1;width=4;height=4:iVBORw0KGgo=\x07\x1b8",
                 id="iterm2-osc1337"),
]


@pytest.fixture(autouse=True)
def _history_off():
    cli._configure_output_history(False, 200)
    yield
    cli._configure_output_history(True, 200)


@pytest.fixture
def terminal():
    """A prompt_toolkit app session whose output is a VT100 terminal writing into a buffer."""
    buf = io.StringIO()
    output = Vt100_Output(buf, lambda: Size(rows=24, columns=80), term="xterm-kitty")
    with create_pipe_input() as pipe, create_app_session(input=pipe, output=output):
        yield buf


def _assert_written_verbatim_once(written: str, raw: str) -> None:
    assert "ZeroWidthEscape" not in written
    assert written.count(raw) == 1
    assert written.index("row") < written.index(raw)


@pytest.mark.parametrize("raw", TRANSMITS)
def test_raw_escape_reaches_the_terminal_verbatim_with_no_app_running(terminal, raw):
    cli._cprint("row", raw=raw)
    _assert_written_verbatim_once(terminal.getvalue(), raw)


@pytest.mark.parametrize("raw", TRANSMITS)
def test_raw_escape_reaches_the_terminal_verbatim_from_the_apps_own_loop(terminal, raw):
    async def run():
        app = Application()
        running = asyncio.ensure_future(app.run_async())
        await async_wait_until(lambda: app.is_running and app.loop is not None, what="app to start")
        cli._cprint("row", raw=raw)
        await async_wait_until(lambda: raw in terminal.getvalue() or "ZeroWidthEscape" in terminal.getvalue(),
                               what="the row to be painted")
        app.exit()
        await running

    asyncio.run(run())
    _assert_written_verbatim_once(terminal.getvalue(), raw)
