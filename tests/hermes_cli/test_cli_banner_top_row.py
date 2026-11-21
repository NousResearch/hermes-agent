"""The classic CLI starts its banner on the terminal's top row and its composer on the last rows.

Startup scrolls what the shell left on screen into scrollback, and the banner's clear prints it from
the top row. With cursor position reports off on POSIX, prompt_toolkit would draw the footer right
under the last startup line, mid-screen. Startup follows the cursor's row through what it writes to
stdout (``_RowFollowingStdout``) and hands the rows left below it to the footer as the chrome floor:
the bottom-aligned footer fills them, so the composer sits on the last rows (#4412) and each print
takes the row under the one before. ``/clear`` inside the running CLI does the same from the top row
of the cleared screen.
"""
import errno
import io
import os
import sys
from collections import deque
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from prompt_toolkit.data_structures import Size
from prompt_toolkit.layout.screen import Screen
from prompt_toolkit.output import DummyOutput
from rich.console import Console

ROWS, COLUMNS = 60, 100
BANNER_LINES = 20
CLEAR_SCREEN = "\x1b[2J"
CLEAR_AND_HOME = "\x1b[2J\x1b[H"  # Rich's Console.clear()
STARTUP_SCROLL = "\n" * (ROWS - 1)


def _terminal_stream():
    return io.TextIOWrapper(io.BytesIO(), encoding="utf-8", write_through=True)


def _rich_lines(out):
    console = Console(file=out, force_terminal=True, width=COLUMNS)
    for index in range(BANNER_LINES):
        console.print(f"line {index}")


def _prompt_toolkit_lines(out):
    from prompt_toolkit import print_formatted_text
    from prompt_toolkit.output.vt100 import Vt100_Output
    print_formatted_text("first\nsecond", output=Vt100_Output(out, lambda: Size(rows=ROWS, columns=COLUMNS)))


@pytest.mark.parametrize("write, rows_below", [
    pytest.param(lambda out: None, None, id="nothing-written-yet"),
    pytest.param(lambda out: out.write("banner\n" * 5), None, id="lines-from-an-unknown-row"),
    pytest.param(lambda out: out.write(STARTUP_SCROLL), 1, id="startup-scroll"),
    pytest.param(lambda out: (out.write(STARTUP_SCROLL + CLEAR_AND_HOME), _rich_lines(out)), ROWS - BANNER_LINES,
                 id="scroll-then-clear-then-rich-lines"),
    pytest.param(lambda out: out.write(CLEAR_AND_HOME + "x" * (2 * COLUMNS + 1) + "\n"), ROWS - 3,
                 id="soft-wrapped-line"),
    pytest.param(lambda out: out.write(CLEAR_AND_HOME + "y" * COLUMNS + "\n"), ROWS - 1, id="exactly-the-width"),
    pytest.param(lambda out: out.write(CLEAR_AND_HOME + "界" * (COLUMNS // 2 + 1) + "\n"), ROWS - 2,
                 id="wide-characters"),
    pytest.param(lambda out: (out.write(CLEAR_AND_HOME), _prompt_toolkit_lines(out)), ROWS - 2,
                 id="prompt-toolkit-bytes"),
    pytest.param(lambda out: out.write(CLEAR_AND_HOME + "z\n" * (ROWS + 5)), 1, id="taller-than-the-screen"),
    pytest.param(lambda out: out.write(CLEAR_AND_HOME + "a\nb\nc\n\x1b[2A"), ROWS - 1, id="cursor-up"),
    pytest.param(lambda out: out.write(STARTUP_SCROLL + "\x1b[5A\x1b[2B"), 4, id="cursor-up-and-down"),
])
def test_the_rows_below_are_those_the_terminal_leaves_under_the_cursor(write, rows_below):
    from hermes_cli.cli_render import _RowFollowingStdout

    out = _RowFollowingStdout(_terminal_stream(), os.terminal_size((COLUMNS, ROWS)))
    write(out)
    assert out.rows_below == rows_below


def _short_banner(**kwargs):
    for index in range(BANNER_LINES):
        kwargs["console"].print(f"banner line {index}")


@pytest.fixture
def shell(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("TERM", "xterm-256color")  # Rich sends no clear to a dumb terminal
    monkeypatch.setattr("shutil.get_terminal_size", lambda *_a, **_k: os.terminal_size((COLUMNS, ROWS)))
    from cli import HermesCLI
    from hermes_cli import cli_render

    monkeypatch.setattr(cli_render, "_CHROME_FLOOR", None)
    obj = HermesCLI.__new__(HermesCLI)
    obj.__dict__.update(
        model="test-model", provider="test", api_key="test", base_url="", _provider_source=None, agent=None,
        session_id=None, compact=False, enabled_toolsets=["hermes-core"], disabled_toolsets=[], _app=None,
        console=Console(file=io.StringIO(), force_terminal=True, width=COLUMNS, height=ROWS))
    return obj


def test_a_banner_outside_interactive_startup_leaves_the_footer_rows_alone(shell):
    """``--list-tools`` and ``/clear`` without the running CLI print the banner from the top, with no
    footer below it."""
    from hermes_cli import cli_render

    with patch("cli.get_tool_definitions", return_value=[]), \
         patch("hermes_cli.banner.load_banner_snapshot", return_value=None), \
         patch("hermes_cli.banner.build_welcome_banner", _short_banner):
        shell.show_banner()

    assert cli_render._chrome_floor() is None


@pytest.mark.parametrize("terminal, scrolls, rows_below", [
    pytest.param("xterm-256color", True, "under the output", id="banner-from-the-top"),
    pytest.param("xterm-256color", False, "under the output", id="scroll-failed-then-clear"),
    pytest.param("dumb", True, 1, id="dumb-terminal-no-clear"),
    pytest.param("quiet", True, 1, id="quiet-console"),
    pytest.param("dumb", False, None, id="dumb-terminal-and-scroll-failed"),
])
def test_startup_hands_the_footer_the_rows_left_below_its_output(shell, monkeypatch, terminal, scrolls, rows_below):
    """Rich sends no clear to a dumb terminal or when quiet: the banner prints from the last row, where
    startup's scroll left the cursor. With neither the scroll nor the clear the row stays unknown."""
    import cli as cli_mod
    from hermes_cli.cli_render import _chrome_floor

    screen = _terminal_stream()
    monkeypatch.setattr(sys, "stdout", screen)
    if terminal == "dumb":
        monkeypatch.setenv("TERM", "dumb")
    if not scrolls:
        monkeypatch.setattr(cli_mod, "_scroll_to_last_row", lambda *_a: None)
    shell.console = Console(force_terminal=True, quiet=terminal == "quiet", width=COLUMNS, height=ROWS)  # on sys.stdout
    shell.__dict__.update(
        _inline_images_mode="off", _resumed=False, preloaded_skills=[], _preload_skills_requested=[],
        _startup_skills_line_shown=False,
        _ensure_gateway_session=lambda: None, _show_security_advisories=lambda: None,
        _maybe_offer_first_run_setup=lambda: None, _tui_startup_prewarm_and_warnings=lambda _skin: None,
        _print_random_tip=lambda: shell._console_print("a tip"), _tui_startup_background_maintenance=lambda: None)
    with patch("cli.get_tool_definitions", return_value=[]), \
         patch("hermes_cli.banner.load_banner_snapshot", return_value=None), \
         patch("hermes_cli.banner.build_welcome_banner", _short_banner):
        shell._tui_startup_screen()

    written = screen.buffer.getvalue().decode()
    assert sys.stdout is screen, "startup gives sys.stdout back"
    if rows_below == "under the output":
        rows_printed = written.rsplit(CLEAR_SCREEN, 1)[1].count("\n")
        assert BANNER_LINES <= rows_printed < ROWS - 1, "the startup output is shorter than the terminal"
        rows_below = ROWS - rows_printed  # the banner started on the top row
    assert _chrome_floor() == rows_below


class _HungUpTty(io.StringIO):
    def write(self, text: str) -> int:
        raise OSError(errno.EIO, "the terminal hung up")


@pytest.mark.parametrize(("stream", "rows"), [(_HungUpTty(), ROWS), (io.StringIO(), 2)], ids=["hung up", "two rows"])
def test_a_scroll_that_cannot_happen_writes_nothing(monkeypatch, stream, rows):
    from hermes_cli import cli_render

    monkeypatch.setattr("shutil.get_terminal_size", lambda *_a, **_k: os.terminal_size((COLUMNS, rows)))

    cli_render._scroll_to_last_row(stream)
    assert stream.getvalue() == ""


@pytest.mark.platforms("posix")  # a pty
def test_the_startup_scroll_measures_the_terminal_it_writes_to(shell):
    """The fixture's ``shutil`` reports ``ROWS``, as a stale exported ``$LINES`` would: the scroll goes
    by the tty's own height, which prompt_toolkit paints by."""
    import fcntl
    import select
    import struct
    import termios

    from hermes_cli import cli_render

    tty_rows = ROWS - 30
    leader, follower = os.openpty()
    try:
        fcntl.ioctl(follower, termios.TIOCSWINSZ, struct.pack("HHHH", tty_rows, COLUMNS, 0, 0))
        with os.fdopen(follower, "w", encoding="utf-8", closefd=False) as tty:
            cli_render._scroll_to_last_row(tty)
            written = b""  # read before the close: macOS drops what is unread then
            # A pty may hand a write over in parts: read until it has been quiet for two seconds.
            while select.select([leader], [], [], 2)[0]:
                written += os.read(leader, 4096)
    finally:
        os.close(follower)
        os.close(leader)

    assert written.count(b"\n") == tty_rows - 1


def _clear_inside_the_cli(shell, monkeypatch, output):
    import cli as cli_mod

    monkeypatch.setattr(cli_mod, "_OUTPUT_HISTORY", deque(maxlen=200))
    real_cprint = cli_mod._cprint
    monkeypatch.setattr(cli_mod, "_cprint", lambda *_a, **_k: None)
    resets = []
    shell._app = SimpleNamespace(output=output, renderer=SimpleNamespace(reset=lambda **_k: resets.append(True)))
    shell._confirm_destructive_slash = lambda *_a, **_k: "once"
    shell._print_random_tip = lambda: None
    shell.new_session = lambda silent=False: None
    with patch("cli.get_tool_definitions", return_value=[]), \
         patch("hermes_cli.banner.build_welcome_banner", lambda **_kw: None):
        shell._cmd_clear("/clear")
    monkeypatch.setattr(cli_mod, "_cprint", real_cprint)
    return resets


class _Terminal(DummyOutput):
    """An output that knows its size, as prompt_toolkit's own does from the tty."""

    def __init__(self, rows):
        super().__init__()
        self.rows, self.moved_to = rows, []

    def get_size(self) -> Size:
        return Size(rows=self.rows, columns=COLUMNS)

    def cursor_goto(self, row: int = 0, column: int = 0) -> None:
        self.moved_to.append((row, column))


def test_clear_inside_the_cli_starts_from_the_top_row(shell, monkeypatch):
    """The rows come from the app's output, as prompt_toolkit paints by it: ``shutil`` reads ``$LINES``
    first, which a shell may have exported at another height (here it reports ``ROWS``)."""
    from hermes_cli import cli_render

    terminal = _Terminal(rows=ROWS - 15)
    resets = _clear_inside_the_cli(shell, monkeypatch, terminal)

    assert terminal.moved_to == [(0, 0)], "the cleared screen's banner starts on the top row"
    assert resets, "the erased paint is forgotten, so the first print starts on that row"
    assert cli_render._chrome_floor() == ROWS - 15, "the footer fills the screen; each print takes its rows"


class _LoopRunningElsewhere:
    """The app's event loop, seen from the thread ``/clear`` and its prints run on."""

    def is_running(self) -> bool:
        return True

    def call_soon_threadsafe(self, callback, *args):
        callback(*args)


def test_a_print_after_clear_takes_its_rows_from_the_footer(shell, monkeypatch):
    """The floor steers the next paint: after a two-line print the footer, last drawn 18 rows tall,
    still fills every row below the print, down to the bottom."""
    import cli as cli_mod
    from hermes_cli import cli_render

    terminal = _Terminal(rows=ROWS)
    _clear_inside_the_cli(shell, monkeypatch, terminal)
    last_paint = Screen()
    last_paint.height = 18
    app = SimpleNamespace(
        _is_running=True, loop=_LoopRunningElsewhere(), output=terminal,
        renderer=SimpleNamespace(_last_screen=last_paint, _min_available_height=0, erase=lambda: None,
                                 reset=lambda: None),
        _running_in_terminal=False, _running_in_terminal_f=None,
        _request_absolute_cursor_position=lambda: None, _redraw=lambda: None)
    with patch("prompt_toolkit.application.get_app_or_none", return_value=app):
        cli_mod._cprint("first line\nsecond line")

    assert cli_render._chrome_floor() == ROWS - 2
