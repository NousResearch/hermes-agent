"""The classic CLI starts its banner, and so its prompt, on the terminal's last row (#4412).

``show_banner`` begins with a screen clear, which puts the cursor back at the top: the scroll startup
did before it was undone, and a banner shorter than the terminal left the prompt mid-screen with
blank rows below it. ``/clear`` inside the running CLI cleared and printed from the top the same way.
Both now start from the last row and arm the chrome floor, the CLI's stand-in for the cursor position
report it leaves off on POSIX, so each print keeps the footer on the last rows.
"""
import errno
import io
import os
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


def _row_after_last_clear(output: str) -> int:
    """The cursor's row at the end of ``output``, from its last screen clear (which homes it): each
    newline moves it down a row, and at the bottom the screen scrolls instead."""
    row = 0
    for character in output.rsplit(CLEAR_SCREEN, 1)[1]:
        if character == "\n":
            row = min(row + 1, ROWS - 1)
    return row


def _short_banner(**kwargs):
    for index in range(BANNER_LINES):
        kwargs["console"].print(f"banner line {index}")


@pytest.fixture
def shell(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("TERM", "xterm-256color")  # Rich sends no clear to a dumb terminal
    monkeypatch.setattr("shutil.get_terminal_size", lambda *_a: os.terminal_size((COLUMNS, ROWS)))
    from cli import HermesCLI
    from hermes_cli import cli_render

    monkeypatch.setattr(cli_render, "_CHROME_FLOOR", None)
    obj = HermesCLI.__new__(HermesCLI)
    obj.__dict__.update(
        model="test-model", provider="test", api_key="test", base_url="", _provider_source=None, agent=None,
        session_id=None, compact=False, enabled_toolsets=["hermes-core"], disabled_toolsets=[],
        console=Console(file=io.StringIO(), force_terminal=True, width=COLUMNS, height=ROWS))
    return obj


def _show_banner(shell, **kwargs):
    with patch("cli.get_tool_definitions", return_value=[]), \
         patch("hermes_cli.banner.load_banner_snapshot", return_value=None), \
         patch("hermes_cli.banner.build_welcome_banner", _short_banner):
        shell.show_banner(**kwargs)


def test_the_startup_banner_starts_on_the_last_row(shell):
    from hermes_cli import cli_render

    _show_banner(shell, from_last_row=True)

    assert _row_after_last_clear(shell.console.file.getvalue()) == ROWS - 1, \
        "a banner shorter than the terminal ends on its last row, where the prompt then starts"
    assert cli_render._chrome_floor() == cli_render._CHROME_FLOOR_AT_LAST_ROW, \
        "the footer's rows down to the bottom are known"


def test_a_banner_outside_interactive_startup_starts_at_the_top(shell):
    """``--list-tools`` and ``/clear`` without the running CLI keep the banner where the clear left it:
    no footer sits below it to keep on the last rows."""
    from hermes_cli import cli_render

    _show_banner(shell)

    assert _row_after_last_clear(shell.console.file.getvalue()) < ROWS - 1
    assert cli_render._chrome_floor() is None


@pytest.mark.parametrize("terminal", ["not a terminal", "TERM=dumb", "quiet"])
def test_a_banner_whose_clear_was_dropped_adds_no_blank_screen(shell, monkeypatch, terminal):
    """Rich drops the clear when it writes to no terminal, to a dumb one, or when quiet: the cursor
    was never sent home, so no screenful of newlines follows, and the footer's rows stay unknown."""
    from hermes_cli import cli_render

    if terminal == "TERM=dumb":
        monkeypatch.setenv("TERM", "dumb")
    shell.console = Console(file=io.StringIO(), force_terminal=terminal != "not a terminal", quiet=terminal == "quiet",
                            width=COLUMNS, height=ROWS)

    _show_banner(shell, from_last_row=True)

    assert "\n" * (ROWS - 1) not in shell.console.file.getvalue()
    assert cli_render._chrome_floor() is None


def test_interactive_startup_prints_the_banner_from_the_last_row(shell):
    from hermes_cli import cli_render

    shell.__dict__.update(
        _inline_images_mode="off", _resumed=False, preloaded_skills=[], _preload_skills_requested=[],
        _startup_skills_line_shown=False,
        _ensure_gateway_session=lambda: None, _show_security_advisories=lambda: None,
        _maybe_offer_first_run_setup=lambda: None, _tui_startup_prewarm_and_warnings=lambda _skin: None,
        _print_random_tip=lambda: None, _tui_startup_background_maintenance=lambda: None)
    with patch("cli.get_tool_definitions", return_value=[]), \
         patch("hermes_cli.banner.load_banner_snapshot", return_value=None), \
         patch("hermes_cli.banner.build_welcome_banner", _short_banner):
        shell._tui_print_startup()

    assert _row_after_last_clear(shell.console.file.getvalue()) == ROWS - 1
    assert cli_render._chrome_floor() == cli_render._CHROME_FLOOR_AT_LAST_ROW


class _HungUpTty(io.StringIO):
    def write(self, text: str) -> int:
        raise OSError(errno.EIO, "the terminal hung up")


@pytest.mark.parametrize(("stream", "rows"), [(_HungUpTty(), ROWS), (io.StringIO(), 2)], ids=["hung up", "two rows"])
def test_a_scroll_that_did_not_happen_says_so(monkeypatch, stream, rows):
    from hermes_cli import cli_render

    monkeypatch.setattr("shutil.get_terminal_size", lambda *_a: os.terminal_size((COLUMNS, rows)))

    assert cli_render._scroll_to_last_row(stream) is False


def test_a_banner_whose_rescroll_failed_leaves_the_footer_rows_unknown(shell, monkeypatch):
    import cli as cli_mod
    from hermes_cli import cli_render

    monkeypatch.setattr(cli_mod, "_scroll_to_last_row", lambda _stream: False)
    _show_banner(shell, from_last_row=True)

    assert cli_render._chrome_floor() is None


@pytest.mark.platforms("posix")  # a pty
def test_the_startup_scroll_measures_the_terminal_it_writes_to(shell):
    """The fixture's ``shutil`` reports ``ROWS``, as a stale exported ``$LINES`` would: the scroll goes
    by the tty's own height, which prompt_toolkit paints by, or the prompt starts mid-screen."""
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
            assert cli_render._scroll_to_last_row(tty)
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
        self.rows, self.rows_moved_to = rows, []

    def get_size(self) -> Size:
        return Size(rows=self.rows, columns=COLUMNS)

    def cursor_goto(self, row: int = 0, column: int = 0) -> None:
        self.rows_moved_to.append(row)


def test_clear_inside_the_cli_starts_on_the_last_row(shell, monkeypatch):
    """The row comes from the app's output, as prompt_toolkit paints by it: ``shutil`` reads ``$LINES``
    first, which a shell may have exported at another height (here it reports ``ROWS``)."""
    from hermes_cli import cli_render

    terminal = _Terminal(rows=ROWS - 15)
    resets = _clear_inside_the_cli(shell, monkeypatch, terminal)

    assert terminal.rows_moved_to == [ROWS - 15], "the cleared screen's banner starts on the last row (CUP rows count from 1)"
    assert resets, "the erased paint is forgotten, so the first print starts on that row"
    assert cli_render._chrome_floor() == cli_render._CHROME_FLOOR_AT_LAST_ROW, "the footer's rows down to the bottom are known again"


class _LoopRunningElsewhere:
    """The app's event loop, seen from the thread ``/clear`` and its prints run on."""

    def is_running(self) -> bool:
        return True

    def call_soon_threadsafe(self, callback, *args):
        callback(*args)


def test_a_print_after_clear_keeps_the_footer_down_to_the_last_row(shell, monkeypatch):
    """The floor steers the next paint: the footer, last drawn 18 rows tall from its top down to the
    bottom, sits two rows lower after a two-line print, so it now spans two rows fewer."""
    import cli as cli_mod
    from hermes_cli import cli_render

    footer_rows = 18
    terminal = _Terminal(rows=ROWS)
    _clear_inside_the_cli(shell, monkeypatch, terminal)
    last_paint = Screen()
    last_paint.height = footer_rows
    app = SimpleNamespace(
        _is_running=True, loop=_LoopRunningElsewhere(), output=terminal,
        renderer=SimpleNamespace(_last_screen=last_paint, _min_available_height=0, erase=lambda: None,
                                 reset=lambda: None),
        _running_in_terminal=False, _running_in_terminal_f=None,
        _request_absolute_cursor_position=lambda: None, _redraw=lambda: None)
    with patch("prompt_toolkit.application.get_app_or_none", return_value=app):
        cli_mod._cprint("first line\nsecond line")

    assert cli_render._chrome_floor() == footer_rows - 2
