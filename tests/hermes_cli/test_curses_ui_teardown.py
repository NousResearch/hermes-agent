"""Regression tests for the shared curses picker's escape-byte handling.

Two invariants are guarded here, both about raw escape bytes:

1. ``flush_stdin()`` must run on EVERY exit path out of ``curses.wrapper()``.
   ``curses.endwin()`` restores terminal modes but does not drain the OS input
   buffer, so leftover CSI bytes from arrow keys survive into the next
   ``input()``.  Every numbered fallback calls ``input()`` as its next
   statement, so the curses-error path is exactly where the leak hurts most.
2. While the type-to-filter prompt is open, a raw ``27`` must be decoded with
   the escape-sequence continuation probe before it is treated as a lone ESC.
   Terminals that deliver cursor keys as raw CSI otherwise wipe the query on
   every arrow press.
"""
import sys

import pytest

# curses (and its _curses C extension) is Unix-only; skip the whole module on Windows.
if sys.platform == "win32":
    pytest.skip("curses is not available on Windows", allow_module_level=True)
import curses

from hermes_cli import curses_ui
from hermes_cli.curses_ui import _KEEP, NAV_SELECT


class FakeStdscr:
    """Minimal stdscr stand-in that replays a queue of getch() byte returns.

    ``getch`` pops from ``keys``; an empty queue yields ``-1`` (matching curses
    non-blocking behavior). Drawing calls are recorded but otherwise inert.
    """

    def __init__(self, keys=()):
        self.keys = list(keys)
        self.timeouts = []

    def getch(self):
        return self.keys.pop(0) if self.keys else -1

    def timeout(self, ms):
        self.timeouts.append(ms)

    def getmaxyx(self):
        return 24, 80

    def clear(self):
        pass

    def refresh(self):
        pass

    def addnstr(self, *args, **kwargs):
        pass


class _TtyStdin:
    """stdin stand-in that reports a real TTY so the non-TTY guard is skipped."""

    def isatty(self):
        return True


@pytest.fixture
def flushes(monkeypatch):
    """Record every ``flush_stdin()`` call made by the picker driver."""
    calls = []
    monkeypatch.setattr("sys.stdin", _TtyStdin())
    monkeypatch.setattr(curses_ui, "flush_stdin", lambda: calls.append("flush"))
    monkeypatch.setattr(curses, "curs_set", lambda _visibility: None)
    monkeypatch.setattr(curses, "has_colors", lambda: False)
    return calls


def _menu_kwargs(**overrides):
    """Minimal ``_run_curses_menu`` wiring: two rows, select resolves to index."""
    kwargs = dict(
        initial_cursor=0,
        item_count=2,
        draw_header=lambda stdscr, max_y, max_x: 2,
        draw_row=lambda stdscr, y, idx, is_cursor, max_x: None,
        on_action=lambda action, cursor: cursor if action == NAV_SELECT else _KEEP,
        fallback=lambda: "fallback",
        cancel_value="cancelled",
    )
    kwargs.update(overrides)
    return kwargs


def test_flush_stdin_runs_when_picker_is_interrupted(monkeypatch, flushes):
    """Ctrl+C out of the picker must still drain the buffered escape bytes."""

    def _interrupt(_draw):
        raise KeyboardInterrupt

    monkeypatch.setattr(curses, "wrapper", _interrupt)

    result = curses_ui._run_curses_menu(**_menu_kwargs())

    assert result == "cancelled"
    assert flushes == ["flush"]


def test_flush_stdin_runs_before_the_numbered_fallback(monkeypatch, flushes):
    """The fallback's first act is ``input()`` — the drain must precede it."""
    flushes_at_fallback = []

    def _explode(_draw):
        raise curses.error("curses unavailable")

    def _fallback():
        flushes_at_fallback.append(len(flushes))
        return "fallback"

    monkeypatch.setattr(curses, "wrapper", _explode)

    result = curses_ui._run_curses_menu(**_menu_kwargs(fallback=_fallback))

    assert result == "fallback"
    # Ordering, not just count: the fallback must observe the drain as done.
    assert flushes_at_fallback == [1]


def test_flush_stdin_still_runs_on_the_normal_path(monkeypatch, flushes):
    """Guards against a regression that MOVES the flush instead of widening it."""

    def _run(draw):
        draw(FakeStdscr([10]))  # ENTER resolves the menu

    monkeypatch.setattr(curses, "wrapper", _run)

    result = curses_ui.curses_single_select("Pick one", ["alpha", "beta"])

    assert result == 0
    assert flushes == ["flush"]


def test_raw_arrow_escape_does_not_wipe_the_active_search_query(monkeypatch, flushes):
    """A raw CSI arrow-down while searching must move the cursor, not clear the query.

    Keys: ``/`` opens search, ``b`` filters to ``beta``/``banana``, then the raw
    three-byte arrow-down ``ESC [ B``, then ENTER. With the query intact the
    cursor lands on ``banana`` (original index 2); if the leading ``27`` is
    mistaken for a lone ESC the query is wiped and ENTER confirms ``beta`` (1).
    """

    def _run(draw):
        draw(FakeStdscr([ord("/"), ord("b"), 27, ord("["), ord("B"), 10]))

    monkeypatch.setattr(curses, "wrapper", _run)

    result = curses_ui.curses_single_select(
        "Pick one", ["alpha", "beta", "banana"], searchable=True
    )

    assert result == 2


def test_lone_escape_still_stops_the_search_and_restores_the_full_list(
    monkeypatch, flushes
):
    """A genuine lone ESC (no continuation byte) keeps its clear-the-query meaning."""

    def _run(draw):
        # The explicit -1 is the continuation probe timing out, i.e. no byte
        # followed the ESC: a real lone ESC. ENTER then confirms the row the
        # cursor was left on, proving the full list came back.
        draw(FakeStdscr([ord("/"), ord("b"), 27, -1, 10]))

    monkeypatch.setattr(curses, "wrapper", _run)

    result = curses_ui.curses_single_select(
        "Pick one", ["alpha", "beta", "banana"], searchable=True
    )

    assert result == 1  # "beta" — its index in the restored, unfiltered list
