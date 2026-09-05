"""Regression tests for arrow-key decoding in the curses menus.

Root cause these guard against: on many terminals/terminfo entries, cursor
keys are delivered to ``getch()`` as raw CSI/SS3 escape byte sequences
(``27, 91, 66`` for arrow-down) even when ``keypad(True)`` is set. The menus
used to treat the leading ``27`` as ESC/cancel, which dumped the setup wizard's
provider/model picker into its numbered "Select [1-N]" fallback the instant a
user pressed up or down.
"""
import sys

import pytest

# curses (and its _curses C extension) is Unix-only; skip the whole module on Windows.
if sys.platform == "win32":
    pytest.skip("curses is not available on Windows", allow_module_level=True)
import curses

from hermes_cli.curses_ui import (
    NAV_CANCEL,
    NAV_DOWN,
    NAV_NONE,
    NAV_SELECT,
    NAV_UP,
    read_menu_key,
)


class FakeStdscr:
    """Minimal stdscr stand-in that replays a queue of getch() byte returns.

    ``getch`` pops from ``keys``; an empty queue yields ``-1`` (matching curses
    non-blocking behavior). ``timeout`` is recorded but otherwise inert.
    """

    def __init__(self, keys):
        self.keys = list(keys)
        self.timeouts = []

    def getch(self):
        return self.keys.pop(0) if self.keys else -1

    def timeout(self, ms):
        self.timeouts.append(ms)




def test_raw_ss3_arrow_keys_decode():
    # Application cursor mode: ESC O B / ESC O A
    assert read_menu_key(FakeStdscr([27, ord("O"), ord("B")])) == NAV_DOWN
    assert read_menu_key(FakeStdscr([27, ord("O"), ord("A")])) == NAV_UP






def test_enter_variants_select():
    assert read_menu_key(FakeStdscr([10])) == NAV_SELECT
    assert read_menu_key(FakeStdscr([13])) == NAV_SELECT
    assert read_menu_key(FakeStdscr([curses.KEY_ENTER])) == NAV_SELECT




