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


def test_raw_csi_arrow_down_decodes_to_down():
    # ESC [ B  -> down, NOT cancel
    assert read_menu_key(FakeStdscr([27, ord("["), ord("B")])) == NAV_DOWN


def test_raw_csi_arrow_up_decodes_to_up():
    # ESC [ A  -> up
    assert read_menu_key(FakeStdscr([27, ord("["), ord("A")])) == NAV_UP


def test_raw_ss3_arrow_keys_decode():
    # Application cursor mode: ESC O B / ESC O A
    assert read_menu_key(FakeStdscr([27, ord("O"), ord("B")])) == NAV_DOWN
    assert read_menu_key(FakeStdscr([27, ord("O"), ord("A")])) == NAV_UP


def test_translated_key_constants_still_work():
    assert read_menu_key(FakeStdscr([curses.KEY_DOWN])) == NAV_DOWN
    assert read_menu_key(FakeStdscr([curses.KEY_UP])) == NAV_UP


def test_vim_keys():
    assert read_menu_key(FakeStdscr([ord("j")])) == NAV_DOWN
    assert read_menu_key(FakeStdscr([ord("k")])) == NAV_UP


def test_lone_escape_is_cancel():
    # ESC with no continuation byte (getch returns -1) -> genuine cancel.
    assert read_menu_key(FakeStdscr([27])) == NAV_CANCEL


def test_q_is_cancel():
    assert read_menu_key(FakeStdscr([ord("q")])) == NAV_CANCEL


def test_enter_variants_select():
    assert read_menu_key(FakeStdscr([10])) == NAV_SELECT
    assert read_menu_key(FakeStdscr([13])) == NAV_SELECT
    assert read_menu_key(FakeStdscr([curses.KEY_ENTER])) == NAV_SELECT


def test_unhandled_csi_sequence_is_consumed_and_ignored():
    # Delete key (ESC [ 3 ~): must be swallowed whole and map to NAV_NONE so
    # its tail bytes don't leak into a subsequent input() call.
    fake = FakeStdscr([27, ord("["), ord("3"), ord("~"), ord("X")])
    assert read_menu_key(fake) == NAV_NONE
    # The trailing 'X' (a genuinely separate keypress) must remain unconsumed.
    assert fake.keys == [ord("X")]


def test_home_end_csi_sequences_ignored():
    # ESC [ H (Home) and ESC [ F (End) -> NAV_NONE, fully consumed.
    assert read_menu_key(FakeStdscr([27, ord("["), ord("H")])) == NAV_NONE
    assert read_menu_key(FakeStdscr([27, ord("["), ord("F")])) == NAV_NONE


def test_escape_uses_short_timeout_then_restores_blocking():
    fake = FakeStdscr([27, ord("["), ord("B")])
    read_menu_key(fake)
    # A short positive timeout is set to wait for the continuation byte, then
    # blocking mode (-1) is restored.
    assert fake.timeouts[0] > 0
    assert fake.timeouts[-1] == -1
