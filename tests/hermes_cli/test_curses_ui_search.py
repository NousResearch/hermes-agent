import sys
from types import SimpleNamespace

from hermes_cli import curses_ui
from hermes_cli.curses_ui import (
    _SearchState,
    _filter_indices,
    _handle_active_search_key,
    _move_filtered_cursor,
    _reconcile_cursor,
)


class _FakeCurses:
    KEY_BACKSPACE = 263
    KEY_DOWN = 258
    KEY_ENTER = 343


def test_filter_indices_keeps_all_items_for_blank_query():
    assert _filter_indices(["Anthropic", "OpenAI"], "") == [0, 1]
    assert _filter_indices(["Anthropic", "OpenAI"], "   ") == [0, 1]


def test_filter_indices_matches_subsequences():
    items = ["claude-opus-4-7", "gpt-5.4-codex", "deepseek-v4"]

    assert _filter_indices(items, "co47") == [0]
    assert _filter_indices(items, "gpt5") == [1]


def test_filter_indices_requires_all_tokens():
    items = ["OpenAI Codex", "OpenAI Chat Completions", "Anthropic Claude"]

    assert _filter_indices(items, "open cod") == [0]


def test_reconcile_cursor_moves_to_first_visible_match():
    assert _reconcile_cursor([2, 4], 0) == (2, 0)
    assert _reconcile_cursor([2, 4], 4) == (4, 1)


def test_move_filtered_cursor_wraps_within_matches():
    filtered = [2, 4, 7]

    assert _move_filtered_cursor(filtered, 2, 0, -1) == 7
    assert _move_filtered_cursor(filtered, 7, 2, 1) == 2


def test_active_search_allows_navigation_keys_to_reach_menu_loop():
    search = _SearchState(active=True, query="opus")

    assert _handle_active_search_key(_FakeCurses, _FakeCurses.KEY_DOWN, search) == (
        False,
        False,
        False,
    )
    assert search.active is True
    assert search.query == "opus"


def test_active_search_consumes_query_editing_and_confirm_keys():
    search = _SearchState(active=True, query="op")

    assert _handle_active_search_key(_FakeCurses, ord("u"), search) == (
        True,
        False,
        True,
    )
    assert search.query == "opu"

    assert _handle_active_search_key(_FakeCurses, _FakeCurses.KEY_ENTER, search) == (
        True,
        True,
        False,
    )


class _DriverScreen:
    def __init__(self, keys):
        self.keys = list(keys)

    def clear(self):
        pass

    def getmaxyx(self):
        return 7, 80

    def addnstr(self, *args):
        pass

    def refresh(self):
        pass

    def getch(self):
        return self.keys.pop(0)

    def timeout(self, milliseconds):
        pass


class _DriverCurses:
    A_NORMAL = 0
    A_BOLD = 1
    A_DIM = 2
    COLOR_GREEN = 2
    COLOR_YELLOW = 3
    COLOR_WHITE = 7
    COLORS = 8
    KEY_UP = 1001
    KEY_DOWN = 1002
    KEY_PPAGE = 1003
    KEY_NPAGE = 1004
    KEY_HOME = 1005
    KEY_END = 1006
    KEY_ENTER = 1007
    KEY_BACKSPACE = 1008

    class error(Exception):
        pass

    def __init__(self, screen):
        self.screen = screen

    def wrapper(self, draw):
        draw(self.screen)

    def has_colors(self):
        return False

    def curs_set(self, visible):
        pass


def test_searchable_driver_pages_through_filtered_original_indices(monkeypatch):
    screen = _DriverScreen([
        ord("/"),
        ord("k"),
        ord("e"),
        ord("e"),
        ord("p"),
        _DriverCurses.KEY_NPAGE,
        _DriverCurses.KEY_ENTER,
    ])
    curses_mod = _DriverCurses(screen)
    monkeypatch.setitem(sys.modules, "curses", curses_mod)
    monkeypatch.setattr(curses_ui.sys, "stdin", SimpleNamespace(isatty=lambda: True))

    selected = curses_ui.curses_radiolist(
        "Filtered paging",
        ["aa", "keep-1", "bb", "keep-2", "cc", "keep-3", "dd", "keep-4"],
        searchable=True,
    )

    assert selected == 7
