"""Regression tests for CJK/Unicode search input in the ``hermes sessions browse`` picker.

Root cause: the curses session browser read keys with ``getch()``, which returns one
UTF-8 byte per keypress. The first byte of a CJK character (0xE4-0xE9) falls outside the
old ASCII-printable range (32-126), so Chinese/Japanese/Korean input was silently dropped
and the filter never applied. The fix switches to ``get_wch()`` (a ``str`` for typed/wide
characters, an ``int`` for special keys) and branches the key handler on the type.
See https://github.com/NousResearch/hermes-agent/issues/40446
"""
import sys

import pytest

if sys.platform == "win32":  # curses is Unix-only
    pytest.skip("curses is not available on Windows", allow_module_level=True)
import curses as _curses

from hermes_cli.sessions_cmd_browse import _CursesBrowser


class _FakeCurses:
    """Minimal curses stand-in exposing just the constants the browser references."""

    KEY_UP = _curses.KEY_UP
    KEY_DOWN = _curses.KEY_DOWN
    KEY_ENTER = _curses.KEY_ENTER
    KEY_BACKSPACE = _curses.KEY_BACKSPACE


SESSIONS = [
    {"id": "s_cn", "title": "PRD 评审 HRMS 系统架构", "preview": "评审", "source": "cli"},
    {"id": "s_en", "title": "Expand power model", "preview": "power", "source": "cli"},
]


def _browser():
    b = _CursesBrowser(_FakeCurses, [dict(s) for s in SESSIONS], delete_fn=None)
    b.search = ""
    b.filtered = list(b.sessions)
    return b


def test_cjk_characters_populate_the_search_filter():
    b = _browser()
    b._handle_key("评")  # get_wch() returns the full character as a str
    assert b.search == "评"
    assert [s["id"] for s in b.filtered] == ["s_cn"]


def test_ascii_characters_still_populate_the_search_filter():
    b = _browser()
    b._handle_key("p")
    b._handle_key("o")
    assert b.search == "po"
    assert [s["id"] for s in b.filtered] == ["s_en"]


def test_non_printable_control_characters_are_ignored():
    b = _browser()
    b._handle_key("\x00")
    assert b.search == ""
    assert len(b.filtered) == len(b.sessions)


def test_backspace_removes_last_search_character():
    b = _browser()
    b._handle_key("中")
    b._handle_key("\x7f")
    assert b.search == ""
    assert len(b.filtered) == len(b.sessions)


def test_escape_str_key_clears_a_populated_search():
    b = _browser()
    b._handle_key("中")
    b._handle_key("\x1b")
    assert b.search == ""
    assert len(b.filtered) == len(b.sessions)


def test_special_keys_arrive_as_ints_and_still_work():
    # get_wch() delivers special keys (arrows, Enter, Backspace, Esc) as ints.
    b = _browser()
    b._handle_key(_FakeCurses.KEY_DOWN)  # focus moves, no crash, search untouched
    assert b.search == ""
    b._handle_key("a")
    assert b.search == "a"


def test_cjk_multi_character_and_mixed_search():
    b = _browser()
    for ch in "评审":
        b._handle_key(ch)
    assert b.search == "评审"
    assert [s["id"] for s in b.filtered] == ["s_cn"]
