"""Regression tests for arrow-key decoding in the curses menus.

Root cause these guard against: on many terminals/terminfo entries, cursor
keys are delivered to ``getch()`` as raw CSI/SS3 escape byte sequences
(``27, 91, 66`` for arrow-down) even when ``keypad(True)`` is set. The menus
used to treat the leading ``27`` as ESC/cancel, which dumped the setup wizard's
provider/model picker into its numbered "Select [1-N]" fallback the instant a
user pressed up or down.
"""
import sys
from types import SimpleNamespace

import pytest

# curses (and its _curses C extension) is Unix-only; skip the whole module on Windows.
if sys.platform == "win32":
    pytest.skip("curses is not available on Windows", allow_module_level=True)
import curses

from hermes_cli.curses_ui import (
    NAV_BACK,
    NAV_CANCEL,
    NAV_DOWN,
    NAV_INTERRUPT,
    NAV_NONE,
    NAV_SELECT,
    NAV_TOGGLE,
    NAV_UP,
    MenuNavigationStart,
    _NumberedNavigation,
    curses_radiolist,
    read_menu_key,
    reset_menu_navigation_handler,
    set_menu_navigation_handler,
)


class FakeStdscr:
    """Minimal stdscr stand-in that replays a queue of getch() byte returns.

    ``getch`` pops from ``keys``; an empty queue yields ``-1`` (matching curses
    non-blocking behavior). ``timeout`` is recorded but otherwise inert.
    """

    def __init__(self, keys):
        self.keys = list(keys)
        self.timeouts = []
        self.writes = []

    def getch(self):
        return self.keys.pop(0) if self.keys else -1

    def timeout(self, ms):
        self.timeouts.append(ms)

    def clear(self):
        pass

    def getmaxyx(self):
        return (12, 80)

    def addnstr(self, *args):
        self.writes.append(args)

    def refresh(self):
        pass


class ExhaustingStdscr(FakeStdscr):
    """Fail instead of spinning if a key sequence is not fully handled."""

    def getch(self):
        if not self.keys:
            raise AssertionError(
                "menu kept polling for input after its scripted keys were consumed"
            )
        return self.keys.pop(0)




def test_raw_ss3_arrow_keys_decode():
    # Application cursor mode: ESC O B / ESC O A
    assert read_menu_key(FakeStdscr([27, ord("O"), ord("B")])) == NAV_DOWN
    assert read_menu_key(FakeStdscr([27, ord("O"), ord("A")])) == NAV_UP


def test_left_arrow_decodes_to_back():
    assert read_menu_key(FakeStdscr([curses.KEY_LEFT])) == NAV_BACK
    assert read_menu_key(FakeStdscr([27, ord("["), ord("D")])) == NAV_BACK
    assert read_menu_key(FakeStdscr([27, ord("O"), ord("D")])) == NAV_BACK


@pytest.mark.parametrize(
    ("keys", "expected"),
    [
        ([27, ord("["), ord("1"), ord("3"), ord("u")], NAV_SELECT),
        (
            [27, ord("["), ord("1"), ord("3"), ord(";"), ord("1"), ord("u")],
            NAV_SELECT,
        ),
        ([27, ord("["), ord("2"), ord("7"), ord("u")], NAV_CANCEL),
        ([27, ord("["), ord("3"), ord("2"), ord("u")], NAV_TOGGLE),
        (
            [27, ord("["), ord("9"), ord("9"), ord(";"), ord("5"), ord("u")],
            NAV_INTERRUPT,
        ),
        ([27, ord("["), ord("1"), ord(";"), ord("1"), ord("D")], NAV_BACK),
        (
            [27, ord("["), ord("2"), ord("7"), ord(";"), ord("1"), ord(";"), ord("1"), ord("3"), ord("~")],
            NAV_SELECT,
        ),
        (
            [27, ord("["), ord("2"), ord("7"), ord(";"), ord("5"), ord(";"), ord("9"), ord("9"), ord("~")],
            NAV_INTERRUPT,
        ),
        (
            [27, ord("["), ord("1"), ord("3"), ord(";"), ord("1"), ord(":"), ord("3"), ord("u")],
            NAV_NONE,
        ),
    ],
)
def test_enhanced_keyboard_sequences_decode(keys, expected):
    assert read_menu_key(FakeStdscr(keys)) == expected


def test_raw_ctrl_c_cancels():
    assert read_menu_key(FakeStdscr([3])) == NAV_INTERRUPT


def test_enhanced_enter_selects_filtered_model_while_search_is_active(monkeypatch):
    fake = ExhaustingStdscr(
        [
            ord("/"),
            ord("g"),
            27,
            ord("["),
            ord("1"),
            ord("3"),
            ord("u"),
        ]
    )
    monkeypatch.setattr(sys.stdin, "isatty", lambda: True)
    monkeypatch.setattr(curses, "wrapper", lambda draw: draw(fake))
    monkeypatch.setattr(curses, "curs_set", lambda _value: None)
    monkeypatch.setattr(curses, "has_colors", lambda: False)
    monkeypatch.setattr("hermes_cli.curses_ui.flush_stdin", lambda: None)

    selected = curses_radiolist(
        "Pick",
        ["alpha", "gpt"],
        searchable=True,
        search_labels=["alpha", "gpt"],
    )

    assert selected == 1


@pytest.mark.parametrize(
    ("enhanced_keys", "expected_event"),
    [
        ([27, ord("["), ord("9"), ord("9"), ord(";"), ord("5"), ord("u")], "cancel"),
        ([27, ord("["), ord("1"), ord(";"), ord("1"), ord("D")], "back"),
    ],
)
def test_enhanced_control_keys_dispatch_while_search_is_active(
    monkeypatch, enhanced_keys, expected_event
):
    class NavigationDispatched(Exception):
        pass

    fake = ExhaustingStdscr([ord("/"), ord("g"), *enhanced_keys])
    events = []
    monkeypatch.setattr(sys.stdin, "isatty", lambda: True)
    monkeypatch.setattr(curses, "wrapper", lambda draw: draw(fake))
    monkeypatch.setattr(curses, "curs_set", lambda _value: None)
    monkeypatch.setattr(curses, "has_colors", lambda: False)

    def handler(event, *_args):
        events.append(event)
        if event == expected_event:
            raise NavigationDispatched()
        return MenuNavigationStart(allow_back=True) if event == "begin" else None

    token = set_menu_navigation_handler(handler)
    try:
        with pytest.raises(NavigationDispatched):
            curses_radiolist(
                "Pick",
                ["alpha", "gpt"],
                searchable=True,
                search_labels=["alpha", "gpt"],
            )
    finally:
        reset_menu_navigation_handler(token)

    assert events == ["begin", expected_event]


def test_numbered_fallback_ctrl_c_dispatches_scoped_cancel(monkeypatch):
    class Cancelled(Exception):
        pass

    fake_error = curses.error("terminal unavailable")
    monkeypatch.setattr(sys.stdin, "isatty", lambda: True)
    monkeypatch.setattr(
        curses,
        "wrapper",
        lambda _draw: (_ for _ in ()).throw(fake_error),
    )
    monkeypatch.setattr(
        "hermes_cli.curses_ui._read_numbered_input",
        lambda _prompt="": _NumberedNavigation.CANCEL,
    )

    def handler(event, *_args):
        if event == "cancel":
            raise Cancelled()
        return MenuNavigationStart() if event == "begin" else None

    token = set_menu_navigation_handler(handler)
    try:
        with pytest.raises(Cancelled):
            curses_radiolist("Pick", ["a", "b"], cancel_returns=-1)
    finally:
        reset_menu_navigation_handler(token)


def test_navigation_handler_programming_error_is_not_hidden_by_fallback(monkeypatch):
    fake = ExhaustingStdscr([13])
    monkeypatch.setattr(sys.stdin, "isatty", lambda: True)
    monkeypatch.setattr(curses, "wrapper", lambda draw: draw(fake))
    monkeypatch.setattr(curses, "curs_set", lambda _value: None)
    monkeypatch.setattr(curses, "has_colors", lambda: False)

    def handler(event, *_args):
        if event == "resolve":
            raise RuntimeError("navigation contract mismatch")
        return MenuNavigationStart() if event == "begin" else None

    token = set_menu_navigation_handler(handler)
    try:
        with pytest.raises(RuntimeError, match="navigation contract mismatch"):
            curses_radiolist("Pick", ["a", "b"])
    finally:
        reset_menu_navigation_handler(token)


def test_standalone_model_flow_renders_previous_and_reopens_provider(monkeypatch):
    from hermes_cli import main as main_mod

    provider_screen = ExhaustingStdscr([13])
    model_back_screen = ExhaustingStdscr([curses.KEY_LEFT])
    provider_reselect_screen = ExhaustingStdscr([13])
    model_select_screen = ExhaustingStdscr([13])
    screens = [
        provider_screen,
        model_back_screen,
        provider_reselect_screen,
        model_select_screen,
    ]
    resolved = []

    monkeypatch.setattr(sys.stdin, "isatty", lambda: True)
    monkeypatch.setattr(curses, "wrapper", lambda draw: draw(screens.pop(0)))
    monkeypatch.setattr(curses, "curs_set", lambda _value: None)
    monkeypatch.setattr(curses, "has_colors", lambda: False)
    monkeypatch.setattr("hermes_cli.curses_ui.flush_stdin", lambda: None)
    monkeypatch.setattr(main_mod, "_require_tty", lambda _command: None)

    def fake_select_provider_and_model(args=None):
        resolved.append(
            curses_radiolist("Select provider:", ["OpenAI"], cancel_returns=-1)
        )
        resolved.append(
            curses_radiolist(
                "Select default model:",
                ["gpt-5.6-sol"],
                cancel_returns=-1,
                searchable=True,
                search_labels=["gpt-5.6-sol"],
            )
        )

    monkeypatch.setattr(
        main_mod, "select_provider_and_model", fake_select_provider_and_model
    )

    main_mod.cmd_model(SimpleNamespace(refresh=False))

    assert resolved == [0, 0, 0]
    assert screens == []
    assert any(
        "\u2190 previous" in str(args[2]) for args in model_back_screen.writes
    )


def test_radiolist_dispatches_contextual_back_navigation(monkeypatch):
    class WentBack(BaseException):
        pass

    events = []
    fake = ExhaustingStdscr([curses.KEY_LEFT])
    monkeypatch.setattr(sys.stdin, "isatty", lambda: True)
    monkeypatch.setattr(curses, "wrapper", lambda draw: draw(fake))
    monkeypatch.setattr(curses, "curs_set", lambda _value: None)
    monkeypatch.setattr(curses, "has_colors", lambda: False)
    monkeypatch.setattr("hermes_cli.curses_ui.flush_stdin", lambda: None)

    def handler(event, *_args):
        events.append(event)
        if event == "back":
            raise WentBack()
        return MenuNavigationStart(allow_back=True) if event == "begin" else None

    token = set_menu_navigation_handler(handler)
    try:
        with pytest.raises(WentBack):
            curses_radiolist("Pick", ["a", "b"])
    finally:
        reset_menu_navigation_handler(token)

    assert events == ["begin", "back"]


def test_radiolist_dispatches_contextual_escape_cancellation(monkeypatch):
    class Cancelled(BaseException):
        pass

    # FakeStdscr (not ExhaustingStdscr): lone-ESC detection legitimately polls one
    # more key and expects the -1 "no continuation byte" answer here.
    fake = FakeStdscr([27])
    monkeypatch.setattr(sys.stdin, "isatty", lambda: True)
    monkeypatch.setattr(curses, "wrapper", lambda draw: draw(fake))
    monkeypatch.setattr(curses, "curs_set", lambda _value: None)
    monkeypatch.setattr(curses, "has_colors", lambda: False)

    def handler(event):
        if event == "cancel":
            raise Cancelled()
        return MenuNavigationStart() if event == "begin" else None

    token = set_menu_navigation_handler(handler)
    try:
        with pytest.raises(Cancelled):
            curses_radiolist("Pick", ["a", "b"])
    finally:
        reset_menu_navigation_handler(token)


def test_lone_escape_is_cancel():
    assert read_menu_key(FakeStdscr([27])) == NAV_CANCEL






def test_enter_variants_select():
    assert read_menu_key(FakeStdscr([10])) == NAV_SELECT
    assert read_menu_key(FakeStdscr([13])) == NAV_SELECT
    assert read_menu_key(FakeStdscr([curses.KEY_ENTER])) == NAV_SELECT
