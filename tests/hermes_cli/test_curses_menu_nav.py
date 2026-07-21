"""Regression tests for read_menu_key_ex — paging, Home/End, and the
type-to-filter (letters_are_nav=False) mode used by the session browser.

These extend test_curses_arrow_keys.py (which covers the base arrow decode)
and guard the second round of Ghostty/raw-escape fixes: the `hermes plugins`
group menu and the `hermes browse` session picker.
"""
import curses

from hermes_cli.curses_ui import (
    NAV_BACKSPACE,
    NAV_CANCEL,
    NAV_DOWN,
    NAV_END,
    NAV_HOME,
    NAV_NONE,
    NAV_PAGE_DOWN,
    NAV_PAGE_UP,
    NAV_SELECT,
    NAV_TOGGLE,
    NAV_UP,
    read_menu_key_ex,
)


class FakeStdscr:
    def __init__(self, keys):
        self.keys = list(keys)
        self.timeouts = []

    def getch(self):
        return self.keys.pop(0) if self.keys else -1

    def timeout(self, ms):
        self.timeouts.append(ms)


# ── tuple contract ──────────────────────────────────────────────────────
def test_ex_returns_action_and_raw_key():
    action, raw = read_menu_key_ex(FakeStdscr([ord("x")]))
    assert action == NAV_NONE
    assert raw == ord("x")


# ── paging via translated keys ──────────────────────────────────────────
def test_translated_paging_and_homeend():
    assert read_menu_key_ex(FakeStdscr([curses.KEY_NPAGE]))[0] == NAV_PAGE_DOWN
    assert read_menu_key_ex(FakeStdscr([curses.KEY_PPAGE]))[0] == NAV_PAGE_UP
    assert read_menu_key_ex(FakeStdscr([curses.KEY_HOME]))[0] == NAV_HOME
    assert read_menu_key_ex(FakeStdscr([curses.KEY_END]))[0] == NAV_END


# ── paging / home / end via raw escape sequences ────────────────────────
def test_raw_csi_pageup_pagedown():
    # ESC [ 5 ~ = PgUp, ESC [ 6 ~ = PgDn
    assert read_menu_key_ex(FakeStdscr([27, ord("["), ord("5"), ord("~")]))[0] == NAV_PAGE_UP
    assert read_menu_key_ex(FakeStdscr([27, ord("["), ord("6"), ord("~")]))[0] == NAV_PAGE_DOWN


def test_raw_csi_home_end_letter_form():
    # ESC [ H = Home, ESC [ F = End
    assert read_menu_key_ex(FakeStdscr([27, ord("["), ord("H")]))[0] == NAV_HOME
    assert read_menu_key_ex(FakeStdscr([27, ord("["), ord("F")]))[0] == NAV_END


def test_raw_csi_home_end_numeric_form():
    # ESC [ 1 ~ / 7 ~ = Home, ESC [ 4 ~ / 8 ~ = End
    assert read_menu_key_ex(FakeStdscr([27, ord("["), ord("1"), ord("~")]))[0] == NAV_HOME
    assert read_menu_key_ex(FakeStdscr([27, ord("["), ord("7"), ord("~")]))[0] == NAV_HOME
    assert read_menu_key_ex(FakeStdscr([27, ord("["), ord("4"), ord("~")]))[0] == NAV_END
    assert read_menu_key_ex(FakeStdscr([27, ord("["), ord("8"), ord("~")]))[0] == NAV_END


def test_delete_csi_3_tilde_ignored():
    # ESC [ 3 ~ (Delete) -> NAV_NONE, fully consumed.
    fake = FakeStdscr([27, ord("["), ord("3"), ord("~"), ord("Z")])
    assert read_menu_key_ex(fake)[0] == NAV_NONE
    assert fake.keys == [ord("Z")]  # trailing real key untouched


def test_modified_arrow_with_params_still_navigates():
    # ESC [ 1 ; 2 B (Shift+Down on some terminals) -> still DOWN.
    seq = [27, ord("["), ord("1"), ord(";"), ord("2"), ord("B")]
    assert read_menu_key_ex(FakeStdscr(seq))[0] == NAV_DOWN


# ── backspace ───────────────────────────────────────────────────────────
def test_backspace_variants():
    for k in (curses.KEY_BACKSPACE, 127, 8):
        assert read_menu_key_ex(FakeStdscr([k]))[0] == NAV_BACKSPACE


# ── letters_are_nav default (True): vim keys + q + space are nav ─────────
def test_letters_are_nav_default_true():
    assert read_menu_key_ex(FakeStdscr([ord("j")]))[0] == NAV_DOWN
    assert read_menu_key_ex(FakeStdscr([ord("k")]))[0] == NAV_UP
    assert read_menu_key_ex(FakeStdscr([ord(" ")]))[0] == NAV_TOGGLE
    assert read_menu_key_ex(FakeStdscr([ord("q")]))[0] == NAV_CANCEL


# ── letters_are_nav=False (session-browser filter mode) ─────────────────
def test_letters_are_nav_false_treats_letters_as_text():
    # j/k/q/space become typeable filter characters (NAV_NONE + raw byte).
    for ch in ("j", "k", "q", " "):
        action, raw = read_menu_key_ex(FakeStdscr([ord(ch)]), letters_are_nav=False)
        assert action == NAV_NONE, ch
        assert raw == ord(ch), ch


def test_letters_are_nav_false_arrows_still_navigate():
    # Real arrow keys (translated + raw escape) MUST still navigate even in
    # filter mode — otherwise the picker is unusable on Ghostty.
    assert read_menu_key_ex(FakeStdscr([curses.KEY_DOWN]), letters_are_nav=False)[0] == NAV_DOWN
    assert read_menu_key_ex(FakeStdscr([27, ord("["), ord("B")]), letters_are_nav=False)[0] == NAV_DOWN
    assert read_menu_key_ex(FakeStdscr([27, ord("O"), ord("A")]), letters_are_nav=False)[0] == NAV_UP


def test_letters_are_nav_false_enter_and_backspace_still_work():
    assert read_menu_key_ex(FakeStdscr([10]), letters_are_nav=False)[0] == NAV_SELECT
    assert read_menu_key_ex(FakeStdscr([127]), letters_are_nav=False)[0] == NAV_BACKSPACE


def test_letters_are_nav_false_escape_seq_never_leaks_as_text():
    # The whole point: an arrow's escape bytes must NOT surface as printable
    # filter characters. raw_key is 27 (the ESC), never '[' or 'B'.
    action, raw = read_menu_key_ex(FakeStdscr([27, ord("["), ord("B")]), letters_are_nav=False)
    assert action == NAV_DOWN
    # And an unhandled sequence returns raw_key == 27 so the 32..126 filter
    # guard rejects it.
    action2, raw2 = read_menu_key_ex(FakeStdscr([27, ord("["), ord("3"), ord("~")]), letters_are_nav=False)
    assert action2 == NAV_NONE
    assert raw2 == 27
    assert not (32 <= raw2 <= 126)


def test_esc_immediately_followed_by_other_key_is_lone_esc():
    # ESC followed immediately by a non-introducer byte (Alt-combo / fast
    # typing / paste): the ESC must register as a cancel and the trailing byte
    # must be pushed back via curses.ungetch, never silently swallowed.
    import curses as _curses

    pushed = []
    orig = _curses.ungetch
    _curses.ungetch = lambda ch: pushed.append(ch)
    try:
        action, raw = read_menu_key_ex(FakeStdscr([27, ord("x")]))
    finally:
        _curses.ungetch = orig
    assert action == NAV_CANCEL
    assert raw == 27
    assert pushed == [ord("x")]  # the 'x' was requeued, not lost

