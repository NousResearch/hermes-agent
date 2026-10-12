"""Keyboard-mode invariants for the existing curses session picker."""

from unittest.mock import MagicMock

import pytest

from hermes_cli.sessions_cmd_browse import _session_browse_picker


@pytest.fixture
def keyboard_picker(monkeypatch):
    import curses

    screen = MagicMock()
    screen.getmaxyx.return_value = (12, 64)
    monkeypatch.setattr(curses, "wrapper", lambda run: run(screen))
    monkeypatch.setattr(curses, "curs_set", lambda *_: None)
    monkeypatch.setattr(curses, "has_colors", lambda: False)
    sessions = [{"id": f"saved_{i}", "title": f"Project {i}", "preview": ""} for i in range(30)]
    sessions[17]["title"] = "j/k target"

    def pick(keys):
        screen.get_wch.side_effect = [chr(key) if key < 128 else key for key in keys]
        screen.getch.return_value = -1  # no continuation after a genuine Escape
        return _session_browse_picker(sessions, keyboard_navigation=True)

    return pick, sessions, screen


@pytest.mark.platforms("posix")
@pytest.mark.parametrize("keys, index", [
    ([ord("j"), ord("j"), ord("k"), 10], 1),
    ([ord("G"), ord("g"), ord("j"), 10], 1),
    ([ord("/"), ord("j"), ord("/"), ord("k"), 10], 17),
])
def test_navigation_keys_are_not_search_until_slash(keyboard_picker, keys, index):
    pick, sessions, _ = keyboard_picker
    assert pick(keys) == sessions[index]["id"]


@pytest.mark.platforms("posix")
def test_page_navigation_uses_visible_rows_and_cancel_is_nonselecting(keyboard_picker):
    import curses
    from hermes_cli.sessions_cmd_browse import _CursesBrowser

    pick, sessions, screen = keyboard_picker
    browser = _CursesBrowser(curses, sessions, None, keyboard_navigation=True)
    browser._draw(screen, *screen.getmaxyx())
    for key in (curses.KEY_NPAGE, curses.KEY_NPAGE, curses.KEY_PPAGE, 10):
        browser._handle_key(key)
    assert browser.result == sessions[browser.page_size]["id"]
    for key in (27, ord("q"), 3):
        assert pick([ord("j"), key]) is None


@pytest.mark.platforms("posix")
@pytest.mark.parametrize("width", [40, 64, 80, 121])
def test_keyboard_rows_use_terminal_cells_and_keep_selected_identity(keyboard_picker, width):
    from wcwidth import wcswidth
    from hermes_cli.sessions_cmd_browse import _CursesBrowser
    import curses

    _, sessions, screen = keyboard_picker
    sessions[0].update(title="界🙂" * 50, preview="selected preview", message_count=3)
    browser = _CursesBrowser(curses, sessions, None, keyboard_navigation=True)
    browser._draw(screen, 12, width)
    drawn = [call.args for call in screen.addstr.call_args_list]

    assert all(wcswidth(text) <= width - 1 for _, _, text, _ in drawn)
    assert any(sessions[0]["id"] in text for _, _, text, _ in drawn)
    assert any("selected preview" in text for _, _, text, _ in drawn)


@pytest.mark.platforms("posix")
@pytest.mark.parametrize("text", ["e\u0301" * 8 + " suffix", "👩\u200d💻" * 4 + " suffix"])
def test_rendering_does_not_clip_codepoints_again_after_cell_clipping(keyboard_picker, text):
    import curses
    from hermes_cli.sessions_cmd_browse import _CursesBrowser, _clip

    class Screen:
        rendered = ""

        def addnstr(self, y, x, value, n, attr):
            self.rendered = value[:n]

        def addstr(self, y, x, value, attr):
            self.rendered = value

    _, sessions, _ = keyboard_picker
    screen = Screen()
    browser = _CursesBrowser(curses, sessions, None, keyboard_navigation=True)
    browser._put(screen, 0, 0, text, 8, curses.A_NORMAL)
    assert screen.rendered == _clip(text, 8)


@pytest.mark.platforms("posix")
def test_compact_metadata_uses_dim_default_foreground_without_changing_standalone(keyboard_picker, monkeypatch):
    import curses
    from hermes_cli.sessions_cmd_browse import _CursesBrowser

    _, sessions, screen = keyboard_picker
    monkeypatch.setattr(curses, "has_colors", lambda: True)
    monkeypatch.setattr(curses, "color_pair", lambda n: n << 8)
    browser = _CursesBrowser(curses, sessions, None, keyboard_navigation=True)
    browser._draw(screen, 12, 64)
    writes = screen.addnstr.call_args_list + screen.addstr.call_args_list
    compact_metadata = [call.args[-1] for call in writes
                        if call.args[0] in (1, 9, 11)]
    assert compact_metadata == [curses.A_DIM] * 3

    screen.reset_mock()
    browser = _CursesBrowser(curses, sessions, None)
    browser._draw(screen, 12, 64)
    writes = screen.addnstr.call_args_list + screen.addstr.call_args_list
    assert all(call.args[-1] == curses.color_pair(4) for call in writes if call.args[0] in (1, 11))


@pytest.mark.platforms("posix")
@pytest.mark.parametrize("width", [40, 64])
def test_row_numbers_keep_original_indices_when_scrolling_and_searching(keyboard_picker, width):
    import curses
    from wcwidth import wcswidth
    from hermes_cli.sessions_cmd_browse import _CursesBrowser

    _, sessions, screen = keyboard_picker
    browser = _CursesBrowser(curses, sessions, None, keyboard_navigation=True)
    browser.cursor = 17
    browser._draw(screen, 12, width)
    rows = [call.args[2] for call in screen.addstr.call_args_list if 3 <= call.args[0] < 8]
    assert any(row.startswith(" → 18  ") and "j/k target" in row for row in rows)
    assert all(wcswidth(row) <= width - 1 for row in rows)

    browser.search = "j/k target"
    browser._refilter()
    screen.reset_mock()
    browser._draw(screen, 12, width)
    row = next(call.args[2] for call in screen.addstr.call_args_list if call.args[0] == 3)
    assert row.startswith(" → 18  j/k target")
