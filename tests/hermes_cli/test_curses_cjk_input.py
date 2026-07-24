"""Tests for CJK/Unicode input support in curses session browser and menu.

The session-browser integration test uses a key-generator that never
exhausts, and patches curses internals so the rendering loop is a no-op.
"""
import curses
from unittest.mock import MagicMock, patch
import pytest


# ---------------------------------------------------------------------------
# _handle_active_search_key: Unicode support (unit tests — no curses needed)
# ---------------------------------------------------------------------------

class TestHandleActiveSearchKeyCJK:
    """Verify _handle_active_search_key accepts string keys from get_wch()."""

    def _make_search(self, query="", active=True):
        from hermes_cli.curses_ui import _SearchState
        s = _SearchState()
        s.query = query
        s.active = active
        return s

    def test_korean_char_appended_to_query(self):
        from hermes_cli.curses_ui import _handle_active_search_key
        search = self._make_search(query="")
        handled, confirm, changed = _handle_active_search_key(curses, "한", search)
        assert handled is True
        assert confirm is False
        assert changed is True
        assert search.query == "한"

    def test_chinese_char_appended_to_query(self):
        from hermes_cli.curses_ui import _handle_active_search_key
        search = self._make_search(query="test")
        handled, confirm, changed = _handle_active_search_key(curses, "中", search)
        assert handled is True
        assert search.query == "test中"

    def test_japanese_char_appended_to_query(self):
        from hermes_cli.curses_ui import _handle_active_search_key
        search = self._make_search(query="")
        handled, confirm, changed = _handle_active_search_key(curses, "あ", search)
        assert handled is True
        assert search.query == "あ"

    def test_emoji_appended_to_query(self):
        from hermes_cli.curses_ui import _handle_active_search_key
        search = self._make_search(query="")
        handled, confirm, changed = _handle_active_search_key(curses, "🎉", search)
        assert handled is True
        assert search.query == "🎉"

    def test_ascii_int_still_works(self):
        """getch()-style integer keys should still work as fallback."""
        from hermes_cli.curses_ui import _handle_active_search_key
        search = self._make_search(query="")
        handled, confirm, changed = _handle_active_search_key(curses, ord("a"), search)
        assert handled is True
        assert search.query == "a"

    def test_non_printable_string_ignored(self):
        """Non-printable strings should be ignored."""
        from hermes_cli.curses_ui import _handle_active_search_key
        search = self._make_search(query="")
        handled, confirm, changed = _handle_active_search_key(curses, "\x00", search)
        assert handled is False
        assert search.query == ""

    def test_escape_key_clears_search(self):
        """Esc (int 27) should still clear the search."""
        from hermes_cli.curses_ui import _handle_active_search_key
        search = self._make_search(query="한글", active=True)
        handled, confirm, changed = _handle_active_search_key(curses, 27, search)
        assert handled is True
        assert search.query == ""
        assert search.active is False

    def test_special_int_key_not_treated_as_char(self):
        """Integer special keys (e.g. KEY_UP=259) should not be treated as text."""
        from hermes_cli.curses_ui import _handle_active_search_key
        search = self._make_search(query="")
        handled, confirm, changed = _handle_active_search_key(curses, 259, search)
        assert handled is False
        assert search.query == ""


# ---------------------------------------------------------------------------
# _session_browse_picker: key-handling integration via mock wrapper
# ---------------------------------------------------------------------------

class TestSessionBrowsePickerCJK:
    """Verify _session_browse_picker accepts CJK input via get_wch()."""

    SAMPLE_SESSIONS = [
        {"id": "s1", "title": "한글 테스트 세션", "preview": "Korean session", "source": "cli"},
        {"id": "s2", "title": "English Session", "preview": "Normal preview", "source": "cli"},
        {"id": "s3", "title": "中文会话测试", "preview": "Chinese session", "source": "gateway"},
        {"id": "s4", "title": "日本語セッション", "preview": "Japanese session", "source": "cli"},
    ]

    def _run_picker(self, sessions, key_sequence):
        """Run _session_browse_picker feeding *key_sequence* via get_wch().

        Uses a non-exhausting key generator: after the sequence is consumed,
        it returns 'q' (quit) to terminate the loop cleanly.
        """
        from hermes_cli.main import _session_browse_picker

        quit_sentinel = object()
        seq = list(key_sequence)

        def key_gen():
            for k in seq:
                yield k
            while True:
                yield "q"  # quit sentinel after sequence

        gen = key_gen()

        mock_win = MagicMock()
        mock_win.getmaxyx.return_value = (40, 120)

        mock_stdscr = MagicMock()
        mock_stdscr.getmaxyx.return_value = (40, 120)
        mock_stdscr.get_wch = MagicMock(side_effect=lambda: next(gen))
        mock_stdscr.getch = MagicMock(side_effect=lambda: next(gen))
        mock_stdscr.derwin = MagicMock(return_value=mock_win)

        def fake_wrapper(fn):
            return fn(mock_stdscr)

        with patch.object(curses, 'wrapper', fake_wrapper), \
             patch.object(curses, 'curs_set'), \
             patch.object(curses, 'has_colors', return_value=False), \
             patch.object(curses, 'start_color'), \
             patch.object(curses, 'use_default_colors'), \
             patch.object(curses, 'init_pair'):
            return _session_browse_picker(sessions)

    def test_korean_input_filters_and_selects(self):
        """Typing Korean '한' then Enter should select the Korean session."""
        result = self._run_picker(self.SAMPLE_SESSIONS, ["한", "\n"])
        assert result == "s1"

    def test_chinese_input_filters_and_selects(self):
        """Typing Chinese '中' then Enter should select the Chinese session."""
        result = self._run_picker(self.SAMPLE_SESSIONS, ["中", "\n"])
        assert result == "s3"

    def test_japanese_input_filters_and_selects(self):
        """Typing Japanese '日' then Enter should select the Japanese session."""
        result = self._run_picker(self.SAMPLE_SESSIONS, ["日", "\n"])
        assert result == "s4"

    def test_emoji_input_filters_and_selects(self):
        """Typing emoji '🎉' then Enter should match and select."""
        sessions = [
            {"id": "e1", "title": "🎉 Party Session", "preview": "fun", "source": "cli"},
            {"id": "e2", "title": "Normal", "preview": "boring", "source": "cli"},
        ]
        result = self._run_picker(sessions, ["🎉", "\n"])
        assert result == "e1"

    def test_multi_char_cjk_search(self):
        """Typing multiple CJK chars narrows the filter."""
        # Type '한글' (2 chars) then Enter
        result = self._run_picker(self.SAMPLE_SESSIONS, ["한", "글", "\n"])
        assert result == "s1"

    def test_getch_fallback_when_get_wch_unavailable(self):
        """When get_wch raises AttributeError, getch() ASCII should still work."""
        from hermes_cli.main import _session_browse_picker

        seq = [ord("E"), ord("n"), 10]  # 'E', 'n', Enter
        quit_seq = list(seq)
        def key_gen():
            for k in quit_seq:
                yield k
            while True:
                yield ord("q")

        gen = key_gen()

        mock_win = MagicMock()
        mock_win.getmaxyx.return_value = (40, 120)

        mock_stdscr = MagicMock()
        mock_stdscr.getmaxyx.return_value = (40, 120)
        mock_stdscr.get_wch = MagicMock(side_effect=AttributeError)
        mock_stdscr.getch = MagicMock(side_effect=lambda: next(gen))
        mock_stdscr.derwin = MagicMock(return_value=mock_win)

        def fake_wrapper(fn):
            return fn(mock_stdscr)

        with patch.object(curses, 'wrapper', fake_wrapper), \
             patch.object(curses, 'curs_set'), \
             patch.object(curses, 'has_colors', return_value=False), \
             patch.object(curses, 'start_color'), \
             patch.object(curses, 'use_default_colors'), \
             patch.object(curses, 'init_pair'):
            result = _session_browse_picker(self.SAMPLE_SESSIONS)

        # "En" matches "English Session"
        assert result == "s2"
