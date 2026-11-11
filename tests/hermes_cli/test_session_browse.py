"""Tests for the interactive session browser (`hermes sessions browse`).

Covers:
- _session_browse_picker logic (curses mocked, fallback tested)
- cmd_sessions 'browse' action integration
- Argument parser registration
"""

import time
from unittest.mock import MagicMock, patch

import pytest


from hermes_cli.sessions_cmd_browse import _session_browse_picker


WHEEL_DOWN = 0x200000  # curses.BUTTON5_PRESSED, absent from curses built against older ncurses


# ─── Sample session data ──────────────────────────────────────────────────────

def _make_sessions(n=5):
    """Generate a list of fake rich-session dicts."""
    now = time.time()
    sessions = []
    for i in range(n):
        sessions.append({
            "id": f"20260308_{i:06d}_abcdef",
            "source": "cli" if i % 2 == 0 else "telegram",
            "model": "test/model",
            "title": f"Session {i}" if i % 3 != 0 else None,
            "preview": f"Hello from session {i}",
            "last_active": now - i * 3600,
            "started_at": now - i * 3600 - 60,
            "message_count": (i + 1) * 5,
        })
    return sessions


SAMPLE_SESSIONS = _make_sessions(5)


# ─── _session_browse_picker ──────────────────────────────────────────────────

class TestSessionBrowsePicker:
    """Tests for the _session_browse_picker function."""

    def test_empty_sessions_returns_none(self, capsys):
        result = _session_browse_picker([])
        assert result is None
        assert "No sessions found" in capsys.readouterr().out


    def test_fallback_mode_valid_selection(self):
        """When curses is unavailable, fallback numbered list should work."""
        sessions = _make_sessions(3)

        # Mock curses import to fail, forcing fallback
        import builtins
        original_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == "curses":
                raise ImportError("no curses")
            return original_import(name, *args, **kwargs)

        with patch.object(builtins, "__import__", side_effect=mock_import):
            with patch("builtins.input", return_value="2"):
                result = _session_browse_picker(sessions)

        assert result == sessions[1]["id"]


    def test_fallback_shows_preview_when_no_title(self, capsys):
        """When no title, show preview."""
        sessions = [{
            "id": "test_002",
            "source": "cli",
            "title": None,
            "preview": "Hello world test message",
            "last_active": time.time(),
        }]

        import builtins
        original_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == "curses":
                raise ImportError("no curses")
            return original_import(name, *args, **kwargs)

        with patch.object(builtins, "__import__", side_effect=mock_import):
            with patch("builtins.input", return_value="q"):
                _session_browse_picker(sessions)

        output = capsys.readouterr().out
        assert "Hello world test message" in output


# ─── Curses-based picker (mocked curses) ────────────────────────────────────

@pytest.mark.platforms("linux")
class TestCursesBrowse:
    """Tests for the curses-based interactive picker via simulated key sequences."""

    def _run_with_keys(self, sessions, key_sequence, mouse_events=()):
        """Simulate running the curses picker with a given key sequence; each ``curses.KEY_MOUSE`` key
        consumes the next ``(y, button_state)`` from *mouse_events*."""

        # Build a mock stdscr that returns keys from the sequence
        mock_stdscr = MagicMock()
        mock_stdscr.getmaxyx.return_value = (30, 120)
        mock_stdscr.getch.side_effect = key_sequence

        # Capture what curses.wrapper receives and call it with our mock
        with patch("curses.wrapper") as mock_wrapper:
            # When wrapper is called, invoke the function with our mock stdscr
            def run_inner(func):
                try:
                    func(mock_stdscr)
                except StopIteration:
                    pass  # key sequence exhausted

            mock_wrapper.side_effect = run_inner
            events = [(0, 0, y, 0, state) for y, state in mouse_events]
            with patch("curses.curs_set"), patch("curses.mousemask") as mousemask, \
                    patch("curses.BUTTON5_PRESSED", WHEEL_DOWN, create=True):
                with patch("curses.has_colors", return_value=False), patch("curses.getmouse", side_effect=events):
                    self.mousemask = mousemask
                    return _session_browse_picker(sessions)



    def test_escape_cancels(self):
        sessions = _make_sessions(3)
        result = self._run_with_keys(sessions, [27])  # Esc
        assert result is None


    def test_type_to_filter_then_enter(self):
        """Typing characters filters the list, Enter selects from filtered."""
        sessions = [
            {"id": "s1", "source": "cli", "title": "Alpha project", "preview": "", "last_active": time.time()},
            {"id": "s2", "source": "cli", "title": "Beta project", "preview": "", "last_active": time.time()},
            {"id": "s3", "source": "cli", "title": "Gamma project", "preview": "", "last_active": time.time()},
        ]
        # Type "Beta" then Enter — should select s2
        keys = [ord(c) for c in "Beta"] + [10]
        result = self._run_with_keys(sessions, keys)
        assert result == "s2"




    def test_mouse_events_are_requested(self):
        import curses

        self._run_with_keys(_make_sessions(3), [27])
        self.mousemask.assert_called_once_with(
            curses.BUTTON1_CLICKED | curses.BUTTON4_PRESSED | WHEEL_DOWN
        )

    def test_click_selects_the_row_under_the_pointer(self):
        import curses

        sessions = _make_sessions(5)
        # Rows start at screen line 3.
        result = self._run_with_keys(sessions, [curses.KEY_MOUSE], [(5, curses.BUTTON1_CLICKED)])
        assert result == sessions[2]["id"]

    @pytest.mark.parametrize("y", [0, 2, 8])
    def test_click_outside_the_rows_selects_nothing(self, y):
        import curses

        result = self._run_with_keys(_make_sessions(3), [curses.KEY_MOUSE, 27], [(y, curses.BUTTON1_CLICKED)])
        assert result is None

    def test_wheel_down_moves_the_cursor_then_enter_selects(self):
        import curses

        sessions = _make_sessions(5)
        result = self._run_with_keys(sessions, [curses.KEY_MOUSE, 10], [(3, WHEEL_DOWN)])
        assert result == sessions[3]["id"]

    def test_wheel_up_stops_at_the_first_row(self):
        import curses

        sessions = _make_sessions(5)
        result = self._run_with_keys(sessions, [curses.KEY_MOUSE, 10], [(3, curses.BUTTON4_PRESSED)])
        assert result == sessions[0]["id"]

    def test_wheel_down_stops_at_the_last_row(self):
        import curses

        sessions = _make_sessions(2)
        result = self._run_with_keys(sessions, [curses.KEY_MOUSE, 10], [(3, WHEEL_DOWN)])
        assert result == sessions[1]["id"]


# ─── Argument parser registration ──────────────────────────────────────────



# ─── Integration: cmd_sessions browse action ────────────────────────────────



# ─── Edge cases ──────────────────────────────────────────────────────────────

class TestEdgeCases:
    """Edge case handling for the session browser."""


    def test_relative_time_formatting(self, capsys):
        """Verify various time deltas format correctly."""
        now = time.time()
        sessions = [
            {"id": "recent", "source": "cli", "title": None, "preview": "just now test", "last_active": now},
            {"id": "hour_ago", "source": "cli", "title": None, "preview": "hour ago test", "last_active": now - 7200},
            {"id": "days_ago", "source": "cli", "title": None, "preview": "days ago test", "last_active": now - 259200},
        ]

        import builtins
        original_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == "curses":
                raise ImportError("no curses")
            return original_import(name, *args, **kwargs)

        with patch.object(builtins, "__import__", side_effect=mock_import):
            with patch("builtins.input", return_value="q"):
                _session_browse_picker(sessions)

        output = capsys.readouterr().out
        assert "just now" in output
        assert "2h ago" in output
        assert "3d ago" in output
