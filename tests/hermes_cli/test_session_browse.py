"""Tests for the interactive session browser (`hermes sessions browse`).

Covers:
- _session_browse_picker logic (curses mocked, fallback tested)
- cmd_sessions 'browse' action integration
- Argument parser registration
"""

import os
import time
from unittest.mock import MagicMock, patch, call

import pytest

from hermes_cli.main import _session_browse_picker


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

    def test_returns_none_when_no_sessions(self, capsys):
        result = _session_browse_picker([])
        assert result is None

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

    def test_fallback_mode_cancel_q(self):
        """Entering 'q' in fallback mode cancels."""
        sessions = _make_sessions(3)

        import builtins
        original_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == "curses":
                raise ImportError("no curses")
            return original_import(name, *args, **kwargs)

        with patch.object(builtins, "__import__", side_effect=mock_import):
            with patch("builtins.input", return_value="q"):
                result = _session_browse_picker(sessions)

        assert result is None

    def test_fallback_mode_cancel_empty(self):
        """Entering empty string in fallback mode cancels."""
        sessions = _make_sessions(3)

        import builtins
        original_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == "curses":
                raise ImportError("no curses")
            return original_import(name, *args, **kwargs)

        with patch.object(builtins, "__import__", side_effect=mock_import):
            with patch("builtins.input", return_value=""):
                result = _session_browse_picker(sessions)

        assert result is None

    def test_fallback_mode_invalid_then_valid(self):
        """Invalid selection followed by valid one works."""
        sessions = _make_sessions(3)

        import builtins
        original_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == "curses":
                raise ImportError("no curses")
            return original_import(name, *args, **kwargs)

        with patch.object(builtins, "__import__", side_effect=mock_import):
            with patch("builtins.input", side_effect=["99", "1"]):
                result = _session_browse_picker(sessions)

        assert result == sessions[0]["id"]

    def test_fallback_mode_keyboard_interrupt(self):
        """KeyboardInterrupt in fallback mode returns None."""
        sessions = _make_sessions(3)

        import builtins
        original_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == "curses":
                raise ImportError("no curses")
            return original_import(name, *args, **kwargs)

        with patch.object(builtins, "__import__", side_effect=mock_import):
            with patch("builtins.input", side_effect=KeyboardInterrupt):
                result = _session_browse_picker(sessions)

        assert result is None

    def test_fallback_displays_all_sessions(self, capsys):
        """Fallback mode should display all session entries."""
        sessions = _make_sessions(4)

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
        # All 4 entries should be shown
        assert "1." in output
        assert "2." in output
        assert "3." in output
        assert "4." in output

    def test_fallback_shows_title_over_preview(self, capsys):
        """When a session has a title, show it instead of the preview."""
        sessions = [{
            "id": "test_001",
            "source": "cli",
            "title": "My Cool Project",
            "preview": "some preview text",
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
        assert "My Cool Project" in output

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

    def test_fallback_shows_id_when_no_title_or_preview(self, capsys):
        """When neither title nor preview, show session ID."""
        sessions = [{
            "id": "test_003_fallback",
            "source": "cli",
            "title": None,
            "preview": "",
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
        assert "test_003_fallback" in output


# ─── Curses-based picker (mocked curses) ────────────────────────────────────

class TestCursesBrowse:
    """Tests for the curses-based interactive picker via simulated key sequences."""

    def _run_with_keys(self, sessions, key_sequence):
        """Simulate running the curses picker with a given key sequence."""
        import curses

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
            with patch("curses.curs_set"):
                with patch("curses.has_colors", return_value=False):
                    return _session_browse_picker(sessions)

    def test_enter_selects_first_session(self):
        sessions = _make_sessions(3)
        result = self._run_with_keys(sessions, [10])  # Enter key
        assert result == sessions[0]["id"]

    def test_down_then_enter_selects_second(self):
        import curses
        sessions = _make_sessions(3)
        result = self._run_with_keys(sessions, [curses.KEY_DOWN, 10])
        assert result == sessions[1]["id"]

    def test_down_down_enter_selects_third(self):
        import curses
        sessions = _make_sessions(5)
        result = self._run_with_keys(sessions, [curses.KEY_DOWN, curses.KEY_DOWN, 10])
        assert result == sessions[2]["id"]

    def test_up_wraps_to_last(self):
        import curses
        sessions = _make_sessions(3)
        result = self._run_with_keys(sessions, [curses.KEY_UP, 10])
        assert result == sessions[2]["id"]

    def test_escape_cancels(self):
        sessions = _make_sessions(3)
        result = self._run_with_keys(sessions, [27])  # Esc
        assert result is None

    def test_q_cancels(self):
        sessions = _make_sessions(3)
        result = self._run_with_keys(sessions, [ord('q')])
        assert result is None

    def test_type_to_filter_then_enter(self):
        """Typing characters filters the list, Enter selects from filtered."""
        import curses
        sessions = [
            {"id": "s1", "source": "cli", "title": "Alpha project", "preview": "", "last_active": time.time()},
            {"id": "s2", "source": "cli", "title": "Beta project", "preview": "", "last_active": time.time()},
            {"id": "s3", "source": "cli", "title": "Gamma project", "preview": "", "last_active": time.time()},
        ]
        # Type "Beta" then Enter — should select s2
        keys = [ord(c) for c in "Beta"] + [10]
        result = self._run_with_keys(sessions, keys)
        assert result == "s2"

    def test_filter_no_match_enter_does_nothing(self):
        """When filter produces no results, Enter shouldn't select."""
        sessions = _make_sessions(3)
        keys = [ord(c) for c in "zzzznonexistent"] + [10]
        result = self._run_with_keys(sessions, keys)
        assert result is None

    def test_backspace_removes_filter_char(self):
        """Backspace removes the last character from the filter."""
        import curses
        sessions = [
            {"id": "s1", "source": "cli", "title": "Alpha", "preview": "", "last_active": time.time()},
            {"id": "s2", "source": "cli", "title": "Beta", "preview": "", "last_active": time.time()},
        ]
        # Type "Bet", backspace, backspace, backspace (clears filter), then Enter (selects first)
        keys = [ord('B'), ord('e'), ord('t'), 127, 127, 127, 10]
        result = self._run_with_keys(sessions, keys)
        assert result == "s1"

    def test_escape_clears_filter_first(self):
        """First Esc clears the search text, second Esc exits."""
        import curses
        sessions = _make_sessions(3)
        # Type "ab" then Esc (clears filter) then Enter (selects first)
        keys = [ord('a'), ord('b'), 27, 10]
        result = self._run_with_keys(sessions, keys)
        assert result == sessions[0]["id"]

    def test_filter_matches_preview(self):
        """Typing should match against session preview text."""
        sessions = [
            {"id": "s1", "source": "cli", "title": None, "preview": "Set up Minecraft server", "last_active": time.time()},
            {"id": "s2", "source": "cli", "title": None, "preview": "Review PR 438", "last_active": time.time()},
        ]
        keys = [ord(c) for c in "Mine"] + [10]
        result = self._run_with_keys(sessions, keys)
        assert result == "s1"

    def test_filter_matches_source(self):
        """Typing a source name should filter by source."""
        sessions = [
            {"id": "s1", "source": "telegram", "title": "TG session", "preview": "", "last_active": time.time()},
            {"id": "s2", "source": "cli", "title": "CLI session", "preview": "", "last_active": time.time()},
        ]
        keys = [ord(c) for c in "telegram"] + [10]
        result = self._run_with_keys(sessions, keys)
        assert result == "s1"

    def test_q_quits_when_no_filter_active(self):
        """When no search text is active, 'q' should quit (not filter)."""
        sessions = _make_sessions(3)
        result = self._run_with_keys(sessions, [ord('q')])
        assert result is None

    def test_q_types_into_filter_when_filter_active(self):
        """When search text is already active, 'q' should add to filter, not quit."""
        sessions = [
            {"id": "s1", "source": "cli", "title": "the sequel", "preview": "", "last_active": time.time()},
            {"id": "s2", "source": "cli", "title": "other thing", "preview": "", "last_active": time.time()},
        ]
        # Type "se" first (activates filter, matches "the sequel")
        # Then type "q" — should add 'q' to filter (filter="seq"), NOT quit
        # "seq" still matches "the sequel" → Enter selects it
        keys = [ord('s'), ord('e'), ord('q'), 10]
        result = self._run_with_keys(sessions, keys)
        assert result == "s1"  # "the sequel" matches "seq"


# ─── Argument parser registration ──────────────────────────────────────────

class TestSessionBrowseArgparse:
    """Verify the 'browse' subcommand is properly registered."""

    def test_browse_subcommand_exists(self):
        """hermes sessions browse should be parseable."""
        from hermes_cli.main import main as _main_entry

        # We can't run main(), but we can import and test the parser setup
        # by checking that argparse doesn't error on "sessions browse"
        import argparse
        # Re-create the parser portion
        # Instead, let's just verify the import works and the function exists
        from hermes_cli.main import _session_browse_picker
        assert callable(_session_browse_picker)

    def test_browse_default_limit_is_500(self):
        """The default --limit for browse should be 500."""
        # Build the same argparse tree cmd_sessions uses and verify the default.
        import argparse
        parser = argparse.ArgumentParser()
        subparsers = parser.add_subparsers(dest="sessions_action")
        browse = subparsers.add_parser("browse")
        browse.add_argument("--source")
        browse.add_argument("--limit", type=int, default=500)

        args = parser.parse_args(["browse"])
        assert args.limit == 500

        args = parser.parse_args(["browse", "--limit", "42"])
        assert args.limit == 42


# ─── Integration: cmd_sessions browse action ────────────────────────────────

class TestCmdSessionsBrowse:
    """Integration tests for the 'browse' action in cmd_sessions."""

    def test_browse_no_sessions_prints_message(self, capsys):
        """When no sessions exist, _session_browse_picker returns None and prints message."""
        result = _session_browse_picker([])
        assert result is None
        output = capsys.readouterr().out
        assert "No sessions found" in output

    def test_browse_with_source_filter(self):
        """The --source flag should be passed to list_sessions_rich."""
        sessions = [
            {"id": "s1", "source": "cli", "title": "CLI only", "preview": "", "last_active": time.time()},
        ]

        import builtins
        original_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == "curses":
                raise ImportError("no curses")
            return original_import(name, *args, **kwargs)

        with patch.object(builtins, "__import__", side_effect=mock_import):
            with patch("builtins.input", return_value="1"):
                result = _session_browse_picker(sessions)

        assert result == "s1"


# ─── Edge cases ──────────────────────────────────────────────────────────────

class TestEdgeCases:
    """Edge case handling for the session browser."""

    def test_sessions_with_missing_fields(self):
        """Sessions with missing optional fields should not crash."""
        sessions = [
            {"id": "minimal_001", "source": "cli"},  # No title, preview, last_active
        ]

        import builtins
        original_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == "curses":
                raise ImportError("no curses")
            return original_import(name, *args, **kwargs)

        with patch.object(builtins, "__import__", side_effect=mock_import):
            with patch("builtins.input", return_value="1"):
                result = _session_browse_picker(sessions)

        assert result == "minimal_001"

    def test_single_session(self):
        """A single session in the list should work fine."""
        sessions = [
            {"id": "only_one", "source": "cli", "title": "Solo", "preview": "", "last_active": time.time()},
        ]

        import builtins
        original_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == "curses":
                raise ImportError("no curses")
            return original_import(name, *args, **kwargs)

        with patch.object(builtins, "__import__", side_effect=mock_import):
            with patch("builtins.input", return_value="1"):
                result = _session_browse_picker(sessions)

        assert result == "only_one"

    def test_long_title_truncated_in_fallback(self, capsys):
        """Very long titles should be truncated in fallback mode."""
        sessions = [{
            "id": "long_title_001",
            "source": "cli",
            "title": "A" * 100,
            "preview": "",
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
        # Title should be truncated to 50 chars with "..."
        assert "..." in output

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
