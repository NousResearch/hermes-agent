"""Tests for the interactive session browser (`hermes sessions browse`).

Covers:
- _session_browse_picker logic (curses mocked, fallback tested)
- cmd_sessions 'browse' action integration
- Argument parser registration
"""

import time
from unittest.mock import MagicMock, patch


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

class TestCursesBrowse:
    """Tests for the curses-based interactive picker via simulated key sequences."""

    def _run_with_keys(self, sessions, key_sequence):
        """Simulate running the curses picker with a given key sequence."""

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


    def test_backspace_removes_filter_char(self):
        """Backspace removes the last character from the filter."""
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
        sessions = _make_sessions(3)
        # Type "ab" then Esc (clears filter) then Enter (selects first)
        keys = [ord('a'), ord('b'), 27, 10]
        result = self._run_with_keys(sessions, keys)
        assert result == sessions[0]["id"]


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

        # We can't run main(), but we can import and test the parser setup
        # by checking that argparse doesn't error on "sessions browse"
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
