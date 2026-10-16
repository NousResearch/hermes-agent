"""Tests for the interactive session browser (`hermes sessions browse`).

Covers:
- _session_browse_picker logic (curses mocked, fallback tested)
- cmd_sessions 'browse' action integration
- Argument parser registration
"""

import argparse
import json
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest


from hermes_cli.sessions_cmd import _cmd_browse, cmd_sessions
from hermes_cli.sessions_cmd_browse import (
    _session_browse_picker,
    session_browse_to_json_record,
    session_browse_to_json_records,
)
from hermes_cli.subcommands.sessions import build_sessions_parser


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








# ─── Argument parser registration ──────────────────────────────────────────

class TestArgumentParser:
    """Verify browse subparser registers --json and --format flags."""

    @pytest.fixture
    def parser(self):
        root = argparse.ArgumentParser()
        subs = root.add_subparsers(dest="command")
        build_sessions_parser(subs, cmd_sessions=MagicMock())
        return root

    def test_browse_json_flag(self, parser):
        args = parser.parse_args(["sessions", "browse", "--json"])
        assert args.command == "sessions"
        assert args.sessions_action == "browse"
        assert args.json is True
        assert args.format is None

    def test_browse_format_json_flag(self, parser):
        args = parser.parse_args(["sessions", "browse", "--format", "json"])
        assert args.format == "json"
        assert args.json is False

    def test_browse_format_text_flag(self, parser):
        args = parser.parse_args(["sessions", "browse", "--format", "text"])
        assert args.format == "text"
        assert args.json is False

    def test_browse_default_flags(self, parser):
        args = parser.parse_args(["sessions", "browse"])
        assert args.json is False
        assert args.format is None
        assert args.limit == 500
        assert args.source is None

    def test_browse_filter_flags(self, parser):
        args = parser.parse_args(["sessions", "browse", "--source", "cli", "--limit", "50", "--json"])
        assert args.source == "cli"
        assert args.limit == 50
        assert args.json is True


# ─── JSON Projection ────────────────────────────────────────────────────────

class TestSessionBrowseJsonProjection:
    """Unit tests for session_browse_to_json_record projection."""

    def test_projection_full_fields(self):
        record = session_browse_to_json_record({
            "id": "sess_123",
            "name": "Custom Name",
            "source": "cli",
            "created_at": "2026-10-15T12:00:00Z",
            "turn_count": 42,
            "tokens": 12345,
            "path": "/custom/path/sess_123.jsonl",
        })
        assert record == {
            "id": "sess_123",
            "name": "Custom Name",
            "source": "cli",
            "created_at": "2026-10-15T12:00:00Z",
            "turn_count": 42,
            "tokens": 12345,
            "path": "/custom/path/sess_123.jsonl",
        }

    def test_name_fallbacks(self):
        assert session_browse_to_json_record({"id": "s1", "title": "My Title"})["name"] == "My Title"
        assert session_browse_to_json_record({"id": "s1", "display_name": "Disp Name"})["name"] == "Disp Name"
        assert session_browse_to_json_record({"id": "s1", "preview": "Hello world"})["name"] == "Hello world"
        assert session_browse_to_json_record({"id": "s1"})["name"] == "s1"

    def test_created_at_from_started_at_timestamp(self):
        record = session_browse_to_json_record({"id": "s1", "started_at": 1728950400.0})
        assert record["created_at"] == "2024-10-15T00:00:00Z"

    def test_turn_count_and_tokens_fallbacks(self):
        record = session_browse_to_json_record({
            "id": "s1",
            "message_count": 10,
            "input_tokens": 100,
            "output_tokens": 50,
        })
        assert record["turn_count"] == 10
        assert record["tokens"] == 150

    def test_path_derived_from_db(self, tmp_path):
        mock_db = MagicMock(db_path=tmp_path / "state.db")
        record = session_browse_to_json_record({"id": "s1"}, session_db=mock_db)
        expected = str(tmp_path / "sessions" / "s1.jsonl")
        assert record["path"] == expected


# ─── Integration: cmd_sessions browse action ────────────────────────────────

class TestCmdBrowseJsonIntegration:
    """Integration tests for _cmd_browse in JSON mode."""

    def test_browse_json_output(self, capsys):
        mock_db = MagicMock()
        mock_db.list_sessions_rich.return_value = [
            {
                "id": "sess_001",
                "title": "Alpha Session",
                "source": "cli",
                "started_at": 1728950400.0,
                "message_count": 5,
                "input_tokens": 200,
                "output_tokens": 100,
            }
        ]
        args = SimpleNamespace(json=True, format=None, limit=10, source=None)
        _cmd_browse(mock_db, args)

        mock_db.close.assert_called_once()
        out = capsys.readouterr().out
        data = json.loads(out)
        assert isinstance(data, list)
        assert len(data) == 1
        assert data[0]["id"] == "sess_001"
        assert data[0]["name"] == "Alpha Session"
        assert data[0]["source"] == "cli"
        assert data[0]["created_at"] == "2024-10-15T00:00:00Z"
        assert data[0]["turn_count"] == 5
        assert data[0]["tokens"] == 300
        assert data[0]["path"].endswith("sess_001.jsonl")

    def test_browse_format_json_output(self, capsys):
        mock_db = MagicMock()
        mock_db.list_sessions_rich.return_value = [
            {"id": "sess_002", "source": "telegram", "started_at": 1728950400.0, "message_count": 2}
        ]
        args = SimpleNamespace(json=False, format="json", limit=10, source="telegram")
        _cmd_browse(mock_db, args)

        mock_db.close.assert_called_once()
        data = json.loads(capsys.readouterr().out)
        assert len(data) == 1
        assert data[0]["id"] == "sess_002"
        assert data[0]["source"] == "telegram"

    def test_browse_json_empty_sessions_emits_empty_array(self, capsys):
        mock_db = MagicMock()
        mock_db.list_sessions_rich.return_value = []
        args = SimpleNamespace(json=True, format=None, limit=500, source=None)
        _cmd_browse(mock_db, args)

        mock_db.close.assert_called_once()
        out = capsys.readouterr().out.strip()
        assert out == "[]"
        assert json.loads(out) == []

    def test_browse_json_passes_filters(self):
        mock_db = MagicMock()
        mock_db.list_sessions_rich.return_value = []
        args = SimpleNamespace(json=True, format=None, limit=25, source="discord")
        _cmd_browse(mock_db, args)

        mock_db.list_sessions_rich.assert_called_once_with(
            source="discord", exclude_sources=None, limit=25
        )

    def test_empty_profile_browse_json(self, tmp_path, capsys):
        """When profile has no state.db and --json is requested, emit []."""
        with patch("hermes_cli.sessions_cmd.get_hermes_home", return_value=tmp_path):
            args = SimpleNamespace(sessions_action="browse", json=True, format=None, source=None, limit=100)
            result = cmd_sessions(args)
            assert result is None
            out = capsys.readouterr().out.strip()
            assert out == "[]"



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
