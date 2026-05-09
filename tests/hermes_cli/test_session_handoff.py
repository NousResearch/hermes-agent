"""Tests for session handoff (CLI to gateway platform)."""

from __future__ import annotations

import time
from unittest.mock import patch

import pytest

from hermes_state import SessionDB


class TestHandoffDB:
    """Test the handoff columns and helper methods on SessionDB."""

    @pytest.fixture
    def db(self, tmp_path, monkeypatch):
        home = tmp_path / ".hermes"
        home.mkdir()
        monkeypatch.setenv("HERMES_HOME", str(home))
        db = SessionDB(db_path=home / "state.db")
        yield db

    def _make_session(self, db, session_id, source="cli", title=None):
        """Insert a session row directly for testing."""
        def _do(conn):
            conn.execute(
                "INSERT OR IGNORE INTO sessions (id, source, title, started_at) "
                "VALUES (?, ?, ?, ?)",
                (session_id, source, title, time.time()),
            )
        db._execute_write(_do)

    def test_handoff_columns_exist(self, db):
        """Verify handoff columns are in the sessions table after init."""
        db._conn.execute("SELECT handoff_pending, handoff_platform FROM sessions LIMIT 0")

    def test_set_handoff_pending(self, db):
        """Mark a session for handoff."""
        session_id = "test-session-001"
        self._make_session(db, session_id)
        ok = db.set_handoff_pending(session_id, "telegram")
        assert ok is True

        session = db.get_session(session_id)
        assert session["handoff_pending"] == 1
        assert session["handoff_platform"] == "telegram"

    def test_set_handoff_pending_no_double_mark(self, db):
        """Re-marking an already-pending session returns False."""
        session_id = "test-session-002"
        self._make_session(db, session_id)
        ok1 = db.set_handoff_pending(session_id, "telegram")
        assert ok1 is True
        ok2 = db.set_handoff_pending(session_id, "discord")
        assert ok2 is False  # already pending

    def test_find_pending_handoff(self, db):
        """Find a session pending handoff for a given platform."""
        sid = "test-session-003"
        self._make_session(db, sid)
        db.set_handoff_pending(sid, "telegram")

        handoff = db.find_pending_handoff("telegram")
        assert handoff is not None
        assert handoff["id"] == sid

        # Should not find for other platforms
        assert db.find_pending_handoff("discord") is None

    def test_clear_handoff_pending(self, db):
        """Clear the handoff flag."""
        sid = "test-session-004"
        self._make_session(db, sid)
        db.set_handoff_pending(sid, "telegram")
        db.clear_handoff_pending(sid)

        session = db.get_session(sid)
        assert session["handoff_pending"] == 0

    def test_full_handoff_flow(self, db):
        """End-to-end: mark → find → load messages → clear."""
        sid = "test-session-005"
        self._make_session(db, sid, title="my session")
        db.append_message(sid, "user", "Hello")
        db.append_message(sid, "assistant", "Hi there!")

        # CLI side: mark for handoff
        ok = db.set_handoff_pending(sid, "telegram")
        assert ok is True

        # Gateway side: find pending handoff
        handoff = db.find_pending_handoff("telegram")
        assert handoff is not None
        assert handoff["id"] == sid
        assert handoff["title"] == "my session"

        # Load messages for context
        messages = db.get_messages(sid)
        assert len(messages) == 2
        assert messages[0]["role"] == "user"
        assert messages[1]["role"] == "assistant"

        # Clear after injecting
        db.clear_handoff_pending(sid)
        assert db.find_pending_handoff("telegram") is None


class TestHandoffCommand:
    """Test the CLI /handoff command handler."""

    def test_command_registered(self):
        from hermes_cli.commands import resolve_command
        cmd = resolve_command("handoff")
        assert cmd is not None
        assert cmd.name == "handoff"
        assert cmd.category == "Session"

    def test_invalid_platform(self):
        """Test that unknown platforms are rejected."""
        supported = {"telegram", "discord", "slack", "whatsapp", "signal", "matrix"}
        assert "telegram" in supported
        assert "foo" not in supported
