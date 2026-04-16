"""Tests for issue #860 — SQLite session transcript deduplication.

Verifies that:
1. _flush_messages_to_session_db uses _last_flushed_db_idx to avoid re-writing
2. Multiple _persist_session calls don't duplicate messages
3. append_to_transcript(skip_db=True) avoids duplicate SQLite writes
4. The gateway doesn't double-write messages the agent already persisted
"""

import json
import os
import sqlite3
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Test: _flush_messages_to_session_db only writes new messages
# ---------------------------------------------------------------------------

class TestFlushDeduplication:
    """Verify _flush_messages_to_session_db tracks what it already wrote."""

    def _make_agent(self, session_db):
        """Create a minimal AIAgent with a real session DB."""
        with patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"}):
            from run_agent import AIAgent
            agent = AIAgent(
                model="test/model",
                quiet_mode=True,
                session_db=session_db,
                session_id="test-session-860",
                skip_context_files=True,
                skip_memory=True,
            )
        return agent

    def test_flush_writes_only_new_messages(self):
        """First flush writes all new messages, second flush writes none."""
        from hermes_state import SessionDB

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            db = SessionDB(db_path=db_path)

            agent = self._make_agent(db)

            conversation_history = [
                {"role": "user", "content": "old message"},
            ]
            messages = list(conversation_history) + [
                {"role": "user", "content": "new question"},
                {"role": "assistant", "content": "new answer"},
            ]

            # First flush — should write 2 new messages
            agent._flush_messages_to_session_db(messages, conversation_history)

            rows = db.get_messages(agent.session_id)
            assert len(rows) == 2, f"Expected 2 messages, got {len(rows)}"

            # Second flush with SAME messages — should write 0 new messages
            agent._flush_messages_to_session_db(messages, conversation_history)

            rows = db.get_messages(agent.session_id)
            assert len(rows) == 2, f"Expected still 2 messages after second flush, got {len(rows)}"

    def test_flush_writes_incrementally(self):
        """Messages added between flushes are written exactly once."""
        from hermes_state import SessionDB

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            db = SessionDB(db_path=db_path)

            agent = self._make_agent(db)

            conversation_history = []
            messages = [
                {"role": "user", "content": "hello"},
            ]

            # First flush — 1 message
            agent._flush_messages_to_session_db(messages, conversation_history)
            rows = db.get_messages(agent.session_id)
            assert len(rows) == 1

            # Add more messages
            messages.append({"role": "assistant", "content": "hi there"})
            messages.append({"role": "user", "content": "follow up"})

            # Second flush — should write only 2 new messages
            agent._flush_messages_to_session_db(messages, conversation_history)
            rows = db.get_messages(agent.session_id)
            assert len(rows) == 3, f"Expected 3 total messages, got {len(rows)}"

    def test_persist_session_multiple_calls_no_duplication(self):
        """Multiple _persist_session calls don't duplicate DB entries."""
        from hermes_state import SessionDB

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            db = SessionDB(db_path=db_path)

            agent = self._make_agent(db)
            # Stub out _save_session_log to avoid file I/O
            agent._save_session_log = MagicMock()

            conversation_history = [{"role": "user", "content": "old"}]
            messages = list(conversation_history) + [
                {"role": "user", "content": "q1"},
                {"role": "assistant", "content": "a1"},
                {"role": "user", "content": "q2"},
                {"role": "assistant", "content": "a2"},
            ]

            # Simulate multiple persist calls (like the agent's many exit paths)
            for _ in range(5):
                agent._persist_session(messages, conversation_history)

            rows = db.get_messages(agent.session_id)
            assert len(rows) == 4, f"Expected 4 messages, got {len(rows)} (duplication bug!)"

    def test_flush_reset_after_compression(self):
        """After compression creates a new session, flush index resets."""
        from hermes_state import SessionDB

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            db = SessionDB(db_path=db_path)

            agent = self._make_agent(db)

            # Write some messages
            messages = [
                {"role": "user", "content": "msg1"},
                {"role": "assistant", "content": "reply1"},
            ]
            agent._flush_messages_to_session_db(messages, [])

            old_session = agent.session_id
            assert agent._last_flushed_db_idx == 2

            # Simulate what _compress_context does: new session, reset idx
            agent.session_id = "compressed-session-new"
            db.create_session(session_id=agent.session_id, source="test")
            agent._last_flushed_db_idx = 0

            # Now flush compressed messages to new session
            compressed_messages = [
                {"role": "user", "content": "summary of conversation"},
            ]
            agent._flush_messages_to_session_db(compressed_messages, [])

            new_rows = db.get_messages(agent.session_id)
            assert len(new_rows) == 1

            # Old session should still have its 2 messages
            old_rows = db.get_messages(old_session)
            assert len(old_rows) == 2

    def test_partial_flush_tracks_persisted_prefix_count(self):
        """A partial DB failure should report the durable prefix length."""
        from hermes_state import SessionDB

        class FlakySessionDB:
            def __init__(self, wrapped):
                self._wrapped = wrapped
                self._append_calls = 0

            def create_session(self, *args, **kwargs):
                return self._wrapped.create_session(*args, **kwargs)

            def ensure_session(self, *args, **kwargs):
                return self._wrapped.ensure_session(*args, **kwargs)

            def append_message(self, *args, **kwargs):
                self._append_calls += 1
                if self._append_calls == 2:
                    raise sqlite3.OperationalError("database is locked")
                return self._wrapped.append_message(*args, **kwargs)

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            real_db = SessionDB(db_path=db_path)
            flaky_db = FlakySessionDB(real_db)

            agent = self._make_agent(flaky_db)
            messages = [
                {"role": "user", "content": "new question"},
                {"role": "assistant", "content": "new answer"},
            ]

            persisted = agent._flush_messages_to_session_db(messages, [])

            rows = real_db.get_messages(agent.session_id)
            assert persisted == 1
            assert agent._last_flushed_db_idx == 1
            assert len(rows) == 1
            assert rows[0]["content"] == "new question"


# ---------------------------------------------------------------------------
# Test: append_to_transcript skip_db parameter
# ---------------------------------------------------------------------------

class TestAppendToTranscriptSkipDb:
    """Verify skip_db=True avoids duplicate SQLite writes."""

    @pytest.fixture()
    def store(self, tmp_path):
        from gateway.config import GatewayConfig
        from gateway.session import SessionStore
        config = GatewayConfig()
        with patch("gateway.session.SessionStore._ensure_loaded"):
            s = SessionStore(sessions_dir=tmp_path, config=config)
        s._db = None  # force legacy fallback mode
        s._loaded = True
        return s

    def test_skip_db_uses_jsonl_fallback_without_sqlite(self, store, tmp_path):
        """Without SQLite, JSONL remains the fallback transcript backend."""
        session_id = "test-skip-db"
        msg = {"role": "assistant", "content": "hello world"}
        store.append_to_transcript(session_id, msg, skip_db=True)

        jsonl_path = store.get_transcript_path(session_id)
        assert jsonl_path.exists()
        with open(jsonl_path) as f:
            lines = f.readlines()
        assert len(lines) == 1
        parsed = json.loads(lines[0])
        assert parsed["content"] == "hello world"

    def test_skip_db_prevents_sqlite_write(self, tmp_path):
        """With skip_db=True and a real DB, SessionStore writes nowhere."""
        from gateway.config import GatewayConfig
        from gateway.session import SessionStore
        from hermes_state import SessionDB

        db_path = tmp_path / "test_skip.db"
        db = SessionDB(db_path=db_path)

        config = GatewayConfig()
        with patch("gateway.session.SessionStore._ensure_loaded"):
            store = SessionStore(sessions_dir=tmp_path, config=config)
        store._db = db
        store._loaded = True

        session_id = "test-skip-db-real"
        db.create_session(session_id=session_id, source="test")

        msg = {"role": "assistant", "content": "hello world"}
        store.append_to_transcript(session_id, msg, skip_db=True)

        rows = db.get_messages(session_id)
        assert len(rows) == 0, f"Expected 0 DB rows with skip_db=True, got {len(rows)}"

        jsonl_path = store.get_transcript_path(session_id)
        assert not jsonl_path.exists()

    def test_default_writes_sqlite_only_when_db_available(self, tmp_path):
        """Without skip_db, SQLite is canonical and JSONL stays untouched."""
        from gateway.config import GatewayConfig
        from gateway.session import SessionStore
        from hermes_state import SessionDB

        db_path = tmp_path / "test_both.db"
        db = SessionDB(db_path=db_path)

        config = GatewayConfig()
        with patch("gateway.session.SessionStore._ensure_loaded"):
            store = SessionStore(sessions_dir=tmp_path, config=config)
        store._db = db
        store._loaded = True

        session_id = "test-default-write"
        db.create_session(session_id=session_id, source="test")

        msg = {"role": "user", "content": "test message"}
        store.append_to_transcript(session_id, msg)

        jsonl_path = store.get_transcript_path(session_id)
        assert not jsonl_path.exists()

        rows = db.get_messages(session_id)
        assert len(rows) == 1

    def test_partial_agent_flush_can_be_completed_without_duplicates(self, tmp_path):
        """Gateway can persist only the unflushed tail after a partial DB write."""
        from gateway.config import GatewayConfig
        from gateway.session import SessionStore
        from hermes_state import SessionDB

        db_path = tmp_path / "test_partial.db"
        db = SessionDB(db_path=db_path)

        config = GatewayConfig()
        with patch("gateway.session.SessionStore._ensure_loaded"):
            store = SessionStore(sessions_dir=tmp_path, config=config)
        store._db = db
        store._loaded = True

        session_id = "test-partial-write"
        db.create_session(session_id=session_id, source="test")
        db.append_message(session_id=session_id, role="user", content="hello")

        new_messages = [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "world"},
        ]
        persisted_count = 1
        for idx, msg in enumerate(new_messages):
            store.append_to_transcript(
                session_id,
                {**msg, "timestamp": "2026-04-16T00:00:00"},
                skip_db=idx < persisted_count,
            )

        rows = db.get_messages_as_conversation(session_id)
        assert [row["content"] for row in rows] == ["hello", "world"]
        assert not store.get_transcript_path(session_id).exists()


# ---------------------------------------------------------------------------
# Test: _last_flushed_db_idx initialization
# ---------------------------------------------------------------------------

class TestFlushIdxInit:
    """Verify _last_flushed_db_idx is properly initialized."""

    def test_init_zero(self):
        """Agent starts with _last_flushed_db_idx = 0."""
        with patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"}):
            from run_agent import AIAgent
            agent = AIAgent(
                model="test/model",
                quiet_mode=True,
                skip_context_files=True,
                skip_memory=True,
            )
        assert agent._last_flushed_db_idx == 0

    def test_no_session_db_noop(self):
        """Without session_db, flush is a no-op and doesn't crash."""
        with patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"}):
            from run_agent import AIAgent
            agent = AIAgent(
                model="test/model",
                quiet_mode=True,
                skip_context_files=True,
                skip_memory=True,
            )
        messages = [{"role": "user", "content": "test"}]
        agent._flush_messages_to_session_db(messages, [])
        # Should not crash, idx should remain 0
        assert agent._last_flushed_db_idx == 0
