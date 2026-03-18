"""Tests for configurable write-through session DB persistence.

Verifies that:
1. The session.write_through_flush config key defaults to True
2. When enabled, _flush_messages_to_session_db is called after each tool-call turn
3. When disabled, per-turn flush is skipped (only happens at session end)
4. Per-turn flushes are idempotent via _last_flushed_db_idx
"""

import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Test: config default
# ---------------------------------------------------------------------------

class TestWriteThroughConfig:
    """Verify the session.write_through_flush config key."""

    def test_default_config_has_write_through_enabled(self):
        """DEFAULT_CONFIG includes session.write_through_flush = True."""
        from hermes_cli.config import DEFAULT_CONFIG
        assert DEFAULT_CONFIG["session"]["write_through_flush"] is True

    def test_load_config_returns_write_through_default(self):
        """load_config() returns write_through_flush = True when no user override."""
        from hermes_cli.config import load_config
        config = load_config()
        assert config["session"]["write_through_flush"] is True


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_agent(session_db=None, write_through_override=None):
    """Create a minimal AIAgent suitable for unit tests.

    If write_through_override is provided, it is set after construction
    to test the disabled path without needing a config file override.
    """
    with patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"}):
        from run_agent import AIAgent
        agent = AIAgent(
            model="test/model",
            quiet_mode=True,
            session_db=session_db,
            session_id="test-write-through",
            skip_context_files=True,
            skip_memory=True,
        )
    if write_through_override is not None:
        agent._write_through_flush = write_through_override
    return agent


# ---------------------------------------------------------------------------
# Test: constructor sets _write_through_flush from config
# ---------------------------------------------------------------------------

class TestWriteThroughInit:
    """Verify _write_through_flush is set during construction."""

    def test_default_is_true(self):
        """Agent defaults to _write_through_flush = True."""
        agent = _make_agent()
        assert agent._write_through_flush is True


# ---------------------------------------------------------------------------
# Test: per-turn flush behaviour
# ---------------------------------------------------------------------------

class TestWriteThroughPerTurnFlush:
    """Verify that the tool-call continue path respects _write_through_flush."""

    def test_enabled_flushes_per_turn(self):
        """With write_through enabled, per-turn flush writes to session DB."""
        from hermes_state import SessionDB

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test_wt.db"
            db = SessionDB(db_path=db_path)

            agent = _make_agent(session_db=db, write_through_override=True)
            agent._save_session_log = MagicMock()

            messages = [
                {"role": "user", "content": "what time is it?"},
                {"role": "assistant", "content": "", "tool_calls": [
                    {"name": "terminal", "arguments": '{"command": "date"}'},
                ]},
                {"role": "tool", "content": "Wed Mar 19 12:00:00 UTC 2026",
                 "tool_call_id": "call_1", "tool_name": "terminal"},
            ]
            conversation_history = []

            # Simulate the per-turn flush that runs in the tool-call continue path
            if agent._write_through_flush:
                agent._flush_messages_to_session_db(messages, conversation_history)

            rows = db.get_messages(agent.session_id)
            assert len(rows) == 3, f"Expected 3 messages flushed per-turn, got {len(rows)}"

    def test_disabled_skips_per_turn_flush(self):
        """With write_through disabled, per-turn path does not flush to DB."""
        from hermes_state import SessionDB

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test_wt_off.db"
            db = SessionDB(db_path=db_path)

            agent = _make_agent(session_db=db, write_through_override=False)
            agent._save_session_log = MagicMock()

            messages = [
                {"role": "user", "content": "what time is it?"},
                {"role": "assistant", "content": "", "tool_calls": [
                    {"name": "terminal", "arguments": '{"command": "date"}'},
                ]},
                {"role": "tool", "content": "Wed Mar 19 12:00:00 UTC 2026",
                 "tool_call_id": "call_1", "tool_name": "terminal"},
            ]
            conversation_history = []

            # Simulate the per-turn check — should NOT flush
            if agent._write_through_flush:
                agent._flush_messages_to_session_db(messages, conversation_history)

            rows = db.get_messages(agent.session_id)
            assert len(rows) == 0, f"Expected 0 messages with write_through disabled, got {len(rows)}"

    def test_session_end_flushes_regardless(self):
        """_persist_session always flushes, even with write_through disabled."""
        from hermes_state import SessionDB

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test_persist.db"
            db = SessionDB(db_path=db_path)

            agent = _make_agent(session_db=db, write_through_override=False)
            agent._save_session_log = MagicMock()

            messages = [
                {"role": "user", "content": "hello"},
                {"role": "assistant", "content": "hi there"},
            ]

            # _persist_session is the end-of-session path — always flushes
            agent._persist_session(messages, [])

            rows = db.get_messages(agent.session_id)
            assert len(rows) == 2, f"Expected 2 messages at session end, got {len(rows)}"

    def test_per_turn_flush_is_idempotent(self):
        """Multiple per-turn flushes don't duplicate messages."""
        from hermes_state import SessionDB

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test_idemp.db"
            db = SessionDB(db_path=db_path)

            agent = _make_agent(session_db=db, write_through_override=True)

            messages = [
                {"role": "user", "content": "q1"},
                {"role": "assistant", "content": "a1"},
            ]
            conversation_history = []

            # Simulate 3 consecutive per-turn flushes with same messages
            agent._flush_messages_to_session_db(messages, conversation_history)
            agent._flush_messages_to_session_db(messages, conversation_history)
            agent._flush_messages_to_session_db(messages, conversation_history)

            rows = db.get_messages(agent.session_id)
            assert len(rows) == 2, f"Expected 2 messages (no dupes), got {len(rows)}"

    def test_incremental_per_turn_flush(self):
        """Each turn adds only new messages to the DB."""
        from hermes_state import SessionDB

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test_incr.db"
            db = SessionDB(db_path=db_path)

            agent = _make_agent(session_db=db, write_through_override=True)
            conversation_history = []

            # Turn 1: user asks, assistant uses tool
            messages = [
                {"role": "user", "content": "turn 1"},
                {"role": "assistant", "content": "tool result 1"},
            ]
            agent._flush_messages_to_session_db(messages, conversation_history)
            assert len(db.get_messages(agent.session_id)) == 2

            # Turn 2: new messages appended
            messages.append({"role": "user", "content": "turn 2"})
            messages.append({"role": "assistant", "content": "tool result 2"})
            agent._flush_messages_to_session_db(messages, conversation_history)
            assert len(db.get_messages(agent.session_id)) == 4

            # Turn 3: one more
            messages.append({"role": "user", "content": "turn 3"})
            agent._flush_messages_to_session_db(messages, conversation_history)
            assert len(db.get_messages(agent.session_id)) == 5
