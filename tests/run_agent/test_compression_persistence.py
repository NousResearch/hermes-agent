"""Tests for context compression persistence in the gateway.

Verifies that when context compression fires during run_conversation(),
the compressed messages are properly persisted to both SQLite (via the
agent) and JSONL (via the gateway).

Bug scenario (pre-fix):
  1. Gateway loads 200-message history, passes to agent
  2. Agent's run_conversation() compresses to ~30 messages mid-run
  3. _compress_context() resets _last_flushed_db_idx = 0
  4. On exit, _flush_messages_to_session_db() calculates:
     flush_from = max(len(conversation_history=200), _last_flushed_db_idx=0) = 200
  5. messages[200:] is empty (only ~30 messages after compression)
  6. Nothing written to new session's SQLite — compressed context lost
  7. Gateway's history_offset was still 200, producing empty new_messages
  8. Fallback wrote only user/assistant pair — summary lost
"""

import os
import sqlite3
import tempfile
from pathlib import Path
from unittest.mock import patch



# ---------------------------------------------------------------------------
# Part 1: Agent-side — _flush_messages_to_session_db after compression
# ---------------------------------------------------------------------------

class TestFlushAfterCompression:
    """Verify that compressed messages are flushed to the new session's SQLite
    even when conversation_history (from the original session) is longer than
    the compressed messages list."""

    def _make_agent(self, session_db):
        with patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"}):
            from run_agent import AIAgent
            agent = AIAgent(
                api_key="test-key",
                base_url="https://openrouter.ai/api/v1",
                model="test/model",
                quiet_mode=True,
                session_db=session_db,
                session_id="original-session",
                skip_context_files=True,
                skip_memory=True,
            )
        return agent

    def test_flush_after_compression_with_long_history(self):
        """The actual bug: conversation_history longer than compressed messages.

        Before the fix, flush_from = max(len(conversation_history), 0) = 200,
        but messages only has ~30 entries, so messages[200:] is empty.
        After the fix, conversation_history is cleared to None after compression,
        so flush_from = max(0, 0) = 0, and ALL compressed messages are written.
        """
        from hermes_state import SessionDB

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            db = SessionDB(db_path=db_path)

            agent = self._make_agent(db)

            # Simulate the original long history (200 messages)
            original_history = [
                {"role": "user" if i % 2 == 0 else "assistant",
                 "content": f"message {i}"}
                for i in range(200)
            ]

            # First, flush original messages to the original session
            agent._flush_messages_to_session_db(original_history, [])
            original_rows = db.get_messages("original-session")
            assert len(original_rows) == 200

            # Now simulate compression: new session, reset idx, shorter messages
            agent.session_id = "compressed-session"
            db.create_session(session_id="compressed-session", source="test")
            agent._last_flushed_db_idx = 0

            # The compressed messages (summary + tail + new turn)
            compressed_messages = [
                {"role": "user", "content": "[CONTEXT COMPACTION] Summary of work..."},
                {"role": "user", "content": "What should we do next?"},
                {"role": "assistant", "content": "Let me check..."},
                {"role": "user", "content": "new question"},
                {"role": "assistant", "content": "new answer"},
            ]

            # THE BUG: passing the original history as conversation_history
            # causes flush_from = max(200, 0) = 200, skipping everything.
            # After the fix, conversation_history should be None.
            agent._flush_messages_to_session_db(compressed_messages, None)

            new_rows = db.get_messages("compressed-session")
            assert len(new_rows) == 5, (
                f"Expected 5 compressed messages in new session, got {len(new_rows)}. "
                f"Compression persistence bug: messages not written to SQLite."
            )

    def test_flush_with_stale_history_loses_messages(self):
        """Demonstrates the bug condition: stale conversation_history causes data loss."""
        from hermes_state import SessionDB

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            db = SessionDB(db_path=db_path)

            agent = self._make_agent(db)

            # Simulate compression reset
            agent.session_id = "new-session"
            db.create_session(session_id="new-session", source="test")
            agent._last_flushed_db_idx = 0

            compressed = [
                {"role": "user", "content": "summary"},
                {"role": "assistant", "content": "continuing..."},
            ]

            # Bug: passing a conversation_history longer than compressed messages
            stale_history = [{"role": "user", "content": f"msg{i}"} for i in range(100)]
            agent._flush_messages_to_session_db(compressed, stale_history)

            rows = db.get_messages("new-session")
            # With the stale history, flush_from = max(100, 0) = 100
            # But compressed only has 2 entries → messages[100:] = empty
            assert len(rows) == 0, (
                "Expected 0 messages with stale conversation_history "
                "(this test verifies the bug condition exists)"
            )


class _FakeCompressor:
    """Minimal stand-in for the context compressor used in rotation tests.

    Returns a fixed, short compacted message list and reports a clean
    (non-aborted) compression so ``compress_context`` proceeds to the
    SQLite session-rotation block.
    """

    def __init__(self, compressed):
        self._compressed = compressed
        self.compression_count = 1
        self._last_compress_aborted = False
        self._last_summary_error = None
        self._last_aux_model_failure_model = None
        self._last_aux_model_failure_error = None

    def compress(self, messages, current_tokens=None, focus_topic=None, force=False):
        return list(self._compressed)


class TestRotationFlushCursorAtomicity:
    """Guard the compression session-rotation against partial DB failures.

    Regression for the data-loss bug where ``_last_flushed_db_idx`` was only
    reset at the very end of the rotation block — *after* the fallible
    ``create_session`` / title / ``update_system_prompt`` calls. A transient
    SQLite error there left ``session_id`` pointing at the new (empty) session
    while the flush cursor still held the stale pre-compression index. The next
    persist then computed ``flush_from`` past the end of the shortened
    compressed list and wrote NOTHING, permanently dropping the entire
    compressed conversation from resumable state.
    """

    def _make_agent(self, session_db):
        with patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"}):
            from run_agent import AIAgent
            agent = AIAgent(
                api_key="test-key",
                base_url="https://openrouter.ai/api/v1",
                model="test/model",
                quiet_mode=True,
                session_db=session_db,
                session_id="original-session",
                skip_context_files=True,
                skip_memory=True,
            )
        return agent

    def test_flush_cursor_reset_when_rotation_db_call_fails(self):
        from hermes_state import SessionDB
        from agent.conversation_compression import compress_context

        with tempfile.TemporaryDirectory() as tmpdir:
            db = SessionDB(db_path=Path(tmpdir) / "test.db")
            db.create_session(session_id="original-session", source="test")

            agent = self._make_agent(db)
            agent._session_db_created = True
            # Skip the just-in-time aux-model feasibility probe (network).
            agent._compression_feasibility_checked = True

            # Seed a long original history and flush it so the cursor advances
            # well past the length of the post-compression list.
            original = [
                {"role": "user" if i % 2 == 0 else "assistant", "content": f"m{i}"}
                for i in range(50)
            ]
            agent._flush_messages_to_session_db(original, [])
            assert agent._last_flushed_db_idx == 50

            compressed = [
                {"role": "user", "content": "[CONTEXT COMPACTION] summary of earlier work"},
                {"role": "user", "content": "recent question"},
                {"role": "assistant", "content": "recent answer"},
            ]
            agent.context_compressor = _FakeCompressor(compressed)

            # Make create_session raise exactly once for the rotated (new)
            # session id — a transient SQLite hiccup mid-rollover. The retry
            # via _ensure_db_session on the next flush must then succeed.
            real_create = db.create_session
            state = {"tripped": False}

            def flaky_create(*args, **kwargs):
                sid = kwargs.get("session_id")
                if sid != "original-session" and not state["tripped"]:
                    state["tripped"] = True
                    raise sqlite3.OperationalError("database is locked")
                return real_create(*args, **kwargs)

            db.create_session = flaky_create

            result, _sp = compress_context(agent, list(original), "system prompt")

            # The rotation happened (id changed) and the DB call failed.
            assert agent.session_id != "original-session"
            assert state["tripped"] is True

            # THE FIX: the flush cursor was reset to 0 atomically with the id
            # rotation, despite the create_session failure. Pre-fix this stayed
            # at 50, which would silently skip every compressed message below.
            assert agent._last_flushed_db_idx == 0, (
                "flush cursor must reset to 0 when the session id rotates, even "
                "if a DB step in the rotation block raised — otherwise the "
                "compressed history is permanently dropped"
            )

            # Recovery: the next persist re-creates the new session row and
            # writes the full compressed list, so nothing is lost.
            agent._flush_messages_to_session_db(result, None)
            new_rows = db.get_messages(agent.session_id)
            assert len(new_rows) == len(compressed), (
                f"Expected {len(compressed)} compressed messages persisted to the "
                f"new session, got {len(new_rows)} — compressed history was lost."
            )


# ---------------------------------------------------------------------------
# Part 2: Gateway-side — history_offset after session split
# ---------------------------------------------------------------------------

class TestGatewayHistoryOffsetAfterSplit:
    """Verify that when the agent creates a new session during compression,
    the gateway uses history_offset=0 so all compressed messages are written
    to the JSONL transcript."""

    def test_history_offset_zero_on_session_split(self):
        """When agent.session_id differs from the original, history_offset must be 0."""
        # This tests the logic in gateway/run.py run_sync():
        # _session_was_split = agent.session_id != session_id
        # _effective_history_offset = 0 if _session_was_split else len(agent_history)

        original_session_id = "session-abc"
        agent_session_id = "session-compressed-xyz"  # Different = compression happened
        agent_history_len = 200

        # Simulate the gateway's offset calculation (post-fix)
        _session_was_split = (agent_session_id != original_session_id)
        _effective_history_offset = 0 if _session_was_split else agent_history_len

        assert _session_was_split is True
        assert _effective_history_offset == 0

    def test_history_offset_preserved_without_split(self):
        """When no compression happened, history_offset is the original length."""
        session_id = "session-abc"
        agent_session_id = "session-abc"  # Same = no compression
        agent_history_len = 200

        _session_was_split = (agent_session_id != session_id)
        _effective_history_offset = 0 if _session_was_split else agent_history_len

        assert _session_was_split is False
        assert _effective_history_offset == 200

    def test_new_messages_extraction_after_split(self):
        """After compression with offset=0, new_messages should be ALL agent messages."""
        # Simulates the gateway's new_messages calculation
        agent_messages = [
            {"role": "user", "content": "[CONTEXT COMPACTION] Summary..."},
            {"role": "user", "content": "recent question"},
            {"role": "assistant", "content": "recent answer"},
            {"role": "user", "content": "new question"},
            {"role": "assistant", "content": "new answer"},
        ]
        history_offset = 0  # After fix: 0 on session split

        new_messages = agent_messages[history_offset:] if len(agent_messages) > history_offset else []
        assert len(new_messages) == 5, (
            f"Expected all 5 messages with offset=0, got {len(new_messages)}"
        )

    def test_new_messages_empty_with_stale_offset(self):
        """Demonstrates the bug: stale offset produces empty new_messages."""
        agent_messages = [
            {"role": "user", "content": "summary"},
            {"role": "assistant", "content": "answer"},
        ]
        # Bug: offset is the pre-compression history length
        history_offset = 200

        new_messages = agent_messages[history_offset:] if len(agent_messages) > history_offset else []
        assert len(new_messages) == 0, (
            "Expected 0 messages with stale offset=200 (demonstrates the bug)"
        )
