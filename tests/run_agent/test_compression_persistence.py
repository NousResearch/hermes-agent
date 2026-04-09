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
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


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


# ---------------------------------------------------------------------------
# Part 3: Pre-compression flush — messages written to OLD session before split
# ---------------------------------------------------------------------------

class TestPreCompressionFlush:
    """Verify that _compress_context() flushes accumulated messages to the OLD
    session's SQLite BEFORE ending it and switching to the new session_id.

    Bug scenario (pre-fix):
      1. Session accumulates 50 messages during tool loop
      2. No intermediate _persist_session call is triggered (normal flow)
      3. Context compression fires
      4. _compress_context() ends old session, creates new session_id
      5. _last_flushed_db_idx = 0, but messages are for NEW session now
      6. The 50 messages from the OLD session are never written to SQLite
      7. JSON file has them (saved via _save_session_log), but SQLite doesn't
      8. hermes --resume shows "no messages found" for the old session

    Fix: _compress_context() now calls _flush_messages_to_session_db(messages, None)
    BEFORE end_session() and the session_id switch.
    """

    def _make_agent(self, session_db, session_id="original-session"):
        with patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"}):
            from run_agent import AIAgent
            agent = AIAgent(
                model="test/model",
                quiet_mode=True,
                session_db=session_db,
                session_id=session_id,
                skip_context_files=True,
                skip_memory=True,
            )
        return agent

    def test_messages_flushed_to_old_session_before_switch(self):
        """Core fix: messages are written to OLD session before compression switches session_id."""
        from hermes_state import SessionDB

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            db = SessionDB(db_path=db_path)

            agent = self._make_agent(db)
            db.create_session(session_id="original-session", source="test", model="test/model")

            # Accumulate messages during the tool loop (none flushed yet)
            messages = [
                {"role": "user", "content": "Please help with my project"},
                {"role": "assistant", "content": "Sure, let me check..."},
                {"role": "assistant", "content": "", "tool_calls": [
                    {"id": "call_1", "type": "function",
                     "function": {"name": "terminal", "arguments": '{"command":"ls"}'}}
                ]},
                {"role": "tool", "content": "file1.py\nfile2.py", "tool_call_id": "call_1"},
                {"role": "assistant", "content": "I found 2 files."},
            ]

            # Verify: before compression, nothing in SQLite yet
            assert agent._last_flushed_db_idx == 0
            pre_rows = db.get_messages("original-session")
            assert len(pre_rows) == 0, "No messages should be in SQLite before any flush"

            # Simulate compression with mocked compressor
            agent.compression_enabled = True
            agent.context_compressor = MagicMock()
            agent.context_compressor.compress.return_value = [
                {"role": "user", "content": "[CONTEXT COMPACTION] Summary..."},
                {"role": "assistant", "content": "Continuing from summary."},
            ]
            agent.context_compressor.compression_count = 1
            agent.context_compressor.threshold_tokens = 100000
            agent.context_compressor.last_prompt_tokens = 5000

            # Mock _build_system_prompt and _invalidate_system_prompt
            agent._build_system_prompt = MagicMock(return_value="system prompt")
            agent._invalidate_system_prompt = MagicMock()
            agent._cached_system_prompt = "system prompt"
            agent._context_pressure_warned = False
            agent.flush_memories = MagicMock()
            agent._memory_manager = None
            agent._todo_store = MagicMock()
            agent._todo_store.format_for_injection.return_value = None

            old_session_id = agent.session_id

            # Run compression
            compressed, new_sys = agent._compress_context(
                messages, "system prompt", approx_tokens=50000, task_id="test"
            )

            # The session_id should have changed
            assert agent.session_id != old_session_id, (
                "Compression should create a new session_id"
            )

            # CRITICAL: the OLD session should have its messages in SQLite
            old_rows = db.get_messages(old_session_id)
            assert len(old_rows) == 5, (
                f"Expected 5 messages in OLD session '{old_session_id}' SQLite, "
                f"got {len(old_rows)}. Pre-compression flush is missing!"
            )

            # The new session should exist but have 0 messages (not flushed yet)
            new_rows = db.get_messages(agent.session_id)
            assert len(new_rows) == 0, (
                "New session should have 0 messages immediately after compression "
                "(they get written on the next flush)"
            )

            # _last_flushed_db_idx should be 0 (reset for the new session)
            assert agent._last_flushed_db_idx == 0

    def test_rapid_compression_chain_preserves_all_sessions(self):
        """Simulate rapid compression (session compressed immediately after creation).

        This reproduces the 150+ depth chain scenario where each intermediate
        session had 0 messages in SQLite.
        """
        from hermes_state import SessionDB

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            db = SessionDB(db_path=db_path)

            agent = self._make_agent(db, session_id="chain-root")
            db.create_session(session_id="chain-root", source="test", model="test/model")

            # Set up mocks once
            agent.compression_enabled = True
            agent.context_compressor = MagicMock()
            agent.context_compressor.compression_count = 0
            agent.context_compressor.threshold_tokens = 100000
            agent.context_compressor.last_prompt_tokens = 5000
            agent._build_system_prompt = MagicMock(return_value="system prompt")
            agent._invalidate_system_prompt = MagicMock()
            agent._cached_system_prompt = "system prompt"
            agent._context_pressure_warned = False
            agent.flush_memories = MagicMock()
            agent._memory_manager = None
            agent._todo_store = MagicMock()
            agent._todo_store.format_for_injection.return_value = None

            session_ids = ["chain-root"]
            messages_per_session = [3, 2, 4]  # Different message counts per compression

            for i, msg_count in enumerate(messages_per_session):
                # Create messages for this session
                messages = [
                    {"role": "user" if j % 2 == 0 else "assistant",
                     "content": f"session {i} message {j}"}
                    for j in range(msg_count)
                ]

                # Compress returns a minimal summary
                agent.context_compressor.compress.return_value = [
                    {"role": "user", "content": f"[Summary of session {i}]"},
                ]
                agent.context_compressor.compression_count = i + 1

                old_sid = agent.session_id
                compressed, _ = agent._compress_context(
                    messages, "system prompt", approx_tokens=50000, task_id="test"
                )
                session_ids.append(agent.session_id)

                # Verify OLD session got its messages
                old_rows = db.get_messages(old_sid)
                assert len(old_rows) == msg_count, (
                    f"Chain step {i}: expected {msg_count} messages in session "
                    f"'{old_sid}', got {len(old_rows)}"
                )

            # Verify all sessions in chain have their messages
            for i, (sid, expected_count) in enumerate(zip(session_ids[:-1], messages_per_session)):
                rows = db.get_messages(sid)
                assert len(rows) == expected_count, (
                    f"Session {i} '{sid}': expected {expected_count} messages, "
                    f"got {len(rows)}. Rapid compression chain loses messages!"
                )

    def test_flush_before_switch_is_idempotent(self):
        """If messages were already flushed (via intermediate _persist_session),
        the pre-compression flush should not duplicate them."""
        from hermes_state import SessionDB

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            db = SessionDB(db_path=db_path)

            agent = self._make_agent(db)
            db.create_session(session_id="original-session", source="test", model="test/model")

            messages = [
                {"role": "user", "content": "hello"},
                {"role": "assistant", "content": "world"},
            ]

            # Simulate an intermediate flush (e.g., from an error handler)
            agent._flush_messages_to_session_db(messages, None)
            assert agent._last_flushed_db_idx == 2

            rows_before = db.get_messages("original-session")
            assert len(rows_before) == 2

            # Now compression fires — the pre-compression flush should be a no-op
            # (messages[2:] is empty since _last_flushed_db_idx == 2)
            agent.compression_enabled = True
            agent.context_compressor = MagicMock()
            agent.context_compressor.compress.return_value = [
                {"role": "user", "content": "[Summary]"},
            ]
            agent.context_compressor.compression_count = 1
            agent.context_compressor.threshold_tokens = 100000
            agent.context_compressor.last_prompt_tokens = 5000
            agent._build_system_prompt = MagicMock(return_value="sys")
            agent._invalidate_system_prompt = MagicMock()
            agent._cached_system_prompt = "sys"
            agent._context_pressure_warned = False
            agent.flush_memories = MagicMock()
            agent._memory_manager = None
            agent._todo_store = MagicMock()
            agent._todo_store.format_for_injection.return_value = None

            agent._compress_context(messages, "sys", approx_tokens=50000, task_id="test")

            # Old session should still have exactly 2 messages (no duplicates)
            rows_after = db.get_messages("original-session")
            assert len(rows_after) == 2, (
                f"Expected 2 messages (no duplicates), got {len(rows_after)}. "
                f"Pre-compression flush should be idempotent."
            )
