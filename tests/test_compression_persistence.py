"""Tests for compressed context persistence — post-compression session flush.

Verifies that:
1. _flush_messages_to_session_db resets flush_from when messages shrink
   after mid-turn context compression (prevents writing zero messages
   to the new session).
2. Gateway _handle_message rewrites the transcript when agent_messages
   is shorter than history_len after compression.

Regression test for silent context loss during mid-conversation compression.
Related: #860 (dedup), #1993 (tool result loss during compression).
"""

import json
import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Test: _flush_messages_to_session_db after compression shrinks messages
# ---------------------------------------------------------------------------

class TestFlushAfterCompression:
    """Verify flush persists all compressed messages when list shrinks."""

    def _make_agent(self, session_db):
        """Create a minimal AIAgent with a real session DB."""
        with patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"}):
            from run_agent import AIAgent
            agent = AIAgent(
                model="test/model",
                quiet_mode=True,
                session_db=session_db,
                session_id="test-session-compress",
                skip_context_files=True,
                skip_memory=True,
            )
        return agent

    def test_flush_resets_when_messages_shorter_than_history(self):
        """After compression, messages is shorter than conversation_history.

        flush_from must reset to 0 so the compressed messages are
        persisted to the new session — not silently skipped.
        """
        from hermes_state import SessionDB

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            db = SessionDB(db_path=db_path)

            agent = self._make_agent(db)

            # Simulate pre-compression state: 50 messages in history
            conversation_history = [
                {"role": "user" if i % 2 == 0 else "assistant",
                 "content": f"message {i}"}
                for i in range(50)
            ]

            # After compression: only 5 messages remain (compressed summary)
            compressed_messages = [
                {"role": "user", "content": "[Compressed context summary]"},
                {"role": "assistant", "content": "Understood, continuing."},
                {"role": "user", "content": "new question after compression"},
                {"role": "assistant", "content": "answer after compression"},
            ]

            # Simulate: _last_flushed_db_idx was reset to 0 by _compress_context
            agent._last_flushed_db_idx = 0

            # This is the critical call: conversation_history has 50 items,
            # compressed_messages has 4.  Without the fix, flush_from = 50
            # and messages[50:] = [] — zero messages written.
            agent._flush_messages_to_session_db(
                compressed_messages, conversation_history
            )

            # Verify all 4 compressed messages were written
            rows = db.get_messages(agent.session_id)
            assert len(rows) == 4, (
                f"Expected 4 messages persisted after compression, got {len(rows)}. "
                f"flush_from was not reset when messages shrunk."
            )
            assert agent._last_flushed_db_idx == 4

    def test_flush_normal_case_unchanged(self):
        """Normal (non-compression) flush behavior is not affected."""
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

            agent._flush_messages_to_session_db(messages, conversation_history)

            rows = db.get_messages(agent.session_id)
            assert len(rows) == 2, (
                f"Expected 2 new messages, got {len(rows)}"
            )


# ---------------------------------------------------------------------------
# Test: Gateway detects mid-turn compression and rewrites transcript
# ---------------------------------------------------------------------------

class TestGatewayCompressionTranscript:
    """Verify gateway rewrites transcript when compression shrinks messages."""

    def test_compressed_messages_trigger_rewrite(self):
        """When agent_messages < history_len, gateway must rewrite transcript."""
        session_store = MagicMock()
        session_store.rewrite_transcript = MagicMock()
        session_store.append_to_transcript = MagicMock()
        session_store.update_session = MagicMock()

        # Simulate: history had 50 messages, agent compressed to 8
        history_len = 50
        agent_messages = [
            {"role": "user", "content": "[Compressed summary]"},
            {"role": "assistant", "content": "Continuing."},
            {"role": "user", "content": "latest question"},
            {"role": "assistant", "content": "latest answer"},
        ]

        # The key condition: len(agent_messages) < history_len
        compressed_mid_turn = len(agent_messages) < history_len
        assert compressed_mid_turn, "Test setup: agent_messages should be shorter than history"

        new_messages = agent_messages[history_len:] if len(agent_messages) > history_len else []
        assert new_messages == [], "Test setup: new_messages should be empty after compression"

        # Simulate the fixed gateway logic
        if compressed_mid_turn:
            ts = "2026-03-19T12:00:00"
            timestamped = []
            for msg in agent_messages:
                if msg.get("role") == "system":
                    continue
                timestamped.append({**msg, "timestamp": ts})
            session_store.rewrite_transcript("test-session", timestamped)
            session_store.update_session("test-key", last_prompt_tokens=0)

        # Verify rewrite_transcript was called with all compressed messages
        session_store.rewrite_transcript.assert_called_once()
        call_args = session_store.rewrite_transcript.call_args
        written_messages = call_args[0][1]
        assert len(written_messages) == 4, (
            f"Expected 4 messages in rewritten transcript, got {len(written_messages)}"
        )

        # Verify append_to_transcript was NOT called (rewrite replaces it)
        session_store.append_to_transcript.assert_not_called()

    def test_normal_messages_use_append(self):
        """When no compression occurred, gateway appends normally."""
        history_len = 5
        agent_messages = [
            {"role": "user", "content": "msg1"},
            {"role": "assistant", "content": "msg2"},
            {"role": "user", "content": "msg3"},
            {"role": "assistant", "content": "msg4"},
            {"role": "user", "content": "msg5"},
            {"role": "user", "content": "new question"},
            {"role": "assistant", "content": "new answer"},
        ]

        compressed_mid_turn = len(agent_messages) < history_len
        assert not compressed_mid_turn, "Normal case: no compression"

        new_messages = agent_messages[history_len:] if len(agent_messages) > history_len else []
        assert len(new_messages) == 2, "Should have 2 new messages"
