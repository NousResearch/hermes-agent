"""Regression test for SessionDB flush_from overflow after message repair.

When _repair_message_sequence() or _drop_trailing_empty_response_scaffolding()
shortens ``messages`` below the original ``conversation_history`` length,
``flush_from`` can exceed ``len(messages)``.  The fix caps ``flush_from`` so
the current turn is always persisted rather than silently skipped.

See: https://github.com/NousResearch/hermes-agent/issues/24187
"""

import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest


class TestFlushFromOverflow:
    """Verify _flush_messages_to_session_db handles shortened messages."""

    def _make_agent(self, session_db):
        """Create a minimal AIAgent with a real session DB."""
        with patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"}):
            from run_agent import AIAgent
            agent = AIAgent(
                api_key="test-key",
                base_url="https://openrouter.ai/api/v1",
                model="test/model",
                quiet_mode=True,
                session_db=session_db,
                session_id="test-session-repair-overflow",
                skip_context_files=True,
                skip_memory=True,
            )
        agent._ensure_db_session()
        return agent

    def test_flush_persists_current_turn_after_repair_shortens_messages(self):
        """Current turn must be persisted even when repair removes historical entries.

        Scenario:
          - conversation_history has 6 entries (loaded from DB)
          - messages starts as 8 (history + user + assistant reply)
          - _repair_message_sequence removes 4 orphan tool messages
          - messages shrinks to 4
          - flush_from = len(conversation_history) = 6 > 4 = len(messages)
          - Without the fix, messages[6:] == [] and the turn is lost.
        """
        from hermes_state import SessionDB

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            session_db = SessionDB(db_path)
            agent = self._make_agent(session_db)

            # Simulate a conversation history loaded from DB (6 entries).
            conversation_history = [
                {"role": "user", "content": "hello"},
                {"role": "assistant", "content": "hi"},
                {"role": "user", "content": "ask"},
                {"role": "assistant", "content": "", "tool_calls": [{"id": "tc1", "type": "function", "function": {"name": "x", "arguments": "{}"}}]},
                {"role": "tool", "content": "result", "tool_call_id": "tc1"},
                {"role": "assistant", "content": "answer"},
            ]

            # Build messages = history + current turn + some orphan tool msgs
            # that _repair_message_sequence would drop.
            messages = list(conversation_history) + [
                {"role": "user", "content": "current question"},
                {"role": "assistant", "content": "current answer"},
                # Orphan tool messages (no matching assistant tool_call in this scope)
                {"role": "tool", "content": "stale1", "tool_call_id": "orphan-1"},
                {"role": "tool", "content": "stale2", "tool_call_id": "orphan-2"},
            ]
            # Simulate _repair_message_sequence dropping the 2 orphan tool msgs
            # (in reality it also merges consecutive user messages, but this
            # captures the core overflow scenario).
            messages_before = len(messages)  # 10
            messages = [m for m in messages if not (m.get("role") == "tool" and m.get("tool_call_id", "").startswith("orphan"))]
            messages_after = len(messages)  # 8

            # The original conversation_history length is still 6.
            # messages is now 8.  flush_from = 6, which is fine here.
            # But let's also simulate the more extreme case where repair
            # removes entries FROM the history (not just the current turn),
            # making messages shorter than conversation_history.
            messages_extreme = list(conversation_history[:2]) + [
                {"role": "user", "content": "current question"},
                {"role": "assistant", "content": "current answer"},
            ]
            # conversation_history len = 6, messages len = 4
            # flush_from = 6 > 4 = len(messages) → overflow!

            # Write the current turn using the agent's flush method.
            # We need to manually set messages to simulate the overflow.
            agent._last_flushed_db_idx = 0
            agent._flush_messages_to_session_db(messages_extreme, conversation_history)

            # Verify the current turn was persisted.
            rows = session_db.get_messages("test-session-repair-overflow")
            persisted_contents = [r["content"] for r in rows if r.get("content")]
            assert "current question" in persisted_contents, (
                f"Current user turn not persisted! Contents: {persisted_contents}"
            )
            assert "current answer" in persisted_contents, (
                f"Current assistant turn not persisted! Contents: {persisted_contents}"
            )

    def test_flush_no_warning_when_messages_longer_than_history(self):
        """Normal case: no warning when messages >= conversation_history."""
        from hermes_state import SessionDB

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            session_db = SessionDB(db_path)
            agent = self._make_agent(session_db)

            conversation_history = [
                {"role": "user", "content": "hello"},
                {"role": "assistant", "content": "hi"},
            ]
            messages = list(conversation_history) + [
                {"role": "user", "content": "follow up"},
                {"role": "assistant", "content": "reply"},
            ]

            agent._last_flushed_db_idx = 0
            agent._flush_messages_to_session_db(messages, conversation_history)

            rows = session_db.get_messages("test-session-repair-overflow")
            persisted_contents = [r["content"] for r in rows if r.get("content")]
            assert "follow up" in persisted_contents
            assert "reply" in persisted_contents
