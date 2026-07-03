"""Regression tests for gateway-origin metadata across compression session rotation.

Room-scoped recall depends on compression-created child sessions preserving the
parent gateway origin metadata (chat/thread identity). When rotation creates a
new child row without `chat_id` / `chat_type` / `thread_id`, scoped recall can
silently lose access to the room's older lineage after a compression boundary.

This test intentionally exercises the real rotation path with
`compression_in_place = False` to pin the fallback behavior even though
in-place compaction is now the default.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from unittest.mock import patch


def _make_agent(session_db, session_id: str):
    with patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"}):
        from run_agent import AIAgent

        agent = AIAgent(
            api_key="test-key",
            base_url="https://openrouter.ai/api/v1",
            model="test/model",
            quiet_mode=True,
            session_db=session_db,
            session_id=session_id,
            skip_context_files=True,
            skip_memory=True,
        )
    agent.compression_in_place = False
    return agent


def _seed_gateway_parent(db, session_id: str) -> None:
    db.create_session(
        session_id=session_id,
        source="telegram",
        user_id="user-1",
        session_key="telegram:chat-123:user-1:thread-456",
        chat_id="chat-123",
        chat_type="group",
        thread_id="thread-456",
        model="test/model",
    )


class TestCompressionRotationPreservesGatewayOrigin:
    def test_compression_child_inherits_gateway_chat_thread_identity(self):
        """A compression-created child row must preserve the parent's gateway
        origin metadata.

        Current room-scoped recall designs need every session in the lineage to
        carry the same chat/thread identity. If the rotation child loses these
        columns, default room-scoped `session_search` can no longer see the
        pre-compression history after the boundary.
        """
        from agent.conversation_compression import compress_context
        from hermes_state import SessionDB

        with tempfile.TemporaryDirectory() as tmp:
            db = SessionDB(db_path=Path(tmp) / "t.db")
            parent_sid = "20260704_000000_parent"
            _seed_gateway_parent(db, parent_sid)

            agent = _make_agent(db, parent_sid)
            agent._ensure_db_session()

            messages = [
                {"role": "user" if i % 2 == 0 else "assistant", "content": f"message {i}"}
                for i in range(12)
            ]

            with patch("agent.context_compressor.call_llm", side_effect=RuntimeError("no provider")):
                compress_context(
                    agent, messages, approx_tokens=100_000, system_message="sys"
                )

            assert agent.session_id != parent_sid
            child = db.get_session(agent.session_id)
            assert child is not None
            assert child["parent_session_id"] == parent_sid

            # Room-scoped recall specifically depends on the gateway-origin
            # columns below. Keep these assertions first so RED points at the
            # real isolation break, not just the generic source label.
            assert child["chat_id"] == "chat-123"
            assert child["chat_type"] == "group"
            assert child["thread_id"] == "thread-456"
            assert child["user_id"] == "user-1"
            assert child["session_key"] == "telegram:chat-123:user-1:thread-456"
            assert child["source"] == "telegram"
