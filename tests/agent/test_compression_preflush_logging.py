"""Regression: compression pre-flush failure must be logged, not swallowed.

``compress_context()`` flushes the current turn to the session DB before
rotation as a best-effort step — it must never block compression on a flush
error. Historically the failure was ``except Exception: pass``: zero
diagnostic output, so a stale persisted transcript was invisible to the
operator. The fix replaces the bare pass with ``logger.warning(..., exc_info=True)``
while keeping the best-effort contract (compression proceeds).

This test pins both halves of the contract:
  1. a flush failure still lets compression continue and return normally;
  2. the failure is observable via the module logger.
"""

from __future__ import annotations

import logging
import os
import time
from unittest.mock import MagicMock, patch

from hermes_state import SessionDB


def _build_agent_with_db(db: SessionDB, session_id: str):
    """Build an AIAgent wired to ``db`` and pinned to ``session_id``."""
    with patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"}):
        from run_agent import AIAgent

        agent = AIAgent(
            api_key="test-key",
            base_url="https://openrouter.ai/api/v1",
            model="test/model",
            quiet_mode=True,
            session_db=db,
            session_id=session_id,
            skip_context_files=True,
            skip_memory=True,
        )

    compressor = MagicMock()

    def _compress(*_a, **_kw):
        time.sleep(0.01)
        return [
            {"role": "user", "content": "[CONTEXT COMPACTION] summary"},
            {"role": "assistant", "content": "compacted"},
        ]

    compressor.compress = _compress
    compressor.compression_count = 1
    compressor.last_prompt_tokens = 0
    compressor.last_completion_tokens = 0
    compressor._last_summary_error = None
    compressor._last_compress_aborted = False
    compressor._last_aux_model_failure_model = None
    compressor._last_aux_model_failure_error = None
    agent.context_compressor = compressor
    agent.compression_in_place = False
    return agent


def test_preflush_failure_logs_warning_and_compression_continues(tmp_path, caplog):
    """A flush failure must be logged at WARNING and must not abort compression."""
    db = SessionDB(db_path=tmp_path / "state.db")
    session_id = "sess_test_preflush_warn"
    agent = _build_agent_with_db(db, session_id)

    # Seed one durable turn so the rotation branch has history to flush, then
    # anchor the current-turn index like turn_context does before preflight.
    db.create_session(session_id, source="desktop")
    db.append_message(session_id, "user", "persisted question")
    db.append_message(session_id, "assistant", "persisted answer")
    messages = [
        *db.get_messages_as_conversation(session_id),
        {"role": "user", "content": "new turn"},
    ]
    agent._persist_user_message_idx = len(messages) - 1

    # Simulate the flush blowing up right where the preflight rotation flush runs.
    def _boom(*_a, **_kw):
        raise RuntimeError("simulated flush failure")

    with (
        patch.object(agent, "_flush_messages_to_session_db", side_effect=_boom),
        caplog.at_level(logging.WARNING, logger="agent.conversation_compression"),
    ):
        agent._compress_context(messages, "sys", approx_tokens=120_000)

    # Compression still completed (best-effort contract preserved).
    assert any(
        "Compression pre-flush" in r.message for r in caplog.records
    ), "flush failure must be logged, not silently swallowed"
