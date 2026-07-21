"""Regression: Cold Desktop resume + rotating preflight compression must NOT
duplicate already-persisted transcript rows in the parent session (#68196).

On the first turn after a cold resume, the legacy rotation path in
``_compress_context`` called::

    agent._flush_messages_to_session_db(messages)

without a ``conversation_history`` boundary.  When the restored transcript
already has durable rows in state.db, the flush treated every dict as new
and appended them all a second time, producing permanent SQLite duplication.

The fix passes the already-durable prefix (anchored by
``_persist_user_message_idx``) as ``conversation_history`` so the identity
skip path in ``_flush_messages_to_session_db`` correctly recognises them.
"""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

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

    # Stub the compressor to return deterministic output without an LLM call.
    compressor = MagicMock()

    def _compress(*_a, **_kw):
        return [
            {"role": "user", "content": "[CONTEXT COMPACTION] summary"},
            {"role": "user", "content": "tail"},
        ]

    compressor.compress.side_effect = _compress
    compressor.compression_count = 1
    compressor.last_prompt_tokens = 0
    compressor.last_completion_tokens = 0
    compressor._last_summary_error = None
    compressor._last_compress_aborted = False
    compressor._last_aux_model_failure_model = None
    compressor._last_aux_model_failure_error = None
    compressor._last_compression_made_progress = True
    compressor._last_summary_fallback_used = False
    agent.context_compressor = compressor
    # Force rotation path (in_place=False) to exercise the bug.
    agent.compression_in_place = False
    return agent


def test_cold_resume_rotation_does_not_duplicate_transcript(tmp_path: Path) -> None:
    """After a cold resume, rotating preflight compression must NOT append
    already-durable messages to the parent session a second time.

    Reproduces the minimal invariant from #68196:

        db.append_message(parent, "user", "persisted question")
        db.append_message(parent, "assistant", "persisted answer")
        loaded = db.get_messages_as_conversation(parent)
        messages = [*loaded, {"role": "user", "content": "new turn"}]
        agent._persist_user_message_idx = len(messages) - 1
        agent._compress_context(messages, "sys", approx_tokens=120_000)

    Before the fix the parent contained 5 rows (duplication).
    After the fix it contains only 3 (the two originals + new turn).
    """
    db = SessionDB(db_path=tmp_path / "state.db")

    parent_sid = "COLD_RESUME_PARENT"
    db.create_session(parent_sid, source="desktop")

    # Simulate a persisted transcript with two messages.
    db.append_message(parent_sid, "user", "persisted question")
    db.append_message(parent_sid, "assistant", "persisted answer")

    # Cold resume: load the durable rows and append a new user turn.
    loaded = db.get_messages_as_conversation(parent_sid)
    assert len(loaded) == 2, f"Expected 2 loaded messages, got {len(loaded)}"

    messages = [*loaded, {"role": "user", "content": "new turn"}]

    agent = _build_agent_with_db(db, parent_sid)

    # Anchor the preflush index — turn_context sets this before preflight
    # compression, pointing past the already-durable prefix.
    agent._persist_user_message_idx = len(messages) - 1

    # Force the legacy rotation path (compression_in_place is already False).
    agent._compress_context(messages, "sys", approx_tokens=120_000)

    # Inspect the parent session — it must NOT have duplicated rows.
    parent_messages = db.get_messages_as_conversation(parent_sid)
    parent_texts = [
        m.get("content") if isinstance(m, dict) else str(m)
        for m in parent_messages
    ]

    # The parent should have exactly 3 rows: the two originals + the new turn.
    # Before the fix it would have 5 (the two originals duplicated + new turn).
    assert len(parent_messages) == 3, (
        f"Parent session has {len(parent_messages)} rows (expected 3). "
        f"Texts: {parent_texts}"
    )

    # Verify the content is correct (no duplication).
    assert parent_texts[0] == "persisted question"
    assert parent_texts[1] == "persisted answer"
    assert parent_texts[2] == "new turn"
