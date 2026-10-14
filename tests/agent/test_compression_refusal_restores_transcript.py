"""Regression: a refused compression candidate hands back the original transcript.

Context engines may mutate the message list they receive (the documented legacy
contract), and that mutation must only become visible if the pass commits. Callers
that pass their own commit fence (gateway session hygiene) or disable the
compression timeout hand the live list to the engine without the facade's deep
snapshot, so every refusal in ``compress_context`` has to put the pre-compression
snapshot back before returning. The compressor-abort, empty-transcript and
would-grow refusals returned the mutated list instead.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from unittest.mock import patch

import pytest

from agent.conversation_compression import CompressionCommitFence
from hermes_state import SessionDB


def _build_agent(db: SessionDB, session_id: str):
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
    # Skip the one-time aux-model feasibility probe (may hit the network).
    agent._compression_feasibility_checked = True
    return agent


def _aborting_engine(compressor):
    def _compress(msgs, **_kw):
        msgs.clear()
        compressor._last_compress_aborted = True
        compressor._last_summary_error = "summary provider unavailable"
        return msgs

    return _compress


def _empty_transcript_engine(_compressor):
    def _compress(msgs, **_kw):
        msgs.clear()
        return []

    return _compress


def _growing_engine(compressor):
    def _compress(msgs, **_kw):
        # A real compressor records the candidate's effect before the commit checks see it.
        compressor._active_compression_telemetry = compressor._last_compression_telemetry = (
            compressor._begin_compression_telemetry(current_tokens=120_000)
        )
        compressor._active_compression_telemetry.update(messages_after=2, tokens_after=9, tokens_reclaimed=99)
        msgs.pop()
        return list(msgs) + [{"role": "assistant", "content": "GROWN " * 20_000}]

    return _compress


def _growing_in_place_engine(_compressor):
    def _compress(msgs, **_kw):
        msgs[:] = [{"role": "assistant", "content": "GROWN " * 20_000}]
        return msgs

    return _compress


def _raising_engine(_compressor):
    def _compress(msgs, **_kw):
        msgs.clear()
        raise RuntimeError("context engine failed")

    return _compress


@pytest.mark.parametrize(
    "make_engine",
    [_aborting_engine, _empty_transcript_engine, _growing_engine, _growing_in_place_engine],
    ids=["compressor_aborted", "empty_transcript", "would_grow", "would_grow_in_place"],
)
def test_refused_candidate_from_mutating_engine_returns_original_transcript(
    tmp_path: Path, make_engine, caplog
) -> None:
    db = SessionDB(db_path=tmp_path / "state.db")
    session_id = "REFUSED_CANDIDATE"
    db.create_session(session_id, source="test")
    agent = _build_agent(db, session_id)
    compressor = agent.context_compressor

    messages = [{"role": "user", "content": f"m{i} " + "x" * 200} for i in range(21)]
    original = [dict(m) for m in messages]

    try:
        with patch.object(type(compressor), "compress", side_effect=make_engine(compressor)), patch.object(
            agent, "commit_memory_session", wraps=agent.commit_memory_session
        ) as commit_memory_session, patch.object(
            db, "archive_and_compact", wraps=db.archive_and_compact
        ) as archive_and_compact:
            # A caller-owned fence selects the direct path that gateway session hygiene uses.
            with caplog.at_level(logging.INFO, logger="agent.conversation_compression"):
                returned, _sp = agent._compress_context(
                    messages, "sys", approx_tokens=120_000, commit_fence=CompressionCommitFence()
                )

        commit_memory_session.assert_not_called()
        archive_and_compact.assert_not_called()
        assert returned == original
        assert messages == original
        assert agent.session_id == session_id
        assert db._conn.execute(
            "SELECT COUNT(*) FROM sessions WHERE parent_session_id = ?", (session_id,)
        ).fetchone()[0] == 0
        assert db.get_session(session_id)["end_reason"] is None
        assert db.get_compression_lock_holder(session_id) is None
        # The record must not claim an effect the refused candidate never had on the transcript.
        prefix = "context compression attempt telemetry: "
        [record] = [json.loads(r.getMessage()[len(prefix):]) for r in caplog.records if r.getMessage().startswith(prefix)]
        assert record["commit_status"] == "aborted"
        assert [record.get(k) for k in ("messages_after", "tokens_after", "tokens_reclaimed")] == [None] * 3
    finally:
        agent.close()


def test_raising_mutating_engine_restores_original_transcript(tmp_path: Path) -> None:
    db = SessionDB(db_path=tmp_path / "state.db")
    session_id = "RAISING_ENGINE"
    db.create_session(session_id, source="test")
    agent = _build_agent(db, session_id)
    compressor = agent.context_compressor
    messages = [{"role": "user", "content": f"m{i} " + "x" * 200} for i in range(21)]
    original = [dict(m) for m in messages]

    try:
        with patch.object(type(compressor), "compress", side_effect=_raising_engine(compressor)):
            with pytest.raises(RuntimeError, match="context engine failed"):
                agent._compress_context(
                    messages, "sys", approx_tokens=120_000, commit_fence=CompressionCommitFence()
                )
        assert messages == original
        assert agent.session_id == session_id
        assert db.get_session(session_id)["end_reason"] is None
        assert db.get_compression_lock_holder(session_id) is None
    finally:
        agent.close()
