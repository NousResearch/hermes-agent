"""Regression: the gateway's ``session_meta`` row must not count as transcript growth.

A gateway session's first turn persists a ``session_meta`` bookkeeping row
(``gateway/run_turn_persistence.py``), and the gateway drops ``session_meta`` /
``system`` rows before it hands the history to the agent (``gateway/run.py``).
Rotation compaction re-reads the durable parent in
``_adopt_grown_durable_parent`` and compares lengths only. Read unfiltered, the
durable copy is one row longer than the live transcript even when nothing was
written, so compaction "adopts" a snapshot with the bookkeeping row spliced in
mid-history. Every later message then sits one index off, and a prepared
summary planned on the live transcript (``compression.prepare_ahead``) no
longer fits: it is cancelled and the summary is generated again inline.
"""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import MagicMock, patch

from hermes_state import SessionDB


def _build_agent_with_db(db: SessionDB, session_id: str):
    """AIAgent on ``db`` in rotation mode with a stub compressor (no LLM call)."""
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
    compressor.compress.side_effect = lambda *_a, **_kw: [
        {"role": "user", "content": "[CONTEXT COMPACTION] summary"},
        {"role": "user", "content": "tail"},
    ]
    compressor.compression_count = 1
    compressor.last_prompt_tokens = 0
    compressor.last_completion_tokens = 0
    compressor._last_summary_error = None
    compressor._last_compress_aborted = False
    compressor._last_aux_model_failure_model = None
    compressor._last_aux_model_failure_error = None
    agent.context_compressor = compressor
    agent._compression_feasibility_checked = True
    agent.compression_in_place = False
    return agent


def _seed_gateway_session(db: SessionDB, session_id: str) -> list:
    """A first gateway turn (user, session_meta, assistant) plus a second exchange, and the
    history the gateway hands the agent: the same rows without ``session_meta``."""
    db.create_session(session_id, source="discord")
    db.append_message(session_id, "user", "first question")
    db.append_message(session_id, "session_meta", None)
    db.append_message(session_id, "assistant", "first answer")
    db.append_message(session_id, "user", "second question")
    db.append_message(session_id, "assistant", "second answer")
    rows = db.get_messages_as_conversation(session_id)
    assert [r["role"] for r in rows].count("session_meta") == 1
    return [r for r in rows if r.get("role") not in ("session_meta", "system")]


def test_session_meta_row_is_not_growth(tmp_path: Path) -> None:
    from agent.conversation_compression_adoption import _adopt_grown_durable_parent

    db = SessionDB(db_path=tmp_path / "state.db")
    live = _seed_gateway_session(db, "GW_PARENT")
    agent = _build_agent_with_db(db, "GW_PARENT")
    lease = MagicMock(db=db, sid="GW_PARENT", watermark=None)

    assert _adopt_grown_durable_parent(agent, lease, live) is None


def test_real_growth_adopts_without_session_meta(tmp_path: Path) -> None:
    from agent.conversation_compression_adoption import _adopt_grown_durable_parent

    db = SessionDB(db_path=tmp_path / "state.db")
    live = _seed_gateway_session(db, "GW_GROWN")
    agent = _build_agent_with_db(db, "GW_GROWN")
    db.append_message("GW_GROWN", "assistant", "concurrent row")
    lease = MagicMock(db=db, sid="GW_GROWN", watermark=0)

    adopted = _adopt_grown_durable_parent(agent, lease, live)

    assert adopted is not None
    assert [m.get("content") for m in adopted] == [
        "first question", "first answer", "second question", "second answer", "concurrent row",
    ]
    assert lease.watermark > 0


def test_compression_input_keeps_live_indices(tmp_path: Path) -> None:
    """End to end: what the summariser receives lines up index for index with the live transcript,
    which is what a prepared summary's window was planned on."""
    db = SessionDB(db_path=tmp_path / "state.db")
    live = _seed_gateway_session(db, "GW_E2E")
    agent = _build_agent_with_db(db, "GW_E2E")
    seen: list = []
    agent.context_compressor.compress.side_effect = lambda msgs, **_kw: (seen.append(list(msgs)) or [
        {"role": "user", "content": "[CONTEXT COMPACTION] summary"},
        {"role": "user", "content": "tail"},
    ])

    agent._compress_context(live, "sys", approx_tokens=120_000)

    assert len(seen) == 1
    assert [m.get("role") for m in seen[0]] == [m.get("role") for m in live]
