"""Regression: a boundary-aware partial compress must not adopt the full parent.

``_compress_context`` re-reads the durable parent under the per-session compression lock and, when
that snapshot is longer than the transcript it was handed, ADOPTS it (the "grew before lease" path
in ``conversation_compression.py``). That is right for a full ``/compress``: a longer durable
parent means a concurrent writer committed a turn, and adopting keeps that turn in the summary
instead of aborting forever.

``/compress here N`` is different. The caller deliberately hands over only the pre-boundary HEAD
and keeps the last ``N`` exchanges verbatim, re-appending them after compression
(``rejoin_compressed_head_and_tail``). The durable parent is *necessarily* longer than that head —
it still contains the tail — so the length comparison fires by construction: the whole transcript
gets summarized, and the rejoin then appends the tail a second time. The user asks to keep the tail
verbatim and gets it duplicated inside a summary of everything.

The check cannot be fixed by inspecting the lists: "intentional head subset" and "a writer
committed a row" are the same shape, because in-memory edits of past turns are legal. The caller's
intent therefore has to be explicit (``partial_head=True``), and the guard must stand down.
"""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import MagicMock, patch

from hermes_state import SessionDB


def _build_agent_with_db(db: SessionDB, session_id: str):
    """Build an AIAgent wired to ``db`` and pinned to ``session_id``.

    Mirrors the helper in ``test_compression_adoption_preserves_live_tail.py``: stub the compressor
    so it returns deterministic output without an LLM call, and pin ``compression_in_place=False``
    so the legacy rotation path (which owns "grew before lease" adoption) is exercised.
    """
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
    compressor.compress.return_value = [
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
    # One-time compression-model feasibility probe would resolve a REAL auxiliary provider;
    # mark it done like test_compression_concurrent_fork.
    agent._compression_feasibility_checked = True
    agent.compression_in_place = False
    return agent


def _contents(rows) -> list:
    return [r.get("content") for r in rows]


def _seed_six_message_session(db: SessionDB, session_id: str) -> list:
    """Three user/assistant exchanges, fully durable. Returns the loaded transcript."""
    db.create_session(session_id, source="desktop")
    for role, content in (
        ("user", "q1"),
        ("assistant", "a1"),
        ("user", "q2"),
        ("assistant", "a2"),
        ("user", "q3"),
        ("assistant", "a3"),
    ):
        db.append_message(session_id, role, content)
    return db.get_messages_as_conversation(session_id)


def test_partial_head_reaches_summarizer_unadopted(tmp_path: Path) -> None:
    """``/compress here 2`` must summarize the head it was handed, not the full transcript.

    The durable parent holds all six rows; the caller passes only the first two and keeps the last
    two exchanges verbatim. Because the durable parent is longer by construction, the adoption
    guard used to replace the head with the full six-row parent, so the summarizer saw — and
    summarized — the tail the user asked to keep untouched.
    """
    db = SessionDB(db_path=tmp_path / "state.db")
    session_id = "PARTIAL_HEAD_NOT_ADOPTED"
    full = _seed_six_message_session(db, session_id)
    head, tail = full[:2], full[2:]

    agent = _build_agent_with_db(db, session_id)

    seen: list = []

    def _recording_compress(first_arg, **_kw):
        seen.append(list(first_arg))
        return [
            {"role": "user", "content": "[CONTEXT COMPACTION] summary"},
            {"role": "user", "content": "tail"},
        ]

    agent.context_compressor.compress.side_effect = _recording_compress

    agent._compress_context(
        head, "sys", approx_tokens=120_000, force=True, partial_head=True
    )

    assert len(seen) == 1, f"expected exactly one summarizer call, got {len(seen)}"
    assert _contents(seen[0]) == ["q1", "a1"], (
        "A partial (/compress here N) call must summarize only the head it was handed. "
        "The durable parent is longer by construction — it still holds the kept tail — so the "
        "length-based adoption guard must stand down for an explicit head subset (#71991). "
        f"Summarizer input contents: {_contents(seen[0])!r}"
    )


def test_partial_head_rejoin_does_not_duplicate_tail(tmp_path: Path) -> None:
    """End-to-end: the kept tail must appear exactly once after the caller's rejoin.

    This is the user-visible failure. Once the head is replaced by the full transcript, the
    summary already covers the tail, and ``rejoin_compressed_head_and_tail`` appends it again.
    """
    from hermes_cli.partial_compress import rejoin_compressed_head_and_tail

    db = SessionDB(db_path=tmp_path / "state.db")
    session_id = "PARTIAL_HEAD_REJOIN"
    full = _seed_six_message_session(db, session_id)
    head, tail = full[:2], full[2:]

    agent = _build_agent_with_db(db, session_id)
    compressed, _ = agent._compress_context(
        head, "sys", approx_tokens=120_000, force=True, partial_head=True
    )
    rejoined = rejoin_compressed_head_and_tail(compressed, tail)

    assert _contents(rejoined).count("q3") == 1, (
        "The verbatim tail must be re-appended exactly once; summarizing it via durable-parent "
        f"adoption duplicates it (#71991). Got {_contents(rejoined)!r}"
    )
