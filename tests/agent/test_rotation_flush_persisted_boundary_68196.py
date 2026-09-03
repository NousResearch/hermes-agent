"""Regression (#68196): rotating preflight compression must not re-append the
already-persisted transcript to the parent session on cold resume.

On the first turn after a cold Desktop resume, the stored rows are handed to
``run_conversation()`` as ``conversation_history`` and live in the message list
as plain dicts that have NOT yet been stamped with ``_DB_PERSISTED_MARKER`` —
the normal turn flush (which stamps them) runs *after* preflight compression.

The legacy rotation branch in ``agent/conversation_compression.py`` flushes the
current turn to the OLD session before ending it (#47202). It used to call
``agent._flush_messages_to_session_db(messages)`` with no history boundary, so
``_flush_messages_to_session_db`` saw an empty ``history_ids`` set and treated
every restored row as new — durably appending the whole transcript to the
parent a second time. The fix passes ``messages[:_persist_user_message_idx]``
(the already-durable prefix ``turn_context`` anchors before preflight runs) as
``conversation_history`` so only the current turn's new messages are written.

Without the fix the parent grows to 5 rows (the two originals + a duplicate of
both + the new turn). With it the parent holds exactly the two originals plus
the single new turn.
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

from hermes_state import SessionDB


def _build_agent_with_db(db: SessionDB, session_id: str):
    """Build an AIAgent wired to ``db`` and pinned to ``session_id``.

    Mirrors the helper in ``test_compression_concurrent_fork.py``: stub the
    compressor so it returns deterministic output without an LLM call, and pin
    ``compression_in_place=False`` so the legacy rotation path is exercised
    regardless of the global default.
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

    def _compress(*_a, **_kw):
        time.sleep(0.01)
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
    agent.context_compressor = compressor
    agent.compression_in_place = False
    return agent


def _contents(rows):
    return [r.get("content") for r in rows]


def test_rotation_flush_does_not_duplicate_persisted_prefix(tmp_path: Path) -> None:
    """Cold-resume + rotating preflight compression keeps the parent transcript
    at (persisted prefix + one new turn) — no second copy of the durable rows."""
    db = SessionDB(db_path=tmp_path / "state.db")

    parent_sid = "COLD_RESUME_PARENT"
    db.create_session(parent_sid, source="desktop")

    # Two durable rows already in the parent.
    db.append_message(parent_sid, "user", "persisted question")
    db.append_message(parent_sid, "assistant", "persisted answer")

    # Cold resume: the stored rows come back as plain dicts, unstamped, and the
    # live turn appends one new user message on top.
    loaded = db.get_messages_as_conversation(parent_sid)
    assert _contents(loaded) == ["persisted question", "persisted answer"]
    messages = [*loaded, {"role": "user", "content": "new turn"}]

    agent = _build_agent_with_db(db, parent_sid)
    # turn_context anchors this at the current-turn user message before preflight
    # compression runs; emulate that anchor.
    agent._persist_user_message_idx = len(messages) - 1

    agent._compress_context(messages, "sys", approx_tokens=120_000)

    # The flush at the rotation boundary lands on the OLD (parent) session,
    # which is then ended. Read it back verbatim (include_inactive to be robust
    # to the end_session bookkeeping).
    parent_rows = db.get_messages_as_conversation(parent_sid, include_inactive=True)
    contents = _contents(parent_rows)

    assert contents.count("persisted question") == 1, (
        "Rotation flush re-appended the already-persisted prefix to the parent "
        f"(#68196). Parent transcript is {contents!r}; expected the two durable "
        "rows plus only the new turn."
    )
    assert contents.count("persisted answer") == 1
    assert contents == ["persisted question", "persisted answer", "new turn"]


def test_rotation_memory_sync_uses_compacted_transcript_once(tmp_path: Path) -> None:
    """Legacy rotation's ``None`` DB baseline must not make provider memory
    append the compacted transcript to its pre-compression snapshot."""
    from agent.conversation_compression import (
        conversation_history_after_compression,
    )

    db = SessionDB(db_path=tmp_path / "state.db")
    parent_sid = "MEMORY_SYNC_PARENT"
    db.create_session(parent_sid, source="desktop")

    messages = [
        {"role": "user", "content": "obsolete question"},
        {"role": "assistant", "content": "obsolete answer"},
        {"role": "user", "content": "new turn"},
    ]
    agent = _build_agent_with_db(db, parent_sid)
    agent._persist_user_message_idx = len(messages) - 1

    compacted, _ = agent._compress_context(
        messages,
        "sys",
        approx_tokens=120_000,
    )
    history = conversation_history_after_compression(agent, compacted, messages)

    # Rotation still needs a cleared persistence baseline for its fresh child,
    # but provider reconciliation has a separate, authoritative transcript.
    assert history is None
    assert agent._external_memory_transcript_authoritative is True

    manager = MagicMock()
    agent._memory_manager = manager
    # Model the provider snapshot following the memory manager's child-session
    # switch. If the compacted list were treated like a stateless exchange, the
    # obsolete parent messages would be prepended here.
    agent._external_memory_messages = [dict(message) for message in messages]
    agent._external_memory_session_id = agent.session_id

    agent._sync_external_memory_for_turn(
        original_user_message="new turn",
        final_response="new answer",
        interrupted=False,
        messages=compacted,
        messages_are_authoritative=(
            agent._external_memory_transcript_authoritative
        ),
    )

    synced = manager.sync_all.call_args.kwargs["messages"]
    assert [message["content"] for message in synced] == [
        "[CONTEXT COMPACTION] summary",
        "tail",
    ]
    assert all("obsolete" not in message["content"] for message in synced)
