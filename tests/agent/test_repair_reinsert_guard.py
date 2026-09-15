"""Regression for #111996 — a repaired/compaction re-materialized transcript must not be
re-INSERTed on top of rows that are still active in the same session.

A failed compaction plus the pre-request alternation repair hand the persistence layer a
re-materialized copy of the transcript (marker-swept compaction assembly, merged/superseded
repair dicts, a reload that lost identity).  The batch write path used to append it verbatim, so
one logical assistant row accumulated N physical copies with an identical ``display_identity``
(and one turn block got re-inserted wholesale), while the live window kept reloading every copy.

Invariant under test: writing a message whose logical identity already exists as an ACTIVE row of
the same session reconciles with that row instead of appending a second physical copy — while a
genuinely distinct message still lands.
"""

import tempfile
from pathlib import Path

SESSION_ID = "reinsert-111996"


def _db(tmpdir):
    from hermes_state import SessionDB

    db = SessionDB(db_path=Path(tmpdir) / "t.db")
    db.create_session(session_id=SESSION_ID, source="test")
    return db


def _transcript(base_ts=1_700_000_000.0):
    """One logical turn: user + assistant(tool_calls) + tool result. Timestamps are explicit, so a
    re-materialized copy is byte-identical (the shape that produced identical display_identity)."""
    return [
        {"role": "user", "content": "run it", "timestamp": base_ts},
        {
            "role": "assistant", "content": "", "timestamp": base_ts + 1, "finish_reason": "tool_calls",
            "tool_calls": [
                {"id": "call_real_1", "type": "function",
                 "function": {"name": "terminal", "arguments": "{}"}}
            ],
        },
        {"role": "tool", "tool_call_id": "call_real_1", "content": "output", "timestamp": base_ts + 2},
    ]


def test_rematerialized_transcript_does_not_duplicate_active_rows(tmp_path):
    """Re-appending an identical (re-materialized) copy must not create a second row."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db = _db(tmpdir)
        try:
            db.append_messages_batch(SESSION_ID, _transcript())
            before = len(db.get_messages(SESSION_ID))
            assert before == 3, f"expected the 3 logical rows, got {before}"

            db.append_messages_batch(SESSION_ID, _transcript())

            rows = db.get_messages(SESSION_ID)
            assert len(rows) == before, (
                f"re-materialized transcript was re-INSERTed: {before} -> {len(rows)} rows "
                f"(every copy is a physical duplicate of an ACTIVE row)"
            )
            assert [r["tool_call_id"] for r in rows if r["tool_call_id"]] == ["call_real_1"]
        finally:
            db.close()


def test_repeated_repair_repack_does_not_grow_the_transcript(tmp_path):
    """Repair runs before every request: repeated re-materialization must stay idempotent."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db = _db(tmpdir)
        try:
            db.append_messages_batch(SESSION_ID, _transcript())
            counts = []
            for _ in range(3):
                db.append_messages_batch(SESSION_ID, _transcript())
                counts.append(len(db.get_messages(SESSION_ID)))
            assert counts == [3, 3, 3], f"transcript grew on every repack: {counts}"
        finally:
            db.close()


def test_distinct_message_with_same_content_still_inserts(tmp_path):
    """The dedup key must stay exact: a later turn repeating the same text is a new row."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db = _db(tmpdir)
        try:
            db.append_messages_batch(SESSION_ID, [{"role": "user", "content": "again", "timestamp": 10.0}])
            db.append_messages_batch(SESSION_ID, [{"role": "user", "content": "again", "timestamp": 20.0}])

            rows = db.get_messages(SESSION_ID)
            assert len(rows) == 2, f"a genuinely distinct turn was deduped away: {len(rows)} rows"
            assert [r["content"] for r in rows] == ["again", "again"]
        finally:
            db.close()


def test_agent_flush_of_rematerialized_turn_does_not_multiply_rows(tmp_path):
    """End-to-end through the agent flush: a failed compaction leaves the durable turns ACTIVE, and
    the pre-request repair re-materializes the live transcript (compaction assembly hands back
    marker-swept copies — ``_fresh_compaction_message_copy``). Flushing that copy used to append a
    whole second generation of the session; it must reconcile instead."""
    import os
    from unittest.mock import patch

    from hermes_state import SessionDB

    with tempfile.TemporaryDirectory() as tmpdir:
        db = SessionDB(db_path=Path(tmpdir) / "t.db")
        try:
            with patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"}):
                from run_agent import AIAgent

                agent = AIAgent(
                    api_key="test-key", base_url="https://openrouter.ai/api/v1", model="test/model",
                    quiet_mode=True, session_db=db, session_id=SESSION_ID,
                    skip_context_files=True, skip_memory=True,
                )
            agent._ensure_db_session()

            messages = _transcript()
            agent._flush_messages_to_session_db(messages, None)
            durable = len(db.get_messages(SESSION_ID))
            assert durable == 3

            # Marker-swept re-materialization (exactly what compaction assembly produces) plus the
            # repair pass that runs before the next request.
            from agent.context_compressor import _fresh_compaction_message_copy
            from agent.agent_runtime_helpers import repair_message_sequence_with_cursor

            repacked = [_fresh_compaction_message_copy(m) for m in messages]
            repair_message_sequence_with_cursor(agent, repacked)
            agent._flush_messages_to_session_db(repacked, None)
            agent._flush_messages_to_session_db(repacked, None)

            rows = db.get_messages(SESSION_ID)
            assert len(rows) == durable, (
                f"the failed-compaction re-materialization re-appended the transcript: "
                f"{durable} -> {len(rows)} rows"
            )
            assert [r["tool_call_id"] for r in rows if r["tool_call_id"]] == ["call_real_1"]
        finally:
            db.close()


def test_same_batch_copies_do_not_self_match(tmp_path):
    """Branch-seed shape (#112044 P1): two no-ID copies sharing a display_identity in ONE batch
    to a fresh session are distinct rows — the guard must only match rows that pre-date the batch."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db = _db(tmpdir)
        try:
            n = db.append_messages_batch(SESSION_ID, [
                {"role": "user", "content": "ok", "timestamp": 1_700_000_000.0},
                {"role": "user", "content": "ok", "timestamp": 1_700_000_000.0},
            ])
            assert n == 2, f"second copy was reconciled with the first in the same batch: {n}"
            assert len(db.get_messages(SESSION_ID)) == 2
        finally:
            db.close()


def test_repair_merge_reconciles_predecessors_atomically(tmp_path):
    """Actual alternation repair (#112044 P1): two adjacent assistant rows (text + tool_calls)
    fuse with a NONZERO repair count, and the flush archives both predecessors in the same txn —
    the durable transcript holds the fused row, not 3 -> 4."""
    import os
    from unittest.mock import patch

    from hermes_state import SessionDB

    with tempfile.TemporaryDirectory() as tmpdir:
        db = SessionDB(db_path=Path(tmpdir) / "t.db")
        try:
            with patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"}):
                from run_agent import AIAgent

                agent = AIAgent(
                    api_key="test-key", base_url="https://openrouter.ai/api/v1", model="test/model",
                    quiet_mode=True, session_db=db, session_id=SESSION_ID,
                    skip_context_files=True, skip_memory=True,
                )
            agent._ensure_db_session()

            db.append_messages_batch(SESSION_ID, [
                {"role": "user", "content": "run it", "timestamp": 1_700_000_000.0},
                {"role": "assistant", "content": "thinking aloud", "timestamp": 1_700_000_001.0},
                {"role": "assistant", "content": "", "timestamp": 1_700_000_002.0,
                 "finish_reason": "tool_calls",
                 "tool_calls": [{"id": "call_merge_1", "type": "function",
                                "function": {"name": "terminal", "arguments": "{}"}}]},
                {"role": "tool", "tool_call_id": "call_merge_1", "content": "output",
                 "timestamp": 1_700_000_003.0},
            ])
            assert len(db.get_messages(SESSION_ID)) == 4

            from agent.agent_runtime_helpers import repair_message_sequence_with_cursor

            live = db.get_messages_as_conversation(SESSION_ID, include_row_ids=True)
            repairs = repair_message_sequence_with_cursor(agent, live)
            assert repairs > 0, "this regression must exercise the _merge_assistant_into path"
            assert [m["role"] for m in live] == ["user", "assistant", "tool"]

            agent._flush_messages_to_session_db(live, None)

            rows = db.get_messages(SESSION_ID)
            assert len(rows) == 3, f"predecessors were not reconciled: {len(rows)} active rows"
            fused = [r for r in rows if r["role"] == "assistant"]
            assert len(fused) == 1
            assert "thinking aloud" in (fused[0]["content"] or "")
            assert fused[0]["tool_calls"], "fused row must carry the unioned tool_calls"
        finally:
            db.close()


def test_distinct_platform_ids_with_same_display_identity_both_insert(tmp_path):
    """Reviewer repro on #112044: same text+timestamp but differing non-null platform ids are
    distinct gateway events and must never dedup; retry of the same pid still reconciles."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db = _db(tmpdir)
        try:
            n1 = db.append_messages_batch(SESSION_ID, [
                {"role": "user", "content": "ok", "timestamp": 1_700_000_000,
                 "platform_message_id": "telegram-100"}])
            n2 = db.append_messages_batch(SESSION_ID, [
                {"role": "user", "content": "ok", "timestamp": 1_700_000_000,
                 "platform_message_id": "telegram-101"}])
            assert (n1, n2) == (1, 1), f"distinct platform events were deduped: {(n1, n2)}"
            assert len(db.get_messages(SESSION_ID)) == 2
            assert db.has_platform_message_id(SESSION_ID, "telegram-100")
            assert db.has_platform_message_id(SESSION_ID, "telegram-101")
            n3 = db.append_messages_batch(SESSION_ID, [
                {"role": "user", "content": "ok", "timestamp": 1_700_000_000,
                 "platform_message_id": "telegram-100"}])
            assert n3 == 0, "retry of the same platform id must reconcile, not insert"
            assert len(db.get_messages(SESSION_ID)) == 2
        finally:
            db.close()
