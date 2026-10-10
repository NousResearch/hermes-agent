"""Regression test: restored rows must not be re-INSERTed by the flush.

Production defect (estate state.db receipts, 2026-10-07): a message dict rebuilt
from durable rows — crash/failed-turn history restore, resume, repaired sequence —
carries its durable ``message_uid`` (identity restoration) but neither the
``_DB_PERSISTED_MARKER`` nor ``_row_id``, so the append-only flush treated it as a
new message and INSERTed a second copy of an already-durable logical message.

Receipts in production state.db: 24k+ re-inserted shadow rows across 47 sessions
(2026-10-01 .. 2026-10-07), each preserving the original uid while landing at a
later timestamp with display metadata lost — e.g. session 20261007_121708_c6c63054
held uid 8f651719 as a durable row at 12:37 and again as a fresh insert at 15:18,
four hours later, from the same logical assistant message.

The fix consults the durable uid set in ``resolve_and_repair_transcript_batch``
(the append's write transaction): a candidate without a row address or CAS
snapshot whose uid already names an ACTIVE durable row is stamped
``_DB_PERSISTED_MARKER`` and skipped — append-only state, no rewrites. Dicts
that still carry ``DB_ROW_SNAPSHOT`` are excluded (the CAS path rewrites their
row in place — a live edit must reach the row, never be skipped), and the
probe runs inside the ``BEGIN IMMEDIATE`` so the active set cannot change
between the read and the INSERT.

This file reproduces the exact signature end-to-end: restore the history from the
DB (fresh dicts, uids preserved, no markers), then flush again — the old code
doubled every row; the fixed code inserts nothing.
"""

from __future__ import annotations

import os
from unittest.mock import patch

import pytest

from agent.context_compressor import _DB_PERSISTED_MARKER


def _make_agent_with_db(tmp_path):
    with patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"}):
        from run_agent import AIAgent
        from hermes_state import SessionDB

        db = SessionDB(tmp_path / "state.db")
        agent = AIAgent(
            api_key="test-key",
            base_url="https://openrouter.ai/api/v1",
            model="test/model",
            quiet_mode=True,
            session_db=db,
            session_id="dup-reinsert-001",
            skip_context_files=True,
            skip_memory=True,
            cwd=str(tmp_path),
        )
    agent._ensure_db_session()
    return agent, db


def _row_count(db, session_id):
    return db._read_one(
        "SELECT COUNT(*) FROM messages WHERE session_id = ?", (session_id,)
    )[0]


def test_restored_history_rows_are_not_reinserted(tmp_path):
    """The production signature: flush a turn, rebuild the dicts from the durable
    rows (restore/resume), flush again — zero new rows may land."""
    agent, db = _make_agent_with_db(tmp_path)
    sid = agent.session_id

    # Turn one: a user message and an assistant reply, flushed durably.
    turn_one = [
        {"role": "user", "content": "first turn", "timestamp": 1000.0},
        {"role": "assistant", "content": "first reply", "timestamp": 1001.0},
    ]
    agent._flush_messages_to_session_db(turn_one)
    n_after_turn_one = _row_count(db, sid)
    assert n_after_turn_one == 2

    # Restore: the gateway reloads the history from the DB after a failed turn /
    # crash, then the dicts cross a boundary that strips the underscore-prefixed
    # persistence marker (_rows_to_conversation stamps "born durable", but
    # transports strip `_db_persisted` before the wire — the marker is in-memory
    # state, the uid is durable identity that survives EVERY projection).
    # This is the exact production shape: uid preserved, marker gone, no _row_id.
    restored = db.get_messages_as_conversation(sid)
    assert len(restored) >= 2
    for m in restored:
        m.pop(_DB_PERSISTED_MARKER, None)  # the wire strip
    assert all(not m.get(_DB_PERSISTED_MARKER) for m in restored)
    restored_uids = [m.get("message_uid") for m in restored]
    assert all(isinstance(u, str) and u for u in restored_uids), (
        "restore must preserve durable uids for this test to reproduce the defect"
    )

    # Re-flush the restored history exactly as the recovered process does —
    # WITHOUT conversation_history (the identity-seed arg): a wire-crossing
    # handoff carries no history reference.
    agent._flush_messages_to_session_db(restored)

    n_after_restore = _row_count(db, sid)
    assert n_after_restore == n_after_turn_one, (
        "flush after restore re-inserted durable rows: "
        f"{n_after_turn_one} -> {n_after_restore}"
    )


def test_new_messages_after_restore_still_persist(tmp_path):
    """The guard must not over-match: after a restore, a genuinely new turn
    (fresh uids) still lands durably."""
    agent, db = _make_agent_with_db(tmp_path)
    sid = agent.session_id

    turn_one = [{"role": "user", "content": "hello", "timestamp": 2000.0}]
    agent._flush_messages_to_session_db(turn_one)
    restored = db.get_messages_as_conversation(sid)

    new_turn = [
        *restored,
        {"role": "user", "content": "second turn", "timestamp": 3000.0},
        {"role": "assistant", "content": "second reply", "timestamp": 3001.0},
    ]
    agent._flush_messages_to_session_db(new_turn, restored)

    rows = db._read_all(
        "SELECT role, content FROM messages WHERE session_id = ? ORDER BY id", (sid,)
    )
    contents = [r["content"] for r in rows]
    assert "second turn" in contents and "second reply" in contents, (
        "new post-restore messages were dropped by the dedupe guard"
    )
    assert contents.count("hello") == 1, "restored row was re-inserted"
    assert len(contents) == 3, f"expected exactly 3 rows, got {len(contents)}: {contents}"


def test_restored_snapshot_rewrite_still_lands(tmp_path):
    """The reviewer's blocker (134799 review, 2026-10-07): a restored dict that
    still carries its ``_db_row_snapshot`` is the CAS repair path's client —
    ``_active_logical_message_row`` finds the active row by (session_id, role,
    message_uid) and rewrites it in place. The durable-uid re-insert guard must
    never stamp-and-skip such a dict: a live edit on it would be silently lost.
    """
    agent, db = _make_agent_with_db(tmp_path)
    sid = agent.session_id

    turn_one = [
        {"role": "user", "content": "q", "timestamp": 5000.0},
        {"role": "assistant", "content": "old answer", "timestamp": 5001.0},
    ]
    agent._flush_messages_to_session_db(turn_one)
    assert _row_count(db, sid) == 2

    # restore with repair_alternation=True: fresh dicts stamped snapshot + marker
    restored = db.get_messages_as_conversation(sid, repair_alternation=True)
    assert len(restored) == 2
    assert all(isinstance(m.get("_db_row_snapshot"), str) for m in restored), (
        "repair_alternation=True read-back must stamp the CAS digest for this test to pin the rewrite path"
    )
    # strip ONLY the persistence marker (underscore-prefixed; the wire strip) —
    # the snapshot and uid stay: this dict is still row-addressable by CAS
    for m in restored:
        m.pop(_DB_PERSISTED_MARKER, None)

    # the live edit that must reach the durable row
    restored[-1]["content"] = "new answer"
    agent._flush_messages_to_session_db(restored)

    final = db.get_messages_as_conversation(sid, repair_alternation=True)
    contents = [m.get("content") for m in final]
    assert contents == ["q", "new answer"], (
        "live edit on a snapshot-carrying restored dict was silently dropped: "
        f"final contents {contents}"
    )
    assert _row_count(db, sid) == 2, "the rewrite must not add rows"