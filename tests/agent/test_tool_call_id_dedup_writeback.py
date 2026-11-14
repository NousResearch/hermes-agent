"""Regression: duplicate tool results must not multiply durable transcript rows."""

import sqlite3
from unittest.mock import patch

from hermes_state import SessionDB


def _tool_rows(db: SessionDB, session_id: str) -> list[dict]:
    return [
        m
        for m in db.get_messages_as_conversation(session_id, include_inactive=True)
        if m.get("role") == "tool"
    ]


def _count_tool_id(db: SessionDB, session_id: str, call_id: str) -> int:
    return sum(1 for m in _tool_rows(db, session_id) if m.get("tool_call_id") == call_id)


def test_first_write_inserts_one_row(tmp_path):
    db = SessionDB(db_path=tmp_path / "state.db")
    sid = "dedup-a1"
    db.create_session(sid, source="cli")
    db.append_messages_batch(
        sid,
        [{"role": "tool", "tool_call_id": "call-1", "content": "payload-a"}],
    )
    assert _count_tool_id(db, sid, "call-1") == 1


def test_same_id_same_content_is_idempotent(tmp_path):
    db = SessionDB(db_path=tmp_path / "state.db")
    sid = "dedup-a2"
    db.create_session(sid, source="cli")
    row = {"role": "tool", "tool_call_id": "call-1", "content": "payload-a"}
    db.append_messages_batch(sid, [row])
    db.append_messages_batch(sid, [dict(row)])
    assert _count_tool_id(db, sid, "call-1") == 1


def test_same_id_different_content_supersedes_in_place(tmp_path):
    db = SessionDB(db_path=tmp_path / "state.db")
    sid = "dedup-a3"
    db.create_session(sid, source="cli")
    db.append_messages_batch(
        sid,
        [{"role": "tool", "tool_call_id": "call-1", "content": "payload-a"}],
    )
    db.append_messages_batch(
        sid,
        [{"role": "tool", "tool_call_id": "call-1", "content": "payload-b"}],
    )
    assert _count_tool_id(db, sid, "call-1") == 1
    rows = _tool_rows(db, sid)
    assert rows[0]["content"] == "payload-b"
    assert rows[0].get("effect_disposition") == "tool_result_superseded"


def test_empty_tool_call_id_is_not_deduped(tmp_path):
    db = SessionDB(db_path=tmp_path / "state.db")
    sid = "dedup-a4"
    db.create_session(sid, source="cli")
    db.append_messages_batch(
        sid,
        [
            {"role": "tool", "tool_call_id": None, "content": "a"},
            {"role": "tool", "tool_call_id": "", "content": "b"},
        ],
    )
    assert len(_tool_rows(db, sid)) == 2


def test_cross_session_same_id_independent(tmp_path):
    db = SessionDB(db_path=tmp_path / "state.db")
    db.create_session("s-one", source="cli")
    db.create_session("s-two", source="cli")
    row = {"role": "tool", "tool_call_id": "shared", "content": "payload"}
    db.append_messages_batch("s-one", [row])
    db.append_messages_batch("s-two", [dict(row)])
    assert _count_tool_id(db, "s-one", "shared") == 1
    assert _count_tool_id(db, "s-two", "shared") == 1


def test_batch_with_duplicates_collapses(tmp_path):
    db = SessionDB(db_path=tmp_path / "state.db")
    sid = "dedup-a6"
    db.create_session(sid, source="cli")
    db.append_messages_batch(
        sid,
        [
            {"role": "tool", "tool_call_id": "call-1", "content": "same"},
            {"role": "tool", "tool_call_id": "call-1", "content": "same"},
            {"role": "tool", "tool_call_id": "call-2", "content": "other"},
        ],
    )
    assert _count_tool_id(db, sid, "call-1") == 1
    assert _count_tool_id(db, sid, "call-2") == 1


def test_legal_tool_call_pair_survives_dedup_writeback(tmp_path):
    """Compression-style rewrite must keep a valid tool_call ↔ tool result pair."""
    db = SessionDB(db_path=tmp_path / "state.db")
    sid = "dedup-pair"
    db.create_session(sid, source="cli")
    pair = [
        {
            "role": "assistant",
            "content": "calling",
            "tool_calls": [
                {
                    "id": "call-pair",
                    "type": "function",
                    "function": {"name": "terminal", "arguments": "{}"},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call-pair", "content": "ok-v1"},
    ]
    db.append_messages_batch(sid, pair)
    # Simulate compression writeback re-emitting the same pairing once more.
    db.append_messages_batch(
        sid,
        [
            pair[0],
            {"role": "tool", "tool_call_id": "call-pair", "content": "ok-v1"},
        ],
    )
    loaded = db.get_messages_as_conversation(sid)
    tool_rows = [m for m in loaded if m.get("role") == "tool"]
    assistant_rows = [m for m in loaded if m.get("role") == "assistant"]
    assert len(tool_rows) == 1
    assert tool_rows[0]["tool_call_id"] == "call-pair"
    assert tool_rows[0]["content"] == "ok-v1"
    assert any(
        isinstance(m.get("tool_calls"), list)
        and any(tc.get("id") == "call-pair" for tc in m["tool_calls"])
        for m in assistant_rows
    )


def test_falsification_without_dedup_accumulates_duplicates(tmp_path):
    """Reverse #3: disabling suppress_duplicate_tool_call_writes must fail the invariant."""
    db = SessionDB(db_path=tmp_path / "state.db")
    sid = "dedup-falsify"
    db.create_session(sid, source="cli")

    def _identity_suppress(conn, session_id, inserted_rows, **kwargs):
        return inserted_rows

    with patch(
        "agent.transcript_repair.suppress_duplicate_tool_call_writes",
        side_effect=_identity_suppress,
    ):
        # Fresh dicts each write: reusing the post-flush object carries
        # ``_db_persisted`` and would skip the second INSERT for a different reason.
        db.append_messages_batch(
            sid,
            [{"role": "tool", "tool_call_id": "call-dup", "content": "legacy"}],
        )
        db.append_messages_batch(
            sid,
            [{"role": "tool", "tool_call_id": "call-dup", "content": "legacy"}],
        )
    assert _count_tool_id(db, sid, "call-dup") == 2


def test_falsification_manual_duplicate_still_readable(tmp_path):
    """Pre-existing duplicate rows remain visible (read path not hiding history)."""
    db = SessionDB(db_path=tmp_path / "state.db")
    sid = "dedup-a7"
    db.create_session(sid, source="cli")
    db.append_messages_batch(
        sid,
        [{"role": "tool", "tool_call_id": "call-dup", "content": "legacy"}],
    )
    with sqlite3.connect(tmp_path / "state.db") as conn:
        conn.execute(
            "INSERT INTO messages (session_id, role, content, tool_call_id, active, timestamp) "
            "VALUES (?, 'tool', ?, ?, 1, ?)",
            (sid, SessionDB._encode_content("legacy"), "call-dup", 1.0),
        )
        conn.commit()
    assert _count_tool_id(db, sid, "call-dup") == 2
    loaded = db.get_messages_as_conversation(sid, include_inactive=True)
    assert sum(1 for m in loaded if m.get("tool_call_id") == "call-dup") == 2
