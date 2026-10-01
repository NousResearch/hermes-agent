"""Durable logical-message identity for unaddressed live-replay rewrites (#129065).

A live replay may intentionally omit the physical SQLite row id. If that dict is edited in
place, the write path must update the logical message it came from without guessing from its
now-mutated payload. The durable message_uid identifies the logical message and the stored-row
snapshot remains the compare-and-swap proof for the version we actually loaded.
"""

from __future__ import annotations

import pytest

from agent.context_compressor import _DB_PERSISTED_MARKER
from agent.message_metadata import DB_ROW_SNAPSHOT
from hermes_state import SessionDB


@pytest.fixture
def db(tmp_path):
    store = SessionDB(tmp_path / "state.db")
    try:
        yield store
    finally:
        store.close()


def _active_rows(store, sid):
    return [
        dict(row)
        for row in store._conn.execute(
            "SELECT id, role, content, timestamp, message_uid, tool_calls "
            "FROM messages WHERE session_id = ? AND active = 1 ORDER BY id",
            (sid,),
        ).fetchall()
    ]


def test_unaddressed_mutated_replay_rewrites_its_original_row(db):
    sid = "mutated-replay"
    db.create_session(sid, "desktop")
    original = {"role": "assistant", "content": "before repair", "timestamp": 1000.0}
    assert db.append_messages_batch(sid, [original]) == 1
    before = _active_rows(db, sid)

    # Live replay without physical row ids is a supported shape (gateway/ACP and identity-losing
    # handoffs). It still carries logical identity and, for a writable replay, the CAS version.
    restored = db.get_messages_as_conversation(sid, repair_alternation=True)
    assert "_row_id" not in restored[0]
    assert restored[0]["message_uid"] == before[0]["message_uid"]
    assert isinstance(restored[0].get(DB_ROW_SNAPSHOT), str)

    # Real rewrite sites change the payload BEFORE clearing the persisted marker.
    restored[0]["content"] = "after repair"
    restored[0].pop(_DB_PERSISTED_MARKER, None)

    assert db.append_messages_batch(sid, restored) == 0
    after = _active_rows(db, sid)
    assert len(after) == 1
    assert after[0]["id"] == before[0]["id"]
    assert after[0]["message_uid"] == before[0]["message_uid"]
    assert after[0]["content"] == "after repair"


def test_fresh_identical_message_is_never_adopted_by_payload(db):
    """Content and timestamps are not identity: two real occurrences may be byte-identical."""
    sid = "identical-twins"
    db.create_session(sid, "desktop")
    first = {"role": "user", "content": "same", "timestamp": 3000.0}
    second = {"role": "user", "content": "same", "timestamp": 3000.0}

    assert db.append_messages_batch(sid, [first]) == 1
    assert DB_ROW_SNAPSHOT not in second and "message_uid" not in second
    assert db.append_messages_batch(sid, [second]) == 1

    rows = _active_rows(db, sid)
    assert len(rows) == 2
    assert rows[0]["message_uid"] != rows[1]["message_uid"]
