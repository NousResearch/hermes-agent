"""Recency order of ``list_sessions_rich(order_by_last_active=True)``.

Only sessions with a child seed the compression-chain walk; every childless row
is ranked by its own last activity. The two kinds must still sort as one list.
"""

import time

import pytest

from hermes_state import SessionDB
import hermes_state_runtime as rt


@pytest.fixture
def db(tmp_path):
    database = SessionDB(tmp_path / "state.db")
    try:
        yield database
    finally:
        database.close()


def _session(db: SessionDB, sid: str, started: float, message_at=None, **kwargs):
    db.create_session(sid, source="cli", **kwargs)
    if message_at is not None:
        db.append_message(session_id=sid, role="user", content=f"hello from {sid}")
        db._conn.execute("UPDATE messages SET timestamp = ? WHERE session_id = ?", (message_at, sid))
    db._conn.execute(
        "UPDATE sessions SET started_at = ?, last_activity_at = NULL, message_count = 1 WHERE id = ?",
        (started, sid),
    )


def test_chained_and_childless_sessions_share_one_recency_order(db):
    base = time.time() - 1000
    # A compression chain whose tip holds the newest message of all.
    _session(db, "root", base)
    db._conn.execute(
        "UPDATE sessions SET ended_at = ?, end_reason = 'compression' WHERE id = 'root'", (base + 10,)
    )
    _session(db, "tip", base + 20, message_at=base + 90, parent_session_id="root")
    # Childless: "fresh" started first but was active later than "stale" started.
    _session(db, "fresh", base + 5, message_at=base + 60)
    _session(db, "stale", base + 30)
    db._conn.commit()

    rows = db.list_sessions_rich(order_by_last_active=True)

    assert [s["id"] for s in rows] == ["tip", "fresh", "stale"]


def test_compression_tip_list_uses_tip_mutation_fence(db):
    """A listed tip id must carry the same durable counters the mutation fence checks."""
    db.create_session("root", source="cli")
    db.append_message(session_id="root", role="user", content="before compression")
    db.end_session("root", "compression")
    db.create_session("tip", source="cli", parent_session_id="root")
    db.append_message(session_id="tip", role="user", content="after compression")
    db._conn.execute(
        "UPDATE sessions SET runtime_revision = 1, runtime_generation = 1 WHERE id = 'root'"
    )
    db._conn.execute(
        "UPDATE sessions SET runtime_revision = 7, runtime_generation = 8 WHERE id = 'tip'"
    )
    db._conn.commit()

    rows = db.list_sessions_rich(compact_rows=True)
    assert len(rows) == 1
    listed = rows[0]
    tip = db.get_session("tip")
    assert listed["id"] == "tip"
    assert (listed["runtime_revision"], listed["runtime_generation"]) == (
        tip["runtime_revision"], tip["runtime_generation"]
    )

    result = rt.mutate_runtime_session(
        db, epoch=rt.begin_runtime_epoch(db, instance_id="owner"),
        principal_id="human", session_id=listed["id"], request_id="rename-tip",
        expected_revision=listed["runtime_revision"],
        expected_generation=listed["runtime_generation"],
        operation="sidebar", payload={"title": "Renamed after compression"},
    )
    assert result["revision"] == 8
    assert db.get_session("tip")["title"] == "Renamed after compression"
