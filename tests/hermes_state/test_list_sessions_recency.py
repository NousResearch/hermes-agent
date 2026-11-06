"""Recency order of ``list_sessions_rich(order_by_last_active=True)``.

Only sessions with a child seed the compression-chain walk; every childless row
is ranked by its own last activity. The two kinds must still sort as one list.
"""

import time

import pytest

from hermes_state import SessionDB


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
