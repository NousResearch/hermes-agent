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


def _compression_pair(db: SessionDB):
    base = time.time() - 100
    db.create_session("root", source="cli")
    db.create_session("tip", source="cli", parent_session_id="root")
    db._conn.execute(
        "UPDATE sessions SET started_at = ?, ended_at = ?, end_reason = 'compression', message_count = 1 WHERE id = 'root'",
        (base, base + 10),
    )
    db._conn.execute(
        "UPDATE sessions SET started_at = ?, message_count = 1 WHERE id = 'tip'",
        (base + 20,),
    )
    db._conn.commit()


def test_archiving_compression_tip_archives_projected_root(db):
    _compression_pair(db)

    assert db.set_session_archived("tip", True) is True

    assert db.get_session("root")["archived"] == 1
    assert db.get_session("tip")["archived"] == 1
    assert [s["id"] for s in db.list_sessions_rich(order_by_last_active=True)] == []
    assert [s["id"] for s in db.list_sessions_rich(order_by_last_active=True, archived_only=True)] == ["tip"]


def test_unarchiving_compression_tip_unarchives_projected_root(db):
    _compression_pair(db)
    db.set_session_archived("tip", True)

    assert db.set_session_archived("tip", False) is True

    assert db.get_session("root")["archived"] == 0
    assert db.get_session("tip")["archived"] == 0
    assert db.get_session("tip")["archived_at"] is None
    assert [s["id"] for s in db.list_sessions_rich(order_by_last_active=True)] == ["tip"]


def test_set_session_archived_stamps_archived_at(db):
    before = time.time()
    db.create_session("old", source="cli")
    db._conn.execute(
        "UPDATE sessions SET started_at = ?, message_count = 1 WHERE id = 'old'",
        (before - 86400 * 40,),
    )
    db._conn.commit()

    assert db.set_session_archived("old", True) is True
    row = db.get_session("old")
    assert row["archived"] == 1
    assert row["archived_at"] is not None
    assert float(row["archived_at"]) >= before

    listed = db.list_sessions_rich(order_by_last_active=True, archived_only=True)
    assert [s["id"] for s in listed] == ["old"]
    assert listed[0]["archived_at"] is not None

    assert db.set_session_archived("old", False) is True
    assert db.get_session("old")["archived_at"] is None


def test_rearchive_keeps_original_archived_at(db):
    db.create_session("s", source="cli")
    assert db.set_session_archived("s", True) is True
    first = float(db.get_session("s")["archived_at"])
    kept = first - 100
    db._conn.execute("UPDATE sessions SET archived_at = ? WHERE id = 's'", (kept,))
    db._conn.commit()

    assert db.set_session_archived("s", True) is True
    assert db.get_session("s")["archived_at"] == pytest.approx(kept)


def test_archived_only_list_orders_by_archived_at_not_last_active(db):
    now = time.time()
    db.create_session("stale_activity", source="cli")
    db.create_session("fresh_activity", source="cli")
    db._conn.execute(
        "UPDATE sessions SET last_activity_at = ?, message_count = 1 WHERE id = 'stale_activity'",
        (now - 86400 * 40,),
    )
    db._conn.execute(
        "UPDATE sessions SET last_activity_at = ?, message_count = 1 WHERE id = 'fresh_activity'",
        (now,),
    )
    db._conn.commit()
    db.set_session_archived("fresh_activity", True)
    db._conn.execute(
        "UPDATE sessions SET archived_at = ? WHERE id = 'fresh_activity'",
        (now - 100,),
    )
    db._conn.commit()
    db.set_session_archived("stale_activity", True)

    listed = db.list_sessions_rich(order_by_last_active=True, archived_only=True)
    assert [s["id"] for s in listed] == ["stale_activity", "fresh_activity"]
