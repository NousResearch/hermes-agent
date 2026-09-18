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
    assert [s["id"] for s in db.list_sessions_rich(order_by_last_active=True)] == ["tip"]


def test_archived_only_view_includes_hidden_archived_sessions(db):
    """The archived-only view is the recovery surface: a session that is both
    archived and hidden (Bot Mode marks its sessions hidden) must appear
    there, otherwise it is unreachable from every UI list (#90946)."""
    db.create_session("plain", source="cli")
    db.create_session("both", source="cli")
    assert db.set_session_hidden("both", True) is True
    assert db.set_session_archived("both", True) is True

    # Default list: hidden rows stay excluded (unchanged behaviour)...
    assert [s["id"] for s in db.list_sessions_rich(order_by_last_active=True)] == ["plain"]
    # ...and the archived-only view must surface the archived+hidden row.
    assert [s["id"] for s in db.list_sessions_rich(order_by_last_active=True, archived_only=True)] == ["both"]
