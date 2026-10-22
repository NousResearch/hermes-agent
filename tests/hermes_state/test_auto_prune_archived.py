"""Unattended auto-prune keeps archived sessions, like pinned ones; manual prune keeps control."""

import time

import pytest

from hermes_state import SessionDB


@pytest.fixture()
def db(tmp_path):
    session_db = SessionDB(db_path=tmp_path / "test_state.db")
    yield session_db
    session_db.close()


def _make_old_ended(db, sid: str, days_old: int = 100) -> None:
    db.create_session(session_id=sid, source="cli")
    db.end_session(sid, end_reason="done")
    db._conn.execute(
        "UPDATE sessions SET started_at = ? WHERE id = ?",
        (time.time() - days_old * 86400, sid),
    )
    db._conn.commit()


def test_auto_prune_preserves_archived_session(db):
    _make_old_ended(db, "archived")
    _make_old_ended(db, "ordinary")
    assert db.set_session_archived("archived", True) is True

    result = db.maybe_auto_prune_and_vacuum(retention_days=90, min_interval_hours=0, vacuum=False)

    assert result["pruned"] == 1
    assert db.get_session("ordinary") is None
    archived = db.get_session("archived")
    assert archived is not None
    assert archived["archived"] == 1


def test_manual_prune_can_still_include_archived(db):
    _make_old_ended(db, "archived")
    assert db.set_session_archived("archived", True) is True

    assert db.prune_sessions(older_than_days=90, archived=None) == 1
    assert db.get_session("archived") is None
