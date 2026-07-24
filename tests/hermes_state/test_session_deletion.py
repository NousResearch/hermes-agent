"""Regression tests for delete_session on compression chains.

Mirrors the archiving coverage in test_session_archiving.py.  Deleting any
node in a compression chain must remove the entire logical conversation —
not just the visible tip (the "onion peeling" bug that shipped before this
fix).  Non-compression branches must be orphaned, not deleted.
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


def _compression_pair(db: SessionDB):
    """Create a root → tip compression chain (both nodes have compression end_reason)."""
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


def _compression_triple(db: SessionDB):
    """Create a root → mid → tip chain (3 segments, all compressed)."""
    base = time.time() - 300
    db.create_session("root", source="cli")
    db.create_session("mid", source="cli", parent_session_id="root")
    db.create_session("tip", source="cli", parent_session_id="mid")
    db._conn.execute(
        "UPDATE sessions SET started_at = ?, ended_at = ?, end_reason = 'compression', message_count = 1 WHERE id = 'root'",
        (base, base + 10),
    )
    db._conn.execute(
        "UPDATE sessions SET started_at = ?, ended_at = ?, end_reason = 'compression', message_count = 1 WHERE id = 'mid'",
        (base + 20, base + 30),
    )
    db._conn.execute(
        "UPDATE sessions SET started_at = ?, message_count = 1 WHERE id = 'tip'",
        (base + 40,),
    )
    db._conn.commit()


def test_delete_compression_tip_removes_entire_chain(db):
    _compression_pair(db)

    assert db.delete_session("tip") is True

    # Both root and tip must be gone.
    assert db.get_session("root") is None
    assert db.get_session("tip") is None
    assert [s["id"] for s in db.list_sessions_rich(order_by_last_active=True)] == []


def test_delete_compression_root_removes_entire_chain(db):
    _compression_pair(db)

    assert db.delete_session("root") is True

    assert db.get_session("root") is None
    assert db.get_session("tip") is None
    assert [s["id"] for s in db.list_sessions_rich(order_by_last_active=True)] == []


def test_delete_compression_middle_removes_entire_chain(db):
    _compression_triple(db)

    assert db.delete_session("mid") is True

    assert db.get_session("root") is None
    assert db.get_session("mid") is None
    assert db.get_session("tip") is None
    assert [s["id"] for s in db.list_sessions_rich(order_by_last_active=True)] == []


def test_delete_compression_chain_orphans_non_compression_branch(db):
    """A branch child that doesn't match the compression contract must survive."""
    base = time.time() - 100
    db.create_session("root", source="cli")
    db.create_session("tip", source="cli", parent_session_id="root")
    db.create_session("branch", source="cli", parent_session_id="root")
    db._conn.execute(
        "UPDATE sessions SET started_at = ?, ended_at = ?, end_reason = 'compression', message_count = 1 WHERE id = 'root'",
        (base, base + 10),
    )
    db._conn.execute(
        "UPDATE sessions SET started_at = ?, message_count = 1 WHERE id = 'tip'",
        (base + 20,),
    )
    # branch was created WHILE root was still live (started_at < ended_at)
    db._conn.execute(
        "UPDATE sessions SET started_at = ?, message_count = 1 WHERE id = 'branch'",
        (base + 5,),
    )
    db._conn.commit()

    assert db.delete_session("tip") is True

    # Compression chain gone; branch orphaned (parent_session_id = NULL).
    assert db.get_session("root") is None
    assert db.get_session("tip") is None
    branch = db.get_session("branch")
    assert branch is not None
    assert branch["parent_session_id"] is None


def test_delete_standalone_session(db):
    db.create_session("solo", source="cli")
    assert db.delete_session("solo") is True
    assert db.get_session("solo") is None


def test_delete_nonexistent_session_returns_false(db):
    assert db.delete_session("does_not_exist") is False
