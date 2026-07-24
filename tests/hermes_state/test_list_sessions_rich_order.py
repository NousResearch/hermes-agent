"""Ordering regression tests for ``SessionDB.list_sessions_rich``.

The default listing sorts by ``started_at DESC``. When two sessions share an
identical ``started_at`` the sort is otherwise non-deterministic, so the query
must fall back to a stable secondary key (``id DESC``) to give callers a
repeatable order. See the ``ORDER BY s.started_at DESC, s.id DESC`` clause in
``hermes_state.py``.
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


def _make_sessions_with_equal_started_at(db: SessionDB, ids):
    """Create top-level sessions that all share one fixed ``started_at``.

    Rows are inserted in ascending-id order so that any natural/insertion
    ordering would surface them ascending; the tiebreaker is what makes the
    default listing return them descending.
    """
    started_at = time.time() - 100
    for session_id in ids:
        db.create_session(session_id, source="cli")
        db._conn.execute(
            "UPDATE sessions SET started_at = ?, message_count = 1 WHERE id = ?",
            (started_at, session_id),
        )
    db._conn.commit()


def test_default_order_breaks_started_at_ties_by_id_desc(db):
    _make_sessions_with_equal_started_at(db, ["sess_001", "sess_002", "sess_003"])

    listed = [s["id"] for s in db.list_sessions_rich()]

    assert listed == ["sess_003", "sess_002", "sess_001"]


def test_tiebreaker_order_is_stable_across_calls(db):
    _make_sessions_with_equal_started_at(db, ["sess_001", "sess_002", "sess_003"])

    first = [s["id"] for s in db.list_sessions_rich()]
    second = [s["id"] for s in db.list_sessions_rich()]

    assert first == second == ["sess_003", "sess_002", "sess_001"]
