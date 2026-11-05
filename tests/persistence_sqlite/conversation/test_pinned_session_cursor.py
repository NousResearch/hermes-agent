"""Pinned listings on SQLite keep the legacy store's first window and its pin back-fill on every window."""

from unittest.mock import patch

import pytest

from persistence.conversation.sessions import NewSession, SessionQuery, SessionWindow
from tests.persistence_sqlite._legacy_store import legacy_store

__all__ = ["legacy_store"]

IDS = ("old", "middle", "pin")


@pytest.fixture
def oracle(tmp_path, legacy_store):
    from hermes_state import SessionDB

    db = SessionDB(db_path=tmp_path / "oracle.db")
    try:
        yield db
    finally:
        db.close()


@pytest.mark.parametrize("pins", [("pin",), ("old",), ("old", "pin")])
@pytest.mark.parametrize("limit", [1, 2, 3])
def test_pin_backfill_preserves_the_first_window(backend, scope_a, oracle, pins, limit):
    sessions = backend.open(scope_a).sessions
    for index, sid in enumerate(IDS):
        with patch("time.time", return_value=1700000000 + index):
            sessions.create(NewSession(sid, "cli"))
            oracle.create_session(sid, "cli")
    for sid in pins:
        sessions.set_pinned(sid, True)
        oracle.set_session_pinned(sid, True)
    query = SessionQuery(include_pinned=True)
    for offset in range(0, len(IDS) + limit, limit):
        expected = [row["id"] for row in oracle.list_sessions_rich(limit=limit, offset=offset, include_pinned=True)]
        window = sessions.list_sessions(query, SessionWindow(limit=limit, offset=offset))
        assert [row.session.session_id for row in window] == expected
    first = sessions.list_sessions(query, SessionWindow(limit=limit))
    assert set(pins) <= {row.session.session_id for row in first}
