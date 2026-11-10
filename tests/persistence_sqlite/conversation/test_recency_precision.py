"""The roster's latest local conversation on SQLite compares the stored epochs exactly."""

import sqlite3

import pytest

from persistence.conversation.sessions import NewSession
from tests.persistence_sqlite.conftest import store_for


@pytest.mark.parametrize(("first", "second", "first_activity", "second_activity"), [
    ("local-a", "local-b", 100.0000004, 100),
    ("a-base", "z-neighbor", 1700000000.123457, 1700000000.1234567),
    ("a-base", "z-neighbor", 100.00000049, 100.0000004),
    ("local-a", "local-b", 100.000001, 100),
])
def test_latest_local_preserves_distinct_epochs(backend, scope_a, first, second, first_activity, second_activity):
    sessions = backend.open(scope_a).sessions
    for sid in (first, second):
        sessions.create(NewSession(sid, "cli", chat_id="local-" + sid, user_id="owner"))
    with sqlite3.connect(store_for(scope_a)) as connection:
        connection.execute("UPDATE sessions SET started_at=200,last_activity_at=? WHERE id=?", (first_activity, first))
        connection.execute("UPDATE sessions SET started_at=100,last_activity_at=? WHERE id=?", (second_activity, second))
    roster = sessions.roster_sessions(owner_user_id="owner", owner=None)
    assert roster.last_session.id == first
    assert roster.last_session.last_active == first_activity
