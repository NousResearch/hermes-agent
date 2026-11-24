"""The roster's latest local conversation on SQLite uses a NULL fallback, not a falsy-value fallback, and skips
hidden rows."""

import sqlite3

import pytest

from persistence.conversation.sessions import NewSession
from tests.persistence_sqlite.conftest import store_for


@pytest.mark.parametrize(("activity", "expected"), [(0, "local-b"), (None, "local-a"), (201, "local-a")])
def test_latest_local_preserves_zero_activity(backend, scope_a, activity, expected):
    sessions = backend.open(scope_a).sessions
    for sid in ("local-a", "local-b", "foreign", "archived", "remote", "hidden"):
        sessions.create(NewSession(sid, "cli", chat_id="local-" + sid, user_id="owner"))
    with sqlite3.connect(store_for(scope_a)) as connection:
        connection.execute("UPDATE sessions SET started_at=100,last_activity_at=100")
        connection.execute("UPDATE sessions SET started_at=200,last_activity_at=? WHERE id='local-a'", (activity,))
        connection.execute("UPDATE sessions SET last_activity_at=300,user_id='other' WHERE id='foreign'")
        connection.execute("UPDATE sessions SET last_activity_at=400,archived=1 WHERE id='archived'")
        connection.execute("UPDATE sessions SET last_activity_at=500,chat_id='remote' WHERE id='remote'")
        connection.execute("UPDATE sessions SET last_activity_at=600 WHERE id='hidden'")
        connection.execute("UPDATE sessions SET hidden=1 WHERE id='hidden'")
    assert sessions.roster_sessions(owner_user_id="owner", owner=None).last_session.id == expected
    assert sessions.roster_sessions(owner_user_id="absent", owner=None).last_session is None
    assert sessions.roster_sessions(owner_user_id=None, owner=None).last_session.id == "foreign"
