"""The roster's local-chat admission on SQLite keeps SQLite LIKE's ASCII case handling."""

import sqlite3

import pytest

from persistence.conversation.sessions import NewSession
from tests.persistence_sqlite.conftest import store_for


@pytest.mark.parametrize(("chat_id", "expected"), [
    ("LOCAL-new", "z-neighbor"),
    ("Local-new", "z-neighbor"),
    ("local-new", "z-neighbor"),
    ("local-", "z-neighbor"),
    ("remote-new", "a-base"),
    ("xlocal-new", "a-base"),
    ("local", "a-base"),
    ("Ｌocal-new", "a-base"),
])
def test_latest_local_ascii_prefix(backend, scope_a, chat_id, expected):
    sessions = backend.open(scope_a).sessions
    for sid, chat in (("a-base", "local-base"), ("z-neighbor", chat_id)):
        sessions.create(NewSession(sid, "cli", chat_id=chat, user_id="owner"))
    with sqlite3.connect(store_for(scope_a)) as connection:
        connection.execute("UPDATE sessions SET started_at=100,last_activity_at=100")
        connection.execute("UPDATE sessions SET last_activity_at=300 WHERE id='z-neighbor'")
    assert sessions.roster_sessions(owner_user_id="owner", owner=None).last_session.id == expected
    assert sessions.roster_sessions(owner_user_id="absent", owner=None).last_session is None
    with sqlite3.connect(store_for(scope_a)) as connection:
        connection.execute("UPDATE sessions SET archived=1 WHERE id='z-neighbor'")
    assert sessions.roster_sessions(owner_user_id="owner", owner=None).last_session.id == "a-base"
