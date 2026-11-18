"""The SQLite gateway and recent listings keep the stored counter and display-name cells."""

import sqlite3

import pytest

from persistence.conversation.messages import NewMessage
from persistence.conversation.sessions import NewSession, RecentBrowseQuery
from tests.persistence_sqlite.conftest import store_for


def _stored_row(scope, session_id):
    with sqlite3.connect(store_for(scope)) as connection:
        return connection.execute("SELECT * FROM sessions WHERE id = ?", (session_id,)).fetchone()


@pytest.mark.parametrize("field", ["input_tokens", "output_tokens"])
@pytest.mark.parametrize("value", [None, "not-an-integer", -0.75, 1.5, 0, 17])
def test_gateway_preserves_stored_cells(backend, scope_a, field, value):
    sessions = backend.open(scope_a).sessions
    sessions.create(NewSession("target", "telegram", session_key="telegram:owner", display_name=""))
    with sqlite3.connect(store_for(scope_a)) as connection:
        connection.execute(f"UPDATE sessions SET {field}=? WHERE id='target'", (value,))
    before = _stored_row(scope_a, "target")
    entries = sessions.list_gateway_sessions(active_only=False)
    assert len(entries) == 1
    assert getattr(entries[0].session, field) == value
    assert entries[0].session.display_name == ""
    assert _stored_row(scope_a, "target") == before


@pytest.mark.parametrize("field", ["message_count", "tool_call_count"])
@pytest.mark.parametrize("value", [None, "not-an-integer", -0.75, 1.5, 0, 17])
def test_recent_preserves_stored_counters(backend, scope_a, field, value):
    store = backend.open(scope_a)
    store.sessions.create(NewSession("target", "cli"))
    store.messages.append("target", NewMessage(role="user", content="hello"))
    with sqlite3.connect(store_for(scope_a)) as connection:
        connection.execute(f"UPDATE sessions SET {field}=? WHERE id='target'", (value,))
    before = _stored_row(scope_a, "target")
    rows = store.sessions.list_recent_sessions(RecentBrowseQuery(limit=10))
    assert len(rows) == 1
    assert getattr(rows[0], field) == value
    assert _stored_row(scope_a, "target") == before
