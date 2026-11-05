"""SQLite summaries keep the stored counter cells the legacy store's rich reads return."""

import sqlite3

import pytest

from persistence.conversation.sessions import NewSession, SessionQuery, SessionWindow
from tests.persistence_sqlite.conftest import store_for


@pytest.mark.parametrize("field", [
    "message_count", "tool_call_count", "input_tokens", "output_tokens",
    "cache_read_tokens", "cache_write_tokens", "reasoning_tokens", "api_call_count",
])
@pytest.mark.parametrize("value", [None, "not-an-integer", 1.5, 17])
def test_summary_preserves_counter_cell(backend, scope_a, field, value):
    sessions = backend.open(scope_a).sessions
    sessions.create(NewSession("target", "acp"))
    with sqlite3.connect(store_for(scope_a)) as connection:
        connection.execute(f"UPDATE sessions SET {field}=? WHERE id='target'", (value,))
    assert getattr(sessions.summary("target").session, field) == value
    page = sessions.list_sessions(SessionQuery(), SessionWindow(limit=10))
    assert len(page) == 1
    assert getattr(page[0].session, field) == value
    assert sessions.summary("missing") is None
