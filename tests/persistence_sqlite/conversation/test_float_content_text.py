"""Float message content on SQLite reads back as the text a SQLite TEXT column stores for it, as main's store does: the installed library's spelling, whatever that build prints."""

from __future__ import annotations

import sqlite3
from contextlib import closing

import pytest

from persistence.conversation.messages import NewMessage
from persistence.conversation.sessions import NewSession
from persistence.core.identity import SessionId

S = SessionId("float-text")

# Values whose text spelling changed between SQLite builds (15 significant digits before, the shortest round-trip text after).
_VALUES = (0.1 + 0.2, 1e16, 5e-324, 100.0, 1 / 3, 2.2250738585072014e-308, 1.7976931348623157e308, 9007199254740993.0)


def _text_column_spelling(value: float) -> str:
    with closing(sqlite3.connect(":memory:")) as conn:
        conn.execute("CREATE TABLE spelled (value TEXT)")
        conn.execute("INSERT INTO spelled (value) VALUES (?)", (value,))
        return conn.execute("SELECT value FROM spelled").fetchone()[0]


@pytest.mark.parametrize("value", _VALUES)
def test_float_content_reads_back_as_a_sqlite_text_column_stores_it(backend, scope_a, value):
    store = backend.open(scope_a)
    store.sessions.create(NewSession(session_id=S, source="cli"))
    store.messages.append(S, NewMessage(role="user", content=value, timestamp=1.0))
    (row,) = store.messages.read_transcript(S)
    assert row.content == _text_column_spelling(value), sqlite3.sqlite_version
