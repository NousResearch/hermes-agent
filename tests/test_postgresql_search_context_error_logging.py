"""Offline regression for #118142: optional PG search context failure is observable."""

import logging
import sqlite3
import traceback
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock

from state_store_postgresql import PostgreSQLStateStore


_CANDIDATE = {
    "id": 17,
    "session_id": "pg-session",
    "role": "user",
    "snippet": "needle in haystack",
    "timestamp": 123.5,
    "tool_name": None,
    "source": "cli",
    "model": "test-model",
    "session_started": 120.0,
    "rank": 0.75,
}


def _store_with_candidate(monkeypatch):
    store = object.__new__(PostgreSQLStateStore)
    store._schema = "hermes_state_store_tenant_" + "a" * 32
    monkeypatch.setattr(store, "_psycopg", SimpleNamespace(rows=SimpleNamespace(dict_row=object())), raising=False)
    cursor = MagicMock()
    cursor.fetchall.return_value = [dict(_CANDIDATE)]
    connection = MagicMock()
    connection.__enter__.return_value = connection
    connection.cursor.return_value.__enter__.return_value = cursor
    monkeypatch.setattr(store, "_connection", Mock(return_value=connection))
    sqlite_connect = Mock(side_effect=AssertionError("SQLite fallback is forbidden"))
    monkeypatch.setattr(sqlite3, "connect", sqlite_connect)
    return store, cursor, sqlite_connect


def test_context_failure_logs_traceback_but_preserves_lexical_candidate(monkeypatch, caplog):
    store, cursor, sqlite_connect = _store_with_candidate(monkeypatch)
    failure = RuntimeError("injected PG search context failure")
    lookup = Mock(side_effect=failure)
    monkeypatch.setattr(store, "_search_contexts", lookup)

    with caplog.at_level(logging.ERROR, logger="state_store_postgresql"):
        rows = store.search_messages("needle")

    assert rows == [{**{key: value for key, value in _CANDIDATE.items() if key != "rank"}, "context": []}]
    lookup.assert_called_once_with([17])
    cursor.execute.assert_called_once()
    sqlite_connect.assert_not_called()
    records = [record for record in caplog.records if record.name == "state_store_postgresql"]
    assert len(records) == 1
    record = records[0]
    assert record.levelno == logging.ERROR
    assert record.getMessage() == "Failed to enrich PostgreSQL search result context"
    assert record.exc_info is not None
    assert record.exc_info[1] is failure
    assert "RuntimeError: injected PG search context failure" in "".join(
        traceback.format_exception(*record.exc_info)
    )


def test_fields_without_context_skip_lookup_and_keep_lexical_fields(monkeypatch, caplog):
    store, cursor, sqlite_connect = _store_with_candidate(monkeypatch)
    lookup = Mock(side_effect=AssertionError("context must not be fetched"))
    monkeypatch.setattr(store, "_search_contexts", lookup)

    with caplog.at_level(logging.ERROR, logger="state_store_postgresql"):
        rows = store.search_messages("needle", fields=("id", "snippet", "session_id"))

    assert rows == [{"id": 17, "session_id": "pg-session", "snippet": "needle in haystack"}]
    lookup.assert_not_called()
    cursor.execute.assert_called_once()
    sqlite_connect.assert_not_called()
    assert not [record for record in caplog.records if record.name == "state_store_postgresql"]
