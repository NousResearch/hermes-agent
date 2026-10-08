"""Offline regression for #118142: PG reset failure keeps its traceback and False result."""

import logging
import sqlite3
import traceback
from unittest.mock import Mock

from state_store_postgresql_sessions import PostgreSQLSessionsMixin


def test_reset_connection_failure_logs_traceback_without_sqlite_fallback(monkeypatch, caplog):
    store = object.__new__(PostgreSQLSessionsMixin)
    connection = Mock(side_effect=RuntimeError("injected PG reset connection failure"))
    sqlite_connect = Mock(side_effect=AssertionError("SQLite fallback is forbidden"))
    monkeypatch.setattr(store, "_connection", connection, raising=False)
    monkeypatch.setattr(sqlite3, "connect", sqlite_connect)

    with caplog.at_level(logging.DEBUG, logger="state_store_postgresql_sessions"):
        assert store.promote_to_session_reset("session") is False

    connection.assert_called_once_with()
    sqlite_connect.assert_not_called()
    records = [record for record in caplog.records if record.name == "state_store_postgresql_sessions"]
    assert len(records) == 1
    record = records[0]
    assert record.levelno == logging.DEBUG
    assert record.exc_info is not None
    assert record.exc_info[0] is RuntimeError
    assert "RuntimeError: injected PG reset connection failure" in "".join(
        traceback.format_exception(*record.exc_info)
    )
    assert "injected PG reset connection failure" not in record.getMessage()


def test_blank_reset_id_does_not_open_connection_or_log(monkeypatch, caplog):
    store = object.__new__(PostgreSQLSessionsMixin)
    connection = Mock(side_effect=AssertionError("blank session must not open PG"))
    sqlite_connect = Mock(side_effect=AssertionError("SQLite fallback is forbidden"))
    monkeypatch.setattr(store, "_connection", connection, raising=False)
    monkeypatch.setattr(sqlite3, "connect", sqlite_connect)

    with caplog.at_level(logging.DEBUG, logger="state_store_postgresql_sessions"):
        assert store.promote_to_session_reset("") is False

    connection.assert_not_called()
    sqlite_connect.assert_not_called()
    assert not [record for record in caplog.records if record.name == "state_store_postgresql_sessions"]
