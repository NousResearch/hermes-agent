"""Schema reconciliation must surface migration failures to its retry owner."""

import sqlite3
from typing import cast

import pytest

from hermes_state_schema import SessionSchemaMixin


class _SingleColumnSchema(SessionSchemaMixin):
    @staticmethod
    def _parse_schema_columns(schema_sql):
        del schema_sql
        return {"sessions": {"missing_column": "TEXT"}}


class _ReconcileCursor:
    def __init__(self, *, probe_error=None, alter_error=None):
        self.probe_error = probe_error
        self.alter_error = alter_error

    def execute(self, statement):
        if statement.startswith("PRAGMA table_info"):
            if self.probe_error:
                raise self.probe_error
            return self
        if statement.startswith("ALTER TABLE"):
            if self.alter_error:
                raise self.alter_error
            return self
        raise AssertionError(f"unexpected statement: {statement}")

    def fetchall(self):
        return [(0, "id", "TEXT", 0, None, 1)]


def test_reconcile_propagates_schema_probe_lock():
    cursor = _ReconcileCursor(
        probe_error=sqlite3.OperationalError("database is locked")
    )

    with pytest.raises(sqlite3.OperationalError, match="database is locked"):
        _SingleColumnSchema()._reconcile_columns(cast(sqlite3.Cursor, cursor))


@pytest.mark.parametrize("message", [
    "database is locked",
    "Cannot add a NOT NULL column with default value NULL",
    "duplicate column constraint failure",
])
def test_reconcile_propagates_database_lock(message):
    cursor = _ReconcileCursor(
        alter_error=sqlite3.OperationalError(message)
    )

    with pytest.raises(sqlite3.OperationalError) as caught:
        _SingleColumnSchema()._reconcile_columns(cast(sqlite3.Cursor, cursor))
    assert caught.value is cursor.alter_error


def test_reconcile_tolerates_duplicate_column_race():
    cursor = _ReconcileCursor(
        alter_error=sqlite3.OperationalError("duplicate column name: missing_column")
    )

    _SingleColumnSchema()._reconcile_columns(cast(sqlite3.Cursor, cursor))
