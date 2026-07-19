"""Tests that verify SQL injection mitigations in insights state queries."""

import re
import threading

import pytest

from hermes_state import INSIGHTS_MAX_ROWS, SessionDB
from state_store.postgres.ops_sessions import PostgresSessionOperations


class _SQLiteCapture:
    def __init__(self) -> None:
        self.calls = []

    def execute(self, query, params):
        self.calls.append((query, params))
        return object()


def _sqlite_insights_query(source):
    db = SessionDB.__new__(SessionDB)
    db._lock = threading.RLock()
    db._conn = _SQLiteCapture()
    db._collect_rows_in_batches = lambda _cursor: []

    assert db.get_insights_sessions(123.5, source) == []
    return db._conn.calls[0]


def _postgres_insights_query(source):
    captured = []
    db = PostgresSessionOperations.__new__(PostgresSessionOperations)
    db._read_all_insight_rows = lambda query, params: captured.append(
        (query, params)
    ) or []

    assert db.get_insights_sessions(123.5, source) == []
    return captured[0]


@pytest.mark.parametrize(
    ("query_factory", "placeholder"),
    [
        (_sqlite_insights_query, "?"),
        (_postgres_insights_query, "%s"),
    ],
)
@pytest.mark.parametrize("source", [None, "cron'; DROP TABLE sessions; --"])
def test_insights_session_filters_are_parameterized(
    query_factory, placeholder, source
):
    query, params = query_factory(source)

    assert f"started_at >= {placeholder}" in query
    assert f"LIMIT {placeholder}" in query
    assert query.count(placeholder) == (3 if source else 2)
    assert params == (
        (123.5, source, INSIGHTS_MAX_ROWS + 1)
        if source
        else (123.5, INSIGHTS_MAX_ROWS + 1)
    )
    if source:
        assert f"source = {placeholder}" in query
        assert source not in query


@pytest.mark.parametrize(
    "query_factory",
    [_sqlite_insights_query, _postgres_insights_query],
)
def test_insights_session_columns_are_safe_identifiers(query_factory):
    query, _params = query_factory(None)
    selected = query.split("SELECT ", 1)[1].split(" FROM sessions", 1)[0]
    safe_identifier = re.compile(r"^[a-zA-Z_][a-zA-Z0-9_]*$")

    for column in (value.strip() for value in selected.split(",")):
        assert safe_identifier.fullmatch(column), (
            f"Column name {column!r} is not a safe SQL identifier"
        )
