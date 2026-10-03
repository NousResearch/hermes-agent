"""A PostgreSQL boolean CASE expression needs a bool, not a SQLite integer."""

from contextlib import contextmanager
from unittest.mock import patch

import pytest

from gateway import delivery_ledger as ledger


class _Cursor:
    def __init__(self, rows=(), rowcount=0):
        self.rows = rows
        self.rowcount = rowcount

    def fetchall(self):
        return self.rows


class _Connection:
    def __init__(self, row):
        self.row = row
        self.claim_parameters = None

    def execute(self, sql, parameters=()):
        if sql.lstrip().startswith("SELECT obligation_id"):
            return _Cursor([self.row])
        if "last_error=CASE WHEN ? THEN" in sql:
            self.claim_parameters = parameters
            return _Cursor(rowcount=1)
        raise AssertionError("Unexpected ledger SQL: " + sql[:100])


@pytest.mark.parametrize("state,last_error,expected", [
    ("pending", None, False),
    ("failed", "flood_control:0.5", True),
])
def test_sweep_claim_binds_sql_boolean(state, last_error, expected):
    row = (
        "synthetic-obligation", "", "telegram", "synthetic-chat", None,
        "synthetic content", state, 0, 100.0, None, None, "default", last_error, 100.0,
    )
    conn = _Connection(row)

    @contextmanager
    def transaction():
        yield conn

    with (
        patch.object(ledger, "_transaction", transaction),
        patch.object(ledger, "_owner_stamp", return_value=(12345, 67890)),
    ):
        claimed = ledger.sweep_recoverable(
            now=101.0,
            deliverable_platforms={"telegram"},
            deliverable_targets={("telegram", "default")},
        )

    assert len(claimed) == 1
    assert conn.claim_parameters is not None
    assert type(conn.claim_parameters[3]) is bool
    assert conn.claim_parameters[3] is expected
