"""Regression guards for the Postgres compression-lock acquire contract.

Background
----------
``SessionDB.try_acquire_compression_lock`` unpacks its worker's return value
into two names regardless of backend::

    acquired, reclaimed_holder = self._execute_write(_do)

The SQLite branch honours that. The Postgres branch — which delegates to
``hermes_state_postgres.acquire_compression_lock_sql`` — returned a bare
``bool``. The resulting ``TypeError`` was raised *after* the worker's INSERT
had been committed by the enclosing transaction, and was then swallowed by
the method's fail-open ``except Exception`` arm, which returns ``False``.

Net effect on a live deployment: every acquire wrote a ``compression_locks`` row,
reported "could not acquire", and left the row to sit until its 300s TTL
expired. Compression could never run on a Postgres-backed session. A long-lived
session grew to tens of thousands of messages because it was unable to rotate,
and operators saw "Compression already in progress (holder: pid=...)" naming
a holder that was the current process blocking itself.

Why these tests are Docker-free
-------------------------------
``tests/test_pg_parity_smoke.py`` already asserts compression-lock mutual
exclusion, but it ``importorskip``s ``psycopg`` + ``testcontainers`` and needs
a live Docker daemon, so it skips in CI — which is how a bare-bool return
shipped. These tests exercise the same contract against a fake connection, so
they run everywhere and fail loudly on a shape regression.

Note also that the parity test's ``== 1`` winner assertion could not have
caught this alone: with the bug, *zero* acquirers win, and a suite that skips
never evaluates it either way.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Tuple

import pytest

from hermes_state_postgres import acquire_compression_lock_sql


class FakeRow(dict):
    """Row supporting ``row["col"]`` like the adapter's ``_PostgresRow``."""


class FakeCursor:
    def __init__(self, row: Optional[FakeRow]) -> None:
        self._row = row

    def fetchone(self) -> Optional[FakeRow]:
        return self._row


class FakeConn:
    """Minimal stand-in for the Postgres adapter connection.

    Models ``compression_locks`` as a single-row-per-session dict and records
    every statement so tests can assert on the advisory-lock ordering.
    """

    def __init__(self) -> None:
        self.rows: Dict[str, Dict[str, Any]] = {}
        self.statements: List[str] = []

    def execute(self, sql: str, params: Tuple[Any, ...] = ()) -> FakeCursor:
        self.statements.append(" ".join(sql.split()))
        upper = sql.strip().upper()

        if "PG_ADVISORY_XACT_LOCK" in upper:
            return FakeCursor(None)

        if upper.startswith("SELECT HOLDER FROM COMPRESSION_LOCKS"):
            session_id = params[0]
            row = self.rows.get(session_id)
            if row is None:
                return FakeCursor(None)
            # The expired-holder probe carries a second bound param (now).
            if "EXPIRES_AT < ?" in upper and not row["expires_at"] < params[1]:
                return FakeCursor(None)
            return FakeCursor(FakeRow(holder=row["holder"]))

        if upper.startswith("DELETE FROM COMPRESSION_LOCKS"):
            session_id, now = params[0], params[1]
            row = self.rows.get(session_id)
            if row is not None and row["expires_at"] < now:
                del self.rows[session_id]
            return FakeCursor(None)

        if "INSERT" in upper and "COMPRESSION_LOCKS" in upper:
            session_id, holder, acquired_at, expires_at = params
            # INSERT OR IGNORE: an existing row wins.
            self.rows.setdefault(
                session_id,
                {
                    "holder": holder,
                    "acquired_at": acquired_at,
                    "expires_at": expires_at,
                },
            )
            return FakeCursor(None)

        raise AssertionError(f"unexpected SQL: {sql}")


# ---------------------------------------------------------------------------
# Return-shape contract — the actual regression
# ---------------------------------------------------------------------------


def test_returns_two_tuple_not_bare_bool():
    """The caller unpacks two values; a bare bool leaks the row it just wrote."""
    conn = FakeConn()
    now = time.time()
    result = acquire_compression_lock_sql(conn, "sess", "holder-a", now, now + 300)

    assert isinstance(result, tuple), (
        f"expected (acquired, reclaimed_holder), got {type(result).__name__} "
        f"{result!r} — this is the exact shape that leaks locks"
    )
    assert len(result) == 2
    acquired, reclaimed = result
    assert acquired is True
    assert reclaimed is None


def test_result_unpacks_the_way_the_caller_unpacks_it():
    """Mirror the caller's statement verbatim so a shape change fails here."""
    conn = FakeConn()
    now = time.time()
    acquired, reclaimed_holder = acquire_compression_lock_sql(
        conn, "sess", "holder-a", now, now + 300
    )
    assert acquired is True
    assert reclaimed_holder is None


# ---------------------------------------------------------------------------
# Mutual exclusion + reclaim semantics
# ---------------------------------------------------------------------------


def test_second_holder_is_refused_while_lock_is_live():
    conn = FakeConn()
    now = time.time()

    first, _ = acquire_compression_lock_sql(conn, "sess", "holder-a", now, now + 300)
    second, reclaimed = acquire_compression_lock_sql(
        conn, "sess", "holder-b", now + 1, now + 301
    )

    assert first is True
    assert second is False, "a live lock must not be stolen"
    assert reclaimed is None
    assert conn.rows["sess"]["holder"] == "holder-a"


def test_expired_lock_is_reclaimed_and_reported():
    """TTL expiry is the one cross-host-safe reclaim signal."""
    conn = FakeConn()
    now = time.time()

    acquire_compression_lock_sql(conn, "sess", "dead-holder", now - 600, now - 300)
    acquired, reclaimed = acquire_compression_lock_sql(
        conn, "sess", "fresh-holder", now, now + 300
    )

    assert acquired is True
    assert reclaimed == "dead-holder", (
        "the reclaimed holder drives the operator's only 'stale lock' warning"
    )
    assert conn.rows["sess"]["holder"] == "fresh-holder"


def test_no_lock_row_is_left_behind_when_acquire_is_refused():
    """A refused acquire must not add a row — that was the leak's shape."""
    conn = FakeConn()
    now = time.time()

    acquire_compression_lock_sql(conn, "sess", "holder-a", now, now + 300)
    before = dict(conn.rows["sess"])
    acquire_compression_lock_sql(conn, "sess", "holder-b", now + 1, now + 301)

    assert conn.rows["sess"] == before
    assert len(conn.rows) == 1


def test_advisory_lock_is_taken_before_any_mutation():
    """Serialization must precede the delete/insert/confirm sequence."""
    conn = FakeConn()
    now = time.time()
    acquire_compression_lock_sql(conn, "sess", "holder-a", now, now + 300)

    assert conn.statements, "no SQL issued"
    assert "pg_advisory_xact_lock" in conn.statements[0].lower(), (
        f"first statement was {conn.statements[0]!r}; without the advisory "
        "lock first, two acquirers can interleave delete/insert"
    )


# ---------------------------------------------------------------------------
# Caller-side guard
# ---------------------------------------------------------------------------


def test_caller_raises_on_bad_shape_instead_of_failing_open():
    """A wrong return shape must surface, not be swallowed as 'lock held'.

    The fail-open ``except Exception`` arm exists for driver faults. When it
    also caught ``TypeError`` from a bad unpack, a code bug became an
    indefinite stall that looked like healthy lock contention.
    """
    import hermes_state

    class Boom(hermes_state.SessionDB):
        def __init__(self):  # bypass real DB setup
            self._is_postgres = True

        def _execute_write(self, fn, patience_s=None, **kwargs):
            return True  # the pre-fix bare-bool shape

    with pytest.raises(TypeError, match="expected a"):
        hermes_state.SessionDB.try_acquire_compression_lock(
            Boom(), "sess", "holder-a"
        )
