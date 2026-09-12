"""Regression tests for the reconnect/retry transaction-boundary fixes.

Blocker 4 — three defects guarded here:

A. Retry wraps the wrong methods
   ``_call_with_retry`` on ``_PostgresConnection`` wrapped ``cursor()``,
   ``commit()``, and ``rollback()``.  A stale TCP connection is normally
   discovered on the actual ``execute()``, not while constructing a cursor, so
   the real failure path was uncovered.

   Fix: ``_PostgresCursor.execute()`` propagates connection loss without
   replaying a statement whose outcome is unknown.

B. Retrying commit() is unsafe
   The original ``commit()`` retry reconnected and called ``commit()`` on the
   NEW connection.  The original transaction's outcome was UNKNOWN — a fresh
   empty commit then reported durable writes that may never have landed.

   Fix: connection loss during or after ``commit()`` now raises a ``RuntimeError``
   describing the UNKNOWN outcome.  ``_execute_write`` sees a genuine failure
   and does NOT treat it as success.

C. ``is_postgres_retryable`` is dead code
   ``is_postgres_retryable()`` was defined but never called.  Serialization
   failures (40001) and deadlocks (40P01) should retry the WHOLE transaction
   (re-run the fn argument from the top of ``_execute_write``), not an individual
   method.

   Fix: ``_execute_write`` catches ``Exception`` at the end of its retry chain
   and calls ``is_postgres_retryable()`` when ``self._is_postgres`` is True.

All tests are Docker-free.  They mock the psycopg layer with lightweight fakes
so they run in standard CI.
"""

from __future__ import annotations

import time
import types
import unittest.mock as mock
from typing import Any, List, Optional

import pytest


# ---------------------------------------------------------------------------
# Minimal fakes for _PostgresConnection internals
# ---------------------------------------------------------------------------


class _FakePsycopgError(Exception):
    """A fake psycopg error that _is_connection_broken_error() will recognise."""

    def __init__(self, msg="SSL connection has been closed unexpectedly"):
        super().__init__(msg)


class _FakeSerializationError(Exception):
    """Fake psycopg error for SQLSTATE 40001 (serialization_failure)."""

    sqlstate = "40001"


class _FakeDeadlockError(Exception):
    """Fake psycopg error for SQLSTATE 40P01 (deadlock_detected)."""

    sqlstate = "40P01"


class _FakePsycopgCursor:
    def __init__(self, *, fail_on_execute=None, description=None):
        self._fail_on_execute = fail_on_execute
        self._description_rows = description or []
        self.description = [(col,) for col in self._description_rows]
        self._rows: list = []
        self._rowcount = 0

    def execute(self, sql, params=()):
        if self._fail_on_execute:
            raise self._fail_on_execute

    def fetchone(self):
        return None

    def fetchall(self):
        return []

    @property
    def rowcount(self):
        return self._rowcount


class _FakePsycopgConn:
    """Minimal psycopg-conn-shaped object for testing the adapter."""

    def __init__(self, *, fail_commit=False, fail_commit_with=None):
        self._fail_commit = fail_commit
        self._fail_commit_with = fail_commit_with
        self.autocommit = False
        self._cursor_call_count = 0

    def cursor(self):
        self._cursor_call_count += 1
        return _FakePsycopgCursor()

    def execute(self, sql, params=()):
        return _FakePsycopgCursor()

    def commit(self):
        if self._fail_commit:
            raise (self._fail_commit_with or _FakePsycopgError("commit failed"))

    def rollback(self):
        pass

    def close(self):
        pass


# ---------------------------------------------------------------------------
# Test A: execute() must NOT replay a write on a fresh connection
# ---------------------------------------------------------------------------


class TestExecuteFailsClosedOnBrokenConnection:
    """Connection loss during a write must propagate, not silently replay.

    A stale TCP connection is usually discovered during execute() rather than
    at cursor() time, which makes reconnect-and-replay look like the obvious
    repair. It is unsafe:

      * ``_reconnect()`` builds the replacement with ``autocommit=True``, so a
        replayed statement does not belong to the ``BEGIN`` that the enclosing
        ``_execute_write`` closure opened. It self-commits while the closure
        believes it is still transactional, so the closure can report success
        over a partially-applied write.
      * The replacement connection does not hold the transaction-scoped
        advisory locks the original acquired.
      * The first execute's outcome is UNKNOWN when the response is lost — the
        server may already have applied it — so replay can double-apply a
        non-idempotent statement.

    Retry belongs at the whole-closure level in ``_execute_write`` and only for
    known-clean aborts (40001 / 40P01), where the server guarantees the
    transaction applied nothing.
    """

    def test_execute_does_not_replay_on_broken_connection(self):
        """A broken-connection error propagates; the statement is not replayed."""
        from hermes_state_postgres import _PostgresConnection, _PostgresCursor

        call_counts = {"execute": 0}

        class _StaleCursor(_FakePsycopgCursor):
            def execute(self, sql, params=()):
                call_counts["execute"] += 1
                raise _FakePsycopgError()  # stale-connection error

        class _FakeConn2:
            autocommit = False

            def cursor(self):
                return _StaleCursor()

            def commit(self):
                pass

            def rollback(self):
                pass

            def close(self):
                pass

        pg_conn = _PostgresConnection(_FakeConn2(), "postgresql://fake/db")

        reconnects = [0]

        def _fake_reconnect():
            reconnects[0] += 1
            pg_conn._conn = _FakeConn2()

        pg_conn._reconnect = _fake_reconnect
        pg_conn._ensure_live = lambda: None

        cur = _PostgresCursor(_FakeConn2().cursor(), conn=pg_conn)
        cur._cursor = _StaleCursor()

        # The error must surface rather than being swallowed by a replay.
        with pytest.raises(Exception):
            cur.execute(
                "INSERT INTO state_meta (key, value) VALUES (?, ?)", ("k", "v")
            )

        assert call_counts["execute"] == 1, (
            "the statement must be attempted exactly once — replaying a write "
            "on a fresh autocommit connection can self-commit outside the "
            "caller's transaction and double-apply a non-idempotent statement"
        )
        assert reconnects[0] == 0, (
            "cursor-level execute must not reconnect; connection loss during a "
            "write has to fail closed so the caller can decide"
        )

    def test_execute_does_not_retry_non_connection_errors(self):
        """Non-connection errors propagate immediately without a reconnect."""
        from hermes_state_postgres import _PostgresCursor

        class _BadSqlCursor(_FakePsycopgCursor):
            def execute(self, sql, params=()):
                raise ValueError("syntax error near '?'")

        reconnects = [0]

        class _FakeConnRef:
            _conn = _FakePsycopgConn()
            _dsn = "postgresql://fake/db"

            def _reconnect(self):
                reconnects[0] += 1

            def _ensure_live(self):
                pass

        cur = _PostgresCursor(_BadSqlCursor(), conn=_FakeConnRef())

        with pytest.raises(ValueError, match="syntax error"):
            cur.execute("SELECT 1")

        assert reconnects[0] == 0, "must not reconnect on a non-connection error"

    def test_execute_does_not_retry_twice(self):
        """A second broken-connection error on the retry propagates immediately."""
        from hermes_state_postgres import _PostgresCursor

        class _AlwaysBrokenCursor(_FakePsycopgCursor):
            def execute(self, sql, params=()):
                raise _FakePsycopgError("ssl connection has been closed unexpectedly")

        reconnects = [0]

        class _AlwaysBrokenPsycopgConn:
            """Inner conn that always returns a cursor that fails on execute."""
            autocommit = False

            def cursor(self):
                return _AlwaysBrokenCursor()

            def commit(self):
                pass

            def rollback(self):
                pass

            def close(self):
                pass

        class _FakeConnRef:
            _dsn = "postgresql://fake/db"
            _conn = _AlwaysBrokenPsycopgConn()

            def _reconnect(self):
                reconnects[0] += 1
                # After reconnect, _conn stays broken (simulate persistent failure)

            def _ensure_live(self):
                pass

        conn_ref = _FakeConnRef()
        cur = _PostgresCursor(_AlwaysBrokenCursor(), conn=conn_ref)

        with pytest.raises(_FakePsycopgError):
            cur.execute("SELECT 1")

        # No reconnect at cursor level: the failure propagates on the first
        # attempt so the caller (_execute_write) owns the retry decision.
        assert reconnects[0] == 0


# ---------------------------------------------------------------------------
# Test B: commit() loss → unknown outcome, NOT synthetic success
# ---------------------------------------------------------------------------


class TestCommitLossIsUnknownOutcome:
    """Connection loss during or after commit must never report success."""

    def test_commit_raises_on_connection_loss(self):
        """_PostgresConnection.commit() raises RuntimeError on dropped connection,
        not an empty-transaction success.
        """
        from hermes_state_postgres import _PostgresConnection

        conn_error = _FakePsycopgError("SSL connection has been closed unexpectedly")

        pg_conn = _PostgresConnection(
            _FakePsycopgConn(fail_commit=True, fail_commit_with=conn_error), "postgresql://fake/db")
        pg_conn._ensure_live = lambda: None

        with pytest.raises(RuntimeError, match="UNKNOWN"):
            pg_conn.commit()

    def test_commit_does_not_reconnect_or_retry(self):
        """commit() must fail closed — no reconnect, no retry commit on new conn."""
        from hermes_state_postgres import _PostgresConnection

        conn_error = _FakePsycopgError("server closed the connection unexpectedly")
        reconnects = [0]

        class _TrackingConn(_FakePsycopgConn):
            def commit(self):
                raise conn_error

        pg_conn = _PostgresConnection(_TrackingConn(), "postgresql://fake/db")
        pg_conn._ensure_live = lambda: None

        original_reconnect = getattr(pg_conn, "_reconnect", None)

        def _spy_reconnect():
            reconnects[0] += 1
            if original_reconnect:
                original_reconnect()

        pg_conn._reconnect = _spy_reconnect

        with pytest.raises(RuntimeError):
            pg_conn.commit()

        assert reconnects[0] == 0, "commit() must NOT reconnect on connection loss"

    def test_non_connection_commit_error_propagates_unchanged(self):
        """A non-connection error from commit() (e.g. constraint) propagates as-is."""
        from hermes_state_postgres import _PostgresConnection

        class _ConstraintError(Exception):
            sqlstate = "23505"  # unique_violation — NOT a connection error

        pg_conn = _PostgresConnection(
            _FakePsycopgConn(fail_commit=True, fail_commit_with=_ConstraintError("unique violation")),
            "postgresql://fake/db")
        pg_conn._ensure_live = lambda: None

        with pytest.raises(_ConstraintError):
            pg_conn.commit()


# ---------------------------------------------------------------------------
# Test C: is_postgres_retryable wired into _execute_write
# ---------------------------------------------------------------------------


class TestIsPostgresRetryableWired:
    """Serialization failures and deadlocks retry the whole transaction fn."""

    def _make_session_db_postgres(self, tmp_path):
        """Return a minimally-wired SessionDB instance set to is_postgres mode."""
        import hermes_state

        db = hermes_state.SessionDB(tmp_path / "state.db")
        db._conn.close()
        db._conn = None
        db._is_postgres = True
        db._wal_active = db._fts_enabled = False
        db._WRITE_PATIENCE_S = 5.0
        return db

    def _make_fake_conn(self):
        """Return a minimal fake connection that _execute_write can call."""
        conn = mock.MagicMock()
        conn.execute.return_value = None
        conn.commit.return_value = None
        conn.rollback.return_value = None
        # Make execute("BEGIN IMMEDIATE") work (adapter translates to BEGIN)
        return conn

    def test_serialization_failure_retries_whole_fn(self, tmp_path):
        """40001 serialization failure retries fn from the top of _execute_write."""
        db = self._make_session_db_postgres(tmp_path)
        fake_conn = self._make_fake_conn()
        db._conn = fake_conn

        call_counts = {"fn": 0}
        results = []

        def _failing_fn(conn):
            call_counts["fn"] += 1
            if call_counts["fn"] == 1:
                raise _FakeSerializationError("could not serialize access")
            results.append("ok")
            return "done"

        outcome = db._execute_write(_failing_fn)

        assert outcome == "done", "should succeed on retry"
        assert call_counts["fn"] == 2, "fn must be called twice (fail + retry)"

    def test_deadlock_retries_whole_fn(self, tmp_path):
        """40P01 deadlock detected retries fn from the top of _execute_write."""
        db = self._make_session_db_postgres(tmp_path)
        fake_conn = self._make_fake_conn()
        db._conn = fake_conn

        call_counts = {"fn": 0}

        def _failing_fn(conn):
            call_counts["fn"] += 1
            if call_counts["fn"] == 1:
                raise _FakeDeadlockError("deadlock detected")
            return "done"

        outcome = db._execute_write(_failing_fn)

        assert outcome == "done"
        assert call_counts["fn"] == 2

    def test_non_retryable_postgres_error_propagates(self, tmp_path):
        """A non-retryable Postgres error (e.g. 23505) propagates immediately."""
        db = self._make_session_db_postgres(tmp_path)
        fake_conn = self._make_fake_conn()
        db._conn = fake_conn

        class _UniqueViolation(Exception):
            sqlstate = "23505"

        call_counts = {"fn": 0}

        def _failing_fn(conn):
            call_counts["fn"] += 1
            raise _UniqueViolation("unique constraint violated")

        with pytest.raises(_UniqueViolation):
            db._execute_write(_failing_fn)

        assert call_counts["fn"] == 1, "non-retryable error must not retry"

    def test_serialization_retry_exhaustion_propagates(self, tmp_path):
        """When patience runs out, the serialization error propagates."""
        db = self._make_session_db_postgres(tmp_path)
        # Very short patience so the loop exits fast in the test
        db._WRITE_PATIENCE_S = 0.05
        db._WRITE_RETRY_SLOW_AFTER_S = 0.01
        db._WRITE_RETRY_MIN_S = 0.001
        db._WRITE_RETRY_MAX_S = 0.005
        db._WRITE_RETRY_SLOW_MIN_S = 0.005
        db._WRITE_RETRY_SLOW_MAX_S = 0.01

        fake_conn = self._make_fake_conn()
        db._conn = fake_conn

        def _always_fails(conn):
            raise _FakeSerializationError("always serialization failure")

        with pytest.raises(_FakeSerializationError):
            db._execute_write(_always_fails)

    def test_sqlite_is_postgres_false_does_not_catch_generic_exceptions(self, tmp_path):
        """SQLite writes propagate unrelated errors from their transaction callback."""
        import hermes_state

        db = hermes_state.SessionDB(tmp_path / "state.db")

        class _RandomError(Exception):
            pass

        def _failing_fn(conn):
            raise _RandomError("unrelated runtime error")

        try:
            with pytest.raises(_RandomError):
                db._execute_write(_failing_fn)
        finally:
            db.close()


# ---------------------------------------------------------------------------
# is_postgres_retryable unit tests
# ---------------------------------------------------------------------------


class TestIsPostgresRetryable:
    """Unit tests for the predicate itself."""

    def test_40001_by_sqlstate(self):
        from hermes_state_postgres import is_postgres_retryable

        exc = _FakeSerializationError()
        assert is_postgres_retryable(exc)

    def test_40p01_by_sqlstate(self):
        from hermes_state_postgres import is_postgres_retryable

        exc = _FakeDeadlockError()
        assert is_postgres_retryable(exc)

    def test_serialization_failure_by_message(self):
        from hermes_state_postgres import is_postgres_retryable

        class _BareError(Exception):
            pass

        # The message-substring fallback checks for "serialization failure"
        # (as psycopg formats it when sqlstate is not exposed on the object).
        exc = _BareError("ERROR:  serialization failure detected on concurrent write")
        assert is_postgres_retryable(exc)

    def test_deadlock_detected_by_message(self):
        from hermes_state_postgres import is_postgres_retryable

        class _BareError(Exception):
            pass

        exc = _BareError("ERROR:  deadlock detected\nDETAIL:  Process 123 waits...")
        assert is_postgres_retryable(exc)

    def test_other_sqlstate_not_retryable(self):
        from hermes_state_postgres import is_postgres_retryable

        class _UniqueViolation(Exception):
            sqlstate = "23505"

        exc = _UniqueViolation("duplicate key value")
        assert not is_postgres_retryable(exc)

    def test_non_postgres_exception_not_retryable(self):
        from hermes_state_postgres import is_postgres_retryable

        exc = ValueError("not a database error at all")
        assert not is_postgres_retryable(exc)
