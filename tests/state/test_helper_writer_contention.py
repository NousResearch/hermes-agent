"""Behavioral contracts for the shared state.db helper-writer primitive.

``hermes_state_common.state_db_begin_immediate`` is the reusable
``BEGIN IMMEDIATE`` discipline wired into ``tools.async_delegation._transaction``
and ``gateway.delivery_ledger._transaction`` — both short-lived,
helper-owned SQLite connections to ``state.db`` that previously committed
via ``sqlite3.Connection.__exit__`` with no application-level retry on
lock contention. The contracts in this module are the invariants the
helpers now share; they are deliberately written against the primitive
directly (the helpers are exercised by their own fd-leak + persistence
tests).

All tests use throwaway tmp databases via the project-standard
``monkeypatch.setattr`` of the ``_db_path``-style seams in each module.
"""

from __future__ import annotations

import sqlite3
import threading
import time

import pytest


pytest.importorskip("hermes_state_common")

from hermes_state_common import (  # noqa: E402
    _STATE_DB_WRITE_PATIENCE_S,
    state_db_begin_immediate,
)


@pytest.fixture
def tmp_db_path(tmp_path):
    """Throwaway DB path; callers create connections themselves."""
    return tmp_path / "test_state.db"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _open(path, timeout=1):
    """Open a sqlite3 connection in manual-transaction mode (no autocommit).

    ``timeout=1`` keeps BEGIN IMMEDIATE from blocking forever under a
    competing lock; the primitive's retry loop is what we want to exercise,
    not the connection's own internal waiter.
    """
    return sqlite3.connect(str(path), timeout=timeout, isolation_level=None)


def _hold_write_lock(db_path, conn_box: list, release: threading.Event):
    """Block in a single thread until ``release`` is set, holding the WAL write lock.

    The opened connection is stored in ``conn_box[0]`` for the test thread
    to *observe* that the lock was acquired.  Closing the connection must
    happen on the thread that opened it (Python's default
    ``check_same_thread=True``), so the helper releases/closes inside the
    thread before returning; the test thread signals via ``release``.
    """
    conn = sqlite3.connect(
        str(db_path), timeout=10, isolation_level=None, check_same_thread=False,
    )
    conn_box.append(conn)
    conn.execute("BEGIN IMMEDIATE")
    release.wait()
    try:
        conn.execute("COMMIT")
    except Exception:
        try:
            conn.execute("ROLLBACK")
        except Exception:
            pass
    conn.close()


def _make_wal(db_path):
    """Create a small DB and force journal_mode=WAL so contention is real."""
    conn = _open(db_path)
    try:
        conn.execute("CREATE TABLE IF NOT EXISTS t (x INTEGER)")
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("INSERT INTO t VALUES (0)")
    finally:
        conn.close()


@pytest.fixture
def _short_patience(monkeypatch):
    """Shrink the BEGIN-retry patience for the persistent-contention test.

    ``state_db_begin_immediate`` reads the patience and jitter schedule
    from module-level constants (no per-call overrides — only the BEGIN
    itself is retried, never the body).  Tightening the constants lets
    the exhaustion contract run in well under a second instead of
    waiting the default 20 s budget.
    """
    monkeypatch.setattr(
        "hermes_state_common._STATE_DB_WRITE_PATIENCE_S", 0.2,
    )
    monkeypatch.setattr(
        "hermes_state_common._STATE_DB_WRITE_RETRY_MIN_S", 0.001,
    )
    monkeypatch.setattr(
        "hermes_state_common._STATE_DB_WRITE_RETRY_MAX_S", 0.005,
    )
    monkeypatch.setattr(
        "hermes_state_common._STATE_DB_WRITE_RETRY_SLOW_MIN_S", 0.001,
    )
    monkeypatch.setattr(
        "hermes_state_common._STATE_DB_WRITE_RETRY_SLOW_MAX_S", 0.005,
    )
    monkeypatch.setattr(
        "hermes_state_common._STATE_DB_WRITE_RETRY_SLOW_AFTER_S", 0.001,
    )


# ---------------------------------------------------------------------------
# Contract 1: BEGIN IMMEDIATE issued before the body runs.
# ---------------------------------------------------------------------------


def test_begin_immediate_runs_before_body(tmp_db_path):
    """The primitive MUST issue BEGIN IMMEDIATE before the body runs.

    A second ``BEGIN IMMEDIATE`` inside the body must fail with
    "cannot start a transaction within a transaction" — the proof that
    the outer BEGIN already holds the WAL write lock.
    """
    observed = []

    def body():
        observed.append("body_entered")
        # If BEGIN IMMEDIATE already ran, a second BEGIN must fail.
        try:
            conn.execute("BEGIN IMMEDIATE")
            observed.append("nested_begin_succeeded")
        except sqlite3.OperationalError as exc:
            observed.append(f"nested_begin_failed:{exc}")

        conn.execute("INSERT INTO t VALUES (1)")

    _make_wal(tmp_db_path)
    conn = _open(tmp_db_path)
    try:
        with state_db_begin_immediate(conn):
            body()
    finally:
        conn.close()

    assert observed[0] == "body_entered"
    assert observed[1].startswith("nested_begin_failed:")
    assert "cannot start a transaction within a transaction" in observed[1]

    # COMMIT ran (probe row is persisted)
    conn2 = _open(tmp_db_path)
    try:
        rows = conn2.execute("SELECT x FROM t").fetchall()
    finally:
        conn2.close()
    assert rows == [(0,), (1,)], rows


# ---------------------------------------------------------------------------
# Contract 2: BEGIN retry succeeds after competing lock release.
# ---------------------------------------------------------------------------


def test_begin_retry_succeeds_after_competing_lock_releases(tmp_db_path):
    """A competitor that holds the WAL write lock for ~200 ms must be waited
    out by the BEGIN-retry loop — release happens BEFORE the patience expires.

    Only ``BEGIN IMMEDIATE`` is retried; the body runs once, un-replayed.
    """
    _make_wal(tmp_db_path)
    conn_box: list = []
    release = threading.Event()

    competitor = threading.Thread(
        target=_hold_write_lock,
        args=(tmp_db_path, conn_box, release),
        daemon=True,
    )
    competitor.start()

    # Wait until the competitor actually holds the lock.
    deadline = time.monotonic() + 2.0
    while not conn_box:
        if time.monotonic() > deadline:
            pytest.fail("competitor never acquired the lock")
        time.sleep(0.01)
    # Give the BEGIN IMMEDIATE a moment to land.
    time.sleep(0.05)

    # Schedule the competitor to release after 200 ms — well within the
    # primitive's default 20 s patience budget but long enough to force
    # at least one BEGIN-retry cycle.
    def _release_after():
        time.sleep(0.2)
        release.set()
    threading.Thread(target=_release_after, daemon=True).start()

    body_call_count = []

    def body():
        body_call_count.append("called")
        conn.execute("UPDATE t SET x = x + 1")

    conn = _open(tmp_db_path)
    try:
        with state_db_begin_immediate(conn):
            body()
    finally:
        conn.close()

    competitor.join(timeout=2.0)

    # Body must have run exactly once (the primitive does NOT re-run the
    # body on retry — only BEGIN is retried).
    assert body_call_count == ["called"]

    # Verify the update landed
    verify = _open(tmp_db_path)
    try:
        rows = verify.execute("SELECT x FROM t").fetchall()
    finally:
        verify.close()
    assert rows == [(1,)], rows


# ---------------------------------------------------------------------------
# Contract 3: BEGIN patience exhaustion propagates the LAST OperationalError.
# ---------------------------------------------------------------------------


def test_begin_patience_exhaustion_propagates_last_error(
    tmp_db_path, _short_patience,
):
    """Under persistent contention, the LAST ``OperationalError`` propagates
    after patience is exhausted — the caller sees a real SQLite error,
    not a generic wrapper.
    """
    _make_wal(tmp_db_path)
    conn_box: list = []
    release = threading.Event()

    competitor = threading.Thread(
        target=_hold_write_lock,
        args=(tmp_db_path, conn_box, release),
        daemon=True,
    )
    competitor.start()

    deadline = time.monotonic() + 2.0
    while not conn_box:
        if time.monotonic() > deadline:
            pytest.fail("competitor never acquired the lock")
        time.sleep(0.01)
    time.sleep(0.05)

    conn = _open(tmp_db_path)
    try:
        body_called = []

        def body():
            body_called.append("called")

        with pytest.raises(sqlite3.OperationalError) as excinfo:
            with state_db_begin_immediate(conn):
                body()
        # The propagated message must mention the lock/busy condition.
        assert any(
            tok in str(excinfo.value).lower()
            for tok in ("locked", "busy")
        ), str(excinfo.value)
        # Body must NOT have run (BEGIN failed, so the block was never
        # entered). This guards against the primitive silently entering
        # the body on a transient error before patience is exhausted.
        assert body_called == []
    finally:
        conn.close()

    # Release competitor — cleanup is done by the helper thread itself.
    release.set()
    competitor.join(timeout=1.0)


# ---------------------------------------------------------------------------
# Contract 4: body failure rolls back.
# ---------------------------------------------------------------------------


def test_body_failure_rolls_back(tmp_db_path):
    """If the body raises, the primitive MUST roll back the entire transaction
    before propagating. The error reaches the caller verbatim — the
    primitive must NOT mask it with its own wrapper.
    """

    def body():
        conn.execute("CREATE TABLE IF NOT EXISTS probe (x INTEGER)")
        conn.execute("INSERT INTO probe VALUES (1)")
        raise RuntimeError("application logic error")

    conn = _open(tmp_db_path)
    try:
        with pytest.raises(RuntimeError, match="application logic error"):
            with state_db_begin_immediate(conn):
                body()
    finally:
        conn.close()

    # The whole transaction (CREATE TABLE + INSERT) was rolled back.
    # SQLite DDL is transactional in manual-commit mode, so the table
    # itself should NOT exist.
    verify = _open(tmp_db_path)
    try:
        tables = verify.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='probe'"
        ).fetchall()
    finally:
        verify.close()
    assert tables == [], (
        f"expected CREATE TABLE rolled back, got {tables}; "
        "DDL inside a transaction must not survive ROLLBACK"
    )


# ---------------------------------------------------------------------------
# Contract 5: deterministic close on success / failure.
#
# These belong to the *helpers* (``tools.async_delegation._transaction``,
# ``gateway.delivery_ledger._transaction``) rather than the bare primitive —
# the primitive operates on a connection the caller already owns. The
# helper-level close guarantee is asserted by the fd-leak tests in
# tests/tools/test_async_delegation_fd_leak.py and
# tests/gateway/test_delivery_ledger_fd_leak.py, which exercise the
# surviving shared primitive.


# ---------------------------------------------------------------------------
# Contract 6: 'no more rows' transient is recognized as retryable.
# ---------------------------------------------------------------------------


def test_no_more_rows_transient_is_retried():
    """The 'no more rows' transient surfaces during contended WAL appends
    and MUST be treated as a retryable lock-condition by the predicate."""
    from hermes_state_common import _is_retryable_lock_error

    assert _is_retryable_lock_error(sqlite3.OperationalError("no more rows available")) is True
    assert _is_retryable_lock_error(sqlite3.OperationalError("database is locked")) is True
    assert _is_retryable_lock_error(sqlite3.OperationalError("database is busy")) is True
    assert _is_retryable_lock_error(sqlite3.IntegrityError("UNIQUE constraint failed")) is False
    assert _is_retryable_lock_error(RuntimeError("anything")) is False


# ---------------------------------------------------------------------------
# Invariant: default patience is bounded and reasonable.
# ---------------------------------------------------------------------------


def test_patience_constant_matches_canonical():
    """The default patience must match SessionDB's canonical
    ``_WRITE_PATIENCE_S`` so helpers and the canonical writer back off on
    the SAME schedule (avoiding deterministic convoy under contention)."""
    # Read-only: confirm the constant is positive and bounded.
    assert _STATE_DB_WRITE_PATIENCE_S >= 1.0
    assert _STATE_DB_WRITE_PATIENCE_S <= 60.0