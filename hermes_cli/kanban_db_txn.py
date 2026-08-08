"""Kanban DB transaction primitives (extracted verbatim from kanban_db.py).

The transaction spine of the kanban board: ``write_txn`` (BEGIN IMMEDIATE /
COMMIT with a jittered BUSY retry and a COMMIT-exhaustion ROLLBACK so the
connection is never poisoned for the next transaction), the torn-extend
file-length invariant, and the busy-error classifier. Bodies are
byte-identical to the source window (kanban_db.py 2731-2838); the
godfile-owned ``_assert_not_delegated_child_mutation`` guard is reached via
a lazy in-body import so ``hermes_cli.kanban_db`` never forms a top-level
import cycle (it re-exports these names at its tail).
"""

import contextlib
import random
import sqlite3
import time


def _check_file_length_invariant(conn: sqlite3.Connection) -> None:
    """Compare SQLite's own page accounting against the file size on disk.

    Raises sqlite3.DatabaseError if the file is shorter than the header claims
    (torn-extend corruption).

    Both sides are read WITHOUT opening the database file. The header side
    comes from ``PRAGMA page_count`` over the existing connection; the on-disk
    side from ``stat()``. An earlier version read the header field with a bare
    ``open(path,"rb")`` -- but ``close()`` cancels every POSIX advisory lock
    this process holds on the file, so that probe silently dropped the locks
    of concurrent writers (and of a running VACUUM) and let other processes
    write into a database a writer still believed it owned. That is the
    documented corruption route in sqlite.org/howtocorrupt.html section 2.2.
    """
    from hermes_cli.sqlite_safe_read import file_length_matches_header

    # In WAL mode a just-committed page can still live in the -wal file, so
    # the main file legitimately lags its page count. Only enforce the
    # invariant under a rollback journal, where every committed page must
    # already be in the main file.
    try:
        row = conn.execute("PRAGMA journal_mode").fetchone()
        journal_mode = str(row[0]).lower() if row and row[0] is not None else ""
    except sqlite3.Error:
        return
    if journal_mode == "wal":
        return

    ok = file_length_matches_header(conn)
    if ok is False:
        raise sqlite3.DatabaseError(
            "torn-extend detected: the database file is shorter than its "
            "header page count claims"
        )


# SQLite's own busy_timeout uses a near-deterministic backoff, so concurrent
# writers re-collide in lockstep under a stampede. A jittered retry on the
# transaction boundary breaks that convoy. Mirrors state.db's _execute_write:
# a fixed 20-150ms jitter band (a 20ms floor prevents a near-zero retry from
# busy-spinning back into the collision). Only BEGIN IMMEDIATE and COMMIT are
# retried -- both are idempotent re-issues that touch no transaction body, so a
# CAS inside write_txn is never replayed. kanban keeps fewer retries than
# state.db (5 vs 15) because its 120s busy_timeout already absorbs most waits;
# the retry is the backstop for the tail SQLite returns BUSY on immediately.
_BUSY_MAX_RETRIES = 5
_BUSY_RETRY_MIN_S = 0.020  # 20ms
_BUSY_RETRY_MAX_S = 0.150  # 150ms


def _is_busy_error(exc: BaseException) -> bool:
    return isinstance(exc, sqlite3.OperationalError) and (
        "database is locked" in str(exc).lower()
        or "database is busy" in str(exc).lower()
    )


def _execute_boundary_with_retry(conn: sqlite3.Connection, sql: str) -> None:
    for attempt in range(_BUSY_MAX_RETRIES + 1):
        try:
            conn.execute(sql)
            return
        except sqlite3.OperationalError as exc:
            if not _is_busy_error(exc) or attempt == _BUSY_MAX_RETRIES:
                raise
            time.sleep(random.uniform(_BUSY_RETRY_MIN_S, _BUSY_RETRY_MAX_S))


@contextlib.contextmanager
def write_txn(conn: sqlite3.Connection):
    """Context manager for an IMMEDIATE write transaction.

    Use for any multi-statement write (creating a task + link, claiming a
    task + recording an event, etc.).  A claim CAS inside this context is
    atomic -- at most one concurrent writer can succeed.

    The explicit ROLLBACK on exception is wrapped in try/except so that
    a SQLite auto-rollback (which leaves no active transaction) does not
    shadow the original exception with a spurious rollback error.
    """
    from hermes_cli.kanban_db import _assert_not_delegated_child_mutation

    _assert_not_delegated_child_mutation()
    _execute_boundary_with_retry(conn, "BEGIN IMMEDIATE")
    try:
        yield conn
    except Exception:
        try:
            conn.execute("ROLLBACK")
        except sqlite3.OperationalError:
            # SQLite has already auto-rolled-back the transaction (typical
            # under EIO, lock contention, or corruption). Nothing to undo;
            # do not let this secondary failure shadow the real one.
            pass
        raise
    else:
        try:
            _execute_boundary_with_retry(conn, "COMMIT")
        except Exception:
            # COMMIT exhausted retries with the txn still open; roll back so the
            # connection isn't poisoned for the next BEGIN IMMEDIATE.
            try:
                conn.execute("ROLLBACK")
            except sqlite3.OperationalError:
                pass
            raise
        # Post-commit file-length check: header page_count must match actual file pages.
        # A discrepancy means a torn-extend — raise now rather than silently corrupt.
        # Lazy import so monkeypatching kanban_db._check_file_length_invariant
        # (as test_kanban_write_txn_busy_retry.py does) keeps intercepting.
        from hermes_cli.kanban_db import _check_file_length_invariant

        _check_file_length_invariant(conn)
