"""POSIX advisory locks must survive Hermes' own database inspection.

close() on ANY file descriptor for a SQLite database cancels every POSIX
advisory lock the process holds on that file -- including a running VACUUM's
EXCLUSIVE lock and an in-flight BEGIN IMMEDIATE's RESERVED lock:

    https://sqlite.org/howtocorrupt.html#_posix_advisory_locks_canceled_by_a_separate_thread_doing_close_

Hermes used to byte-probe live databases in several places (kanban's
post-commit page-count check, the zeroed-state.db detector run on every
SessionDB construction, backup header verification). Under `hermes sessions
optimize` this let an external process write into a database while VACUUM was
rewriting it, producing "database disk image is malformed".

These tests pin the behavioural contract: an external process must stay locked
out across Hermes' inspection calls.
"""

from __future__ import annotations

import sqlite3
import subprocess
import sys
import textwrap

import pytest

from hermes_cli.sqlite_safe_read import (
    file_length_matches_header,
    has_live_connection,
    page_count_bytes,
    read_header_bytes_preopen,
    track_connection,
    untrack_connection,
)


_INTRUDER = textwrap.dedent(
    """
    import sqlite3, sys
    conn = sqlite3.connect(sys.argv[1], isolation_level=None, timeout=0)
    try:
        conn.execute("PRAGMA busy_timeout=0")
        conn.execute("BEGIN IMMEDIATE")
        conn.execute("INSERT INTO t(v) VALUES ('intruder')")
        conn.execute("COMMIT")
        print("ACQUIRED")
    except sqlite3.OperationalError:
        print("BLOCKED")
    """
)


def _external_writer_can_break_in(db_path) -> bool:
    """True when a separate process managed to write to a locked database."""
    result = subprocess.run(
        [sys.executable, "-c", _INTRUDER, str(db_path)],
        capture_output=True,
        text=True,
        timeout=60,
    )
    return "ACQUIRED" in result.stdout


def _make_db(path, journal_mode: str) -> None:
    conn = sqlite3.connect(str(path), isolation_level=None)
    conn.execute(f"PRAGMA journal_mode={journal_mode}")
    conn.execute("CREATE TABLE t(v TEXT)")
    conn.executemany("INSERT INTO t(v) VALUES (?)", [(f"row{i}",) for i in range(200)])
    conn.close()


@pytest.fixture
def clean_registry():
    yield
    # Keep the module-level registry from leaking across tests.
    import hermes_cli.sqlite_safe_read as mod

    with mod._live_lock:
        mod._live_connections.clear()


@pytest.mark.parametrize("journal_mode", ["DELETE", "WAL"])
def test_write_lock_survives_file_length_check(tmp_path, journal_mode, clean_registry):
    """kanban's post-commit invariant check must not cancel the write lock."""
    from hermes_cli.kanban_db import _check_file_length_invariant

    db = tmp_path / "kanban.db"
    _make_db(db, journal_mode)

    holder = sqlite3.connect(str(db), isolation_level=None, timeout=0.5)
    holder.execute(f"PRAGMA journal_mode={journal_mode}")
    holder.execute("BEGIN IMMEDIATE")
    holder.execute("INSERT INTO t(v) VALUES ('held')")
    try:
        assert not _external_writer_can_break_in(db), (
            "precondition failed: the external writer was not locked out "
            "before the inspection call"
        )

        worker = sqlite3.connect(str(db), isolation_level=None, timeout=0.5)
        try:
            _check_file_length_invariant(worker)
        finally:
            worker.close()

        assert not _external_writer_can_break_in(db), (
            "_check_file_length_invariant cancelled this process's POSIX "
            "advisory locks -- an external process wrote into a database "
            "that a writer still believed it held exclusively"
        )
    finally:
        holder.close()


@pytest.mark.parametrize("journal_mode", ["DELETE", "WAL"])
def test_write_lock_survives_zeroed_state_db_probe(
    tmp_path, journal_mode, clean_registry
):
    """SessionDB's zeroed-file detector must not cancel locks once connected."""
    from hermes_state import is_zeroed_state_db

    db = tmp_path / "state.db"
    _make_db(db, journal_mode)

    holder = sqlite3.connect(str(db), isolation_level=None, timeout=0.5)
    holder.execute(f"PRAGMA journal_mode={journal_mode}")
    holder.execute("BEGIN IMMEDIATE")
    holder.execute("INSERT INTO t(v) VALUES ('held')")
    track_connection(db)
    try:
        assert not _external_writer_can_break_in(db)

        assert is_zeroed_state_db(db) is False

        assert not _external_writer_can_break_in(db), (
            "is_zeroed_state_db cancelled this process's POSIX advisory "
            "locks on a live database"
        )
    finally:
        untrack_connection(db)
        holder.close()


def test_preopen_read_refused_while_connection_is_live(tmp_path, clean_registry):
    """The byte-level probe is allowed pre-open and refused once connected."""
    db = tmp_path / "state.db"
    _make_db(db, "WAL")

    assert not has_live_connection(db)
    head = read_header_bytes_preopen(db, length=16)
    assert head == b"SQLite format 3\x00"

    track_connection(db)
    try:
        assert has_live_connection(db)
        assert read_header_bytes_preopen(db, length=16) is None
        # An explicit override stays available for offline artifacts.
        assert read_header_bytes_preopen(db, length=16, force=True) is not None
    finally:
        untrack_connection(db)

    assert not has_live_connection(db)
    assert read_header_bytes_preopen(db, length=16) is not None


def test_tracking_registry_does_not_leak_across_close_paths(tmp_path, clean_registry):
    """A drifting counter would silently disable the probe guard forever.

    Opens are easy to count; closes happen in many places. If the registry
    ever over-counts, ``has_live_connection`` stays true for a path with no
    live connection and every later byte-probe is refused — turning the
    safety guard into a permanent outage of zeroed-file / header detection.
    """
    import contextlib

    from hermes_cli.sqlite_safe_read import connect_tracked

    db = tmp_path / "state.db"
    boot = connect_tracked(db, isolation_level=None)
    boot.execute("CREATE TABLE t(v TEXT)")
    boot.close()
    assert not has_live_connection(db)

    # plain close
    connect_tracked(db).close()
    assert not has_live_connection(db)

    # contextlib.closing
    with contextlib.closing(connect_tracked(db)):
        assert has_live_connection(db)
    assert not has_live_connection(db)

    # `with conn:` is a TRANSACTION scope, not a close — must stay tracked
    conn = connect_tracked(db, isolation_level=None)
    with conn:
        conn.execute("INSERT INTO t(v) VALUES ('x')")
    assert has_live_connection(db), "transaction scope must not untrack"
    conn.close()
    assert not has_live_connection(db)

    # double close is idempotent (must not under-count into negatives)
    dup = connect_tracked(db)
    dup.close()
    dup.close()
    assert not has_live_connection(db)

    # nested lifetimes: still live until the last one closes
    first = connect_tracked(db)
    second = connect_tracked(db)
    first.close()
    assert has_live_connection(db)
    second.close()
    assert not has_live_connection(db)

    # churn must not drift
    for _ in range(100):
        connect_tracked(db).close()
    assert not has_live_connection(db)


def test_caller_supplied_connection_factory_still_works(tmp_path, clean_registry):
    """A caller's own factory wins; tracking is skipped rather than crashing.

    Tracking is an optimisation for the probe guard, never a precondition for
    opening the database — passing a custom factory must not raise.
    """
    from hermes_cli.sqlite_safe_read import connect_tracked

    class CustomConnection(sqlite3.Connection):
        pass

    db = tmp_path / "state.db"
    _make_db(db, "WAL")

    conn = connect_tracked(db, factory=CustomConnection)
    try:
        assert isinstance(conn, CustomConnection)
        assert conn.execute("SELECT COUNT(*) FROM t").fetchone()[0] == 200
    finally:
        conn.close()


def test_page_count_bytes_matches_on_disk_size(tmp_path):
    """The PRAGMA route reports the same size the header field encodes."""
    db = tmp_path / "state.db"
    _make_db(db, "DELETE")

    conn = sqlite3.connect(str(db))
    try:
        logical = page_count_bytes(conn)
        assert logical is not None
        assert logical == db.stat().st_size
        assert file_length_matches_header(conn) is True
    finally:
        conn.close()


def test_file_length_check_never_reports_truncated_db_as_healthy(tmp_path):
    """A short file must not come back as a clean 'file length matches'.

    On a truncated database SQLite refuses the pragma outright, so the helper
    returns None (inconclusive) rather than False. Either way the contract that
    matters is the same: a torn file is never reported as healthy.
    """
    db = tmp_path / "state.db"
    _make_db(db, "DELETE")

    conn = sqlite3.connect(str(db))
    try:
        logical = page_count_bytes(conn)
        assert logical is not None
        assert file_length_matches_header(conn) is True

        # Truncate behind SQLite's back to simulate a torn extend.
        with open(db, "r+b") as handle:
            handle.truncate(logical // 2)

        assert file_length_matches_header(conn) is not True
    finally:
        conn.close()
