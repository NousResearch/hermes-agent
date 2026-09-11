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
import threading

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


def test_failed_close_keeps_connection_tracked(tmp_path, clean_registry):
    """A raising close must not release the registry (#75629).

    Untracking before ``super().close()`` leaves the descriptor open while
    ``has_live_connection`` reports false, so the byte-probe guard permits
    ``open``/``close`` on a live database — cancelling POSIX advisory locks.
    """
    from hermes_cli.sqlite_safe_read import connect_tracked

    class ControllableConnection(sqlite3.Connection):
        def close(self):
            if getattr(self, "_hermes_fail_close", False):
                raise sqlite3.ProgrammingError(
                    "SQLite objects created in a thread can only be used in "
                    "that same thread"
                )
            return super().close()

    db = tmp_path / "state.db"
    _make_db(db, "WAL")

    conn = connect_tracked(db, factory=ControllableConnection)
    assert has_live_connection(db)
    assert read_header_bytes_preopen(db, length=16) is None

    conn._hermes_fail_close = True
    with pytest.raises(sqlite3.ProgrammingError):
        conn.close()

    assert has_live_connection(db), "failed close must leave the registry entry"
    assert read_header_bytes_preopen(db, length=16) is None

    conn._hermes_fail_close = False
    conn.close()
    assert not has_live_connection(db)
    assert read_header_bytes_preopen(db, length=16) is not None


def test_probe_and_connect_do_not_race(tmp_path, clean_registry, monkeypatch):
    """The check and the raw read must be atomic w.r.t. connection lifecycle.

    Deterministic interleaving: pause the probe *inside* the byte read — after
    it has decided "nothing is live" and opened its descriptor — then let
    another thread open a tracked connection and take a write lock, then let
    the probe close.

    With the guard holding ``_live_lock`` across check+open+read+close, the
    other thread blocks on that lock until the probe is done, so no
    interleaving is possible. If the lock is only held across the check, that
    thread slips in and the probe's ``close()`` cancels its POSIX locks.
    """
    import hermes_cli.sqlite_safe_read as ssr

    db = tmp_path / "state.db"
    _make_db(db, "DELETE")

    inside_read = threading.Event()
    may_close = threading.Event()
    failures: list[str] = []
    real_open = open

    def slow_open(*args, **kwargs):
        handle = real_open(*args, **kwargs)
        inside_read.set()
        # Hold the descriptor open while the writer tries to get in.
        may_close.wait(10)
        return handle

    def writer():
        inside_read.wait(10)
        # If the guard is correct this blocks until the probe releases the
        # lock; if not, it opens and locks inside the probe's window.
        conn = ssr.connect_tracked(db, isolation_level=None, timeout=0.5)
        try:
            conn.execute("BEGIN IMMEDIATE")
            conn.execute("INSERT INTO t(v) VALUES ('holder')")
            may_close.set()  # let the probe's close() land
            if _external_writer_can_break_in(db):
                failures.append(
                    "external writer broke in: the probe's close() cancelled "
                    "this connection's POSIX locks"
                )
            conn.execute("COMMIT")
        except sqlite3.Error as exc:  # a cancelled lock surfaces here too
            failures.append(f"holder transaction failed: {exc}")
        finally:
            conn.close()

    monkeypatch.setattr(ssr, "open", slow_open, raising=False)
    t = threading.Thread(target=writer, daemon=True)
    t.start()
    try:
        read_header_bytes_preopen(db, length=16)
    finally:
        may_close.set()  # never wedge the probe if the writer died
    t.join(20)

    assert not failures, failures[0]




def test_session_db_read_only_is_tracked(tmp_path, clean_registry, monkeypatch):
    """End-to-end: a real read-only SessionDB blocks byte-probes."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    from hermes_state import SessionDB

    db_path = tmp_path / "state.db"
    seed = SessionDB(db_path=db_path)
    seed.create_session("s1", source="cli")
    seed.close()

    ro = SessionDB(db_path=db_path, read_only=True)
    try:
        assert has_live_connection(db_path)
        assert read_header_bytes_preopen(db_path, length=16) is None
    finally:
        ro.close()

    assert not has_live_connection(db_path)
    assert read_header_bytes_preopen(db_path, length=16) is not None








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
