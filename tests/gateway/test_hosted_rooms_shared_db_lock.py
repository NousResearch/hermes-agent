"""Cross-process first-open serialization for the shared hosted-rooms db.

Every profile gateway opens the same install-root ``state.db`` through
``hosted_rooms._connect``. On a simultaneous fleet restart those first opens
overlapped inside the journal-mode pragma and left a file whose header was no
longer a SQLite database (#102120). The invariant under test: concurrent first
opens of one hosted-rooms db can never leave a non-database behind.
"""

from __future__ import annotations

import sqlite3
import threading
import time
from pathlib import Path

import pytest

import hermes_state
from gateway import hosted_rooms


SQLITE_MAGIC = b"SQLite format 3\x00"


def _assert_healthy_database(db_path: Path) -> None:
    assert db_path.is_file()
    header = db_path.read_bytes()[: len(SQLITE_MAGIC)]
    assert header == SQLITE_MAGIC, f"shared db lost its SQLite header: {header!r}"
    conn = sqlite3.connect(db_path)
    try:
        assert conn.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
    finally:
        conn.close()


def test_concurrent_first_open_cannot_leave_a_non_database(tmp_path, monkeypatch):
    """Two racing first opens must both succeed against a healthy file.

    ``apply_wal_with_fallback`` is replaced by a deterministic stand-in for the
    unserialized failure: if two callers are ever inside journal-mode setup at
    the same moment, it scrambles the on-disk header and every later open of
    that file raises the observed ``file is not a database``. Under the lock the
    two callers cannot overlap, so the stand-in never fires.
    """

    db_path = tmp_path / "state.db"
    real_apply = hermes_state.apply_wal_with_fallback
    overlap_gate = threading.Barrier(2)
    poisoned = threading.Event()

    def racing_apply(conn: sqlite3.Connection, **kwargs: object) -> str:
        if poisoned.is_set():
            raise sqlite3.DatabaseError("file is not a database")
        mode = real_apply(conn, **kwargs)  # type: ignore[arg-type]
        try:
            overlap_gate.wait(timeout=0.5)
        except threading.BrokenBarrierError:
            return mode
        # Two openers were inside journal-mode setup simultaneously.
        poisoned.set()
        with db_path.open("r+b") as handle:
            handle.write(b"\x00" * len(SQLITE_MAGIC))
        raise sqlite3.DatabaseError("file is not a database")

    monkeypatch.setattr(hermes_state, "apply_wal_with_fallback", racing_apply)

    failures: list[BaseException] = []

    def open_and_prune() -> None:
        try:
            hosted_rooms.prune_disbanded_rooms(db_path)
        except BaseException as exc:  # noqa: BLE001 - reported as a failure
            failures.append(exc)

    threads = [threading.Thread(target=open_and_prune) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=30)
        assert not thread.is_alive()

    assert not failures, f"concurrent first open failed: {failures!r}"
    monkeypatch.setattr(hermes_state, "apply_wal_with_fallback", real_apply)
    _assert_healthy_database(db_path)


def test_second_opener_waits_for_the_lock_instead_of_skipping(tmp_path, monkeypatch):
    """A busy lock defers the newcomer; it must not skip the open."""

    monkeypatch.setattr(hosted_rooms, "_SHARED_DB_OPEN_LOCK_TIMEOUT_SECONDS", 5.0)
    db_path = tmp_path / "state.db"
    hold_seconds = 0.3
    holder_ready = threading.Event()

    def hold_lock() -> None:
        with hosted_rooms._shared_db_open_lock(db_path) as acquired:
            assert acquired
            holder_ready.set()
            time.sleep(hold_seconds)

    holder = threading.Thread(target=hold_lock)
    holder.start()
    try:
        assert holder_ready.wait(timeout=5)
        started = time.monotonic()
        conn = hosted_rooms._connect(db_path)
        waited = time.monotonic() - started
    finally:
        holder.join(timeout=10)
    try:
        assert conn.execute("SELECT 1 FROM hosted_rooms LIMIT 1").fetchall() == []
    finally:
        conn.close()
    assert waited >= hold_seconds / 2, "second opener did not wait for the holder"
    _assert_healthy_database(db_path)


def test_lock_file_lives_next_to_the_shared_db(tmp_path):
    """The hosted-rooms lock must not collide with repair/FTS lock files."""

    db_path = tmp_path / "state.db"
    with hosted_rooms._shared_db_open_lock(db_path) as acquired:
        assert acquired
        lock_path = db_path.with_name("state.db.hosted-rooms.lock")
        assert lock_path.is_file()


def test_first_open_retries_transient_not_a_database(tmp_path, monkeypatch):
    """A transient scrambled read on first open is retried, not surfaced."""

    db_path = tmp_path / "state.db"
    real_apply = hermes_state.apply_wal_with_fallback
    calls: list[int] = []

    def flaky_apply(conn: sqlite3.Connection, **kwargs: object) -> str:
        calls.append(1)
        mode = real_apply(conn, **kwargs)  # type: ignore[arg-type]
        if len(calls) == 1:
            raise sqlite3.DatabaseError("file is not a database")
        return mode

    monkeypatch.setattr(hermes_state, "apply_wal_with_fallback", flaky_apply)
    conn = hosted_rooms._connect(db_path)
    conn.close()
    assert len(calls) == 2
    monkeypatch.setattr(hermes_state, "apply_wal_with_fallback", real_apply)
    _assert_healthy_database(db_path)


def test_persistent_not_a_database_still_raises(tmp_path, monkeypatch):
    """Retries are bounded: a genuinely corrupt file still fails loudly."""

    db_path = tmp_path / "state.db"
    real_apply = hermes_state.apply_wal_with_fallback
    calls: list[int] = []

    def broken_apply(conn: sqlite3.Connection, **kwargs: object) -> str:
        calls.append(1)
        real_apply(conn, **kwargs)  # type: ignore[arg-type]
        raise sqlite3.DatabaseError("file is not a database")

    monkeypatch.setattr(hermes_state, "apply_wal_with_fallback", broken_apply)
    with pytest.raises(sqlite3.DatabaseError):
        hosted_rooms._connect(db_path)
    assert len(calls) == hosted_rooms._FIRST_OPEN_CORRUPTION_RETRIES
