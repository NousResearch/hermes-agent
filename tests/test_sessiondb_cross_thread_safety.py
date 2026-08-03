"""Regression coverage for cross-thread SessionDB connection safety.

The gateway intentionally shares one SessionDB instance through asyncio.to_thread.
These tests cover the safety measures added after intermittent
"no more rows available" persistence failures (2026-07-28 root-cause: SQLite
error-state scrambling race on unlocked handoff reads):

* shared writer connections disable CPython's prepared-statement cache
  (defensive hardening, relevant on Python 3.12+);
* compression-lock reads serialize with the shared writer lock;
* handoff reads use _read_ctx() to avoid the errmsg-scrambling race.
"""

from __future__ import annotations

import concurrent.futures
import sqlite3
import threading
import time
from pathlib import Path

import pytest

from hermes_state import SessionDB


# Exposure window for the concurrency stress test.
#
# The race is probabilistic, so the test has to keep the writer and the
# handoff watchers overlapping long enough for it to surface. Measured on
# pre-fix code: 6/6 runs scrambled, between 0.76s and 3.67s, at write counts
# ranging from 35 to 248 — so elapsed exposure, not write count, is what
# determines whether the race appears. The window is set at ~2x the slowest
# observed failure. A failing run exits the moment it scrambles; only a
# passing run spends the full window.
_EXPOSURE_S = 8.0
# Independently, a run that achieved no writes proves nothing, so require a
# floor of real work before trusting a pass.
_MIN_PROGRESS_WRITES = 10


def _new_db(path: Path) -> SessionDB:
    db = SessionDB(db_path=path)
    db.create_session("session", source="test", model="test-model")
    return db


def _enable_wal(db: SessionDB) -> None:
    """Put *db* on the WAL read path, or skip the test if unavailable.

    _read_ctx() only hands out the per-thread read-only connection when
    _wal_active is true; without this the WAL branch silently goes uncovered.
    """
    journal_mode = db._conn.execute("PRAGMA journal_mode=WAL").fetchone()[0]
    assert journal_mode.lower() == "wal", (
        f"expected WAL journal mode for this test, got {journal_mode!r}"
    )
    db._wal_active = True


def test_compression_lock_read_waits_for_writer_lock(tmp_path: Path) -> None:
    db = _new_db(tmp_path / "state.db")
    try:
        assert db.try_acquire_compression_lock("session", "holder") is True
        entered = threading.Event()
        release = threading.Event()
        result: list[str | None] = []

        def hold_writer_lock() -> None:
            with db._lock:
                entered.set()
                release.wait(timeout=5)

        holder = threading.Thread(target=hold_writer_lock)
        holder.start()
        assert entered.wait(timeout=2)

        reader = threading.Thread(
            target=lambda: result.append(db.get_compression_lock_holder("session"))
        )
        reader.start()
        time.sleep(0.05)
        # If get_compression_lock_holder bypasses db._lock, this can already be
        # populated. The correct implementation remains blocked.
        assert result == []
        release.set()
        holder.join(timeout=2)
        reader.join(timeout=2)
        assert result == ["holder"]
    finally:
        db.close()


def test_concurrent_append_and_lock_reads_do_not_escape_sqlite_errors(tmp_path: Path) -> None:
    db = _new_db(tmp_path / "state.db")
    errors: list[BaseException] = []
    errors_lock = threading.Lock()

    def append_worker(worker: int) -> None:
        try:
            for index in range(150):
                db.append_message(
                    "session",
                    "tool",
                    content=(f"worker={worker} index={index} " + "x" * 256),
                    tool_name="terminal",
                )
        except BaseException as exc:  # assertion reports exact unexpected type
            with errors_lock:
                errors.append(exc)

    def lock_reader() -> None:
        try:
            for _ in range(700):
                db.get_compression_lock_holder("session")
        except BaseException as exc:
            with errors_lock:
                errors.append(exc)

    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=9) as pool:
            futures = [pool.submit(append_worker, worker) for worker in range(8)]
            futures.append(pool.submit(lock_reader))
            for future in futures:
                future.result(timeout=30)
        assert errors == []
        assert len(db.get_messages("session")) == 8 * 150
        assert db._conn.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
    finally:
        db.close()


def test_shared_writer_connect_request_has_cache_disabled(monkeypatch, tmp_path: Path) -> None:
    """Pin the actual sqlite connect kwargs rather than a private attribute."""
    import hermes_state

    captured: list[dict] = []
    real_connect = hermes_state.sqlite3.connect

    def spy_connect(*args, **kwargs):
        captured.append(dict(kwargs))
        return real_connect(*args, **kwargs)

    monkeypatch.setattr(hermes_state.sqlite3, "connect", spy_connect)
    db = SessionDB(db_path=tmp_path / "state.db")
    try:
        assert any(call.get("cached_statements") == 0 for call in captured)
    finally:
        db.close()


def test_per_thread_read_connection_has_cache_disabled(monkeypatch, tmp_path: Path) -> None:
    import hermes_state

    captured: list[dict] = []
    real_connect = hermes_state.sqlite3.connect

    def spy_connect(*args, **kwargs):
        captured.append(dict(kwargs))
        return real_connect(*args, **kwargs)

    monkeypatch.setattr(hermes_state.sqlite3, "connect", spy_connect)
    db = _new_db(tmp_path / "state.db")
    try:
        # The test harness may intentionally force DELETE mode for SQLite
        # portability. This unit pins the WAL-only per-thread read branch.
        db._wal_active = True
        done = threading.Event()

        def read_in_worker() -> None:
            assert db.get_session("session") is not None
            done.set()

        thread = threading.Thread(target=read_in_worker)
        thread.start()
        thread.join(timeout=5)
        assert done.is_set()
        assert sum(1 for call in captured if call.get("cached_statements") == 0) >= 2
    finally:
        db.close()


def test_handoff_reads_do_not_scramble_writer_error_state(tmp_path: Path) -> None:
    """Root-cause regression for 'no more rows available' (2026-07-28).

    The gateway's handoff watcher polls list_pending_handoffs() /
    get_handoff_state() every 2s through asyncio.to_thread on the SHARED
    writer connection. When those reads bypass the writer lock, their
    sqlite3_step() -> SQLITE_DONE runs with the GIL released and overwrites
    the db handle's global error state. A concurrent writer that hits a real
    SQLITE_BUSY then raises with sqlite3_errmsg(db) == errstr(SQLITE_DONE)
    == 'no more rows available' instead of 'database is locked', so
    _execute_write's locked/busy retry classifier cannot retry it and the
    transcript flush fails.

    Fails within seconds while the handoff reads are unlocked; passes once
    they go through _read_ctx() (per-thread RO connection or writer lock).

    WAL-path coverage: _read_ctx() uses the per-thread read-only connection
    only when _wal_active is True. This test explicitly enables WAL so the
    intended production path is exercised, not just the writer-lock fallback.
    """
    db = _new_db(tmp_path / "state.db")
    # Enable WAL so _read_ctx() takes the per-thread read-only connection
    # path instead of the writer-lock fallback (production behavior).
    # Asserted, not assumed: if a future SQLite build or Hermes' WAL guard
    # refuses WAL, this test would silently revert to covering only the
    # fallback branch — the exact gap this coverage exists to close.
    _enable_wal(db)
    stop = threading.Event()
    scrambled: list[BaseException] = []
    scramble_seen = threading.Event()
    progress = {"writes": 0}

    def contention() -> None:
        # Mirrors SessionStore/CLI/cron writers: an independent connection
        # that holds the WAL write lock most of the time so the shared
        # writer's BEGIN IMMEDIATE regularly hits SQLITE_BUSY.
        conn = sqlite3.connect(str(tmp_path / "state.db"), timeout=0.05)
        try:
            while not stop.is_set():
                try:
                    conn.execute("BEGIN IMMEDIATE")
                    conn.execute(
                        "UPDATE sessions SET model = 'contender' "
                        "WHERE id = 'session'")
                    time.sleep(0.002)
                    conn.commit()
                except sqlite3.Error:
                    try:
                        conn.rollback()
                    except Exception:
                        pass
        finally:
            conn.close()

    def watcher() -> None:
        # Handoff watcher mirror (gateway/run.py::_handoff_watcher).
        while not stop.is_set():
            db.list_pending_handoffs()
            db.get_handoff_state("session")

    def writer() -> None:
        success = 0
        while not stop.is_set():
            try:
                db.append_message("session", "user", content="x")
                success += 1
                progress["writes"] = success
            except sqlite3.Error as exc:
                msg = str(exc).lower()
                if "locked" not in msg and "busy" not in msg:
                    scrambled.append(exc)
                    scramble_seen.set()
                    stop.set()
                    return

    threads = [threading.Thread(target=contention, daemon=True),
               threading.Thread(target=writer, daemon=True)]
    threads += [threading.Thread(target=watcher, daemon=True)
                for _ in range(3)]
    for t in threads:
        t.start()
    # A regression trips this immediately, so a failing run costs a fraction
    # of a second; only a healthy run spends the full exposure window.
    scramble_seen.wait(timeout=_EXPOSURE_S)
    stop.set()
    for t in threads:
        t.join(timeout=10)
    db.close()
    # Order matters: a scramble aborts the writer, which also suppresses its
    # write count. Report the real defect first, or a genuine regression gets
    # misdiagnosed as "the writer made no progress".
    assert not scrambled, (
        "writer exception scrambled by unlocked shared-connection read: "
        f"{type(scrambled[0]).__name__}: {scrambled[0]}"
    )
    # Guard against a vacuous pass: without real writes the watchers never
    # raced anything, so a clean result would mean nothing.
    assert progress["writes"] >= _MIN_PROGRESS_WRITES, (
        f"writer completed only {progress['writes']} appends in "
        f"{_EXPOSURE_S}s (expected >= {_MIN_PROGRESS_WRITES}) — contention or "
        "handoff reads may be blocking unexpectedly"
    )


def test_handoff_reads_use_read_ctx_on_wal_path(tmp_path: Path) -> None:
    """Both handoff reads must resolve through _read_ctx() under WAL.

    The stress test above proves the race is gone; this pins *how*. Under
    WAL, _read_ctx() yields a per-thread read-only connection, so neither
    handoff read may touch the shared writer connection — that is precisely
    what allowed a reader's sqlite3_step() to overwrite the writer's error
    state. Asserting on behavior (which connection served the read, and that
    the rows are still correct) rather than on source text keeps this honest
    if the internals are refactored.
    """
    db = _new_db(tmp_path / "state.db")
    try:
        _enable_wal(db)
        assert db.request_handoff("session", "telegram") is True

        used: list[sqlite3.Connection] = []
        real_get_read_conn = db._get_read_conn

        def spy_get_read_conn():
            conn = real_get_read_conn()
            used.append(conn)
            return conn

        db._get_read_conn = spy_get_read_conn

        state = db.get_handoff_state("session")
        pending = db.list_pending_handoffs()

        # Both helpers went through the read path...
        assert len(used) == 2, (
            f"expected both handoff reads to consult _read_ctx(), saw {len(used)}"
        )
        # ...and got a real per-thread read-only connection, not the shared
        # writer connection whose error state the race corrupted.
        assert all(conn is not None for conn in used), (
            "WAL read path returned no connection; _read_ctx() fell back to "
            "the shared writer connection"
        )
        assert all(conn is not db._conn for conn in used), (
            "handoff read was served by the shared writer connection"
        )

        # Routing is only worth anything if the data is still right.
        assert state is not None and state["state"] == "pending"
        assert state["platform"] == "telegram"
        assert [row["id"] for row in pending] == ["session"]

        # And the read-only connection genuinely cannot write, which is what
        # makes it safe to use off the writer lock.
        with pytest.raises(sqlite3.OperationalError):
            used[0].execute("UPDATE sessions SET model = 'nope' WHERE id = 'session'")
    finally:
        db.close()
