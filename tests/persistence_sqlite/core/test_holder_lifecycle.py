"""StoreHolder lifecycle on real files: the accounting queue across close(), the read barrier on a loop thread,
write-lock replay and patience, a read with its own lock wait, and the read-only open that never creates or migrates a store.
"""

import asyncio
import hashlib
import logging
import os
import sqlite3
import threading
import time
from contextlib import closing
from pathlib import Path

import pytest

from persistence.sqlite import _holder
from persistence.sqlite import _registry as registry
from persistence.sqlite._holder import _READ_BUSY_TIMEOUT_S, StoreHolder, set_token_delta_applier


@pytest.fixture(autouse=True)
def _clean_state():
    registry.close_all()
    registered = _holder._token_delta_applier
    yield
    set_token_delta_applier(registered)
    registry.close_all()


@pytest.fixture
def db_path():
    return Path(os.environ["HERMES_HOME"]) / "state.db"


@pytest.fixture
def holder(db_path):
    store = StoreHolder(db_path)
    yield store
    store.close()


def _delta_keys(path):
    reader = StoreHolder(path, read_only=True)
    try:
        return [row[0] for row in reader._read_all("SELECT key FROM state_meta WHERE key LIKE 'delta:%' ORDER BY key")]
    finally:
        reader.close()


class _RecordingApplier:
    """Writes each delta as one state_meta row through the holder's transaction; the first waits on ``gate``."""

    def __init__(self, block_first=True):
        self.applied = []
        self.entered = threading.Event()
        self.gate = threading.Event()
        if not block_first:
            self.gate.set()

    def __call__(self, holder, session_id, kwargs):
        if not self.applied:
            self.entered.set()
            assert self.gate.wait(10.0)
        self.applied.append((session_id, kwargs["seq"]))
        holder._execute_write(lambda conn: conn.execute(
            "INSERT INTO state_meta (key, value) VALUES (?, ?)", (f"delta:{kwargs['seq']:03d}", session_id)))


def _queue(holder, seq):
    # Alternating sessions never coalesce, so every delta is applied on its own.
    session_id = "s-a" if seq % 2 else "s-b"
    holder.queue_token_counts(session_id, seq=seq, input_tokens=seq)
    return session_id, seq


class TestAccountingQueueAcrossClose:
    def test_close_waits_for_the_writer_and_applies_every_delta_once_in_order(self, holder, db_path):
        applier = _RecordingApplier()
        set_token_delta_applier(applier)
        expected = [_queue(holder, 0)]
        assert applier.entered.wait(5.0)
        expected += [_queue(holder, seq) for seq in range(1, 6)]

        closer = threading.Thread(target=holder.close)
        closer.start()
        deadline = time.monotonic() + 5.0
        while not holder._token_writer_stop and time.monotonic() < deadline:
            time.sleep(0.01)
        assert holder._token_writer_stop
        # Queued after close() began, while the writer is still mid-batch: the writer drains it too.
        expected.append(_queue(holder, 6))
        closer.join(0.3)
        assert closer.is_alive(), "close() returned while the writer was mid-batch"
        assert holder._conn is not None

        applier.gate.set()
        closer.join(10.0)
        assert not closer.is_alive()
        assert holder._conn is None
        assert holder._token_writer_thread is None
        assert not holder._token_queue
        assert applier.applied == expected
        assert _delta_keys(db_path) == [f"delta:{seq:03d}" for _, seq in expected]

    def test_queue_after_close_applies_inline(self, holder, db_path, caplog):
        applier = _RecordingApplier(block_first=False)
        set_token_delta_applier(applier)
        _queue(holder, 0)
        assert holder.flush_token_counts()
        holder.close()
        assert holder._conn is None

        with caplog.at_level(logging.WARNING, logger="hermes_state"):
            _queue(holder, 1)

        assert not holder._token_queue, "a delta queued after close() must not park on a stopped queue"
        assert applier.applied == [("s-b", 0), ("s-a", 1)]
        assert any("reopening" in record.getMessage() for record in caplog.records)
        holder.close()
        assert _delta_keys(db_path) == ["delta:000", "delta:001"]

    def test_flush_returns_true_once_the_queue_drains(self, holder):
        applier = _RecordingApplier()
        set_token_delta_applier(applier)
        expected = [_queue(holder, 0)]
        assert applier.entered.wait(5.0)
        expected += [_queue(holder, seq) for seq in range(1, 4)]

        assert holder.flush_token_counts(timeout=0.2) is False
        applier.gate.set()
        assert holder.flush_token_counts() is True
        assert applier.applied == expected
        assert not holder._token_queue and not holder._token_writer_busy



class TestReadBarrierOnALoopThread:
    def test_a_read_on_a_loop_thread_never_waits_for_a_stuck_writer(self, holder):
        applier = _RecordingApplier()
        set_token_delta_applier(applier)
        expected = [_queue(holder, 0)]
        assert applier.entered.wait(5.0)

        async def read_on_loop():
            started = time.monotonic()
            drained = holder.flush_for_read()
            return drained, time.monotonic() - started

        drained, elapsed = asyncio.run(read_on_loop())
        assert drained is False and elapsed < 2.0, f"loop-thread read waited {elapsed:.2f}s on the token writer"
        applier.gate.set()
        # Off the loop the read barrier still waits, so the queued delta is applied before the read.
        assert holder.flush_for_read() is True
        assert applier.applied == expected
        assert not holder._token_queue and not holder._token_writer_busy


def _hold_write_lock(path, hold_s, started):
    conn = sqlite3.connect(str(path), timeout=1.0, isolation_level=None)
    try:
        conn.execute("BEGIN IMMEDIATE")
        started.set()
        time.sleep(hold_s)  # health: allow HX013 -- the sibling holding the write lock this long is the contention under test
        conn.execute("COMMIT")
    finally:
        conn.close()


class TestWriteLockReplay:
    def test_write_replays_until_a_sibling_releases_the_lock(self, holder, db_path, monkeypatch):
        retries = []
        original = holder._sleep_before_write_retry

        def counting(deadline, patience_s):
            retries.append(patience_s)
            return original(deadline, patience_s)

        monkeypatch.setattr(holder, "_sleep_before_write_retry", counting)
        started = threading.Event()
        # Held past the writer connection's 1s busy timeout, so BEGIN IMMEDIATE fails and is retried.
        sibling = threading.Thread(target=_hold_write_lock, args=(db_path, 1.3, started))
        sibling.start()
        try:
            assert started.wait(5.0)
            holder.set_meta("k", "after-sibling")
        finally:
            sibling.join(10.0)
        assert retries, "the locked BEGIN IMMEDIATE was not retried"
        assert holder.get_meta("k") == "after-sibling"

    def test_callback_raising_busy_is_rolled_back_and_replayed(self, holder):
        calls = []

        def write(conn):
            calls.append(1)
            conn.execute("INSERT INTO state_meta (key, value) VALUES ('replayed', ?)", (str(len(calls)),))
            if len(calls) == 1:
                raise sqlite3.OperationalError("database is locked")
            return len(calls)

        assert holder._execute_write(write) == 2
        assert holder.get_meta("replayed") == "2"

    def test_exhausted_patience_raises_the_locked_error(self, holder, db_path):
        started = threading.Event()
        sibling = threading.Thread(target=_hold_write_lock, args=(db_path, 1.5, started))
        sibling.start()
        try:
            assert started.wait(5.0)
            with pytest.raises(sqlite3.OperationalError) as excinfo:
                holder._write_sql("INSERT INTO state_meta (key, value) VALUES ('k', 'v')", patience_s=0.05)
        finally:
            sibling.join(10.0)
        text = str(excinfo.value)
        assert text.startswith("database is locked (another Hermes process held the state.db write lock")
        assert "the database itself is healthy" in text
        assert holder.get_meta("k") is None


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _schema_rows(path):
    with closing(sqlite3.connect(str(path))) as conn:
        return conn.execute("SELECT type, name, sql FROM sqlite_master ORDER BY name").fetchall()


class TestReadOnlyHolder:
    def test_never_creates_a_missing_store(self, db_path):
        missing_dir = db_path.parent / "absent"
        for path in (db_path, missing_dir / "state.db"):
            with pytest.raises(sqlite3.OperationalError):
                StoreHolder(path, read_only=True)
            assert not path.exists()
            assert not Path(f"{path}-wal").exists() and not Path(f"{path}-shm").exists()
        assert not missing_dir.exists()

    def test_never_migrates_an_old_store(self, db_path):
        with closing(sqlite3.connect(str(db_path))) as conn:
            conn.execute("CREATE TABLE state_meta (key TEXT PRIMARY KEY, value TEXT)")
            conn.execute("INSERT INTO state_meta VALUES ('who', 'old')")
            conn.commit()
        schema = _schema_rows(db_path)
        digest = _sha256(db_path)

        reader = StoreHolder(db_path, read_only=True)
        try:
            assert reader.read_only and not reader._fts_enabled
            assert reader._read_one("SELECT value FROM state_meta WHERE key = 'who'")[0] == "old"
            with pytest.raises(sqlite3.OperationalError):
                reader.set_meta("who", "new")
        finally:
            reader.close()

        assert _schema_rows(db_path) == schema
        assert _sha256(db_path) == digest
        assert not Path(f"{db_path}-wal").exists()


class TestOpenFailureLatch:
    def test_a_failed_open_is_the_last_open_failure_the_notice_names(self, db_path, monkeypatch):
        from persistence.sqlite._state import guard as hermes_state_guard
        import storage_user_copy

        monkeypatch.setattr(storage_user_copy, "_last_init_error", None)
        monkeypatch.setattr(hermes_state_guard, "_last_init_error", None)
        with pytest.raises(sqlite3.OperationalError) as excinfo:
            StoreHolder(db_path, read_only=True)
        assert storage_user_copy.get_last_init_error() == f"OperationalError: {excinfo.value}"
        assert hermes_state_guard.get_last_init_error() is None


class TestReadWithOwnTimeout:
    """A read that keeps a legacy call site's own lock wait (``timeout=2`` in main's
    hermes_cli/worktree_ops.py reads) on the holder's pooled reader, and gives the reader its budget back."""

    def test_the_block_waits_the_given_time_and_the_reader_gets_its_budget_back(self, holder):
        with holder._read_ctx_with_timeout(2) as conn:
            assert conn.execute("PRAGMA busy_timeout").fetchone()[0] == 2000
            assert conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0] == 0
        with holder._read_ctx() as again:
            assert again is conn
            assert again.execute("PRAGMA busy_timeout").fetchone()[0] == int(_READ_BUSY_TIMEOUT_S * 1000)

    def test_the_budget_is_restored_when_the_read_raises(self, holder):
        with pytest.raises(sqlite3.OperationalError):
            with holder._read_ctx_with_timeout(0.05) as conn:
                conn.execute("SELECT * FROM no_such_table")
        with holder._read_ctx() as again:
            assert again is conn
            assert again.execute("PRAGMA busy_timeout").fetchone()[0] == int(_READ_BUSY_TIMEOUT_S * 1000)
