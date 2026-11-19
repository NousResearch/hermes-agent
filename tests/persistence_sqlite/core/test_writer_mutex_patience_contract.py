"""Write patience bounds the in-process writer mutex wait without losing a write (field report).

A field probe held ``StoreHolder._lock`` and saw a ``patience_s=0.05`` write wait ~0.27 s and then run.
The patience budget now covers the mutex: a write that cannot take it in time is refused before
BEGIN with :class:`WriterMutexTimeout`, which callers already treat as lock contention (retry or
defer, never damage). These tests pin the contract end to end:

* refused inside the budget, classified as in-process contention (not a foreign SQLite writer);
* the refusal names the in-process holder (thread, call site, held-for) without SQL or values;
* queued token accounting that hits the refusal keeps the delta and applies it once free.
"""

import logging
import sqlite3
import threading
import time

import pytest

from persistence.sqlite._holder import StoreHolder
from persistence.sqlite._state.errors import classify_persistence_error, is_sqlite_lock_error
from persistence.sqlite._state.write import WriterMutexTimeout
from persistence.sqlite.conversation._store.sessions import create_session, get_session
from tests.waiting import wait_until


@pytest.fixture()
def db(tmp_path):
    handle = StoreHolder(tmp_path / "state.db")
    create_session(handle, "owned", "test")
    yield handle
    handle.close()


def _hold_writer_mutex(db, hold_s):
    """Hold the writer mutex from a named thread through the real write path (a slow callback)."""
    held, release = threading.Event(), threading.Event()

    def slow_write(conn):
        held.set()
        release.wait(hold_s)
        conn.execute("UPDATE sessions SET title = ? WHERE id = ?", ("kept", "owned"))

    holder = threading.Thread(target=lambda: db._execute_write(slow_write), name="slow-transcript-writer")
    holder.start()
    assert held.wait(5), "holder never took the writer mutex"
    return holder, release


def test_refused_within_patience_and_classified_as_in_process_contention(db):
    holder, release = _hold_writer_mutex(db, hold_s=5)
    try:
        started = time.monotonic()
        with pytest.raises(WriterMutexTimeout) as raised:
            db._write_sql("UPDATE sessions SET title = ? WHERE id = ?", ("never", "owned"), patience_s=0.05)
        elapsed = time.monotonic() - started
    finally:
        release.set()
        holder.join(5)
    assert elapsed < 0.25, f"a 0.05 s write waited {elapsed:.3f}s for the writer mutex"
    exc = raised.value
    # Existing callers retry/defer on the lock class: the refusal must stay inside it.
    assert isinstance(exc, sqlite3.OperationalError) and is_sqlite_lock_error(exc)
    assert classify_persistence_error(exc) == "locked"
    # ... but it is told apart from a foreign SQLite writer.
    assert exc.sqlite_errorcode == sqlite3.SQLITE_BUSY
    assert "in-process" in str(exc)
    assert "another Hermes process" not in str(exc)
    assert get_session(db, "owned")["title"] == "kept"  # the refused write never ran


def test_refusal_names_the_in_process_holder_without_sql_or_values(db, caplog):
    holder, release = _hold_writer_mutex(db, hold_s=5)
    try:
        with caplog.at_level(logging.WARNING, logger="hermes_state"), pytest.raises(WriterMutexTimeout):
            db._write_sql("UPDATE sessions SET title = ? WHERE id = ?", ("secret-title-value", "owned"),
                          patience_s=0.05)
    finally:
        release.set()
        holder.join(5)
    lines = [r.getMessage() for r in caplog.records if "writer mutex" in r.getMessage()]
    assert len(lines) == 1, lines
    line = lines[0]
    assert "slow-transcript-writer" in line  # holder thread
    assert "slow_write" in line  # holder call site (function name, not source text)
    assert "state.db" in line
    assert "held for" in line
    assert "secret-title-value" not in line and "UPDATE" not in line


def test_holder_record_is_cleared_after_release(db):
    holder, release = _hold_writer_mutex(db, hold_s=5)
    release.set()
    holder.join(5)
    db._write_sql("UPDATE sessions SET title = ? WHERE id = ?", ("after", "owned"), patience_s=0.05)
    assert get_session(db, "owned")["title"] == "after"
    assert getattr(db, "_writer_owner", None) is None


def test_queued_token_delta_survives_a_mutex_refusal(db, monkeypatch, caplog):
    """The token writer applies queued deltas with routine patience; a refusal must requeue, not drop."""
    caplog.set_level(logging.DEBUG, logger="hermes_state")
    for name in ("_WRITE_PATIENCE_S", "_TRANSCRIPT_WRITE_PATIENCE_S"):
        monkeypatch.setattr(StoreHolder, name, 0.05)
    holder, release = _hold_writer_mutex(db, hold_s=5)
    try:
        db.queue_token_counts("owned", input_tokens=7, output_tokens=3, model="m")
        # Several 0.05 s attempts are refused while the mutex is held: wait for the first refusal to land.
        wait_until(lambda: any("requeued" in r.getMessage() for r in caplog.records), timeout=5.0,
                   what="a token-writer requeue")
        assert db._token_queue or db._token_writer_busy, "the refused delta left the queue"
    finally:
        release.set()
        holder.join(5)
    assert db.flush_token_counts(timeout=10.0)
    row = get_session(db, "owned")
    assert (row["input_tokens"], row["output_tokens"]) == (7, 3), "queued token delta was lost"


def _queue_without_worker(db, monkeypatch, **values):
    with monkeypatch.context() as paused:
        paused.setattr(db, "_token_writer_loop", lambda: None)
        db.queue_token_counts("owned", **values)
        db._token_writer_thread.join(5)
        assert not db._token_writer_thread.is_alive()


def test_queued_tokens_keep_lost_commit_ack_unsettled(db, monkeypatch):
    """An observed committed value cannot serve as a proof to replay the delta."""
    conn = db._conn

    class LostCommitAcknowledgement:
        def __getattr__(self, name):
            return getattr(conn, name)

        def commit(self):
            conn.commit()
            raise OSError("lost commit acknowledgement")

    _queue_without_worker(db, monkeypatch, input_tokens=7, model="saved", api_call_count=1)
    with monkeypatch.context() as wire:
        wire.setattr(db, "_conn", LostCommitAcknowledgement())
        assert db.flush_token_counts(timeout=2) is False
    assert db._read_one("SELECT input_tokens FROM sessions WHERE id='owned'")[0] == 7
    assert db.flush_token_counts(timeout=2) is False
    assert db._read_one("SELECT input_tokens FROM sessions WHERE id='owned'")[0] == 7


def test_queued_tokens_retry_after_begin_proves_no_write(db, monkeypatch):
    import sqlite3

    conn = db._conn

    class RefusedBegin:
        def __getattr__(self, name):
            return getattr(conn, name)

        def execute(self, sql, *args):
            if sql == "BEGIN IMMEDIATE":
                raise sqlite3.OperationalError("disk I/O error")
            return conn.execute(sql, *args)

    _queue_without_worker(db, monkeypatch, input_tokens=7, model="saved", api_call_count=1)
    with monkeypatch.context() as wire:
        wire.setattr(db, "_conn", RefusedBegin())
        wire.setattr(db, "_sleep_before_write_retry", lambda *_args: False)
        assert db.flush_token_counts(timeout=0.1) is False
    assert db._read_one("SELECT input_tokens FROM sessions WHERE id='owned'")[0] == 0
    assert db.flush_token_counts(timeout=2) is True
    assert db._read_one("SELECT input_tokens FROM sessions WHERE id='owned'")[0] == 7
