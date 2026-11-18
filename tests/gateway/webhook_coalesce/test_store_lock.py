"""Cross-process locking for the durable webhook coalescing store."""

import queue
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor

import pytest

from gateway.platforms.webhook_coalesce.store import CoalesceStore
from gateway.platforms.webhook_coalesce import store as store_module


def _writer_attempt(path):
    result = subprocess.run(
        [sys.executable, "-c", """
import sqlite3
import sys

with sqlite3.connect(sys.argv[1], timeout=0) as conn:
    try:
        conn.execute('BEGIN IMMEDIATE')
    except sqlite3.OperationalError as exc:
        if exc.sqlite_errorcode != sqlite3.SQLITE_BUSY:
            raise
        print('BLOCKED')
    else:
        print('ACQUIRED')
""", str(path)],
        capture_output=True,
        text=True,
        check=True,
        timeout=10,
    )
    return result.stdout.strip()


@pytest.mark.platforms("posix")
def test_read_does_not_release_another_connection_writer_lock(tmp_path):
    store = CoalesceStore(tmp_path)
    with store.connection() as conn:
        assert conn.in_transaction
        assert _writer_attempt(store.path) == 'BLOCKED'
        assert store.seen('missing') is False
        assert conn.in_transaction
        assert _writer_attempt(store.path) == 'BLOCKED', 'read released the writer lock'
    assert _writer_attempt(store.path) == 'ACQUIRED'


@pytest.mark.platforms("posix")
def test_first_creation_close_does_not_release_another_instance_writer(tmp_path, monkeypatch):
    creator = CoalesceStore(tmp_path)
    writer = CoalesceStore(tmp_path)
    closing, release_creator = threading.Event(), threading.Event()
    creator_closed, resume_creator = threading.Event(), threading.Event()
    writer_active, release_writer = threading.Event(), threading.Event()
    checkpoints = queue.Queue()
    real_close = store_module.os.close
    lock = threading.Lock()

    class ObservedOpenLock:
        def __enter__(self):
            if not lock.acquire(blocking=False):
                checkpoints.put('blocked')
                lock.acquire()

        def __exit__(self, *args):
            lock.release()

    def close(fd):
        if threading.current_thread().name == 'creator':
            closing.set()
            assert release_creator.wait(10)
            real_close(fd)
            creator_closed.set()
            assert resume_creator.wait(10)
        else:
            real_close(fd)

    def create():
        threading.current_thread().name = 'creator'
        with creator.connection(write=False):
            pass

    def write():
        with writer.connection() as conn:
            assert conn.in_transaction
            assert _writer_attempt(writer.path) == 'BLOCKED'
            writer_active.set()
            checkpoints.put('writing')
            assert release_writer.wait(10)
            assert conn.in_transaction

    monkeypatch.setattr(store_module.os, 'close', close)
    monkeypatch.setattr(store_module, '_open_lock', ObservedOpenLock(), raising=False)
    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(create)
        try:
            assert closing.wait(10)
            second = pool.submit(write)
            checkpoint = checkpoints.get(timeout=10)
            release_creator.set()
            assert creator_closed.wait(10)
            if checkpoint == 'blocked':
                resume_creator.set()
            assert writer_active.wait(10)
            assert _writer_attempt(writer.path) == 'BLOCKED', 'creator close released the writer lock'
        finally:
            release_creator.set()
            resume_creator.set()
            release_writer.set()
        first.result(timeout=10)
        second.result(timeout=10)
    assert _writer_attempt(writer.path) == 'ACQUIRED'
