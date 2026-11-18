"""Cross-process locking for the durable webhook coalescing store."""

import subprocess
import sys

import pytest

from gateway.platforms.webhook_coalesce.store import CoalesceStore


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
