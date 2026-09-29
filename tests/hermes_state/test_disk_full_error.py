"""is_disk_full_error classifies ENOSPC / SQLITE_FULL failures."""

from __future__ import annotations

import errno
import sqlite3

import pytest

from hermes_state_errors import describe_sqlite_error, is_disk_full_error


def test_enospc_oserror():
    assert is_disk_full_error(OSError(errno.ENOSPC, "No space left on device")) is True


def test_sqlite_full_operational_error():
    assert is_disk_full_error(sqlite3.OperationalError("database or disk is full")) is True


def test_string_markers():
    assert is_disk_full_error("disk full: session storage could not be written") is True
    assert is_disk_full_error("ENOSPC writing state.db") is True
    assert is_disk_full_error("This is often a full disk — free some space") is True


def test_unrelated_errors():
    assert is_disk_full_error(None) is False
    assert is_disk_full_error(OSError(errno.EACCES, "Permission denied")) is False
    assert is_disk_full_error(RuntimeError("network timeout")) is False
    assert is_disk_full_error("session not found") is False
    assert is_disk_full_error("session storage could not be written: permission denied") is False


def test_sqlite_full_from_a_connection_ceiling_still_reads_as_disk(tmp_path):
    """SQLITE_FULL is also raised with the filesystem healthy, and the text is identical.

    ``PRAGMA max_page_count`` is a per-connection ceiling, not free space: the write is
    refused with ``database or disk is full`` on a filesystem with room to spare (measured:
    SQLite 3.53.1, 59 GiB free). The prose therefore cannot separate the two, which is why
    the persistence log line records the result code instead — this test is the proof that
    the prose alone is insufficient (#RIC-103).
    """
    conn = sqlite3.connect(tmp_path / "state.db")
    conn.execute("CREATE TABLE t(id INTEGER PRIMARY KEY, body TEXT)")
    conn.executemany("INSERT INTO t(body) VALUES (?)", [("x" * 4096,) for _ in range(50)])
    conn.commit()
    conn.execute("PRAGMA max_page_count = 30")  # ceiling, unrelated to free space
    with pytest.raises(sqlite3.OperationalError) as excinfo:
        conn.executemany("INSERT INTO t(body) VALUES (?)", [("y" * 4096,) for _ in range(200)])
        conn.commit()
    err = excinfo.value
    assert "database or disk is full" in str(err)
    assert is_disk_full_error(err) is True  # indistinguishable by prose...
    detail = describe_sqlite_error(err)  # ...but not by provenance
    assert "SQLITE_FULL" in detail and "sqlite_errorcode=13" in detail
    conn.close()


def test_describe_sqlite_error_separates_identical_messages():
    """Same message, different provenance: only the class / result code / errno tells them apart."""
    ceiling = sqlite3.OperationalError("database or disk is full")
    enospc = OSError(errno.ENOSPC, "database or disk is full")
    assert "database or disk is full" in str(enospc)
    # Both land in the disk bucket by prose - the classification is identical ...
    assert is_disk_full_error(ceiling) is True and is_disk_full_error(enospc) is True
    # ... so only the logged provenance can name which one happened.
    assert describe_sqlite_error(ceiling).startswith("OperationalError:")
    detail = describe_sqlite_error(enospc)
    assert detail.startswith(f"OSError errno={errno.ENOSPC}:") and f"errno={errno.ENOSPC}" in detail
    assert describe_sqlite_error(ceiling) != describe_sqlite_error(enospc)


def test_describe_sqlite_error_never_raises_on_odd_input():
    class _OddError(Exception):
        sqlite_errorcode = None
        sqlite_errorname = None

    assert describe_sqlite_error(None) == "none"
    assert describe_sqlite_error("plain string") == "str: plain string"
    assert describe_sqlite_error(RuntimeError("boom")) == "RuntimeError: boom"
    # An exception that carries the attributes as None must not print "None" as a code.
    assert "sqlite_errorcode" not in describe_sqlite_error(_OddError("boom"))

