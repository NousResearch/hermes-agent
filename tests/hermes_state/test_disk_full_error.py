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
    refused with ``database or disk is full`` on a filesystem with room to spare (measured
    on this host: SQLite 3.46.1, 55 GiB free, ``sqlite_errorcode=13`` /
    ``sqlite_errorname='SQLITE_FULL'`` / ``errno`` None). The prose therefore cannot
    separate the two, which is why the persistence log line records the result code
    instead — this test is the proof that the prose alone is insufficient (#RIC-103).
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


def test_describe_sqlite_error_surfaces_engine_provenance_for_identical_messages():
    """Same message, same engine code: provenance narrows the search but cannot rule ENOSPC in.

    ``describe_sqlite_error`` was originally justified by the belief that a real ENOSPC-backed
    SQLITE_FULL arrives as an ``OSError`` (carrying ``errno``) while a ceiling refusal arrives as
    an ``OperationalError``, so the two could be told apart. **Measured on this host (CPython
    3.13, SQLite 3.46.1), that is false:** filling a 1 MiB tmpfs — a real ``ENOSPC``, confirmed
    by ``OSError: [Errno 28] No space left on device`` from a raw file write to the same mount —
    raises ``sqlite3.OperationalError('database or disk is full')`` with
    ``sqlite_errorcode=13``, ``sqlite_errorname='SQLITE_FULL'``, ``errno`` **None** and
    ``isinstance(exc, OSError)`` **False**. A ``max_page_count`` ceiling on a filesystem with
    55 GiB free produces byte-identical provenance.

    So the record adds the code/name/path an operator can search on, and a non-13 code or a
    named limit still rules disk-full *out* — but the text+code pair never rules it *in*, which
    is why the operator copy keeps "check free space" as its first lead. This test pins the
    measured contract so the false discriminator cannot be reintroduced.
    """
    ceiling = sqlite3.OperationalError("database or disk is full")
    # The shape a real ENOSPC actually takes out of SQLite's C layer: an OperationalError
    # with a code and no errno - NOT an OSError.
    real_enospc = sqlite3.OperationalError("database or disk is full")
    real_enospc.sqlite_errorcode = 13
    real_enospc.sqlite_errorname = "SQLITE_FULL"

    # Both land in the disk bucket by prose - the classification is identical...
    assert is_disk_full_error(ceiling) is True and is_disk_full_error(real_enospc) is True

    # ...and neither carries an errno, so nothing in the record separates them. That is the
    # measured truth, not a shortcoming of the formatter.
    for exc in (ceiling, real_enospc):
        assert isinstance(exc, OSError) is False
        assert getattr(exc, "errno", None) is None
        assert "errno=" not in describe_sqlite_error(exc)

    detail = describe_sqlite_error(real_enospc)
    assert detail.startswith("OperationalError sqlite_errorcode=13 sqlite_errorname=SQLITE_FULL:")
    assert str(describe_sqlite_error(ceiling)) == "OperationalError: database or disk is full"


def test_describe_sqlite_error_surfaces_errno_when_the_os_layer_raises_one():
    """An ``OSError`` that does carry ENOSPC still prints it - that path exists, and is useful.

    It is the shape another caller may hold (an ``OSError`` raised around the SQLite call rather
    than by it), so the formatter must not drop the errno. This is a formatter contract only; it
    is NOT evidence that SQLite itself produces that shape for a full disk (see the test above).
    """
    enospc = OSError(errno.ENOSPC, "database or disk is full")
    assert is_disk_full_error(enospc) is True
    detail = describe_sqlite_error(enospc)
    assert detail.startswith(f"OSError errno={errno.ENOSPC}:") and f"errno={errno.ENOSPC}" in detail


def test_describe_sqlite_error_never_raises_on_odd_input():
    class _OddError(Exception):
        sqlite_errorcode = None
        sqlite_errorname = None

    assert describe_sqlite_error(None) == "none"
    assert describe_sqlite_error("plain string") == "str: plain string"
    assert describe_sqlite_error(RuntimeError("boom")) == "RuntimeError: boom"
    # An exception that carries the attributes as None must not print "None" as a code.
    assert "sqlite_errorcode" not in describe_sqlite_error(_OddError("boom"))

