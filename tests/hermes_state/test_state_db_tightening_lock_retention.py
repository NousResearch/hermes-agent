"""Regression for #109966 / #109687: tightening mode bits on an existing state.db
must not drop a live connection's POSIX locks (the #109509 -> #109841 regression).

``_secure_state_db_files`` used to ``open(O_WRONLY|O_CREAT)`` + ``close()`` an
*existing* main database file while tightening 0600.  Closing any descriptor for
a file cancels every POSIX ``fcntl`` lock this process holds on it — including
the WAL-mode SHARED lock and the ``-shm`` DMS read-mark of an already-open
SQLite connection to the same database.  With the holder's locks gone, the next
ordinary opener's close-time checkpoint sees no peer, checkpoints and unlinks
``-wal``/``-shm`` while long-lived holders (gateway, dashboard) keep using the
deleted inodes; every later opener is then refused by the deleted-WAL guard.

The helper must therefore only take a descriptor for a *brand-new* inode
(``O_EXCL``) and tighten pre-existing files with ``chmod(2)`` on the path.
"""

from __future__ import annotations

import os
import sqlite3
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from hermes_state import _secure_state_db_files
from hermes_state_dbfile import iter_deleted_sqlite_sidecar_holders


def _file_ident(path: Path) -> tuple[int, int, int]:
    st = os.stat(path)
    return (os.major(st.st_dev), os.minor(st.st_dev), st.st_ino)


def _held_locks(pid: int, inodes: set[tuple[int, int, int]]) -> set[tuple[str, str, str]]:
    """The pid's advisory locks on the given files, per ``/proc/locks``.

    Field quirks: ``MAJ:MIN`` are hex, the inode is decimal, ``RW``/``START``/``END``
    are strings on the line.
    """
    held: set[tuple[str, str, str]] = set()
    for line in Path("/proc/locks").read_text().splitlines():
        if "->" in line:  # a waiter queued behind someone else's lock
            continue
        fields = line.split()
        if len(fields) < 8 or fields[4] != str(pid):
            continue
        major, minor, inode = fields[5].split(":")
        if (int(major, 16), int(minor, 16), int(inode)) in inodes:
            held.add((fields[3], fields[6], fields[7]))
    return held


def _open_wal_writer(path: Path) -> sqlite3.Connection:
    """A long-lived WAL connection with an open read transaction (what a gateway holds)."""
    conn = sqlite3.connect(str(path), timeout=5.0, isolation_level=None)
    mode = conn.execute("PRAGMA journal_mode=WAL").fetchone()[0]
    if mode != "wal":
        conn.close()
        pytest.skip("WAL not active on this filesystem")
    conn.execute("CREATE TABLE t (x INTEGER)")
    conn.execute("INSERT INTO t VALUES (1)")
    conn.execute("BEGIN")
    conn.execute("SELECT * FROM t").fetchall()
    return conn


def _tighten_both_shapes(path: Path) -> None:
    """The two call shapes production uses: SessionDB init and the sidecar pass."""
    _secure_state_db_files(path, create_main=True)
    _secure_state_db_files(path)


# A second, ordinary opener: connect, read, close.  On the fixed tree its
# close-time checkpoint must leave the sidecars alone; on the pre-fix tree it
# believed the (lock-less) holder was gone and unlinked the live generation.
_CLOSER_CHILD = textwrap.dedent(
    """
    import sqlite3, sys
    conn = sqlite3.connect(sys.argv[1], timeout=5.0)
    conn.execute("SELECT count(*) FROM t").fetchall()
    conn.close()
    """
)


@pytest.mark.linux_only  # /proc/locks is the only view that catches the lock drop
def test_tightening_keeps_the_holders_posix_locks(tmp_path):
    path = tmp_path / "state.db"
    conn = _open_wal_writer(path)
    try:
        watched = {
            _file_ident(Path(str(path) + suffix))
            for suffix in ("", "-wal", "-shm")
            if Path(str(path) + suffix).exists()
        }
        before = _held_locks(os.getpid(), watched)
        assert before, "expected the open WAL writer to hold /proc/locks entries"

        _tighten_both_shapes(path)

        assert _held_locks(os.getpid(), watched) == before, (
            "the tightening pass dropped the holder's POSIX locks"
        )
    finally:
        conn.close()


@pytest.mark.linux_only  # the deleted-WAL guard it protects is Linux-only
def test_second_opener_cannot_unlink_sidecars_under_a_locked_holder(tmp_path):
    path = tmp_path / "state.db"
    conn = _open_wal_writer(path)
    try:
        _tighten_both_shapes(path)
        assert Path(str(path) + "-wal").exists()

        closer = subprocess.run(
            [sys.executable, "-c", _CLOSER_CHILD, str(path)],
            cwd=tmp_path,
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert closer.returncode == 0, closer.stderr

        assert Path(str(path) + "-wal").exists(), "a second opener's close unlinked the live -wal"
        assert Path(str(path) + "-shm").exists(), "a second opener's close unlinked the live -shm"
        assert iter_deleted_sqlite_sidecar_holders(path) == []
    finally:
        conn.close()
