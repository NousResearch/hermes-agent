"""Regression tests: restore paths must fail closed under an IN-PROCESS
live connection to the destination database (#90837 / #90950 own-holder gap).

The foreign-pid scan (`_foreign_db_holder_pids`) excludes the calling
process by design, so before the fix both restore paths would unlink the
destination DB and its -wal/-shm sidecars while THIS process held a tracked
connection — leaving the process on deleted-inode fds (the split-brain
fingerprint from issue #90837's field reports).
"""

import os
import sqlite3
import sys

import pytest

from hermes_cli import backup as backup_mod
from hermes_cli import update_cmd
from hermes_cli.sqlite_safe_read import connect_tracked


def _make_db(path, marker):
    conn = sqlite3.connect(str(path))
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("CREATE TABLE t (v TEXT)")
    conn.execute("INSERT INTO t VALUES (?)", (marker,))
    conn.commit()
    conn.close()


def _read_marker(path):
    conn = sqlite3.connect(str(path))
    try:
        return conn.execute("SELECT v FROM t LIMIT 1").fetchone()[0]
    finally:
        conn.close()


def _own_deleted_fds(needle):
    """Paths of this process's fds pointing at deleted files matching needle."""
    if not sys.platform.startswith("linux"):
        return []
    out = []
    fd_dir = f"/proc/{os.getpid()}/fd"
    for fd in os.listdir(fd_dir):
        try:
            target = os.readlink(f"{fd_dir}/{fd}")
        except OSError:
            continue
        if needle in target and "(deleted)" in target:
            out.append(target)
    return out


@pytest.fixture
def live_held_db(tmp_path):
    """A WAL-mode dst held open by a tracked connection in this process,
    with a corrupt header so the backup-API restore path genuinely fails
    (which is what routes _safe_restore_db into its unlink+move fallback)."""
    dst = tmp_path / "state.db"
    src = tmp_path / "snap.db"
    _make_db(src, "snapshot-good")
    _make_db(dst, "live-old")
    held = connect_tracked(
        dst, connect_fn=sqlite3.connect, check_same_thread=False
    )
    held.execute("PRAGMA journal_mode=WAL")
    held.execute("INSERT INTO t VALUES ('held-write')")
    held.commit()
    assert (tmp_path / "state.db-wal").exists()
    with open(dst, "r+b") as fh:  # offline-fixture mutation, doomed file
        fh.write(b"\x00" * 100)
    try:
        yield src, dst
    finally:
        try:
            held.close()
        except Exception:
            pass


def test_safe_restore_fallback_refuses_under_own_live_connection(live_held_db):
    src, dst = live_held_db
    wal = dst.with_name(dst.name + "-wal")
    wal_ino = wal.stat().st_ino

    assert backup_mod._safe_restore_db(src, dst) is False

    # The held generation must survive: same WAL inode, no deleted-fd ghosts.
    assert wal.exists() and wal.stat().st_ino == wal_ino
    assert _own_deleted_fds(str(dst.name)) == []


def test_update_autorestore_refuses_under_own_live_connection(
    live_held_db, capsys
):
    src, dst = live_held_db
    wal = dst.with_name(dst.name + "-wal")
    wal_ino = wal.stat().st_ino

    assert update_cmd._restore_state_db_from_snapshot(dst, src) is False

    out = capsys.readouterr().out
    assert "Auto-restore refused" in out
    assert wal.exists() and wal.stat().st_ino == wal_ino
    assert _own_deleted_fds(str(dst.name)) == []


def test_safe_restore_fallback_still_works_without_holder(tmp_path):
    dst = tmp_path / "state.db"
    src = tmp_path / "snap.db"
    _make_db(src, "snapshot-good")
    _make_db(dst, "live-old")
    with open(dst, "r+b") as fh:
        fh.write(b"\x00" * 100)

    assert backup_mod._safe_restore_db(src, dst) is True
    assert _read_marker(dst) == "snapshot-good"


def test_update_autorestore_still_works_without_holder(tmp_path):
    dst = tmp_path / "state.db"
    src = tmp_path / "snap.db"
    _make_db(src, "snapshot-good")
    _make_db(dst, "live-old")
    # Stale WAL from a crashed writer must still be cleared.
    dst.with_name(dst.name + "-wal").write_bytes(b"\x00" * 1024)
    with open(dst, "r+b") as fh:
        fh.write(b"\x00" * 100)

    assert update_cmd._restore_state_db_from_snapshot(dst, src) is True
    assert _read_marker(dst) == "snapshot-good"
    assert not dst.with_name(dst.name + "-wal").exists()
