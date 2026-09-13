"""Permission hardening must not retire another live SQLite generation."""

import errno
import os
import stat
import subprocess
import sys

import pytest

from hermes_state import SessionDB, _secure_state_db_files


@pytest.mark.macos_only
def test_permission_hardening_preserves_wal_across_processes(tmp_path):
    path = tmp_path / "state.db"
    first = SessionDB(path)
    second = SessionDB(path)
    try:
        identities = {suffix: os.stat(str(path) + suffix).st_ino for suffix in ("-wal", "-shm")}
        for _ in range(3):
            _secure_state_db_files(path, create_main=True)
            subprocess.run(
                [sys.executable, "-c", "import sqlite3,sys; c=sqlite3.connect(sys.argv[1]); "
                 "c.execute('SELECT count(*) FROM sessions').fetchone(); c.close()", str(path)],
                check=True, capture_output=True, timeout=10,
            )
            assert {suffix: os.stat(str(path) + suffix).st_ino for suffix in identities
                    if os.path.exists(str(path) + suffix)} == identities
            first._raise_if_db_replaced()
            first._conn.execute("BEGIN IMMEDIATE")
            first._conn.rollback()
    finally:
        second.close()
        first.close()


@pytest.mark.macos_only
@pytest.mark.parametrize("suffix", ["", "-wal", "-shm"])
def test_permission_hardening_keeps_private_creation_and_symlink_rejection(tmp_path, suffix):
    path = tmp_path / "state.db"
    old_umask = os.umask(0)
    try:
        _secure_state_db_files(path, create_main=True)
    finally:
        os.umask(old_umask)
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    sidecar = path.with_name(path.name + suffix)
    sidecar.unlink(missing_ok=True)
    target = tmp_path / "unrelated"
    target.write_text("keep")
    target.chmod(0o644)
    sidecar.symlink_to(target)
    with pytest.raises(OSError) as caught:
        _secure_state_db_files(path, create_main=True)
    assert caught.value.errno == errno.ELOOP
    assert stat.S_IMODE(target.stat().st_mode) == 0o644
    assert target.read_text() == "keep"
