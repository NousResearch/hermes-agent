"""#110601 — close the lstat→chmod tightening window in ``_secure_state_db_files``.

Pre-fix, an existing file was classified first (``os.lstat``: skip symlinks and
non-regular nodes) and only then ``chmod(0o600)`` landed, leaving a world-readable
0644 ``state.db`` observable to concurrent processes for the whole classification
window. The chmod now runs first and unconditionally: ``chmod(2)`` never opens the
file (so it leaves POSIX lock state untouched) and only narrows access, so following
a planted symlink tightens the target instead of leaking — the lstat stays as a
logged diagnostic, not a gate.
"""

from __future__ import annotations

import os
import stat

import pytest

from hermes_state import _secure_state_db_files


def _mode(path) -> int:
    return stat.S_IMODE(os.lstat(path).st_mode)


def test_regular_0644_db_is_tightened(tmp_path):
    db = tmp_path / "state.db"
    db.write_bytes(bytes(4096))
    os.chmod(db, 0o644)

    _secure_state_db_files(db)

    assert _mode(db) == 0o600


def test_symlink_target_is_tightened_and_logged(tmp_path, caplog):
    """The chmod runs before the symlink classification (#110601).

    Pre-fix, the lstat gate skipped symlinks entirely — a planted symlink kept its
    0644 target untouched, which is the leak direction. Now the tighten follows the
    link (0600 only narrows) and the symlink is logged.
    """
    target = tmp_path / "real-state.db"
    target.write_bytes(bytes(4096))
    os.chmod(target, 0o644)
    link = tmp_path / "state.db"
    os.symlink(target.name, link)

    with caplog.at_level("WARNING", logger="hermes_state"):
        _secure_state_db_files(link)

    assert _mode(target) == 0o600
    assert any("symlink" in rec.message for rec in caplog.records)


def test_missing_sidecars_are_ignored(tmp_path):
    db = tmp_path / "state.db"
    db.write_bytes(bytes(4096))
    os.chmod(db, 0o644)
    # No -wal/-shm sidecars on disk.

    _secure_state_db_files(db)

    assert _mode(db) == 0o600
    assert not (tmp_path / "state.db-wal").exists()


def test_create_main_uses_o_excl_0600(tmp_path):
    db = tmp_path / "state.db"

    _secure_state_db_files(db, create_main=True)

    assert db.exists()
    assert _mode(db) == 0o600


def test_create_main_never_overwrites_existing(tmp_path):
    db = tmp_path / "state.db"
    db.write_bytes(bytes(4096))
    os.chmod(db, 0o640)

    _secure_state_db_files(db, create_main=True)

    # O_EXCL: the existing inode (and its bytes) survives; then it is tightened.
    assert db.read_bytes() == bytes(4096)
    assert _mode(db) == 0o600
