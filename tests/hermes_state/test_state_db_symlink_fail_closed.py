"""Writable state.db hardening fails closed on symlinked main/-wal/-shm paths (#109857)."""

import errno
import os
import sqlite3
import stat
from pathlib import Path

import pytest

from hermes_state import SessionDB, _secure_state_db_files


class TestSecureStateDbSymlinkFailClosed:
    """A planted state.db/-wal/-shm symlink must stop the writable open (#109857)."""

    @pytest.mark.skipif(os.name == "nt", reason="POSIX symlinks and permission bits")
    @pytest.mark.parametrize(
        "planted_name", ["state.db", "state.db-wal", "state.db-shm"]
    )
    def test_planted_symlink_fails_closed(self, tmp_path, planted_name):
        """Hardening raises ELOOP instead of skipping, so the caller's
        sqlite3.connect() never follows the planted link."""
        target = tmp_path / "target.db"
        sqlite3.connect(target).close()
        os.chmod(target, 0o666)
        planted = tmp_path / planted_name
        planted.symlink_to(target)

        with pytest.raises(OSError) as exc_info:
            _secure_state_db_files(
                tmp_path / "state.db", create_main=True
            )

        assert exc_info.value.errno == errno.ELOOP
        assert stat.S_IMODE(target.stat().st_mode) == 0o666

    @pytest.mark.skipif(os.name == "nt", reason="POSIX symlinks and permission bits")
    def test_swap_after_lstat_cannot_chmod_replacement_target(
        self, tmp_path, monkeypatch
    ):
        """The chmod must never re-resolve the pathname to a swapped-in link.

        lstat() validates one inode; a bare chmod(path) would follow a
        replacement symlink planted after the check and change the mode of an
        unrelated target (#109857).
        """
        db_path = tmp_path / "state.db"
        db_path.write_bytes(b"regular")
        target = tmp_path / "unrelated"
        target.write_bytes(b"unrelated")
        os.chmod(target, 0o666)

        real_lstat = os.lstat

        def swap_after_check(path):
            st = real_lstat(path)
            if Path(path) == db_path:
                db_path.unlink()
                db_path.symlink_to(target)
            return st

        monkeypatch.setattr(os, "lstat", swap_after_check)
        if hasattr(os, "O_PATH"):
            # Linux: the O_PATH|O_NOFOLLOW pin sees the swapped-in link and
            # fails closed before the caller's sqlite3.connect() follows it.
            with pytest.raises(OSError) as exc_info:
                _secure_state_db_files(db_path)
            assert exc_info.value.errno == errno.ELOOP
        else:
            _secure_state_db_files(db_path)

        assert stat.S_IMODE(target.stat().st_mode) == 0o666

    @pytest.mark.skipif(os.name == "nt", reason="POSIX symlinks and permission bits")
    def test_session_db_open_refuses_planted_symlink(self, tmp_path):
        """End to end: the writable SessionDB open fails before sqlite3
        follows the planted link or the hardening touches its target."""
        target = tmp_path / "target.db"
        seed = sqlite3.connect(target)
        seed.execute("CREATE TABLE marker (x)")
        seed.commit()
        seed.close()
        os.chmod(target, 0o666)
        planted = tmp_path / "state.db"
        planted.symlink_to(target)

        with pytest.raises(OSError):
            SessionDB(db_path=planted)

        assert stat.S_IMODE(target.stat().st_mode) == 0o666
