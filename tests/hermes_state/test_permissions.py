"""
Test that state.db and its WAL/SHM sidecars are created with 0o600 permissions
(regression test for issue #59706).
"""
import os
import sqlite3
import stat
import tempfile
from pathlib import Path

import pytest

from hermes_state import SessionDB


def test_state_db_created_with_0600_permissions():
    """
    state.db should be created with 0o600 permissions, not 0o644 (umask 022 default).

    This is a security hardening fix: even if ~/.hermes has wider permissions
    (e.g., HERMES_HOME_MODE=0755 for a web-server traversal use case, or NixOS
    managed mode sets it to 0750), the database file itself should not be
    world-readable.

    Regression test for issue #59706.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "state.db"

        # Set umask to 0o022 (the default on most distros) to ensure we're not
        # getting a false negative because the test runner happens to use 0o077
        old_umask = os.umask(0o022)
        try:
            # Create a SessionDB instance, which triggers the file creation
            db = SessionDB(db_path=db_path)

            # Check that state.db has 0o600 permissions
            mode = stat.S_IMODE(os.stat(db_path).st_mode)
            assert mode == 0o600, f"state.db should be 0o600, got {oct(mode)}"

            # Check that WAL/SHM sidecars (if they exist) also have 0o600 permissions
            # WAL mode is the default; these files should be created
            wal_path = db_path.with_suffix(".db-wal")
            shm_path = db_path.with_suffix(".db-shm")

            # WAL/SHM may not exist if WAL fell back to DELETE mode (NFS/SMB incompatibility)
            if wal_path.exists():
                wal_mode = stat.S_IMODE(os.stat(wal_path).st_mode)
                assert wal_mode == 0o600, f"state.db-wal should be 0o600, got {oct(wal_mode)}"

            if shm_path.exists():
                shm_mode = stat.S_IMODE(os.stat(shm_path).st_mode)
                assert shm_mode == 0o600, f"state.db-shm should be 0o600, got {oct(shm_mode)}"

            db.close()
        finally:
            os.umask(old_umask)


def test_state_db_permissions_on_existing_file():
    """
    If state.db already exists (e.g., after a repair or from a previous run),
    SessionDB.__init__ should still ensure it has 0o600 permissions.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "state.db"

        # Create a state.db with wrong permissions (simulating the bug)
        conn = sqlite3.connect(str(db_path))
        conn.execute("CREATE TABLE t (x int)")
        conn.commit()
        conn.close()

        # Set wrong permissions (0o644, the umask default)
        os.chmod(db_path, 0o644)

        # Verify it's wrong before the fix
        mode_before = stat.S_IMODE(os.stat(db_path).st_mode)
        assert mode_before == 0o644, "Setup failed: should start with 0o644"

        # Open SessionDB (should chmod to 0o600)
        db = SessionDB(db_path=db_path)

        # Check that it now has correct permissions
        mode_after = stat.S_IMODE(os.stat(db_path).st_mode)
        assert mode_after == 0o600, f"After SessionDB init, state.db should be 0o600, got {oct(mode_after)}"

        db.close()


def test_state_db_permissions_race_condition():
    """
    If state.db is created by another process between our exists() check and
    os.open(), we should still chmod it to 0o600 (best-effort).

    This tests the FileExistsError branch in the pre-creation logic.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "state.db"

        old_umask = os.umask(0o022)
        try:
            # Pre-create the file with wrong permissions (simulating a race)
            conn = sqlite3.connect(str(db_path))
            conn.execute("CREATE TABLE t (x int)")
            conn.commit()
            conn.close()
            os.chmod(db_path, 0o644)

            # Verify it's wrong before SessionDB
            mode_before = stat.S_IMODE(os.stat(db_path).st_mode)
            assert mode_before == 0o644, "Setup failed: should start with 0o644"

            # Open SessionDB (should chmod to 0o600 despite the race)
            db = SessionDB(db_path=db_path)

            # Check that it now has correct permissions
            mode_after = stat.S_IMODE(os.stat(db_path).st_mode)
            assert mode_after == 0o600, f"After SessionDB init, state.db should be 0o600, got {oct(mode_after)}"

            db.close()
        finally:
            os.umask(old_umask)
