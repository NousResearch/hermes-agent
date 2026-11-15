"""Writable state.db hardening fails closed on symlinked main/-wal/-shm paths (#109857)."""

import errno
import os
import sqlite3
import stat
from pathlib import Path

import pytest

from hermes_state import SessionDB, _secure_state_db_files

pytestmark = pytest.mark.platforms("posix")  # POSIX symlinks and permission bits


class TestSecureStateDbSymlinkFailClosed:
    """A planted state.db/-wal/-shm symlink must stop the writable open (#109857)."""

    @pytest.mark.parametrize("planted_name", ["state.db-wal", "state.db-shm"])
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

    def test_swap_after_lstat_cannot_chmod_replacement_target(
        self, tmp_path, monkeypatch
    ):
        """The chmod must never re-resolve the pathname to a swapped-in link.

        lstat() validates one inode; a bare chmod(path) would follow a
        replacement symlink planted after the check and change the mode of an
        unrelated target (#109857). The swap happens at the pin (os.open, Linux)
        or the chmod (elsewhere): the test runner's home-I/O guard re-enters a
        patched os.lstat, which would land the swap before the check instead.
        """
        db_path = tmp_path / "state.db"
        db_path.write_bytes(b"regular")
        target = tmp_path / "unrelated"
        target.write_bytes(b"unrelated")
        os.chmod(target, 0o666)
        swapped = []

        def swap_then(real):
            def call(path, *args, **kwargs):
                if not swapped and Path(path) == db_path:
                    swapped.append(path)
                    db_path.unlink()
                    db_path.symlink_to(target)
                return real(path, *args, **kwargs)
            return call

        monkeypatch.setattr(os, "open", swap_then(os.open))
        monkeypatch.setattr(os, "chmod", swap_then(os.chmod))
        if hasattr(os, "O_PATH"):
            # Linux: the O_PATH|O_NOFOLLOW pin sees the swapped-in link and
            # fails closed before the caller's sqlite3.connect() follows it.
            with pytest.raises(OSError) as exc_info:
                _secure_state_db_files(db_path)
            assert exc_info.value.errno == errno.ELOOP
        else:
            _secure_state_db_files(db_path)

        assert swapped
        assert stat.S_IMODE(target.stat().st_mode) == 0o666

    @pytest.mark.parametrize("foreign", ["link", "target"])
    def test_main_db_link_owned_by_another_user_fails_closed(
        self, tmp_path, monkeypatch, foreign
    ):
        """A state.db link counts as the operator's only when the link and its
        target are both this user's; either end owned by someone else is planted."""
        target = tmp_path / "target.db"
        sqlite3.connect(target).close()
        os.chmod(target, 0o666)
        link = tmp_path / "state.db"
        link.symlink_to(target)

        def owned_by_another_user(real, end):
            def call(path, *args, **kwargs):
                st = real(path, *args, **kwargs)
                if end == foreign and Path(path) == link:
                    fields = list(st[:10])
                    fields[4] = os.geteuid() + 1  # st_uid
                    return os.stat_result(fields)
                return st
            return call

        monkeypatch.setattr(os, "lstat", owned_by_another_user(os.lstat, "link"))
        monkeypatch.setattr(os, "stat", owned_by_another_user(os.stat, "target"))
        with pytest.raises(OSError) as exc_info:
            _secure_state_db_files(link, create_main=True)

        assert exc_info.value.errno == errno.ELOOP
        assert stat.S_IMODE(target.stat().st_mode) == 0o666

    def test_session_db_open_refuses_dangling_link(self, tmp_path):
        """End to end: a state.db link whose target is missing (an unmounted
        volume) fails closed instead of SQLite creating a store behind it."""
        target = tmp_path / "unmounted" / "state.db"
        target.parent.mkdir()
        link = tmp_path / "state.db"
        link.symlink_to(target)

        with pytest.raises(OSError):
            SessionDB(db_path=link)

        assert not target.exists()

    def test_registry_refuses_dangling_link_before_resolving_it(self, tmp_path):
        """The shared registry (gateway, CLI) resolves the path before SessionDB
        sees it, so it must judge the link first: a dangling one never gets a
        store created behind it."""
        import hermes_state_registry

        target = tmp_path / "unmounted" / "state.db"
        target.parent.mkdir()
        link = tmp_path / "state.db"
        link.symlink_to(target)

        with pytest.raises(OSError) as exc_info:
            hermes_state_registry.acquire(link)

        assert exc_info.value.errno == errno.ELOOP
        assert not target.exists()

    def test_registry_opens_through_operator_store_link(self, tmp_path):
        import hermes_state_registry

        store = tmp_path / "dataset" / "state.db"
        store.parent.mkdir()
        SessionDB(db_path=store).close()
        link = tmp_path / "state.db"
        link.symlink_to(store)

        db = hermes_state_registry.acquire(link)
        try:
            assert Path(db.db_path) == store.resolve()
        finally:
            hermes_state_registry.release_or_close(db)

    def test_session_db_opens_through_operator_store_link(self, tmp_path):
        """A store the operator moved to other storage and linked back
        (#145075) keeps working: sessions land in the link's target."""
        store = tmp_path / "dataset" / "state.db"
        store.parent.mkdir()
        SessionDB(db_path=store).close()
        link = tmp_path / "home" / "state.db"
        link.parent.mkdir()
        link.symlink_to(store)

        db = SessionDB(db_path=link)
        try:
            db.create_session(session_id="through-link", source="cli")
        finally:
            db.close()

        with sqlite3.connect(store) as conn:
            ids = [row[0] for row in conn.execute("SELECT id FROM sessions")]
        assert "through-link" in ids
        assert link.is_symlink()
