"""The backend's writable open fails closed on symlinked state.db/-wal/-shm paths (#109857), except a state.db link the
operator made to a store of their own (#145075)."""

import errno
import os
import sqlite3
import stat
from pathlib import Path

import pytest

from persistence.conversation.sessions import NewSession
from persistence.sqlite import _registry
from persistence.sqlite._holder import StoreHolder, _secure_state_db_files
from tests.persistence_sqlite.conftest import store_for

pytestmark = pytest.mark.platforms("posix")  # POSIX symlinks and permission bits


@pytest.mark.parametrize("planted_name", ["state.db-wal", "state.db-shm"])
def test_planted_symlink_fails_closed(tmp_path, planted_name):
    """Hardening raises ELOOP instead of skipping, so the caller's sqlite3.connect() never follows the planted link."""
    target = tmp_path / "target.db"
    sqlite3.connect(target).close()
    os.chmod(target, 0o666)
    (tmp_path / planted_name).symlink_to(target)

    with pytest.raises(OSError) as exc_info:
        _secure_state_db_files(tmp_path / "state.db", create_main=True)

    assert exc_info.value.errno == errno.ELOOP
    assert stat.S_IMODE(target.stat().st_mode) == 0o666


def test_swap_after_lstat_cannot_chmod_replacement_target(tmp_path, monkeypatch):
    """The chmod never re-resolves the pathname to a swapped-in link.

    lstat() validates one inode; a bare chmod(path) would follow a replacement symlink planted after the check and
    change the mode of an unrelated target (#109857). The swap happens at the pin (os.open, Linux) or the chmod
    (elsewhere): the test runner's home-I/O guard re-enters a patched os.lstat, which would land the swap before the
    check instead.
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
        # Linux: the O_PATH|O_NOFOLLOW pin sees the swapped-in link and fails closed before the caller's
        # sqlite3.connect() follows it.
        with pytest.raises(OSError) as exc_info:
            _secure_state_db_files(db_path)
        assert exc_info.value.errno == errno.ELOOP
    else:
        _secure_state_db_files(db_path)

    assert swapped
    assert stat.S_IMODE(target.stat().st_mode) == 0o666


@pytest.mark.parametrize("foreign", ["link", "target"])
def test_main_db_link_owned_by_another_user_fails_closed(tmp_path, monkeypatch, foreign):
    """A state.db link counts as the operator's only when the link and its target are both this user's; either end
    owned by someone else is planted."""
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


def test_holder_open_refuses_dangling_link(tmp_path):
    """A state.db link whose target is missing (an unmounted volume) fails closed instead of SQLite creating a store
    behind it."""
    target = tmp_path / "unmounted" / "state.db"
    target.parent.mkdir()
    link = tmp_path / "state.db"
    link.symlink_to(target)

    with pytest.raises(OSError):
        StoreHolder(db_path=link)

    assert not target.exists()


def test_registry_refuses_dangling_link_before_resolving_it(tmp_path):
    """The shared registry resolves the path before the holder sees it, so it judges the link first: a dangling one
    never gets a store created behind it."""
    target = tmp_path / "unmounted" / "state.db"
    target.parent.mkdir()
    link = tmp_path / "state.db"
    link.symlink_to(target)

    with pytest.raises(OSError) as exc_info:
        _registry.acquire(link)

    assert exc_info.value.errno == errno.ELOOP
    assert not target.exists()


def test_registry_opens_through_operator_store_link(tmp_path):
    store = tmp_path / "dataset" / "state.db"
    store.parent.mkdir()
    StoreHolder(db_path=store).close()
    link = tmp_path / "state.db"
    link.symlink_to(store)

    holder = _registry.acquire(link)
    try:
        assert Path(holder.db_path) == store.resolve()
    finally:
        _registry.release_or_close(holder)


def test_backend_opens_through_operator_store_link(tmp_path, backend, scope_a):
    """A store the operator moved to other storage and linked back (#145075) keeps working: sessions land in the
    link's target."""
    store = tmp_path / "dataset" / "state.db"
    store.parent.mkdir()
    StoreHolder(db_path=store).close()
    link = store_for(scope_a)
    link.symlink_to(store)

    backend.open(scope_a).sessions.create(NewSession(session_id="through-link", source="cli"))
    _registry.close_all()

    with sqlite3.connect(store) as conn:
        ids = [row[0] for row in conn.execute("SELECT id FROM sessions")]
    assert "through-link" in ids
    assert link.is_symlink()
