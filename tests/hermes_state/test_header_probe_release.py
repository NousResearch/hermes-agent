"""``close_all_under`` releases cached header-probe fds for state.db files under the directory (#149274).

The write-path probe (``_pread_db_range``) caches one O_RDONLY fd per state.db path for the life
of the process, because closing it while this process holds SQLite POSIX locks on the same inode
would cancel them (#100519). Once ``close_all_under`` has torn down every connection under a
profile home, no such locks remain, so the cached fd must be released: an unserved or deleted
profile should leave no handle behind.
"""

import os
import sqlite3
from pathlib import Path

import pytest

import hermes_state_dbfile as dbfile
from hermes_state_registry import close_all_under


@pytest.fixture(autouse=True)
def _restore_probe_cache():
    """Leave the module-level probe cache exactly as found (mirrors tests/gateway/conftest.py)."""
    existing = set(dbfile._HEADER_PROBE_FDS)
    yield
    for key in set(dbfile._HEADER_PROBE_FDS) - existing:
        fd, _dev, _ino = dbfile._HEADER_PROBE_FDS.pop(key)
        try:
            os.close(fd)
        except OSError:
            pass


def _make_db(path: Path) -> None:
    conn = sqlite3.connect(path)
    conn.execute("CREATE TABLE t(x)")
    conn.commit()
    conn.close()


def _probe(path: Path) -> int:
    dbfile._read_sqlite_application_id(path)
    return dbfile._HEADER_PROBE_FDS[str(path)][0]


def _fd_open(fd: int) -> bool:
    try:
        os.fstat(fd)
        return True
    except OSError:
        return False


def test_close_all_under_releases_probe_fd(tmp_path):
    home = tmp_path / "profiles" / "p1"
    home.mkdir(parents=True)
    db = home / "state.db"
    _make_db(db)
    fd = _probe(db)
    assert _fd_open(fd)

    close_all_under(home)

    assert str(db) not in dbfile._HEADER_PROBE_FDS
    assert not _fd_open(fd)


def test_probe_fd_outside_directory_survives(tmp_path):
    home_a = tmp_path / "profiles" / "a"
    home_b = tmp_path / "profiles" / "b"
    home_a.mkdir(parents=True)
    home_b.mkdir(parents=True)
    db_a, db_b = home_a / "state.db", home_b / "state.db"
    _make_db(db_a)
    _make_db(db_b)
    _probe(db_a)
    fd_b = _probe(db_b)

    close_all_under(home_a)

    assert str(db_b) in dbfile._HEADER_PROBE_FDS
    assert _fd_open(fd_b)


def test_sibling_profile_prefix_is_not_released(tmp_path):
    """Path scoping is by directory boundary: ``profiles/ab`` is not under ``profiles/a``."""
    home_a = tmp_path / "profiles" / "a"
    home_ab = tmp_path / "profiles" / "ab"
    home_a.mkdir(parents=True)
    home_ab.mkdir(parents=True)
    db_ab = home_ab / "state.db"
    _make_db(db_ab)
    fd_ab = _probe(db_ab)

    close_all_under(home_a)

    assert str(db_ab) in dbfile._HEADER_PROBE_FDS
    assert _fd_open(fd_ab)


def test_release_header_probes_under_returns_count(tmp_path):
    home = tmp_path / "home"
    home.mkdir()
    db = home / "state.db"
    _make_db(db)
    _probe(db)

    assert dbfile.release_header_probes_under(home) == 1
    assert dbfile.release_header_probes_under(home) == 0


def test_probe_fd_on_inode_with_live_connection_elsewhere_survives(tmp_path):
    """Hard-linked state.db: the probe fd under the torn-down directory shares an inode with a
    live connection's file elsewhere — closing it would cancel that connection's POSIX locks."""
    home_a = tmp_path / "profiles" / "a"
    home_a.mkdir(parents=True)
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    db_a = home_a / "state.db"
    db_b = elsewhere / "state.db"
    _make_db(db_a)
    os.link(db_a, db_b)

    from hermes_state_registry import acquire
    live = acquire(db_b)
    try:
        fd_a = _probe(db_a)
        close_all_under(home_a)
        assert str(db_a) in dbfile._HEADER_PROBE_FDS
        assert _fd_open(fd_a)
    finally:
        live.close() if hasattr(live, "close") else None


def test_raw_untracked_connection_on_same_inode_blocks_release(tmp_path):
    """A raw sqlite3 connection (invisible to the registry, #149276) on the same inode still
    appears in the /proc/self/fd inventory and must block the close."""
    home_a = tmp_path / "profiles" / "a"
    home_a.mkdir(parents=True)
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    db_a = home_a / "state.db"
    db_b = elsewhere / "state.db"
    _make_db(db_a)
    os.link(db_a, db_b)

    raw = sqlite3.connect(db_b)
    try:
        fd_a = _probe(db_a)
        close_all_under(home_a)
        assert str(db_a) in dbfile._HEADER_PROBE_FDS
        assert _fd_open(fd_a)
    finally:
        raw.close()


def test_release_without_fd_inventory_closes_nothing(tmp_path, monkeypatch):
    """Where the open-fd inventory is unavailable, the bounded leak is kept over an unsafe close."""
    home = tmp_path / "home"
    home.mkdir()
    db = home / "state.db"
    _make_db(db)
    fd = _probe(db)
    monkeypatch.setattr(dbfile, "_self_open_inodes", lambda exclude: None)

    assert dbfile.release_header_probes_under(home) == 0
    assert str(db) in dbfile._HEADER_PROBE_FDS
    assert _fd_open(fd)
