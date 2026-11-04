"""close_all_under releases the cached header-probe fds of the homes it sweeps.

hermes_state_dbfile keeps one O_RDONLY fd per state.db for lock-safe header reads. Without
this release a multiplexed gateway keeps one such fd for every profile it has unserved, until
the process exits. The fd may only close once nothing else in the process has the inode open,
because closing any descriptor on the file cancels the process's POSIX locks on it.
"""

import os
import sqlite3
import threading
from pathlib import Path

import pytest

import hermes_state_dbfile as dbfile
import hermes_state_registry as registry


@pytest.fixture(autouse=True)
def _clean_state(tmp_path):
    registry.close_all()
    registry._generations.clear()
    registry._retired.clear()
    registry._opening.clear()
    registry._tearing_down.clear()
    yield
    registry.close_all()
    # Only this test's probes: production keeps probes open on purpose (see the module docstring).
    with dbfile._HEADER_PROBE_LOCK:
        for key in [k for k in dbfile._HEADER_PROBE_FDS if Path(k).is_relative_to(tmp_path)]:
            fd, _dev, _ino = dbfile._HEADER_PROBE_FDS.pop(key)
            os.close(fd)


def _probe(db_path):
    assert dbfile._pread_db_header(db_path, 16) == b"SQLite format 3\x00"
    return dbfile._HEADER_PROBE_FDS[str(db_path)][0]


def _is_open(fd):
    try:
        os.fstat(fd)
    except OSError:
        return False
    return True


def _home(tmp_path, name):
    home = tmp_path / "profiles" / name
    home.mkdir(parents=True)
    db_path = home / "state.db"
    registry.release(registry.acquire(db_path))  # creates the schema; final release closes it
    return home, db_path


@pytest.mark.platforms("posix")
def test_sweep_closes_the_probe_fd_of_a_closed_home(tmp_path):
    home, db_path = _home(tmp_path, "work")
    db = registry.acquire(db_path)
    fd = _probe(db_path)

    assert registry.close_all_under(home) == 1
    assert db._conn is None
    assert str(db_path) not in dbfile._HEADER_PROBE_FDS
    assert not _is_open(fd)


@pytest.mark.platforms("posix")
def test_probe_left_by_an_earlier_close_is_released(tmp_path):
    """The gateway case: the store already closed, only the probe fd remains."""
    home, db_path = _home(tmp_path, "parked")
    fd = _probe(db_path)

    assert registry.close_all_under(home) == 0
    assert not _is_open(fd)


@pytest.mark.platforms("posix")
def test_probe_kept_while_another_descriptor_holds_the_inode(tmp_path):
    """A connection the registry cannot see (a bare sqlite3 one) keeps the probe open."""
    home, db_path = _home(tmp_path, "busy")
    fd = _probe(db_path)
    conn = sqlite3.connect(db_path)
    conn.execute("SELECT 1 FROM sqlite_master").fetchall()
    try:
        registry.close_all_under(home)
        assert _is_open(fd)
        assert str(db_path) in dbfile._HEADER_PROBE_FDS
    finally:
        conn.close()

    registry.close_all_under(home)
    assert not _is_open(fd)


@pytest.mark.platforms("posix")
def test_probe_kept_when_a_tracked_connection_opens_after_the_scan(tmp_path, monkeypatch):
    """A tracked connection the fd scan missed (it opened after the scan) keeps the probe open."""
    home, db_path = _home(tmp_path, "tracked")
    fd = _probe(db_path)
    conn = dbfile._connect_tracked_db(db_path)
    monkeypatch.setattr(dbfile, "inodes_with_other_descriptors", lambda identities: set(), raising=False)
    try:
        registry.close_all_under(home)
        assert _is_open(fd)
        assert str(db_path) in dbfile._HEADER_PROBE_FDS
    finally:
        conn.close()


@pytest.mark.platforms("posix")
def test_probe_kept_when_a_writer_is_published_after_the_sweep(tmp_path, monkeypatch):
    """The #127811 shape: a caller acquires the path after the sweep tore it down."""
    home, db_path = _home(tmp_path, "reopened")
    fd = _probe(db_path)
    late: list = []
    sweep = registry._teardown_swept_generations

    def _sweep_then_reopen(*args, **kwargs):
        closed = sweep(*args, **kwargs)
        late.append(registry.acquire(db_path))
        return closed

    monkeypatch.setattr(registry, "_teardown_swept_generations", _sweep_then_reopen)
    registry.close_all_under(home)
    assert late and late[0]._conn is not None
    assert _is_open(fd)
    assert str(db_path) in dbfile._HEADER_PROBE_FDS
    registry.release(late[0])


@pytest.mark.platforms("posix")
def test_probe_kept_while_an_open_is_in_flight(tmp_path, monkeypatch):
    """An acquire that is mid-open when the release runs keeps the probe open."""
    home, db_path = _home(tmp_path, "opening")
    fd = _probe(db_path)
    entered, resume = threading.Event(), threading.Event()
    real_open = registry._open_session_db

    def _paused_open(path):
        entered.set()
        assert resume.wait(10.0)
        return real_open(path)

    monkeypatch.setattr(registry, "_open_session_db", _paused_open)
    opened: list = []
    opener = threading.Thread(target=lambda: opened.append(registry.acquire(db_path)), daemon=True)
    opener.start()
    try:
        assert entered.wait(5.0)
        assert registry._opening.get(db_path.resolve()) is not None
        sweeper = threading.Thread(target=registry.close_all_under, args=(home,), daemon=True)
        sweeper.start()
        sweeper.join(0.5)  # it waits on the lifecycle lock the opener holds
        resume.set()
        opener.join(10.0)
        sweeper.join(10.0)
        assert not sweeper.is_alive()
    finally:
        resume.set()
    assert opened and opened[0]._conn is not None
    assert _is_open(fd)
    registry.release(opened[0])


@pytest.mark.platforms("posix")
def test_probes_outside_the_swept_home_are_untouched(tmp_path):
    home, _db_path = _home(tmp_path, "one")
    _other_home, other_db = _home(tmp_path, "two")
    other_fd = _probe(other_db)

    registry.close_all_under(home)
    assert _is_open(other_fd)
    assert str(other_db) in dbfile._HEADER_PROBE_FDS


@pytest.mark.platforms("posix")
def test_two_spellings_of_one_path_do_not_pin_each_other(tmp_path):
    """Probes hold no locks, so a second probe on the same inode must not keep the first open."""
    home, db_path = _home(tmp_path, "aliased")
    link = tmp_path / "alias"
    link.symlink_to(home)
    fd = _probe(db_path)
    alias_fd = _probe(link / "state.db")

    registry.close_all_under(home)
    assert not _is_open(fd)
    assert not _is_open(alias_fd)
