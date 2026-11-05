"""Registry lifecycle on real stores: close and reacquire, concurrent cold acquire, profile delete
and recreate, and inode replacement, each with the registry census checked.
"""

import os
import shutil
import threading
import time
from pathlib import Path

import pytest

from hermes_cli import sqlite_safe_read
from hermes_state_errors import StateDbReplacedError
from persistence.sqlite import _registry as registry
from persistence.sqlite._holder import StoreHolder


@pytest.fixture(autouse=True)
def _clean_registry():
    """Isolate the process-global registry between tests."""
    registry.close_all()
    registry._generations.clear()
    registry._retired.clear()
    registry._opening.clear()
    registry._tearing_down.clear()
    registry._path_lifecycle_locks.clear()
    yield
    registry.close_all()
    registry._generations.clear()
    registry._retired.clear()
    registry._opening.clear()
    registry._tearing_down.clear()


@pytest.fixture
def home():
    return Path(os.environ["HERMES_HOME"])


def _census(live, retired, refcounts):
    return {"live_generations": live, "retired_generations": retired, "total_refcounts": refcounts}


def _live_connections(path):
    """Tracked connections this process holds on *path* (writer plus pooled readers)."""
    with sqlite_safe_read._live_lock:
        return sqlite_safe_read._live_connections.get(sqlite_safe_read._key(path), 0)


def test_close_releases_one_reference_and_reacquire_reads_the_data(home):
    path = home / "state.db"
    first = registry.acquire(path)
    first.set_meta("k", "written-before-close")
    second = registry.acquire(path)
    assert second is first
    assert registry.stats() == _census(1, 0, 2)

    first.close()  # registry-owned: releases one reference, the connection stays
    assert second._conn is not None
    assert registry.stats() == _census(1, 0, 1)

    assert registry.release(second) is True  # final release closes physically
    assert first._conn is None
    assert registry.stats() == _census(0, 0, 0)
    assert _live_connections(path) == 0

    reopened = registry.acquire(path)
    try:
        assert reopened is not first
        assert reopened.get_meta("k") == "written-before-close"
    finally:
        assert registry.release(reopened) is True
    assert registry.stats() == _census(0, 0, 0)


def test_concurrent_cold_acquire_opens_one_holder_and_one_writer(home, monkeypatch):
    path = home / "state.db"
    callers = 16
    ready = threading.Barrier(callers + 1)
    release_open = threading.Event()
    opens, writer_conns, results, errors = [], [], [], []
    real_open, real_writer_conn = registry._open_holder, StoreHolder._open_writer_conn

    def gated_open(open_path):
        opens.append(open_path)
        assert release_open.wait(5.0)
        return real_open(open_path)

    def counting_writer_conn(self):
        writer_conns.append(self)
        return real_writer_conn(self)

    monkeypatch.setattr(registry, "_open_holder", gated_open)
    monkeypatch.setattr(StoreHolder, "_open_writer_conn", counting_writer_conn)

    def acquire():
        try:
            ready.wait()
            results.append(registry.acquire(path))
        except BaseException as exc:  # pragma: no cover - failure path
            errors.append(exc)

    threads = [threading.Thread(target=acquire) for _ in range(callers)]
    for thread in threads:
        thread.start()
    ready.wait()
    time.sleep(0.1)  # health: allow HX013 -- no signal shows a caller blocked inside acquire(); give them time to get there
    release_open.set()
    for thread in threads:
        thread.join(10.0)
        assert not thread.is_alive(), "concurrent acquire deadlocked"

    assert errors == []
    assert len(opens) == 1
    assert len(writer_conns) == 1
    assert len(results) == callers and all(db is results[0] for db in results)
    assert registry.stats() == _census(1, 0, callers)
    # Tracked connections are the one writer plus the idle pooled readers the open used.
    assert _live_connections(path) == 1 + results[0]._read_pool.qsize()
    for db in results:
        assert registry.release(db) is True
    assert results[0]._conn is None
    assert registry.stats() == _census(0, 0, 0)
    assert _live_connections(path) == 0


def test_profile_delete_and_recreate_gets_a_fresh_holder(home):
    profile = home / "profiles" / "p1"
    profile.mkdir(parents=True)
    path = profile / "state.db"
    stale = registry.acquire(path)
    stale.set_meta("who", "deleted-profile")

    assert registry.close_all_under(profile) == 1
    assert stale._conn is None
    assert not stale._shared_registry_owned
    assert registry.stats() == _census(0, 0, 0)
    assert registry.close_all_under(profile) == 0

    shutil.rmtree(profile)
    profile.mkdir(parents=True)

    fresh = registry.acquire(path)
    try:
        assert fresh is not stale
        assert fresh.get_meta("who") is None, "data from the deleted profile survived"
        fresh.set_meta("who", "recreated-profile")
        # The torn-down handle refuses instead of writing into the recreated profile's file.
        with pytest.raises(StateDbReplacedError):
            stale.set_meta("who", "stale-write")
        assert stale._conn is None
        assert fresh.get_meta("who") == "recreated-profile"
    finally:
        assert registry.release(fresh) is True
    assert registry.stats() == _census(0, 0, 0)


def test_inode_replacement_retires_the_old_generation(home):
    path = home / "state.db"
    donor = StoreHolder(home / "donor" / "state.db")
    donor.set_meta("who", "replacement")
    donor.close()

    old = registry.acquire(path)
    peer = registry.acquire(path)
    assert peer is old
    old.set_meta("who", "original")
    # Quiesce the WAL so a new connection reads the replacement's pages, not the old generation's frames.
    with old._lock:
        old._conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    os.replace(home / "donor" / "state.db", path)

    current = registry.acquire(path)
    assert current is not old
    assert registry.stats() == _census(1, 1, 1)
    assert current.get_meta("who") == "replacement"
    assert registry.acquire(path) is current  # the retired generation is never lent again
    assert registry.release(current) is True

    assert old._conn is not None, "a retired generation was closed under its holders"
    with pytest.raises(StateDbReplacedError):
        old.set_meta("who", "stale-write")
    current.set_meta("who", "written-by-new-generation")

    assert registry.release(old) is True
    assert old._conn is not None, "one holder's release tore down the other's generation"
    assert registry.stats() == _census(1, 1, 1)
    assert registry.release(peer) is True
    assert old._conn is None
    assert current._conn is not None
    assert registry.stats() == _census(1, 0, 1)
    # The retired generation's teardown left the current generation's WAL intact.
    current.set_meta("who", "written-after-retired-teardown")

    assert registry.release(current) is True
    assert current._conn is None
    assert registry.stats() == _census(0, 0, 0)
    reader = StoreHolder(path, read_only=True)
    try:
        assert reader._read_one("SELECT value FROM state_meta WHERE key = 'who'")[0] == "written-after-retired-teardown"
    finally:
        reader.close()
