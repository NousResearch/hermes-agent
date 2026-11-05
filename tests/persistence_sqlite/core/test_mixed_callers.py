"""Legacy and backend callers in one process on one ``state.db``. A SessionDB serves through the backend's store
holder, and both sides take their long-lived handles from the backend's registry, so the process holds one writer
connection per file whichever side opened it, and one release, eviction or forced close reaches both sides. Writers
that stay separate (a SessionDB on a holder opened outside the registry) still share one WAL lock guard, one read
budget, one corruption latch, one repair claim and one leak-sweep registry per file. These hold through close and
reacquire, profile delete and recreate, inode replacement and pending accounting. Linux-only: OFD locks + ``/proc``.
"""

import os
import shutil
import sqlite3
import subprocess
import sys
from pathlib import Path

import pytest

import hermes_state
import hermes_state_lockguard as lockguard
import hermes_state_registry as legacy_registry
import hermes_state_wal
import persistence_assembly
from hermes_state_errors import StateDbCorruptError, StateDbReplacedError
from hermes_state_guard import _test_instance_registry
from hermes_state_health import reset_storage_state
from hermes_state_readpool import _read_budget_for
from hermes_state_repair import _claim_repair_attempt
from persistence import StorageState
from persistence.sqlite import _holder
from persistence.sqlite import _registry as backend_registry
from persistence.sqlite.admin import SQLiteStoreAdmin
from persistence.sqlite.services import SQLiteStorageHealth

pytestmark = pytest.mark.platforms("linux")

SESSION = "mixed-1"


@pytest.fixture(autouse=True)
def _wal_and_empty_registries(monkeypatch):
    """Pin WAL (this host's SQLite may be reset-vulnerable) and leave the registry empty."""
    monkeypatch.setattr(hermes_state_wal, "is_sqlite_wal_reset_vulnerable", lambda version_info=None: False)
    monkeypatch.setattr(hermes_state_wal, "resolve_journal_mode", lambda: "wal")
    yield
    legacy_registry.close_all()
    backend_registry.close_all()


@pytest.fixture
def home():
    return Path(os.environ["HERMES_HOME"])


@pytest.fixture
def path(home):
    return home / "state.db"


def _open_both(path):
    """The long-lived handle each side takes from the registry."""
    legacy, backend = legacy_registry.acquire(path), backend_registry.acquire(path)
    assert legacy._wal_active and backend._wal_active
    return legacy, backend


def _open_separate(path):
    """A SessionDB on a writer of its own beside the backend's registry handle."""
    legacy, backend = hermes_state.SessionDB(holder=_holder.StoreHolder(path)), backend_registry.acquire(path)
    assert legacy._wal_active and backend._wal_active
    assert legacy._conn is not backend._conn
    return legacy, backend


def _identity(path):
    st = os.stat(path)
    return st.st_dev, st.st_ino


def _guards(path):
    return lockguard._HANDLES.get(_identity(path), 0)


def _sibling_takes_exclusive(path) -> bool:
    """Another process tries the EXCLUSIVE a close-time WAL reset needs; True = nothing in this process guards."""
    code = (
        "import fcntl, os, struct, sys\n"
        f"fd = os.open({str(path)!r}, os.O_RDWR)\n"
        "lk = struct.pack('@hhqqi', fcntl.F_WRLCK, 0, 0x40000002, 510, 0)\n"
        "try:\n    fcntl.fcntl(fd, 37, lk); print('EXCLUSIVE_ACQUIRED')\n"
        "except BlockingIOError:\n    print('REFUSED')\n"
    )
    return "EXCLUSIVE_ACQUIRED" in subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True).stdout


def _drop_idle_readers(*holders):
    for holder in holders:
        while holder._evict_one_idle_read_conn():
            pass


def _read(holder):
    with holder._read_ctx() as conn:
        return conn.execute("SELECT value FROM state_meta WHERE key = 'k'").fetchone()


@pytest.mark.parametrize("first", ["backend", "legacy"])
def test_one_lock_guard_counts_both_writers_and_only_the_last_release_lifts_it(path, first):
    legacy, backend = _open_separate(path)
    legacy.set_meta("k", "legacy")
    backend.set_meta("k", "backend")
    assert _guards(path) == 2
    assert not _sibling_takes_exclusive(path)

    sides = {"legacy": (legacy, legacy.close), "backend": (backend, lambda: backend_registry.release(backend))}
    released, release = sides[first]
    survivor, survivor_release = sides["legacy" if first == "backend" else "backend"]
    release()
    assert released._conn is None
    assert _guards(path) == 1
    assert not _sibling_takes_exclusive(path), "one side's last release unlocked the other side's descriptors"
    survivor.set_meta("k", "after")

    survivor_release()
    assert _guards(path) == 0
    assert _sibling_takes_exclusive(path)


def test_both_sides_draw_on_one_read_budget_for_the_file(path):
    legacy, backend = _open_separate(path)
    budget = _read_budget_for(path)
    assert legacy._read_budget is budget and backend._read_budget is budget
    _drop_idle_readers(legacy, backend)
    _read(legacy)
    assert legacy._read_pool.qsize() == 1

    taken = 0
    while budget.permits.acquire(blocking=False):
        taken += 1
    try:
        _read(backend)
        # The file's last permit sat with the legacy store's idle reader; the backend reclaimed it.
        assert legacy._read_pool.qsize() == 0
        assert backend._read_pool.qsize() == 1
    finally:
        for _ in range(taken):
            budget.permits.release()


@pytest.mark.parametrize("reporter", ["legacy", "backend"])
def test_corruption_one_side_reports_quarantines_the_other(path, reporter):
    legacy, backend = _open_separate(path)
    sides = {"legacy": legacy, "backend": backend}
    other = sides["backend" if reporter == "legacy" else "legacy"]
    try:
        with pytest.raises(StateDbCorruptError):
            sides[reporter]._halt_db_corrupt(sqlite3.DatabaseError("database disk image is malformed"))
        assert SQLiteStorageHealth(path).storage_state() is StorageState.CORRUPT
        with pytest.raises(StateDbCorruptError, match="reported earlier in this process"):
            other.set_meta("k", "refused")
    finally:
        reset_storage_state(path)


def test_the_one_repair_attempt_per_file_is_claimed_once_for_both_sides(path):
    try:
        assert _holder._claim_repair_attempt(path) is True
        assert _claim_repair_attempt(path) is False
    finally:
        hermes_state._repair_attempted_paths.discard(str(path))


def test_the_registry_lends_one_writer_and_the_backend_vacuums_through_it(path):
    legacy, backend = _open_both(path)
    assert legacy_registry.acquire(path) is legacy
    assert backend_registry.acquire(path) is backend
    assert legacy._holder is backend
    assert legacy_registry.stats()["live_generations"] == backend_registry.stats()["live_generations"] == 1
    assert backend in _test_instance_registry
    assert SQLiteStoreAdmin(path, read_only=False).vacuum_blockers() == ()


def test_closing_and_reacquiring_the_backend_keeps_the_legacy_writer_and_the_shared_state(path):
    legacy, backend = _open_separate(path)
    legacy.set_meta("k", "legacy")
    assert backend_registry.release(backend) is True
    assert backend._conn is None and legacy._conn is not None
    assert _guards(path) == 1

    reacquired = backend_registry.acquire(path)
    assert reacquired is not backend
    assert reacquired.get_meta("k") == "legacy"
    reacquired.set_meta("k", "backend")
    assert legacy.get_meta("k") == "backend"
    assert _guards(path) == 2
    assert reacquired._read_budget is legacy._read_budget is _read_budget_for(path)


def test_releasing_one_side_keeps_the_shared_writer_for_the_other(path):
    legacy, backend = _open_both(path)
    legacy.set_meta("k", "legacy")
    assert backend_registry.release(backend) is True
    assert backend._conn is not None and legacy._conn is backend._conn
    assert legacy.get_meta("k") == "legacy"
    assert _guards(path) == 1

    assert backend_registry.acquire(path) is backend
    legacy.close()
    assert backend._conn is not None
    assert backend_registry.release(backend) is True
    assert backend._conn is None and legacy._conn is None
    assert _guards(path) == 0


def test_profile_delete_closes_both_sides_and_a_recreated_profile_starts_fresh(home):
    profile = home / "profiles" / "p1"
    profile.mkdir(parents=True)
    path = profile / "state.db"
    legacy, backend = _open_both(path)
    legacy.set_meta("k", "deleted-profile")
    deleted = _identity(path)

    # Profile delete releases this process's handles on both sides before the rmtree.
    assert legacy_registry.close_all_under(profile) == 1
    assert persistence_assembly.close_stores_under(profile) == 0
    assert legacy._conn is None and backend._conn is None
    assert deleted not in lockguard._HANDLES
    shutil.rmtree(profile)
    profile.mkdir(parents=True)

    fresh_legacy, fresh_backend = _open_both(path)
    assert fresh_legacy is not legacy and fresh_backend is not backend
    assert fresh_backend.get_meta("k") is None, "data from the deleted profile survived"
    fresh_legacy.set_meta("k", "recreated-profile")
    assert fresh_backend.get_meta("k") == "recreated-profile"
    for stale in (legacy, backend):
        with pytest.raises(StateDbReplacedError):
            stale.set_meta("k", "stale-write")
    assert _guards(path) == 1
    assert fresh_legacy._read_budget is fresh_backend._read_budget


def test_inode_replacement_retires_both_sides_and_their_old_guards_go_with_them(home, path):
    donor = home / "donor" / "state.db"
    seed = _holder.StoreHolder(donor)
    seed.set_meta("k", "replacement")
    seed.close()
    legacy, backend = _open_both(path)
    legacy.set_meta("k", "original")
    # Quiesce the WAL so new connections read the replacement's pages, not the old generation's frames.
    with backend._lock:
        backend._conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    replaced = _identity(path)
    os.replace(donor, path)

    current_legacy, current_backend = _open_both(path)
    assert current_legacy is not legacy and current_backend is not backend
    assert current_legacy._holder is current_backend
    assert current_legacy.get_meta("k") == current_backend.get_meta("k") == "replacement"
    for old in (legacy, backend):
        with pytest.raises(StateDbReplacedError):
            old.set_meta("k", "stale-write")
    assert _guards(path) == 1

    assert legacy_registry.release(legacy) is True
    assert backend._conn is not None, "the retired generation closed while the backend still held it"
    assert backend_registry.release(backend) is True
    assert legacy._conn is None and backend._conn is None
    assert replaced not in lockguard._HANDLES
    assert _guards(path) == 1
    current_backend.set_meta("k", "written-after-retired-teardown")
    assert current_legacy.get_meta("k") == "written-after-retired-teardown"


def test_pending_accounting_from_both_sides_lands_once_on_the_shared_row(path):
    legacy, backend = _open_both(path)
    legacy.create_session(SESSION, "cli", model="model-a")
    for _ in range(3):
        legacy.queue_token_counts(SESSION, input_tokens=1, output_tokens=1, api_call_count=1, model="model-a",
                                  source="cli")
        backend.queue_token_counts(SESSION, input_tokens=10, output_tokens=1, api_call_count=1, model="model-a",
                                   source="cli")

    # The backend's release leaves the shared writer open for the legacy side, whose flush drains the one queue.
    assert backend_registry.release(backend) is True
    assert legacy.flush_token_counts(5.0) is True
    row = legacy.get_session(SESSION)
    assert (row["input_tokens"], row["output_tokens"], row["api_call_count"]) == (33, 6, 6)


def test_pending_accounting_of_a_separate_writer_drains_at_its_close(path):
    legacy, backend = _open_separate(path)
    backend.set_meta("k", "v")
    legacy.create_session(SESSION, "cli", model="model-a")
    for _ in range(3):
        legacy.queue_token_counts(SESSION, input_tokens=1, output_tokens=1, api_call_count=1, model="model-a",
                                  source="cli")
    legacy.close()
    with hermes_state.SessionDB(path, read_only=True) as reader:
        row = reader.get_session(SESSION)
    assert (row["input_tokens"], row["output_tokens"], row["api_call_count"]) == (3, 3, 3)


def test_a_mixed_process_holds_one_writer_connection_per_file(path):
    legacy, backend = _open_both(path)
    assert legacy._conn is backend._conn
    assert _guards(path) == 1
    assert legacy_registry.stats() == backend_registry.stats() == {
        "live_generations": 1, "retired_generations": 0, "total_refcounts": 2}


def test_a_backend_writer_is_an_in_process_holder_for_the_legacy_auto_vacuum(path):
    seed = hermes_state.SessionDB(path)
    seed.create_session("old", "cli")
    seed.end_session("old", "done")
    seed.close()
    backend = backend_registry.acquire(path)
    maintainer = hermes_state.SessionDB(holder=_holder.StoreHolder(path))
    try:
        maintainer.set_meta("last_auto_prune", "0")
        result = maintainer.maybe_auto_prune_and_vacuum(
            retention_days=0, min_interval_hours=0, min_vacuum_interval_days=0, min_vacuum_freelist_ratio=-1.0)
        assert result["pruned"] >= 1, result
        assert result["vacuumed"] is False, "VACUUM ran under a backend writer in this process"
        assert result.get("vacuum_skipped_holders") == 1, result
    finally:
        maintainer.close()
    assert backend._conn is not None


@pytest.mark.parametrize("closer", ["legacy", "backend"])
def test_one_profile_release_closes_both_sides_handles(home, closer):
    profile = home / "profiles" / "p1"
    profile.mkdir(parents=True)
    path = profile / "state.db"
    legacy, backend = _open_both(path)
    legacy.set_meta("k", "v")
    backend.set_meta("k", "v")
    held = _identity(path)

    close_under = {"legacy": legacy_registry.close_all_under, "backend": persistence_assembly.close_stores_under}[closer]
    assert close_under(profile) == 1
    assert legacy._conn is None and backend._conn is None
    assert held not in lockguard._HANDLES
    shutil.rmtree(profile)
