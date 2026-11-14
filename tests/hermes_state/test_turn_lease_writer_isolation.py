"""Turn lease upkeep must not queue behind transcript/FTS work (#139959)."""

import threading
import sqlite3
import time
from concurrent.futures import ThreadPoolExecutor

import pytest

from hermes_state import SessionDB


def _upkeep(db, operation):
    if operation == "refresh":
        return db.refresh_session_turn_lease("s", "owner")
    return db.release_session_turn_lease("s", "owner")


def test_lease_upkeep_finishes_while_transcript_mutex_is_held(tmp_path):
    with SessionDB(tmp_path / "state.db") as db:
        db.create_session("root", source="test")
        db.end_session("root", "compression")
        db.create_session("child", source="test", parent_session_id="root")
        assert db.try_acquire_session_turn_lease("root", "owner", ttl_seconds=30)
        expiry = db.session_turn_lease_expires_at("root", "owner")
        entered, finished = threading.Event(), threading.Event()

        def upkeep():
            entered.set()
            try:
                return db.refresh_session_turn_lease("child", "owner", ttl_seconds=60, patience_s=1)
            finally:
                finished.set()

        with ThreadPoolExecutor(max_workers=1) as pool:
            with db._lock:
                future = pool.submit(upkeep)
                assert entered.wait(5)
                completed_without_transcript = finished.wait(5)
            result = future.result(timeout=5)
        assert completed_without_transcript, "lease upkeep waited behind transcript work"
        assert result is True
        assert db.session_turn_lease_expires_at("root", "owner") > expiry
        assert not db.try_acquire_session_turn_lease("child", "other")


@pytest.mark.parametrize("blocker", ["sqlite", "lifecycle"])
def test_renewal_waits_only_within_its_budget_and_recovers(tmp_path, blocker):
    with SessionDB(tmp_path / "state.db") as db:
        db.create_session("s", source="test")
        assert db.try_acquire_session_turn_lease("s", "owner")
        expiry = db.session_turn_lease_expires_at("s", "owner")
        with sqlite3.connect(db.db_path, timeout=0) as competing:
            if blocker == "sqlite":
                competing.execute("BEGIN IMMEDIATE")
            else:
                db._read_conns_lock.acquire()
            started = time.monotonic()
            try:
                with pytest.raises(sqlite3.OperationalError, match="locked"):
                    db.refresh_session_turn_lease("s", "owner", patience_s=0.1)
                assert time.monotonic() - started < 3
            finally:
                competing.rollback()
                if blocker == "lifecycle":
                    db._read_conns_lock.release()
        assert db.session_turn_lease_expires_at("s", "owner") == expiry
        assert db.refresh_session_turn_lease("s", "owner", patience_s=1)


@pytest.mark.parametrize("operation", ["refresh", "release"])
def test_upkeep_refuses_a_quarantined_store(tmp_path, operation):
    from hermes_state_errors import StateDbCorruptError
    from hermes_state_health import mark_storage_corrupt

    with SessionDB(tmp_path / "state.db") as db:
        db.create_session("s", source="test")
        assert db.try_acquire_session_turn_lease("s", "owner")
        mark_storage_corrupt(db.db_path, sqlite3.DatabaseError("database disk image is malformed"))
        with pytest.raises(StateDbCorruptError):
            _upkeep(db, operation)


@pytest.mark.platforms("posix")
@pytest.mark.parametrize("operation", ["refresh", "release"])
def test_upkeep_never_opens_a_replacement_database(tmp_path, operation):
    from hermes_state_errors import StateDbReplacedError

    path = tmp_path / "state.db"
    with SessionDB(path) as db:
        db.create_session("s", source="test")
        assert db.try_acquire_session_turn_lease("s", "owner")
        path.rename(tmp_path / "original.db")
        with pytest.raises(StateDbReplacedError):
            _upkeep(db, operation)
        assert not path.exists(), "upkeep must never recreate a missing state.db"


@pytest.mark.platforms("linux")
@pytest.mark.parametrize("operation", ["refresh", "release"])
def test_upkeep_refuses_a_lost_wal_generation(tmp_path, monkeypatch, operation):
    from hermes_state_errors import DeletedWalGenerationError
    from tests.hermes_state._wal_generation_harness import lose_sidecars, pin_wal, require_wal

    pin_wal(monkeypatch)
    with SessionDB(tmp_path / "state.db") as db:
        db.create_session("s", source="test")
        assert db.try_acquire_session_turn_lease("s", "owner")
        require_wal(db)
        lose_sidecars(db.db_path, rename=False)
        with pytest.raises(DeletedWalGenerationError):
            _upkeep(db, operation)


@pytest.mark.parametrize("action", ["read", "close"])
def test_busy_renewal_does_not_hold_reader_or_teardown_mutex(tmp_path, monkeypatch, action):
    from tests.hermes_state._wal_generation_harness import pin_wal, require_wal

    pin_wal(monkeypatch)
    with SessionDB(tmp_path / "state.db") as db:
        db.create_session("s", source="test")
        assert db.try_acquire_session_turn_lease("s", "owner")
        require_wal(db)
        refused, resume = threading.Event(), threading.Event()

        def retry(deadline, patience):
            refused.set()
            assert resume.wait(10)
            return time.monotonic() < deadline

        monkeypatch.setattr(db, "_sleep_before_write_retry", retry)
        with sqlite3.connect(db.db_path, timeout=0) as blocker, ThreadPoolExecutor(max_workers=2) as pool:
            blocker.execute("BEGIN IMMEDIATE")
            renewal = pool.submit(db.refresh_session_turn_lease, "s", "owner", patience_s=15)
            try:
                assert refused.wait(5)
                other = pool.submit(db.get_session, "s") if action == "read" else pool.submit(db.close)
                result = other.result(timeout=5)
                if action == "read":
                    assert result["id"] == "s"
            finally:
                blocker.rollback()
                resume.set()
            assert renewal.result(timeout=5) is True
