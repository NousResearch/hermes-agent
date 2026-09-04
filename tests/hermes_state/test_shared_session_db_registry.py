"""Shared SessionDB registry lifecycle regressions (#90837 review).

Covers the three ownership invariants the PR review demanded:

1. INODE REPLACEMENT — a generation with live holders must NEVER be
   closed by a third caller's acquire.  Retire-and-drain, not
   revoke-by-pathname: existing holders keep a working handle, new
   callers get the fresh generation, and each generation's final
   release tears down exactly that generation.
2. REPLACEMENT-OPEN FAILURE — if the fresh open fails after an inode
   change retired the old generation, the registry must hold NO entry
   for the path (never a closed stale object), and the next acquire
   retries fresh.
3. CLOSE OUTSIDE THE LOCK — a final release's teardown must not run
   under the registry lock (it stops the token writer, checkpoints the
   WAL, drains the read pool — none of which may stall acquisition for
   every state.db in the process).
"""

import contextlib
import os
import shutil
import threading
import time
from pathlib import Path

import pytest

import hermes_state_registry as registry


@pytest.fixture(autouse=True)
def _clean_registry():
    """Isolate the process-global registry between tests."""
    registry.close_all()
    registry._generations.clear()
    registry._retired.clear()
    registry._opening.clear()
    yield
    registry.close_all()
    registry._generations.clear()
    registry._retired.clear()
    registry._opening.clear()


def _replace_file_preserving_schema(src: Path, dst: Path) -> None:
    """Simulate snapshot-restore / recovery: new inode, same logical DB.

    Copies the live DB to a temp name, removes the original, and renames
    the copy into place — the replacement has a different inode.
    """
    tmp = dst.with_suffix(".replacement.tmp")
    shutil.copy2(dst, tmp)
    os.unlink(dst)
    os.rename(tmp, dst)


class TestInodeReplacement:
    def test_live_holders_keep_working_handle_across_replacement(self, tmp_path):
        """Two active refs → inode replacement → third caller gets NEW
        generation; the first two keep a working handle and their
        releases tear down only their own generation."""
        db_path = tmp_path / "state.db"

        a = registry.acquire(db_path)
        b = registry.acquire(db_path)
        assert a is b

        _replace_file_preserving_schema(db_path, db_path)

        c = registry.acquire(db_path)
        assert c is not a, "new caller must get the fresh generation"

        # A and B still hold the OLD generation — it must be alive, not
        # closed underneath them (the review's core blocker).  The old
        # generation's own write path detects the replacement and fails
        # with the typed StateDbReplacedError (existing protection); the
        # registry's job is that the connection object stays VALID —
        # a catchable, typed error, never a use-after-close segfault or
        # "Cannot operate on a closed database".
        assert a._conn is not None, "retired generation closed while holders live"
        from hermes_state import StateDbReplacedError

        with pytest.raises(StateDbReplacedError):
            a.create_session(
                session_id="old-gen-session",
                source="cli",
                model="m",
                model_config={},
                system_prompt=None,
            )

        # New generation works independently.
        c.create_session(
            session_id="new-gen-session",
            source="cli",
            model="m",
            model_config={},
            system_prompt=None,
        )
        assert c.get_session("new-gen-session") is not None

        # Releases route to the right generation: A and B release the
        # OLD one (object-keyed), C releases the NEW one.
        assert registry.release(a) is True
        assert a._conn is not None, "one holder releasing must not tear down the other"
        assert registry.release(b) is True
        assert a._conn is None, "final old-generation release tears it down"
        assert c._conn is not None, "old-generation teardown must not touch the new one"

        assert registry.release(c) is True
        assert c._conn is None
        stats = registry.stats()
        assert stats["live_generations"] == 0
        assert stats["retired_generations"] == 0

    def test_retired_generation_never_relent_even_after_drain(self, tmp_path):
        """After replacement, repeated acquires all return the NEW
        generation — the retired one is never lent again, even while it
        still has live holders."""
        db_path = tmp_path / "state.db"
        first = registry.acquire(db_path)

        _replace_file_preserving_schema(db_path, db_path)

        second = registry.acquire(db_path)
        third = registry.acquire(db_path)
        assert second is third
        assert second is not first
        # Retired generation still drainable by its holder.
        assert registry.release(first) is True
        assert first._conn is None

    def test_open_failure_after_replacement_leaves_no_stale_entry(self, tmp_path, monkeypatch):
        """Replacement-open failure must not leave a closed stale object
        as the registry's authority for the path."""
        db_path = tmp_path / "state.db"
        old = registry.acquire(db_path)

        _replace_file_preserving_schema(db_path, db_path)

        calls = {"n": 0}

        def _fail_open(path):
            calls["n"] += 1
            raise OSError("disk temporarily gone")

        monkeypatch.setattr(registry, "_open_session_db", _fail_open)

        with pytest.raises(OSError):
            registry.acquire(db_path)

        # No live entry for the path — the next acquire retries fresh.
        assert db_path not in registry._generations
        assert stats_live_for(db_path) is None

        monkeypatch.setattr(
            registry,
            "_open_session_db",
            lambda path: _make_session_db(path),
        )
        fresh = registry.acquire(db_path)
        assert fresh is not old
        assert fresh._conn is not None

        # The old generation still drains correctly through its holder.
        assert registry.release(old) is True
        assert old._conn is None


def _make_session_db(path):
    from hermes_state import SessionDB

    return SessionDB(db_path=Path(path))


def stats_live_for(path: Path):
    generation = registry._generations.get(Path(path))
    return generation


class TestTeardownOutsideLock:
    def test_concurrent_cold_acquire_opens_one_writer(self, tmp_path, monkeypatch):
        """Concurrent first callers must not construct redundant writers.

        Returning one winning object is not enough: every losing constructor
        has already opened its own writable SQLite connection by then.  Hold
        the first construction so peer callers overlap deterministically and
        assert the registry single-flights the open itself.
        """
        db_path = tmp_path / "state.db"
        callers = 6
        ready = threading.Barrier(callers + 1)
        release_open = threading.Event()
        count_lock = threading.Lock()
        open_calls = 0
        results = []
        errors = []

        class _FakeDB:
            def __init__(self, path):
                self.db_path = path
                self._shared_registry_owned = False
                self.closed = False

            def close(self):
                self.closed = True

        def _blocked_open(path):
            nonlocal open_calls
            with count_lock:
                open_calls += 1
            assert release_open.wait(5.0)
            return _FakeDB(path)

        monkeypatch.setattr(registry, "_open_session_db", _blocked_open)

        def _acquire():
            try:
                ready.wait()
                results.append(registry.acquire(db_path))
            except BaseException as exc:  # pragma: no cover - failure path
                errors.append(exc)

        threads = [threading.Thread(target=_acquire) for _ in range(callers)]
        for thread in threads:
            thread.start()
        ready.wait()
        time.sleep(0.1)
        release_open.set()
        for thread in threads:
            thread.join(10.0)
            assert not thread.is_alive(), "concurrent acquire deadlocked"

        assert errors == []
        assert open_calls == 1
        assert len({id(db) for db in results}) == 1
        for db in results:
            assert registry.release(db) is True

    def test_waiter_retries_after_cold_open_failure(self, tmp_path, monkeypatch):
        """A failed elected opener must wake a peer to retry the path."""
        db_path = tmp_path / "state.db"
        first_entered = threading.Event()
        release_failure = threading.Event()
        open_calls = 0
        results = []
        errors = []

        class _FakeDB:
            def __init__(self, path):
                self.db_path = path
                self._shared_registry_owned = False

            def close(self):
                pass

        def _fail_then_open(path):
            nonlocal open_calls
            open_calls += 1
            if open_calls == 1:
                first_entered.set()
                assert release_failure.wait(5.0)
                raise OSError("transient open failure")
            return _FakeDB(path)

        monkeypatch.setattr(registry, "_open_session_db", _fail_then_open)

        def _acquire():
            try:
                results.append(registry.acquire(db_path))
            except BaseException as exc:
                errors.append(exc)

        first = threading.Thread(target=_acquire)
        second = threading.Thread(target=_acquire)
        first.start()
        assert first_entered.wait(5.0)
        second.start()
        time.sleep(0.1)
        release_failure.set()
        first.join(10.0)
        second.join(10.0)

        assert not first.is_alive()
        assert not second.is_alive()
        assert open_calls == 2
        assert len(errors) == 1
        assert isinstance(errors[0], OSError)
        assert len(results) == 1
        assert registry.release(results[0]) is True

    def test_equivalent_path_spellings_share_generation(self, tmp_path):
        """Registry identity is the resolved file, not caller spelling."""
        db_path = tmp_path / "nested" / "state.db"
        equivalent = tmp_path / "nested" / ".." / "nested" / "state.db"

        first = registry.acquire(db_path)
        second = registry.acquire(equivalent)
        assert first is second
        assert registry.release(first) is True
        assert registry.release(second) is True

    def test_final_release_does_not_hold_registry_lock_during_close(self, tmp_path, monkeypatch):
        """A final release's teardown (token-writer stop, WAL checkpoint,
        read-pool drain) must run OUTSIDE the registry lock — otherwise
        one state.db's close stalls acquisition for every other."""
        db_path = tmp_path / "state.db"
        db = registry.acquire(db_path)

        teardown_entered = threading.Event()
        lock_released_during_teardown = threading.Event()

        original_teardown = registry._teardown

        def _slow_teardown(target):
            teardown_entered.set()
            # If teardown runs while the registry lock is held, this
            # acquire from another thread will deadlock or block until
            # teardown finishes.  Give it a moment to observe.
            try:
                acquired = registry._lock.acquire(timeout=2.0)
                if acquired:
                    lock_released_during_teardown.set()
                    registry._lock.release()
            except Exception:
                pass
            original_teardown(target)

        monkeypatch.setattr(registry, "_teardown", _slow_teardown)

        result = threading.Event()

        def _release():
            assert registry.release(db) is True
            result.set()

        t = threading.Thread(target=_release)
        t.start()
        assert teardown_entered.wait(5.0), "teardown never ran"
        assert lock_released_during_teardown.wait(5.0), (
            "registry lock was HELD during teardown close — a slow WAL "
            "checkpoint here stalls every other state.db acquisition"
        )
        t.join(10.0)
        assert result.is_set()
        assert db._conn is None

    def test_concurrent_acquire_and_release_no_deadlock(self, tmp_path):
        """Hammer acquire/release from multiple threads — teardown
        contention must not deadlock or corrupt refcounts."""
        db_path = tmp_path / "state.db"
        errors = []

        def _worker(n):
            try:
                for index in range(20):
                    db = registry.acquire(db_path)
                    try:
                        db.create_session(
                            session_id=f"worker-{n}-{index}",
                            source="test",
                            model="test-model",
                            model_config={},
                            system_prompt=None,
                        )
                    finally:
                        registry.release(db)
            except Exception as exc:  # pragma: no cover - failure path
                errors.append(exc)

        threads = [threading.Thread(target=_worker, args=(i,)) for i in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(30.0)
            assert not t.is_alive(), "worker deadlocked"

        assert errors == []
        verifier = registry.acquire(db_path)
        try:
            with verifier._lock:
                assert verifier._conn.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
        finally:
            registry.release(verifier)
        stats = registry.stats()
        assert stats["live_generations"] == 0
        assert stats["retired_generations"] == 0


class TestLegacyCloseSemantics:
    def test_close_on_shared_instance_releases_one_refcount(self, tmp_path):
        """Legacy ``db.close()`` call sites must not leak refcounts: close()
        on a shared instance releases ONE reference — so the gateway's
        pre-registry close paths stay balanced — while never tearing down
        the connection other holders still use."""
        db_path = tmp_path / "state.db"
        a = registry.acquire(db_path)
        b = registry.acquire(db_path)
        assert a is b

        # Legacy close: decrements, does not tear down (b still holds).
        a.close()
        assert b._conn is not None, "close() must not tear down a shared instance"

        # The refcount is now 1 (b's); releasing b tears down.
        assert registry.release(b) is True
        assert b._conn is None
        stats = registry.stats()
        assert stats["live_generations"] == 0

    def test_close_only_call_site_does_not_leak_refcount(self, tmp_path):
        """A call site that acquires and only calls close() (the pre-#90837
        cleanup idiom) must return its reference — the exact leak class
        the 4-angle review flagged."""
        db_path = tmp_path / "state.db"
        for _ in range(5):
            db = registry.acquire(db_path)
            db.close()
        stats = registry.stats()
        assert stats["live_generations"] == 0, (
            f"acquire+close cycles leaked refcounts: {stats}"
        )
        assert stats["retired_generations"] == 0


class TestAcquireSingleFlight:
    def test_concurrent_first_acquires_share_one_generation(self, tmp_path, monkeypatch):
        """Two threads acquiring a cold path concurrently must end up
        sharing ONE generation, with the loser's instance torn down."""
        db_path = tmp_path / "state.db"
        real_open = registry._open_session_db
        gate = threading.Event()
        opened = []

        def _gated_open(path):
            db = real_open(path)
            opened.append(db)
            # Hold the first open so a second thread can race in.
            if len(opened) == 1:
                gate.wait(5.0)
            return db

        monkeypatch.setattr(registry, "_open_session_db", _gated_open)

        results = []
        errors = []

        def _acquire():
            try:
                results.append(registry.acquire(db_path))
            except Exception as exc:  # pragma: no cover
                errors.append(exc)

        t1 = threading.Thread(target=_acquire)
        t1.start()
        # Wait until the first open is in flight inside the lock window.
        deadline = time.monotonic() + 5.0
        while not opened and time.monotonic() < deadline:
            time.sleep(0.01)
        t2 = threading.Thread(target=_acquire)
        t2.start()
        gate.set()
        t1.join(10.0)
        t2.join(10.0)

        assert errors == []
        assert len(results) == 2
        assert results[0] is results[1], "concurrent acquires must share one generation"
        assert len(opened) >= 1
        registry.release(results[0])
        registry.release(results[1])
