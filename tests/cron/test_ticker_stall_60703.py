"""Regression tests for #60703 — cron ticker silently stalls after gateway restart.

Three fixes under test:

1. ``_jobs_lock()`` bounds its cross-process flock: when another process holds
   ``.jobs.lock`` indefinitely, acquisition times out, logs at ERROR, and falls
   through to in-process-only locking — instead of blocking the calling thread
   (and, transitively, the cron ticker heartbeat) forever.

2. Claim freshness checks are bounded on both sides (``0 <= age < ttl``): a
   ``fire_claim``/``run_claim`` stamped in the FUTURE (clock/TZ skew across a
   restart) is treated as stale/overwritable, not eternally fresh.

3. ``_execute_job_now`` no longer mislabels a missing job as "already being fired".
"""

import json
import os
import threading
import time
from datetime import timedelta
from pathlib import Path

import pytest

import cron.jobs as jobs_mod
import cron.scheduler as scheduler_mod
from cron.jobs import (
    _jobs_lock,
    claim_job_for_fire,
    create_job,
    get_due_jobs,
    get_job,
    load_jobs,
    save_jobs,
)


try:
    import fcntl
except ImportError:  # pragma: no cover - non-POSIX
    fcntl = None


pytestmark = pytest.mark.skipif(fcntl is None, reason="flock semantics are POSIX-only")


def _hold_jobs_flock(path: Path, release: threading.Event, held: threading.Event):
    """Hold an exclusive flock on *path* from a separate fd until released.

    flock locks are per-open-file-description, so a second open() in the SAME
    process contends exactly like another process would.
    """
    fd = open(path, "a+", encoding="utf-8")
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        held.set()
        release.wait(timeout=30)
    finally:
        try:
            fcntl.flock(fd, fcntl.LOCK_UN)
        except OSError:
            pass
        fd.close()


class TestBoundedJobsLock:
    def test_lock_acquisition_times_out_and_degrades(self, monkeypatch, caplog):
        """A foreign holder of .jobs.lock must NOT block _jobs_lock forever."""
        jobs_mod.ensure_dirs()
        lock_path = jobs_mod._jobs_lock_file()
        lock_path.touch()

        monkeypatch.setattr(jobs_mod, "_JOBS_LOCK_TIMEOUT_SECONDS", 1.0)

        release = threading.Event()
        held = threading.Event()
        holder = threading.Thread(
            target=_hold_jobs_flock, args=(lock_path, release, held), daemon=True
        )
        holder.start()
        assert held.wait(timeout=10), "test holder failed to take the flock"

        try:
            start = time.monotonic()
            entered = False
            with caplog.at_level("ERROR", logger="cron.jobs"):
                with _jobs_lock():
                    entered = True
            elapsed = time.monotonic() - start

            assert entered, "critical section must still run in degraded mode"
            assert elapsed < 10, f"lock wait was not bounded (took {elapsed:.1f}s)"
            assert any("Timed out" in r.message for r in caplog.records), (
                "degraded-mode fallback must be logged at ERROR"
            )
        finally:
            release.set()
            holder.join(timeout=10)

    def test_uncontended_lock_is_fast_and_silent(self, caplog):
        jobs_mod.ensure_dirs()
        start = time.monotonic()
        with caplog.at_level("ERROR", logger="cron.jobs"):
            with _jobs_lock():
                pass
        assert time.monotonic() - start < 5
        assert not [r for r in caplog.records if "Timed out" in r.message]

    def test_reentrant_nesting_still_works(self):
        with _jobs_lock():
            with _jobs_lock():  # must not deadlock or re-flock
                pass


class TestFutureDatedClaims:
    def _make_job(self, **kw):
        return create_job(name="claim job", schedule="0 7 * * *", prompt="x", **kw)

    def test_future_fire_claim_is_treated_as_stale(self):
        """A fire_claim stamped in the future must not block claiming forever."""
        job = self._make_job()
        jobs = load_jobs()
        for j in jobs:
            if j["id"] == job["id"]:
                future = jobs_mod._hermes_now() + timedelta(hours=6)
                j["fire_claim"] = {"at": future.isoformat(), "by": "other-host:1"}
        save_jobs(jobs)

        assert claim_job_for_fire(job["id"]) is True, (
            "future-dated claim must be overwritable, not eternally fresh"
        )

    def test_fresh_past_fire_claim_still_blocks(self):
        job = self._make_job()
        assert claim_job_for_fire(job["id"]) is True
        # Immediately re-claiming must be refused — claim is genuinely fresh.
        assert claim_job_for_fire(job["id"]) is False

    def test_expired_fire_claim_is_reclaimable(self):
        job = self._make_job()
        jobs = load_jobs()
        for j in jobs:
            if j["id"] == job["id"]:
                past = jobs_mod._hermes_now() - timedelta(hours=6)
                j["fire_claim"] = {"at": past.isoformat(), "by": "other-host:1"}
        save_jobs(jobs)
        assert claim_job_for_fire(job["id"]) is True


class TestHonestRunSkipMessages:
    def test_missing_job_not_reported_as_already_firing(self):
        from tools.cronjob_tools import _execute_job_now

        res = _execute_job_now({"id": "does-not-exist-123"})
        assert res["claimed"] is False
        assert "no longer exists" in (res["error"] or "").lower()


class TestManualRunLocalOwnership:
    def test_live_fire_owner_blocks_replacement_local_reservation(self):
        """A swept pre-execution guard must not admit a replacement while its body is live."""
        job_id = "manual-owner-still-live"
        execution_token = object()
        with scheduler_mod._running_lock:
            scheduler_mod._running_job_ids.discard(job_id)
            scheduler_mod._running_fire_owners[job_id] = {
                execution_token: ("store-owner-a", Path.cwd())
            }
        try:
            assert scheduler_mod.try_register_running_job(job_id) is False
        finally:
            with scheduler_mod._running_lock:
                scheduler_mod._running_fire_owners.pop(job_id, None)
            scheduler_mod.release_running_job(job_id)

    def test_old_owner_cannot_release_replacement_reservation(self):
        """A delayed finally from owner A must not remove owner B's local dedupe guard."""
        job_id = "manual-owner-replaced"
        owner_a = "reservation-a"
        owner_b = "reservation-b"
        try:
            assert scheduler_mod.try_register_running_job(
                job_id, reservation_token=owner_a
            )
            assert scheduler_mod.release_running_job(
                job_id, expected_reservation_token=owner_a
            )
            assert scheduler_mod.try_register_running_job(
                job_id, reservation_token=owner_b
            )

            assert scheduler_mod.release_running_job(
                job_id, expected_reservation_token=owner_a
            ) is False
            assert job_id in scheduler_mod.get_running_job_ids()
        finally:
            scheduler_mod.release_running_job(
                job_id, expected_reservation_token=owner_b
            )

    def test_stale_ticker_finally_cannot_release_replacement_manual_reservation(
        self, monkeypatch
    ):
        """A swept ticker worker must not remove a newer manual owner when it starts late."""

        class FakeFuture:
            def done(self):
                return False

        class DeferredPool:
            submitted = None

            def submit(self, fn):
                self.submitted = fn
                return FakeFuture()

        job_id = "ticker-owner-replaced"
        manual_owner = "manual-owner-b"
        pool = DeferredPool()
        ran = []
        monkeypatch.setattr(
            scheduler_mod,
            "create_execution",
            lambda *_args, **_kwargs: {"id": "execution-a"},
        )
        try:
            assert scheduler_mod._submit_with_guard(
                {"id": job_id, "name": job_id, "schedule": {"kind": "interval", "minutes": 5}},
                pool,
                lambda _job: ran.append("ticker-a"),
            ) is not None
            with scheduler_mod._running_lock:
                ticker_owner = scheduler_mod._running_reservation_tokens.get(job_id)
            assert scheduler_mod.release_running_job(
                job_id, expected_reservation_token=ticker_owner
            )
            assert scheduler_mod.try_register_running_job(
                job_id, reservation_token=manual_owner
            )
            assert scheduler_mod.release_running_job(job_id) is False
            assert job_id in scheduler_mod.get_running_job_ids()

            assert pool.submitted is not None
            pool.submitted()
            assert ran == ["ticker-a"]
            assert job_id in scheduler_mod.get_running_job_ids()
        finally:
            scheduler_mod.release_running_job(
                job_id, expected_reservation_token=manual_owner
            )

    def test_late_stale_ticker_record_cannot_clear_manual_store_claim(
        self, monkeypatch
    ):
        """Stale telemetry is store-fenced after local sweep admits manual owner B."""
        from tools.cronjob_tools import (
            _claim_for_manual_run,
            _release_manual_run_reservation,
        )

        class DoneFuture:
            def done(self):
                return True

        job = create_job(name="durable aba", schedule="every 5m", prompt="x")
        job_id = job["id"]
        ticker_owner = object()
        delayed = []
        original_record = scheduler_mod._record_stale_release
        assert scheduler_mod.try_register_running_job(
            job_id, reservation_token=ticker_owner
        )
        with scheduler_mod._running_lock:
            scheduler_mod._running_since[job_id] = time.time() - 7200
            scheduler_mod._running_futures[job_id] = DoneFuture()
        monkeypatch.setattr(
            scheduler_mod,
            "_record_stale_release",
            lambda *args: delayed.append(args),
        )
        claimed_job = None
        try:
            assert scheduler_mod.sweep_stale_inflight([get_job(job_id)]) == [job_id]
            assert len(delayed) == 1
            claimed_job, err = _claim_for_manual_run(job_id, "durable ABA test")
            assert err is None
            assert claimed_job is not None
            fire_owner = claimed_job["fire_claim"]["by"]

            original_record(*delayed[0])

            after = get_job(job_id)
            assert after is not None
            assert after["fire_claim"]["by"] == fire_owner
            assert after["repeat"]["completed"] == 0
        finally:
            if claimed_job is not None:
                _release_manual_run_reservation(claimed_job)
            scheduler_mod.release_running_job(
                job_id, expected_reservation_token=ticker_owner
            )
