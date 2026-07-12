"""Regression tests for #60703 — cron ticker silently stalls after gateway restart.

Three fixes under test:

1. ``_jobs_lock()`` bounds its cross-process flock: when another process holds
   ``.jobs.lock`` indefinitely, acquisition times out, logs at ERROR, and falls
   through to in-process-only locking — instead of blocking the calling thread
   (and, transitively, the cron ticker heartbeat) forever.

2. Claim freshness checks are bounded on both sides (``0 <= age < ttl``): a
   ``fire_claim``/``run_claim`` stamped in the FUTURE (clock/TZ skew across a
   restart) is treated as stale/overwritable, not eternally fresh.

3. ``_execute_job_now`` no longer mislabels paused/disabled/missing jobs as
   "already being fired".
"""

import json
import os
import threading
import time
from datetime import timedelta
from pathlib import Path

import pytest

import cron.jobs as jobs_mod
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


@pytest.fixture(autouse=True)
def isolated_cron_store(tmp_path, monkeypatch):
    """Never let these lock/claim regressions touch a user's live cron store."""
    hermes_home = tmp_path / ".hermes"
    cron_dir = hermes_home / "cron"
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    monkeypatch.setattr(jobs_mod, "HERMES_DIR", hermes_home)
    monkeypatch.setattr(jobs_mod, "CRON_DIR", cron_dir)
    monkeypatch.setattr(jobs_mod, "JOBS_FILE", cron_dir / "jobs.json")
    monkeypatch.setattr(jobs_mod, "OUTPUT_DIR", cron_dir / "output")


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

    def test_future_run_claim_does_not_skip_oneshot_forever(self):
        """A one-shot with a future-dated run_claim must still become due."""
        past_fire = (jobs_mod._hermes_now() - timedelta(seconds=30)).isoformat()
        job = create_job(name="oneshot", schedule=past_fire, prompt="x")
        jobs = load_jobs()
        for j in jobs:
            if j["id"] == job["id"]:
                future = jobs_mod._hermes_now() + timedelta(hours=6)
                j["run_claim"] = {"at": future.isoformat(), "by": "other-host:1"}
                j["next_run_at"] = past_fire
        save_jobs(jobs)

        due_ids = {j["id"] for j in get_due_jobs()}
        assert job["id"] in due_ids, (
            "future-dated run_claim must be treated as stale, not fresh"
        )


class TestHonestRunSkipMessages:
    def test_paused_job_not_reported_as_already_firing(self):
        from tools.cronjob_tools import _execute_job_now

        job = create_job(name="paused job", schedule="0 7 * * *", prompt="x")
        from cron.jobs import pause_job

        pause_job(job["id"])
        res = _execute_job_now(get_job(job["id"]))
        assert res["claimed"] is False
        assert "paused" in (res["error"] or "").lower()
        assert "already being fired" not in (res["error"] or "").lower()

    def test_missing_job_not_reported_as_already_firing(self):
        from tools.cronjob_tools import _execute_job_now

        res = _execute_job_now({"id": "does-not-exist-123"})
        assert res["claimed"] is False
        assert "no longer exists" in (res["error"] or "").lower()
