"""Tests for the persistent parallel pool and running-job guard in cron/scheduler.py.

These verify the fix for the tick-blocking issue where as_completed(timeout=600)
prevented the ticker thread from firing, causing all other jobs to be fast-forwarded.
"""

import concurrent.futures
import threading
import time
from unittest.mock import patch

import pytest


class TestPersistentPool:
    """_get_parallel_pool returns a persistent ThreadPoolExecutor."""

    def test_pool_is_reused(self, monkeypatch):
        """Same pool instance returned when max_workers doesn't change."""
        import cron.scheduler as sched

        # Reset module state.
        sched._parallel_pool = None
        sched._parallel_pool_max_workers = None

        pool1 = sched._get_parallel_pool(4)
        pool2 = sched._get_parallel_pool(4)
        assert pool1 is pool2

        # Cleanup.
        sched._shutdown_parallel_pool()

    def test_pool_is_recreated_on_worker_change(self, monkeypatch):
        """New pool when max_workers changes."""
        import cron.scheduler as sched

        sched._parallel_pool = None
        sched._parallel_pool_max_workers = None

        pool1 = sched._get_parallel_pool(2)
        pool2 = sched._get_parallel_pool(4)
        assert pool1 is not pool2

        sched._shutdown_parallel_pool()

    def test_shutdown_clears_pool(self, monkeypatch):
        """_shutdown_parallel_pool resets state."""
        import cron.scheduler as sched

        sched._parallel_pool = None
        sched._parallel_pool_max_workers = None
        sched._get_parallel_pool(2)

        sched._shutdown_parallel_pool()
        assert sched._parallel_pool is None
        assert sched._parallel_pool_max_workers is None


class TestRunningJobGuard:
    """_running_job_ids prevents double-dispatch of active jobs."""

    def test_submit_claimed_job_returns_before_completion_and_tracks_run(self, monkeypatch):
        """Manual gateway dispatch uses the persistent pool and running set."""
        import cron.scheduler as sched

        sched._parallel_pool = None
        sched._parallel_pool_max_workers = None
        sched._running_job_ids.clear()
        job = {"id": "manual-job", "name": "manual", "prompt": "test"}
        barrier = threading.Barrier(2, timeout=5)

        def slow_run(_job, **_kwargs):
            barrier.wait()
            return True

        monkeypatch.setattr(sched, "run_one_job", slow_run)

        future = sched.submit_claimed_job(job)

        assert future is not None
        assert "manual-job" in sched.get_running_job_ids()
        assert future.done() is False

        barrier.wait()
        assert future.result(timeout=5) is True
        assert "manual-job" not in sched.get_running_job_ids()
        sched._shutdown_parallel_pool()

    def test_running_set_prevents_double_dispatch(self, tmp_path, monkeypatch):
        """A job already in _running_job_ids is skipped on the next tick."""
        import cron.scheduler as sched

        # Reset state.
        sched._parallel_pool = None
        sched._parallel_pool_max_workers = None
        sched._running_job_ids.clear()

        job = {
            "id": "guard-job",
            "name": "guard-test",
            "prompt": "test",
            "schedule": "every 5m",
            "enabled": True,
            "next_run_at": "2020-01-01T00:00:00",
            "deliver": "local",
        }

        # Simulate the job already running.
        sched._running_job_ids.add("guard-job")

        dispatched = []
        monkeypatch.setattr(sched, "get_due_jobs", lambda: [job])
        monkeypatch.setattr(sched, "advance_next_run", lambda *_a, **_kw: None)
        monkeypatch.setattr(sched, "run_job", lambda j, **_kw: dispatched.append(j["id"]) or (True, "out", "resp", None))
        monkeypatch.setattr(sched, "save_job_output", lambda *_a, **_kw: None)
        monkeypatch.setattr(sched, "mark_job_run", lambda *_a, **_kw: None)
        monkeypatch.setattr(sched, "_deliver_result", lambda *_a, **_kw: None)

        n = sched.tick(verbose=False)
        assert n == 0  # skipped, not dispatched
        assert dispatched == []

        sched._running_job_ids.discard("guard-job")
        sched._shutdown_parallel_pool()


class TestSyncMode:
    """tick() blocks by default (sync=True); tick(sync=False) returns immediately."""

    def test_sync_true_blocks_and_returns_correct_count(self, tmp_path, monkeypatch):
        """sync=True waits for jobs and returns actual results."""
        import cron.scheduler as sched

        sched._parallel_pool = None
        sched._parallel_pool_max_workers = None
        sched._running_job_ids.clear()

        jobs = [
            {"id": f"job-{i}", "name": f"Job {i}", "prompt": "test",
             "schedule": "every 5m", "enabled": True,
             "next_run_at": "2020-01-01T00:00:00", "deliver": "local"}
            for i in range(3)
        ]

        monkeypatch.setattr(sched, "get_due_jobs", lambda: jobs)
        monkeypatch.setattr(sched, "advance_next_run", lambda *_a, **_kw: None)
        monkeypatch.setattr(sched, "run_job", lambda j, **_kw: (True, "out", "resp", None))
        monkeypatch.setattr(sched, "save_job_output", lambda *_a, **_kw: "/tmp/out")
        monkeypatch.setattr(sched, "mark_job_run", lambda *_a, **_kw: None)
        monkeypatch.setattr(sched, "_deliver_result", lambda *_a, **_kw: None)

        n = sched.tick(verbose=False)
        assert n == 3

        sched._shutdown_parallel_pool()

    def test_sync_false_returns_immediately(self, tmp_path, monkeypatch):
        """sync=False returns before parallel jobs finish (optimistic count)."""
        import cron.scheduler as sched

        sched._parallel_pool = None
        sched._parallel_pool_max_workers = None
        sched._running_job_ids.clear()

        job = {
            "id": "slow-job",
            "name": "slow",
            "prompt": "test",
            "schedule": "every 5m",
            "enabled": True,
            "next_run_at": "2020-01-01T00:00:00",
            "deliver": "local",
        }

        barrier = threading.Barrier(2, timeout=5)

        def slow_run(j, *, defer_agent_teardown=None):
            barrier.wait()  # blocks until test thread also waits
            return True, "out", "resp", None

        monkeypatch.setattr(sched, "get_due_jobs", lambda: [job])
        monkeypatch.setattr(sched, "advance_next_run", lambda *_a, **_kw: None)
        monkeypatch.setattr(sched, "run_job", slow_run)
        monkeypatch.setattr(sched, "save_job_output", lambda *_a, **_kw: "/tmp/out")
        monkeypatch.setattr(sched, "mark_job_run", lambda *_a, **_kw: None)
        monkeypatch.setattr(sched, "_deliver_result", lambda *_a, **_kw: None)

        start = time.monotonic()
        n = sched.tick(verbose=False, sync=False)  # opt-in: non-blocking
        elapsed = time.monotonic() - start

        assert n == 1  # optimistic count
        assert elapsed < 1.0  # returned immediately, didn't wait for slow_run

        # Let the job finish so cleanup works.
        barrier.wait()
        time.sleep(0.1)
        sched._shutdown_parallel_pool()


class TestSequentialPool:
    """Sequential (workdir) jobs use the persistent cron-seq pool.

    Verifies the follow-up fix: env-mutating jobs no longer run inline
    in the ticker thread, so a long workdir job can't starve the
    schedule the same way the parallel path used to.
    """

    def test_sequential_job_does_not_block_ticker(self, tmp_path, monkeypatch):
        """sync=False returns immediately even when a workdir job is slow."""
        import cron.scheduler as sched

        sched._parallel_pool = None
        sched._parallel_pool_max_workers = None
        sched._sequential_pool = None
        sched._running_job_ids.clear()

        job = {
            "id": "slow-workdir",
            "name": "slow-workdir",
            "prompt": "test",
            "schedule": "every 5m",
            "enabled": True,
            "next_run_at": "2020-01-01T00:00:00",
            "deliver": "local",
            "workdir": str(tmp_path),  # makes it sequential
        }

        barrier = threading.Barrier(2, timeout=5)

        def slow_run(j, *, defer_agent_teardown=None):
            barrier.wait()
            return True, "out", "resp", None

        monkeypatch.setattr(sched, "get_due_jobs", lambda: [job])
        monkeypatch.setattr(sched, "advance_next_run", lambda *_a, **_kw: None)
        monkeypatch.setattr(sched, "run_job", slow_run)
        monkeypatch.setattr(sched, "save_job_output", lambda *_a, **_kw: "/tmp/out")
        monkeypatch.setattr(sched, "mark_job_run", lambda *_a, **_kw: None)
        monkeypatch.setattr(sched, "_deliver_result", lambda *_a, **_kw: None)

        start = time.monotonic()
        n = sched.tick(verbose=False, sync=False)
        elapsed = time.monotonic() - start

        assert n == 1  # optimistic count
        assert elapsed < 1.0  # did NOT block on the slow workdir job

        barrier.wait()
        time.sleep(0.1)
        sched._shutdown_parallel_pool()

    def test_sequential_running_guard_prevents_double_dispatch(self, tmp_path, monkeypatch):
        """A workdir job already in _running_job_ids is skipped on next tick."""
        import cron.scheduler as sched

        sched._parallel_pool = None
        sched._parallel_pool_max_workers = None
        sched._sequential_pool = None
        sched._running_job_ids.clear()

        job = {
            "id": "guard-seq",
            "name": "guard-seq",
            "prompt": "test",
            "schedule": "every 5m",
            "enabled": True,
            "next_run_at": "2020-01-01T00:00:00",
            "deliver": "local",
            "workdir": str(tmp_path),
        }

        # Simulate the job already running.
        sched._running_job_ids.add("guard-seq")

        dispatched = []
        monkeypatch.setattr(sched, "get_due_jobs", lambda: [job])
        monkeypatch.setattr(sched, "advance_next_run", lambda *_a, **_kw: None)
        monkeypatch.setattr(sched, "run_job", lambda j, **_kw: dispatched.append(j["id"]) or (True, "out", "resp", None))
        monkeypatch.setattr(sched, "save_job_output", lambda *_a, **_kw: None)
        monkeypatch.setattr(sched, "mark_job_run", lambda *_a, **_kw: None)
        monkeypatch.setattr(sched, "_deliver_result", lambda *_a, **_kw: None)

        n = sched.tick(verbose=False)
        assert n == 0  # skipped, not dispatched
        assert dispatched == []

        sched._running_job_ids.discard("guard-seq")
        sched._shutdown_parallel_pool()

    def test_get_sequential_pool_is_persistent(self):
        """_get_sequential_pool returns the same single-thread pool."""
        import cron.scheduler as sched

        sched._sequential_pool = None
        pool1 = sched._get_sequential_pool()
        pool2 = sched._get_sequential_pool()
        assert pool1 is pool2

        sched._shutdown_parallel_pool()
        assert sched._sequential_pool is None
