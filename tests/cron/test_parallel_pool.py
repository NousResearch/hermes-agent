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
        monkeypatch.setattr(sched, "run_job", lambda j: dispatched.append(j["id"]) or (True, "out", "resp", None))
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
        monkeypatch.setattr(sched, "run_job", lambda j: (True, "out", "resp", None))
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

        def slow_run(j):
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

        def slow_run(j):
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
        monkeypatch.setattr(sched, "run_job", lambda j: dispatched.append(j["id"]) or (True, "out", "resp", None))
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


class TestLockReleasedBeforeDispatch:
    """The tick file lock is released after the get_due_jobs + advance_next_run
    critical section, BEFORE any due job is dispatched.

    Regression for #27485: previously the lock was held through the entire
    ThreadPoolExecutor lifetime, so a long-running delegation kept every
    subsequent 60s ticker attempt blocked on the lock and overdue interval
    jobs were silently dropped.  The lock must now only serialize the
    read-modify-write of schedule state; job execution runs lock-free.

    The test records a "lock_release" event whenever the scheduler unlocks
    its file lock and a "run_job" event when a job body starts, then asserts
    the release happens before the first dispatch.  Pre-fix (lock held across
    dispatch) the order is run_job, run_job, lock_release and the assertion
    fails.
    """

    def _instrument(self, sched, monkeypatch, events, events_lock, jobs):
        """Patch the scheduler so lock-release and job-start are both recorded
        in a single ordered ``events`` list, and stub out the per-job side
        effects."""

        def _record(name):
            with events_lock:
                events.append(name)

        # Record every lock release.  fcntl is the Unix path; msvcrt the
        # Windows path.  Wrap whichever the module actually bound so the
        # test is cross-platform.
        if sched.fcntl is not None:
            real_flock = sched.fcntl.flock

            def _flock(fd, op):
                if op == sched.fcntl.LOCK_UN:
                    _record("lock_release")
                return real_flock(fd, op)

            monkeypatch.setattr(sched.fcntl, "flock", _flock)
        elif sched.msvcrt is not None:  # pragma: no cover - Windows only
            real_locking = sched.msvcrt.locking

            def _locking(fileno, mode, nbytes):
                if mode == sched.msvcrt.LK_UNLCK:
                    _record("lock_release")
                return real_locking(fileno, mode, nbytes)

            monkeypatch.setattr(sched.msvcrt, "locking", _locking)

        def _run_job(j):
            _record("run_job")
            return True, "out", "resp", None

        monkeypatch.setattr(sched, "get_due_jobs", lambda: jobs)
        monkeypatch.setattr(sched, "advance_next_run", lambda *_a, **_kw: None)
        monkeypatch.setattr(sched, "run_job", _run_job)
        monkeypatch.setattr(sched, "save_job_output", lambda *_a, **_kw: "/tmp/out")
        monkeypatch.setattr(sched, "mark_job_run", lambda *_a, **_kw: None)
        monkeypatch.setattr(sched, "_deliver_result", lambda *_a, **_kw: None)

    def test_lock_released_before_dispatch_sync(self, tmp_path, monkeypatch):
        """sync=True path: the lock is released before any job runs."""
        import cron.scheduler as sched

        sched._parallel_pool = None
        sched._parallel_pool_max_workers = None
        sched._sequential_pool = None
        sched._running_job_ids.clear()

        jobs = [
            {"id": f"job-{i}", "name": f"Job {i}", "prompt": "test",
             "schedule": "every 5m", "enabled": True,
             "next_run_at": "2020-01-01T00:00:00", "deliver": "local"}
            for i in range(2)
        ]

        events: list = []
        events_lock = threading.Lock()
        self._instrument(sched, monkeypatch, events, events_lock, jobs)

        n = sched.tick(verbose=False)  # sync=True default
        assert n == 2

        # The lock must be released before the first job is dispatched.
        assert "lock_release" in events, events
        assert "run_job" in events, events
        assert events.index("lock_release") < events.index("run_job"), events

        sched._shutdown_parallel_pool()

    def test_lock_released_before_dispatch_async(self, tmp_path, monkeypatch):
        """Production sync=False path (gateway ticker): the lock is released
        before the dispatched job body runs, even though tick() returns
        without waiting."""
        import cron.scheduler as sched

        sched._parallel_pool = None
        sched._parallel_pool_max_workers = None
        sched._sequential_pool = None
        sched._running_job_ids.clear()

        job = {
            "id": "async-job", "name": "Async", "prompt": "test",
            "schedule": "every 5m", "enabled": True,
            "next_run_at": "2020-01-01T00:00:00", "deliver": "local",
        }

        events: list = []
        events_lock = threading.Lock()
        started = threading.Event()

        def _run_job(j):
            with events_lock:
                events.append("run_job")
            started.set()
            return True, "out", "resp", None

        # Reuse the lock-release instrumentation, then override run_job to
        # also signal completion so the assertion doesn't race the pool.
        self._instrument(sched, monkeypatch, events, events_lock, [job])
        monkeypatch.setattr(sched, "run_job", _run_job)

        n = sched.tick(verbose=False, sync=False)
        assert n == 1  # optimistic count, returns immediately

        # Wait for the pool thread to actually start the job.
        assert started.wait(timeout=5), "job never dispatched"
        time.sleep(0.05)  # let run_job append its event

        assert "lock_release" in events, events
        assert "run_job" in events, events
        assert events.index("lock_release") < events.index("run_job"), events

        time.sleep(0.05)
        sched._shutdown_parallel_pool()
