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


class TestEnvMutationGate:
    """_EnvMutationGate: env-mutating jobs never overlap parallel jobs."""

    def test_shared_holders_run_concurrently(self):
        """Two shared acquisitions overlap (no serialization of parallel jobs)."""
        import cron.scheduler as sched

        gate = sched._EnvMutationGate()
        both_inside = threading.Barrier(2, timeout=5)

        def reader():
            with gate.shared():
                both_inside.wait()  # only passes if both hold shared at once

        t1 = threading.Thread(target=reader)
        t2 = threading.Thread(target=reader)
        t1.start(); t2.start()
        t1.join(timeout=5); t2.join(timeout=5)
        assert not t1.is_alive() and not t2.is_alive()

    def test_exclusive_waits_for_shared_and_blocks_new_shared(self):
        """Writer waits for in-flight readers; queued writer blocks new readers."""
        import cron.scheduler as sched

        gate = sched._EnvMutationGate()
        reader_inside = threading.Event()
        release_reader = threading.Event()
        writer_inside = threading.Event()
        writer_done = threading.Event()
        late_reader_inside = threading.Event()

        def reader():
            with gate.shared():
                reader_inside.set()
                release_reader.wait(timeout=5)

        def writer():
            with gate.exclusive():
                writer_inside.set()
                time.sleep(0.2)
            writer_done.set()

        def late_reader():
            with gate.shared():
                late_reader_inside.set()

        t_reader = threading.Thread(target=reader)
        t_reader.start()
        assert reader_inside.wait(timeout=5)

        t_writer = threading.Thread(target=writer)
        t_writer.start()
        time.sleep(0.2)
        # Writer must not enter while the reader holds the gate.
        assert not writer_inside.is_set()

        # A reader arriving behind a waiting writer queues (writer priority).
        t_late = threading.Thread(target=late_reader)
        t_late.start()
        time.sleep(0.2)
        assert not late_reader_inside.is_set()

        release_reader.set()
        assert writer_inside.wait(timeout=5)
        assert writer_done.wait(timeout=5)
        assert late_reader_inside.wait(timeout=5)
        for t in (t_reader, t_writer, t_late):
            t.join(timeout=5)
            assert not t.is_alive()

    def _make_job(self, job_id, **extra):
        job = {
            "id": job_id, "name": job_id, "prompt": "test",
            "schedule": "every 5m", "enabled": True,
            "next_run_at": "2020-01-01T00:00:00", "deliver": "local",
        }
        job.update(extra)
        return job

    def _patch_common(self, sched, monkeypatch, due):
        monkeypatch.setattr(sched, "get_due_jobs", lambda: due)
        monkeypatch.setattr(sched, "advance_next_run", lambda *_a, **_kw: None)
        monkeypatch.setattr(sched, "save_job_output", lambda *_a, **_kw: "/tmp/out")
        monkeypatch.setattr(sched, "mark_job_run", lambda *_a, **_kw: None)
        monkeypatch.setattr(sched, "_deliver_result", lambda *_a, **_kw: None)

    def test_sequential_job_waits_for_inflight_parallel_job(self, tmp_path, monkeypatch):
        """A workdir job dispatched while a parallel job runs must not start
        mutating env state until the parallel job finishes."""
        import cron.scheduler as sched

        sched._parallel_pool = None
        sched._parallel_pool_max_workers = None
        sched._sequential_pool = None
        sched._running_job_ids.clear()
        monkeypatch.setattr(sched, "_env_gate", sched._EnvMutationGate())

        parallel_started = threading.Event()
        release_parallel = threading.Event()
        sequential_started = threading.Event()

        def fake_run(j):
            if j["id"] == "par":
                parallel_started.set()
                release_parallel.wait(timeout=10)
            else:
                sequential_started.set()
            return True, "out", "resp", None

        due = [self._make_job("par")]
        self._patch_common(sched, monkeypatch, due)
        monkeypatch.setattr(sched, "run_job", fake_run)

        try:
            sched.tick(verbose=False, sync=False)
            assert parallel_started.wait(timeout=5)

            due[:] = [self._make_job("seq", workdir=str(tmp_path))]
            sched.tick(verbose=False, sync=False)

            time.sleep(0.3)
            assert not sequential_started.is_set(), (
                "sequential (workdir) job overlapped an in-flight parallel job"
            )

            release_parallel.set()
            assert sequential_started.wait(timeout=5)
        finally:
            release_parallel.set()
            sched._shutdown_parallel_pool()
            sched._running_job_ids.clear()

    def test_parallel_job_waits_for_inflight_sequential_job(self, tmp_path, monkeypatch):
        """A parallel job dispatched while a workdir job runs must not
        start (and read mutated env state) until the sequential job finishes."""
        import cron.scheduler as sched

        sched._parallel_pool = None
        sched._parallel_pool_max_workers = None
        sched._sequential_pool = None
        sched._running_job_ids.clear()
        monkeypatch.setattr(sched, "_env_gate", sched._EnvMutationGate())

        sequential_started = threading.Event()
        release_sequential = threading.Event()
        parallel_started = threading.Event()

        def fake_run(j):
            if j["id"] == "seq":
                sequential_started.set()
                release_sequential.wait(timeout=10)
            else:
                parallel_started.set()
            return True, "out", "resp", None

        due = [self._make_job("seq", workdir=str(tmp_path))]
        self._patch_common(sched, monkeypatch, due)
        monkeypatch.setattr(sched, "run_job", fake_run)

        try:
            sched.tick(verbose=False, sync=False)
            assert sequential_started.wait(timeout=5)

            due[:] = [self._make_job("par")]
            sched.tick(verbose=False, sync=False)

            time.sleep(0.3)
            assert not parallel_started.is_set(), (
                "parallel job overlapped an in-flight workdir job"
            )

            release_sequential.set()
            assert parallel_started.wait(timeout=5)
        finally:
            release_sequential.set()
            sched._shutdown_parallel_pool()
            sched._running_job_ids.clear()
