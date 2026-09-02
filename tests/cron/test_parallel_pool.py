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
        monkeypatch.setattr(sched, "claim_job_for_fire", lambda *_a, **_kw: True)
        monkeypatch.setattr(sched, "run_job", lambda j, **_kw: dispatched.append(j["id"]) or (True, "out", "resp", None))
        monkeypatch.setattr(sched, "save_job_output", lambda *_a, **_kw: None)
        monkeypatch.setattr(sched, "mark_job_run", lambda *_a, **_kw: None)
        monkeypatch.setattr(sched, "_deliver_result", lambda *_a, **_kw: None)

        n = sched.tick(verbose=False)
        assert n == 0  # skipped, not dispatched
        assert dispatched == []

        sched._running_job_ids.discard("guard-job")
        sched._shutdown_parallel_pool()


    def test_fire_claim_is_acquired_only_when_executor_worker_starts(self, monkeypatch):
        """Queue wait must not consume the durable claim TTL."""
        import cron.scheduler as sched

        sched._running_job_ids.clear()
        job = {
            "id": "queued-job",
            "name": "queued",
            "prompt": "test",
            "schedule": "every 5m",
            "enabled": True,
            "next_run_at": "2020-01-01T00:00:00",
            "deliver": "local",
        }
        submitted = []
        claim_calls = []

        class DeferredPool:
            def submit(self, callback):
                future = concurrent.futures.Future()
                submitted.append((callback, future))
                return future

        monkeypatch.setattr(sched, "get_due_jobs", lambda: [job])
        monkeypatch.setattr(sched, "_get_parallel_pool", lambda _workers: DeferredPool())
        monkeypatch.setattr(
            sched,
            "create_execution",
            lambda *_a, **_kw: {"id": "execution-1"},
        )
        monkeypatch.setattr(
            sched,
            "claim_job_for_fire",
            lambda job_id, **kwargs: claim_calls.append((job_id, kwargs))
            or {**job, "fire_claim": {"by": "worker-owner", "at": "now"}},
        )
        monkeypatch.setattr(sched, "run_one_job", lambda *_a, **_kw: True)

        assert sched.tick(verbose=False, sync=False) == 1
        assert claim_calls == []
        assert len(submitted) == 1

        callback, future = submitted[0]
        result = callback()
        future.set_result(result)

        assert claim_calls == [("queued-job", {"return_job": True})]
        assert "queued-job" not in sched._running_job_ids


    def test_create_execution_failure_does_not_wedge_running_set(self, tmp_path, monkeypatch):
        """create_execution failures clear the running lock and still allow next jobs."""
        import cron.scheduler as sched

        sched._parallel_pool = None
        sched._parallel_pool_max_workers = None
        sched._running_job_ids.clear()

        failing_job = {
            "id": "failing-job",
            "name": "failing-job",
            "prompt": "test",
            "schedule": "every 5m",
            "enabled": True,
            "next_run_at": "2020-01-01T00:00:00",
            "deliver": "local",
        }
        healthy_job = {
            "id": "healthy-job",
            "name": "healthy-job",
            "prompt": "test",
            "schedule": "every 5m",
            "enabled": True,
            "next_run_at": "2020-01-01T00:00:00",
            "deliver": "local",
        }

        called = []

        def create_execution_side_effect(job_id, source):
            if job_id == "failing-job":
                raise RuntimeError("execution ledger unavailable")
            return {"id": f"{job_id}-execution"}

        monkeypatch.setattr(sched, "get_due_jobs", lambda: [failing_job, healthy_job])
        monkeypatch.setattr(sched, "advance_next_runs", lambda *_a, **_kw: 0)
        monkeypatch.setattr(sched, "create_execution", create_execution_side_effect)
        monkeypatch.setattr(sched, "run_job", lambda j, **_kw: called.append(j["id"]) or (True, "out", "resp", None))
        monkeypatch.setattr(sched, "save_job_output", lambda *_a, **_kw: None)
        monkeypatch.setattr(sched, "mark_job_run", lambda *_a, **_kw: None)
        monkeypatch.setattr(sched, "_deliver_result", lambda *_a, **_kw: None)
        monkeypatch.setattr(sched, "finish_execution", lambda *_a, **_kw: None)
        monkeypatch.setattr(sched, "claim_dispatch", lambda *_a, **_kw: True)
        monkeypatch.setattr(
            sched,
            "claim_job_for_fire",
            lambda job_id, **_kw: dict(
                healthy_job, fire_claim={"by": "test-owner", "at": "now"}
            )
            if job_id == "healthy-job"
            else None,
        )
        monkeypatch.setattr(sched, "mark_execution_running", lambda *_a, **_kw: None)
        monkeypatch.setattr(sched, "heartbeat_fire_claim", lambda *_a, **_kw: True)

        n = sched.tick(verbose=False)

        assert n == 1
        assert called == ["healthy-job"]
        assert "failing-job" not in sched._running_job_ids
        assert "healthy-job" not in sched._running_job_ids

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
        monkeypatch.setattr(sched, "claim_job_for_fire", lambda *_a, **_kw: True)
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

        def slow_run(j, *, defer_agent_teardown=None, **_kw):
            barrier.wait()  # blocks until test thread also waits
            return True, "out", "resp", None

        monkeypatch.setattr(sched, "get_due_jobs", lambda: [job])
        monkeypatch.setattr(sched, "claim_job_for_fire", lambda *_a, **_kw: True)
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


class TestWorkdirParallelPool:
    """Task-scoped workdir jobs use the normal persistent parallel pool."""

    def test_workdir_job_does_not_block_ticker(self, tmp_path, monkeypatch):
        """sync=False returns immediately even when a workdir job is slow."""
        import cron.scheduler as sched

        sched._parallel_pool = None
        sched._parallel_pool_max_workers = None
        sched._running_job_ids.clear()

        job = {
            "id": "slow-workdir",
            "name": "slow-workdir",
            "prompt": "test",
            "schedule": "every 5m",
            "enabled": True,
            "next_run_at": "2020-01-01T00:00:00",
            "deliver": "local",
            "workdir": str(tmp_path),
        }

        barrier = threading.Barrier(2, timeout=5)

        def slow_run(j, *, defer_agent_teardown=None, **_kw):
            barrier.wait()
            return True, "out", "resp", None

        monkeypatch.setattr(sched, "get_due_jobs", lambda: [job])
        monkeypatch.setattr(sched, "claim_job_for_fire", lambda *_a, **_kw: True)
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

    def test_workdir_running_guard_prevents_double_dispatch(self, tmp_path, monkeypatch):
        """A workdir job already in _running_job_ids is skipped on next tick."""
        import cron.scheduler as sched

        sched._parallel_pool = None
        sched._parallel_pool_max_workers = None
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
        monkeypatch.setattr(sched, "claim_job_for_fire", lambda *_a, **_kw: True)
        monkeypatch.setattr(sched, "run_job", lambda j, **_kw: dispatched.append(j["id"]) or (True, "out", "resp", None))
        monkeypatch.setattr(sched, "save_job_output", lambda *_a, **_kw: None)
        monkeypatch.setattr(sched, "mark_job_run", lambda *_a, **_kw: None)
        monkeypatch.setattr(sched, "_deliver_result", lambda *_a, **_kw: None)

        n = sched.tick(verbose=False)
        assert n == 0  # skipped, not dispatched
        assert dispatched == []

        sched._running_job_ids.discard("guard-seq")
        sched._shutdown_parallel_pool()

class TestTickBatchAdvance:
    """The tick's pre-dispatch advance must go through advance_next_runs
    exactly once with the whole due set — a revert to the per-job loop
    (or back to advance_next_run) must fail this test, not slip past the
    helper-level I/O pin."""

    def test_tick_calls_advance_next_runs_once_with_all_due_ids(self, tmp_path, monkeypatch):
        import cron.scheduler as sched

        sched._parallel_pool = None
        sched._parallel_pool_max_workers = None
        sched._running_job_ids.clear()

        jobs = [
            {"id": f"job-{i}", "name": f"Job {i}", "prompt": "test",
             "schedule": "every 5m", "enabled": True,
             "next_run_at": "2020-01-01T00:00:00", "deliver": "local"}
            for i in range(4)
        ]

        advance_calls = []
        monkeypatch.setattr(sched, "get_due_jobs", lambda: jobs)
        monkeypatch.setattr(
            sched, "advance_next_runs",
            lambda ids: advance_calls.append(list(ids)) or len(list(ids)))
        monkeypatch.setattr(sched, "run_job", lambda j, **_kw: (True, "out", "resp", None))
        monkeypatch.setattr(sched, "save_job_output", lambda *_a, **_kw: "/tmp/out")
        monkeypatch.setattr(sched, "mark_job_run", lambda *_a, **_kw: None)
        monkeypatch.setattr(sched, "_deliver_result", lambda *_a, **_kw: None)

        n = sched.tick(verbose=False)

        assert n == 4
        assert advance_calls == [["job-0", "job-1", "job-2", "job-3"]], (
            f"tick must batch-advance the due set in ONE call; got {advance_calls}")

        sched._shutdown_parallel_pool()
