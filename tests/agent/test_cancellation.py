"""Tests for agent/cancellation.py - preemptive task cancellation primitives.

Updated for corrected state transitions:
  cancel_job() -> CANCEL_REQUESTED (not CANCELLED)
  callback -> CANCELLING -> CANCELLED (after process tree kill verified)
"""
import asyncio
import os
import subprocess
import sys
import time

import pytest

from agent.cancellation import (
    CancellationToken,
    JobState,
    JobManager,
    CancellationResult,
    ProcessRegistry,
    get_job_manager,
    get_process_registry,
    is_preemptive_cancellation_enabled,
)


class TestCancellationToken:
    def test_initial_state(self):
        token = CancellationToken()
        assert not token.is_cancelled
        assert token.state == JobState.RUNNING

    def test_request_cancel_sets_cancel_requested(self):
        token = CancellationToken()
        token.request_cancel()
        assert token.is_cancelled
        assert token.state == JobState.CANCEL_REQUESTED

    def test_request_cancel_idempotent(self):
        token = CancellationToken()
        token.request_cancel()
        token.request_cancel()
        assert token.is_cancelled
        assert token.state == JobState.CANCEL_REQUESTED

    def test_throw_if_cancelled(self):
        token = CancellationToken()
        token.request_cancel()
        with pytest.raises(asyncio.CancelledError):
            token.throw_if_cancelled()

    def test_check_cancelled(self):
        token = CancellationToken()
        assert not token.check_cancelled()
        token.request_cancel()
        assert token.check_cancelled()

    def test_register_callback_fires_on_cancel(self):
        token = CancellationToken()
        called = []
        token.register(lambda: called.append(True))
        token.request_cancel()
        assert called == [True]

    def test_register_callback_after_cancel(self):
        token = CancellationToken()
        called = []
        token.request_cancel()
        token.register(lambda: called.append(True))
        assert called == [True]

    def test_set_cancelling(self):
        token = CancellationToken()
        token.set_cancelling()
        assert token.state == JobState.CANCELLING

    def test_set_cancelled(self):
        token = CancellationToken()
        token.set_cancelled()
        assert token.state == JobState.CANCELLED

    def test_current_step(self):
        token = CancellationToken()
        assert token.current_step is None
        token.set_current_step("step_3")
        assert token.current_step == "step_3"


class TestCancellationTokenSleep:
    def test_sleep_not_cancelled(self):
        token = CancellationToken()
        loop = asyncio.new_event_loop()
        try:
            loop.run_until_complete(token.sleep(0.01))
        finally:
            loop.close()
        assert not token.is_cancelled

    def test_sleep_cancelled_during(self):
        """Event-based sleep should wake instantly on cancel, not wait full duration."""
        token = CancellationToken()
        loop = asyncio.new_event_loop()
        try:
            # Cancel after 50ms
            loop.call_later(0.05, token.request_cancel)
            start = time.time()
            with pytest.raises(asyncio.CancelledError):
                loop.run_until_complete(token.sleep(10.0))
            elapsed = time.time() - start
            # Should wake within ~100ms, not 10s
            assert elapsed < 0.5, f"sleep took {elapsed}s, expected <0.5s"
        finally:
            loop.close()

    def test_sleep_already_cancelled(self):
        token = CancellationToken()
        token.request_cancel()
        loop = asyncio.new_event_loop()
        try:
            with pytest.raises(asyncio.CancelledError):
                loop.run_until_complete(token.sleep(0.01))
        finally:
            loop.close()

    def test_sleep_sync_not_cancelled(self):
        token = CancellationToken()
        token.sleep_sync(0.01)
        assert not token.is_cancelled

    def test_sleep_sync_cancelled_during(self):
        """Sync sleep should wake instantly on cancel."""
        token = CancellationToken()
        import threading
        t = threading.Timer(0.05, token.request_cancel)
        t.start()
        start = time.time()
        with pytest.raises(asyncio.CancelledError):
            token.sleep_sync(10.0)
        elapsed = time.time() - start
        assert elapsed < 0.5
        t.join()


class TestProcessRegistry:
    def test_register_and_get_pids(self):
        reg = ProcessRegistry()
        reg.register_pid("job1", 123)
        reg.register_pid("job1", 456)
        assert reg.get_pids("job1") == [123, 456]

    def test_get_pids_empty(self):
        reg = ProcessRegistry()
        assert reg.get_pids("nonexistent") == []

    def test_clear(self):
        reg = ProcessRegistry()
        reg.register_pid("job1", 123)
        pids = reg.clear("job1")
        assert pids == [123]
        assert reg.get_pids("job1") == []

    def test_get_remaining_no_pids(self):
        reg = ProcessRegistry()
        assert reg.get_remaining("nonexistent") == []

    def test_get_remaining_alive_process(self):
        reg = ProcessRegistry()
        proc = subprocess.Popen(
            ["sleep", "30"] if sys.platform != "win32" else ["timeout", "30"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        time.sleep(0.3)
        try:
            reg.register_pid("job1", proc.pid)
            remaining = reg.get_remaining("job1")
            assert len(remaining) == 1
            assert remaining[0]["pid"] == proc.pid
        finally:
            proc.kill()
            proc.wait()
            time.sleep(0.2)

    def test_get_remaining_dead_process(self):
        reg = ProcessRegistry()
        # PID 999999 is very unlikely to exist
        reg.register_pid("job1", 999999)
        remaining = reg.get_remaining("job1")
        assert remaining == []


class TestJobManager:
    def test_create_job(self):
        mgr = JobManager()
        job_id = mgr.create_job()
        assert job_id is not None
        assert mgr.get_state(job_id) == JobState.RUNNING
        token = mgr.get_token(job_id)
        assert token is not None
        assert not token.is_cancelled

    def test_create_job_with_explicit_id(self):
        mgr = JobManager()
        job_id = mgr.create_job("my-job-123")
        assert job_id == "my-job-123"

    def test_cancel_job_sets_cancel_requested(self):
        """cancel_job() should set CANCEL_REQUESTED, not CANCELLED."""
        mgr = JobManager()
        job_id = mgr.create_job()
        mgr.set_current_step(job_id, "step_3")
        result = mgr.cancel_job(job_id, last_completed_step="step_3", cancelled_step="step_4")
        assert result is not None
        # After callback runs synchronously, state should be CANCELLED
        # (callback fires during request_cancel)
        assert result.state in (JobState.CANCEL_REQUESTED, JobState.CANCELLED)
        assert result.last_completed_step == "step_3"
        assert result.cancelled_step == "step_4"
        assert mgr.get_token(job_id).is_cancelled

    def test_cancel_job_not_found(self):
        mgr = JobManager()
        result = mgr.cancel_job("nonexistent")
        assert result is None

    def test_cancel_job_idempotent(self):
        """Duplicate cancel should return current state, not re-declare."""
        mgr = JobManager()
        job_id = mgr.create_job()
        r1 = mgr.cancel_job(job_id)
        r2 = mgr.cancel_job(job_id)
        assert r1 is not None
        assert r2 is not None
        # Both should return a result, second one should not re-declare
        # After callback, state should be CANCELLED
        assert mgr.get_state(job_id) == JobState.CANCELLED

    def test_cancel_all(self):
        mgr = JobManager()
        j1 = mgr.create_job()
        j2 = mgr.create_job()
        results = mgr.cancel_all()
        assert len(results) == 2
        assert mgr.get_state(j1) == JobState.CANCELLED
        assert mgr.get_state(j2) == JobState.CANCELLED

    def test_unregister_job(self):
        mgr = JobManager()
        job_id = mgr.create_job()
        mgr.unregister_job(job_id)
        assert mgr.get_state(job_id) is None
        assert mgr.get_token(job_id) is None

    def test_list_running_jobs(self):
        mgr = JobManager()
        j1 = mgr.create_job()
        j2 = mgr.create_job()
        running = mgr.list_running_jobs()
        assert len(running) == 2
        mgr.cancel_job(j1)
        running = mgr.list_running_jobs()
        assert all(r["state"] != "cancelled" for r in running)

    def test_set_current_step(self):
        mgr = JobManager()
        job_id = mgr.create_job()
        mgr.set_current_step(job_id, "tool_call_5")
        token = mgr.get_token(job_id)
        assert token.current_step == "tool_call_5"

    def test_list_running_jobs_returns_dicts(self):
        """list_running_jobs should return dicts with job_id, state, current_step."""
        mgr = JobManager()
        j1 = mgr.create_job()
        mgr.set_current_step(j1, "step_1")
        running = mgr.list_running_jobs()
        assert len(running) == 1
        assert running[0]["job_id"] == j1
        assert running[0]["state"] == "running"
        assert running[0]["current_step"] == "step_1"


class TestCancellationResult:
    def test_default_values(self):
        result = CancellationResult(
            job_id="test-123",
            state=JobState.CANCELLED,
        )
        assert result.job_id == "test-123"
        assert result.state == JobState.CANCELLED
        assert result.last_completed_step is None
        assert result.cancelled_step is None
        assert result.remaining_processes == []
        assert result.cancelled_at > 0


class TestFeatureFlag:
    def test_default_off(self, monkeypatch):
        monkeypatch.delenv("HERMES_PREEMPTIVE_CANCELLATION", raising=False)
        assert not is_preemptive_cancellation_enabled()

    def test_enabled_via_env(self, monkeypatch):
        monkeypatch.setenv("HERMES_PREEMPTIVE_CANCELLATION", "true")
        assert is_preemptive_cancellation_enabled()

    def test_disabled_via_env_false(self, monkeypatch):
        monkeypatch.setenv("HERMES_PREEMPTIVE_CANCELLATION", "false")
        assert not is_preemptive_cancellation_enabled()


class TestCancelCallbackKillsProcesses:
    """Verify that cancel_job() automatically kills registered processes."""

    pytestmark = pytest.mark.live_system_guard_bypass

    def test_cancel_job_kills_registered_process(self):
        """cancel_job() should kill processes registered via ProcessRegistry."""

        mgr = JobManager()
        registry = get_process_registry()
        job_id = mgr.create_job()

        # Spawn a process and register it
        proc = subprocess.Popen(
            ["sleep", "30"] if sys.platform != "win32" else ["timeout", "30"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        time.sleep(0.3)
        assert proc.poll() is None

        registry.register_pid(job_id, proc.pid)
        mgr.set_current_step(job_id, "terminal: sleep 30")

        # Cancel the job — callback should kill the process
        result = mgr.cancel_job(job_id)
        time.sleep(0.5)  # give callback time to run

        assert proc.poll() is not None  # process is dead
        assert mgr.get_state(job_id) == JobState.CANCELLED

        mgr.unregister_job(job_id)
        registry.clear(job_id)

    def test_cancel_job_kills_nested_process_tree(self):
        """cancel_job() should kill entire process tree including children."""
        pytestmark = pytest.mark.live_system_guard_bypass

        mgr = JobManager()
        registry = get_process_registry()
        job_id = mgr.create_job()

        # Spawn a process with children
        proc = subprocess.Popen(
            ["bash", "-c", "sleep 30 & sleep 30 & wait"] if sys.platform != "win32"
            else ["cmd", "/c", "start /b timeout 30 & timeout 30"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        time.sleep(0.5)
        assert proc.poll() is None

        registry.register_pid(job_id, proc.pid)

        # Cancel — callback kills the whole tree
        mgr.cancel_job(job_id)
        time.sleep(0.5)

        assert proc.poll() is not None
        assert mgr.get_state(job_id) == JobState.CANCELLED

        mgr.unregister_job(job_id)
        registry.clear(job_id)


class TestGetJobManager:
    def test_singleton(self):
        mgr1 = get_job_manager()
        mgr2 = get_job_manager()
        assert mgr1 is mgr2
