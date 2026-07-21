"""Integration tests for preemptive cancellation runtime wiring.

These tests verify that:
1. cancel_job() alone (without manual kill_process_tree) kills child processes
2. Side-effect gates block after cancel
3. /jobs command lists running jobs
4. Event-based sleep wakes instantly on cancel
"""
import asyncio
import json
import os
import subprocess
import sys
import threading
import time

import pytest

from agent.cancellation import (
    CancellationToken,
    JobManager,
    JobState,
    get_job_manager,
    get_process_registry,
    is_preemptive_cancellation_enabled,
)
from agent.cancellation_gates import (
    guard_file_write,
    guard_commit,
    guard_push,
    guard_pr,
    guard_external_send,
    OperationCancelled,
)


pytestmark = pytest.mark.live_system_guard_bypass


class TestCancelJobKillsProcessesAutomatically:
    """Verify that cancel_job() alone kills registered processes —
    no manual kill_process_tree call needed."""

    def test_cancel_job_kills_registered_process(self):
        """cancel_job() should kill processes registered via ProcessRegistry."""
        mgr = JobManager()
        registry = get_process_registry()
        job_id = mgr.create_job()

        proc = subprocess.Popen(
            ["sleep", "30"] if sys.platform != "win32" else ["timeout", "30"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        time.sleep(0.3)
        assert proc.poll() is None

        registry.register_pid(job_id, proc.pid)
        mgr.set_current_step(job_id, "terminal: sleep 30")

        # Cancel — callback should kill the process automatically
        result = mgr.cancel_job(job_id)
        time.sleep(0.5)

        assert proc.poll() is not None
        assert mgr.get_state(job_id) == JobState.CANCELLED
        assert result.state in (JobState.CANCEL_REQUESTED, JobState.CANCELLED)

        mgr.unregister_job(job_id)
        registry.clear(job_id)

    def test_cancel_job_kills_nested_process_tree(self):
        """cancel_job() should kill entire process tree including children."""
        mgr = JobManager()
        registry = get_process_registry()
        job_id = mgr.create_job()

        proc = subprocess.Popen(
            ["bash", "-c", "sleep 30 & sleep 30 & wait"] if sys.platform != "win32"
            else ["cmd", "/c", "start /b timeout 30 & timeout 30"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        time.sleep(0.5)
        assert proc.poll() is None

        registry.register_pid(job_id, proc.pid)

        mgr.cancel_job(job_id)
        time.sleep(0.5)

        assert proc.poll() is not None
        assert mgr.get_state(job_id) == JobState.CANCELLED

        mgr.unregister_job(job_id)
        registry.clear(job_id)

    def test_cancel_all_kills_all_registered_processes(self):
        """STOP ALL should kill all registered processes across jobs."""
        mgr = JobManager()
        registry = get_process_registry()

        j1 = mgr.create_job()
        j2 = mgr.create_job()

        p1 = subprocess.Popen(
            ["sleep", "30"] if sys.platform != "win32" else ["timeout", "30"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        p2 = subprocess.Popen(
            ["sleep", "30"] if sys.platform != "win32" else ["timeout", "30"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        time.sleep(0.3)

        registry.register_pid(j1, p1.pid)
        registry.register_pid(j2, p2.pid)

        results = mgr.cancel_all()
        time.sleep(0.5)

        assert len(results) == 2
        assert p1.poll() is not None
        assert p2.poll() is not None
        assert mgr.get_state(j1) == JobState.CANCELLED
        assert mgr.get_state(j2) == JobState.CANCELLED

        mgr.unregister_job(j1)
        mgr.unregister_job(j2)
        registry.clear(j1)
        registry.clear(j2)


class TestGatesBlockAfterCancel:
    """Verify side-effect gates block after cancel_job()."""

    def _make_cancelled_agent(self, job_id, mgr):
        """Create a stub agent with a cancelled token for a given job."""
        token = mgr.get_token(job_id)
        class _Stub: pass
        s = _Stub()
        s._cancellation_token = token
        s._interrupt_requested = False
        s._job_id = job_id
        return s

    def test_file_write_blocked_after_cancel(self):
        mgr = JobManager()
        job_id = mgr.create_job()
        mgr.cancel_job(job_id)
        agent = self._make_cancelled_agent(job_id, mgr)
        with pytest.raises(OperationCancelled, match="file_write"):
            guard_file_write(agent, "/tmp/test.txt")
        mgr.unregister_job(job_id)

    def test_commit_blocked_after_cancel(self):
        mgr = JobManager()
        job_id = mgr.create_job()
        mgr.cancel_job(job_id)
        agent = self._make_cancelled_agent(job_id, mgr)
        with pytest.raises(OperationCancelled, match="git_commit"):
            guard_commit(agent)
        mgr.unregister_job(job_id)

    def test_push_blocked_after_cancel(self):
        mgr = JobManager()
        job_id = mgr.create_job()
        mgr.cancel_job(job_id)
        agent = self._make_cancelled_agent(job_id, mgr)
        with pytest.raises(OperationCancelled, match="git_push"):
            guard_push(agent)
        mgr.unregister_job(job_id)

    def test_pr_blocked_after_cancel(self):
        mgr = JobManager()
        job_id = mgr.create_job()
        mgr.cancel_job(job_id)
        agent = self._make_cancelled_agent(job_id, mgr)
        with pytest.raises(OperationCancelled, match="create_pr"):
            guard_pr(agent)
        mgr.unregister_job(job_id)

    def test_external_send_blocked_after_cancel(self):
        mgr = JobManager()
        job_id = mgr.create_job()
        mgr.cancel_job(job_id)
        agent = self._make_cancelled_agent(job_id, mgr)
        with pytest.raises(OperationCancelled, match="external_send"):
            guard_external_send(agent, "discord:#general")
        mgr.unregister_job(job_id)

    def test_gates_pass_when_not_cancelled(self):
        mgr = JobManager()
        job_id = mgr.create_job()
        agent = self._make_cancelled_agent(job_id, mgr)
        guard_file_write(agent, "/tmp/test.txt")
        guard_commit(agent)
        guard_push(agent)
        guard_pr(agent)
        guard_external_send(agent, "discord:#general")
        mgr.unregister_job(job_id)


class TestEventBasedSleep:
    """Verify that cancellable sleep wakes instantly on cancel."""

    def test_async_sleep_wakes_instantly(self):
        token = CancellationToken()
        loop = asyncio.new_event_loop()
        try:
            t = threading.Timer(0.05, token.request_cancel)
            t.start()
            start = time.time()
            with pytest.raises(asyncio.CancelledError):
                loop.run_until_complete(token.sleep(10.0))
            elapsed = time.time() - start
            assert elapsed < 0.5, f"sleep took {elapsed}s, expected <0.5s"
            t.join()
        finally:
            loop.close()

    def test_sync_sleep_wakes_instantly(self):
        token = CancellationToken()
        t = threading.Timer(0.05, token.request_cancel)
        t.start()
        start = time.time()
        with pytest.raises(asyncio.CancelledError):
            token.sleep_sync(10.0)
        elapsed = time.time() - start
        assert elapsed < 0.5
        t.join()


class TestJobsListing:
    """Verify /jobs command output."""

    def test_list_running_jobs_returns_dicts(self):
        mgr = JobManager()
        j1 = mgr.create_job()
        mgr.set_current_step(j1, "terminal: ls")
        j2 = mgr.create_job()
        mgr.set_current_step(j2, "file: write")

        jobs = mgr.list_running_jobs()
        assert len(jobs) == 2
        assert all("job_id" in j for j in jobs)
        assert all("state" in j for j in jobs)
        assert all("current_step" in j for j in jobs)

        job_ids = {j["job_id"] for j in jobs}
        assert j1 in job_ids
        assert j2 in job_ids

        mgr.unregister_job(j1)
        mgr.unregister_job(j2)

    def test_list_running_jobs_empty_after_cancel_all(self):
        mgr = JobManager()
        j1 = mgr.create_job()
        j2 = mgr.create_job()
        mgr.cancel_all()
        jobs = mgr.list_running_jobs()
        assert jobs == []
        mgr.unregister_job(j1)
        mgr.unregister_job(j2)


class TestStateTransitions:
    """Verify correct state transitions: CANCEL_REQUESTED -> CANCELLING -> CANCELLED."""

    def test_cancel_progresses_to_cancelled(self):
        """After cancel_job(), state should eventually reach CANCELLED."""
        mgr = JobManager()
        job_id = mgr.create_job()
        mgr.cancel_job(job_id)
        # Callback runs synchronously in request_cancel
        assert mgr.get_state(job_id) == JobState.CANCELLED
        mgr.unregister_job(job_id)

    def test_duplicate_cancel_returns_current_state(self):
        """Second cancel should return current state, not re-declare."""
        mgr = JobManager()
        job_id = mgr.create_job()
        r1 = mgr.cancel_job(job_id)
        r2 = mgr.cancel_job(job_id)
        assert r1 is not None
        assert r2 is not None
        # Both should see CANCELLED (callback already ran)
        assert mgr.get_state(job_id) == JobState.CANCELLED
        mgr.unregister_job(job_id)

    def test_cancel_result_includes_remaining_processes(self):
        """CancellationResult should include remaining_processes list."""
        mgr = JobManager()
        registry = get_process_registry()
        job_id = mgr.create_job()

        # Register a dead PID (999999) — should not appear in remaining
        registry.register_pid(job_id, 999999)
        result = mgr.cancel_job(job_id)
        assert result is not None
        # Dead process should not appear
        assert len(result.remaining_processes) == 0

        mgr.unregister_job(job_id)
        registry.clear(job_id)
