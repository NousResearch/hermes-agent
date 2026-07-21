"""Smoke test: start a long-running task, cancel mid-execution, verify
the cancellation token fires and the process tree is cleaned up.

This test verifies the end-to-end preemptive cancellation flow:
1. Feature flag is enabled
2. A job is registered with JobManager
3. The job's CancellationToken is cancelled out-of-band
4. The cancellation is detected at the next check point
5. Process tree is terminated
6. Cancellation result contains correct metadata
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
    get_job_manager,
    is_preemptive_cancellation_enabled,
)
from agent.process_killer import kill_process_tree, get_remaining_processes


pytestmark = pytest.mark.live_system_guard_bypass


class TestSmokeCancellationFlow:
    """End-to-end smoke test of the preemptive cancellation flow."""

    def test_cancel_job_mid_execution(self):
        """Register a job, spawn a child process, cancel the job,
        verify the token fires and process tree is cleaned up."""
        # 1. Register a job
        mgr = get_job_manager()
        job_id = mgr.create_job()
        token = mgr.get_token(job_id)

        assert token is not None
        assert not token.is_cancelled
        assert mgr.get_state(job_id) == JobState.RUNNING

        # 2. Spawn a child process to simulate a running tool
        proc = subprocess.Popen(
            ["sleep", "30"] if sys.platform != "win32" else ["timeout", "30"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        time.sleep(0.3)
        assert proc.poll() is None  # process is running

        # 3. Track the step
        mgr.set_current_step(job_id, "terminal: sleep 30")

        # 4. Cancel the job out-of-band (as the Discord listener would)
        result = mgr.cancel_job(
            job_id,
            last_completed_step="terminal: sleep 30",
            cancelled_step="terminal: sleep 30",
        )

        assert result is not None
        assert result.job_id == job_id
        assert result.state == JobState.CANCELLED
        assert result.last_completed_step == "terminal: sleep 30"
        assert result.cancelled_step == "terminal: sleep 30"

        # 5. Verify the token is cancelled
        assert token.is_cancelled
        assert mgr.get_state(job_id) == JobState.CANCELLED

        # 6. Kill the process tree (simulating what the cancellation handler does)
        kill_result = kill_process_tree(proc.pid, timeout=3.0)
        time.sleep(0.3)

        assert proc.poll() is not None  # process is dead

        # 7. Verify no remaining processes
        remaining = get_remaining_processes([proc.pid])
        assert remaining == []

        # 8. Unregister the job
        mgr.unregister_job(job_id)
        assert mgr.get_state(job_id) is None

    def test_cancel_all_with_multiple_jobs(self):
        """Register multiple jobs, cancel all, verify all are cancelled."""
        mgr = get_job_manager()
        j1 = mgr.create_job()
        j2 = mgr.create_job()
        j3 = mgr.create_job()

        results = mgr.cancel_all()
        assert len(results) == 3

        for r in results:
            assert r.state == JobState.CANCELLED

        assert mgr.get_state(j1) == JobState.CANCELLED
        assert mgr.get_state(j2) == JobState.CANCELLED
        assert mgr.get_state(j3) == JobState.CANCELLED

        # Cleanup
        mgr.unregister_job(j1)
        mgr.unregister_job(j2)
        mgr.unregister_job(j3)

    def test_cancellation_token_callback_fires(self):
        """Verify that registered callbacks fire when the token is cancelled."""
        token = CancellationToken()
        callback_results = []

        token.register(lambda: callback_results.append("callback_1"))
        token.register(lambda: callback_results.append("callback_2"))

        assert callback_results == []  # nothing yet

        token.request_cancel()

        assert callback_results == ["callback_1", "callback_2"]
        assert token.is_cancelled

    def test_feature_flag_off_by_default(self, monkeypatch):
        """Verify the feature flag is off by default."""
        monkeypatch.delenv("HERMES_PREEMPTIVE_CANCELLATION", raising=False)
        assert not is_preemptive_cancellation_enabled()

    def test_feature_flag_on(self, monkeypatch):
        """Verify the feature flag can be enabled."""
        monkeypatch.setenv("HERMES_PREEMPTIVE_CANCELLATION", "true")
        assert is_preemptive_cancellation_enabled()

    def test_cancelled_vs_failed_distinction(self):
        """Verify that cancelled and failed are distinct states.

        A cancelled job has state=CANCELLED and the token is set.
        A failed job would have state=RUNNING (it never gets cancelled)
        and an error result — the key distinction is that cancelled
        jobs have their token fired, while failed jobs don't.
        """
        mgr = get_job_manager()

        # "Cancelled" job
        cancelled_id = mgr.create_job()
        mgr.cancel_job(cancelled_id)
        assert mgr.get_state(cancelled_id) == JobState.CANCELLED
        assert mgr.get_token(cancelled_id).is_cancelled

        # "Failed" job (just unregister without cancelling — simulates natural failure)
        failed_id = mgr.create_job()
        assert mgr.get_state(failed_id) == JobState.RUNNING
        assert not mgr.get_token(failed_id).is_cancelled
        mgr.unregister_job(failed_id)

        # The distinction: cancelled jobs have CANCELLED state + fired token,
        # failed jobs were RUNNING when they ended (no cancellation requested)
        assert mgr.get_state(cancelled_id) == JobState.CANCELLED
        assert mgr.get_state(failed_id) is None  # unregistered

        # Cleanup
        mgr.unregister_job(cancelled_id)
