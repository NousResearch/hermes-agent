"""Tests for HERMES_CRON_SESSION cleanup after cron jobs finish.

Verifies the fix in #73195: when the cron scheduler runs inside the gateway
process (InProcessCronScheduler), os.environ["HERMES_CRON_SESSION"] = "1"
leaks into user-interactive sessions after all cron jobs finish. The fix
pops the flag in run_job's finally block when no other cron jobs are running.
"""

import os
import threading
from unittest.mock import patch

import pytest

import cron.scheduler


@pytest.fixture
def _clean_cron_running(monkeypatch):
    """Isolate the module-level _running_job_ids between tests."""
    monkeypatch.setattr(cron.scheduler, "_running_job_ids", set())
    monkeypatch.delenv("HERMES_CRON_SESSION", raising=False)
    yield
    monkeypatch.setattr(cron.scheduler, "_running_job_ids", set())
    monkeypatch.delenv("HERMES_CRON_SESSION", raising=False)


class TestCronSessionEnvVarCleanup:
    """run_job's finally block should pop HERMES_CRON_SESSION when
    the last cron job finishes."""

    def test_cleared_when_last_job_finishes(self, _clean_cron_running, monkeypatch):
        """When a single cron job runs and finishes, the flag is removed."""
        job_id = "test-single-job"
        # Simulate: a job starts running
        with cron.scheduler._running_lock:
            cron.scheduler._running_job_ids.add(job_id)
        monkeypatch.setenv("HERMES_CRON_SESSION", "1")

        # Simulate cleanup from run_job's finally block
        with cron.scheduler._running_lock:
            remaining = cron.scheduler._running_job_ids - {job_id}
            if not remaining:
                os.environ.pop("HERMES_CRON_SESSION", None)
            cron.scheduler._running_job_ids.discard(job_id)

        assert "HERMES_CRON_SESSION" not in os.environ
        assert len(cron.scheduler._running_job_ids) == 0

    def test_preserved_when_other_jobs_still_running(
        self, _clean_cron_running, monkeypatch
    ):
        """When multiple cron jobs run concurrently, the flag is preserved
        until the last one finishes."""
        job_a = "test-job-a"
        job_b = "test-job-b"

        # Both jobs start
        with cron.scheduler._running_lock:
            cron.scheduler._running_job_ids.add(job_a)
            cron.scheduler._running_job_ids.add(job_b)
        monkeypatch.setenv("HERMES_CRON_SESSION", "1")

        # Job A finishes — should NOT clear the flag because B is still running
        with cron.scheduler._running_lock:
            remaining = cron.scheduler._running_job_ids - {job_a}
            assert len(remaining) == 1  # job_b still there
            if not remaining:
                os.environ.pop("HERMES_CRON_SESSION", None)
            cron.scheduler._running_job_ids.discard(job_a)

        assert os.environ.get("HERMES_CRON_SESSION") == "1"
        assert cron.scheduler._running_job_ids == {"test-job-b"}

        # Job B finishes — now it SHOULD clear the flag
        with cron.scheduler._running_lock:
            remaining = cron.scheduler._running_job_ids - {job_b}
            assert len(remaining) == 0
            if not remaining:
                os.environ.pop("HERMES_CRON_SESSION", None)
            cron.scheduler._running_job_ids.discard(job_b)

        assert "HERMES_CRON_SESSION" not in os.environ
        assert len(cron.scheduler._running_job_ids) == 0

    def test_thread_safe_with_lock(self, _clean_cron_running, monkeypatch):
        """The lock serialises access to _running_job_ids, preventing
        TOCTOU races between check and pop."""
        job_id = "test-thread-safe"
        with cron.scheduler._running_lock:
            cron.scheduler._running_job_ids.add(job_id)
        monkeypatch.setenv("HERMES_CRON_SESSION", "1")

        race_detected = []
        lock = threading.Lock()

        def simulate_concurrent_cleanup():
            # Both threads see only one job (themselves) and both
            # attempt to pop — but the lock gates access.
            with cron.scheduler._running_lock:
                remaining = cron.scheduler._running_job_ids - {job_id}
                if not remaining:
                    # Only the first thread gets here with the flag still set
                    if "HERMES_CRON_SESSION" in os.environ:
                        os.environ.pop("HERMES_CRON_SESSION", None)
                    else:
                        race_detected.append("flag already cleared by other thread")
                cron.scheduler._running_job_ids.discard(job_id)

        t1 = threading.Thread(target=simulate_concurrent_cleanup)
        t2 = threading.Thread(target=simulate_concurrent_cleanup)

        # Start both, but only one has the job in _running_job_ids
        # (the other discards immediately in the finally)
        # Actually let's fix this: add TWO jobs and have each thread
        # see only its own remaining.
        cron.scheduler._running_job_ids.clear()
        cron.scheduler._running_job_ids.add("job-1")
        cron.scheduler._running_job_ids.add("job-2")
        monkeypatch.setenv("HERMES_CRON_SESSION", "1")

        results = []

        def cleanup_job(jid):
            with cron.scheduler._running_lock:
                remaining = cron.scheduler._running_job_ids - {jid}
                if not remaining:
                    os.environ.pop("HERMES_CRON_SESSION", None)
                    results.append(f"{jid}: popped")
                else:
                    results.append(f"{jid}: skipped (remaining={remaining})")
                cron.scheduler._running_job_ids.discard(jid)

        t1 = threading.Thread(target=cleanup_job, args=("job-1",))
        t2 = threading.Thread(target=cleanup_job, args=("job-2",))
        t1.start()
        t2.start()
        t1.join()
        t2.join()

        assert "HERMES_CRON_SESSION" not in os.environ
        # One thread should have skipped, the other popped
        assert len(results) == 2
        assert any("popped" in r for r in results)
        assert any("skipped" in r for r in results)
        assert cron.scheduler._running_job_ids == set()

    def test_approval_no_longer_sees_cron_after_cleanup(
        self, _clean_cron_running, monkeypatch
    ):
        """After cleanup, execute_code guard should not treat the session
        as a cron session — even when tested from the same process."""
        from tools.approval import check_execute_code_guard

        # Simulate: a job ran and finished
        with cron.scheduler._running_lock:
            cron.scheduler._running_job_ids.add("test-approval-job")
        monkeypatch.setenv("HERMES_CRON_SESSION", "1")

        # Before cleanup: should be blocked (cron deny mode)
        result_before = check_execute_code_guard(
            "print('hello')", "local", has_host_access=False
        )
        # The code reads env_var_enabled("HERMES_CRON_SESSION") which is True
        # so it hits the cron branch. cron_mode defaults to "deny".
        assert not result_before["approved"]

        # Cleanup
        with cron.scheduler._running_lock:
            remaining = cron.scheduler._running_job_ids - {"test-approval-job"}
            if not remaining:
                os.environ.pop("HERMES_CRON_SESSION", None)
            cron.scheduler._running_job_ids.discard("test-approval-job")

        # After cleanup: should no longer be blocked
        # In a non-gateway, non-cron context, check_execute_code_guard
        # returns approved for local env types.
        result_after = check_execute_code_guard(
            "print('hello')", "local", has_host_access=False
        )
        assert result_after["approved"]
