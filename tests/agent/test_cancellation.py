"""Tests for agent/cancellation.py - preemptive task cancellation primitives."""
import asyncio
import pytest

from agent.cancellation import (
    CancellationToken,
    JobState,
    JobManager,
    CancellationResult,
    get_job_manager,
    is_preemptive_cancellation_enabled,
)


class TestCancellationToken:
    def test_initial_state(self):
        token = CancellationToken()
        assert not token.is_cancelled
        assert token.state == JobState.RUNNING

    def test_request_cancel(self):
        token = CancellationToken()
        token.request_cancel()
        assert token.is_cancelled
        assert token.state == JobState.CANCEL_REQUESTED

    def test_request_cancel_idempotent(self):
        token = CancellationToken()
        token.request_cancel()
        token.request_cancel()  # should not raise
        assert token.is_cancelled

    def test_throw_if_cancelled(self):
        token = CancellationToken()
        token.request_cancel()
        with pytest.raises(asyncio.CancelledError):
            token.throw_if_cancelled()

    def test_throw_if_cancelled_not_cancelled(self):
        token = CancellationToken()
        token.throw_if_cancelled()  # should not raise

    def test_check_cancelled(self):
        token = CancellationToken()
        assert not token.check_cancelled()
        token.request_cancel()
        assert token.check_cancelled()

    def test_register_callback(self):
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


class TestCancellationTokenAsync:
    def test_sleep_not_cancelled(self):
        token = CancellationToken()
        loop = asyncio.new_event_loop()
        try:
            loop.run_until_complete(token.sleep(0.01))
        finally:
            loop.close()
        assert not token.is_cancelled

    def test_sleep_cancelled_after(self):
        token = CancellationToken()
        token.request_cancel()
        loop = asyncio.new_event_loop()
        try:
            with pytest.raises(asyncio.CancelledError):
                loop.run_until_complete(token.sleep(0.01))
        finally:
            loop.close()


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

    def test_cancel_job(self):
        mgr = JobManager()
        job_id = mgr.create_job()
        mgr.set_current_step(job_id, "step_3")
        result = mgr.cancel_job(job_id, last_completed_step="step_3", cancelled_step="step_4")
        assert result is not None
        assert result.job_id == job_id
        assert result.state == JobState.CANCELLED
        assert result.last_completed_step == "step_3"
        assert result.cancelled_step == "step_4"
        assert mgr.get_state(job_id) == JobState.CANCELLED
        assert mgr.get_token(job_id).is_cancelled

    def test_cancel_job_not_found(self):
        mgr = JobManager()
        result = mgr.cancel_job("nonexistent")
        assert result is None

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
        assert set(running) == {j1, j2}
        mgr.cancel_job(j1)
        running = mgr.list_running_jobs()
        assert j1 not in running
        assert j2 in running

    def test_set_current_step(self):
        mgr = JobManager()
        job_id = mgr.create_job()
        mgr.set_current_step(job_id, "tool_call_5")
        token = mgr.get_token(job_id)
        assert token.current_step == "tool_call_5"


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

    def test_enabled_via_env_1(self, monkeypatch):
        monkeypatch.setenv("HERMES_PREEMPTIVE_CANCELLATION", "1")
        assert is_preemptive_cancellation_enabled()

    def test_disabled_via_env_false(self, monkeypatch):
        monkeypatch.setenv("HERMES_PREEMPTIVE_CANCELLATION", "false")
        assert not is_preemptive_cancellation_enabled()


class TestGetJobManager:
    def test_singleton(self):
        mgr1 = get_job_manager()
        mgr2 = get_job_manager()
        assert mgr1 is mgr2
