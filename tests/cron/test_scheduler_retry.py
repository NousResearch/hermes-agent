"""Tests for cron job retry-on-timeout behaviour in _process_job."""
import pytest
from unittest.mock import MagicMock


_JOB = {"id": "job-1", "name": "test-job"}


def _make_run_job(results):
    it = iter(results)
    calls = []

    def run_job(job):
        calls.append(job)
        value = next(it)
        if isinstance(value, BaseException):
            raise value
        return value

    run_job.calls = calls
    return run_job


def _ok():
    return (True, "output", "response", None)


@pytest.fixture(autouse=True)
def _base_mocks(monkeypatch):
    monkeypatch.setattr("cron.scheduler.get_due_jobs", lambda *a, **kw: [_JOB])
    monkeypatch.setattr("cron.scheduler.advance_next_run", lambda *a, **kw: None)
    monkeypatch.setattr("cron.scheduler.save_job_output", lambda *a, **kw: "/tmp/out.txt")
    monkeypatch.setattr("cron.scheduler._deliver_result", lambda *a, **kw: None)
    monkeypatch.setattr("cron.scheduler.mark_job_run", lambda *a, **kw: None)
    monkeypatch.setattr("cron.scheduler.fcntl", MagicMock())


def test_retries_once_on_timeout_then_succeeds(monkeypatch):
    stub = _make_run_job([TimeoutError("slow"), _ok()])
    monkeypatch.setattr("cron.scheduler.run_job", stub)

    from cron.scheduler import tick
    result = tick(verbose=False)

    assert len(stub.calls) == 2
    assert result == 1  # job succeeded


def test_no_retry_on_success(monkeypatch):
    stub = _make_run_job([_ok()])
    monkeypatch.setattr("cron.scheduler.run_job", stub)

    from cron.scheduler import tick
    result = tick(verbose=False)

    assert len(stub.calls) == 1
    assert result == 1


def test_double_timeout_fails_after_two_attempts(monkeypatch):
    stub = _make_run_job([TimeoutError("slow"), TimeoutError("still slow")])
    monkeypatch.setattr("cron.scheduler.run_job", stub)

    from cron.scheduler import tick
    result = tick(verbose=False)

    assert len(stub.calls) == 2
    assert result == 0  # job failed


def test_non_timeout_exception_not_retried(monkeypatch):
    stub = _make_run_job([RuntimeError("bad input")])
    monkeypatch.setattr("cron.scheduler.run_job", stub)

    from cron.scheduler import tick
    result = tick(verbose=False)

    assert len(stub.calls) == 1
    assert result == 0  # job failed, no retry
