"""Tests for shared process-containment primitives."""

from runtime import processes


def test_attach_self_is_noop_off_windows(monkeypatch):
    monkeypatch.setattr(processes, "_IS_WINDOWS", False)
    monkeypatch.setattr(processes, "_JOB_HANDLE", None)
    assert processes.attach_self_to_kill_on_close_job() is False


def test_attach_self_is_idempotent_once_job_is_owned(monkeypatch):
    sentinel = object()
    monkeypatch.setattr(processes, "_JOB_HANDLE", sentinel)
    assert processes.attach_self_to_kill_on_close_job() is True
    assert processes._JOB_HANDLE is sentinel
