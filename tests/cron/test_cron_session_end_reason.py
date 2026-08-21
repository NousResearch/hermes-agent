"""Regression test for issue #88443: failed cron sessions must be stamped
end_reason="cron_failed", not "cron_complete".

run_job's finally block closes the cron session via end_session()
unconditionally. Before the fix it always passed end_reason="cron_complete",
so a FAILED cron run was stamped as if it had completed — masking the failure
and leaving the session indistinguishable from a healthy one. These tests
drive run_job down both the failure and success paths and assert the correct
end_reason is recorded by the SessionDB.
"""

from __future__ import annotations

import pytest

import cron.scheduler as cron_scheduler


class _RecordingSessionDB:
    """Captures end_session calls so the test can assert the end_reason."""

    def __init__(self):
        self.ended = []

    def set_session_title(self, *args, **kwargs):
        pass

    def end_session(self, session_id, end_reason):
        self.ended.append((session_id, end_reason))

    def close(self):
        pass


class _FailingCronAgent:
    """Agent whose run_conversation raises to force run_job's failure path."""

    def __init__(self, *args, **kwargs):
        self.kwargs = kwargs

    def run_conversation(self, prompt):
        raise RuntimeError("Connection error")

    def close(self):
        pass


class _SucceedingCronAgent:
    """Agent that returns a normal result to force run_job's success path."""

    def __init__(self, *args, **kwargs):
        self.kwargs = kwargs

    def run_conversation(self, prompt):
        return {
            "completed": True,
            "failed": False,
            "final_response": "ok",
            "turn_exit_reason": "",
        }

    def close(self):
        pass


def _install_patch_set(monkeypatch, tmp_path, session_db, fake_agent_cls):
    """Apply the exact patch set the isolation test uses to make run_job
    execute cleanly down both the success and failure paths."""
    monkeypatch.setattr("hermes_state.SessionDB", lambda: session_db)
    monkeypatch.setattr("run_agent.AIAgent", fake_agent_cls)
    monkeypatch.setattr(
        "hermes_constants.resolve_reasoning_config", lambda *_a, **_k: None
    )
    monkeypatch.setattr(
        "hermes_cli.runtime_provider.resolve_runtime_provider",
        lambda **_k: {
            "api_key": "test-key",
            "base_url": None,
            "provider": "test-provider",
            "api_mode": None,
            "command": None,
            "args": None,
        },
    )
    monkeypatch.setattr("tools.mcp_tool.discover_mcp_tools", lambda: [])
    monkeypatch.setattr(cron_scheduler, "_get_hermes_home", lambda: tmp_path)
    monkeypatch.setattr(cron_scheduler, "get_fallback_chain", lambda _cfg: [])
    monkeypatch.setattr(
        cron_scheduler, "_guard_job_credential_exfil", lambda _job: None
    )


_JOB = {
    "id": "end-reason",
    "name": "End Reason",
    "prompt": "Run the job",
    "schedule_display": "manual",
}


def test_run_job_failed_run_stamps_end_reason_cron_failed(monkeypatch, tmp_path):
    """A failing cron run must close its session with end_reason='cron_failed'."""
    session_db = _RecordingSessionDB()
    _install_patch_set(monkeypatch, tmp_path, session_db, _FailingCronAgent)

    success, _output, final_response, error = cron_scheduler.run_job(_JOB)

    assert success is False
    assert error == "RuntimeError: Connection error"
    # The session must have been closed exactly once, with the failure reason.
    assert len(session_db.ended) == 1
    session_id, end_reason = session_db.ended[0]
    assert end_reason == "cron_failed"
    assert end_reason != "cron_complete"


def test_run_job_successful_run_stamps_end_reason_cron_complete(
    monkeypatch, tmp_path
):
    """A successful cron run must still close with end_reason='cron_complete'
    (the fix must not regress the success path)."""
    session_db = _RecordingSessionDB()
    _install_patch_set(monkeypatch, tmp_path, session_db, _SucceedingCronAgent)

    success, _output, final_response, error = cron_scheduler.run_job(_JOB)

    assert success is True
    assert error is None
    assert final_response == "ok"
    assert len(session_db.ended) == 1
    session_id, end_reason = session_db.ended[0]
    assert end_reason == "cron_complete"
