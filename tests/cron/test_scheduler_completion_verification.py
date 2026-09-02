"""Fail-closed completion booking for cron runs (#93820).

The scheduler booked every finished run as ``cron_complete`` based on the run
lifecycle alone: a job whose agent turn died after a tool call, mid-API-wait,
or without any assistant text still surfaced as a healthy run (one audited
day held 10 such silently-failed sessions). The fix classifies the session's
LAST message row through the existing ``session_lifecycle_statuses`` helper
before ``end_session``: only a real assistant reply — a plain answer or the
``[SILENT]`` sentinel, both assistant-text rows — books as ``cron_complete``;
anything else books as ``cron_incomplete_no_output``. Classification is
best-effort: a probe failure keeps the historical reason rather than
mislabeling a healthy run.
"""

import os

import pytest

import cron.scheduler as cron_scheduler
from gateway.session_context import reset_session_vars


class _FakeCronAgent:
    def __init__(self, *args, **kwargs):
        pass

    def run_conversation(self, prompt):
        return {
            "completed": True,
            "failed": False,
            "final_response": "done",
            "turn_exit_reason": "",
        }

    def close(self):
        pass


class _RecordingSessionDB:
    """SessionDB double with a configurable lifecycle classification."""

    def __init__(self, *args, **kwargs):
        self.ended: list[tuple[str, str]] = []
        self.lifecycle = type(self).next_lifecycle

    next_lifecycle = "complete"

    def set_session_title(self, *args, **kwargs):
        return True

    def get_compression_tip(self, session_id):
        return None

    def session_lifecycle_statuses(self, session_ids):
        if isinstance(type(self).next_lifecycle, Exception):
            raise type(self).next_lifecycle
        return {sid: type(self).next_lifecycle for sid in session_ids}

    def end_session(self, session_id, reason):
        self.ended.append((session_id, reason))

    def close(self):
        pass


def _run_booked_job(monkeypatch, tmp_path):
    import hermes_state
    import run_agent

    instances: list[_RecordingSessionDB] = []
    real_init = _RecordingSessionDB.__init__

    def _capture_init(self, *args, **kwargs):
        real_init(self, *args, **kwargs)
        instances.append(self)

    monkeypatch.setattr(_RecordingSessionDB, "__init__", _capture_init)
    monkeypatch.setattr(hermes_state, "SessionDB", _RecordingSessionDB)
    monkeypatch.setattr(run_agent, "AIAgent", _FakeCronAgent)
    monkeypatch.setattr(
        "hermes_constants.resolve_reasoning_config", lambda *_a, **_k: None
    )
    # The runtime key is read from the environment (never a literal here);
    # AIAgent and SessionDB are fakes above, so the value is never used.
    monkeypatch.setenv("HERMES_TEST_RUNTIME_KEY", "unused-placeholder")

    def _fake_runtime(**_kwargs):
        return {
            "api_key": os.environ.get("HERMES_TEST_RUNTIME_KEY", ""),
            "base_url": None,
            "provider": "test-provider",
            "api_mode": None,
            "command": None,
            "args": None,
        }

    monkeypatch.setattr(
        "hermes_cli.runtime_provider.resolve_runtime_provider", _fake_runtime
    )
    monkeypatch.setattr("tools.mcp_tool.discover_mcp_tools", lambda: [])
    monkeypatch.setattr(cron_scheduler, "_get_hermes_home", lambda: tmp_path)
    monkeypatch.setattr(cron_scheduler, "get_fallback_chain", lambda _cfg: [])
    monkeypatch.setattr(
        cron_scheduler, "_guard_job_credential_exfil", lambda _job: None
    )
    cron_scheduler.run_job(
        {
            "id": "verify-complete",
            "name": "Verification",
            "prompt": "Do the thing",
            "schedule_display": "manual",
        }
    )
    return instances


@pytest.fixture(autouse=True)
def _clean_state():
    reset_session_vars()
    _RecordingSessionDB.next_lifecycle = "complete"
    yield
    reset_session_vars()


def test_run_without_final_assistant_message_books_incomplete(monkeypatch, tmp_path):
    """Last row a tool result / pending call (lifecycle 'interrupted') must
    not surface as a healthy complete run."""
    _RecordingSessionDB.next_lifecycle = "interrupted"

    instances = _run_booked_job(monkeypatch, tmp_path)

    assert instances, "SessionDB was never constructed"
    reasons = [reason for _sid, reason in instances[0].ended]
    assert reasons == ["cron_incomplete_no_output"]


def test_run_with_final_assistant_reply_books_complete(monkeypatch, tmp_path):
    """A real assistant reply (plain answer or [SILENT] — both assistant
    text rows) keeps the healthy booking."""
    instances = _run_booked_job(monkeypatch, tmp_path)

    reasons = [reason for _sid, reason in instances[0].ended]
    assert reasons == ["cron_complete"]


def test_classification_probe_failure_keeps_historical_reason(monkeypatch, tmp_path):
    """Best-effort metadata: a failing classifier must not mislabel a run."""
    _RecordingSessionDB.next_lifecycle = RuntimeError("db busy")

    instances = _run_booked_job(monkeypatch, tmp_path)

    reasons = [reason for _sid, reason in instances[0].ended]
    assert reasons == ["cron_complete"]
