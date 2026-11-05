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
from types import SimpleNamespace

from gateway import session_cron

import pytest

import cron.scheduler as cron_scheduler
from gateway.session_context import reset_session_vars
from persistence.conversation.sessions import SessionLifecycleStatus


class _FakeCronAgent:
    def __init__(self, *args, **kwargs):
        pass

    def run_conversation(self, prompt, *, task_id):
        return {
            "completed": True,
            "failed": False,
            "final_response": "done",
            "turn_exit_reason": "",
        }

    def close(self):
        pass


class _RecordingStore:
    """Storage interface handle double with a configurable lifecycle classification."""

    def __init__(self, *args, **kwargs):
        self.ended: list[tuple[str, str]] = []
        self.lifecycle = type(self).next_lifecycle
        self.sessions = self.lineage = self

    next_lifecycle = "complete"

    def set_title(self, *args, **kwargs):
        return True

    def tip(self, session_id):
        return session_id

    def lifecycle_statuses(self, session_ids):
        if isinstance(type(self).next_lifecycle, Exception):
            raise type(self).next_lifecycle
        return {sid: SessionLifecycleStatus(type(self).next_lifecycle) for sid in session_ids}

    def end(self, session_id, reason):
        self.ended.append((session_id, reason))

    def close(self):
        pass


def _run_booked_job(monkeypatch, tmp_path):
    import run_agent

    from persistence_assembly import profile_scope

    db = _RecordingStore()
    owner = SimpleNamespace(persistence=SimpleNamespace(scope=profile_scope(home=tmp_path), ready=lambda: None))
    monkeypatch.setattr("persistence_assembly.open_persistence", lambda _scope, eager=False: db)
    monkeypatch.setattr(run_agent, "AIAgent", _FakeCronAgent)
    monkeypatch.setattr(
        "hermes_constants.resolve_reasoning_config", lambda *_a, **_k: None
    )
    # The runtime key is read from the environment (never a literal here);
    # AIAgent and the store are fakes above, so the value is never used.
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
    monkeypatch.setattr("tools.mcp_tool_discovery.discover_mcp_tools", list)
    monkeypatch.setattr(cron_scheduler, "_get_hermes_home", lambda: tmp_path)
    monkeypatch.setattr(cron_scheduler, "get_fallback_chain", lambda _cfg: [])
    monkeypatch.setattr(
        cron_scheduler, "_guard_job_credential_exfil", lambda _job: None
    )
    token = session_cron._execution.set((owner, "verify-session", "verify-complete", "verify-fire"))
    try:
        result = cron_scheduler.run_job(
            {"id": "verify-complete", "name": "Verification", "prompt": "Do the thing",
             "schedule_display": "manual"}, execution_id="verify-fire")
    finally:
        session_cron._execution.reset(token)
    assert result[0] and result[2] == "done", result
    assert db.ended[0][0] == "verify-session"
    return [db]


@pytest.fixture(autouse=True)
def _clean_state():
    reset_session_vars()
    _RecordingStore.next_lifecycle = "complete"
    yield
    reset_session_vars()


def test_run_without_final_assistant_message_books_incomplete(monkeypatch, tmp_path):
    """Last row a tool result / pending call (lifecycle 'interrupted') must
    not surface as a healthy complete run."""
    _RecordingStore.next_lifecycle = "interrupted"

    instances = _run_booked_job(monkeypatch, tmp_path)

    assert instances, "the store was never opened"
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
    _RecordingStore.next_lifecycle = RuntimeError("db busy")

    instances = _run_booked_job(monkeypatch, tmp_path)

    reasons = [reason for _sid, reason in instances[0].ended]
    assert reasons == ["cron_complete"]
