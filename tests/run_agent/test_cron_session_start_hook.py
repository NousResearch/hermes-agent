from types import SimpleNamespace

import pytest

from agent.conversation_loop import _restore_or_build_system_prompt


def test_cron_session_start_can_block_before_runtime(monkeypatch):
    captured = {}

    def invoke(name, **kwargs):
        captured.update(kwargs)
        return [{"action": "block", "reason": "registration failed"}]

    monkeypatch.setattr("hermes_cli.lifecycle.invoke_hook", invoke)
    monkeypatch.setattr(
        "agent.credits_tracker.seed_credits_at_session_start", lambda _agent: None
    )
    agent = SimpleNamespace(
        _session_db=None,
        _cached_system_prompt=None,
        _build_system_prompt=lambda _message: "prompt",
        session_id="cron-session",
        model="test-model",
        platform="cron",
        cron_job_id="job-1",
        cron_job_name="triage",
        cron_max_turns=12,
    )

    with pytest.raises(RuntimeError, match="registration failed"):
        _restore_or_build_system_prompt(agent, None, [])

    assert captured["session_id"] == "cron-session"
    assert captured["cron_job_id"] == "job-1"
    assert captured["cron_max_turns"] == 12
