"""Revived from a 9 Sep autostash after the 2026-09-18 update (real incidents,
never landed on main):

1. ``job["no_fallback"]`` opts a cron job out of the fallback chain entirely
   (weekly-ops-review incident, 2026-08-23: the local fallback model made
   zero tool calls and fabricated a plausible-looking report instead of
   erroring — worse than an honest failure for a tool-dependent job).
2. A raw "You've hit your session/usage limit" body delivered with HTTP 200
   (no real ``failed``/429 signal) must not be delivered verbatim as if it
   were the agent's real answer (2026-08-09, 2026-08-30 weekly-ops-review
   runs shipped exactly this to Phil).
"""

from __future__ import annotations

import pytest

from cron import scheduler
from cron.scheduler import _CronAgentSetup, _final_response_from_result


class _AIAgent:
    pass


def test_no_fallback_true_clears_the_fallback_chain(monkeypatch):
    monkeypatch.setattr(scheduler, "get_fallback_chain", lambda cfg: ["local-model"])
    monkeypatch.setattr(scheduler, "_load_prefill_messages", lambda cfg, job_id: None)
    monkeypatch.setattr(scheduler, "_guard_job_credential_exfil", lambda job: None)
    monkeypatch.setattr(scheduler, "_preflight_or_block", lambda job, jid, jname, cfg: None)
    monkeypatch.setattr(scheduler, "_resolve_job_runtime", lambda job, jid, jc: ({}, "gpt-5"))
    monkeypatch.setattr(scheduler, "_resolve_job_reasoning_config", lambda job, cfg, model: None)
    monkeypatch.setattr(scheduler, "_load_credential_pool", lambda runtime, jid: None)
    monkeypatch.setattr(scheduler, "_init_cron_mcp_tools", lambda jid: None)
    monkeypatch.setattr(scheduler, "_cron_preflight_enabled", lambda cfg: False)

    class _JobConfig:
        cfg = {}
        model = "gpt-5"

    setup = scheduler._resolve_cron_agent_setup(
        {"id": "weekly-ops-review", "no_fallback": True}, "weekly-ops-review", "Weekly Ops Review", _JobConfig(),
    )
    assert setup.fallback_model is None


def test_no_fallback_unset_preserves_the_fallback_chain(monkeypatch):
    monkeypatch.setattr(scheduler, "get_fallback_chain", lambda cfg: ["local-model"])
    monkeypatch.setattr(scheduler, "_load_prefill_messages", lambda cfg, job_id: None)
    monkeypatch.setattr(scheduler, "_guard_job_credential_exfil", lambda job: None)
    monkeypatch.setattr(scheduler, "_preflight_or_block", lambda job, jid, jname, cfg: None)
    monkeypatch.setattr(scheduler, "_resolve_job_runtime", lambda job, jid, jc: ({}, "gpt-5"))
    monkeypatch.setattr(scheduler, "_resolve_job_reasoning_config", lambda job, cfg, model: None)
    monkeypatch.setattr(scheduler, "_load_credential_pool", lambda runtime, jid: None)
    monkeypatch.setattr(scheduler, "_init_cron_mcp_tools", lambda jid: None)
    monkeypatch.setattr(scheduler, "_cron_preflight_enabled", lambda cfg: False)

    class _JobConfig:
        cfg = {}
        model = "gpt-5"

    setup = scheduler._resolve_cron_agent_setup(
        {"id": "some-other-job"}, "some-other-job", "Some Other Job", _JobConfig(),
    )
    assert setup.fallback_model == ["local-model"]


def test_bare_session_limit_200_body_raises_instead_of_delivering():
    result = {
        "final_response": "You've hit your session limit · resets 9:30pm (Europe/London)",
        "failed": False, "completed": True, "turn_exit_reason": "done",
        "messages": [], "api_calls": 1,
    }
    with pytest.raises(RuntimeError, match="session limit"):
        _final_response_from_result(result, "weekly-ops-review", "Weekly Ops Review", _AIAgent)


def test_bare_usage_limit_200_body_also_raises():
    result = {
        "final_response": "You've hit your weekly usage limit, resets Monday",
        "failed": False, "completed": True, "turn_exit_reason": "done",
        "messages": [], "api_calls": 1,
    }
    with pytest.raises(RuntimeError):
        _final_response_from_result(result, "weekly-ops-review", "Weekly Ops Review", _AIAgent)


def test_a_genuine_answer_that_merely_mentions_limits_is_delivered_normally():
    result = {
        "final_response": "Reminder: your API plan's usage limit resets Monday, but here's this week's report...",
        "failed": False, "completed": True, "turn_exit_reason": "done",
        "messages": [], "api_calls": 3,
    }
    # Doesn't match the anchored "you've hit your ... limit" signature, so it's a real answer.
    assert "this week's report" in _final_response_from_result(
        result, "weekly-ops-review", "Weekly Ops Review", _AIAgent,
    )
