"""The cron fire audit row carries what a provider/cost investigation needs.

Per-fire ``usage_audit.jsonl`` records gain four additive keys:
``estimated_cost_usd`` / ``cost_status`` / ``cost_source`` (the agent already
folds per-call cost deltas and status/source onto the session) and
``provider_response_ids`` (the per-call ``response.id`` list, deduped — the
handle a provider support request or aggregation-diagnosis turns on).

Contract pinned here:
- a fire whose turn result carries the fields writes them into the row;
- a fire with no cost (usage absent) writes ``null`` + empty list — the row
  must still appear (script-only/no-LLM fires keep writing rows);
- existing keys are unchanged, so additive-contract consumers keep parsing.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from cron import scheduler


@pytest.fixture
def tmp_hermes_home(tmp_path, monkeypatch):
    """Redirect _get_hermes_home() so the audit logger writes under tmp_path."""
    fake_home = tmp_path / "home" / ".hermes"
    fake_home.mkdir(parents=True)
    monkeypatch.setattr(scheduler, "_get_hermes_home", lambda: fake_home)
    return fake_home


def test_fire_audit_row_carries_cost_and_ids(tmp_hermes_home):
    audit = scheduler._FireAudit({"deliver": "telegram"}, "job-1", "m/x")
    result = {
        "prompt_tokens": 100,
        "completion_tokens": 7,
        "total_tokens": 107,
        "response_silent": False,
        "estimated_cost_usd": 0.008306,
        "cost_status": "ok",
        "cost_source": "estimator",
        "provider_response_ids": ["gen-1789466448-a", "gen-1789466448-b"],
    }
    with patch.object(scheduler.time, "monotonic", side_effect=[0.0, 1.5]):
        audit.write(result, None)

    path = scheduler._usage_audit_path()
    row = json.loads(path.read_text(encoding="utf-8").splitlines()[-1])
    assert row["job_id"] == "job-1"
    assert row["estimated_cost_usd"] == 0.008306
    assert row["cost_status"] == "ok"
    assert row["cost_source"] == "estimator"
    assert row["provider_response_ids"] == ["gen-1789466448-a", "gen-1789466448-b"]
    # pre-existing keys unchanged (additive contract)
    assert row["prompt_tokens"] == 100 and row["total_tokens"] == 107


def test_fire_audit_row_without_cost_still_writes_null_fields(tmp_hermes_home):
    """A fire with no LLM usage (script-only) or a failed fire writes the new keys
    as null/empty instead of being dropped — rows must stay schema-stable."""
    audit = scheduler._FireAudit({"deliver": None}, "job-2", None)
    audit.write({}, "TimeoutError: idle")

    row = json.loads(scheduler._usage_audit_path().read_text(encoding="utf-8").splitlines()[-1])
    assert row["estimated_cost_usd"] is None
    assert row["cost_status"] is None
    assert row["cost_source"] is None
    assert row["provider_response_ids"] == []
    assert row["error"] == "TimeoutError: idle"


def test_finalize_turn_exposes_provider_response_ids():
    """mod-11 half: the turn result is the transport the fire audit reads — a
    finalized turn must expose the agent's captured response ids."""
    import sys
    from types import SimpleNamespace

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "agent"))
    from tests.agent.test_turn_finalizer_cleanup_guard import _StubAgent, _run

    agent = _StubAgent(raise_in=())
    agent.session_provider_response_ids = ["gen-bite-10", "gen-bite-11"]
    result = _run(agent)
    assert result["provider_response_ids"] == ["gen-bite-10", "gen-bite-11"]


def test_finalize_turn_absent_ids_default_to_empty():
    """Agents that never captured an id (older flow, or no LLM call) expose [] —
    the key must exist so consumers don't need getattr guards."""
    from tests.agent.test_turn_finalizer_cleanup_guard import _StubAgent, _run

    agent = _StubAgent(raise_in=())
    result = _run(agent)
    assert result["provider_response_ids"] == []


def test_record_response_usage_captures_response_id(tmp_path, monkeypatch, caplog):
    """mod-10 half: record_response_usage appends the response id to the agent's
    deduped session list (and omits nothing when the id repeats across calls)."""
    from types import SimpleNamespace

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    from run_agent import AIAgent

    agent = AIAgent(api_key="k", base_url="https://inference-api.nousresearch.com/v1",
                    provider="nous", api_mode="chat_completions",
                    model="anthropic/claude-fable-5.1", session_id="t", platform="cli",
                    quiet_mode=True, skip_context_files=True, skip_memory=True,
                    save_trajectories=False, enabled_toolsets=["file"])
    try:
        usage = SimpleNamespace(prompt_tokens=100, completion_tokens=7, total_tokens=107,
                                prompt_tokens_details=SimpleNamespace(cached_tokens=0, cache_write_tokens=0),
                                completion_tokens_details=None)
        resp = SimpleNamespace(usage=usage, id="gen-bite-9", provider=None,
                               model="anthropic/claude-fable-5.1")
        from agent import turn_usage
        turn_usage.record_response_usage(agent, resp, messages=[{"role": "user", "content": "hi"}],
                                         api_call_count=1, api_duration=0.2,
                                         compression_attempts=0, max_compression_attempts=3)
        turn_usage.record_response_usage(agent, resp, messages=[{"role": "user", "content": "hi"}],
                                         api_call_count=2, api_duration=0.2,
                                         compression_attempts=0, max_compression_attempts=3)
        assert agent.session_provider_response_ids == ["gen-bite-9"]
    finally:
        agent.close()
