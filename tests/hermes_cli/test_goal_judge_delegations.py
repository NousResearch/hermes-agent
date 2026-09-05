"""The goal judge knows about delegated subagents.

In a fan-out run 4 of 5 /goal nudges fired 6-153 s after a turn that had said "waiting on workers,
nothing to dispatch": the judge prompt had no WAIT branch for delegated subagents (only for registry
processes), so it returned CONTINUE and each nudge bought a status recap (19 API calls, ~$4.9).
"""
from types import SimpleNamespace
from unittest.mock import patch

from hermes_cli import goals


def _resp(content):
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))])


def test_judge_prompt_states_active_delegations_and_a_wait_branch_for_them():
    seen = {}

    def fake_call_llm(*a, **kw):
        seen["prompt"] = kw.get("messages") or a
        return _resp('{"verdict": "wait", "wait_for_seconds": 900, "reason": "waiting on workers"}')

    with patch("agent.auxiliary_client.call_llm", side_effect=fake_call_llm):
        verdict, _reason, parse_failed, directive, _transport = goals.judge_goal(
            "refactor everything", "Waiting on 4 workers; nothing to dispatch.", active_delegations=4)
    text = str(seen["prompt"])
    assert "Active delegations: the agent has 4 delegated subagent batch(es) still running" in text
    assert "delegated subagents still running" in goals.JUDGE_SYSTEM_PROMPT
    assert (verdict, parse_failed) == ("wait", False) and directive.get("seconds") == 900

    with patch("agent.auxiliary_client.call_llm", side_effect=fake_call_llm):
        goals.judge_goal("g", "r", active_delegations=0)
    assert "Active delegations" not in str(seen["prompt"])


def test_count_active_delegations_is_scoped_to_the_spawning_session():
    from tools import async_delegation as ad

    fake = {
        "a": {"status": "running", "parent_session_id": "root", "session_key": "", "origin_ui_session_id": ""},
        "b": {"status": "completed", "parent_session_id": "root", "session_key": "", "origin_ui_session_id": ""},
        "c": {"status": "running", "parent_session_id": "other", "session_key": "", "origin_ui_session_id": ""},
    }
    with patch.object(ad, "_records", fake):
        assert goals.count_active_delegations("root") == 1
        assert goals.count_active_delegations("other") == 1
        assert goals.count_active_delegations(None) == 0


def test_a_delegation_wait_lifts_when_a_batch_returns_not_only_when_the_timer_runs_out(tmp_path, monkeypatch):
    """Independent-review witness: results arrived with 1,199 s left on a 1,200 s WAIT and nothing
    re-judged; integration sat unfinished until the timer ran out."""
    from pathlib import Path
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes")); (tmp_path / ".hermes").mkdir()
    goals._DB_CACHE.clear()
    mgr = goals.GoalManager(session_id="root-wait")
    mgr.set("integrate the rounds")
    with patch.object(goals, "count_active_delegations", return_value=4):
        mgr.wait_for_seconds(1200, reason="4 batches running", on_delegations=4)
        assert mgr.is_waiting() is True                     # all four still live: parked
    with patch.object(goals, "count_active_delegations", return_value=3):
        assert mgr.is_waiting() is False                    # one returned: barrier lifted early
    assert mgr.state.waiting_until == 0.0 and mgr.state.waiting_on_delegations == 0
    # a plain timed wait (no delegations) is unaffected by the delegation count
    mgr.wait_for_seconds(1200, reason="cooldown")
    with patch.object(goals, "count_active_delegations", return_value=0):
        assert mgr.is_waiting() is True
    goals._DB_CACHE.clear()
