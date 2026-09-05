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
