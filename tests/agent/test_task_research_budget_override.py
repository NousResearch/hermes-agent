"""Agent binding tests for the per-Kanban-worker research policy envelope."""

from __future__ import annotations

import json
from types import SimpleNamespace

from agent.agent_init import _apply_display_config
from agent.tool_guardrails import RESEARCH_BUDGET_ENV


def _agent():
    return SimpleNamespace()


def test_task_override_wins_in_memory_without_mutating_profile_config(monkeypatch):
    profile_policy = {"web_search_max": 12, "browser_extract_max": 6}
    config = {"tool_loop_guardrails": {"research_budget": profile_policy}}
    monkeypatch.setenv(RESEARCH_BUDGET_ENV, json.dumps({"web_search_max": 2}))

    agent = _agent()
    _apply_display_config(agent, config, "cron")

    policy = agent._tool_guardrails.config.research_budget
    assert policy.web_search_max == 2
    assert policy.browser_extract_max is None
    assert config["tool_loop_guardrails"]["research_budget"] == profile_policy


def test_missing_or_invalid_task_override_preserves_profile_policy(monkeypatch):
    profile_policy = {"web_search_max": 12}
    config = {"tool_loop_guardrails": {"research_budget": profile_policy}}

    monkeypatch.delenv(RESEARCH_BUDGET_ENV, raising=False)
    agent = _agent()
    _apply_display_config(agent, config, "cron")
    assert agent._tool_guardrails.config.research_budget.web_search_max == 12

    monkeypatch.setenv(RESEARCH_BUDGET_ENV, "{not-json")
    invalid_agent = _agent()
    _apply_display_config(invalid_agent, config, "cron")
    assert invalid_agent._tool_guardrails.config.research_budget.web_search_max == 12
    assert config["tool_loop_guardrails"]["research_budget"] == profile_policy
