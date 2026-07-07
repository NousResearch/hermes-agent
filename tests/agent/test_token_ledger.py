"""Tests for operational token budget attribution."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from agent.token_ledger import compute_token_ledger, find_budget_violations


def _make_agent(
    *,
    stable: str = "identity and guidance",
    context: str = "# Project Context\nFollow AGENTS.md",
    volatile: str = "runtime footer",
    memory: str = "MEMORY: durable note",
    user_profile: str = "USER: prefers concise receipts",
    tools: list | None = None,
    context_length: int = 200_000,
):
    agent = MagicMock()
    agent.model = "openai/gpt-5.5"
    agent.tools = tools or [
        {"type": "function", "function": {"name": "terminal", "description": "run shell commands"}},
        {"type": "function", "function": {"name": "mcp_demo_lookup", "description": "mcp lookup"}},
        {"type": "function", "function": {"name": "delegate_task", "description": "spawn subagent"}},
    ]
    store = MagicMock()
    store.format_for_system_prompt.side_effect = lambda target: memory if target == "memory" else user_profile
    agent._memory_store = store
    agent._memory_enabled = True
    agent._user_profile_enabled = True
    agent.context_compressor = MagicMock(context_length=context_length, last_prompt_tokens=0)
    parts = {"stable": stable, "context": context, "volatile": volatile}
    return agent, parts


def test_token_ledger_sources_sum_to_estimated_total_without_unknown_bucket():
    stable = (
        "base guidance\n"
        "<available_skills>\n  coding:\n    - test-driven-development: TDD\n</available_skills>"
    )
    messages = [
        {"role": "user", "content": "Please inspect the failing test"},
        {"role": "assistant", "content": "I will run pytest"},
        {"role": "tool", "tool_call_id": "call_1", "content": "pytest output: 1 failed, 2 passed"},
    ]
    agent, parts = _make_agent(stable=stable)

    with patch("agent.system_prompt.build_system_prompt_parts", return_value=parts):
        ledger = compute_token_ledger(agent, messages)

    assert ledger.estimated_total == sum(segment.token_count for segment in ledger.segments)
    assert ledger.by_source["system_prompt"] > 0
    assert ledger.by_source["skills"] > 0
    assert ledger.by_source["rules"] > 0
    assert ledger.by_source["memory"] > 0
    assert ledger.by_source["tool_definitions"] > 0
    assert ledger.by_source["mcp"] > 0
    assert ledger.by_source["subagent_definitions"] > 0
    assert ledger.by_source["conversation"] > 0
    assert ledger.by_source.get("unknown", 0) == 0


def test_token_ledger_marks_stable_prefix_and_prunable_segments():
    messages = [{"role": "user", "content": "latest active task"}]
    agent, parts = _make_agent()

    with patch("agent.system_prompt.build_system_prompt_parts", return_value=parts):
        ledger = compute_token_ledger(agent, messages)

    by_source = {segment.source: segment for segment in ledger.segments}
    for source in {"system_prompt", "rules", "memory", "tool_definitions"}:
        assert by_source[source].stable_prefix is True
        assert by_source[source].prunable is False
    assert by_source["conversation"].stable_prefix is False
    assert by_source["conversation"].prunable is True


def test_budget_violations_report_source_overflow_below_global_threshold():
    messages = [{"role": "tool", "tool_call_id": "call_big", "content": "x" * 1200}]
    agent, parts = _make_agent(context_length=100_000)

    with patch("agent.system_prompt.build_system_prompt_parts", return_value=parts):
        ledger = compute_token_ledger(agent, messages)

    assert ledger.estimated_total < 50_000  # below the normal 50% compression threshold
    violations = find_budget_violations(ledger, {"conversation": 100})

    assert [v.source for v in violations] == ["conversation"]
    assert violations[0].tokens == ledger.by_source["conversation"]
    assert violations[0].cap == 100
    assert violations[0].excess == ledger.by_source["conversation"] - 100
