"""Tests for live session context breakdown."""

from unittest.mock import MagicMock, patch

from agent.context_breakdown import compute_session_context_breakdown


def _make_agent(
    *,
    stable: str = "identity and guidance",
    context: str = "",
    volatile: str = "timestamp line",
    tools: list | None = None,
    context_length: int = 200_000,
    last_prompt_tokens: int = 0,
):
    agent = MagicMock()
    agent.model = "openai/gpt-5.4"
    agent.tools = tools or [
        {"type": "function", "function": {"name": "terminal", "description": "run"}},
        {"type": "function", "function": {"name": "mcp_demo_tool", "description": "mcp"}},
        {"type": "function", "function": {"name": "delegate_task", "description": "spawn"}},
    ]
    agent._memory_store = None
    agent._memory_enabled = True
    agent._user_profile_enabled = True
    agent.context_compressor = MagicMock(
        context_length=context_length,
        last_prompt_tokens=last_prompt_tokens,
    )
    return agent, {"stable": stable, "context": context, "volatile": volatile}


def test_breakdown_includes_major_categories():
    stable = (
        "base guidance\n"
        "<available_skills>\n  demo:\n    - hello: hi\n</available_skills>"
    )
    context = "# Project Context\nFollow AGENTS.md"
    volatile = "Current time: now"
    history = [{"role": "user", "content": "hello there"}]
    agent, parts = _make_agent(stable=stable, context=context, volatile=volatile)

    with patch("agent.system_prompt.build_system_prompt_parts", return_value=parts):
        data = compute_session_context_breakdown(agent, history)

    ids = {item["id"] for item in data["categories"]}
    assert {"system_prompt", "tool_definitions", "rules", "skills", "mcp", "subagent_definitions", "conversation"} <= ids
    assert data["context_max"] == 200_000
    assert data["estimated_total"] > 0


def test_breakdown_uses_measured_context_when_available():
    agent, parts = _make_agent(last_prompt_tokens=42_000)

    with patch("agent.system_prompt.build_system_prompt_parts", return_value=parts):
        data = compute_session_context_breakdown(agent, [])

    assert data["context_used"] == 42_000
    assert data["context_percent"] == 21


def test_breakdown_scales_category_rows_to_measured_total():
    agent, parts = _make_agent(
        stable="system prompt " * 100,
        context="rules " * 100,
        volatile="volatile " * 100,
        tools=[{"type": "function", "function": {"name": "terminal", "description": "run " * 100}}],
        last_prompt_tokens=42_000,
    )
    history = [{"role": "user", "content": "conversation " * 1000}]

    with patch("agent.system_prompt.build_system_prompt_parts", return_value=parts):
        data = compute_session_context_breakdown(agent, history)

    assert data["context_used"] == 42_000
    assert data["estimated_total"] != 42_000
    assert sum(item["tokens"] for item in data["categories"]) == data["context_used"]
    assert all(item["tokens"] <= data["context_used"] for item in data["categories"])
