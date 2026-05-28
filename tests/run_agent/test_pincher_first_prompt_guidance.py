"""Regression tests for Pincher-first prompt guidance injection."""

from unittest.mock import MagicMock, patch

from run_agent import AIAgent


def _make_tool_defs(*names: str) -> list[dict]:
    """Build minimal tool definitions accepted by AIAgent.__init__."""
    return [
        {
            "type": "function",
            "function": {
                "name": name,
                "description": f"{name} tool",
                "parameters": {"type": "object", "properties": {}},
            },
        }
        for name in names
    ]


def _make_agent_with_tools(*tool_names: str) -> AIAgent:
    with (
        patch("run_agent.get_tool_definitions", return_value=_make_tool_defs(*tool_names)),
        patch("run_agent.check_toolset_requirements", return_value={}),
        patch("run_agent.OpenAI"),
    ):
        agent = AIAgent(
            api_key="test-key-1234567890",
            base_url="https://openrouter.ai/api/v1",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
        )
        agent.client = MagicMock()
        return agent


def test_pincher_first_guidance_injected_when_pincher_mcp_tools_loaded():
    agent = _make_agent_with_tools(
        "mcp_pincher_search",
        "mcp_pincher_context",
        "mcp_pincher_trace",
        "mcp_pincher_changes",
        "read_file",
    )

    prompt = agent._build_system_prompt()

    assert "mcp_pincher_search" in prompt
    assert "mcp_pincher_context" in prompt
    assert "mcp_pincher_trace" in prompt
    assert "mcp_pincher_changes" in prompt
    assert "Do NOT call Read / Grep / Glob for code navigation until Pincher returns no result" in prompt


def test_pincher_first_guidance_not_injected_without_pincher_mcp_tools():
    agent = _make_agent_with_tools("read_file", "search_files", "terminal")

    prompt = agent._build_system_prompt()

    assert "mcp_pincher_search" not in prompt
    assert "Do NOT call Read / Grep / Glob for code navigation until Pincher returns no result" not in prompt
