"""Tests for stripping tools from fallback provider requests.

When the agent falls back to a secondary provider, tool definitions are
stripped from the API kwargs to avoid 400 errors from models with smaller
context windows (e.g. Grok on OpenRouter).
"""

from unittest.mock import MagicMock, patch

from run_agent import AIAgent


def _make_tool_defs(*names: str) -> list:
    return [
        {
            "type": "function",
            "function": {
                "name": n,
                "description": f"{n} tool",
                "parameters": {"type": "object", "properties": {}},
            },
        }
        for n in names
    ]


def _make_agent(tools=None):
    """Create a minimal AIAgent with tools loaded."""
    tool_defs = tools or _make_tool_defs("web_search", "read_file", "edit_file")
    with (
        patch("run_agent.get_tool_definitions", return_value=tool_defs),
        patch("run_agent.check_toolset_requirements", return_value={}),
        patch("run_agent.OpenAI"),
    ):
        agent = AIAgent(
            api_key="test-key",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
        )
        agent.client = MagicMock()
        return agent


class TestStripToolsOnFallback:
    def test_tools_present_on_primary(self):
        agent = _make_agent()
        assert agent._fallback_activated is False
        kwargs = agent._build_api_kwargs([{"role": "user", "content": "hi"}])
        assert "tools" in kwargs
        assert len(kwargs["tools"]) == 3

    def test_tools_stripped_on_fallback(self):
        agent = _make_agent()
        agent._fallback_activated = True
        kwargs = agent._build_api_kwargs([{"role": "user", "content": "hi"}])
        assert "tools" not in kwargs
        assert "tool_choice" not in kwargs

    def test_no_tools_no_error_on_fallback(self):
        agent = _make_agent(tools=[])
        agent._fallback_activated = True
        kwargs = agent._build_api_kwargs([{"role": "user", "content": "hi"}])
        assert "tools" not in kwargs
