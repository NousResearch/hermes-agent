"""Test that OAuth Anthropic path does NOT add mcp_ prefix to tool names.

Regression test for issue #28849: the mcp_ prefix on all tools triggered
Anthropic's overage gate for Pro/Max subscribers, causing HTTP 400 errors
on every tool-bearing request.
"""
import pytest


def _make_tools(names):
    """Create minimal tool schemas in OpenAI format."""
    return [
        {"type": "function", "function": {
            "name": n, "description": f"tool {n}",
            "parameters": {"type": "object", "properties": {}}
        }}
        for n in names
    ]


def _make_tool_use_history(tool_name):
    """Create a message history with a tool_use block."""
    return [
        {"role": "user", "content": "test"},
        {"role": "assistant", "content": [
            {"type": "tool_use", "id": "tu_1", "name": tool_name,
             "input": {}},
        ]},
    ]


class TestNoMcpPrefixOnOAuth:
    """The mcp_ prefix must NOT be added to built-in tool names."""

    def test_tool_names_not_prefixed_oauth(self):
        """On the OAuth path, tool names should keep their original names."""
        from agent.anthropic_adapter import build_anthropic_kwargs

        tools = _make_tools(["terminal", "read_file", "web_search"])
        kwargs = build_anthropic_kwargs(
            model="claude-sonnet-4-5",
            messages=[{"role": "user", "content": "hi"}],
            tools=tools,
            max_tokens=1024,
            reasoning_config=None,
            is_oauth=True,
        )
        result_names = [t["name"] for t in kwargs["tools"]]
        assert result_names == ["terminal", "read_file", "web_search"]

    def test_tool_names_not_prefixed_non_oauth(self):
        """On the non-OAuth path, tool names should also keep original names."""
        from agent.anthropic_adapter import build_anthropic_kwargs

        tools = _make_tools(["terminal", "execute_code"])
        kwargs = build_anthropic_kwargs(
            model="claude-sonnet-4-5",
            messages=[{"role": "user", "content": "hi"}],
            tools=tools,
            max_tokens=1024,
            reasoning_config=None,
            is_oauth=False,
        )
        result_names = [t["name"] for t in kwargs["tools"]]
        assert result_names == ["terminal", "execute_code"]

    def test_history_tool_use_names_not_prefixed(self):
        """Tool names in message history should not be mangled with mcp_."""
        from agent.anthropic_adapter import build_anthropic_kwargs

        messages = _make_tool_use_history("terminal")
        kwargs = build_anthropic_kwargs(
            model="claude-sonnet-4-5",
            messages=messages,
            tools=_make_tools(["terminal"]),
            max_tokens=1024,
            reasoning_config=None,
            is_oauth=True,
        )
        # The tool_use block in the converted messages should keep "terminal"
        assistant_blocks = [
            b for m in kwargs["messages"]
            if isinstance(m.get("content"), list)
            for b in m["content"]
            if isinstance(b, dict) and b.get("type") == "tool_use"
        ]
        for block in assistant_blocks:
            assert not block["name"].startswith("mcp_"), \
                f"tool_use name '{block['name']}' should not have mcp_ prefix"

    def test_no_tools_no_crash(self):
        """OAuth path should work fine with no tools at all."""
        from agent.anthropic_adapter import build_anthropic_kwargs

        kwargs = build_anthropic_kwargs(
            model="claude-sonnet-4-5",
            messages=[{"role": "user", "content": "hi"}],
            tools=None,
            max_tokens=1024,
            reasoning_config=None,
            is_oauth=True,
        )
        assert "tools" not in kwargs or kwargs.get("tools") is None
