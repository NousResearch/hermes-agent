"""Tests for GH-25255: Anthropic OAuth mcp__ prefix stripping.

When strip_tool_prefix=True (Anthropic OAuth path), the transport must only
strip the ``mcp__`` prefix from OAuth-injected tools, NOT from Hermes-native
MCP server tools that are registered under their full ``mcp_<server>_<tool>``
name in the tool registry.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch



# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_tool_use_block(name: str, block_id: str = "tc_1", input_data: dict | None = None):
    """Create a fake Anthropic tool_use content block."""
    return SimpleNamespace(
        type="tool_use",
        id=block_id,
        name=name,
        input=input_data or {"query": "test"},
    )


def _make_response(*blocks, stop_reason="end_turn"):
    """Create a fake Anthropic Messages response."""
    return SimpleNamespace(
        content=list(blocks),
        stop_reason=stop_reason,
        model="claude-sonnet-4",
        usage=SimpleNamespace(input_tokens=100, output_tokens=50),
    )


class _FakeRegistry:
    """Minimal fake tool registry for testing prefix stripping logic."""

    def __init__(self, registered_names: set[str]):
        self._names = registered_names

    def get_entry(self, name: str):
        if name in self._names:
            return SimpleNamespace(name=name)  # truthy = tool exists
        return None


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestAnthropicMcpPrefixStrip:
    """Verify that strip_tool_prefix only strips OAuth-injected prefixes."""

    def _get_transport(self):
        from agent.transports.anthropic import AnthropicTransport
        return AnthropicTransport()

    def test_strips_prefix_for_oauth_injected_tool(self):
        """OAuth tools: mcp__read_file -> read_file (stripped).

        The tool was registered as 'read_file' in the registry.
        Anthropic sees 'mcp__read_file' because Hermes adds the prefix.
        On response, we must strip it back to 'read_file'.
        """
        transport = self._get_transport()
        block = _make_tool_use_block("mcp__read_file")
        response = _make_response(block)

        registry = _FakeRegistry({"read_file", "terminal", "web_search"})
        with patch("tools.registry.registry", registry):
            result = transport.normalize_response(response, strip_tool_prefix=True)

        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].name == "read_file"

    def test_preserves_native_mcp_server_tool_name(self):
        """Native MCP tools: mcp_composio_SEARCH -> mcp_composio_SEARCH (kept).

        The tool is registered with the full mcp_ prefix in the registry.
        Stripping would break registry lookup.
        """
        transport = self._get_transport()
        block = _make_tool_use_block("mcp_composio_COMPOSIO_SEARCH_TOOLS")
        response = _make_response(block)

        registry = _FakeRegistry({
            "mcp_composio_COMPOSIO_SEARCH_TOOLS",
            "mcp_composio_COMPOSIO_GET_TOOL_SCHEMAS",
            "read_file",
        })
        with patch("tools.registry.registry", registry):
            result = transport.normalize_response(response, strip_tool_prefix=True)

        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].name == "mcp_composio_COMPOSIO_SEARCH_TOOLS"

    def test_no_strip_when_flag_false(self):
        """When strip_tool_prefix=False, names are never modified."""
        transport = self._get_transport()
        block = _make_tool_use_block("mcp__read_file")
        response = _make_response(block)

        registry = _FakeRegistry({"read_file"})
        with patch("tools.registry.registry", registry):
            result = transport.normalize_response(response, strip_tool_prefix=False)

        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].name == "mcp__read_file"

    def test_no_strip_when_not_mcp_prefixed(self):
        """Non-mcp__ names are untouched regardless of strip flag."""
        transport = self._get_transport()
        block = _make_tool_use_block("web_search")
        response = _make_response(block)

        registry = _FakeRegistry({"web_search"})
        with patch("tools.registry.registry", registry):
            result = transport.normalize_response(response, strip_tool_prefix=True)

        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].name == "web_search"

    def test_preserves_name_when_neither_in_registry(self):
        """When neither stripped nor full name is in registry, keep full name.

        Safety fallback: if we can't determine the type, prefer the full name
        since it's what the LLM was told about.
        """
        transport = self._get_transport()
        block = _make_tool_use_block("mcp__unknown_tool")
        response = _make_response(block)

        registry = _FakeRegistry({"read_file"})  # neither name registered
        with patch("tools.registry.registry", registry):
            result = transport.normalize_response(response, strip_tool_prefix=True)

        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].name == "mcp__unknown_tool"

    def test_mixed_tools_same_response(self):
        """Both OAuth and native MCP tools in the same response."""
        transport = self._get_transport()
        block1 = _make_tool_use_block("mcp__read_file", block_id="tc_1")
        block2 = _make_tool_use_block("mcp_composio_SEARCH", block_id="tc_2")
        block3 = _make_tool_use_block("mcp_composio_SEARCH", block_id="tc_3")  # also registered natively
        response = _make_response(block1, block2, block3)

        registry = _FakeRegistry({
            "read_file",  # OAuth-injected
            "mcp_composio_SEARCH",  # native MCP
        })
        with patch("tools.registry.registry", registry):
            result = transport.normalize_response(response, strip_tool_prefix=True)

        assert len(result.tool_calls) == 3
        # OAuth tool: stripped
        assert result.tool_calls[0].name == "read_file"
        # Native MCP: preserved (doesn't start with mcp__ so never enters strip path)
        assert result.tool_calls[1].name == "mcp_composio_SEARCH"
        assert result.tool_calls[2].name == "mcp_composio_SEARCH"

    def test_both_stripped_and_full_registered_prefers_full(self):
        """Edge case: both 'foo' and 'mcp__foo' exist in registry.

        Keep 'mcp__foo' (the original name) since it's what the LLM requested.
        """
        transport = self._get_transport()
        block = _make_tool_use_block("mcp__foo")
        response = _make_response(block)

        registry = _FakeRegistry({"foo", "mcp__foo"})
        with patch("tools.registry.registry", registry):
            result = transport.normalize_response(response, strip_tool_prefix=True)

        assert len(result.tool_calls) == 1
        # Both exist — the condition `get_entry(stripped) and not get_entry(name)`
        # is False because get_entry(name) IS truthy, so we keep the full name.
        assert result.tool_calls[0].name == "mcp__foo"

    def test_legacy_single_underscore_native_mcp_not_stripped(self):
        """Legacy mcp_ (single underscore) native MCP tools are NOT stripped.

        They don't start with mcp__ so the strip path is never entered.
        """
        transport = self._get_transport()
        block = _make_tool_use_block("mcp_github_create_issue")
        response = _make_response(block)

        registry = _FakeRegistry({"mcp_github_create_issue"})
        with patch("tools.registry.registry", registry):
            result = transport.normalize_response(response, strip_tool_prefix=True)

        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].name == "mcp_github_create_issue"


class TestAnthropicOAuthOutgoingPrefix:
    """Verify the outgoing-side companion fix: build_anthropic_kwargs must not
    double-prefix tool names that already start with ``mcp_`` or ``mcp__``
    (native MCP server tools registered as ``mcp_<server>_<tool>``). GH-25255."""

    def _build(self, tools, is_oauth=True):
        from agent.anthropic_adapter import build_anthropic_kwargs
        return build_anthropic_kwargs(
            model="claude-sonnet-4-6",
            messages=[{"role": "user", "content": "Hi"}],
            tools=tools,
            max_tokens=4096,
            reasoning_config=None,
            is_oauth=is_oauth,
        )

    def test_oauth_adds_prefix_to_bare_tool_name(self):
        """OAuth + bare name → mcp__ prefix added (Claude Code convention)."""
        kwargs = self._build([{
            "type": "function",
            "function": {"name": "read_file", "description": "x", "parameters": {}},
        }])
        names = [t["name"] for t in kwargs["tools"]]
        assert names == ["mcp__read_file"]

    def test_oauth_does_not_double_prefix_native_mcp_tool(self):
        """OAuth + already-prefixed native MCP name → left alone."""
        kwargs = self._build([{
            "type": "function",
            "function": {
                "name": "mcp_composio_COMPOSIO_SEARCH_TOOLS",
                "description": "x",
                "parameters": {},
            },
        }])
        names = [t["name"] for t in kwargs["tools"]]
        # Must NOT become "mcp__mcp_composio_..." — that breaks the round-trip
        # because normalize_response only strips ONE mcp__ prefix.
        assert names == ["mcp_composio_COMPOSIO_SEARCH_TOOLS"]

    def test_oauth_mixed_native_and_bare_tools(self):
        """Mixed: native MCP preserved, bare names prefixed with mcp__."""
        kwargs = self._build([
            {"type": "function", "function": {"name": "read_file",
                                               "description": "x", "parameters": {}}},
            {"type": "function", "function": {"name": "mcp_composio_SEARCH",
                                               "description": "y", "parameters": {}}},
            {"type": "function", "function": {"name": "terminal",
                                               "description": "z", "parameters": {}}},
        ])
        names = sorted(t["name"] for t in kwargs["tools"])
        assert names == ["mcp__read_file", "mcp__terminal", "mcp_composio_SEARCH"]

    def test_non_oauth_path_untouched(self):
        """Non-OAuth requests never get the prefix — schemas pass through as-is."""
        kwargs = self._build([
            {"type": "function", "function": {"name": "read_file",
                                               "description": "x", "parameters": {}}},
            {"type": "function", "function": {"name": "mcp_composio_SEARCH",
                                               "description": "y", "parameters": {}}},
        ], is_oauth=False)
        names = sorted(t["name"] for t in kwargs["tools"])
        assert names == ["mcp_composio_SEARCH", "read_file"]
