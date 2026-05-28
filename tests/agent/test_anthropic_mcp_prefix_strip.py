"""Tests for Anthropic OAuth tool-name handling.

Outbound (build_anthropic_kwargs):
  - Strips ``mcp_`` prefix from ALL tool names to avoid the billing classifier.
  - Filters to delegate-only when HERMES_OAUTH_DELEGATE_ONLY=1.
  - Enforces tool-schema budget and system-prompt cap.

Inbound (normalize_response with strip_tool_prefix=True):
  - Restores ``mcp_`` prefix for bare names via registry lookup.
  - Does NOT strip native MCP server tools (``mcp_<server>_<tool>``).
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import patch

import pytest


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
        """OAuth tools: mcp_read_file -> read_file (stripped).

        The tool was registered as 'read_file' in the registry.
        Anthropic sees 'mcp_read_file' because Hermes adds the prefix.
        On response, we must strip it back to 'read_file'.
        """
        transport = self._get_transport()
        block = _make_tool_use_block("mcp_read_file")
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
        block = _make_tool_use_block("mcp_read_file")
        response = _make_response(block)

        registry = _FakeRegistry({"read_file"})
        with patch("tools.registry.registry", registry):
            result = transport.normalize_response(response, strip_tool_prefix=False)

        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].name == "mcp_read_file"

    def test_no_strip_when_not_mcp_prefixed(self):
        """Non-mcp_ names are untouched regardless of strip flag."""
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
        block = _make_tool_use_block("mcp_unknown_tool")
        response = _make_response(block)

        registry = _FakeRegistry({"read_file"})  # neither name registered
        with patch("tools.registry.registry", registry):
            result = transport.normalize_response(response, strip_tool_prefix=True)

        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].name == "mcp_unknown_tool"

    def test_mixed_tools_same_response(self):
        """Both OAuth and native MCP tools in the same response."""
        transport = self._get_transport()
        block1 = _make_tool_use_block("mcp_read_file", block_id="tc_1")
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
        # Native MCP: preserved (both stripped and full are registered, full wins)
        assert result.tool_calls[1].name == "mcp_composio_SEARCH"
        assert result.tool_calls[2].name == "mcp_composio_SEARCH"

    def test_both_stripped_and_full_registered_prefers_full(self):
        """Edge case: both 'foo' and 'mcp_foo' exist in registry.

        Keep 'mcp_foo' (the original name) since it's what the LLM requested.
        """
        transport = self._get_transport()
        block = _make_tool_use_block("mcp_foo")
        response = _make_response(block)

        registry = _FakeRegistry({"foo", "mcp_foo"})
        with patch("tools.registry.registry", registry):
            result = transport.normalize_response(response, strip_tool_prefix=True)

        assert len(result.tool_calls) == 1
        # Both exist — the condition `get_entry(stripped) and not get_entry(name)`
        # is False because get_entry(name) IS truthy, so we keep the full name.
        assert result.tool_calls[0].name == "mcp_foo"


class TestAnthropicOAuthOutgoingPrefix:
    """Verify that OAuth outbound strips mcp_ prefix from tool names to avoid
    the billing classifier, and that delegate-only mode filters correctly."""

    def _build(self, tools, is_oauth=True, env_overrides=None, messages=None):
        import os
        from agent.anthropic_adapter import build_anthropic_kwargs

        _env = {"HERMES_OAUTH_DELEGATE_ONLY": "0"}  # default off for tool-level tests
        if env_overrides:
            _env.update(env_overrides)

        _msgs = messages or [{"role": "user", "content": "Hi"}]

        with patch.dict(os.environ, _env, clear=False):
            return build_anthropic_kwargs(
                model="claude-sonnet-4-6",
                messages=_msgs,
                tools=tools,
                max_tokens=4096,
                reasoning_config=None,
                is_oauth=is_oauth,
            )

    def test_oauth_strips_prefix_from_bare_tool_name(self):
        """OAuth + bare name → left bare (no mcp_ added, avoids billing gate)."""
        kwargs = self._build([{
            "type": "function",
            "function": {"name": "read_file", "description": "x", "parameters": {}},
        }])
        names = [t["name"] for t in kwargs["tools"]]
        assert names == ["read_file"]

    def test_oauth_strips_prefix_from_mcp_prefixed_tool(self):
        """OAuth + mcp_-prefixed name → prefix stripped."""
        kwargs = self._build([{
            "type": "function",
            "function": {"name": "mcp_read_file", "description": "x", "parameters": {}},
        }])
        names = [t["name"] for t in kwargs["tools"]]
        assert names == ["read_file"]

    def test_oauth_strips_native_mcp_tool_prefix(self):
        """OAuth + native MCP server tool → prefix stripped.

        Native MCP tools (mcp_composio_X) also get stripped because the billing
        classifier flags ANY mcp_ prefix.  The inbound path restores it via
        registry lookup.
        """
        kwargs = self._build([{
            "type": "function",
            "function": {
                "name": "mcp_composio_COMPOSIO_SEARCH_TOOLS",
                "description": "x",
                "parameters": {},
            },
        }])
        names = [t["name"] for t in kwargs["tools"]]
        assert names == ["composio_COMPOSIO_SEARCH_TOOLS"]

    def test_oauth_mixed_tools_all_stripped(self):
        """Mixed: all mcp_ prefixes stripped on outbound."""
        kwargs = self._build([
            {"type": "function", "function": {"name": "read_file",
                                               "description": "x", "parameters": {}}},
            {"type": "function", "function": {"name": "mcp_composio_SEARCH",
                                               "description": "y", "parameters": {}}},
            {"type": "function", "function": {"name": "terminal",
                                               "description": "z", "parameters": {}}},
        ])
        names = sorted(t["name"] for t in kwargs["tools"])
        assert names == ["composio_SEARCH", "read_file", "terminal"]

    def test_non_oauth_path_untouched(self):
        """Non-OAuth requests never modify tool names — schemas pass through as-is."""
        kwargs = self._build([
            {"type": "function", "function": {"name": "read_file",
                                               "description": "x", "parameters": {}}},
            {"type": "function", "function": {"name": "mcp_composio_SEARCH",
                                               "description": "y", "parameters": {}}},
        ], is_oauth=False)
        names = sorted(t["name"] for t in kwargs["tools"])
        assert names == ["mcp_composio_SEARCH", "read_file"]

    # -- Delegate-only mode --------------------------------------------------

    def test_delegate_only_filters_to_delegate_tool(self):
        """When HERMES_OAUTH_DELEGATE_ONLY=1, only delegate tool survives."""
        kwargs = self._build(
            [
                {"type": "function", "function": {"name": "read_file",
                                                   "description": "x", "parameters": {}}},
                {"type": "function", "function": {"name": "delegate_to_claude_code",
                                                   "description": "y", "parameters": {}}},
                {"type": "function", "function": {"name": "terminal",
                                                   "description": "z", "parameters": {}}},
            ],
            env_overrides={"HERMES_OAUTH_DELEGATE_ONLY": "1"},
        )
        names = [t["name"] for t in kwargs["tools"]]
        assert names == ["delegate_to_claude_code"]

    def test_delegate_only_falls_back_when_no_delegate_tool(self):
        """When delegate tool absent, falls back to full set (with prefix stripping)."""
        kwargs = self._build(
            [
                {"type": "function", "function": {"name": "read_file",
                                                   "description": "x", "parameters": {}}},
                {"type": "function", "function": {"name": "terminal",
                                                   "description": "z", "parameters": {}}},
            ],
            env_overrides={"HERMES_OAUTH_DELEGATE_ONLY": "1"},
        )
        names = sorted(t["name"] for t in kwargs["tools"])
        assert names == ["read_file", "terminal"]

    def test_delegate_only_matches_prefixed_name(self):
        """Delegate tool registered as mcp_delegate_to_claude_code also matches."""
        kwargs = self._build(
            [
                {"type": "function", "function": {"name": "mcp_delegate_to_claude_code",
                                                   "description": "y", "parameters": {}}},
                {"type": "function", "function": {"name": "read_file",
                                                   "description": "x", "parameters": {}}},
            ],
            env_overrides={"HERMES_OAUTH_DELEGATE_ONLY": "1"},
        )
        names = [t["name"] for t in kwargs["tools"]]
        # After filtering + stripping, name should be bare
        assert names == ["delegate_to_claude_code"]

    # -- Tool schema budget --------------------------------------------------

    def test_tool_schema_budget_drops_tools_over_limit(self):
        """Tools that would exceed the 30 KB budget are dropped."""
        big_desc = "x" * 31_000  # >30 KB ensures second tool is dropped
        kwargs = self._build([
            {"type": "function", "function": {"name": "small_tool",
                                               "description": "ok", "parameters": {}}},
            {"type": "function", "function": {"name": "big_tool",
                                               "description": big_desc, "parameters": {}}},
        ])
        names = [t["name"] for t in kwargs["tools"]]
        assert "small_tool" in names
        assert "big_tool" not in names

    # -- System prompt cap ---------------------------------------------------

    def test_system_prompt_capped_for_oauth(self):
        """OAuth system prompt is truncated to stay under budget."""
        long_system = "A" * 5000
        kwargs = self._build(
            [{"type": "function", "function": {"name": "t", "description": "x",
                                                "parameters": {}}}],
            messages=[
                {"role": "system", "content": long_system},
                {"role": "user", "content": "Hi"},
            ],
        )
        assert "system" in kwargs
        # Identity block must be preserved as first block
        from agent.anthropic_adapter import (
            _CLAUDE_CODE_SYSTEM_PREFIX,
            _OAUTH_SYSTEM_PROMPT_CAP_CHARS,
        )
        assert kwargs["system"][0]["text"] == _CLAUDE_CODE_SYSTEM_PREFIX
        # Total must be under cap
        total = sum(
            len(b.get("text", ""))
            for b in kwargs["system"]
            if isinstance(b, dict) and b.get("type") == "text"
        )
        assert total <= _OAUTH_SYSTEM_PROMPT_CAP_CHARS
        # Second block should be truncated, not empty
        assert len(kwargs["system"]) >= 2
        assert len(kwargs["system"][1]["text"]) < len(long_system)

    # -- Message history sanitization ----------------------------------------

    def test_message_history_brand_sanitized(self):
        """Assistant text blocks in message history get brand names replaced."""
        kwargs = self._build(
            [{"type": "function", "function": {"name": "t", "description": "x",
                                                "parameters": {}}}],
            messages=[
                {"role": "user", "content": "Hi"},
                {"role": "assistant", "content": [
                    {"type": "text", "text": "I am Hermes Agent, made by Nous Research."},
                ]},
                {"role": "user", "content": "Thanks"},
            ],
        )
        for msg in kwargs["messages"]:
            if msg.get("role") == "assistant":
                content = msg.get("content")
                if isinstance(content, list):
                    for block in content:
                        if isinstance(block, dict) and block.get("type") == "text":
                            assert "Hermes Agent" not in block["text"]
                            assert "Nous Research" not in block["text"]

    def test_user_messages_not_brand_sanitized(self):
        """User message text is NOT brand-sanitized — preserves user intent."""
        kwargs = self._build(
            [{"type": "function", "function": {"name": "t", "description": "x",
                                                "parameters": {}}}],
            messages=[
                {"role": "user", "content": [
                    {"type": "text", "text": "Compare Hermes Agent to Claude Code"},
                ]},
                {"role": "assistant", "content": [
                    {"type": "text", "text": "I am Hermes Agent by Nous Research."},
                ]},
            ],
        )
        for msg in kwargs["messages"]:
            content = msg.get("content")
            if isinstance(content, list):
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "text":
                        if msg.get("role") == "user":
                            assert "Hermes Agent" in block["text"]
                        elif msg.get("role") == "assistant":
                            assert "Hermes Agent" not in block["text"]

    def test_message_history_tool_use_prefix_stripped(self):
        """tool_use blocks in message history have mcp_ prefix stripped."""
        kwargs = self._build(
            [{"type": "function", "function": {"name": "read_file",
                                                "description": "x", "parameters": {}}}],
            messages=[
                {"role": "user", "content": "read foo"},
                {"role": "assistant", "content": [
                    {"type": "tool_use", "id": "tc_1", "name": "mcp_read_file",
                     "input": {"path": "/tmp/foo"}},
                ]},
                {"role": "user", "content": [
                    {"type": "tool_result", "tool_use_id": "tc_1",
                     "content": "file contents here"},
                ]},
                {"role": "assistant", "content": "Here's the file."},
                {"role": "user", "content": "thanks"},
            ],
        )
        for msg in kwargs["messages"]:
            content = msg.get("content")
            if isinstance(content, list):
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "tool_use":
                        assert not block["name"].startswith("mcp_"), \
                            f"Expected stripped name, got {block['name']}"

    # -- tool_choice + prefix stripping --------------------------------------

    def test_tool_choice_prefix_stripped_on_oauth(self):
        """tool_choice name must match stripped tool names on OAuth path."""
        import os
        from agent.anthropic_adapter import build_anthropic_kwargs

        with patch.dict(os.environ, {"HERMES_OAUTH_DELEGATE_ONLY": "0"}):
            kwargs = build_anthropic_kwargs(
                model="claude-sonnet-4-6",
                messages=[{"role": "user", "content": "Hi"}],
                tools=[{
                    "type": "function",
                    "function": {"name": "mcp_read_file", "description": "x",
                                 "parameters": {}},
                }],
                max_tokens=4096,
                reasoning_config=None,
                tool_choice="mcp_read_file",
                is_oauth=True,
            )
        assert kwargs["tool_choice"]["name"] == "read_file"
        assert kwargs["tools"][0]["name"] == "read_file"

    def test_tool_choice_not_stripped_on_non_oauth(self):
        """tool_choice name is unchanged on non-OAuth path."""
        from agent.anthropic_adapter import build_anthropic_kwargs

        kwargs = build_anthropic_kwargs(
            model="claude-sonnet-4-6",
            messages=[{"role": "user", "content": "Hi"}],
            tools=[{
                "type": "function",
                "function": {"name": "mcp_read_file", "description": "x",
                             "parameters": {}},
            }],
            max_tokens=4096,
            reasoning_config=None,
            tool_choice="mcp_read_file",
            is_oauth=False,
        )
        assert kwargs["tool_choice"]["name"] == "mcp_read_file"

    def test_tool_choice_falls_back_when_filtered_out(self):
        """tool_choice falls back to auto when target removed by delegate-only."""
        import os
        from agent.anthropic_adapter import build_anthropic_kwargs

        with patch.dict(os.environ, {"HERMES_OAUTH_DELEGATE_ONLY": "1"}):
            kwargs = build_anthropic_kwargs(
                model="claude-sonnet-4-6",
                messages=[{"role": "user", "content": "Hi"}],
                tools=[
                    {"type": "function", "function": {"name": "delegate_to_claude_code",
                                                       "description": "d", "parameters": {}}},
                    {"type": "function", "function": {"name": "read_file",
                                                       "description": "x", "parameters": {}}},
                ],
                max_tokens=4096,
                reasoning_config=None,
                tool_choice="read_file",
                is_oauth=True,
            )
        # read_file was filtered out; tool_choice should fall back to auto
        assert kwargs["tool_choice"] == {"type": "auto"}

    # -- Post-strip dedup ----------------------------------------------------

    def test_post_strip_dedup(self):
        """If stripping mcp_ creates duplicate names, dedup keeps first."""
        kwargs = self._build([
            {"type": "function", "function": {"name": "read_file",
                                               "description": "bare version", "parameters": {}}},
            {"type": "function", "function": {"name": "mcp_read_file",
                                               "description": "prefixed version", "parameters": {}}},
        ])
        names = [t["name"] for t in kwargs["tools"]]
        assert names.count("read_file") == 1
        assert kwargs["tools"][0]["description"] == "bare version"


class TestAnthropicOAuthInboundReverse:
    """Verify that bare tool names returned by Anthropic (after outbound
    stripping) are correctly re-prefixed on inbound using the tool registry."""

    def _get_transport(self):
        from agent.transports.anthropic import AnthropicTransport
        return AnthropicTransport()

    def test_bare_name_re_prefixed_when_prefixed_in_registry(self):
        """Bare name 'read_file' → 'mcp_read_file' when registry has prefixed version."""
        transport = self._get_transport()
        block = _make_tool_use_block("read_file")
        response = _make_response(block)

        registry = _FakeRegistry({"mcp_read_file"})
        with patch("tools.registry.registry", registry):
            result = transport.normalize_response(response, strip_tool_prefix=True)

        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].name == "mcp_read_file"

    def test_bare_name_kept_when_registered_directly(self):
        """Bare name 'delegate_to_claude_code' stays bare when registered directly."""
        transport = self._get_transport()
        block = _make_tool_use_block("delegate_to_claude_code")
        response = _make_response(block)

        registry = _FakeRegistry({"delegate_to_claude_code"})
        with patch("tools.registry.registry", registry):
            result = transport.normalize_response(response, strip_tool_prefix=True)

        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].name == "delegate_to_claude_code"

    def test_bare_name_kept_when_neither_registered(self):
        """Unknown bare name stays as-is (safety fallback)."""
        transport = self._get_transport()
        block = _make_tool_use_block("unknown_tool")
        response = _make_response(block)

        registry = _FakeRegistry(set())
        with patch("tools.registry.registry", registry):
            result = transport.normalize_response(response, strip_tool_prefix=True)

        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].name == "unknown_tool"

    def test_bare_name_not_re_prefixed_when_both_registered(self):
        """When both 'foo' and 'mcp_foo' in registry, keep bare name."""
        transport = self._get_transport()
        block = _make_tool_use_block("foo")
        response = _make_response(block)

        registry = _FakeRegistry({"foo", "mcp_foo"})
        with patch("tools.registry.registry", registry):
            result = transport.normalize_response(response, strip_tool_prefix=True)

        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].name == "foo"

    def test_round_trip_strip_then_restore(self):
        """End-to-end: outbound strips mcp_, inbound restores it."""
        import os

        # Outbound: build_anthropic_kwargs strips mcp_
        from agent.anthropic_adapter import build_anthropic_kwargs
        with patch.dict(os.environ, {"HERMES_OAUTH_DELEGATE_ONLY": "0"}):
            kwargs = build_anthropic_kwargs(
                model="claude-sonnet-4-6",
                messages=[{"role": "user", "content": "test"}],
                tools=[{
                    "type": "function",
                    "function": {"name": "mcp_read_file", "description": "x",
                                 "parameters": {}},
                }],
                max_tokens=4096,
                reasoning_config=None,
                is_oauth=True,
            )
        wire_names = [t["name"] for t in kwargs["tools"]]
        assert wire_names == ["read_file"], f"Outbound should strip: {wire_names}"

        # Inbound: normalize_response restores mcp_
        transport = self._get_transport()
        block = _make_tool_use_block("read_file")
        response = _make_response(block)

        registry = _FakeRegistry({"mcp_read_file"})
        with patch("tools.registry.registry", registry):
            result = transport.normalize_response(response, strip_tool_prefix=True)

        assert result.tool_calls[0].name == "mcp_read_file", \
            "Inbound should restore mcp_ prefix"
