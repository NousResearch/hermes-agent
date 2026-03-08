"""Tests for agent/anthropic_adapter.py — Anthropic Messages API adapter."""

import json
from types import SimpleNamespace
from unittest.mock import patch, MagicMock

import pytest

from agent.anthropic_adapter import (
    build_anthropic_client,
    build_anthropic_kwargs,
    convert_messages_to_anthropic,
    convert_tools_to_anthropic,
    normalize_anthropic_response,
)


class TestBuildAnthropicClient:
    def test_setup_token_uses_auth_token(self):
        with patch("agent.anthropic_adapter.anthropic.Anthropic") as mock_cls:
            build_anthropic_client("sk-ant-oat01-abcdefghijklmnop" + "x" * 60)
            kwargs = mock_cls.call_args[1]
            assert "auth_token" in kwargs
            assert kwargs["default_headers"]["anthropic-beta"] == "oauth-2025-04-20"
            assert "api_key" not in kwargs

    def test_api_key_uses_api_key(self):
        with patch("agent.anthropic_adapter.anthropic.Anthropic") as mock_cls:
            build_anthropic_client("sk-ant-api03-something")
            kwargs = mock_cls.call_args[1]
            assert kwargs["api_key"] == "sk-ant-api03-something"
            assert "auth_token" not in kwargs


class TestConvertTools:
    def test_converts_openai_to_anthropic_format(self):
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "search",
                    "description": "Search the web",
                    "parameters": {
                        "type": "object",
                        "properties": {"query": {"type": "string"}},
                        "required": ["query"],
                    },
                },
            }
        ]
        result = convert_tools_to_anthropic(tools)
        assert len(result) == 1
        assert result[0]["name"] == "search"
        assert result[0]["description"] == "Search the web"
        assert result[0]["input_schema"]["properties"]["query"]["type"] == "string"

    def test_empty_tools(self):
        assert convert_tools_to_anthropic([]) == []
        assert convert_tools_to_anthropic(None) == []


class TestConvertMessages:
    def test_extracts_system_prompt(self):
        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hello"},
        ]
        system, result = convert_messages_to_anthropic(messages)
        assert system == "You are helpful."
        assert len(result) == 1
        assert result[0]["role"] == "user"

    def test_converts_tool_calls(self):
        messages = [
            {"role": "assistant", "content": "Let me search.", "tool_calls": [
                {"id": "tc_1", "function": {"name": "search", "arguments": '{"query": "test"}'}}
            ]},
            {"role": "tool", "tool_call_id": "tc_1", "content": "search results"},
        ]
        _, result = convert_messages_to_anthropic(messages)
        blocks = result[0]["content"]
        assert blocks[0] == {"type": "text", "text": "Let me search."}
        assert blocks[1]["type"] == "tool_use"
        assert blocks[1]["id"] == "tc_1"
        assert blocks[1]["input"] == {"query": "test"}

    def test_converts_tool_results(self):
        messages = [
            {"role": "tool", "tool_call_id": "tc_1", "content": "result data"},
        ]
        _, result = convert_messages_to_anthropic(messages)
        assert result[0]["role"] == "user"
        assert result[0]["content"][0]["type"] == "tool_result"
        assert result[0]["content"][0]["tool_use_id"] == "tc_1"

    def test_merges_consecutive_tool_results(self):
        messages = [
            {"role": "tool", "tool_call_id": "tc_1", "content": "result 1"},
            {"role": "tool", "tool_call_id": "tc_2", "content": "result 2"},
        ]
        _, result = convert_messages_to_anthropic(messages)
        assert len(result) == 1
        assert len(result[0]["content"]) == 2

    def test_strips_orphaned_tool_use(self):
        messages = [
            {"role": "assistant", "content": "", "tool_calls": [
                {"id": "tc_orphan", "function": {"name": "x", "arguments": "{}"}}
            ]},
            {"role": "user", "content": "never mind"},
        ]
        _, result = convert_messages_to_anthropic(messages)
        # tc_orphan has no matching tool_result, should be stripped
        assistant_blocks = result[0]["content"]
        assert all(b.get("type") != "tool_use" for b in assistant_blocks)


class TestBuildAnthropicKwargs:
    def test_basic_kwargs(self):
        messages = [
            {"role": "system", "content": "Be helpful."},
            {"role": "user", "content": "Hi"},
        ]
        kwargs = build_anthropic_kwargs(
            model="claude-sonnet-4-20250514",
            messages=messages,
            tools=None,
            max_tokens=4096,
            reasoning_config=None,
        )
        assert kwargs["model"] == "claude-sonnet-4-20250514"
        assert kwargs["system"] == "Be helpful."
        assert kwargs["max_tokens"] == 4096
        assert "tools" not in kwargs

    def test_reasoning_config_maps_to_thinking(self):
        kwargs = build_anthropic_kwargs(
            model="claude-sonnet-4-20250514",
            messages=[{"role": "user", "content": "think hard"}],
            tools=None,
            max_tokens=4096,
            reasoning_config={"enabled": True, "effort": "high"},
        )
        assert kwargs["thinking"]["type"] == "adaptive"
        assert kwargs["thinking"]["budget_tokens"] == 16000
        assert kwargs["max_tokens"] >= 16000 + 4096

    def test_reasoning_disabled(self):
        kwargs = build_anthropic_kwargs(
            model="claude-sonnet-4-20250514",
            messages=[{"role": "user", "content": "quick"}],
            tools=None,
            max_tokens=4096,
            reasoning_config={"enabled": False},
        )
        assert "thinking" not in kwargs


class TestNormalizeResponse:
    def _make_response(self, content_blocks, stop_reason="end_turn"):
        resp = SimpleNamespace()
        resp.content = content_blocks
        resp.stop_reason = stop_reason
        resp.usage = SimpleNamespace(input_tokens=100, output_tokens=50)
        return resp

    def test_text_response(self):
        block = SimpleNamespace(type="text", text="Hello world")
        msg, reason = normalize_anthropic_response(self._make_response([block]))
        assert msg.content == "Hello world"
        assert reason == "stop"
        assert msg.tool_calls is None

    def test_tool_use_response(self):
        blocks = [
            SimpleNamespace(type="text", text="Searching..."),
            SimpleNamespace(
                type="tool_use",
                id="tc_1",
                name="search",
                input={"query": "test"},
            ),
        ]
        msg, reason = normalize_anthropic_response(
            self._make_response(blocks, "tool_use")
        )
        assert msg.content == "Searching..."
        assert reason == "tool_calls"
        assert len(msg.tool_calls) == 1
        assert msg.tool_calls[0].function.name == "search"
        assert json.loads(msg.tool_calls[0].function.arguments) == {"query": "test"}

    def test_thinking_response(self):
        blocks = [
            SimpleNamespace(type="thinking", thinking="Let me reason about this..."),
            SimpleNamespace(type="text", text="The answer is 42."),
        ]
        msg, reason = normalize_anthropic_response(self._make_response(blocks))
        assert msg.content == "The answer is 42."
        assert msg.reasoning == "Let me reason about this..."

    def test_stop_reason_mapping(self):
        block = SimpleNamespace(type="text", text="x")
        _, r1 = normalize_anthropic_response(self._make_response([block], "end_turn"))
        _, r2 = normalize_anthropic_response(self._make_response([block], "tool_use"))
        _, r3 = normalize_anthropic_response(self._make_response([block], "max_tokens"))
        assert r1 == "stop"
        assert r2 == "tool_calls"
        assert r3 == "length"
