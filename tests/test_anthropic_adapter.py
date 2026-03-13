"""Tests for agent/anthropic_adapter.py — Anthropic Messages API adapter."""

import json
import time
from types import SimpleNamespace
from unittest.mock import patch, MagicMock

import pytest

from agent.anthropic_adapter import (
    _is_oauth_token,
    build_anthropic_client,
    build_anthropic_kwargs,
    convert_messages_to_anthropic,
    convert_tools_to_anthropic,
    is_claude_code_token_valid,
    normalize_anthropic_response,
    normalize_model_name,
    read_claude_code_credentials,
    resolve_anthropic_token,
)


# ---------------------------------------------------------------------------
# Auth helpers
# ---------------------------------------------------------------------------


class TestIsOAuthToken:
    def test_setup_token(self):
        assert _is_oauth_token("sk-ant-oat01-abcdef1234567890") is True

    def test_api_key(self):
        assert _is_oauth_token("sk-ant-api03-abcdef1234567890") is False

    def test_managed_key(self):
        # Managed keys from ~/.claude.json are NOT regular API keys
        assert _is_oauth_token("ou1R1z-ft0A-bDeZ9wAA") is True

    def test_jwt_token(self):
        # JWTs from OAuth flow
        assert _is_oauth_token("eyJhbGciOiJSUzI1NiJ9.test") is True

    def test_empty(self):
        assert _is_oauth_token("") is False


class TestBuildAnthropicClient:
    def test_setup_token_uses_auth_token(self):
        with patch("agent.anthropic_adapter._anthropic_sdk") as mock_sdk:
            build_anthropic_client("sk-ant-oat01-" + "x" * 60)
            kwargs = mock_sdk.Anthropic.call_args[1]
            assert "auth_token" in kwargs
            betas = kwargs["default_headers"]["anthropic-beta"]
            assert "oauth-2025-04-20" in betas
            assert "interleaved-thinking-2025-05-14" in betas
            assert "fine-grained-tool-streaming-2025-05-14" in betas
            assert "api_key" not in kwargs

    def test_api_key_uses_api_key(self):
        with patch("agent.anthropic_adapter._anthropic_sdk") as mock_sdk:
            build_anthropic_client("sk-ant-api03-something")
            kwargs = mock_sdk.Anthropic.call_args[1]
            assert kwargs["api_key"] == "sk-ant-api03-something"
            assert "auth_token" not in kwargs
            # API key auth should still get common betas
            betas = kwargs["default_headers"]["anthropic-beta"]
            assert "interleaved-thinking-2025-05-14" in betas
            assert "oauth-2025-04-20" not in betas  # OAuth-only beta NOT present

    def test_custom_base_url(self):
        with patch("agent.anthropic_adapter._anthropic_sdk") as mock_sdk:
            build_anthropic_client("sk-ant-api03-x", base_url="https://custom.api.com")
            kwargs = mock_sdk.Anthropic.call_args[1]
            assert kwargs["base_url"] == "https://custom.api.com"


class TestReadClaudeCodeCredentials:
    def test_reads_valid_credentials(self, tmp_path, monkeypatch):
        cred_file = tmp_path / ".claude" / ".credentials.json"
        cred_file.parent.mkdir(parents=True)
        cred_file.write_text(json.dumps({
            "claudeAiOauth": {
                "accessToken": "sk-ant-oat01-test-token",
                "refreshToken": "sk-ant-ort01-refresh",
                "expiresAt": int(time.time() * 1000) + 3600_000,
            }
        }))
        monkeypatch.setattr("agent.anthropic_adapter.Path.home", lambda: tmp_path)
        creds = read_claude_code_credentials()
        assert creds is not None
        assert creds["accessToken"] == "sk-ant-oat01-test-token"
        assert creds["refreshToken"] == "sk-ant-ort01-refresh"

    def test_returns_none_for_missing_file(self, tmp_path, monkeypatch):
        monkeypatch.setattr("agent.anthropic_adapter.Path.home", lambda: tmp_path)
        assert read_claude_code_credentials() is None

    def test_returns_none_for_missing_oauth_key(self, tmp_path, monkeypatch):
        cred_file = tmp_path / ".claude" / ".credentials.json"
        cred_file.parent.mkdir(parents=True)
        cred_file.write_text(json.dumps({"someOtherKey": {}}))
        monkeypatch.setattr("agent.anthropic_adapter.Path.home", lambda: tmp_path)
        assert read_claude_code_credentials() is None

    def test_returns_none_for_empty_access_token(self, tmp_path, monkeypatch):
        cred_file = tmp_path / ".claude" / ".credentials.json"
        cred_file.parent.mkdir(parents=True)
        cred_file.write_text(json.dumps({
            "claudeAiOauth": {"accessToken": "", "refreshToken": "x"}
        }))
        monkeypatch.setattr("agent.anthropic_adapter.Path.home", lambda: tmp_path)
        assert read_claude_code_credentials() is None


class TestIsClaudeCodeTokenValid:
    def test_valid_token(self):
        creds = {"accessToken": "tok", "expiresAt": int(time.time() * 1000) + 3600_000}
        assert is_claude_code_token_valid(creds) is True

    def test_expired_token(self):
        creds = {"accessToken": "tok", "expiresAt": int(time.time() * 1000) - 3600_000}
        assert is_claude_code_token_valid(creds) is False

    def test_no_expiry_but_has_token(self):
        creds = {"accessToken": "tok", "expiresAt": 0}
        assert is_claude_code_token_valid(creds) is True


class TestResolveAnthropicToken:
    def test_prefers_api_key(self, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-api03-mykey")
        monkeypatch.setenv("ANTHROPIC_TOKEN", "sk-ant-oat01-mytoken")
        assert resolve_anthropic_token() == "sk-ant-api03-mykey"

    def test_falls_back_to_token(self, monkeypatch):
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        monkeypatch.setenv("ANTHROPIC_TOKEN", "sk-ant-oat01-mytoken")
        assert resolve_anthropic_token() == "sk-ant-oat01-mytoken"

    def test_returns_none_with_no_creds(self, monkeypatch, tmp_path):
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        monkeypatch.delenv("ANTHROPIC_TOKEN", raising=False)
        monkeypatch.delenv("CLAUDE_CODE_OAUTH_TOKEN", raising=False)
        monkeypatch.setattr("agent.anthropic_adapter.Path.home", lambda: tmp_path)
        assert resolve_anthropic_token() is None


# ---------------------------------------------------------------------------
# Model name normalization
# ---------------------------------------------------------------------------


class TestNormalizeModelName:
    def test_strips_anthropic_prefix(self):
        assert normalize_model_name("anthropic/claude-sonnet-4-20250514") == "claude-sonnet-4-20250514"

    def test_leaves_bare_name(self):
        assert normalize_model_name("claude-sonnet-4-20250514") == "claude-sonnet-4-20250514"


# ---------------------------------------------------------------------------
# Tool conversion
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Message conversion
# ---------------------------------------------------------------------------


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
            {
                "role": "assistant",
                "content": "Let me search.",
                "tool_calls": [
                    {
                        "id": "tc_1",
                        "function": {
                            "name": "search",
                            "arguments": '{"query": "test"}',
                        },
                    }
                ],
            },
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
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {"id": "tc_orphan", "function": {"name": "x", "arguments": "{}"}}
                ],
            },
            {"role": "user", "content": "never mind"},
        ]
        _, result = convert_messages_to_anthropic(messages)
        # tc_orphan has no matching tool_result, should be stripped
        assistant_blocks = result[0]["content"]
        assert all(b.get("type") != "tool_use" for b in assistant_blocks)

    def test_system_with_cache_control(self):
        messages = [
            {
                "role": "system",
                "content": [
                    {"type": "text", "text": "System prompt", "cache_control": {"type": "ephemeral"}},
                ],
            },
            {"role": "user", "content": "Hi"},
        ]
        system, result = convert_messages_to_anthropic(messages)
        # When cache_control is present, system should be a list of blocks
        assert isinstance(system, list)
        assert system[0]["cache_control"] == {"type": "ephemeral"}


# ---------------------------------------------------------------------------
# Build kwargs
# ---------------------------------------------------------------------------


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

    def test_strips_anthropic_prefix(self):
        kwargs = build_anthropic_kwargs(
            model="anthropic/claude-sonnet-4-20250514",
            messages=[{"role": "user", "content": "Hi"}],
            tools=None,
            max_tokens=4096,
            reasoning_config=None,
        )
        assert kwargs["model"] == "claude-sonnet-4-20250514"

    def test_reasoning_config_maps_to_manual_thinking_for_pre_4_6_models(self):
        kwargs = build_anthropic_kwargs(
            model="claude-sonnet-4-20250514",
            messages=[{"role": "user", "content": "think hard"}],
            tools=None,
            max_tokens=4096,
            reasoning_config={"enabled": True, "effort": "high"},
        )
        assert kwargs["thinking"]["type"] == "enabled"
        assert kwargs["thinking"]["budget_tokens"] == 16000
        assert kwargs["temperature"] == 1
        assert kwargs["max_tokens"] >= 16000 + 4096
        assert "output_config" not in kwargs

    def test_reasoning_config_maps_to_adaptive_thinking_for_4_6_models(self):
        kwargs = build_anthropic_kwargs(
            model="claude-opus-4-6",
            messages=[{"role": "user", "content": "think hard"}],
            tools=None,
            max_tokens=4096,
            reasoning_config={"enabled": True, "effort": "high"},
        )
        assert kwargs["thinking"] == {"type": "adaptive"}
        assert kwargs["output_config"] == {"effort": "high"}
        assert "budget_tokens" not in kwargs["thinking"]
        assert "temperature" not in kwargs
        assert kwargs["max_tokens"] == 4096

    def test_reasoning_config_maps_xhigh_to_max_effort_for_4_6_models(self):
        kwargs = build_anthropic_kwargs(
            model="claude-sonnet-4-6",
            messages=[{"role": "user", "content": "think harder"}],
            tools=None,
            max_tokens=4096,
            reasoning_config={"enabled": True, "effort": "xhigh"},
        )
        assert kwargs["thinking"] == {"type": "adaptive"}
        assert kwargs["output_config"] == {"effort": "max"}

    def test_reasoning_disabled(self):
        kwargs = build_anthropic_kwargs(
            model="claude-sonnet-4-20250514",
            messages=[{"role": "user", "content": "quick"}],
            tools=None,
            max_tokens=4096,
            reasoning_config={"enabled": False},
        )
        assert "thinking" not in kwargs

    def test_default_max_tokens(self):
        kwargs = build_anthropic_kwargs(
            model="claude-sonnet-4-20250514",
            messages=[{"role": "user", "content": "Hi"}],
            tools=None,
            max_tokens=None,
            reasoning_config=None,
        )
        assert kwargs["max_tokens"] == 16384


# ---------------------------------------------------------------------------
# Response normalization
# ---------------------------------------------------------------------------


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
        _, r1 = normalize_anthropic_response(
            self._make_response([block], "end_turn")
        )
        _, r2 = normalize_anthropic_response(
            self._make_response([block], "tool_use")
        )
        _, r3 = normalize_anthropic_response(
            self._make_response([block], "max_tokens")
        )
        assert r1 == "stop"
        assert r2 == "tool_calls"
        assert r3 == "length"

    def test_no_text_content(self):
        block = SimpleNamespace(
            type="tool_use", id="tc_1", name="search", input={"q": "hi"}
        )
        msg, reason = normalize_anthropic_response(
            self._make_response([block], "tool_use")
        )
        assert msg.content is None
        assert len(msg.tool_calls) == 1


# ---------------------------------------------------------------------------
# Role alternation
# ---------------------------------------------------------------------------


class TestRoleAlternation:
    def test_merges_consecutive_user_messages(self):
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "user", "content": "World"},
        ]
        _, result = convert_messages_to_anthropic(messages)
        assert len(result) == 1
        assert result[0]["role"] == "user"
        assert "Hello" in result[0]["content"]
        assert "World" in result[0]["content"]

    def test_preserves_proper_alternation(self):
        messages = [
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello!"},
            {"role": "user", "content": "How are you?"},
        ]
        _, result = convert_messages_to_anthropic(messages)
        assert len(result) == 3
        assert [m["role"] for m in result] == ["user", "assistant", "user"]


# ---------------------------------------------------------------------------
# Tool choice
# ---------------------------------------------------------------------------


class TestToolChoice:
    _DUMMY_TOOL = [
        {
            "type": "function",
            "function": {
                "name": "test",
                "description": "x",
                "parameters": {"type": "object", "properties": {}},
            },
        }
    ]

    def test_auto_tool_choice(self):
        kwargs = build_anthropic_kwargs(
            model="claude-sonnet-4-20250514",
            messages=[{"role": "user", "content": "Hi"}],
            tools=self._DUMMY_TOOL,
            max_tokens=4096,
            reasoning_config=None,
            tool_choice="auto",
        )
        assert kwargs["tool_choice"] == {"type": "auto"}

    def test_required_tool_choice(self):
        kwargs = build_anthropic_kwargs(
            model="claude-sonnet-4-20250514",
            messages=[{"role": "user", "content": "Hi"}],
            tools=self._DUMMY_TOOL,
            max_tokens=4096,
            reasoning_config=None,
            tool_choice="required",
        )
        assert kwargs["tool_choice"] == {"type": "any"}

    def test_specific_tool_choice(self):
        kwargs = build_anthropic_kwargs(
            model="claude-sonnet-4-20250514",
            messages=[{"role": "user", "content": "Hi"}],
            tools=self._DUMMY_TOOL,
            max_tokens=4096,
            reasoning_config=None,
            tool_choice="search",
        )
        assert kwargs["tool_choice"] == {"type": "tool", "name": "search"}
