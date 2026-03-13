"""Tests for agent/anthropic_adapter.py — Anthropic Messages API adapter."""

import json
import time
from types import SimpleNamespace
from unittest.mock import patch, MagicMock

import pytest

from agent.anthropic_adapter import (
    _is_oauth_token,
    _refresh_oauth_token,
    _write_claude_code_credentials,
    build_anthropic_client,
    build_anthropic_kwargs,
    convert_messages_to_anthropic,
    convert_tools_to_anthropic,
    is_claude_code_token_valid,
    normalize_anthropic_response,
    normalize_model_name,
    read_claude_code_credentials,
    resolve_anthropic_token,
    run_oauth_setup_token,
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
            assert "claude-code-20250219" in betas
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
            assert "claude-code-20250219" not in betas  # OAuth-only beta NOT present

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

    def test_falls_back_to_claude_code_oauth_token(self, monkeypatch, tmp_path):
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        monkeypatch.delenv("ANTHROPIC_TOKEN", raising=False)
        monkeypatch.setenv("CLAUDE_CODE_OAUTH_TOKEN", "sk-ant-oat01-test-token")
        monkeypatch.setattr("agent.anthropic_adapter.Path.home", lambda: tmp_path)
        assert resolve_anthropic_token() == "sk-ant-oat01-test-token"

    def test_falls_back_to_claude_code_credentials(self, monkeypatch, tmp_path):
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        monkeypatch.delenv("ANTHROPIC_TOKEN", raising=False)
        monkeypatch.delenv("CLAUDE_CODE_OAUTH_TOKEN", raising=False)
        cred_file = tmp_path / ".claude" / ".credentials.json"
        cred_file.parent.mkdir(parents=True)
        cred_file.write_text(json.dumps({
            "claudeAiOauth": {
                "accessToken": "cc-auto-token",
                "refreshToken": "refresh",
                "expiresAt": int(time.time() * 1000) + 3600_000,
            }
        }))
        monkeypatch.setattr("agent.anthropic_adapter.Path.home", lambda: tmp_path)
        assert resolve_anthropic_token() == "cc-auto-token"


class TestRefreshOauthToken:
    def test_returns_none_without_refresh_token(self):
        creds = {"accessToken": "expired", "refreshToken": "", "expiresAt": 0}
        assert _refresh_oauth_token(creds) is None

    def test_successful_refresh(self, tmp_path, monkeypatch):
        monkeypatch.setattr("agent.anthropic_adapter.Path.home", lambda: tmp_path)

        creds = {
            "accessToken": "old-token",
            "refreshToken": "refresh-123",
            "expiresAt": int(time.time() * 1000) - 3600_000,
        }

        mock_response = json.dumps({
            "access_token": "new-token-abc",
            "refresh_token": "new-refresh-456",
            "expires_in": 7200,
        }).encode()

        with patch("urllib.request.urlopen") as mock_urlopen:
            mock_ctx = MagicMock()
            mock_ctx.__enter__ = MagicMock(return_value=MagicMock(
                read=MagicMock(return_value=mock_response)
            ))
            mock_ctx.__exit__ = MagicMock(return_value=False)
            mock_urlopen.return_value = mock_ctx

            result = _refresh_oauth_token(creds)

        assert result == "new-token-abc"
        # Verify credentials were written back
        cred_file = tmp_path / ".claude" / ".credentials.json"
        assert cred_file.exists()
        written = json.loads(cred_file.read_text())
        assert written["claudeAiOauth"]["accessToken"] == "new-token-abc"
        assert written["claudeAiOauth"]["refreshToken"] == "new-refresh-456"

    def test_failed_refresh_returns_none(self):
        creds = {
            "accessToken": "old",
            "refreshToken": "refresh-123",
            "expiresAt": 0,
        }

        with patch("urllib.request.urlopen", side_effect=Exception("network error")):
            assert _refresh_oauth_token(creds) is None


class TestWriteClaudeCodeCredentials:
    def test_writes_new_file(self, tmp_path, monkeypatch):
        monkeypatch.setattr("agent.anthropic_adapter.Path.home", lambda: tmp_path)
        _write_claude_code_credentials("tok", "ref", 12345)
        cred_file = tmp_path / ".claude" / ".credentials.json"
        assert cred_file.exists()
        data = json.loads(cred_file.read_text())
        assert data["claudeAiOauth"]["accessToken"] == "tok"
        assert data["claudeAiOauth"]["refreshToken"] == "ref"
        assert data["claudeAiOauth"]["expiresAt"] == 12345

    def test_preserves_existing_fields(self, tmp_path, monkeypatch):
        monkeypatch.setattr("agent.anthropic_adapter.Path.home", lambda: tmp_path)
        cred_dir = tmp_path / ".claude"
        cred_dir.mkdir()
        cred_file = cred_dir / ".credentials.json"
        cred_file.write_text(json.dumps({"otherField": "keep-me"}))
        _write_claude_code_credentials("new-tok", "new-ref", 99999)
        data = json.loads(cred_file.read_text())
        assert data["otherField"] == "keep-me"
        assert data["claudeAiOauth"]["accessToken"] == "new-tok"


class TestResolveWithRefresh:
    def test_auto_refresh_on_expired_creds(self, monkeypatch, tmp_path):
        """When cred file has expired token + refresh token, auto-refresh is attempted."""
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        monkeypatch.delenv("ANTHROPIC_TOKEN", raising=False)
        monkeypatch.delenv("CLAUDE_CODE_OAUTH_TOKEN", raising=False)

        # Set up expired creds with a refresh token
        cred_file = tmp_path / ".claude" / ".credentials.json"
        cred_file.parent.mkdir(parents=True)
        cred_file.write_text(json.dumps({
            "claudeAiOauth": {
                "accessToken": "expired-tok",
                "refreshToken": "valid-refresh",
                "expiresAt": int(time.time() * 1000) - 3600_000,
            }
        }))
        monkeypatch.setattr("agent.anthropic_adapter.Path.home", lambda: tmp_path)

        # Mock refresh to succeed
        with patch("agent.anthropic_adapter._refresh_oauth_token", return_value="refreshed-token"):
            result = resolve_anthropic_token()

        assert result == "refreshed-token"


class TestRunOauthSetupToken:
    def test_raises_when_claude_not_installed(self, monkeypatch):
        monkeypatch.setattr("shutil.which", lambda _: None)
        with pytest.raises(FileNotFoundError, match="claude.*CLI.*not installed"):
            run_oauth_setup_token()

    def test_returns_token_from_credential_files(self, monkeypatch, tmp_path):
        """After subprocess completes, reads credentials from Claude Code files."""
        monkeypatch.setattr("shutil.which", lambda _: "/usr/bin/claude")
        monkeypatch.delenv("CLAUDE_CODE_OAUTH_TOKEN", raising=False)
        monkeypatch.delenv("ANTHROPIC_TOKEN", raising=False)

        # Pre-create credential files that will be found after subprocess
        cred_file = tmp_path / ".claude" / ".credentials.json"
        cred_file.parent.mkdir(parents=True)
        cred_file.write_text(json.dumps({
            "claudeAiOauth": {
                "accessToken": "from-cred-file",
                "refreshToken": "refresh",
                "expiresAt": int(time.time() * 1000) + 3600_000,
            }
        }))
        monkeypatch.setattr("agent.anthropic_adapter.Path.home", lambda: tmp_path)

        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0)
            token = run_oauth_setup_token()

        assert token == "from-cred-file"
        mock_run.assert_called_once()

    def test_returns_token_from_env_var(self, monkeypatch, tmp_path):
        """Falls back to CLAUDE_CODE_OAUTH_TOKEN env var when no cred files."""
        monkeypatch.setattr("shutil.which", lambda _: "/usr/bin/claude")
        monkeypatch.setenv("CLAUDE_CODE_OAUTH_TOKEN", "from-env-var")
        monkeypatch.delenv("ANTHROPIC_TOKEN", raising=False)
        monkeypatch.setattr("agent.anthropic_adapter.Path.home", lambda: tmp_path)

        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0)
            token = run_oauth_setup_token()

        assert token == "from-env-var"

    def test_returns_none_when_no_creds_found(self, monkeypatch, tmp_path):
        """Returns None when subprocess completes but no credentials are found."""
        monkeypatch.setattr("shutil.which", lambda _: "/usr/bin/claude")
        monkeypatch.delenv("CLAUDE_CODE_OAUTH_TOKEN", raising=False)
        monkeypatch.delenv("ANTHROPIC_TOKEN", raising=False)
        monkeypatch.setattr("agent.anthropic_adapter.Path.home", lambda: tmp_path)

        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0)
            token = run_oauth_setup_token()

        assert token is None

    def test_returns_none_on_keyboard_interrupt(self, monkeypatch):
        """Returns None gracefully when user interrupts the flow."""
        monkeypatch.setattr("shutil.which", lambda _: "/usr/bin/claude")

        with patch("subprocess.run", side_effect=KeyboardInterrupt):
            token = run_oauth_setup_token()

        assert token is None


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
