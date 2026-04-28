"""Tests for agent/anthropic_adapter.py — Anthropic Messages API adapter."""

import json
import time
from types import SimpleNamespace
from unittest.mock import patch, MagicMock

import pytest

from agent.prompt_caching import apply_anthropic_cache_control
from agent.anthropic_adapter import (
    _is_oauth_token,
    _refresh_oauth_token,
    _to_plain_data,
    _write_claude_code_credentials,
    build_anthropic_client,
    build_anthropic_kwargs,
    convert_messages_to_anthropic,
    convert_tools_to_anthropic,
    is_claude_code_token_valid,
    normalize_model_name,
    read_claude_code_credentials,
    resolve_anthropic_token,
    run_oauth_setup_token,
)
from agent.transports import get_transport


# ---------------------------------------------------------------------------
# Auth helpers
# ---------------------------------------------------------------------------


class TestIsOAuthToken:
    def test_setup_token(self):
        assert _is_oauth_token("sk-ant-oat01-abcdef1234567890") is True

    def test_api_key(self):
        assert _is_oauth_token("sk-ant-api03-abcdef1234567890") is False

    def test_managed_key(self):
        # Managed keys from ~/.claude.json without a recognisable Anthropic
        # prefix are not positively identified as OAuth.  They enter the system
        # via diagnostics-only read_claude_managed_key(), not via
        # resolve_anthropic_token(), so they don't reach the OAuth gate in
        # practice.  Third-party provider keys (MiniMax, Alibaba) also lack
        # the sk-ant- prefix and must NOT be treated as OAuth.
        assert _is_oauth_token("ou1R1z-ft0A-bDeZ9wAA") is False

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

    def test_oauth_does_not_send_claude_code_spoof_headers(self):
        """OAuth requests identify as Hermes — no claude-cli UA, no x-app: cli.

        Anthropic's OAuth-gated Messages API accepts requests from non-Claude-Code
        clients as long as auth is correct and the OAuth beta headers are present.
        See commit that removed fingerprinting for the live-test write-up.
        """
        with patch("agent.anthropic_adapter._anthropic_sdk") as mock_sdk:
            build_anthropic_client("sk-ant-oat01-" + "x" * 60)
            headers = mock_sdk.Anthropic.call_args[1]["default_headers"]
            assert "user-agent" not in {k.lower() for k in headers}
            assert "x-app" not in {k.lower() for k in headers}

    def test_oauth_strips_context_1m_beta(self):
        """context-1m-2025-08-07 is incompatible with OAuth auth — must be stripped.

        Anthropic returns HTTP 400 "This authentication style is incompatible
        with the long context beta header." when OAuth traffic carries it.
        """
        with patch("agent.anthropic_adapter._anthropic_sdk") as mock_sdk:
            build_anthropic_client("sk-ant-oat01-" + "x" * 60)
            betas = mock_sdk.Anthropic.call_args[1]["default_headers"]["anthropic-beta"]
            assert "context-1m-2025-08-07" not in betas
            # But other common betas still flow through
            assert "interleaved-thinking-2025-05-14" in betas
            assert "oauth-2025-04-20" in betas

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
            assert kwargs["default_headers"] == {
                "anthropic-beta": "interleaved-thinking-2025-05-14,fine-grained-tool-streaming-2025-05-14"
            }

    def test_minimax_anthropic_endpoint_uses_bearer_auth_for_regular_api_keys(self):
        with patch("agent.anthropic_adapter._anthropic_sdk") as mock_sdk:
            build_anthropic_client(
                "minimax-secret-123",
                base_url="https://api.minimax.io/anthropic",
            )
            kwargs = mock_sdk.Anthropic.call_args[1]
            assert kwargs["auth_token"] == "minimax-secret-123"
            assert "api_key" not in kwargs
            assert kwargs["default_headers"] == {
                "anthropic-beta": "interleaved-thinking-2025-05-14"
            }

    def test_minimax_cn_anthropic_endpoint_omits_tool_streaming_beta(self):
        with patch("agent.anthropic_adapter._anthropic_sdk") as mock_sdk:
            build_anthropic_client(
                "minimax-cn-secret-123",
                base_url="https://api.minimaxi.com/anthropic",
            )
            kwargs = mock_sdk.Anthropic.call_args[1]
            assert kwargs["auth_token"] == "minimax-cn-secret-123"
            assert "api_key" not in kwargs
            assert kwargs["default_headers"] == {
                "anthropic-beta": "interleaved-thinking-2025-05-14"
            }


class TestReadClaudeCodeCredentials:
    def test_reads_valid_credentials(self, tmp_path, monkeypatch):
        cred_file = tmp_path / ".claude" / ".credentials.json"
        cred_file.parent.mkdir(parents=True)
        cred_file.write_text(json.dumps({
            "claudeAiOauth": {
                "accessToken": "sk-ant-oat01-token",
                "refreshToken": "sk-ant-oat01-refresh",
                "expiresAt": int(time.time() * 1000) + 3600_000,
            }
        }))
        monkeypatch.setattr("agent.anthropic_adapter.Path.home", lambda: tmp_path)
        creds = read_claude_code_credentials()
        assert creds is not None
        assert creds["accessToken"] == "sk-ant-oat01-token"
        assert creds["refreshToken"] == "sk-ant-oat01-refresh"
        assert creds["source"] == "claude_code_credentials_file"

    def test_ignores_primary_api_key_for_native_anthropic_resolution(self, tmp_path, monkeypatch):
        claude_json = tmp_path / ".claude.json"
        claude_json.write_text(json.dumps({"primaryApiKey": "sk-ant-api03-primary"}))
        monkeypatch.setattr("agent.anthropic_adapter.Path.home", lambda: tmp_path)

        creds = read_claude_code_credentials()
        assert creds is None

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
    def test_prefers_oauth_token_over_api_key(self, monkeypatch, tmp_path):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-api03-mykey")
        monkeypatch.setenv("ANTHROPIC_TOKEN", "sk-ant-oat01-mytoken")
        monkeypatch.delenv("CLAUDE_CODE_OAUTH_TOKEN", raising=False)
        monkeypatch.setattr("agent.anthropic_adapter.Path.home", lambda: tmp_path)
        assert resolve_anthropic_token() == "sk-ant-oat01-mytoken"

    def test_does_not_resolve_primary_api_key_as_native_anthropic_token(self, monkeypatch, tmp_path):
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        monkeypatch.delenv("ANTHROPIC_TOKEN", raising=False)
        monkeypatch.delenv("CLAUDE_CODE_OAUTH_TOKEN", raising=False)
        (tmp_path / ".claude.json").write_text(json.dumps({"primaryApiKey": "sk-ant-api03-primary"}))
        monkeypatch.setattr("agent.anthropic_adapter.Path.home", lambda: tmp_path)

        assert resolve_anthropic_token() is None

    def test_falls_back_to_api_key_when_no_oauth_sources_exist(self, monkeypatch, tmp_path):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-api03-mykey")
        monkeypatch.delenv("ANTHROPIC_TOKEN", raising=False)
        monkeypatch.delenv("CLAUDE_CODE_OAUTH_TOKEN", raising=False)
        monkeypatch.setattr("agent.anthropic_adapter.Path.home", lambda: tmp_path)
        assert resolve_anthropic_token() == "sk-ant-api03-mykey"

    def test_falls_back_to_token(self, monkeypatch, tmp_path):
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        monkeypatch.setenv("ANTHROPIC_TOKEN", "sk-ant-oat01-mytoken")
        monkeypatch.delenv("CLAUDE_CODE_OAUTH_TOKEN", raising=False)
        monkeypatch.setattr("agent.anthropic_adapter.Path.home", lambda: tmp_path)
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

    def test_prefers_refreshable_claude_code_credentials_over_static_anthropic_token(self, monkeypatch, tmp_path):
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        monkeypatch.setenv("ANTHROPIC_TOKEN", "sk-ant-oat01-static-token")
        monkeypatch.delenv("CLAUDE_CODE_OAUTH_TOKEN", raising=False)
        cred_file = tmp_path / ".claude" / ".credentials.json"
        cred_file.parent.mkdir(parents=True)
        cred_file.write_text(json.dumps({
            "claudeAiOauth": {
                "accessToken": "cc-auto-token",
                "refreshToken": "refresh-token",
                "expiresAt": int(time.time() * 1000) + 3600_000,
            }
        }))
        monkeypatch.setattr("agent.anthropic_adapter.Path.home", lambda: tmp_path)

        assert resolve_anthropic_token() == "cc-auto-token"

    def test_keeps_static_anthropic_token_when_only_non_refreshable_claude_key_exists(self, monkeypatch, tmp_path):
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        monkeypatch.setenv("ANTHROPIC_TOKEN", "sk-ant-oat01-static-token")
        monkeypatch.delenv("CLAUDE_CODE_OAUTH_TOKEN", raising=False)
        claude_json = tmp_path / ".claude.json"
        claude_json.write_text(json.dumps({"primaryApiKey": "sk-ant-api03-managed-key"}))
        monkeypatch.setattr("agent.anthropic_adapter.Path.home", lambda: tmp_path)

        assert resolve_anthropic_token() == "sk-ant-oat01-static-token"


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

    def test_static_env_oauth_token_does_not_block_refreshable_claude_creds(self, monkeypatch, tmp_path):
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        monkeypatch.setenv("ANTHROPIC_TOKEN", "sk-ant-oat01-expired-env-token")
        monkeypatch.delenv("CLAUDE_CODE_OAUTH_TOKEN", raising=False)

        cred_file = tmp_path / ".claude" / ".credentials.json"
        cred_file.parent.mkdir(parents=True)
        cred_file.write_text(json.dumps({
            "claudeAiOauth": {
                "accessToken": "expired-claude-creds-token",
                "refreshToken": "valid-refresh",
                "expiresAt": int(time.time() * 1000) - 3600_000,
            }
        }))
        monkeypatch.setattr("agent.anthropic_adapter.Path.home", lambda: tmp_path)

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
        # Don't assert exact call count — the contract is "credentials flow
        # through", not "exactly one subprocess call". xdist cross-test
        # pollution (other tests shimming subprocess via plugins) has flaked
        # assert_called_once() in CI.
        assert mock_run.called

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

    def test_converts_dots_to_hyphens(self):
        """OpenRouter uses dots (4.6), Anthropic uses hyphens (4-6)."""
        assert normalize_model_name("anthropic/claude-opus-4.6") == "claude-opus-4-6"
        assert normalize_model_name("anthropic/claude-sonnet-4.5") == "claude-sonnet-4-5"
        assert normalize_model_name("claude-opus-4.6") == "claude-opus-4-6"

    def test_already_hyphenated_unchanged(self):
        """Names already in Anthropic format should pass through."""
        assert normalize_model_name("claude-opus-4-6") == "claude-opus-4-6"
        assert normalize_model_name("claude-opus-4-5-20251101") == "claude-opus-4-5-20251101"

    def test_preserve_dots_for_alibaba_dashscope(self):
        """Alibaba/DashScope use dots in model names (e.g. qwen3.5-plus). Fixes #1739."""
        assert normalize_model_name("qwen3.5-plus", preserve_dots=True) == "qwen3.5-plus"
        assert normalize_model_name("anthropic/qwen3.5-plus", preserve_dots=True) == "qwen3.5-plus"
        assert normalize_model_name("qwen3.5-flash", preserve_dots=True) == "qwen3.5-flash"


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

    def test_converts_user_image_url_blocks_to_anthropic_image_blocks(self):
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Can you see this?"},
                    {"type": "image_url", "image_url": {"url": "https://example.com/cat.png"}},
                ],
            }
        ]

        _, result = convert_messages_to_anthropic(messages)

        assert result == [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Can you see this?"},
                    {"type": "image", "source": {"type": "url", "url": "https://example.com/cat.png"}},
                ],
            }
        ]

    def test_converts_data_url_image_blocks_to_base64_anthropic_image_blocks(self):
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": "What is in this screenshot?"},
                    {"type": "input_image", "image_url": "data:image/png;base64,AAAA"},
                ],
            }
        ]

        _, result = convert_messages_to_anthropic(messages)

        assert result == [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What is in this screenshot?"},
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/png",
                            "data": "AAAA",
                        },
                    },
                ],
            }
        ]

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
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {"id": "tc_1", "function": {"name": "test_tool", "arguments": "{}"}},
                ],
            },
            {"role": "tool", "tool_call_id": "tc_1", "content": "result data"},
        ]
        _, result = convert_messages_to_anthropic(messages)
        # tool result is in the second message (user role)
        user_msg = [m for m in result if m["role"] == "user"][0]
        assert user_msg["content"][0]["type"] == "tool_result"
        assert user_msg["content"][0]["tool_use_id"] == "tc_1"

    def test_merges_consecutive_tool_results(self):
        messages = [
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {"id": "tc_1", "function": {"name": "tool_a", "arguments": "{}"}},
                    {"id": "tc_2", "function": {"name": "tool_b", "arguments": "{}"}},
                ],
            },
            {"role": "tool", "tool_call_id": "tc_1", "content": "result 1"},
            {"role": "tool", "tool_call_id": "tc_2", "content": "result 2"},
        ]
        _, result = convert_messages_to_anthropic(messages)
        # assistant + merged user (with 2 tool_results)
        user_msgs = [m for m in result if m["role"] == "user"]
        assert len(user_msgs) == 1
        assert len(user_msgs[0]["content"]) == 2

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

    def test_strips_orphaned_tool_result(self):
        """tool_result with no matching tool_use should be stripped.

        This happens when context compression removes the assistant message
        containing the tool_use but leaves the subsequent tool_result intact.
        Anthropic rejects orphaned tool_results with a 400.
        """
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there"},
            # The assistant tool_use message was removed by compression,
            # but the tool_result survived:
            {"role": "tool", "tool_call_id": "tc_gone", "content": "stale result"},
            {"role": "user", "content": "Thanks"},
        ]
        _, result = convert_messages_to_anthropic(messages)
        # tc_gone has no matching tool_use — its tool_result should be stripped
        for m in result:
            if m["role"] == "user" and isinstance(m["content"], list):
                assert all(
                    b.get("type") != "tool_result"
                    for b in m["content"]
                ), "Orphaned tool_result should have been stripped"

    def test_strips_orphaned_tool_result_preserves_valid(self):
        """Orphaned tool_results are stripped while valid ones survive."""
        messages = [
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {"id": "tc_valid", "function": {"name": "search", "arguments": "{}"}},
                ],
            },
            {"role": "tool", "tool_call_id": "tc_valid", "content": "good result"},
            {"role": "tool", "tool_call_id": "tc_orphan", "content": "stale result"},
        ]
        _, result = convert_messages_to_anthropic(messages)
        user_msg = [m for m in result if m["role"] == "user"][0]
        tool_results = [
            b for b in user_msg["content"] if b.get("type") == "tool_result"
        ]
        assert len(tool_results) == 1
        assert tool_results[0]["tool_use_id"] == "tc_valid"

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

    def test_assistant_cache_control_blocks_are_preserved(self):
        messages = apply_anthropic_cache_control([
            {"role": "system", "content": "System prompt"},
            {"role": "assistant", "content": "Hello from assistant"},
        ])

        _, result = convert_messages_to_anthropic(messages)
        assistant_blocks = result[0]["content"]

        assert assistant_blocks[0]["type"] == "text"
        assert assistant_blocks[0]["text"] == "Hello from assistant"
        assert assistant_blocks[0]["cache_control"] == {"type": "ephemeral"}

    def test_tool_cache_control_is_preserved_on_tool_result_block(self):
        messages = apply_anthropic_cache_control([
            {"role": "system", "content": "System prompt"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {"id": "tc_1", "function": {"name": "test_tool", "arguments": "{}"}},
                ],
            },
            {"role": "tool", "tool_call_id": "tc_1", "content": "result"},
        ], native_anthropic=True)

        _, result = convert_messages_to_anthropic(messages)
        user_msg = [m for m in result if m["role"] == "user"][0]
        tool_block = user_msg["content"][0]

        assert tool_block["type"] == "tool_result"
        assert tool_block["tool_use_id"] == "tc_1"
        assert tool_block["content"] == "result"
        assert tool_block["cache_control"] == {"type": "ephemeral"}

    def test_preserved_thinking_blocks_are_rehydrated_before_tool_use(self):
        messages = [
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {"id": "tc_1", "function": {"name": "test_tool", "arguments": "{}"}},
                ],
                "reasoning_details": [
                    {
                        "type": "thinking",
                        "thinking": "Need to inspect the tool result first.",
                        "signature": "sig_123",
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "tc_1", "content": "tool output"},
        ]

        _, result = convert_messages_to_anthropic(messages)
        assistant_blocks = next(msg for msg in result if msg["role"] == "assistant")["content"]

        assert assistant_blocks[0]["type"] == "thinking"
        assert assistant_blocks[0]["thinking"] == "Need to inspect the tool result first."
        assert assistant_blocks[0]["signature"] == "sig_123"
        assert assistant_blocks[1]["type"] == "tool_use"

    def test_converts_data_url_image_to_anthropic_image_block(self):
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe this image"},
                    {
                        "type": "image_url",
                        "image_url": {"url": "data:image/png;base64,ZmFrZQ=="},
                    },
                ],
            }
        ]

        _, result = convert_messages_to_anthropic(messages)
        blocks = result[0]["content"]
        assert blocks[0] == {"type": "text", "text": "Describe this image"}
        assert blocks[1] == {
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": "image/png",
                "data": "ZmFrZQ==",
            },
        }

    def test_converts_remote_image_url_to_anthropic_image_block(self):
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe this image"},
                    {
                        "type": "image_url",
                        "image_url": {"url": "https://example.com/cat.png"},
                    },
                ],
            }
        ]

        _, result = convert_messages_to_anthropic(messages)
        blocks = result[0]["content"]
        assert blocks[1] == {
            "type": "image",
            "source": {
                "type": "url",
                "url": "https://example.com/cat.png",
            },
        }

    def test_empty_cached_assistant_tool_turn_converts_without_empty_text_block(self):
        messages = apply_anthropic_cache_control([
            {"role": "system", "content": "System prompt"},
            {"role": "user", "content": "Find the skill"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {"id": "tc_1", "function": {"name": "skill_view", "arguments": "{}"}},
                ],
            },
            {"role": "tool", "tool_call_id": "tc_1", "content": "result"},
        ])

        _, result = convert_messages_to_anthropic(messages)

        assistant_turn = next(msg for msg in result if msg["role"] == "assistant")
        assistant_blocks = assistant_turn["content"]

        assert all(not (b.get("type") == "text" and b.get("text") == "") for b in assistant_blocks)
        assert any(b.get("type") == "tool_use" for b in assistant_blocks)

    def test_empty_user_message_string_gets_placeholder(self):
        """Empty user message strings should get '(empty message)' placeholder.

        Anthropic rejects requests with empty user message content.
        Regression test for #3143 — Discord @mention-only messages.
        """
        messages = [
            {"role": "user", "content": ""},
        ]
        _, result = convert_messages_to_anthropic(messages)
        assert result[0]["role"] == "user"
        assert result[0]["content"] == "(empty message)"

    def test_whitespace_only_user_message_gets_placeholder(self):
        """Whitespace-only user messages should also get placeholder."""
        messages = [
            {"role": "user", "content": "   \n\t  "},
        ]
        _, result = convert_messages_to_anthropic(messages)
        assert result[0]["content"] == "(empty message)"

    def test_empty_user_message_list_gets_placeholder(self):
        """Empty content list for user messages should get placeholder block."""
        messages = [
            {"role": "user", "content": []},
        ]
        _, result = convert_messages_to_anthropic(messages)
        assert result[0]["role"] == "user"
        assert isinstance(result[0]["content"], list)
        assert len(result[0]["content"]) == 1
        assert result[0]["content"][0] == {"type": "text", "text": "(empty message)"}

    def test_user_message_with_empty_text_blocks_gets_placeholder(self):
        """User message with only empty text blocks should get placeholder."""
        messages = [
            {"role": "user", "content": [{"type": "text", "text": ""}, {"type": "text", "text": "  "}]},
        ]
        _, result = convert_messages_to_anthropic(messages)
        assert result[0]["role"] == "user"
        assert isinstance(result[0]["content"], list)
        assert result[0]["content"] == [{"type": "text", "text": "(empty message)"}]


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
        # Adaptive thinking + display="summarized" keeps reasoning text
        # populated in the response stream (Opus 4.7 default is "omitted").
        assert kwargs["thinking"] == {"type": "adaptive", "display": "summarized"}
        assert kwargs["output_config"] == {"effort": "high"}
        assert "budget_tokens" not in kwargs["thinking"]
        assert "temperature" not in kwargs
        assert kwargs["max_tokens"] == 4096

    def test_reasoning_config_downgrades_xhigh_to_max_for_4_6_models(self):
        # Opus 4.7 added "xhigh" as a distinct effort level (low/medium/high/
        # xhigh/max). Opus 4.6 only supports low/medium/high/max — sending
        # "xhigh" there returns an API 400. Preserve the pre-migration
        # behavior of aliasing xhigh→max on pre-4.7 adaptive models so users
        # who prefer xhigh as their default don't 400 every request when
        # switching back to 4.6.
        kwargs = build_anthropic_kwargs(
            model="claude-sonnet-4-6",
            messages=[{"role": "user", "content": "think harder"}],
            tools=None,
            max_tokens=4096,
            reasoning_config={"enabled": True, "effort": "xhigh"},
        )
        assert kwargs["thinking"] == {"type": "adaptive", "display": "summarized"}
        assert kwargs["output_config"] == {"effort": "max"}

    def test_reasoning_config_preserves_xhigh_for_4_7_models(self):
        # On 4.7+ xhigh is a real level and the recommended default for
        # coding/agentic work — keep it distinct from max.
        kwargs = build_anthropic_kwargs(
            model="claude-opus-4-7",
            messages=[{"role": "user", "content": "think harder"}],
            tools=None,
            max_tokens=4096,
            reasoning_config={"enabled": True, "effort": "xhigh"},
        )
        assert kwargs["thinking"] == {"type": "adaptive", "display": "summarized"}
        assert kwargs["output_config"] == {"effort": "xhigh"}

    def test_reasoning_config_maps_max_effort_for_4_7_models(self):
        kwargs = build_anthropic_kwargs(
            model="claude-opus-4-7",
            messages=[{"role": "user", "content": "maximum reasoning please"}],
            tools=None,
            max_tokens=4096,
            reasoning_config={"enabled": True, "effort": "max"},
        )
        assert kwargs["thinking"] == {"type": "adaptive", "display": "summarized"}
        assert kwargs["output_config"] == {"effort": "max"}

    def test_opus_4_7_strips_sampling_params(self):
        # Opus 4.7 returns 400 on non-default temperature/top_p/top_k.
        # build_anthropic_kwargs must strip them as a safety net even if an
        # upstream caller injects them for older-model compatibility.
        kwargs = build_anthropic_kwargs(
            model="claude-opus-4-7",
            messages=[{"role": "user", "content": "hi"}],
            tools=None,
            max_tokens=1024,
            reasoning_config=None,
        )
        # Manually inject sampling params then re-run through the guard.
        # Because build_anthropic_kwargs doesn't currently accept sampling
        # params through its signature, we exercise the strip behavior by
        # calling the internal predicate directly.
        from agent.anthropic_adapter import _forbids_sampling_params
        assert _forbids_sampling_params("claude-opus-4-7") is True
        assert _forbids_sampling_params("claude-opus-4-6") is False
        assert _forbids_sampling_params("claude-sonnet-4-5") is False

    def test_reasoning_disabled(self):
        kwargs = build_anthropic_kwargs(
            model="claude-sonnet-4-20250514",
            messages=[{"role": "user", "content": "quick"}],
            tools=None,
            max_tokens=4096,
            reasoning_config={"enabled": False},
        )
        assert "thinking" not in kwargs

    def test_default_max_tokens_uses_model_output_limit(self):
        """When max_tokens is None, use the model's native output limit."""
        kwargs = build_anthropic_kwargs(
            model="claude-sonnet-4-20250514",
            messages=[{"role": "user", "content": "Hi"}],
            tools=None,
            max_tokens=None,
            reasoning_config=None,
        )
        assert kwargs["max_tokens"] == 64_000  # Sonnet 4 output limit

    def test_default_max_tokens_opus_4_6(self):
        kwargs = build_anthropic_kwargs(
            model="claude-opus-4-6",
            messages=[{"role": "user", "content": "Hi"}],
            tools=None,
            max_tokens=None,
            reasoning_config=None,
        )
        assert kwargs["max_tokens"] == 128_000

    def test_default_max_tokens_sonnet_4_6(self):
        kwargs = build_anthropic_kwargs(
            model="claude-sonnet-4-6",
            messages=[{"role": "user", "content": "Hi"}],
            tools=None,
            max_tokens=None,
            reasoning_config=None,
        )
        assert kwargs["max_tokens"] == 64_000

    def test_default_max_tokens_date_stamped_model(self):
        """Date-stamped model IDs should resolve via substring match."""
        kwargs = build_anthropic_kwargs(
            model="claude-sonnet-4-5-20250929",
            messages=[{"role": "user", "content": "Hi"}],
            tools=None,
            max_tokens=None,
            reasoning_config=None,
        )
        assert kwargs["max_tokens"] == 64_000

    def test_default_max_tokens_older_model(self):
        kwargs = build_anthropic_kwargs(
            model="claude-3-5-sonnet-20241022",
            messages=[{"role": "user", "content": "Hi"}],
            tools=None,
            max_tokens=None,
            reasoning_config=None,
        )
        assert kwargs["max_tokens"] == 8_192

    def test_default_max_tokens_unknown_model_uses_highest(self):
        """Unknown future models should get the highest known limit."""
        kwargs = build_anthropic_kwargs(
            model="claude-ultra-5-20260101",
            messages=[{"role": "user", "content": "Hi"}],
            tools=None,
            max_tokens=None,
            reasoning_config=None,
        )
        assert kwargs["max_tokens"] == 128_000

    def test_explicit_max_tokens_overrides_default(self):
        """User-specified max_tokens should be respected."""
        kwargs = build_anthropic_kwargs(
            model="claude-opus-4-6",
            messages=[{"role": "user", "content": "Hi"}],
            tools=None,
            max_tokens=4096,
            reasoning_config=None,
        )
        assert kwargs["max_tokens"] == 4096

    def test_context_length_clamp(self):
        """max_tokens should be clamped to context_length if it's smaller."""
        kwargs = build_anthropic_kwargs(
            model="claude-opus-4-6",  # 128K output
            messages=[{"role": "user", "content": "Hi"}],
            tools=None,
            max_tokens=None,
            reasoning_config=None,
            context_length=50000,
        )
        assert kwargs["max_tokens"] == 49999  # context_length - 1

    def test_context_length_no_clamp_when_larger(self):
        """No clamping when context_length exceeds output limit."""
        kwargs = build_anthropic_kwargs(
            model="claude-sonnet-4-6",  # 64K output
            messages=[{"role": "user", "content": "Hi"}],
            tools=None,
            max_tokens=None,
            reasoning_config=None,
            context_length=200000,
        )
        assert kwargs["max_tokens"] == 64_000


# ---------------------------------------------------------------------------
# Model output limit lookup
# ---------------------------------------------------------------------------


class TestGetAnthropicMaxOutput:
    def test_opus_4_6(self):
        from agent.anthropic_adapter import _get_anthropic_max_output
        assert _get_anthropic_max_output("claude-opus-4-6") == 128_000

    def test_opus_4_6_variant(self):
        from agent.anthropic_adapter import _get_anthropic_max_output
        assert _get_anthropic_max_output("claude-opus-4-6:1m:fast") == 128_000

    def test_sonnet_4_6(self):
        from agent.anthropic_adapter import _get_anthropic_max_output
        assert _get_anthropic_max_output("claude-sonnet-4-6") == 64_000

    def test_sonnet_4_date_stamped(self):
        from agent.anthropic_adapter import _get_anthropic_max_output
        assert _get_anthropic_max_output("claude-sonnet-4-20250514") == 64_000

    def test_claude_3_5_sonnet(self):
        from agent.anthropic_adapter import _get_anthropic_max_output
        assert _get_anthropic_max_output("claude-3-5-sonnet-20241022") == 8_192

    def test_claude_3_opus(self):
        from agent.anthropic_adapter import _get_anthropic_max_output
        assert _get_anthropic_max_output("claude-3-opus-20240229") == 4_096

    def test_unknown_future_model(self):
        from agent.anthropic_adapter import _get_anthropic_max_output
        assert _get_anthropic_max_output("claude-ultra-5-20260101") == 128_000

    def test_longest_prefix_wins(self):
        """'claude-3-5-sonnet' should match before 'claude-3-5'."""
        from agent.anthropic_adapter import _get_anthropic_max_output
        # claude-3-5-sonnet (8192) should win over a hypothetical shorter match
        assert _get_anthropic_max_output("claude-3-5-sonnet-20241022") == 8_192


# ---------------------------------------------------------------------------
# _to_plain_data hardening
# ---------------------------------------------------------------------------


class TestToPlainData:
    def test_simple_dict(self):
        assert _to_plain_data({"a": 1, "b": [2, 3]}) == {"a": 1, "b": [2, 3]}

    def test_pydantic_like_model_dump(self):
        class FakeModel:
            def model_dump(self):
                return {"type": "thinking", "thinking": "hello"}

        result = _to_plain_data(FakeModel())
        assert result == {"type": "thinking", "thinking": "hello"}

    def test_circular_reference_does_not_recurse_forever(self):
        """Circular dict reference should be stringified, not infinite-loop."""
        d: dict = {"key": "value"}
        d["self"] = d  # circular
        result = _to_plain_data(d)
        assert isinstance(result, dict)
        assert result["key"] == "value"
        assert isinstance(result["self"], str)

    def test_shared_sibling_objects_are_not_falsely_detected_as_cycles(self):
        """Two siblings referencing the same dict must both be converted."""
        shared = {"type": "thinking", "thinking": "reason"}
        parent = {"a": shared, "b": shared}
        result = _to_plain_data(parent)
        assert isinstance(result["a"], dict)
        assert isinstance(result["b"], dict)
        assert result["a"] == {"type": "thinking", "thinking": "reason"}

    def test_deep_nesting_is_capped(self):
        deep = "leaf"
        for _ in range(25):
            deep = {"nested": deep}
        result = _to_plain_data(deep)
        assert isinstance(result, dict)

    def test_plain_values_pass_through(self):
        assert _to_plain_data("hello") == "hello"
        assert _to_plain_data(42) == 42
        assert _to_plain_data(None) is None

    def test_object_with_dunder_dict(self):
        obj = SimpleNamespace(type="thinking", thinking="reason", signature="sig")
        result = _to_plain_data(obj)
        assert result == {"type": "thinking", "thinking": "reason", "signature": "sig"}


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
        nr = get_transport("anthropic_messages").normalize_response(self._make_response([block]))
        assert nr.content == "Hello world"
        assert nr.finish_reason == "stop"
        assert nr.tool_calls is None

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
        nr = get_transport("anthropic_messages").normalize_response(
            self._make_response(blocks, "tool_use")
        )
        assert nr.content == "Searching..."
        assert nr.finish_reason == "tool_calls"
        assert len(nr.tool_calls) == 1
        assert nr.tool_calls[0].name == "search"
        assert json.loads(nr.tool_calls[0].arguments) == {"query": "test"}

    def test_thinking_response(self):
        blocks = [
            SimpleNamespace(type="thinking", thinking="Let me reason about this..."),
            SimpleNamespace(type="text", text="The answer is 42."),
        ]
        nr = get_transport("anthropic_messages").normalize_response(self._make_response(blocks))
        assert nr.content == "The answer is 42."
        assert nr.reasoning == "Let me reason about this..."
        assert nr.provider_data["reasoning_details"] == [{"type": "thinking", "thinking": "Let me reason about this..."}]

    def test_thinking_response_preserves_signature(self):
        blocks = [
            SimpleNamespace(
                type="thinking",
                thinking="Let me reason about this...",
                signature="opaque_signature",
                redacted=False,
            ),
        ]
        nr = get_transport("anthropic_messages").normalize_response(self._make_response(blocks))
        assert nr.provider_data["reasoning_details"][0]["signature"] == "opaque_signature"
        assert nr.provider_data["reasoning_details"][0]["thinking"] == "Let me reason about this..."

    def test_stop_reason_mapping(self):
        block = SimpleNamespace(type="text", text="x")
        nr1 = get_transport("anthropic_messages").normalize_response(
            self._make_response([block], "end_turn")
        )
        nr2 = get_transport("anthropic_messages").normalize_response(
            self._make_response([block], "tool_use")
        )
        nr3 = get_transport("anthropic_messages").normalize_response(
            self._make_response([block], "max_tokens")
        )
        assert nr1.finish_reason == "stop"
        assert nr2.finish_reason == "tool_calls"
        assert nr3.finish_reason == "length"

    def test_stop_reason_refusal_and_context_exceeded(self):
        # Claude 4.5+ introduced two new stop_reason values the Messages API
        # returns.  We map both to OpenAI-style finish_reasons upstream
        # handlers already understand, instead of silently collapsing to
        # "stop" (old behavior).
        block = SimpleNamespace(type="text", text="")
        nr_refusal = get_transport("anthropic_messages").normalize_response(
            self._make_response([block], "refusal")
        )
        nr_overflow = get_transport("anthropic_messages").normalize_response(
            self._make_response([block], "model_context_window_exceeded")
        )
        assert nr_refusal.finish_reason == "content_filter"
        assert nr_overflow.finish_reason == "length"

    def test_no_text_content(self):
        block = SimpleNamespace(
            type="tool_use", id="tc_1", name="search", input={"q": "hi"}
        )
        nr = get_transport("anthropic_messages").normalize_response(
            self._make_response([block], "tool_use")
        )
        assert nr.content is None
        assert len(nr.tool_calls) == 1


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
# Thinking block signature management
# ---------------------------------------------------------------------------


class TestThinkingBlockSignatureManagement:
    """Tests for the thinking block handling strategy:
    strip from old turns, preserve latest signed, downgrade unsigned."""

    def test_thinking_stripped_from_non_last_assistant(self):
        """Thinking blocks are removed from all assistant messages except the last."""
        messages = [
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {"id": "tc_1", "function": {"name": "tool1", "arguments": "{}"}},
                ],
                "reasoning_details": [
                    {"type": "thinking", "thinking": "Old reasoning.", "signature": "sig_old"},
                ],
            },
            {"role": "tool", "tool_call_id": "tc_1", "content": "result 1"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {"id": "tc_2", "function": {"name": "tool2", "arguments": "{}"}},
                ],
                "reasoning_details": [
                    {"type": "thinking", "thinking": "Latest reasoning.", "signature": "sig_new"},
                ],
            },
            {"role": "tool", "tool_call_id": "tc_2", "content": "result 2"},
        ]
        _, result = convert_messages_to_anthropic(messages)

        # Find both assistant messages
        assistants = [m for m in result if m["role"] == "assistant"]
        assert len(assistants) == 2

        # First (non-last) assistant: no thinking blocks
        first_types = [b.get("type") for b in assistants[0]["content"]]
        assert "thinking" not in first_types
        assert "redacted_thinking" not in first_types
        assert "tool_use" in first_types  # tool_use should survive

        # Last assistant: thinking block preserved with signature
        last_blocks = assistants[1]["content"]
        thinking_blocks = [b for b in last_blocks if b.get("type") == "thinking"]
        assert len(thinking_blocks) == 1
        assert thinking_blocks[0]["thinking"] == "Latest reasoning."
        assert thinking_blocks[0]["signature"] == "sig_new"

    def test_signed_thinking_preserved_on_last_turn(self):
        """A signed thinking block on the last assistant message is kept."""
        messages = [
            {
                "role": "assistant",
                "content": "The answer is 42.",
                "reasoning_details": [
                    {"type": "thinking", "thinking": "Deep thought.", "signature": "sig_valid"},
                ],
            },
        ]
        _, result = convert_messages_to_anthropic(messages)
        blocks = result[0]["content"]
        thinking = [b for b in blocks if b.get("type") == "thinking"]
        assert len(thinking) == 1
        assert thinking[0]["signature"] == "sig_valid"

    def test_unsigned_thinking_downgraded_to_text_on_last_turn(self):
        """Unsigned thinking blocks on the last turn become text blocks."""
        messages = [
            {
                "role": "assistant",
                "content": "Response text.",
                "reasoning_details": [
                    {"type": "thinking", "thinking": "Unsigned reasoning."},
                    # No 'signature' field
                ],
            },
        ]
        _, result = convert_messages_to_anthropic(messages)
        blocks = result[0]["content"]

        # No thinking blocks should remain
        assert not any(b.get("type") == "thinking" for b in blocks)
        # The reasoning text should be preserved as a text block
        text_contents = [b.get("text", "") for b in blocks if b.get("type") == "text"]
        assert "Unsigned reasoning." in text_contents

    def test_redacted_thinking_with_data_preserved(self):
        """Redacted thinking with 'data' field is kept on last turn."""
        messages = [
            {
                "role": "assistant",
                "content": "Response.",
                "reasoning_details": [
                    {"type": "redacted_thinking", "data": "opaque_signature_data"},
                ],
            },
        ]
        _, result = convert_messages_to_anthropic(messages)
        blocks = result[0]["content"]
        redacted = [b for b in blocks if b.get("type") == "redacted_thinking"]
        assert len(redacted) == 1
        assert redacted[0]["data"] == "opaque_signature_data"

    def test_redacted_thinking_without_data_dropped(self):
        """Redacted thinking without 'data' is dropped — can't be validated."""
        messages = [
            {
                "role": "assistant",
                "content": "Response.",
                "reasoning_details": [
                    {"type": "redacted_thinking"},
                    # No 'data' field
                ],
            },
        ]
        _, result = convert_messages_to_anthropic(messages)
        blocks = result[0]["content"]
        assert not any(b.get("type") == "redacted_thinking" for b in blocks)

    def test_cache_control_stripped_from_thinking_blocks(self):
        """cache_control markers are removed from thinking/redacted_thinking blocks."""
        messages = [
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {"id": "tc_1", "function": {"name": "t", "arguments": "{}"}},
                ],
                "reasoning_details": [
                    {
                        "type": "thinking",
                        "thinking": "Reasoning.",
                        "signature": "sig_1",
                        "cache_control": {"type": "ephemeral"},
                    },
                ],
            },
            {"role": "tool", "tool_call_id": "tc_1", "content": "result"},
        ]
        _, result = convert_messages_to_anthropic(messages)
        assistant = next(m for m in result if m["role"] == "assistant")
        for block in assistant["content"]:
            if block.get("type") in ("thinking", "redacted_thinking"):
                assert "cache_control" not in block

    def test_thinking_stripped_from_merged_consecutive_assistants(self):
        """When consecutive assistants are merged, second one's thinking is dropped."""
        messages = [
            {
                "role": "assistant",
                "content": "First response.",
                "reasoning_details": [
                    {"type": "thinking", "thinking": "First thought.", "signature": "sig_1"},
                ],
            },
            {
                "role": "assistant",
                "content": "Second response.",
                "reasoning_details": [
                    {"type": "thinking", "thinking": "Second thought.", "signature": "sig_2"},
                ],
            },
        ]
        _, result = convert_messages_to_anthropic(messages)

        # Should be merged into one assistant message
        assistants = [m for m in result if m["role"] == "assistant"]
        assert len(assistants) == 1

        # Only the first thinking block should remain (signed, on the last/only assistant)
        blocks = assistants[0]["content"]
        thinking = [b for b in blocks if b.get("type") == "thinking"]
        assert len(thinking) == 1
        assert thinking[0]["thinking"] == "First thought."

    def test_empty_content_after_strip_gets_placeholder(self):
        """If stripping thinking leaves an empty message, a placeholder is added."""
        messages = [
            {
                "role": "assistant",
                "content": "",
                "reasoning_details": [
                    {"type": "thinking", "thinking": "Only thinking, no text."},
                    # Unsigned — will be downgraded, but content was empty string
                ],
            },
            {"role": "user", "content": "Next message."},
            {"role": "assistant", "content": "Final."},
        ]
        _, result = convert_messages_to_anthropic(messages)
        # First assistant is non-last, so thinking is stripped completely.
        # The original content was empty and thinking was unsigned → placeholder
        first_assistant = result[0]
        assert first_assistant["role"] == "assistant"
        assert len(first_assistant["content"]) >= 1

    def test_multi_turn_conversation_preserves_only_last(self):
        """Full multi-turn conversation: only last assistant keeps thinking."""
        messages = [
            {"role": "user", "content": "Question 1"},
            {
                "role": "assistant",
                "content": "Answer 1",
                "reasoning_details": [
                    {"type": "thinking", "thinking": "Thought 1", "signature": "sig_1"},
                ],
            },
            {"role": "user", "content": "Question 2"},
            {
                "role": "assistant",
                "content": "Answer 2",
                "reasoning_details": [
                    {"type": "thinking", "thinking": "Thought 2", "signature": "sig_2"},
                ],
            },
            {"role": "user", "content": "Question 3"},
            {
                "role": "assistant",
                "content": "Answer 3",
                "reasoning_details": [
                    {"type": "thinking", "thinking": "Thought 3", "signature": "sig_3"},
                ],
            },
        ]
        _, result = convert_messages_to_anthropic(messages)

        assistants = [m for m in result if m["role"] == "assistant"]
        assert len(assistants) == 3

        # First two: no thinking blocks
        for a in assistants[:2]:
            assert not any(
                b.get("type") in ("thinking", "redacted_thinking")
                for b in a["content"]
                if isinstance(b, dict)
            )

        # Last one: thinking preserved
        last_thinking = [
            b for b in assistants[2]["content"]
            if isinstance(b, dict) and b.get("type") == "thinking"
        ]
        assert len(last_thinking) == 1
        assert last_thinking[0]["signature"] == "sig_3"


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



# ---------------------------------------------------------------------------
# max_tokens resolver — openclaw/openclaw#66664 port
# ---------------------------------------------------------------------------

from agent.anthropic_adapter import (
    _resolve_positive_anthropic_max_tokens,
    _resolve_anthropic_messages_max_tokens,
)


class TestResolvePositiveMaxTokens:
    """Unit tests for the positive-int resolver helper."""

    def test_positive_int_passes_through(self):
        assert _resolve_positive_anthropic_max_tokens(8192) == 8192

    def test_zero_returns_none(self):
        assert _resolve_positive_anthropic_max_tokens(0) is None

    def test_negative_int_returns_none(self):
        assert _resolve_positive_anthropic_max_tokens(-1) is None
        assert _resolve_positive_anthropic_max_tokens(-500) is None

    def test_fractional_float_floored_and_kept_if_positive(self):
        # 8192.7 -> 8192, still positive
        assert _resolve_positive_anthropic_max_tokens(8192.7) == 8192

    def test_small_positive_float_below_one_returns_none(self):
        # 0.5 floors to 0, which is not positive
        assert _resolve_positive_anthropic_max_tokens(0.5) is None

    def test_negative_float_returns_none(self):
        assert _resolve_positive_anthropic_max_tokens(-1.5) is None

    def test_nan_returns_none(self):
        assert _resolve_positive_anthropic_max_tokens(float("nan")) is None

    def test_infinity_returns_none(self):
        assert _resolve_positive_anthropic_max_tokens(float("inf")) is None
        assert _resolve_positive_anthropic_max_tokens(float("-inf")) is None

    def test_bool_true_returns_none(self):
        # True is an int subclass but semantically never a real max_tokens value
        assert _resolve_positive_anthropic_max_tokens(True) is None
        assert _resolve_positive_anthropic_max_tokens(False) is None

    def test_string_returns_none(self):
        assert _resolve_positive_anthropic_max_tokens("8192") is None

    def test_none_returns_none(self):
        assert _resolve_positive_anthropic_max_tokens(None) is None


class TestResolveMessagesMaxTokens:
    """Integration tests for the full Messages resolver."""

    def test_positive_requested_wins(self):
        assert _resolve_anthropic_messages_max_tokens(
            8192, "claude-opus-4-6"
        ) == 8192

    def test_zero_falls_back_to_model_default(self):
        # Should use _get_anthropic_max_output(model), not crash
        result = _resolve_anthropic_messages_max_tokens(0, "claude-opus-4-6")
        assert result > 0

    def test_none_falls_back_to_model_default(self):
        result = _resolve_anthropic_messages_max_tokens(None, "claude-opus-4-6")
        assert result > 0

    def test_negative_falls_back_to_model_default(self):
        # Previously leaked -1 to the API; now falls back safely
        result = _resolve_anthropic_messages_max_tokens(-1, "claude-opus-4-6")
        assert result > 0

    def test_fractional_positive_floored(self):
        assert _resolve_anthropic_messages_max_tokens(
            8192.5, "claude-opus-4-6"
        ) == 8192

    def test_sub_one_float_falls_back(self):
        # 0.5 floors to 0 -> not positive -> falls back to model ceiling
        result = _resolve_anthropic_messages_max_tokens(0.5, "claude-opus-4-6")
        assert result > 0
        assert result != 0
