"""Tests for agent.auxiliary_client resolution chain, provider overrides, and model overrides."""

import json
import os
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from agent.auxiliary_client import (
    get_text_auxiliary_client,
    get_available_vision_backends,
    resolve_vision_provider_client,
    resolve_provider_client,
    auxiliary_max_tokens_param,
    call_llm,
    _read_codex_access_token,
    _get_auxiliary_provider,
    _get_provider_chain,
    _is_payment_error,
    _try_payment_fallback,
    _resolve_auto,
)


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    """Strip provider env vars so each test starts clean."""
    for key in (
        "OPENROUTER_API_KEY", "OPENAI_BASE_URL", "OPENAI_API_KEY",
        "OPENAI_MODEL", "LLM_MODEL", "NOUS_INFERENCE_BASE_URL",
        "ANTHROPIC_API_KEY", "ANTHROPIC_TOKEN", "CLAUDE_CODE_OAUTH_TOKEN",
        # Per-task provider/model/direct-endpoint overrides
        "AUXILIARY_VISION_PROVIDER", "AUXILIARY_VISION_MODEL",
        "AUXILIARY_VISION_BASE_URL", "AUXILIARY_VISION_API_KEY",
        "AUXILIARY_WEB_EXTRACT_PROVIDER", "AUXILIARY_WEB_EXTRACT_MODEL",
        "AUXILIARY_WEB_EXTRACT_BASE_URL", "AUXILIARY_WEB_EXTRACT_API_KEY",
        "CONTEXT_COMPRESSION_PROVIDER", "CONTEXT_COMPRESSION_MODEL",
    ):
        monkeypatch.delenv(key, raising=False)


@pytest.fixture
def codex_auth_dir(tmp_path, monkeypatch):
    """Provide a writable ~/.codex/ directory with a valid auth.json."""
    codex_dir = tmp_path / ".codex"
    codex_dir.mkdir()
    auth_file = codex_dir / "auth.json"
    auth_file.write_text(json.dumps({
        "tokens": {
            "access_token": "codex-test-token-abc123",
            "refresh_token": "codex-refresh-xyz",
        }
    }))
    monkeypatch.setattr(
        "agent.auxiliary_client._read_codex_access_token",
        lambda: "codex-test-token-abc123",
    )
    return codex_dir


class TestReadCodexAccessToken:
    def test_valid_auth_store(self, tmp_path, monkeypatch):
        hermes_home = tmp_path / "hermes"
        hermes_home.mkdir(parents=True, exist_ok=True)
        (hermes_home / "auth.json").write_text(json.dumps({
            "version": 1,
            "providers": {
                "openai-codex": {
                    "tokens": {"access_token": "tok-123", "refresh_token": "r-456"},
                },
            },
        }))
        monkeypatch.setenv("HERMES_HOME", str(hermes_home))
        result = _read_codex_access_token()
        assert result == "tok-123"

    def test_pool_without_selected_entry_falls_back_to_auth_store(self, tmp_path, monkeypatch):
        hermes_home = tmp_path / "hermes"
        hermes_home.mkdir(parents=True, exist_ok=True)
        monkeypatch.setenv("HERMES_HOME", str(hermes_home))

        valid_jwt = "eyJhbGciOiJSUzI1NiJ9.eyJleHAiOjk5OTk5OTk5OTl9.sig"
        with patch("agent.auxiliary_client._select_pool_entry", return_value=(True, None)), \
             patch("hermes_cli.auth._read_codex_tokens", return_value={
                 "tokens": {"access_token": valid_jwt, "refresh_token": "refresh"}
             }):
            result = _read_codex_access_token()

        assert result == valid_jwt

    def test_missing_returns_none(self, tmp_path, monkeypatch):
        hermes_home = tmp_path / "hermes"
        hermes_home.mkdir(parents=True, exist_ok=True)
        (hermes_home / "auth.json").write_text(json.dumps({"version": 1, "providers": {}}))
        monkeypatch.setenv("HERMES_HOME", str(hermes_home))
        result = _read_codex_access_token()
        assert result is None

    def test_empty_token_returns_none(self, tmp_path, monkeypatch):
        hermes_home = tmp_path / "hermes"
        hermes_home.mkdir(parents=True, exist_ok=True)
        (hermes_home / "auth.json").write_text(json.dumps({
            "version": 1,
            "providers": {
                "openai-codex": {
                    "tokens": {"access_token": "  ", "refresh_token": "r"},
                },
            },
        }))
        monkeypatch.setenv("HERMES_HOME", str(hermes_home))
        result = _read_codex_access_token()
        assert result is None

    def test_malformed_json_returns_none(self, tmp_path):
        codex_dir = tmp_path / ".codex"
        codex_dir.mkdir()
        (codex_dir / "auth.json").write_text("{bad json")
        with patch("agent.auxiliary_client.Path.home", return_value=tmp_path):
            result = _read_codex_access_token()
        assert result is None

    def test_missing_tokens_key_returns_none(self, tmp_path):
        codex_dir = tmp_path / ".codex"
        codex_dir.mkdir()
        (codex_dir / "auth.json").write_text(json.dumps({"other": "data"}))
        with patch("agent.auxiliary_client.Path.home", return_value=tmp_path):
            result = _read_codex_access_token()
        assert result is None


    def test_expired_jwt_returns_none(self, tmp_path, monkeypatch):
        """Expired JWT tokens should be skipped so auto chain continues."""
        import base64
        import time as _time

        # Build a JWT with exp in the past
        header = base64.urlsafe_b64encode(b'{"alg":"RS256","typ":"JWT"}').rstrip(b"=").decode()
        payload_data = json.dumps({"exp": int(_time.time()) - 3600}).encode()
        payload = base64.urlsafe_b64encode(payload_data).rstrip(b"=").decode()
        expired_jwt = f"{header}.{payload}.fakesig"

        hermes_home = tmp_path / "hermes"
        hermes_home.mkdir(parents=True, exist_ok=True)
        (hermes_home / "auth.json").write_text(json.dumps({
            "version": 1,
            "providers": {
                "openai-codex": {
                    "tokens": {"access_token": expired_jwt, "refresh_token": "r"},
                },
            },
        }))
        monkeypatch.setenv("HERMES_HOME", str(hermes_home))
        result = _read_codex_access_token()
        assert result is None, "Expired JWT should return None"

    def test_valid_jwt_returns_token(self, tmp_path, monkeypatch):
        """Non-expired JWT tokens should be returned."""
        import base64
        import time as _time

        header = base64.urlsafe_b64encode(b'{"alg":"RS256","typ":"JWT"}').rstrip(b"=").decode()
        payload_data = json.dumps({"exp": int(_time.time()) + 3600}).encode()
        payload = base64.urlsafe_b64encode(payload_data).rstrip(b"=").decode()
        valid_jwt = f"{header}.{payload}.fakesig"

        hermes_home = tmp_path / "hermes"
        hermes_home.mkdir(parents=True, exist_ok=True)
        (hermes_home / "auth.json").write_text(json.dumps({
            "version": 1,
            "providers": {
                "openai-codex": {
                    "tokens": {"access_token": valid_jwt, "refresh_token": "r"},
                },
            },
        }))
        monkeypatch.setenv("HERMES_HOME", str(hermes_home))
        result = _read_codex_access_token()
        assert result == valid_jwt

    def test_non_jwt_token_passes_through(self, tmp_path, monkeypatch):
        """Non-JWT tokens (no dots) should be returned as-is."""
        hermes_home = tmp_path / "hermes"
        hermes_home.mkdir(parents=True, exist_ok=True)
        (hermes_home / "auth.json").write_text(json.dumps({
            "version": 1,
            "providers": {
                "openai-codex": {
                    "tokens": {"access_token": "plain-token-no-jwt", "refresh_token": "r"},
                },
            },
        }))
        monkeypatch.setenv("HERMES_HOME", str(hermes_home))
        result = _read_codex_access_token()
        assert result == "plain-token-no-jwt"


class TestAnthropicOAuthFlag:
    """Test that OAuth tokens get is_oauth=True in auxiliary Anthropic client."""

    def test_oauth_token_sets_flag(self, monkeypatch):
        """OAuth tokens (sk-ant-oat01-*) should create client with is_oauth=True."""
        monkeypatch.setenv("ANTHROPIC_TOKEN", "sk-ant-oat01-test-token")
        with patch("agent.anthropic_adapter.build_anthropic_client") as mock_build:
            mock_build.return_value = MagicMock()
            from agent.auxiliary_client import _try_anthropic, AnthropicAuxiliaryClient
            client, model = _try_anthropic()
            assert client is not None
            assert isinstance(client, AnthropicAuxiliaryClient)
            # The adapter inside should have is_oauth=True
            adapter = client.chat.completions
            assert adapter._is_oauth is True

    def test_api_key_no_oauth_flag(self, monkeypatch):
        """Regular API keys (sk-ant-api-*) should create client with is_oauth=False."""
        with patch("agent.anthropic_adapter.resolve_anthropic_token", return_value="sk-ant-api03-testkey1234"), \
             patch("agent.anthropic_adapter.build_anthropic_client") as mock_build, \
             patch("agent.auxiliary_client._select_pool_entry", return_value=(False, None)):
            mock_build.return_value = MagicMock()
            from agent.auxiliary_client import _try_anthropic, AnthropicAuxiliaryClient
            client, model = _try_anthropic()
            assert client is not None
            assert isinstance(client, AnthropicAuxiliaryClient)
            adapter = client.chat.completions
            assert adapter._is_oauth is False

    def test_pool_entry_takes_priority_over_legacy_resolution(self):
        class _Entry:
            access_token = "sk-ant-oat01-pooled"
            base_url = "https://api.anthropic.com"

        class _Pool:
            def has_credentials(self):
                return True

            def select(self):
                return _Entry()

        with (
            patch("agent.auxiliary_client.load_pool", return_value=_Pool()),
            patch("agent.anthropic_adapter.resolve_anthropic_token", side_effect=AssertionError("legacy path should not run")),
            patch("agent.anthropic_adapter.build_anthropic_client", return_value=MagicMock()) as mock_build,
        ):
            from agent.auxiliary_client import _try_anthropic

            client, model = _try_anthropic()

        assert client is not None
        assert model == "claude-haiku-4-5-20251001"
        assert mock_build.call_args.args[0] == "sk-ant-oat01-pooled"


class TestTryCodex:
    def test_pool_without_selected_entry_falls_back_to_auth_store(self):
        with (
            patch("agent.auxiliary_client._select_pool_entry", return_value=(True, None)),
            patch("agent.auxiliary_client._read_codex_access_token", return_value="codex-auth-token"),
            patch("agent.auxiliary_client.OpenAI") as mock_openai,
        ):
            mock_openai.return_value = MagicMock()
            from agent.auxiliary_client import _try_codex

            client, model = _try_codex()

        assert client is not None
        assert model == "gpt-5.2-codex"
        assert mock_openai.call_args.kwargs["api_key"] == "codex-auth-token"
        assert mock_openai.call_args.kwargs["base_url"] == "https://chatgpt.com/backend-api/codex"


class TestExpiredCodexFallback:
    """Test that expired Codex tokens don't block the auto chain."""

    def test_expired_codex_falls_through_to_next(self, tmp_path, monkeypatch):
        """When Codex token is expired, auto chain should skip it and try next provider."""
        import base64
        import time as _time

        # Expired Codex JWT
        header = base64.urlsafe_b64encode(b'{"alg":"RS256","typ":"JWT"}').rstrip(b"=").decode()
        payload_data = json.dumps({"exp": int(_time.time()) - 3600}).encode()
        payload = base64.urlsafe_b64encode(payload_data).rstrip(b"=").decode()
        expired_jwt = f"{header}.{payload}.fakesig"

        hermes_home = tmp_path / "hermes"
        hermes_home.mkdir(parents=True, exist_ok=True)
        (hermes_home / "auth.json").write_text(json.dumps({
            "version": 1,
            "providers": {
                "openai-codex": {
                    "tokens": {"access_token": expired_jwt, "refresh_token": "r"},
                },
            },
        }))
        monkeypatch.setenv("HERMES_HOME", str(hermes_home))

        # Set up Anthropic as fallback
        monkeypatch.setenv("ANTHROPIC_TOKEN", "sk-ant-oat01-test-fallback")
        with patch("agent.anthropic_adapter.build_anthropic_client") as mock_build:
            mock_build.return_value = MagicMock()
            from agent.auxiliary_client import _resolve_auto, AnthropicAuxiliaryClient
            client, model = _resolve_auto()
            # Should NOT be Codex, should be Anthropic (or another available provider)
            assert not isinstance(client, type(None)), "Should find a provider after expired Codex"


    def test_expired_codex_openrouter_wins(self, tmp_path, monkeypatch):
        """With expired Codex + OpenRouter key, OpenRouter should win (1st in chain)."""
        import base64
        import time as _time

        header = base64.urlsafe_b64encode(b'{"alg":"RS256","typ":"JWT"}').rstrip(b"=").decode()
        payload_data = json.dumps({"exp": int(_time.time()) - 3600}).encode()
        payload = base64.urlsafe_b64encode(payload_data).rstrip(b"=").decode()
        expired_jwt = f"{header}.{payload}.fakesig"

        hermes_home = tmp_path / "hermes"
        hermes_home.mkdir(parents=True, exist_ok=True)
        (hermes_home / "auth.json").write_text(json.dumps({
            "version": 1,
            "providers": {
                "openai-codex": {
                    "tokens": {"access_token": expired_jwt, "refresh_token": "r"},
                },
            },
        }))
        monkeypatch.setenv("HERMES_HOME", str(hermes_home))
        monkeypatch.setenv("OPENROUTER_API_KEY", "or-test-key")

        with patch("agent.auxiliary_client.OpenAI") as mock_openai:
            mock_openai.return_value = MagicMock()
            from agent.auxiliary_client import _resolve_auto
            client, model = _resolve_auto()
            assert client is not None
            # OpenRouter is 1st in chain, should win
            mock_openai.assert_called()

    def test_expired_codex_custom_endpoint_wins(self, tmp_path, monkeypatch):
        """With expired Codex + custom endpoint (Ollama), custom should win (3rd in chain)."""
        import base64
        import time as _time

        header = base64.urlsafe_b64encode(b'{"alg":"RS256","typ":"JWT"}').rstrip(b"=").decode()
        payload_data = json.dumps({"exp": int(_time.time()) - 3600}).encode()
        payload = base64.urlsafe_b64encode(payload_data).rstrip(b"=").decode()
        expired_jwt = f"{header}.{payload}.fakesig"

        hermes_home = tmp_path / "hermes"
        hermes_home.mkdir(parents=True, exist_ok=True)
        (hermes_home / "auth.json").write_text(json.dumps({
            "version": 1,
            "providers": {
                "openai-codex": {
                    "tokens": {"access_token": expired_jwt, "refresh_token": "r"},
                },
            },
        }))
        monkeypatch.setenv("HERMES_HOME", str(hermes_home))

        # Simulate Ollama or custom endpoint
        with patch("agent.auxiliary_client._resolve_custom_runtime",
                   return_value=("http://localhost:11434/v1", "sk-dummy")):
            with patch("agent.auxiliary_client.OpenAI") as mock_openai:
                mock_openai.return_value = MagicMock()
                from agent.auxiliary_client import _resolve_auto
                client, model = _resolve_auto()
                assert client is not None


    def test_hermes_oauth_file_sets_oauth_flag(self, monkeypatch):
        """OAuth-style tokens should get is_oauth=*** (token is not sk-ant-api-*)."""
        # Mock resolve_anthropic_token to return an OAuth-style token
        with patch("agent.anthropic_adapter.resolve_anthropic_token", return_value="hermes-oauth-jwt-token"), \
             patch("agent.anthropic_adapter.build_anthropic_client") as mock_build, \
             patch("agent.auxiliary_client._select_pool_entry", return_value=(False, None)):
            mock_build.return_value = MagicMock()
            from agent.auxiliary_client import _try_anthropic, AnthropicAuxiliaryClient
            client, model = _try_anthropic()
            assert client is not None, "Should resolve token"
            adapter = client.chat.completions
            assert adapter._is_oauth is True, "Non-sk-ant-api token should set is_oauth=True"

    def test_jwt_missing_exp_passes_through(self, tmp_path, monkeypatch):
        """JWT with valid JSON but no exp claim should pass through."""
        import base64
        header = base64.urlsafe_b64encode(b'{"alg":"RS256","typ":"JWT"}').rstrip(b"=").decode()
        payload_data = json.dumps({"sub": "user123"}).encode()  # no exp
        payload = base64.urlsafe_b64encode(payload_data).rstrip(b"=").decode()
        no_exp_jwt = f"{header}.{payload}.fakesig"

        hermes_home = tmp_path / "hermes"
        hermes_home.mkdir(parents=True, exist_ok=True)
        (hermes_home / "auth.json").write_text(json.dumps({
            "version": 1,
            "providers": {
                "openai-codex": {
                    "tokens": {"access_token": no_exp_jwt, "refresh_token": "r"},
                },
            },
        }))
        monkeypatch.setenv("HERMES_HOME", str(hermes_home))
        result = _read_codex_access_token()
        assert result == no_exp_jwt, "JWT without exp should pass through"

    def test_jwt_invalid_json_payload_passes_through(self, tmp_path, monkeypatch):
        """JWT with valid base64 but invalid JSON payload should pass through."""
        import base64
        header = base64.urlsafe_b64encode(b'{"alg":"RS256"}').rstrip(b"=").decode()
        payload = base64.urlsafe_b64encode(b"not-json-content").rstrip(b"=").decode()
        bad_jwt = f"{header}.{payload}.fakesig"

        hermes_home = tmp_path / "hermes"
        hermes_home.mkdir(parents=True, exist_ok=True)
        (hermes_home / "auth.json").write_text(json.dumps({
            "version": 1,
            "providers": {
                "openai-codex": {
                    "tokens": {"access_token": bad_jwt, "refresh_token": "r"},
                },
            },
        }))
        monkeypatch.setenv("HERMES_HOME", str(hermes_home))
        result = _read_codex_access_token()
        assert result == bad_jwt, "JWT with invalid JSON payload should pass through"

    def test_claude_code_oauth_env_sets_flag(self, monkeypatch):
        """CLAUDE_CODE_OAUTH_TOKEN env var should get is_oauth=True."""
        monkeypatch.setenv("CLAUDE_CODE_OAUTH_TOKEN", "cc-oauth-token-test")
        monkeypatch.delenv("ANTHROPIC_TOKEN", raising=False)
        with patch("agent.anthropic_adapter.build_anthropic_client") as mock_build:
            mock_build.return_value = MagicMock()
            from agent.auxiliary_client import _try_anthropic, AnthropicAuxiliaryClient
            client, model = _try_anthropic()
            assert client is not None
            adapter = client.chat.completions
            assert adapter._is_oauth is True


class TestExplicitProviderRouting:
    """Test explicit provider selection bypasses auto chain correctly."""

    def test_explicit_anthropic_oauth(self, monkeypatch):
        """provider='anthropic' + OAuth token should work with is_oauth=True."""
        monkeypatch.setenv("ANTHROPIC_TOKEN", "sk-ant-oat01-explicit-test")
        with patch("agent.anthropic_adapter.build_anthropic_client") as mock_build:
            mock_build.return_value = MagicMock()
            client, model = resolve_provider_client("anthropic")
            assert client is not None
            # Verify OAuth flag propagated
            adapter = client.chat.completions
            assert adapter._is_oauth is True

    def test_explicit_anthropic_api_key(self, monkeypatch):
        """provider='anthropic' + regular API key should work with is_oauth=False."""
        with patch("agent.anthropic_adapter.resolve_anthropic_token", return_value="sk-ant-api-regular-key"), \
             patch("agent.anthropic_adapter.build_anthropic_client") as mock_build, \
             patch("agent.auxiliary_client._select_pool_entry", return_value=(False, None)):
            mock_build.return_value = MagicMock()
            client, model = resolve_provider_client("anthropic")
            assert client is not None
            adapter = client.chat.completions
            assert adapter._is_oauth is False

    def test_explicit_openrouter(self, monkeypatch):
        """provider='openrouter' should use OPENROUTER_API_KEY."""
        monkeypatch.setenv("OPENROUTER_API_KEY", "or-explicit")
        with patch("agent.auxiliary_client.OpenAI") as mock_openai:
            mock_openai.return_value = MagicMock()
            client, model = resolve_provider_client("openrouter")
            assert client is not None

    def test_explicit_kimi(self, monkeypatch):
        """provider='kimi-coding' should use KIMI_API_KEY."""
        monkeypatch.setenv("KIMI_API_KEY", "kimi-test-key")
        with patch("agent.auxiliary_client.OpenAI") as mock_openai:
            mock_openai.return_value = MagicMock()
            client, model = resolve_provider_client("kimi-coding")
            assert client is not None

    def test_explicit_minimax(self, monkeypatch):
        """provider='minimax' should use MINIMAX_API_KEY."""
        monkeypatch.setenv("MINIMAX_API_KEY", "mm-test-key")
        with patch("agent.auxiliary_client.OpenAI") as mock_openai:
            mock_openai.return_value = MagicMock()
            client, model = resolve_provider_client("minimax")
            assert client is not None

    def test_explicit_deepseek(self, monkeypatch):
        """provider='deepseek' should use DEEPSEEK_API_KEY."""
        monkeypatch.setenv("DEEPSEEK_API_KEY", "ds-test-key")
        with patch("agent.auxiliary_client.OpenAI") as mock_openai:
            mock_openai.return_value = MagicMock()
            client, model = resolve_provider_client("deepseek")
            assert client is not None

    def test_explicit_zai(self, monkeypatch):
        """provider='zai' should use GLM_API_KEY."""
        monkeypatch.setenv("GLM_API_KEY", "zai-test-key")
        with patch("agent.auxiliary_client.OpenAI") as mock_openai:
            mock_openai.return_value = MagicMock()
            client, model = resolve_provider_client("zai")
            assert client is not None

    def test_explicit_google_alias_uses_gemini_credentials(self):
        """provider='google' should route through the gemini API-key provider."""
        with (
            patch("hermes_cli.auth.resolve_api_key_provider_credentials", return_value={
                "api_key": "gemini-key",
                "base_url": "https://generativelanguage.googleapis.com/v1beta/openai",
            }),
            patch("agent.auxiliary_client.OpenAI") as mock_openai,
        ):
            mock_openai.return_value = MagicMock()
            client, model = resolve_provider_client("google", model="gemini-3.1-pro-preview")

        assert client is not None
        assert model == "gemini-3.1-pro-preview"
        assert mock_openai.call_args.kwargs["api_key"] == "gemini-key"
        assert mock_openai.call_args.kwargs["base_url"] == "https://generativelanguage.googleapis.com/v1beta/openai"

    def test_explicit_unknown_returns_none(self, monkeypatch):
        """Unknown provider should return None."""
        client, model = resolve_provider_client("nonexistent-provider")
        assert client is None


class TestGetTextAuxiliaryClient:
    """Test the full resolution chain for get_text_auxiliary_client."""

    def test_openrouter_takes_priority(self, monkeypatch, codex_auth_dir):
        monkeypatch.setenv("OPENROUTER_API_KEY", "or-key")
        with patch("agent.auxiliary_client.OpenAI") as mock_openai:
            client, model = get_text_auxiliary_client()
        assert model == "google/gemini-3-flash-preview"
        mock_openai.assert_called_once()
        call_kwargs = mock_openai.call_args
        assert call_kwargs.kwargs["api_key"] == "or-key"

    def test_nous_takes_priority_over_codex(self, monkeypatch, codex_auth_dir):
        with patch("agent.auxiliary_client._read_nous_auth") as mock_nous, \
             patch("agent.auxiliary_client.OpenAI") as mock_openai:
            mock_nous.return_value = {"access_token": "nous-tok"}
            client, model = get_text_auxiliary_client()
        assert model == "google/gemini-3-flash-preview"

    def test_custom_endpoint_over_codex(self, monkeypatch, codex_auth_dir):
        config = {
            "model": {
                "provider": "custom",
                "base_url": "http://localhost:1234/v1",
                "default": "my-local-model",
            }
        }
        monkeypatch.setenv("OPENAI_API_KEY", "lm-studio-key")
        monkeypatch.setattr("hermes_cli.config.load_config", lambda: config)
        monkeypatch.setattr("hermes_cli.runtime_provider.load_config", lambda: config)
        # Override the autouse monkeypatch for codex
        monkeypatch.setattr(
            "agent.auxiliary_client._read_codex_access_token",
            lambda: "codex-test-token-abc123",
        )
        with patch("agent.auxiliary_client._read_nous_auth", return_value=None), \
             patch("agent.auxiliary_client.OpenAI") as mock_openai:
            client, model = get_text_auxiliary_client()
        assert model == "my-local-model"
        call_kwargs = mock_openai.call_args
        assert call_kwargs.kwargs["base_url"] == "http://localhost:1234/v1"

    def test_task_direct_endpoint_override(self, monkeypatch):
        monkeypatch.setenv("OPENROUTER_API_KEY", "or-key")
        monkeypatch.setenv("AUXILIARY_WEB_EXTRACT_BASE_URL", "http://localhost:2345/v1")
        monkeypatch.setenv("AUXILIARY_WEB_EXTRACT_API_KEY", "task-key")
        monkeypatch.setenv("AUXILIARY_WEB_EXTRACT_MODEL", "task-model")
        with patch("agent.auxiliary_client.OpenAI") as mock_openai:
            client, model = get_text_auxiliary_client("web_extract")
        assert model == "task-model"
        assert mock_openai.call_args.kwargs["base_url"] == "http://localhost:2345/v1"
        assert mock_openai.call_args.kwargs["api_key"] == "task-key"

    def test_task_direct_endpoint_without_openai_key_uses_placeholder(self, monkeypatch):
        """Local endpoints without an API key should use 'no-key-required' placeholder."""
        monkeypatch.setenv("OPENROUTER_API_KEY", "or-key")
        monkeypatch.setenv("AUXILIARY_WEB_EXTRACT_BASE_URL", "http://localhost:2345/v1")
        monkeypatch.setenv("AUXILIARY_WEB_EXTRACT_MODEL", "task-model")
        with patch("agent.auxiliary_client.OpenAI") as mock_openai:
            client, model = get_text_auxiliary_client("web_extract")
        assert client is not None
        assert model == "task-model"
        assert mock_openai.call_args.kwargs["api_key"] == "no-key-required"
        assert mock_openai.call_args.kwargs["base_url"] == "http://localhost:2345/v1"

    def test_custom_endpoint_uses_config_saved_base_url(self, monkeypatch):
        config = {
            "model": {
                "provider": "custom",
                "base_url": "http://localhost:1234/v1",
                "default": "my-local-model",
            }
        }
        monkeypatch.setenv("OPENAI_API_KEY", "lm-studio-key")
        monkeypatch.setattr("hermes_cli.config.load_config", lambda: config)
        monkeypatch.setattr("hermes_cli.runtime_provider.load_config", lambda: config)

        with patch("agent.auxiliary_client._read_nous_auth", return_value=None), \
             patch("agent.auxiliary_client._read_codex_access_token", return_value=None), \
             patch("agent.auxiliary_client._resolve_api_key_provider", return_value=(None, None)), \
             patch("agent.auxiliary_client.OpenAI") as mock_openai:
            client, model = get_text_auxiliary_client()

        assert client is not None
        assert model == "my-local-model"
        call_kwargs = mock_openai.call_args
        assert call_kwargs.kwargs["base_url"] == "http://localhost:1234/v1"

    def test_codex_fallback_when_nothing_else(self, codex_auth_dir):
        with patch("agent.auxiliary_client._read_nous_auth", return_value=None), \
             patch("agent.auxiliary_client.OpenAI") as mock_openai:
            client, model = get_text_auxiliary_client()
        assert model == "gpt-5.2-codex"
        # Returns a CodexAuxiliaryClient wrapper, not a raw OpenAI client
        from agent.auxiliary_client import CodexAuxiliaryClient
        assert isinstance(client, CodexAuxiliaryClient)

    def test_codex_pool_entry_takes_priority_over_auth_store(self):
        class _Entry:
            access_token = "pooled-codex-token"
            base_url = "https://chatgpt.com/backend-api/codex"

        class _Pool:
            def has_credentials(self):
                return True

            def select(self):
                return _Entry()

        with (
            patch("agent.auxiliary_client.load_pool", return_value=_Pool()),
            patch("agent.auxiliary_client.OpenAI"),
            patch("hermes_cli.auth._read_codex_tokens", side_effect=AssertionError("legacy codex store should not run")),
        ):
            from agent.auxiliary_client import _try_codex

            client, model = _try_codex()

        from agent.auxiliary_client import CodexAuxiliaryClient

        assert isinstance(client, CodexAuxiliaryClient)
        assert model == "gpt-5.2-codex"

    def test_returns_none_when_nothing_available(self, monkeypatch):
        monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
        with patch("agent.auxiliary_client._read_nous_auth", return_value=None), \
             patch("agent.auxiliary_client._read_codex_access_token", return_value=None), \
             patch("agent.auxiliary_client._resolve_api_key_provider", return_value=(None, None)):
            client, model = get_text_auxiliary_client()
        assert client is None
        assert model is None


class TestVisionClientFallback:
    """Vision client auto mode resolves known-good multimodal backends."""

    def test_vision_auto_includes_active_provider_when_configured(self, monkeypatch):
        """Active provider appears in available backends when credentials exist."""
        monkeypatch.setenv("ANTHROPIC_API_KEY", "***")
        with (
            patch("agent.auxiliary_client._read_nous_auth", return_value=None),
            patch("agent.auxiliary_client._read_main_provider", return_value="anthropic"),
            patch("agent.auxiliary_client._read_main_model", return_value="claude-sonnet-4"),
            patch("agent.anthropic_adapter.build_anthropic_client", return_value=MagicMock()),
            patch("agent.anthropic_adapter.resolve_anthropic_token", return_value="***"),
        ):
            backends = get_available_vision_backends()

        assert "anthropic" in backends

    def test_resolve_provider_client_returns_native_anthropic_wrapper(self, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-api03-key")
        with (
            patch("agent.auxiliary_client._read_nous_auth", return_value=None),
            patch("agent.anthropic_adapter.build_anthropic_client", return_value=MagicMock()),
            patch("agent.anthropic_adapter.resolve_anthropic_token", return_value="sk-ant-api03-key"),
        ):
            client, model = resolve_provider_client("anthropic")

        assert client is not None
        assert client.__class__.__name__ == "AnthropicAuxiliaryClient"
        assert model == "claude-haiku-4-5-20251001"


class TestAuxiliaryPoolAwareness:
    def test_try_nous_uses_pool_entry(self):
        class _Entry:
            access_token = "pooled-access-token"
            agent_key = "pooled-agent-key"
            inference_base_url = "https://inference.pool.example/v1"

        class _Pool:
            def has_credentials(self):
                return True

            def select(self):
                return _Entry()

        with (
            patch("agent.auxiliary_client.load_pool", return_value=_Pool()),
            patch("agent.auxiliary_client.OpenAI") as mock_openai,
        ):
            from agent.auxiliary_client import _try_nous

            client, model = _try_nous()

        assert client is not None
        assert model == "gemini-3-flash"
        call_kwargs = mock_openai.call_args.kwargs
        assert call_kwargs["api_key"] == "pooled-agent-key"
        assert call_kwargs["base_url"] == "https://inference.pool.example/v1"

    def test_resolve_provider_client_copilot_uses_runtime_credentials(self, monkeypatch):
        monkeypatch.delenv("GITHUB_TOKEN", raising=False)
        monkeypatch.delenv("GH_TOKEN", raising=False)

        with (
            patch(
                "hermes_cli.auth.resolve_api_key_provider_credentials",
                return_value={
                    "provider": "copilot",
                    "api_key": "gh-cli-token",
                    "base_url": "https://api.githubcopilot.com",
                    "source": "gh auth token",
                },
            ),
            patch("agent.auxiliary_client.OpenAI") as mock_openai,
        ):
            client, model = resolve_provider_client("copilot", model="gpt-5.4")

        assert client is not None
        assert model == "gpt-5.4"
        call_kwargs = mock_openai.call_args.kwargs
        assert call_kwargs["api_key"] == "gh-cli-token"
        assert call_kwargs["base_url"] == "https://api.githubcopilot.com"
        assert call_kwargs["default_headers"]["Editor-Version"]

    def test_vision_auto_prefers_active_provider_over_openrouter(self, monkeypatch):
        """Active provider is tried before OpenRouter in vision auto."""
        monkeypatch.setenv("OPENROUTER_API_KEY", "or-key")
        monkeypatch.setenv("ANTHROPIC_API_KEY", "***")

        with (
            patch("agent.auxiliary_client._read_nous_auth", return_value=None),
            patch("agent.auxiliary_client._read_main_provider", return_value="anthropic"),
            patch("agent.auxiliary_client._read_main_model", return_value="claude-sonnet-4"),
            patch("agent.anthropic_adapter.build_anthropic_client", return_value=MagicMock()),
            patch("agent.anthropic_adapter.resolve_anthropic_token", return_value="***"),
        ):
            provider, client, model = resolve_vision_provider_client()

        # Active provider should win over OpenRouter
        assert provider == "anthropic"

    def test_vision_auto_uses_named_custom_as_active_provider(self, monkeypatch):
        """Named custom provider works as active provider fallback in vision auto."""
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        with patch("agent.auxiliary_client._read_nous_auth", return_value=None), \
             patch("agent.auxiliary_client._select_pool_entry", return_value=(False, None)), \
             patch("agent.auxiliary_client._read_main_provider", return_value="custom:local"), \
             patch("agent.auxiliary_client._read_main_model", return_value="my-local-model"), \
             patch("agent.auxiliary_client.resolve_provider_client",
                   return_value=(MagicMock(), "my-local-model")) as mock_resolve:
            provider, client, model = resolve_vision_provider_client()
        assert client is not None
        assert provider == "custom:local"

    def test_vision_config_google_provider_uses_gemini_credentials(self, monkeypatch):
        config = {
            "auxiliary": {
                "vision": {
                    "provider": "google",
                    "model": "gemini-3.1-pro-preview",
                }
            }
        }
        monkeypatch.setattr("hermes_cli.config.load_config", lambda: config)
        with (
            patch("hermes_cli.auth.resolve_api_key_provider_credentials", return_value={
                "api_key": "gemini-key",
                "base_url": "https://generativelanguage.googleapis.com/v1beta/openai",
            }),
            patch("agent.auxiliary_client.OpenAI") as mock_openai,
        ):
            resolved_provider, client, model = resolve_vision_provider_client()

        assert resolved_provider == "gemini"
        assert client is not None
        assert model == "gemini-3.1-pro-preview"
        assert mock_openai.call_args.kwargs["api_key"] == "gemini-key"
        assert mock_openai.call_args.kwargs["base_url"] == "https://generativelanguage.googleapis.com/v1beta/openai"



class TestGetAuxiliaryProvider:
    """Tests for _get_auxiliary_provider env var resolution."""

    def test_no_task_returns_auto(self):
        assert _get_auxiliary_provider() == "auto"
        assert _get_auxiliary_provider("") == "auto"

    def test_auxiliary_prefix_takes_priority(self, monkeypatch):
        monkeypatch.setenv("AUXILIARY_VISION_PROVIDER", "openrouter")
        assert _get_auxiliary_provider("vision") == "openrouter"

    def test_context_prefix_fallback(self, monkeypatch):
        monkeypatch.setenv("CONTEXT_COMPRESSION_PROVIDER", "nous")
        assert _get_auxiliary_provider("compression") == "nous"

    def test_auxiliary_prefix_over_context_prefix(self, monkeypatch):
        monkeypatch.setenv("AUXILIARY_COMPRESSION_PROVIDER", "openrouter")
        monkeypatch.setenv("CONTEXT_COMPRESSION_PROVIDER", "nous")
        assert _get_auxiliary_provider("compression") == "openrouter"

    def test_auto_value_treated_as_auto(self, monkeypatch):
        monkeypatch.setenv("AUXILIARY_VISION_PROVIDER", "auto")
        assert _get_auxiliary_provider("vision") == "auto"

    def test_whitespace_stripped(self, monkeypatch):
        monkeypatch.setenv("AUXILIARY_VISION_PROVIDER", "  openrouter  ")
        assert _get_auxiliary_provider("vision") == "openrouter"

    def test_case_insensitive(self, monkeypatch):
        monkeypatch.setenv("AUXILIARY_VISION_PROVIDER", "OpenRouter")
        assert _get_auxiliary_provider("vision") == "openrouter"

    def test_main_provider(self, monkeypatch):
        monkeypatch.setenv("AUXILIARY_WEB_EXTRACT_PROVIDER", "main")
        assert _get_auxiliary_provider("web_extract") == "main"


class TestTaskSpecificOverrides:
    """Integration tests for per-task provider routing via get_text_auxiliary_client(task=...)."""

    def test_text_with_vision_provider_override(self, monkeypatch):
        """AUXILIARY_VISION_PROVIDER should not affect text tasks."""
        monkeypatch.setenv("AUXILIARY_VISION_PROVIDER", "nous")
        monkeypatch.setenv("OPENROUTER_API_KEY", "or-key")
        with patch("agent.auxiliary_client.OpenAI"):
            client, model = get_text_auxiliary_client()  # no task → auto
        assert model == "google/gemini-3-flash-preview"  # OpenRouter, not Nous

    def test_compression_task_reads_context_prefix(self, monkeypatch):
        """Compression task should check CONTEXT_COMPRESSION_PROVIDER env var."""
        monkeypatch.setenv("CONTEXT_COMPRESSION_PROVIDER", "nous")
        monkeypatch.setenv("OPENROUTER_API_KEY", "or-key")  # would win in auto
        with patch("agent.auxiliary_client._read_nous_auth") as mock_nous, \
             patch("agent.auxiliary_client.OpenAI"):
            mock_nous.return_value = {"access_token": "***"}
            client, model = get_text_auxiliary_client("compression")
        # Config-first: model comes from config.yaml summary_model default,
        # but provider is forced to Nous via env var
        assert client is not None

    def test_web_extract_task_override(self, monkeypatch):
        monkeypatch.setenv("AUXILIARY_WEB_EXTRACT_PROVIDER", "openrouter")
        monkeypatch.setenv("OPENROUTER_API_KEY", "or-key")
        with patch("agent.auxiliary_client.OpenAI"):
            client, model = get_text_auxiliary_client("web_extract")
        assert model == "google/gemini-3-flash-preview"

    def test_task_direct_endpoint_from_config(self, monkeypatch, tmp_path):
        hermes_home = tmp_path / "hermes"
        hermes_home.mkdir(parents=True, exist_ok=True)
        (hermes_home / "config.yaml").write_text(
            """auxiliary:
  web_extract:
    base_url: http://localhost:3456/v1
    api_key: config-key
    model: config-model
"""
        )
        monkeypatch.setenv("HERMES_HOME", str(hermes_home))
        with patch("agent.auxiliary_client.OpenAI") as mock_openai:
            client, model = get_text_auxiliary_client("web_extract")
        assert model == "config-model"
        assert mock_openai.call_args.kwargs["base_url"] == "http://localhost:3456/v1"
        assert mock_openai.call_args.kwargs["api_key"] == "config-key"

    def test_task_without_override_uses_auto(self, monkeypatch):
        """A task with no provider env var falls through to auto chain."""
        monkeypatch.setenv("OPENROUTER_API_KEY", "or-key")
        with patch("agent.auxiliary_client.OpenAI"):
            client, model = get_text_auxiliary_client("compression")
        assert model == "google/gemini-3-flash-preview"  # auto → OpenRouter

    def test_compression_summary_base_url_from_config(self, monkeypatch, tmp_path):
        """compression.summary_base_url should produce a custom-endpoint client."""
        hermes_home = tmp_path / "hermes"
        hermes_home.mkdir(parents=True, exist_ok=True)
        (hermes_home / "config.yaml").write_text(
            """compression:
  summary_provider: custom
  summary_model: glm-4.7
  summary_base_url: https://api.z.ai/api/coding/paas/v4
"""
        )
        monkeypatch.setenv("HERMES_HOME", str(hermes_home))
        # Custom endpoints need an API key to build the client
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")
        with patch("agent.auxiliary_client.OpenAI") as mock_openai:
            client, model = get_text_auxiliary_client("compression")
        assert model == "glm-4.7"
        assert mock_openai.call_args.kwargs["base_url"] == "https://api.z.ai/api/coding/paas/v4"


class TestAuxiliaryMaxTokensParam:
    def test_codex_fallback_uses_max_tokens(self, monkeypatch):
        """Codex adapter translates max_tokens internally, so we return max_tokens."""
        with patch("agent.auxiliary_client._read_nous_auth", return_value=None), \
             patch("agent.auxiliary_client._read_codex_access_token", return_value="tok"):
            result = auxiliary_max_tokens_param(1024)
        assert result == {"max_tokens": 1024}

    def test_openrouter_uses_max_tokens(self, monkeypatch):
        monkeypatch.setenv("OPENROUTER_API_KEY", "or-key")
        result = auxiliary_max_tokens_param(1024)
        assert result == {"max_tokens": 1024}

    def test_no_provider_uses_max_tokens(self):
        with patch("agent.auxiliary_client._read_nous_auth", return_value=None), \
             patch("agent.auxiliary_client._read_codex_access_token", return_value=None):
            result = auxiliary_max_tokens_param(1024)
        assert result == {"max_tokens": 1024}


# ── Payment / credit exhaustion fallback ─────────────────────────────────


class TestIsPaymentError:
    """_is_payment_error detects 402 and credit-related errors."""

    def test_402_status_code(self):
        exc = Exception("Payment Required")
        exc.status_code = 402
        assert _is_payment_error(exc) is True

    def test_402_with_credits_message(self):
        exc = Exception("You requested up to 65535 tokens, but can only afford 8029")
        exc.status_code = 402
        assert _is_payment_error(exc) is True

    def test_429_with_credits_message(self):
        exc = Exception("insufficient credits remaining")
        exc.status_code = 429
        assert _is_payment_error(exc) is True

    def test_429_without_credits_message_is_not_payment(self):
        """Normal rate limits should NOT be treated as payment errors."""
        exc = Exception("Rate limit exceeded, try again in 2 seconds")
        exc.status_code = 429
        assert _is_payment_error(exc) is False

    def test_generic_500_is_not_payment(self):
        exc = Exception("Internal server error")
        exc.status_code = 500
        assert _is_payment_error(exc) is False

    def test_no_status_code_with_billing_message(self):
        exc = Exception("billing: payment required for this request")
        assert _is_payment_error(exc) is True

    def test_no_status_code_no_message(self):
        exc = Exception("connection reset")
        assert _is_payment_error(exc) is False


class TestGetProviderChain:
    """_get_provider_chain() resolves functions at call time (testable)."""

    def test_returns_five_entries(self):
        chain = _get_provider_chain()
        assert len(chain) == 5
        labels = [label for label, _ in chain]
        assert labels == ["openrouter", "nous", "local/custom", "openai-codex", "api-key"]

    def test_picks_up_patched_functions(self):
        """Patches on _try_* functions must be visible in the chain."""
        sentinel = lambda: ("patched", "model")
        with patch("agent.auxiliary_client._try_openrouter", sentinel):
            chain = _get_provider_chain()
        assert chain[0] == ("openrouter", sentinel)


class TestTryPaymentFallback:
    """_try_payment_fallback skips the failed provider and tries alternatives."""

    def test_skips_failed_provider(self):
        mock_client = MagicMock()
        with patch("agent.auxiliary_client._try_openrouter", return_value=(None, None)), \
             patch("agent.auxiliary_client._try_nous", return_value=(mock_client, "nous-model")), \
             patch("agent.auxiliary_client._read_main_provider", return_value="openrouter"):
            client, model, label = _try_payment_fallback("openrouter", task="compression")
        assert client is mock_client
        assert model == "nous-model"
        assert label == "nous"

    def test_returns_none_when_no_fallback(self):
        with patch("agent.auxiliary_client._try_openrouter", return_value=(None, None)), \
             patch("agent.auxiliary_client._try_nous", return_value=(None, None)), \
             patch("agent.auxiliary_client._try_custom_endpoint", return_value=(None, None)), \
             patch("agent.auxiliary_client._try_codex", return_value=(None, None)), \
             patch("agent.auxiliary_client._resolve_api_key_provider", return_value=(None, None)), \
             patch("agent.auxiliary_client._read_main_provider", return_value="openrouter"):
            client, model, label = _try_payment_fallback("openrouter")
        assert client is None
        assert label == ""

    def test_codex_alias_maps_to_chain_label(self):
        """'codex' should map to 'openai-codex' in the skip set."""
        mock_client = MagicMock()
        with patch("agent.auxiliary_client._try_openrouter", return_value=(mock_client, "or-model")), \
             patch("agent.auxiliary_client._try_codex", return_value=(None, None)), \
             patch("agent.auxiliary_client._read_main_provider", return_value="openai-codex"):
            client, model, label = _try_payment_fallback("openai-codex", task="vision")
        assert client is mock_client
        assert label == "openrouter"

    def test_skips_to_codex_when_or_and_nous_fail(self):
        mock_codex = MagicMock()
        with patch("agent.auxiliary_client._try_openrouter", return_value=(None, None)), \
             patch("agent.auxiliary_client._try_nous", return_value=(None, None)), \
             patch("agent.auxiliary_client._try_custom_endpoint", return_value=(None, None)), \
             patch("agent.auxiliary_client._try_codex", return_value=(mock_codex, "gpt-5.2-codex")), \
             patch("agent.auxiliary_client._read_main_provider", return_value="openrouter"):
            client, model, label = _try_payment_fallback("openrouter")
        assert client is mock_codex
        assert model == "gpt-5.2-codex"
        assert label == "openai-codex"


class TestCallLlmPaymentFallback:
    """call_llm() retries with a different provider on 402 / payment errors."""

    def _make_402_error(self, msg="Payment Required: insufficient credits"):
        exc = Exception(msg)
        exc.status_code = 402
        return exc

    def test_402_triggers_fallback(self, monkeypatch):
        """When the primary provider returns 402, call_llm tries the next one."""
        monkeypatch.setenv("OPENROUTER_API_KEY", "or-key")

        primary_client = MagicMock()
        primary_client.chat.completions.create.side_effect = self._make_402_error()

        fallback_client = MagicMock()
        fallback_response = MagicMock()
        fallback_client.chat.completions.create.return_value = fallback_response

        with patch("agent.auxiliary_client._get_cached_client",
                    return_value=(primary_client, "google/gemini-3-flash-preview")), \
             patch("agent.auxiliary_client._resolve_task_provider_model",
                    return_value=("openrouter", "google/gemini-3-flash-preview", None, None)), \
             patch("agent.auxiliary_client._try_payment_fallback",
                    return_value=(fallback_client, "gpt-5.2-codex", "openai-codex")) as mock_fb:
            result = call_llm(
                task="compression",
                messages=[{"role": "user", "content": "hello"}],
            )

        assert result is fallback_response
        mock_fb.assert_called_once_with("openrouter", "compression")
        # Fallback call should use the fallback model
        fb_kwargs = fallback_client.chat.completions.create.call_args.kwargs
        assert fb_kwargs["model"] == "gpt-5.2-codex"

    def test_non_payment_error_not_caught(self, monkeypatch):
        """Non-payment errors (500, connection, etc.) should NOT trigger fallback."""
        monkeypatch.setenv("OPENROUTER_API_KEY", "or-key")

        primary_client = MagicMock()
        server_err = Exception("Internal Server Error")
        server_err.status_code = 500
        primary_client.chat.completions.create.side_effect = server_err

        with patch("agent.auxiliary_client._get_cached_client",
                    return_value=(primary_client, "google/gemini-3-flash-preview")), \
             patch("agent.auxiliary_client._resolve_task_provider_model",
                    return_value=("openrouter", "google/gemini-3-flash-preview", None, None)):
            with pytest.raises(Exception, match="Internal Server Error"):
                call_llm(
                    task="compression",
                    messages=[{"role": "user", "content": "hello"}],
                )

    def test_402_with_no_fallback_reraises(self, monkeypatch):
        """When 402 hits and no fallback is available, the original error propagates."""
        monkeypatch.setenv("OPENROUTER_API_KEY", "or-key")

        primary_client = MagicMock()
        primary_client.chat.completions.create.side_effect = self._make_402_error()

        with patch("agent.auxiliary_client._get_cached_client",
                    return_value=(primary_client, "google/gemini-3-flash-preview")), \
             patch("agent.auxiliary_client._resolve_task_provider_model",
                    return_value=("openrouter", "google/gemini-3-flash-preview", None, None)), \
             patch("agent.auxiliary_client._try_payment_fallback",
                    return_value=(None, None, "")):
            with pytest.raises(Exception, match="insufficient credits"):
                call_llm(
                    task="compression",
                    messages=[{"role": "user", "content": "hello"}],
                )


# ---------------------------------------------------------------------------
# Gate: _resolve_api_key_provider must skip anthropic when not configured
# ---------------------------------------------------------------------------


def test_resolve_api_key_provider_skips_unconfigured_anthropic(monkeypatch):
    """_resolve_api_key_provider must not try anthropic when user never configured it."""
    from collections import OrderedDict
    from hermes_cli.auth import ProviderConfig

    # Build a minimal registry with only "anthropic" so the loop is guaranteed
    # to reach it without being short-circuited by earlier providers.
    fake_registry = OrderedDict({
        "anthropic": ProviderConfig(
            id="anthropic",
            name="Anthropic",
            auth_type="api_key",
            inference_base_url="https://api.anthropic.com",
            api_key_env_vars=("ANTHROPIC_API_KEY",),
        ),
    })

    called = []

    def mock_try_anthropic():
        called.append("anthropic")
        return None, None

    monkeypatch.setattr("agent.auxiliary_client._try_anthropic", mock_try_anthropic)
    monkeypatch.setattr("hermes_cli.auth.PROVIDER_REGISTRY", fake_registry)
    monkeypatch.setattr(
        "hermes_cli.auth.is_provider_explicitly_configured",
        lambda pid: False,
    )

    from agent.auxiliary_client import _resolve_api_key_provider
    _resolve_api_key_provider()

    assert "anthropic" not in called, \
        "_try_anthropic() should not be called when anthropic is not explicitly configured"
