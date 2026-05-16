"""Tests for agent.auxiliary_client resolution chain, provider overrides, and model overrides."""

import json
import logging
import os
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch, MagicMock, AsyncMock

import pytest

from agent.auxiliary_client import (
    get_text_auxiliary_client,
    get_available_vision_backends,
    resolve_vision_provider_client,
    resolve_provider_client,
    auxiliary_max_tokens_param,
    call_llm,
    async_call_llm,
    _build_call_kwargs,
    _read_codex_access_token,
    _get_provider_chain,
    _is_payment_error,
    _is_rate_limit_error,
    _normalize_aux_provider,
    _try_payment_fallback,
    _resolve_auto,
    _resolve_xai_oauth_for_aux,
    _CodexCompletionsAdapter,
)


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    """Strip provider env vars so each test starts clean."""
    for key in (
        "OPENROUTER_API_KEY", "OPENAI_BASE_URL", "OPENAI_API_KEY",
        "OPENAI_MODEL", "LLM_MODEL", "NOUS_INFERENCE_BASE_URL",
        "ANTHROPIC_API_KEY", "ANTHROPIC_TOKEN", "CLAUDE_CODE_OAUTH_TOKEN",
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


class TestAuxiliaryMaxTokensParam:
    def test_uses_max_completion_tokens_for_github_copilot_custom_base(self):
        with patch("agent.auxiliary_client._resolve_custom_runtime", return_value=("https://api.githubcopilot.com", "key", None)), \
             patch("agent.auxiliary_client._read_nous_auth", return_value=None):
            assert auxiliary_max_tokens_param(2048) == {"max_completion_tokens": 2048}

    def test_uses_max_completion_tokens_for_github_copilot_custom_base_path(self):
        with patch("agent.auxiliary_client._resolve_custom_runtime", return_value=("https://api.githubcopilot.com/chat/completions", "key", None)), \
             patch("agent.auxiliary_client._read_nous_auth", return_value=None):
            assert auxiliary_max_tokens_param(2048) == {"max_completion_tokens": 2048}


class TestNormalizeAuxProvider:
    def test_maps_github_copilot_aliases(self):
        assert _normalize_aux_provider("github") == "copilot"
        assert _normalize_aux_provider("github-copilot") == "copilot"
        assert _normalize_aux_provider("github-models") == "copilot"

    def test_maps_github_copilot_acp_aliases(self):
        assert _normalize_aux_provider("github-copilot-acp") == "copilot-acp"
        assert _normalize_aux_provider("copilot-acp-agent") == "copilot-acp"


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
        with patch("agent.auxiliary_client._select_pool_entry", return_value=(False, None)):
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
        with patch("agent.auxiliary_client._select_pool_entry", return_value=(False, None)):
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


class TestResolveXaiOAuthForAux:
    def test_uses_pool_backed_credentials_without_singleton(self, tmp_path, monkeypatch):
        """Auxiliary xAI OAuth must see pool-only credentials.

        ``hermes auth status`` already reports these as logged in; compression
        should not fall through to "no auxiliary provider configured" just
        because the singleton auth-store entry is absent.
        """
        from agent.credential_pool import AUTH_TYPE_OAUTH, PooledCredential, load_pool
        from hermes_cli.auth import DEFAULT_XAI_OAUTH_BASE_URL

        hermes_home = tmp_path / "hermes"
        hermes_home.mkdir(parents=True, exist_ok=True)
        (hermes_home / "auth.json").write_text(json.dumps({
            "version": 1,
            "providers": {},
        }))
        monkeypatch.setenv("HERMES_HOME", str(hermes_home))
        monkeypatch.delenv("HERMES_XAI_BASE_URL", raising=False)
        monkeypatch.delenv("XAI_BASE_URL", raising=False)

        pool = load_pool("xai-oauth")
        pool.add_entry(PooledCredential(
            provider="xai-oauth",
            id="xai123",
            label="pool-only",
            auth_type=AUTH_TYPE_OAUTH,
            priority=0,
            source="manual:xai_pkce",
            access_token="pool-access-token",
            refresh_token="pool-refresh-token",
            base_url=DEFAULT_XAI_OAUTH_BASE_URL,
        ))

        assert _resolve_xai_oauth_for_aux() == (
            "pool-access-token",
            DEFAULT_XAI_OAUTH_BASE_URL,
        )

    def test_pool_backed_credentials_honor_base_url_env_override(self, tmp_path, monkeypatch):
        from agent.credential_pool import AUTH_TYPE_OAUTH, PooledCredential, load_pool
        from hermes_cli.auth import DEFAULT_XAI_OAUTH_BASE_URL

        hermes_home = tmp_path / "hermes"
        hermes_home.mkdir(parents=True, exist_ok=True)
        (hermes_home / "auth.json").write_text(json.dumps({
            "version": 1,
            "providers": {},
        }))
        monkeypatch.setenv("HERMES_HOME", str(hermes_home))
        monkeypatch.setenv("HERMES_XAI_BASE_URL", "https://example.x.ai/v1/")

        pool = load_pool("xai-oauth")
        pool.add_entry(PooledCredential(
            provider="xai-oauth",
            id="xai456",
            label="pool-only",
            auth_type=AUTH_TYPE_OAUTH,
            priority=0,
            source="manual:xai_pkce",
            access_token="pool-access-token",
            refresh_token="pool-refresh-token",
            base_url=DEFAULT_XAI_OAUTH_BASE_URL,
        ))

        assert _resolve_xai_oauth_for_aux() == (
            "pool-access-token",
            "https://example.x.ai/v1",
        )


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


class TestBuildCodexClient:
    def test_pool_without_selected_entry_falls_back_to_auth_store(self):
        with (
            patch("agent.auxiliary_client._select_pool_entry", return_value=(True, None)),
            patch("agent.auxiliary_client._read_codex_access_token", return_value="codex-auth-token"),
            patch("agent.auxiliary_client.OpenAI") as mock_openai,
        ):
            mock_openai.return_value = MagicMock()
            from agent.auxiliary_client import _build_codex_client

            client, model = _build_codex_client("gpt-5.4")

        assert client is not None
        assert model == "gpt-5.4"
        assert mock_openai.call_args.kwargs["api_key"] == "codex-auth-token"
        assert mock_openai.call_args.kwargs["base_url"] == "https://chatgpt.com/backend-api/codex"

    def test_rejects_missing_model(self):
        """Callers must pass an explicit model; no hardcoded default."""
        from agent.auxiliary_client import _build_codex_client

        client, model = _build_codex_client("")
        assert client is None
        assert model is None

    def test_cached_codex_client_rebuilds_when_pool_entry_changes(self):
        import agent.auxiliary_client as aux

        class _Entry:
            def __init__(self, entry_id, token):
                self.id = entry_id
                self.runtime_api_key = token
                self.runtime_base_url = "https://chatgpt.com/backend-api/codex"

        class _Pool:
            def __init__(self):
                self.entry = _Entry("cred-a", "tok-a")

            def has_credentials(self):
                return True

            def current(self):
                return self.entry

            def peek(self):
                return self.entry

            def select(self):
                return self.entry

        pool = _Pool()
        client_a = MagicMock(name="codex-client-a")
        client_b = MagicMock(name="codex-client-b")

        with (
            patch("agent.auxiliary_client.load_pool", return_value=pool),
            patch("agent.auxiliary_client.OpenAI", side_effect=[client_a, client_b]) as mock_openai,
        ):
            aux.shutdown_cached_clients()
            try:
                first_client, first_model = aux._get_cached_client("openai-codex", "gpt-5.4")
                pool.entry = _Entry("cred-b", "tok-b")
                second_client, second_model = aux._get_cached_client("openai-codex", "gpt-5.4")
            finally:
                aux.shutdown_cached_clients()

        assert first_client is not second_client
        assert first_model == "gpt-5.4"
        assert second_model == "gpt-5.4"
        assert mock_openai.call_count == 2


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
        with patch("agent.anthropic_adapter.resolve_anthropic_token", return_value="sk-ant-oat-hermes-token"), \
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
        monkeypatch.setenv("CLAUDE_CODE_OAUTH_TOKEN", "sk-ant-oat-cc-test-token")
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

    def test_explicit_openrouter_pool_exhausted_logs_precise_warning(self, monkeypatch, caplog):
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
        with patch("agent.auxiliary_client._select_pool_entry", return_value=(True, None)):
            with caplog.at_level(logging.WARNING, logger="agent.auxiliary_client"):
                client, model = resolve_provider_client("openrouter")
        assert client is None
        assert model is None
        assert any(
            "credential pool has no usable entries" in record.message
            for record in caplog.records
        )
        assert not any(
            "OPENROUTER_API_KEY not set" in record.message
            for record in caplog.records
        )

    def test_explicit_openrouter_missing_env_keeps_not_set_warning(self, monkeypatch, caplog):
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
        with patch("agent.auxiliary_client._select_pool_entry", return_value=(False, None)):
            with caplog.at_level(logging.WARNING, logger="agent.auxiliary_client"):
                client, model = resolve_provider_client("openrouter")
        assert client is None
        assert model is None
        assert any(
            "OPENROUTER_API_KEY not set" in record.message
            for record in caplog.records
        )

class TestGetTextAuxiliaryClient:
    """Test the full resolution chain for get_text_auxiliary_client."""

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
            from agent.auxiliary_client import _build_codex_client

            client, model = _build_codex_client("gpt-5.4")

        from agent.auxiliary_client import CodexAuxiliaryClient

        assert isinstance(client, CodexAuxiliaryClient)
        assert model == "gpt-5.4"

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

    def test_custom_endpoint_uses_codex_wrapper_when_runtime_requests_responses_api(self):
        with patch("agent.auxiliary_client._resolve_custom_runtime",
                   return_value=("https://api.openai.com/v1", "sk-test", "codex_responses")), \
             patch("agent.auxiliary_client._read_main_model", return_value="gpt-5.3-codex"), \
             patch("agent.auxiliary_client.OpenAI") as mock_openai:
            client, model = get_text_auxiliary_client()

        from agent.auxiliary_client import CodexAuxiliaryClient
        assert isinstance(client, CodexAuxiliaryClient)
        assert model == "gpt-5.3-codex"
        assert mock_openai.call_args.kwargs["base_url"] == "https://api.openai.com/v1"
        assert mock_openai.call_args.kwargs["api_key"] == "sk-test"


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
        monkeypatch.setenv("ANTHROPIC_API_KEY", "***")
        with (
            patch("agent.auxiliary_client._read_nous_auth", return_value=None),
            patch("agent.anthropic_adapter.build_anthropic_client", return_value=MagicMock()),
            patch("agent.anthropic_adapter.resolve_anthropic_token", return_value="***"),
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
            patch("hermes_cli.models.get_nous_recommended_aux_model", return_value=None),
        ):
            from agent.auxiliary_client import _try_nous

            client, model = _try_nous()

        assert client is not None
        assert model == "google/gemini-3-flash-preview"
        assert mock_openai.call_args.kwargs["api_key"] == "pooled-agent-key"
        assert mock_openai.call_args.kwargs["base_url"] == "https://inference.pool.example/v1"

    def test_try_nous_uses_portal_recommendation_for_text(self):
        """When the Portal recommends a compaction model, _try_nous honors it."""
        fresh_base = "https://inference-api.nousresearch.com/v1"
        with (
            patch("agent.auxiliary_client._read_nous_auth", return_value={"access_token": "***"}),
            patch("agent.auxiliary_client._resolve_nous_runtime_api", return_value=("fresh-agent-key", fresh_base)),
            patch("hermes_cli.models.get_nous_recommended_aux_model", return_value="minimax/minimax-m2.7") as mock_rec,
            patch("agent.auxiliary_client.OpenAI") as mock_openai,
        ):
            from agent.auxiliary_client import _try_nous

            mock_openai.return_value = MagicMock()
            client, model = _try_nous(vision=False)

        assert client is not None
        assert model == "minimax/minimax-m2.7"
        assert mock_rec.call_args.kwargs["vision"] is False

    def test_try_nous_uses_portal_recommendation_for_vision(self):
        """Vision tasks should ask for the vision-specific recommendation."""
        fresh_base = "https://inference-api.nousresearch.com/v1"
        with (
            patch("agent.auxiliary_client._read_nous_auth", return_value={"access_token": "***"}),
            patch("agent.auxiliary_client._resolve_nous_runtime_api", return_value=("fresh-agent-key", fresh_base)),
            patch("hermes_cli.models.get_nous_recommended_aux_model", return_value="google/gemini-3-flash-preview") as mock_rec,
            patch("agent.auxiliary_client.OpenAI"),
        ):
            from agent.auxiliary_client import _try_nous
            client, model = _try_nous(vision=True)

        assert client is not None
        assert model == "google/gemini-3-flash-preview"
        assert mock_rec.call_args.kwargs["vision"] is True

    def test_try_nous_falls_back_when_recommendation_lookup_raises(self):
        """If the Portal lookup throws, we must still return a usable model."""
        fresh_base = "https://inference-api.nousresearch.com/v1"
        with (
            patch("agent.auxiliary_client._read_nous_auth", return_value={"access_token": "***"}),
            patch("agent.auxiliary_client._resolve_nous_runtime_api", return_value=("fresh-agent-key", fresh_base)),
            patch("hermes_cli.models.get_nous_recommended_aux_model", side_effect=RuntimeError("portal down")),
            patch("agent.auxiliary_client.OpenAI"),
        ):
            from agent.auxiliary_client import _try_nous
            client, model = _try_nous()

        assert client is not None
        assert model == "google/gemini-3-flash-preview"

    def test_call_llm_retries_nous_after_401(self):
        class _Auth401(Exception):
            status_code = 401

        stale_client = MagicMock()
        stale_client.base_url = "https://inference-api.nousresearch.com/v1"
        stale_client.chat.completions.create.side_effect = _Auth401("stale nous key")

        fresh_client = MagicMock()
        fresh_client.base_url = "https://inference-api.nousresearch.com/v1"
        fresh_client.chat.completions.create.return_value = {"ok": True}

        with (
            patch("agent.auxiliary_client._resolve_task_provider_model", return_value=("nous", "nous-model", None, None, None)),
            patch("agent.auxiliary_client._get_cached_client", return_value=(stale_client, "nous-model")),
            patch("agent.auxiliary_client.OpenAI", return_value=fresh_client),
            patch("agent.auxiliary_client._validate_llm_response", side_effect=lambda resp, _task: resp),
            patch("agent.auxiliary_client._resolve_nous_runtime_api", return_value=("fresh-agent-key", "https://inference-api.nousresearch.com/v1")),
        ):
            result = call_llm(
                task="compression",
                messages=[{"role": "user", "content": "hi"}],
            )

        assert result == {"ok": True}
        assert stale_client.chat.completions.create.call_count == 1
        assert fresh_client.chat.completions.create.call_count == 1

    @pytest.mark.asyncio
    async def test_async_call_llm_retries_nous_after_401(self):
        class _Auth401(Exception):
            status_code = 401

        stale_client = MagicMock()
        stale_client.base_url = "https://inference-api.nousresearch.com/v1"
        stale_client.chat.completions.create = AsyncMock(side_effect=_Auth401("stale nous key"))

        fresh_async_client = MagicMock()
        fresh_async_client.base_url = "https://inference-api.nousresearch.com/v1"
        fresh_async_client.chat.completions.create = AsyncMock(return_value={"ok": True})

        with (
            patch("agent.auxiliary_client._resolve_task_provider_model", return_value=("nous", "nous-model", None, None, None)),
            patch("agent.auxiliary_client._get_cached_client", return_value=(stale_client, "nous-model")),
            patch("agent.auxiliary_client._to_async_client", return_value=(fresh_async_client, "nous-model")),
            patch("agent.auxiliary_client._validate_llm_response", side_effect=lambda resp, _task: resp),
            patch("agent.auxiliary_client._resolve_nous_runtime_api", return_value=("fresh-agent-key", "https://inference-api.nousresearch.com/v1")),
        ):
            result = await async_call_llm(
                task="session_search",
                messages=[{"role": "user", "content": "hi"}],
            )

        assert result == {"ok": True}
        assert stale_client.chat.completions.create.await_count == 1
        assert fresh_async_client.chat.completions.create.await_count == 1

    def test_cached_gmi_client_keeps_explicit_slash_model_override(self):
        import agent.auxiliary_client as aux

        fake_client = MagicMock()

        with patch(
            "agent.auxiliary_client.resolve_provider_client",
            return_value=(fake_client, "google/gemini-3.1-flash-lite-preview"),
        ) as mock_resolve:
            aux.shutdown_cached_clients()
            try:
                client, model = aux._get_cached_client(
                    "gmi",
                    "google/gemini-3.1-flash-lite-preview",
                    base_url="https://api.gmi-serving.com/v1",
                    api_key="gmi-key",
                )
                assert client is fake_client
                assert model == "google/gemini-3.1-flash-lite-preview"

                client, model = aux._get_cached_client(
                    "gmi",
                    "openai/gpt-5.4-mini",
                    base_url="https://api.gmi-serving.com/v1",
                    api_key="gmi-key",
                )
            finally:
                aux.shutdown_cached_clients()

        assert client is fake_client
        assert model == "openai/gpt-5.4-mini"
        assert mock_resolve.call_count == 1


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


class TestIsRateLimitError:
    """_is_rate_limit_error detects 429 rate-limit errors warranting fallback."""

    def test_429_with_rate_limit_message(self):
        exc = Exception("Rate limit exceeded, try again in 2 seconds")
        exc.status_code = 429
        assert _is_rate_limit_error(exc) is True

    def test_429_with_resets_in_message(self):
        """Nous-style 429: 'resets in 3508s'."""
        exc = Exception("Hold up for a bit, you've exceeded the rate limit on your API key")
        exc.status_code = 429
        assert _is_rate_limit_error(exc) is True

    def test_429_with_too_many_requests(self):
        exc = Exception("Too many requests")
        exc.status_code = 429
        assert _is_rate_limit_error(exc) is True

    def test_429_without_billing_keywords_is_rate_limit(self):
        """Generic 429 without billing keywords = likely a rate limit."""
        exc = Exception("Something went wrong")
        exc.status_code = 429
        assert _is_rate_limit_error(exc) is True

    def test_429_with_credits_message_is_not_rate_limit(self):
        """Billing-related 429 should NOT be classified as rate limit."""
        exc = Exception("insufficient credits remaining")
        exc.status_code = 429
        assert _is_rate_limit_error(exc) is False

    def test_429_with_billing_message_is_not_rate_limit(self):
        exc = Exception("you can only afford 1000 tokens")
        exc.status_code = 429
        assert _is_rate_limit_error(exc) is False

    def test_402_is_not_rate_limit(self):
        exc = Exception("Payment Required")
        exc.status_code = 402
        assert _is_rate_limit_error(exc) is False

    def test_500_is_not_rate_limit(self):
        exc = Exception("Internal Server Error")
        exc.status_code = 500
        assert _is_rate_limit_error(exc) is False

    def test_openai_ratelimiterror_classname(self):
        """OpenAI SDK RateLimitError may omit .status_code — detect by class name."""
        class RateLimitError(Exception):
            pass
        exc = RateLimitError("rate limit exceeded")
        # No status_code set, but class name matches
        assert _is_rate_limit_error(exc) is True

    def test_no_status_code_no_keywords_is_not_rate_limit(self):
        exc = Exception("connection reset")
        assert _is_rate_limit_error(exc) is False


class TestGetProviderChain:
    """_get_provider_chain() resolves functions at call time (testable)."""

    def test_returns_four_entries(self):
        chain = _get_provider_chain()
        assert len(chain) == 4
        labels = [label for label, _ in chain]
        assert labels == ["openrouter", "nous", "local/custom", "api-key"]
        # Codex is deliberately NOT in this chain — see _get_provider_chain
        # docstring. ChatGPT-account Codex has a shifting model allow-list;
        # guessing a model to fall back on breaks more often than it helps.
        assert "openai-codex" not in labels

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
             patch("agent.auxiliary_client._resolve_api_key_provider", return_value=(None, None)), \
             patch("agent.auxiliary_client._read_main_provider", return_value="openrouter"):
            client, model, label = _try_payment_fallback("openrouter")
        assert client is None
        assert label == ""

    def test_codex_alias_maps_to_chain_label(self):
        """'codex' should map to 'openai-codex' in the skip set."""
        mock_client = MagicMock()
        with patch("agent.auxiliary_client._try_openrouter", return_value=(mock_client, "or-model")), \
             patch("agent.auxiliary_client._read_main_provider", return_value="openai-codex"):
            client, model, label = _try_payment_fallback("openai-codex", task="vision")
        assert client is mock_client
        assert label == "openrouter"

    def test_codex_not_in_fallback_chain(self):
        """Codex is deliberately NOT a fallback rung (shifting model allow-list).

        When OR/Nous/custom/api-key all fail, payment-fallback returns None —
        Codex is never tried with a guessed model.
        """
        with patch("agent.auxiliary_client._try_openrouter", return_value=(None, None)), \
             patch("agent.auxiliary_client._try_nous", return_value=(None, None)), \
             patch("agent.auxiliary_client._try_custom_endpoint", return_value=(None, None)), \
             patch("agent.auxiliary_client._resolve_api_key_provider", return_value=(None, None)), \
             patch("agent.auxiliary_client._read_main_provider", return_value="openrouter"):
            client, model, label = _try_payment_fallback("openrouter")
        assert client is None
        assert model is None
        assert label == ""


class TestCallLlmPaymentFallback:
    """call_llm() retries with a different provider on 402 / payment / rate-limit errors."""

    def _make_402_error(self, msg="Payment Required: insufficient credits"):
        exc = Exception(msg)
        exc.status_code = 402
        return exc

    def _make_429_rate_limit_error(self, msg="Rate limit exceeded, try again in 60 seconds"):
        exc = Exception(msg)
        exc.status_code = 429
        return exc

    def test_non_payment_error_not_caught(self, monkeypatch):
        """Non-payment/non-connection errors (500) should NOT trigger fallback."""
        monkeypatch.setenv("OPENROUTER_API_KEY", "or-key")

        primary_client = MagicMock()
        server_err = Exception("Internal Server Error")
        server_err.status_code = 500
        primary_client.chat.completions.create.side_effect = server_err

        with patch("agent.auxiliary_client._get_cached_client",
                    return_value=(primary_client, "google/gemini-3-flash-preview")), \
             patch("agent.auxiliary_client._resolve_task_provider_model",
                    return_value=("auto", "google/gemini-3-flash-preview", None, None, None)):
            with pytest.raises(Exception, match="Internal Server Error"):
                call_llm(
                    task="compression",
                    messages=[{"role": "user", "content": "hello"}],
                )

    def test_429_rate_limit_triggers_fallback(self, monkeypatch):
        """429 rate-limit errors should trigger fallback to next provider."""
        monkeypatch.setenv("OPENROUTER_API_KEY", "or-key")

        primary_client = MagicMock()
        rate_err = self._make_429_rate_limit_error()
        primary_client.chat.completions.create.side_effect = rate_err

        fallback_client = MagicMock()
        fallback_client.chat.completions.create.return_value = MagicMock(choices=[
            MagicMock(message=MagicMock(content="fallback response"))
        ])

        with patch("agent.auxiliary_client._get_cached_client",
                    return_value=(primary_client, "xiaomi/mimo-v2-pro")), \
             patch("agent.auxiliary_client._resolve_task_provider_model",
                    return_value=("auto", "xiaomi/mimo-v2-pro", None, None, None)), \
             patch("agent.auxiliary_client._try_payment_fallback",
                    return_value=(fallback_client, "fallback-model", "openrouter")):
            result = call_llm(
                task="session_search",
                messages=[{"role": "user", "content": "hello"}],
            )
        # Fallback client should have been used
        assert fallback_client.chat.completions.create.called

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


# ---------------------------------------------------------------------------
# model="default" elimination (#7512)
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# _try_payment_fallback reason parameter (#7512 bug 3)
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# _is_connection_error coverage
# ---------------------------------------------------------------------------


class TestIsConnectionError:
    """Tests for _is_connection_error detection."""

    def test_connection_refused(self):
        from agent.auxiliary_client import _is_connection_error
        err = Exception("Connection refused")
        assert _is_connection_error(err) is True

    def test_timeout(self):
        from agent.auxiliary_client import _is_connection_error
        err = Exception("Request timed out.")
        assert _is_connection_error(err) is True

    def test_dns_failure(self):
        from agent.auxiliary_client import _is_connection_error
        err = Exception("Name or service not known")
        assert _is_connection_error(err) is True

    def test_normal_api_error_not_connection(self):
        from agent.auxiliary_client import _is_connection_error
        err = Exception("Bad Request: invalid model")
        err.status_code = 400
        assert _is_connection_error(err) is False

    def test_500_not_connection(self):
        from agent.auxiliary_client import _is_connection_error
        err = Exception("Internal Server Error")
        err.status_code = 500
        assert _is_connection_error(err) is False


class TestKimiTemperatureOmitted:
    """Kimi/Moonshot models should have temperature OMITTED from API kwargs.

    The Kimi gateway selects the correct temperature server-side based on the
    active mode (thinking → 1.0, non-thinking → 0.6).  Sending any temperature
    value conflicts with gateway-managed defaults.
    """

    @pytest.mark.parametrize(
        "model",
        [
            "kimi-for-coding",
            "kimi-k2.5",
            "kimi-k2.6",
            "kimi-k2-turbo-preview",
            "kimi-k2-0905-preview",
            "kimi-k2-thinking",
            "kimi-k2-thinking-turbo",
            "kimi-k2-instruct",
            "kimi-k2-instruct-0905",
            "moonshotai/kimi-k2.5",
            "moonshotai/Kimi-K2-Thinking",
            "moonshotai/Kimi-K2-Instruct",
        ],
    )
    def test_kimi_models_omit_temperature(self, model):
        """No kimi model should have a temperature key in kwargs."""
        from agent.auxiliary_client import _build_call_kwargs

        kwargs = _build_call_kwargs(
            provider="kimi-coding",
            model=model,
            messages=[{"role": "user", "content": "hello"}],
            temperature=0.3,
        )

        assert "temperature" not in kwargs

    def test_kimi_for_coding_no_temperature_when_none(self):
        """When caller passes temperature=None, still no temperature key."""
        from agent.auxiliary_client import _build_call_kwargs

        kwargs = _build_call_kwargs(
            provider="kimi-coding",
            model="kimi-for-coding",
            messages=[{"role": "user", "content": "hello"}],
            temperature=None,
        )

        assert "temperature" not in kwargs

    def test_sync_call_omits_temperature(self):
        client = MagicMock()
        client.base_url = "https://api.kimi.com/coding/v1"
        response = MagicMock()
        client.chat.completions.create.return_value = response

        with patch(
            "agent.auxiliary_client._get_cached_client",
            return_value=(client, "kimi-for-coding"),
        ), patch(
            "agent.auxiliary_client._resolve_task_provider_model",
            return_value=("auto", "kimi-for-coding", None, None, None),
        ):
            result = call_llm(
                task="session_search",
                messages=[{"role": "user", "content": "hello"}],
                temperature=0.1,
            )

        assert result is response
        kwargs = client.chat.completions.create.call_args.kwargs
        assert kwargs["model"] == "kimi-for-coding"
        assert "temperature" not in kwargs

    @pytest.mark.asyncio
    async def test_async_call_omits_temperature(self):
        client = MagicMock()
        client.base_url = "https://api.kimi.com/coding/v1"
        response = MagicMock()
        client.chat.completions.create = AsyncMock(return_value=response)

        with patch(
            "agent.auxiliary_client._get_cached_client",
            return_value=(client, "kimi-for-coding"),
        ), patch(
            "agent.auxiliary_client._resolve_task_provider_model",
            return_value=("auto", "kimi-for-coding", None, None, None),
        ):
            result = await async_call_llm(
                task="session_search",
                messages=[{"role": "user", "content": "hello"}],
                temperature=0.1,
            )

        assert result is response
        kwargs = client.chat.completions.create.call_args.kwargs
        assert kwargs["model"] == "kimi-for-coding"
        assert "temperature" not in kwargs

    @pytest.mark.parametrize(
        "model",
        [
            "anthropic/claude-sonnet-4-6",
            "gpt-5.4",
            "deepseek-chat",
        ],
    )
    def test_non_kimi_models_preserve_temperature(self, model):
        from agent.auxiliary_client import _build_call_kwargs

        kwargs = _build_call_kwargs(
            provider="openrouter",
            model=model,
            messages=[{"role": "user", "content": "hello"}],
            temperature=0.3,
        )

        assert kwargs["temperature"] == 0.3

    @pytest.mark.parametrize(
        "base_url",
        [
            "https://api.moonshot.ai/v1",
            "https://api.moonshot.cn/v1",
            "https://api.kimi.com/coding/v1",
        ],
    )
    def test_kimi_k2_5_omits_temperature_regardless_of_endpoint(self, base_url):
        """Temperature is omitted regardless of which Kimi endpoint is used."""
        from agent.auxiliary_client import _build_call_kwargs

        kwargs = _build_call_kwargs(
            provider="kimi-coding",
            model="kimi-k2.5",
            messages=[{"role": "user", "content": "hello"}],
            temperature=0.1,
            base_url=base_url,
        )

        assert "temperature" not in kwargs


# ---------------------------------------------------------------------------
# async_call_llm payment / connection fallback (#7512 bug 2)
# ---------------------------------------------------------------------------


class TestStaleBaseUrlWarning:
    """_resolve_auto() warns when OPENAI_BASE_URL conflicts with config provider (#5161)."""

    def test_warns_when_openai_base_url_set_with_named_provider(self, monkeypatch, caplog):
        """Warning fires when OPENAI_BASE_URL is set but provider is a named provider."""
        import agent.auxiliary_client as mod
        # Reset the module-level flag so the warning fires
        monkeypatch.setattr(mod, "_stale_base_url_warned", False)
        monkeypatch.setenv("OPENAI_BASE_URL", "http://localhost:11434/v1")
        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")

        with patch("agent.auxiliary_client._read_main_provider", return_value="openrouter"), \
             patch("agent.auxiliary_client._read_main_model", return_value="google/gemini-flash"), \
             caplog.at_level(logging.WARNING, logger="agent.auxiliary_client"):
            _resolve_auto()

        assert any("OPENAI_BASE_URL is set" in rec.message for rec in caplog.records), \
            "Expected a warning about stale OPENAI_BASE_URL"
        assert mod._stale_base_url_warned is True


class TestAuxiliaryTaskExtraBody:
    def test_sync_call_merges_task_extra_body_from_config(self):
        client = MagicMock()
        client.base_url = "https://api.example.com/v1"
        response = MagicMock()
        client.chat.completions.create.return_value = response

        config = {
            "auxiliary": {
                "session_search": {
                    "extra_body": {
                        "enable_thinking": False,
                        "reasoning": {"effort": "none"},
                    }
                }
            }
        }

        with patch("hermes_cli.config.load_config", return_value=config), patch(
            "agent.auxiliary_client._get_cached_client",
            return_value=(client, "glm-4.5-air"),
        ):
            result = call_llm(
                task="session_search",
                messages=[{"role": "user", "content": "hello"}],
                extra_body={"metadata": {"source": "test"}},
            )

        assert result is response
        kwargs = client.chat.completions.create.call_args.kwargs
        assert kwargs["extra_body"]["enable_thinking"] is False
        assert kwargs["extra_body"]["reasoning"] == {"effort": "none"}
        assert kwargs["extra_body"]["metadata"] == {"source": "test"}

    @pytest.mark.asyncio
    async def test_async_call_explicit_extra_body_overrides_task_config(self):
        client = MagicMock()
        client.base_url = "https://api.example.com/v1"
        response = MagicMock()
        client.chat.completions.create = AsyncMock(return_value=response)

        config = {
            "auxiliary": {
                "session_search": {
                    "extra_body": {"enable_thinking": False}
                }
            }
        }

        with patch("hermes_cli.config.load_config", return_value=config), patch(
            "agent.auxiliary_client._get_cached_client",
            return_value=(client, "glm-4.5-air"),
        ):
            result = await async_call_llm(
                task="session_search",
                messages=[{"role": "user", "content": "hello"}],
                extra_body={"enable_thinking": True},
            )

        assert result is response
        kwargs = client.chat.completions.create.call_args.kwargs
        assert kwargs["extra_body"]["enable_thinking"] is True

    def test_no_warning_when_provider_is_custom(self, monkeypatch, caplog):
        """No warning when the provider is 'custom' — OPENAI_BASE_URL is expected."""
        import agent.auxiliary_client as mod
        monkeypatch.setattr(mod, "_stale_base_url_warned", False)
        monkeypatch.setenv("OPENAI_BASE_URL", "http://localhost:11434/v1")
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")

        with patch("agent.auxiliary_client._read_main_provider", return_value="custom"), \
             patch("agent.auxiliary_client._read_main_model", return_value="llama3"), \
             patch("agent.auxiliary_client._resolve_custom_runtime",
                   return_value=("http://localhost:11434/v1", "test-key", None)), \
             patch("agent.auxiliary_client.OpenAI") as mock_openai, \
             caplog.at_level(logging.WARNING, logger="agent.auxiliary_client"):
            mock_openai.return_value = MagicMock()
            _resolve_auto()

        assert not any("OPENAI_BASE_URL is set" in rec.message for rec in caplog.records), \
            "Should NOT warn when provider is 'custom'"

    def test_no_warning_when_provider_is_named_custom(self, monkeypatch, caplog):
        """No warning when the provider is 'custom:myname' — base_url comes from config."""
        import agent.auxiliary_client as mod
        monkeypatch.setattr(mod, "_stale_base_url_warned", False)
        monkeypatch.setenv("OPENAI_BASE_URL", "http://localhost:11434/v1")
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")

        with patch("agent.auxiliary_client._read_main_provider", return_value="custom:ollama-local"), \
             patch("agent.auxiliary_client._read_main_model", return_value="llama3"), \
             patch("agent.auxiliary_client.resolve_provider_client",
                   return_value=(MagicMock(), "llama3")), \
             caplog.at_level(logging.WARNING, logger="agent.auxiliary_client"):
            _resolve_auto()

        assert not any("OPENAI_BASE_URL is set" in rec.message for rec in caplog.records), \
            "Should NOT warn when provider is 'custom:*'"

    def test_no_warning_when_openai_base_url_not_set(self, monkeypatch, caplog):
        """No warning when OPENAI_BASE_URL is absent."""
        import agent.auxiliary_client as mod
        monkeypatch.setattr(mod, "_stale_base_url_warned", False)
        monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")

        with patch("agent.auxiliary_client._read_main_provider", return_value="openrouter"), \
             patch("agent.auxiliary_client._read_main_model", return_value="google/gemini-flash"), \
             caplog.at_level(logging.WARNING, logger="agent.auxiliary_client"):
            _resolve_auto()

        assert not any("OPENAI_BASE_URL is set" in rec.message for rec in caplog.records), \
            "Should NOT warn when OPENAI_BASE_URL is not set"

# ---------------------------------------------------------------------------
# Anthropic-compatible image block conversion
# ---------------------------------------------------------------------------

class TestAnthropicCompatImageConversion:
    """Tests for _is_anthropic_compat_endpoint and _convert_openai_images_to_anthropic."""

    def test_known_providers_detected(self):
        from agent.auxiliary_client import _is_anthropic_compat_endpoint
        assert _is_anthropic_compat_endpoint("minimax", "")
        assert _is_anthropic_compat_endpoint("minimax-cn", "")

    def test_openrouter_not_detected(self):
        from agent.auxiliary_client import _is_anthropic_compat_endpoint
        assert not _is_anthropic_compat_endpoint("openrouter", "")
        assert not _is_anthropic_compat_endpoint("anthropic", "")

    def test_url_based_detection(self):
        from agent.auxiliary_client import _is_anthropic_compat_endpoint
        assert _is_anthropic_compat_endpoint("custom", "https://api.minimax.io/anthropic")
        assert _is_anthropic_compat_endpoint("custom", "https://example.com/anthropic/v1")
        assert not _is_anthropic_compat_endpoint("custom", "https://api.openai.com/v1")

    def test_base64_image_converted(self):
        from agent.auxiliary_client import _convert_openai_images_to_anthropic
        messages = [{
            "role": "user",
            "content": [
                {"type": "text", "text": "describe"},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,iVBOR="}}
            ]
        }]
        result = _convert_openai_images_to_anthropic(messages)
        img_block = result[0]["content"][1]
        assert img_block["type"] == "image"
        assert img_block["source"]["type"] == "base64"
        assert img_block["source"]["media_type"] == "image/png"
        assert img_block["source"]["data"] == "iVBOR="

    def test_url_image_converted(self):
        from agent.auxiliary_client import _convert_openai_images_to_anthropic
        messages = [{
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": "https://example.com/img.jpg"}}
            ]
        }]
        result = _convert_openai_images_to_anthropic(messages)
        img_block = result[0]["content"][0]
        assert img_block["type"] == "image"
        assert img_block["source"]["type"] == "url"
        assert img_block["source"]["url"] == "https://example.com/img.jpg"

    def test_text_only_messages_unchanged(self):
        from agent.auxiliary_client import _convert_openai_images_to_anthropic
        messages = [{"role": "user", "content": "Hello"}]
        result = _convert_openai_images_to_anthropic(messages)
        assert result[0] is messages[0]  # same object, not copied

    def test_jpeg_media_type_parsed(self):
        from agent.auxiliary_client import _convert_openai_images_to_anthropic
        messages = [{
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,/9j/="}}
            ]
        }]
        result = _convert_openai_images_to_anthropic(messages)
        assert result[0]["content"][0]["source"]["media_type"] == "image/jpeg"


class _AuxAuth401(Exception):
    status_code = 401

    def __init__(self, message="Provided authentication token is expired"):
        super().__init__(message)


class _DummyResponse:
    def __init__(self, text="ok"):
        self.choices = [MagicMock(message=MagicMock(content=text))]


class _FailingThenSuccessCompletions:
    def __init__(self):
        self.calls = 0

    def create(self, **kwargs):
        self.calls += 1
        if self.calls == 1:
            raise _AuxAuth401()
        return _DummyResponse("sync-ok")


class _AsyncFailingThenSuccessCompletions:
    def __init__(self):
        self.calls = 0

    async def create(self, **kwargs):
        self.calls += 1
        if self.calls == 1:
            raise _AuxAuth401()
        return _DummyResponse("async-ok")


class TestAuxiliaryAuthRefreshRetry:
    def test_call_llm_refreshes_codex_on_401_for_vision(self):
        failing_client = MagicMock()
        failing_client.base_url = "https://chatgpt.com/backend-api/codex"
        failing_client.chat.completions = _FailingThenSuccessCompletions()

        fresh_client = MagicMock()
        fresh_client.base_url = "https://chatgpt.com/backend-api/codex"
        fresh_client.chat.completions.create.return_value = _DummyResponse("fresh-sync")

        with (
            patch(
                "agent.auxiliary_client.resolve_vision_provider_client",
                side_effect=[("openai-codex", failing_client, "gpt-5.4"), ("openai-codex", fresh_client, "gpt-5.4")],
            ),
            patch("agent.auxiliary_client._refresh_provider_credentials", return_value=True) as mock_refresh,
        ):
            resp = call_llm(
                task="vision",
                provider="openai-codex",
                model="gpt-5.4",
                messages=[{"role": "user", "content": "hi"}],
            )

        assert resp.choices[0].message.content == "fresh-sync"
        mock_refresh.assert_called_once_with("openai-codex")

    def test_call_llm_refreshes_codex_on_401_for_non_vision(self):
        stale_client = MagicMock()
        stale_client.base_url = "https://chatgpt.com/backend-api/codex"
        stale_client.chat.completions.create.side_effect = _AuxAuth401("stale codex token")

        fresh_client = MagicMock()
        fresh_client.base_url = "https://chatgpt.com/backend-api/codex"
        fresh_client.chat.completions.create.return_value = _DummyResponse("fresh-non-vision")

        with (
            patch("agent.auxiliary_client._resolve_task_provider_model", return_value=("openai-codex", "gpt-5.4", None, None, None)),
            patch("agent.auxiliary_client._get_cached_client", side_effect=[(stale_client, "gpt-5.4"), (fresh_client, "gpt-5.4")]),
            patch("agent.auxiliary_client._refresh_provider_credentials", return_value=True) as mock_refresh,
        ):
            resp = call_llm(
                task="compression",
                provider="openai-codex",
                model="gpt-5.4",
                messages=[{"role": "user", "content": "hi"}],
            )

        assert resp.choices[0].message.content == "fresh-non-vision"
        mock_refresh.assert_called_once_with("openai-codex")
        assert stale_client.chat.completions.create.call_count == 1
        assert fresh_client.chat.completions.create.call_count == 1

    def test_call_llm_refreshes_anthropic_on_401_for_non_vision(self):
        stale_client = MagicMock()
        stale_client.base_url = "https://api.anthropic.com"
        stale_client.chat.completions.create.side_effect = _AuxAuth401("anthropic token expired")

        fresh_client = MagicMock()
        fresh_client.base_url = "https://api.anthropic.com"
        fresh_client.chat.completions.create.return_value = _DummyResponse("fresh-anthropic")

        with (
            patch("agent.auxiliary_client._resolve_task_provider_model", return_value=("anthropic", "claude-haiku-4-5-20251001", None, None, None)),
            patch("agent.auxiliary_client._get_cached_client", side_effect=[(stale_client, "claude-haiku-4-5-20251001"), (fresh_client, "claude-haiku-4-5-20251001")]),
            patch("agent.auxiliary_client._refresh_provider_credentials", return_value=True) as mock_refresh,
        ):
            resp = call_llm(
                task="compression",
                provider="anthropic",
                model="claude-haiku-4-5-20251001",
                messages=[{"role": "user", "content": "hi"}],
            )

        assert resp.choices[0].message.content == "fresh-anthropic"
        mock_refresh.assert_called_once_with("anthropic")
        assert stale_client.chat.completions.create.call_count == 1
        assert fresh_client.chat.completions.create.call_count == 1

    @pytest.mark.asyncio
    async def test_async_call_llm_refreshes_codex_on_401_for_vision(self):
        failing_client = MagicMock()
        failing_client.base_url = "https://chatgpt.com/backend-api/codex"
        failing_client.chat.completions = _AsyncFailingThenSuccessCompletions()

        fresh_client = MagicMock()
        fresh_client.base_url = "https://chatgpt.com/backend-api/codex"
        fresh_client.chat.completions.create = AsyncMock(return_value=_DummyResponse("fresh-async"))

        with (
            patch(
                "agent.auxiliary_client.resolve_vision_provider_client",
                side_effect=[("openai-codex", failing_client, "gpt-5.4"), ("openai-codex", fresh_client, "gpt-5.4")],
            ),
            patch("agent.auxiliary_client._refresh_provider_credentials", return_value=True) as mock_refresh,
        ):
            resp = await async_call_llm(
                task="vision",
                provider="openai-codex",
                model="gpt-5.4",
                messages=[{"role": "user", "content": "hi"}],
            )

        assert resp.choices[0].message.content == "fresh-async"
        mock_refresh.assert_called_once_with("openai-codex")

    def test_refresh_provider_credentials_force_refreshes_anthropic_oauth_and_evicts_cache(self, monkeypatch):
        stale_client = MagicMock()
        cache_key = ("anthropic", False, None, None, None)

        monkeypatch.setenv("ANTHROPIC_TOKEN", "")
        monkeypatch.setenv("CLAUDE_CODE_OAUTH_TOKEN", "")
        monkeypatch.setenv("ANTHROPIC_API_KEY", "")

        with (
            patch("agent.auxiliary_client._client_cache", {cache_key: (stale_client, "claude-haiku-4-5-20251001", None)}),
            patch("agent.anthropic_adapter.read_claude_code_credentials", return_value={
                "accessToken": "expired-token",
                "refreshToken": "refresh-token",
                "expiresAt": 0,
            }),
            patch("agent.anthropic_adapter.refresh_anthropic_oauth_pure", return_value={
                "access_token": "fresh-token",
                "refresh_token": "refresh-token-2",
                "expires_at_ms": 9999999999999,
            }) as mock_refresh_oauth,
            patch("agent.anthropic_adapter._write_claude_code_credentials") as mock_write,
        ):
            from agent.auxiliary_client import _refresh_provider_credentials

            assert _refresh_provider_credentials("anthropic") is True

        mock_refresh_oauth.assert_called_once_with("refresh-token", use_json=False)
        mock_write.assert_called_once_with("fresh-token", "refresh-token-2", 9999999999999)
        stale_client.close.assert_called_once()

    @pytest.mark.asyncio
    async def test_async_call_llm_refreshes_anthropic_on_401_for_non_vision(self):
        stale_client = MagicMock()
        stale_client.base_url = "https://api.anthropic.com"
        stale_client.chat.completions.create = AsyncMock(side_effect=_AuxAuth401("anthropic token expired"))

        fresh_client = MagicMock()
        fresh_client.base_url = "https://api.anthropic.com"
        fresh_client.chat.completions.create = AsyncMock(return_value=_DummyResponse("fresh-async-anthropic"))

        with (
            patch("agent.auxiliary_client._resolve_task_provider_model", return_value=("anthropic", "claude-haiku-4-5-20251001", None, None, None)),
            patch("agent.auxiliary_client._get_cached_client", side_effect=[(stale_client, "claude-haiku-4-5-20251001"), (fresh_client, "claude-haiku-4-5-20251001")]),
            patch("agent.auxiliary_client._refresh_provider_credentials", return_value=True) as mock_refresh,
        ):
            resp = await async_call_llm(
                task="compression",
                provider="anthropic",
                model="claude-haiku-4-5-20251001",
                messages=[{"role": "user", "content": "hi"}],
            )

        assert resp.choices[0].message.content == "fresh-async-anthropic"
        mock_refresh.assert_called_once_with("anthropic")
        assert stale_client.chat.completions.create.await_count == 1
        assert fresh_client.chat.completions.create.await_count == 1


class TestAuxiliaryPoolRotationRetry:
    def test_call_llm_rotates_explicit_codex_pool_on_429(self):
        rate_err = Exception("usage limit reached")
        rate_err.status_code = 429

        stale_client = MagicMock()
        stale_client.base_url = "https://chatgpt.com/backend-api/codex"
        stale_client.chat.completions.create.side_effect = [rate_err, rate_err]

        fresh_client = MagicMock()
        fresh_client.base_url = "https://chatgpt.com/backend-api/codex"
        fresh_client.chat.completions.create.return_value = _DummyResponse("rotated-sync")

        class _Pool:
            def __init__(self):
                self.rotate_calls = []

            def has_credentials(self):
                return True

            def try_refresh_current(self):
                return None

            def mark_exhausted_and_rotate(self, **kwargs):
                self.rotate_calls.append(kwargs)
                return SimpleNamespace(id="cred-b")

        pool = _Pool()

        with (
            patch("agent.auxiliary_client._resolve_task_provider_model", return_value=("openai-codex", "gpt-5.4", None, None, None)),
            patch("agent.auxiliary_client._get_cached_client", side_effect=[(stale_client, "gpt-5.4"), (fresh_client, "gpt-5.4")]),
            patch("agent.auxiliary_client._refresh_provider_credentials", return_value=False),
            patch("agent.auxiliary_client.load_pool", return_value=pool),
            patch("agent.auxiliary_client._try_payment_fallback") as mock_fallback,
        ):
            resp = call_llm(
                task="compression",
                provider="openai-codex",
                model="gpt-5.4",
                messages=[{"role": "user", "content": "hi"}],
            )

        assert resp.choices[0].message.content == "rotated-sync"
        assert stale_client.chat.completions.create.call_count == 2
        assert fresh_client.chat.completions.create.call_count == 1
        assert len(pool.rotate_calls) == 1
        assert pool.rotate_calls[0]["status_code"] == 429
        mock_fallback.assert_not_called()

    @pytest.mark.asyncio
    async def test_async_call_llm_rotates_explicit_codex_pool_on_429(self):
        rate_err = Exception("usage limit reached")
        rate_err.status_code = 429

        stale_client = MagicMock()
        stale_client.base_url = "https://chatgpt.com/backend-api/codex"
        stale_client.chat.completions.create = AsyncMock(side_effect=[rate_err, rate_err])

        fresh_client = MagicMock()
        fresh_client.base_url = "https://chatgpt.com/backend-api/codex"
        fresh_client.chat.completions.create = AsyncMock(return_value=_DummyResponse("rotated-async"))

        class _Pool:
            def __init__(self):
                self.rotate_calls = []

            def has_credentials(self):
                return True

            def try_refresh_current(self):
                return None

            def mark_exhausted_and_rotate(self, **kwargs):
                self.rotate_calls.append(kwargs)
                return SimpleNamespace(id="cred-b")

        pool = _Pool()

        with (
            patch("agent.auxiliary_client._resolve_task_provider_model", return_value=("openai-codex", "gpt-5.4", None, None, None)),
            patch("agent.auxiliary_client._get_cached_client", side_effect=[(stale_client, "gpt-5.4"), (fresh_client, "gpt-5.4")]),
            patch("agent.auxiliary_client._refresh_provider_credentials", return_value=False),
            patch("agent.auxiliary_client.load_pool", return_value=pool),
            patch("agent.auxiliary_client._try_payment_fallback") as mock_fallback,
        ):
            resp = await async_call_llm(
                task="compression",
                provider="openai-codex",
                model="gpt-5.4",
                messages=[{"role": "user", "content": "hi"}],
            )

        assert resp.choices[0].message.content == "rotated-async"
        assert stale_client.chat.completions.create.await_count == 2
        assert fresh_client.chat.completions.create.await_count == 1
        assert len(pool.rotate_calls) == 1
        assert pool.rotate_calls[0]["status_code"] == 429
        mock_fallback.assert_not_called()


class TestCodexAdapterReasoningTranslation:
    """Verify _CodexCompletionsAdapter translates extra_body.reasoning
    into the Responses API's top-level reasoning + include fields, matching
    agent/transports/codex.py::build_kwargs() behavior.

    Regression for user feedback (Apr 26): auxiliary callers that configure
    reasoning via auxiliary.<task>.extra_body.reasoning had that config
    silently dropped because the adapter only forwarded messages/model/tools.
    """

    @staticmethod
    def _build_adapter():
        """Build a _CodexCompletionsAdapter with a mocked responses.stream()."""
        from agent.auxiliary_client import _CodexCompletionsAdapter
        from types import SimpleNamespace

        # Mock the stream context manager: yields no events, get_final_response
        # returns a minimal empty-output response.
        fake_final = SimpleNamespace(
            output=[SimpleNamespace(
                type="message",
                content=[SimpleNamespace(type="output_text", text="hi")],
            )],
            usage=SimpleNamespace(input_tokens=1, output_tokens=1, total_tokens=2),
        )

        class _FakeStream:
            def __enter__(self): return self
            def __exit__(self, *a): return False
            def __iter__(self): return iter([])
            def get_final_response(self): return fake_final

        captured_kwargs = {}

        def _stream(**kwargs):
            captured_kwargs.update(kwargs)
            return _FakeStream()

        real_client = MagicMock()
        real_client.responses.stream = _stream
        adapter = _CodexCompletionsAdapter(real_client, "gpt-5.3-codex")
        return adapter, captured_kwargs

    def test_reasoning_effort_medium_translated_to_top_level(self):
        adapter, captured = self._build_adapter()
        adapter.create(
            messages=[{"role": "user", "content": "hi"}],
            extra_body={"reasoning": {"effort": "medium"}},
        )
        assert captured.get("reasoning") == {"effort": "medium", "summary": "auto"}
        assert captured.get("include") == ["reasoning.encrypted_content"]

    def test_reasoning_effort_minimal_clamped_to_low(self):
        """Codex backend rejects 'minimal'; adapter clamps to 'low' per main transport."""
        adapter, captured = self._build_adapter()
        adapter.create(
            messages=[{"role": "user", "content": "hi"}],
            extra_body={"reasoning": {"effort": "minimal"}},
        )
        assert captured.get("reasoning") == {"effort": "low", "summary": "auto"}
        assert captured.get("include") == ["reasoning.encrypted_content"]

    def test_reasoning_effort_low_passed_through(self):
        adapter, captured = self._build_adapter()
        adapter.create(
            messages=[{"role": "user", "content": "hi"}],
            extra_body={"reasoning": {"effort": "low"}},
        )
        assert captured.get("reasoning") == {"effort": "low", "summary": "auto"}

    def test_reasoning_effort_high_passed_through(self):
        adapter, captured = self._build_adapter()
        adapter.create(
            messages=[{"role": "user", "content": "hi"}],
            extra_body={"reasoning": {"effort": "high"}},
        )
        assert captured.get("reasoning") == {"effort": "high", "summary": "auto"}

    def test_reasoning_disabled_omits_reasoning_and_include(self):
        adapter, captured = self._build_adapter()
        adapter.create(
            messages=[{"role": "user", "content": "hi"}],
            extra_body={"reasoning": {"enabled": False}},
        )
        assert "reasoning" not in captured
        assert "include" not in captured

    def test_reasoning_default_effort_when_only_enabled_flag(self):
        """extra_body={"reasoning": {}} (truthy enabled by omission) → default 'medium'."""
        adapter, captured = self._build_adapter()
        adapter.create(
            messages=[{"role": "user", "content": "hi"}],
            extra_body={"reasoning": {}},
        )
        assert captured.get("reasoning") == {"effort": "medium", "summary": "auto"}
        assert captured.get("include") == ["reasoning.encrypted_content"]

    def test_no_extra_body_means_no_reasoning_keys(self):
        """Baseline: without extra_body, no reasoning/include is sent (preserves
        current behavior for callers that don't opt in)."""
        adapter, captured = self._build_adapter()
        adapter.create(messages=[{"role": "user", "content": "hi"}])
        assert "reasoning" not in captured
        assert "include" not in captured

    def test_extra_body_without_reasoning_key_is_noop(self):
        adapter, captured = self._build_adapter()
        adapter.create(
            messages=[{"role": "user", "content": "hi"}],
            extra_body={"metadata": {"source": "test"}},
        )
        assert "reasoning" not in captured
        assert "include" not in captured

    def test_non_dict_reasoning_value_is_ignored_gracefully(self):
        """Defensive: if a caller accidentally passes a string/None, we
        silently skip instead of crashing inside the adapter."""
        adapter, captured = self._build_adapter()
        adapter.create(
            messages=[{"role": "user", "content": "hi"}],
            extra_body={"reasoning": "medium"},  # wrong shape — must not crash
        )
        assert "reasoning" not in captured

    def test_reasoning_effort_null_falls_back_to_medium(self):
        """Parity with agent/transports/codex.py::build_kwargs() — falsy
        ``effort`` (None / empty / 0) keeps the default ``medium`` instead
        of being forwarded to Codex.  Codex rejects ``{"effort": null}``
        with HTTP 400 (Invalid value for parameter `reasoning.effort`)."""
        adapter, captured = self._build_adapter()
        adapter.create(
            messages=[{"role": "user", "content": "hi"}],
            extra_body={"reasoning": {"effort": None}},
        )
        assert captured.get("reasoning") == {"effort": "medium", "summary": "auto"}
        assert captured.get("include") == ["reasoning.encrypted_content"]

    def test_reasoning_effort_empty_string_falls_back_to_medium(self):
        """Empty-string effort (e.g. ``effort: ""`` in YAML) is falsy in
        the main-agent path's truthy check; mirror that here so the same
        config produces the same result."""
        adapter, captured = self._build_adapter()
        adapter.create(
            messages=[{"role": "user", "content": "hi"}],
            extra_body={"reasoning": {"effort": ""}},
        )
        assert captured.get("reasoning") == {"effort": "medium", "summary": "auto"}
        assert captured.get("include") == ["reasoning.encrypted_content"]

    def test_reasoning_effort_zero_falls_back_to_medium(self):
        """Numeric ``0`` is also falsy — the docstring lists it explicitly,
        so cover the contract.  Codex would reject ``{"effort": 0}`` the
        same way it rejects ``null``."""
        adapter, captured = self._build_adapter()
        adapter.create(
            messages=[{"role": "user", "content": "hi"}],
            extra_body={"reasoning": {"effort": 0}},
        )
        assert captured.get("reasoning") == {"effort": "medium", "summary": "auto"}
        assert captured.get("include") == ["reasoning.encrypted_content"]


class TestVisionAutoSkipsKimiCoding:
    """_resolve_auto vision branch skips providers that have no vision on
    their main endpoint (e.g. Kimi Coding Plan /coding) and falls through
    to the aggregator chain instead of handing back a client that will 404
    on every request (#17076).
    """

    def test_kimi_coding_skipped_falls_through_to_openrouter(self, monkeypatch):
        """kimi-coding as main + vision auto → OpenRouter (not kimi)."""
        fake_or_client = MagicMock(name="openrouter_client")

        monkeypatch.setattr(
            "agent.auxiliary_client._read_main_provider", lambda: "kimi-coding",
        )
        monkeypatch.setattr(
            "agent.auxiliary_client._read_main_model", lambda: "kimi-code",
        )
        # Guard: if the skip doesn't fire, _resolve_strict_vision_backend
        # and resolve_provider_client both would try kimi-coding — detect
        # either via the main-provider call and fail loud.
        rpc_mock = MagicMock(side_effect=AssertionError(
            "resolve_provider_client should NOT be called for kimi-coding "
            "on the vision auto path"))
        monkeypatch.setattr(
            "agent.auxiliary_client.resolve_provider_client", rpc_mock,
        )

        def fake_strict(provider, model=None):
            if provider == "openrouter":
                return fake_or_client, "google/gemini-3-flash-preview"
            if provider == "nous":
                return None, None
            raise AssertionError(
                f"strict vision backend should not be called for {provider!r} "
                "when main provider is kimi-coding"
            )
        monkeypatch.setattr(
            "agent.auxiliary_client._resolve_strict_vision_backend",
            fake_strict,
        )

        provider, client, model = resolve_vision_provider_client()
        assert provider == "openrouter"
        assert client is fake_or_client
        assert model == "google/gemini-3-flash-preview"

    def test_kimi_coding_cn_skipped_too(self, monkeypatch):
        """Same skip applies to the CN variant."""
        fake_or_client = MagicMock(name="openrouter_client")

        monkeypatch.setattr(
            "agent.auxiliary_client._read_main_provider", lambda: "kimi-coding-cn",
        )
        monkeypatch.setattr(
            "agent.auxiliary_client._read_main_model", lambda: "kimi-code",
        )
        rpc_mock = MagicMock(side_effect=AssertionError(
            "resolve_provider_client should NOT be called for kimi-coding-cn"))
        monkeypatch.setattr(
            "agent.auxiliary_client.resolve_provider_client", rpc_mock,
        )
        monkeypatch.setattr(
            "agent.auxiliary_client._resolve_strict_vision_backend",
            lambda p, m=None: (fake_or_client, "gemini")
            if p == "openrouter"
            else (None, None),
        )

        provider, client, _ = resolve_vision_provider_client()
        assert provider == "openrouter"
        assert client is fake_or_client

    def test_explicit_override_to_kimi_coding_still_honored(self, monkeypatch):
        """When a user *explicitly* requests kimi-coding for vision (e.g.
        they know what they're doing, or are running a future build that
        adds image_in capability to Kimi Code), the explicit path still
        routes to kimi-coding — only the auto branch applies the skip.
        """
        monkeypatch.setattr(
            "agent.auxiliary_client._read_main_provider", lambda: "openrouter",
        )
        fake_kimi_client = MagicMock(name="kimi_client")
        gcc_mock = MagicMock(return_value=(fake_kimi_client, "kimi-code"))
        monkeypatch.setattr(
            "agent.auxiliary_client._get_cached_client", gcc_mock,
        )

        provider, client, model = resolve_vision_provider_client(
            provider="kimi-coding",
        )
        assert provider == "kimi-coding"
        assert client is fake_kimi_client
        gcc_mock.assert_called_once()

    def test_skip_set_covers_exactly_known_entries(self):
        """Guard against accidental widening of the skip list."""
        from agent.auxiliary_client import _PROVIDERS_WITHOUT_VISION
        assert _PROVIDERS_WITHOUT_VISION == frozenset({
            "kimi-coding",
            "kimi-coding-cn",
        })


class TestCodexAuxiliaryAdapterTimeout:
    def test_forwards_timeout_to_responses_stream(self):
        class FakeStream:
            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def __iter__(self):
                return iter(())

            def get_final_response(self):
                return SimpleNamespace(
                    output=[SimpleNamespace(
                        type="message",
                        content=[SimpleNamespace(type="output_text", text="summary")],
                    )],
                    usage=None,
                )

        class FakeResponses:
            def __init__(self):
                self.kwargs = None

            def stream(self, **kwargs):
                self.kwargs = kwargs
                return FakeStream()

        fake_client = SimpleNamespace(responses=FakeResponses())
        adapter = _CodexCompletionsAdapter(fake_client, "gpt-5.5")

        response = adapter.create(
            messages=[{"role": "user", "content": "summarize this"}],
            timeout=12.5,
        )

        assert fake_client.responses.kwargs["timeout"] == 12.5
        assert response.choices[0].message.content == "summary"

    def test_enforces_total_timeout_while_stream_keeps_emitting_events(self):
        class SlowAliveStream:
            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def __iter__(self):
                for _ in range(5):
                    time.sleep(0.03)
                    yield SimpleNamespace(type="response.in_progress")

            def get_final_response(self):
                return SimpleNamespace(
                    output=[SimpleNamespace(
                        type="message",
                        content=[SimpleNamespace(type="output_text", text="late")],
                    )],
                    usage=None,
                )

        class FakeResponses:
            def stream(self, **kwargs):
                return SlowAliveStream()

        fake_client = SimpleNamespace(responses=FakeResponses(), close=lambda: None)
        adapter = _CodexCompletionsAdapter(fake_client, "gpt-5.5")

        started = time.monotonic()
        with pytest.raises(TimeoutError):
            adapter.create(
                messages=[{"role": "user", "content": "summarize this"}],
                timeout=0.05,
            )

        assert time.monotonic() - started < 0.14


# ---------------------------------------------------------------------------
# Issue #23432 — auxiliary timeout poisons cached client; later aux calls fail
# ---------------------------------------------------------------------------

class TestAuxiliaryClientPoisonedCacheEviction:
    """Connection/timeout errors must evict the cached aux client.

    Otherwise the next auxiliary call (compression retry, memory flush,
    background review) reuses the closed httpx transport and fails with
    ``Connection error`` even though the main provider route is healthy.
    See https://github.com/NousResearch/hermes-agent/issues/23432.
    """

    def test_evict_cached_client_instance_drops_direct_match(self):
        from agent.auxiliary_client import (
            _client_cache, _client_cache_lock, _evict_cached_client_instance,
        )

        target = MagicMock(name="target_client")
        other = MagicMock(name="other_client")
        with _client_cache_lock:
            _client_cache.clear()
            _client_cache[("openrouter", False, None, None, None)] = (target, "x", None)
            _client_cache[("anthropic", False, None, None, None)] = (other, "y", None)
        try:
            assert _evict_cached_client_instance(target) is True
            assert ("openrouter", False, None, None, None) not in _client_cache
            assert ("anthropic", False, None, None, None) in _client_cache
        finally:
            with _client_cache_lock:
                _client_cache.clear()

    def test_evict_cached_client_instance_walks_codex_wrapper(self):
        """Closing the underlying OpenAI client must evict the Codex shim."""
        from agent.auxiliary_client import (
            _client_cache, _client_cache_lock, _evict_cached_client_instance,
            CodexAuxiliaryClient,
        )

        real = SimpleNamespace(api_key="k", base_url="https://chatgpt.com/backend-api/codex",
                               responses=SimpleNamespace(stream=lambda **k: None),
                               close=lambda: None)
        wrapper = CodexAuxiliaryClient(real, "gpt-5.5")
        with _client_cache_lock:
            _client_cache.clear()
            _client_cache[("openai-codex", False, None, None, None)] = (wrapper, "gpt-5.5", None)
        try:
            # Eviction by the inner OpenAI client must remove the wrapper entry.
            assert _evict_cached_client_instance(real) is True
            assert ("openai-codex", False, None, None, None) not in _client_cache
        finally:
            with _client_cache_lock:
                _client_cache.clear()

    def test_evict_cached_client_instance_handles_none_and_misses(self):
        from agent.auxiliary_client import _evict_cached_client_instance

        assert _evict_cached_client_instance(None) is False
        assert _evict_cached_client_instance(MagicMock()) is False

    def test_evict_cached_client_instance_walks_async_wrapper(self):
        """async_mode is part of the cache key so sync and async share the same
        underlying OpenAI client across two distinct cache entries. A single
        timeout that closes the leaf must evict BOTH — otherwise the async
        entry survives, keeps reusing the dead transport, and every async
        aux call (compression, vision, session_search) fails fast with
        'Connection error' until gateway restart even while the sync route
        recovers.

        Regression for the async-side gap left by #23482, which fixed the
        sync wrapper's _real_client walk but missed the async wrappers.
        """
        from agent.auxiliary_client import (
            _client_cache, _client_cache_lock, _evict_cached_client_instance,
            CodexAuxiliaryClient, AsyncCodexAuxiliaryClient,
        )

        real = SimpleNamespace(api_key="k", base_url="https://chatgpt.com/backend-api/codex",
                               responses=SimpleNamespace(stream=lambda **k: None),
                               close=lambda: None)
        sync_wrapper = CodexAuxiliaryClient(real, "gpt-5.5")
        async_wrapper = AsyncCodexAuxiliaryClient(sync_wrapper)
        with _client_cache_lock:
            _client_cache.clear()
            _client_cache[("openai-codex", False, None, None, None)] = (sync_wrapper, "gpt-5.5", None)
            _client_cache[("openai-codex", True, None, None, None)] = (async_wrapper, "gpt-5.5", None)
        try:
            assert _evict_cached_client_instance(real) is True
            assert ("openai-codex", False, None, None, None) not in _client_cache
            assert ("openai-codex", True, None, None, None) not in _client_cache, (
                "async cache entry survived eviction — wrapper is missing _real_client"
            )
        finally:
            with _client_cache_lock:
                _client_cache.clear()

    def test_codex_timeout_evicts_cached_wrapper(self):
        """The timeout closer evicts the cache entry that wraps the closed client."""
        from agent.auxiliary_client import (
            _client_cache, _client_cache_lock,
            _CodexCompletionsAdapter, CodexAuxiliaryClient,
        )

        class SlowAliveStream:
            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def __iter__(self):
                for _ in range(20):
                    time.sleep(0.01)
                    yield SimpleNamespace(type="response.in_progress")

            def get_final_response(self):  # pragma: no cover — timeout fires first
                return SimpleNamespace(output=[], usage=None)

        closed = {"flag": False}

        class FakeClient:
            def __init__(self):
                self.responses = SimpleNamespace(stream=lambda **k: SlowAliveStream())
                self.api_key = "k"
                self.base_url = "https://chatgpt.com/backend-api/codex"

            def close(self):
                closed["flag"] = True

        fake_real = FakeClient()
        wrapper = CodexAuxiliaryClient(fake_real, "gpt-5.5")
        cache_key = ("openai-codex", False, None, None, None)
        with _client_cache_lock:
            _client_cache.clear()
            _client_cache[cache_key] = (wrapper, "gpt-5.5", None)
        try:
            adapter = _CodexCompletionsAdapter(fake_real, "gpt-5.5")
            with pytest.raises(TimeoutError):
                adapter.create(
                    messages=[{"role": "user", "content": "x"}],
                    timeout=0.05,
                )
            assert closed["flag"] is True, "timeout closer must close inner client"
            assert cache_key not in _client_cache, (
                "timeout closer must evict cache entry that wraps the closed client"
            )
        finally:
            with _client_cache_lock:
                _client_cache.clear()

    def test_call_llm_evicts_on_connection_error_with_explicit_provider(self):
        """Connection error on an explicit provider must drop the cached client.

        This is the exact reporter scenario: ``auxiliary.compression.provider:
        main`` (resolves to ``openai-codex``) → no fallback chain runs (not
        auto), but the cached client was poisoned by a prior timeout and must
        be evicted so the next call rebuilds.
        """
        from agent.auxiliary_client import _client_cache, _client_cache_lock

        poisoned = MagicMock(name="poisoned_client")
        poisoned.base_url = "https://chatgpt.com/backend-api/codex"
        poisoned.chat.completions.create.side_effect = ConnectionError("transport closed")

        cache_key = ("openai-codex", False, None, None, None)
        with _client_cache_lock:
            _client_cache.clear()
            _client_cache[cache_key] = (poisoned, "gpt-5.5", None)

        try:
            with patch(
                "agent.auxiliary_client._resolve_task_provider_model",
                return_value=("openai-codex", "gpt-5.5", None, None, None),
            ), patch(
                "agent.auxiliary_client._get_cached_client",
                return_value=(poisoned, "gpt-5.5"),
            ):
                with pytest.raises(ConnectionError):
                    call_llm(
                        task="compression",
                        messages=[{"role": "user", "content": "x"}],
                    )
            assert cache_key not in _client_cache, (
                "connection error must evict cached client so the next call rebuilds"
            )
        finally:
            with _client_cache_lock:
                _client_cache.clear()

    @pytest.mark.asyncio
    async def test_async_call_llm_evicts_on_connection_error_with_explicit_provider(self):
        from agent.auxiliary_client import _client_cache, _client_cache_lock

        poisoned = MagicMock(name="poisoned_async_client")
        poisoned.base_url = "https://chatgpt.com/backend-api/codex"
        poisoned.chat.completions.create = AsyncMock(side_effect=ConnectionError("transport closed"))

        cache_key = ("openai-codex", True, None, None, None)
        with _client_cache_lock:
            _client_cache.clear()
            _client_cache[cache_key] = (poisoned, "gpt-5.5", None)

        try:
            with patch(
                "agent.auxiliary_client._resolve_task_provider_model",
                return_value=("openai-codex", "gpt-5.5", None, None, None),
            ), patch(
                "agent.auxiliary_client._get_cached_client",
                return_value=(poisoned, "gpt-5.5"),
            ):
                with pytest.raises(ConnectionError):
                    await async_call_llm(
                        task="compression",
                        messages=[{"role": "user", "content": "x"}],
                    )
            assert cache_key not in _client_cache
        finally:
            with _client_cache_lock:
                _client_cache.clear()


# ---------------------------------------------------------------------------
# _build_call_kwargs — tool dedup at API boundary
# ---------------------------------------------------------------------------

class TestBuildCallKwargsToolDedup:
    """_build_call_kwargs must deduplicate tool names before passing to API.

    Providers like Google Vertex, Azure, and Bedrock reject requests with
    duplicate tool names (HTTP 400).  This guard converts a hard failure into
    a warning log so agent turns succeed even if an upstream injection path
    regresses.  See: https://github.com/NousResearch/hermes-agent/issues/18478
    """

    def _make_tool(self, name: str) -> dict:
        return {
            "type": "function",
            "function": {
                "name": name,
                "description": f"Tool {name}",
                "parameters": {"type": "object", "properties": {}},
            },
        }

    def test_unique_tools_pass_through_unchanged(self):
        tools = [self._make_tool("alpha"), self._make_tool("beta")]
        kwargs = _build_call_kwargs(
            provider="openai", model="gpt-4o", messages=[], tools=tools,
        )
        assert len(kwargs["tools"]) == 2
        names = [t["function"]["name"] for t in kwargs["tools"]]
        assert names == ["alpha", "beta"]

    def test_duplicate_tool_names_are_deduplicated(self):
        """RED test — must fail until dedup guard is added."""
        tools = [
            self._make_tool("lcm_grep"),
            self._make_tool("lcm_describe"),
            self._make_tool("lcm_grep"),  # duplicate
            self._make_tool("lcm_expand"),
            self._make_tool("lcm_describe"),  # duplicate
        ]
        kwargs = _build_call_kwargs(
            provider="google", model="gemini-2.5-pro", messages=[], tools=tools,
        )
        result_tools = kwargs["tools"]
        names = [t["function"]["name"] for t in result_tools]
        # Must be deduplicated — no repeated names
        assert len(names) == len(set(names)), (
            f"Duplicate tool names found: {names}"
        )
        assert len(result_tools) == 3  # lcm_grep, lcm_describe, lcm_expand

    def test_empty_tools_unchanged(self):
        kwargs = _build_call_kwargs(
            provider="openai", model="gpt-4o", messages=[], tools=[],
        )
        assert kwargs.get("tools") == [] or "tools" not in kwargs

    def test_none_tools_unchanged(self):
        kwargs = _build_call_kwargs(
            provider="openai", model="gpt-4o", messages=[], tools=None,
        )
        assert "tools" not in kwargs


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    """Strip provider env vars so each test starts clean."""
    for key in (
        "OPENROUTER_API_KEY", "OPENAI_BASE_URL", "OPENAI_API_KEY",
        "NVIDIA_API_KEY", "NVIDIA_BASE_URL",
    ):
        monkeypatch.delenv(key, raising=False)


class TestNvidiaBillingHeaders:
    """NVIDIA NIM billing-origin headers are scoped to NVIDIA cloud."""

    def test_resolve_provider_client_cloud_adds_billing_origin_header(self, monkeypatch):
        monkeypatch.setenv("NVIDIA_API_KEY", "nvidia-key")
        monkeypatch.delenv("NVIDIA_BASE_URL", raising=False)
        mock_openai = MagicMock()
        mock_openai.return_value = MagicMock(name="nvidia-client")

        with patch("agent.auxiliary_client.OpenAI", mock_openai):
            client, model = resolve_provider_client(
                provider="nvidia",
                model="nvidia/test-model",
            )

        assert client is not None
        assert model == "nvidia/test-model"
        call_kwargs = mock_openai.call_args[1]
        headers = call_kwargs["default_headers"]
        assert headers["X-BILLING-INVOKE-ORIGIN"] == "HermesAgent"

    def test_resolve_provider_client_local_nim_skips_billing_origin_header(self, monkeypatch):
        monkeypatch.setenv("NVIDIA_API_KEY", "nvidia-key")
        monkeypatch.setenv("NVIDIA_BASE_URL", "http://localhost:8000/v1")
        mock_openai = MagicMock()
        mock_openai.return_value = MagicMock(name="nvidia-local-client")

        with patch("agent.auxiliary_client.OpenAI", mock_openai):
            client, model = resolve_provider_client(
                provider="nvidia",
                model="nvidia/test-model",
            )

        assert client is not None
        assert model == "nvidia/test-model"
        call_kwargs = mock_openai.call_args[1]
        headers = call_kwargs.get("default_headers", {})
        assert "X-BILLING-INVOKE-ORIGIN" not in headers


class TestOpenRouterExplicitApiKey:
    """Test that explicit_api_key is correctly propagated to _try_openrouter()."""

    def test_resolve_provider_client_passes_explicit_api_key_to_openrouter(
        self, monkeypatch
    ):
        """
        When resolve_provider_client() is called with explicit_api_key for OpenRouter,
        the explicit key should be passed to the OpenAI client instead of falling back
        to OPENROUTER_API_KEY env var.
        """
        # Set up env var as fallback (should NOT be used when explicit_api_key is provided)
        monkeypatch.setenv("OPENROUTER_API_KEY", "env-fallback-key")

        # Mock OpenAI to capture the api_key used
        mock_openai = MagicMock()
        mock_openai.return_value = MagicMock(name="openrouter-client")

        with patch("agent.auxiliary_client.OpenAI", mock_openai):
            client, model = resolve_provider_client(
                provider="openrouter",
                explicit_api_key="explicit-pool-key",
            )

            # Verify a client was created
            assert client is not None
            # Verify the explicit key was used, not the env var fallback
            mock_openai.assert_called_once()
            call_kwargs = mock_openai.call_args[1]
            assert call_kwargs["api_key"] == "explicit-pool-key", (
                f"Expected explicit_api_key to be passed, got: {call_kwargs['api_key']}"
            )
            assert call_kwargs["api_key"] != "env-fallback-key", (
                "Should NOT fall back to OPENROUTER_API_KEY when explicit_api_key is provided"
            )

    def test_resolve_provider_client_without_explicit_api_key_falls_back_to_env(
        self, monkeypatch
    ):
        """
        When resolve_provider_client() is called WITHOUT explicit_api_key for OpenRouter,
        it should fall back to OPENROUTER_API_KEY env var.
        """
        # Set up env var as fallback (should be used when explicit_api_key is NOT provided)
        monkeypatch.setenv("OPENROUTER_API_KEY", "env-fallback-key")

        # Mock OpenAI to capture the api_key used
        mock_openai = MagicMock()
        mock_openai.return_value = MagicMock(name="openrouter-client")

        with patch("agent.auxiliary_client.OpenAI", mock_openai):
            client, model = resolve_provider_client(
                provider="openrouter",
                explicit_api_key=None,
            )

            # Verify a client was created
            assert client is not None
            # Verify the env var fallback was used
            mock_openai.assert_called_once()
            call_kwargs = mock_openai.call_args[1]
            assert call_kwargs["api_key"] == "env-fallback-key", (
                f"Expected env fallback key to be used when explicit_api_key is None, got: {call_kwargs['api_key']}"
            )


class TestAnthropicExplicitApiKey:
    """Test that explicit_api_key is correctly propagated to _try_anthropic().

    Parity with the OpenRouter fix in #18768: resolve_provider_client() passes
    explicit_api_key to _try_openrouter(), but the anthropic branch was not
    updated — _try_anthropic() always fell back to resolve_anthropic_token()
    even when an explicit key was supplied (e.g. from a fallback_model entry).
    """

    def test_try_anthropic_uses_explicit_api_key_over_env(self):
        """_try_anthropic(explicit_api_key) must use the supplied key, not the env fallback."""
        with patch("agent.anthropic_adapter.resolve_anthropic_token", return_value="env-fallback-key"), \
             patch("agent.anthropic_adapter.build_anthropic_client") as mock_build, \
             patch("agent.auxiliary_client._select_pool_entry", return_value=(False, None)):
            mock_build.return_value = MagicMock()
            from agent.auxiliary_client import _try_anthropic
            client, model = _try_anthropic("explicit-pool-key")
        assert client is not None
        assert mock_build.call_args.args[0] == "explicit-pool-key", (
            f"Expected explicit_api_key to be passed, got: {mock_build.call_args.args[0]}"
        )
        assert mock_build.call_args.args[0] != "env-fallback-key"

    def test_try_anthropic_without_explicit_key_falls_back_to_resolve(self):
        """Without explicit_api_key, _try_anthropic falls back to resolve_anthropic_token."""
        with patch("agent.anthropic_adapter.resolve_anthropic_token", return_value="env-fallback-key"), \
             patch("agent.anthropic_adapter.build_anthropic_client") as mock_build, \
             patch("agent.auxiliary_client._select_pool_entry", return_value=(False, None)):
            mock_build.return_value = MagicMock()
            from agent.auxiliary_client import _try_anthropic
            client, model = _try_anthropic()
        assert client is not None
        assert mock_build.call_args.args[0] == "env-fallback-key"

    def test_resolve_provider_client_passes_explicit_api_key_to_anthropic(self):
        """resolve_provider_client(provider='anthropic', explicit_api_key=...) must propagate the key."""
        with patch("agent.anthropic_adapter.resolve_anthropic_token", return_value="env-key"), \
             patch("agent.anthropic_adapter.build_anthropic_client") as mock_build, \
             patch("agent.auxiliary_client._select_pool_entry", return_value=(False, None)):
            mock_build.return_value = MagicMock()
            client, model = resolve_provider_client(
                provider="anthropic",
                explicit_api_key="explicit-fallback-key",
            )
        assert client is not None
        assert mock_build.call_args.args[0] == "explicit-fallback-key", (
            "resolve_provider_client must forward explicit_api_key to _try_anthropic()"
        )


# ── Auxiliary unhealthy-provider TTL cache (issue #23570) ────────────────


class TestAuxUnhealthyCache:
    """Recently-402'd providers are skipped on subsequent aux calls.

    Without this, every compression / title-gen / session-search call on a
    long session retries a depleted OpenRouter (~1 RTT to 402) before
    falling back to the next provider. The TTL cache hides the unhealthy
    provider for ``_AUX_UNHEALTHY_TTL_SECONDS`` so the chain skips it.
    """

    def setup_method(self):
        from agent.auxiliary_client import _reset_aux_unhealthy_cache
        _reset_aux_unhealthy_cache()

    def teardown_method(self):
        from agent.auxiliary_client import _reset_aux_unhealthy_cache
        _reset_aux_unhealthy_cache()

    def test_mark_then_skip(self):
        from agent.auxiliary_client import (
            _mark_provider_unhealthy,
            _is_provider_unhealthy,
        )
        assert _is_provider_unhealthy("openrouter") is False
        _mark_provider_unhealthy("openrouter")
        assert _is_provider_unhealthy("openrouter") is True

    def test_ttl_expiry_evicts(self):
        from agent.auxiliary_client import (
            _mark_provider_unhealthy,
            _is_provider_unhealthy,
            _aux_unhealthy_until,
        )
        _mark_provider_unhealthy("openrouter", ttl=0.01)
        assert _is_provider_unhealthy("openrouter") is True
        import time
        time.sleep(0.02)
        # Lazy eviction: first lookup after expiry returns False AND removes the entry.
        assert _is_provider_unhealthy("openrouter") is False
        assert "openrouter" not in _aux_unhealthy_until

    def test_alias_normalization(self):
        """'codex' should normalize to 'openai-codex' so the cache lookup
        matches the chain label."""
        from agent.auxiliary_client import (
            _mark_provider_unhealthy,
            _is_provider_unhealthy,
        )
        _mark_provider_unhealthy("codex")
        assert _is_provider_unhealthy("openai-codex") is True

    def test_resolve_auto_skips_unhealthy_step2(self):
        """_resolve_auto Step-2 chain skips unhealthy providers."""
        from agent.auxiliary_client import (
            _resolve_auto,
            _mark_provider_unhealthy,
        )
        nous_client = MagicMock()
        # Mark OpenRouter unhealthy → chain should skip it and pick nous.
        _mark_provider_unhealthy("openrouter")
        with patch("agent.auxiliary_client._read_main_provider", return_value=""), \
             patch("agent.auxiliary_client._read_main_model", return_value=""), \
             patch("agent.auxiliary_client._try_openrouter") as or_try, \
             patch("agent.auxiliary_client._try_nous", return_value=(nous_client, "nous-model")), \
             patch("agent.auxiliary_client._try_custom_endpoint", return_value=(None, None)), \
             patch("agent.auxiliary_client._resolve_api_key_provider", return_value=(None, None)):
            client, model = _resolve_auto()
        assert client is nous_client
        assert model == "nous-model"
        # The skipped provider's _try_* should NOT have been called at all.
        or_try.assert_not_called()

    def test_resolve_auto_skips_unhealthy_main_in_step1(self):
        """Step-1 also consults the unhealthy cache so a depleted main
        provider doesn't burn a 402 RTT every aux call. Falls through to
        Step-2 chain (which also respects the cache)."""
        from agent.auxiliary_client import (
            _resolve_auto,
            _mark_provider_unhealthy,
        )
        nous_client = MagicMock()
        _mark_provider_unhealthy("openrouter")
        with patch("agent.auxiliary_client._read_main_provider", return_value="openrouter"), \
             patch("agent.auxiliary_client._read_main_model", return_value="anthropic/claude-sonnet-4.6"), \
             patch("agent.auxiliary_client.resolve_provider_client") as step1, \
             patch("agent.auxiliary_client._try_openrouter") as or_try, \
             patch("agent.auxiliary_client._try_nous", return_value=(nous_client, "n-model")), \
             patch("agent.auxiliary_client._try_custom_endpoint", return_value=(None, None)), \
             patch("agent.auxiliary_client._resolve_api_key_provider", return_value=(None, None)):
            client, model = _resolve_auto()
        # Step-1 was bypassed — resolve_provider_client never invoked
        step1.assert_not_called()
        # Step-2 also skipped openrouter and landed on nous
        or_try.assert_not_called()
        assert client is nous_client

    def test_payment_fallback_skips_unhealthy(self):
        """_try_payment_fallback also consults the unhealthy cache so a 402
        on OpenRouter doesn't cause a second OR call within the same chain
        iteration if it gets re-entered."""
        from agent.auxiliary_client import (
            _try_payment_fallback,
            _mark_provider_unhealthy,
        )
        nous_client = MagicMock()
        # Mark BOTH the failed provider (openrouter) and a sibling (custom)
        # unhealthy. The chain should still find nous.
        _mark_provider_unhealthy("local/custom")
        with patch("agent.auxiliary_client._read_main_provider", return_value="openrouter"), \
             patch("agent.auxiliary_client._try_openrouter") as or_try, \
             patch("agent.auxiliary_client._try_nous", return_value=(nous_client, "n-model")), \
             patch("agent.auxiliary_client._try_custom_endpoint") as custom_try, \
             patch("agent.auxiliary_client._resolve_api_key_provider", return_value=(None, None)):
            client, model, label = _try_payment_fallback("openrouter", task="compression")
        assert client is nous_client
        assert label == "nous"
        # OR is skipped via skip_chain_labels (failed provider), custom via unhealthy cache.
        or_try.assert_not_called()
        custom_try.assert_not_called()

    def test_call_llm_marks_provider_unhealthy_on_402(self, monkeypatch):
        """A 402 from call_llm causes the provider to be marked unhealthy
        so the next call skips it instead of re-trying the same depleted
        endpoint."""
        from agent.auxiliary_client import (
            call_llm,
            _is_provider_unhealthy,
        )
        monkeypatch.setenv("OPENROUTER_API_KEY", "or-key")

        primary_client = MagicMock()
        # base_url tells _recoverable_pool_provider() that this is OpenRouter
        # (resolved_provider="auto" doesn't carry that information by itself).
        primary_client.base_url = "https://openrouter.ai/api/v1/"
        err = Exception("Payment Required: insufficient credits")
        err.status_code = 402
        primary_client.chat.completions.create.side_effect = err

        nous_client = MagicMock()
        nous_resp = MagicMock()
        nous_resp.choices = [MagicMock(message=MagicMock(content="ok"))]
        nous_client.chat.completions.create.return_value = nous_resp

        with patch("agent.auxiliary_client._get_cached_client",
                    return_value=(primary_client, "google/gemini-3-flash-preview")), \
             patch("agent.auxiliary_client._resolve_task_provider_model",
                    return_value=("auto", "google/gemini-3-flash-preview", None, None, None)), \
             patch("agent.auxiliary_client._try_payment_fallback",
                    return_value=(nous_client, "n-model", "nous")), \
             patch("agent.auxiliary_client._build_call_kwargs",
                    return_value={"model": "n-model", "messages": [{"role": "user", "content": "hi"}]}):
            assert _is_provider_unhealthy("openrouter") is False
            call_llm(
                task="compression",
                messages=[{"role": "user", "content": "hi"}],
            )
            # After the 402, OpenRouter is in the unhealthy cache.
            assert _is_provider_unhealthy("openrouter") is True
