"""Tests for agent.auxiliary_client resolution chain, provider overrides, and model overrides."""

import json
import logging
import os
from pathlib import Path
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
    _read_codex_access_token,
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
            from agent.auxiliary_client import _try_codex

            client, model = _try_codex()

        from agent.auxiliary_client import CodexAuxiliaryClient

        assert isinstance(client, CodexAuxiliaryClient)
        assert model == "gpt-5.2-codex"

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


class TestKimiForCodingTemperature:
    """kimi-for-coding now requires temperature=0.6 exactly."""

    def test_build_call_kwargs_forces_fixed_temperature(self):
        from agent.auxiliary_client import _build_call_kwargs

        kwargs = _build_call_kwargs(
            provider="kimi-coding",
            model="kimi-for-coding",
            messages=[{"role": "user", "content": "hello"}],
            temperature=0.3,
        )

        assert kwargs["temperature"] == 0.6

    def test_build_call_kwargs_injects_temperature_when_missing(self):
        from agent.auxiliary_client import _build_call_kwargs

        kwargs = _build_call_kwargs(
            provider="kimi-coding",
            model="kimi-for-coding",
            messages=[{"role": "user", "content": "hello"}],
            temperature=None,
        )

        assert kwargs["temperature"] == 0.6

    def test_auto_routed_kimi_for_coding_sync_call_uses_fixed_temperature(self):
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
        assert kwargs["temperature"] == 0.6

    @pytest.mark.asyncio
    async def test_auto_routed_kimi_for_coding_async_call_uses_fixed_temperature(self):
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
        assert kwargs["temperature"] == 0.6

    def test_non_kimi_model_still_preserves_temperature(self):
        from agent.auxiliary_client import _build_call_kwargs

        kwargs = _build_call_kwargs(
            provider="kimi-coding",
            model="kimi-k2.5",
            messages=[{"role": "user", "content": "hello"}],
            temperature=0.3,
        )

        assert kwargs["temperature"] == 0.3


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
