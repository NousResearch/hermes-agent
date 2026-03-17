"""Tests for agent.auxiliary_client resolution chain, provider overrides, and model overrides."""

import json
import os
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from agent.auxiliary_client import (
    get_text_auxiliary_client,
    get_vision_auxiliary_client,
    get_available_vision_backends,
    resolve_provider_client,
    auxiliary_max_tokens_param,
    _read_codex_access_token,
    _get_auxiliary_provider,
    _resolve_forced_provider,
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
        assert model == "gemini-3-flash"

    def test_custom_endpoint_over_codex(self, monkeypatch, codex_auth_dir):
        monkeypatch.setenv("OPENAI_BASE_URL", "http://localhost:1234/v1")
        monkeypatch.setenv("OPENAI_API_KEY", "lm-studio-key")
        monkeypatch.setenv("OPENAI_MODEL", "my-local-model")
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

    def test_task_direct_endpoint_without_openai_key_does_not_fall_back(self, monkeypatch):
        monkeypatch.setenv("OPENROUTER_API_KEY", "or-key")
        monkeypatch.setenv("AUXILIARY_WEB_EXTRACT_BASE_URL", "http://localhost:2345/v1")
        monkeypatch.setenv("AUXILIARY_WEB_EXTRACT_MODEL", "task-model")
        with patch("agent.auxiliary_client.OpenAI") as mock_openai:
            client, model = get_text_auxiliary_client("web_extract")
        assert client is None
        assert model is None
        mock_openai.assert_not_called()

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

    def test_vision_returns_none_without_any_credentials(self):
        with (
            patch("agent.auxiliary_client._read_nous_auth", return_value=None),
            patch("agent.auxiliary_client._try_anthropic", return_value=(None, None)),
        ):
            client, model = get_vision_auxiliary_client()
        assert client is None
        assert model is None

    def test_vision_auto_includes_anthropic_when_configured(self, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-api03-key")
        with (
            patch("agent.auxiliary_client._read_nous_auth", return_value=None),
            patch("agent.anthropic_adapter.build_anthropic_client", return_value=MagicMock()),
            patch("agent.anthropic_adapter.resolve_anthropic_token", return_value="sk-ant-api03-key"),
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

    def test_vision_auto_uses_anthropic_when_no_higher_priority_backend(self, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-api03-key")
        with (
            patch("agent.auxiliary_client._read_nous_auth", return_value=None),
            patch("agent.anthropic_adapter.build_anthropic_client", return_value=MagicMock()),
            patch("agent.anthropic_adapter.resolve_anthropic_token", return_value="sk-ant-api03-key"),
        ):
            client, model = get_vision_auxiliary_client()

        assert client is not None
        assert client.__class__.__name__ == "AnthropicAuxiliaryClient"
        assert model == "claude-haiku-4-5-20251001"

    def test_selected_anthropic_provider_is_preferred_for_vision_auto(self, monkeypatch):
        monkeypatch.setenv("OPENROUTER_API_KEY", "or-key")
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-api03-key")

        def fake_load_config():
            return {"model": {"provider": "anthropic", "default": "claude-sonnet-4-6"}}

        with (
            patch("agent.auxiliary_client._read_nous_auth", return_value=None),
            patch("agent.anthropic_adapter.build_anthropic_client", return_value=MagicMock()),
            patch("agent.anthropic_adapter.resolve_anthropic_token", return_value="sk-ant-api03-key"),
            patch("agent.auxiliary_client.OpenAI") as mock_openai,
            patch("hermes_cli.config.load_config", fake_load_config),
        ):
            client, model = get_vision_auxiliary_client()

        assert client is not None
        assert client.__class__.__name__ == "AnthropicAuxiliaryClient"
        assert model == "claude-haiku-4-5-20251001"

    def test_vision_auto_includes_codex(self, codex_auth_dir):
        """Codex supports vision (gpt-5.3-codex), so auto mode should use it."""
        with patch("agent.auxiliary_client._read_nous_auth", return_value=None), \
             patch("agent.auxiliary_client.OpenAI"):
            client, model = get_vision_auxiliary_client()
        from agent.auxiliary_client import CodexAuxiliaryClient
        assert isinstance(client, CodexAuxiliaryClient)
        assert model == "gpt-5.2-codex"

    def test_vision_auto_falls_back_to_custom_endpoint(self, monkeypatch):
        """Custom endpoint is used as fallback in vision auto mode.

        Many local models (Qwen-VL, LLaVA, etc.) support vision.
        When no OpenRouter/Nous/Codex is available, try the custom endpoint.
        """
        monkeypatch.setenv("OPENAI_BASE_URL", "http://localhost:1234/v1")
        monkeypatch.setenv("OPENAI_API_KEY", "local-key")
        with patch("agent.auxiliary_client._read_nous_auth", return_value=None), \
             patch("agent.auxiliary_client.OpenAI") as mock_openai:
            client, model = get_vision_auxiliary_client()
        assert client is not None  # Custom endpoint picked up as fallback

    def test_vision_direct_endpoint_override(self, monkeypatch):
        monkeypatch.setenv("OPENROUTER_API_KEY", "or-key")
        monkeypatch.setenv("AUXILIARY_VISION_BASE_URL", "http://localhost:4567/v1")
        monkeypatch.setenv("AUXILIARY_VISION_API_KEY", "vision-key")
        monkeypatch.setenv("AUXILIARY_VISION_MODEL", "vision-model")
        with patch("agent.auxiliary_client.OpenAI") as mock_openai:
            client, model = get_vision_auxiliary_client()
        assert model == "vision-model"
        assert mock_openai.call_args.kwargs["base_url"] == "http://localhost:4567/v1"
        assert mock_openai.call_args.kwargs["api_key"] == "vision-key"

    def test_vision_direct_endpoint_requires_openai_api_key(self, monkeypatch):
        monkeypatch.setenv("OPENROUTER_API_KEY", "or-key")
        monkeypatch.setenv("AUXILIARY_VISION_BASE_URL", "http://localhost:4567/v1")
        monkeypatch.setenv("AUXILIARY_VISION_MODEL", "vision-model")
        with patch("agent.auxiliary_client.OpenAI") as mock_openai:
            client, model = get_vision_auxiliary_client()
        assert client is None
        assert model is None
        mock_openai.assert_not_called()

    def test_vision_uses_openrouter_when_available(self, monkeypatch):
        monkeypatch.setenv("OPENROUTER_API_KEY", "or-key")
        with patch("agent.auxiliary_client.OpenAI") as mock_openai:
            client, model = get_vision_auxiliary_client()
        assert model == "google/gemini-3-flash-preview"
        assert client is not None

    def test_vision_uses_nous_when_available(self, monkeypatch):
        with patch("agent.auxiliary_client._read_nous_auth") as mock_nous, \
             patch("agent.auxiliary_client.OpenAI"):
            mock_nous.return_value = {"access_token": "nous-tok"}
            client, model = get_vision_auxiliary_client()
        assert model == "gemini-3-flash"
        assert client is not None

    def test_vision_forced_main_uses_custom_endpoint(self, monkeypatch):
        """When explicitly forced to 'main', vision CAN use custom endpoint."""
        monkeypatch.setenv("AUXILIARY_VISION_PROVIDER", "main")
        monkeypatch.setenv("OPENAI_BASE_URL", "http://localhost:1234/v1")
        monkeypatch.setenv("OPENAI_API_KEY", "local-key")
        monkeypatch.setenv("OPENAI_MODEL", "my-local-model")
        with patch("agent.auxiliary_client._read_nous_auth", return_value=None), \
             patch("agent.auxiliary_client.OpenAI") as mock_openai:
            client, model = get_vision_auxiliary_client()
        assert client is not None
        assert model == "my-local-model"

    def test_vision_forced_main_returns_none_without_creds(self, monkeypatch):
        """Forced main with no credentials still returns None."""
        monkeypatch.setenv("AUXILIARY_VISION_PROVIDER", "main")
        monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        with patch("agent.auxiliary_client._read_nous_auth", return_value=None), \
             patch("agent.auxiliary_client._read_codex_access_token", return_value=None), \
             patch("agent.auxiliary_client._resolve_api_key_provider", return_value=(None, None)):
            client, model = get_vision_auxiliary_client()
        assert client is None
        assert model is None

    def test_vision_forced_codex(self, monkeypatch, codex_auth_dir):
        """When forced to 'codex', vision uses Codex OAuth."""
        monkeypatch.setenv("AUXILIARY_VISION_PROVIDER", "codex")
        with patch("agent.auxiliary_client._read_nous_auth", return_value=None), \
             patch("agent.auxiliary_client.OpenAI"):
            client, model = get_vision_auxiliary_client()
        from agent.auxiliary_client import CodexAuxiliaryClient
        assert isinstance(client, CodexAuxiliaryClient)
        assert model == "gpt-5.2-codex"


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


class TestResolveForcedProvider:
    """Tests for _resolve_forced_provider with explicit provider selection."""

    def test_forced_openrouter(self, monkeypatch):
        monkeypatch.setenv("OPENROUTER_API_KEY", "or-key")
        with patch("agent.auxiliary_client.OpenAI") as mock_openai:
            client, model = _resolve_forced_provider("openrouter")
        assert model == "google/gemini-3-flash-preview"
        assert client is not None

    def test_forced_openrouter_no_key(self, monkeypatch):
        with patch("agent.auxiliary_client._read_nous_auth", return_value=None):
            client, model = _resolve_forced_provider("openrouter")
        assert client is None
        assert model is None

    def test_forced_nous(self, monkeypatch):
        with patch("agent.auxiliary_client._read_nous_auth") as mock_nous, \
             patch("agent.auxiliary_client.OpenAI"):
            mock_nous.return_value = {"access_token": "nous-tok"}
            client, model = _resolve_forced_provider("nous")
        assert model == "gemini-3-flash"
        assert client is not None

    def test_forced_nous_not_configured(self, monkeypatch):
        with patch("agent.auxiliary_client._read_nous_auth", return_value=None):
            client, model = _resolve_forced_provider("nous")
        assert client is None
        assert model is None

    def test_forced_main_uses_custom(self, monkeypatch):
        monkeypatch.setenv("OPENAI_BASE_URL", "http://local:8080/v1")
        monkeypatch.setenv("OPENAI_API_KEY", "local-key")
        monkeypatch.setenv("OPENAI_MODEL", "my-local-model")
        with patch("agent.auxiliary_client._read_nous_auth", return_value=None), \
             patch("agent.auxiliary_client.OpenAI") as mock_openai:
            client, model = _resolve_forced_provider("main")
        assert model == "my-local-model"

    def test_forced_main_uses_config_saved_custom_endpoint(self, monkeypatch):
        config = {
            "model": {
                "provider": "custom",
                "base_url": "http://local:8080/v1",
                "default": "my-local-model",
            }
        }
        monkeypatch.setenv("OPENAI_API_KEY", "local-key")
        monkeypatch.setattr("hermes_cli.config.load_config", lambda: config)
        monkeypatch.setattr("hermes_cli.runtime_provider.load_config", lambda: config)
        with patch("agent.auxiliary_client._read_nous_auth", return_value=None), \
             patch("agent.auxiliary_client._read_codex_access_token", return_value=None), \
             patch("agent.auxiliary_client._resolve_api_key_provider", return_value=(None, None)), \
             patch("agent.auxiliary_client.OpenAI") as mock_openai:
            client, model = _resolve_forced_provider("main")
        assert client is not None
        assert model == "my-local-model"
        call_kwargs = mock_openai.call_args
        assert call_kwargs.kwargs["base_url"] == "http://local:8080/v1"

    def test_forced_main_skips_openrouter_nous(self, monkeypatch):
        """Even if OpenRouter key is set, 'main' skips it."""
        monkeypatch.setenv("OPENROUTER_API_KEY", "or-key")
        monkeypatch.setenv("OPENAI_BASE_URL", "http://local:8080/v1")
        monkeypatch.setenv("OPENAI_API_KEY", "local-key")
        monkeypatch.setenv("OPENAI_MODEL", "my-local-model")
        with patch("agent.auxiliary_client._read_nous_auth", return_value=None), \
             patch("agent.auxiliary_client.OpenAI") as mock_openai:
            client, model = _resolve_forced_provider("main")
        # Should use custom endpoint, not OpenRouter
        assert model == "my-local-model"

    def test_forced_main_falls_to_codex(self, codex_auth_dir, monkeypatch):
        with patch("agent.auxiliary_client._read_nous_auth", return_value=None), \
             patch("agent.auxiliary_client.OpenAI"):
            client, model = _resolve_forced_provider("main")
        from agent.auxiliary_client import CodexAuxiliaryClient
        assert isinstance(client, CodexAuxiliaryClient)
        assert model == "gpt-5.2-codex"

    def test_forced_codex(self, codex_auth_dir, monkeypatch):
        with patch("agent.auxiliary_client._read_nous_auth", return_value=None), \
             patch("agent.auxiliary_client.OpenAI"):
            client, model = _resolve_forced_provider("codex")
        from agent.auxiliary_client import CodexAuxiliaryClient
        assert isinstance(client, CodexAuxiliaryClient)
        assert model == "gpt-5.2-codex"

    def test_forced_codex_no_token(self, monkeypatch):
        with patch("agent.auxiliary_client._read_codex_access_token", return_value=None):
            client, model = _resolve_forced_provider("codex")
        assert client is None
        assert model is None

    def test_forced_unknown_returns_none(self, monkeypatch):
        with patch("agent.auxiliary_client._read_nous_auth", return_value=None), \
             patch("agent.auxiliary_client._read_codex_access_token", return_value=None):
            client, model = _resolve_forced_provider("invalid-provider")
        assert client is None
        assert model is None


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
