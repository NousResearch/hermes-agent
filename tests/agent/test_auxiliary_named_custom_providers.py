"""Tests for named custom provider and 'main' alias resolution in auxiliary_client."""

import os
from unittest.mock import patch, MagicMock

import pytest


@pytest.fixture(autouse=True)
def _isolate(tmp_path, monkeypatch):
    """Redirect HERMES_HOME and clear module caches."""
    hermes_home = tmp_path / ".hermes"
    hermes_home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    # Write a minimal config so load_config doesn't fail
    (hermes_home / "config.yaml").write_text("model:\n  default: test-model\n")


def _write_config(tmp_path, config_dict):
    """Write a config.yaml to the test HERMES_HOME."""
    import yaml
    config_path = tmp_path / ".hermes" / "config.yaml"
    config_path.write_text(yaml.dump(config_dict))


class TestNormalizeVisionProvider:
    """_normalize_vision_provider should resolve 'main' to actual main provider."""

    def test_main_resolves_to_named_custom(self, tmp_path):
        _write_config(tmp_path, {
            "model": {"default": "my-model", "provider": "custom:beans"},
            "custom_providers": [{"name": "beans", "base_url": "http://localhost/v1"}],
        })
        from agent.auxiliary_client import _normalize_vision_provider
        assert _normalize_vision_provider("main") == "custom:beans"

    def test_main_resolves_to_openrouter(self, tmp_path):
        _write_config(tmp_path, {
            "model": {"default": "anthropic/claude-sonnet-4", "provider": "openrouter"},
        })
        from agent.auxiliary_client import _normalize_vision_provider
        assert _normalize_vision_provider("main") == "openrouter"

    def test_main_resolves_to_deepseek(self, tmp_path):
        _write_config(tmp_path, {
            "model": {"default": "deepseek-chat", "provider": "deepseek"},
        })
        from agent.auxiliary_client import _normalize_vision_provider
        assert _normalize_vision_provider("main") == "deepseek"

    def test_main_falls_back_to_custom_when_no_provider(self, tmp_path):
        _write_config(tmp_path, {"model": {"default": "gpt-4o"}})
        from agent.auxiliary_client import _normalize_vision_provider
        assert _normalize_vision_provider("main") == "custom"

    def test_bare_provider_name_unchanged(self):
        from agent.auxiliary_client import _normalize_vision_provider
        assert _normalize_vision_provider("beans") == "beans"
        assert _normalize_vision_provider("deepseek") == "deepseek"

    def test_custom_colon_named_provider_preserved(self):
        from agent.auxiliary_client import _normalize_vision_provider
        assert _normalize_vision_provider("custom:beans") == "beans"

    def test_codex_alias_still_works(self):
        from agent.auxiliary_client import _normalize_vision_provider
        assert _normalize_vision_provider("codex") == "openai-codex"

    def test_auto_unchanged(self):
        from agent.auxiliary_client import _normalize_vision_provider
        assert _normalize_vision_provider("auto") == "auto"
        assert _normalize_vision_provider(None) == "auto"


class TestResolveProviderClientMainAlias:
    """resolve_provider_client('main', ...) should resolve to actual main provider."""

    def test_main_resolves_to_named_custom_provider(self, tmp_path):
        _write_config(tmp_path, {
            "model": {"default": "my-model", "provider": "beans"},
            "custom_providers": [
                {"name": "beans", "base_url": "http://beans.local/v1", "api_key": "k"},
            ],
        })
        from agent.auxiliary_client import resolve_provider_client
        client, model = resolve_provider_client("main", "override-model")
        assert client is not None
        assert model == "override-model"
        assert "beans.local" in str(client.base_url)

    def test_main_with_custom_colon_prefix(self, tmp_path):
        _write_config(tmp_path, {
            "model": {"default": "my-model", "provider": "custom:beans"},
            "custom_providers": [
                {"name": "beans", "base_url": "http://beans.local/v1", "api_key": "k"},
            ],
        })
        from agent.auxiliary_client import resolve_provider_client
        client, model = resolve_provider_client("main", "test")
        assert client is not None
        assert "beans.local" in str(client.base_url)


class TestResolveProviderClientNamedCustom:
    """resolve_provider_client should resolve named custom providers directly."""

    def test_named_custom_provider(self, tmp_path):
        _write_config(tmp_path, {
            "model": {"default": "test-model"},
            "custom_providers": [
                {"name": "beans", "base_url": "http://beans.local/v1", "api_key": "k"},
            ],
        })
        from agent.auxiliary_client import resolve_provider_client
        client, model = resolve_provider_client("beans", "my-model")
        assert client is not None
        assert model == "my-model"
        assert "beans.local" in str(client.base_url)

    def test_named_custom_provider_default_model(self, tmp_path):
        _write_config(tmp_path, {
            "model": {"default": "main-model"},
            "custom_providers": [
                {"name": "beans", "base_url": "http://beans.local/v1", "api_key": "k"},
            ],
        })
        from agent.auxiliary_client import resolve_provider_client
        client, model = resolve_provider_client("beans")
        assert client is not None
        # Should use _read_main_model() fallback
        assert model == "main-model"

    def test_named_custom_no_api_key_uses_fallback(self, tmp_path):
        _write_config(tmp_path, {
            "model": {"default": "test"},
            "custom_providers": [
                {"name": "local", "base_url": "http://localhost:8080/v1"},
            ],
        })
        from agent.auxiliary_client import resolve_provider_client
        client, model = resolve_provider_client("local", "test")
        assert client is not None
        # no-key-required should be used

    def test_nonexistent_named_custom_falls_through(self, tmp_path):
        _write_config(tmp_path, {
            "model": {"default": "test"},
            "custom_providers": [
                {"name": "beans", "base_url": "http://beans.local/v1"},
            ],
        })
        from agent.auxiliary_client import resolve_provider_client
        # "coffee" doesn't exist in custom_providers
        client, model = resolve_provider_client("coffee", "test")
        assert client is None


class TestResolveProviderClientModelNormalization:
    """Direct-provider auxiliary routing should normalize models like main runtime."""

    def test_matching_native_prefix_is_stripped_for_main_provider(self, tmp_path):
        _write_config(tmp_path, {
            "model": {"default": "zai/glm-5.1", "provider": "zai"},
        })
        with (
            patch("hermes_cli.auth.resolve_api_key_provider_credentials", return_value={
                "api_key": "glm-key",
                "base_url": "https://api.z.ai/api/paas/v4",
            }),
            patch("agent.auxiliary_client.OpenAI") as mock_openai,
        ):
            mock_openai.return_value = MagicMock()
            from agent.auxiliary_client import resolve_provider_client

            client, model = resolve_provider_client("main", "zai/glm-5.1")

        assert client is not None
        assert model == "glm-5.1"

    def test_non_matching_prefix_is_preserved_for_direct_provider(self, tmp_path):
        _write_config(tmp_path, {
            "model": {"default": "zai/glm-5.1", "provider": "zai"},
        })
        with (
            patch("hermes_cli.auth.resolve_api_key_provider_credentials", return_value={
                "api_key": "glm-key",
                "base_url": "https://api.z.ai/api/paas/v4",
            }),
            patch("agent.auxiliary_client.OpenAI") as mock_openai,
        ):
            mock_openai.return_value = MagicMock()
            from agent.auxiliary_client import resolve_provider_client

            client, model = resolve_provider_client("zai", "google/gemini-2.5-pro")

        assert client is not None
        assert model == "google/gemini-2.5-pro"

    def test_aggregator_vendor_slug_is_preserved(self, monkeypatch):
        monkeypatch.setenv("OPENROUTER_API_KEY", "or-key")
        with patch("agent.auxiliary_client.OpenAI") as mock_openai:
            mock_openai.return_value = MagicMock()
            from agent.auxiliary_client import resolve_provider_client

            client, model = resolve_provider_client(
                "openrouter", "anthropic/claude-sonnet-4.6"
            )

        assert client is not None
        assert model == "anthropic/claude-sonnet-4.6"


class TestResolveVisionProviderClientModelNormalization:
    """Vision auto-routing should reuse the same provider-specific normalization."""

    def test_vision_auto_strips_matching_main_provider_prefix(self, tmp_path):
        _write_config(tmp_path, {
            "model": {"default": "zai/glm-5.1", "provider": "zai"},
        })
        with (
            patch("agent.auxiliary_client._read_nous_auth", return_value=None),
            patch("hermes_cli.auth.resolve_api_key_provider_credentials", return_value={
                "api_key": "glm-key",
                "base_url": "https://api.z.ai/api/paas/v4",
            }),
            patch("agent.auxiliary_client.OpenAI") as mock_openai,
        ):
            mock_openai.return_value = MagicMock()
            from agent.auxiliary_client import resolve_vision_provider_client

            provider, client, model = resolve_vision_provider_client()

        assert provider == "zai"
        assert client is not None
        assert model == "glm-5v-turbo"  # zai has dedicated vision model in _PROVIDER_VISION_MODELS


class TestVisionPathApiMode:
    """Vision path should propagate api_mode to _get_cached_client."""

    def test_explicit_provider_passes_api_mode(self, tmp_path):
        _write_config(tmp_path, {
            "model": {"default": "test-model"},
            "auxiliary": {"vision": {"api_mode": "chat_completions"}},
        })
        with patch("agent.auxiliary_client._get_cached_client") as mock_gcc:
            mock_gcc.return_value = (MagicMock(), "test-model")
            from agent.auxiliary_client import resolve_vision_provider_client

            provider, client, model = resolve_vision_provider_client(provider="deepseek")

        mock_gcc.assert_called_once()
        _, kwargs = mock_gcc.call_args
        assert kwargs.get("api_mode") == "chat_completions"
