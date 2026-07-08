"""Test for _normalize_main_model_assignment custom provider preservation (#50944)."""
import pytest
from unittest.mock import patch

from hermes_cli.web_server import _normalize_main_model_assignment


class TestNormalizeMainModelAssignmentCustomProvider:
    """Test that user-configured custom providers are preserved during normalization.

    Regression test for #50944: When a user configures a custom provider in the
    providers: block (e.g., litellm-proxy), and then selects it + a model with a slash
    (e.g., nvidia/nim/deepseek-v4-pro) via the Desktop model picker, the provider should
    NOT be fallback-ed to openrouter just because it's not in _KNOWN_PROVIDER_NAMES.
    """

    @pytest.fixture
    def mock_config_with_custom_provider(self):
        """Mock config with a custom provider in providers: block."""
        return {
            "model": {"provider": "litellm-proxy", "default": "nvidia/nim/deepseek-v4-pro"},
            "providers": {
                "litellm-proxy": {"base_url": "http://localhost:4000"},
                "ollama": {"base_url": "http://localhost:11434"},
            },
        }

    @pytest.fixture
    def mock_config_with_legacy_custom_provider(self):
        """Mock config with legacy custom_providers format."""
        return {
            "model": {"provider": "my-proxy", "default": "custom-model"},
            "custom_providers": [
                {"id": "my-proxy", "base_url": "http://localhost:8080"},
                {"name": "another-proxy", "base_url": "http://localhost:9090"},
            ],
        }

    @pytest.fixture
    def mock_config_without_custom_provider(self):
        """Mock config without the provider in question."""
        return {
            "model": {"provider": "openrouter", "default": "anthropic/claude-3-5-sonnet"},
            "providers": {
                "openrouter": {"api_key_env": "OPENROUTER_API_KEY"},
            },
        }

    def test_custom_provider_with_slashed_model_preserved(self, mock_config_with_custom_provider):
        """Custom provider + model with slash should be preserved, not fallback to openrouter."""
        with patch("hermes_cli.web_server.load_config", return_value=mock_config_with_custom_provider):
            provider, model = _normalize_main_model_assignment(
                "litellm-proxy", "nvidia/nim/deepseek-v4-pro"
            )

            # Should preserve the custom provider, NOT fallback to openrouter
            assert provider == "litellm-proxy", f"Expected 'litellm-proxy', got '{provider}'"
            assert model == "nvidia/nim/deepseek-v4-pro", f"Expected model unchanged, got '{model}'"

    def test_legacy_custom_provider_preserved(self, mock_config_with_legacy_custom_provider):
        """Legacy custom_providers format should also be recognized."""
        with patch("hermes_cli.web_server.load_config", return_value=mock_config_with_legacy_custom_provider):
            provider, model = _normalize_main_model_assignment(
                "my-proxy", "custom-model"
            )

            # Should preserve the legacy custom provider
            assert provider == "my-proxy", f"Expected 'my-proxy', got '{provider}'"
            assert model == "custom-model", f"Expected model unchanged, got '{model}'"

    def test_legacy_custom_provider_with_name_field(self, mock_config_with_legacy_custom_provider):
        """Legacy custom_providers with 'name' field should be recognized."""
        with patch("hermes_cli.web_server.load_config", return_value=mock_config_with_legacy_custom_provider):
            provider, model = _normalize_main_model_assignment(
                "another-proxy", "vendor/model"
            )

            # Should preserve the custom provider with 'name' field
            assert provider == "another-proxy", f"Expected 'another-proxy', got '{provider}'"
            assert model == "vendor/model", f"Expected model unchanged, got '{model}'"

    def test_unknown_provider_fallback_to_openrouter(self, mock_config_without_custom_provider):
        """Unknown provider NOT in config should still fallback to openrouter (analytics fallback).

        Note: 'anthropic' is in _KNOWN_PROVIDER_NAMES, so we use 'poolside' which is NOT.
        """
        with patch("hermes_cli.web_server.load_config", return_value=mock_config_without_custom_provider):
            # Simulate analytics fallback: unknown vendor name as provider
            provider, model = _normalize_main_model_assignment(
                "poolside", "poolside/model"
            )

            # Should fallback to openrouter (current provider is an aggregator)
            assert provider == "openrouter", f"Expected 'openrouter', got '{provider}'"
            # Model should be preserved
            assert model == "poolside/model", f"Expected model unchanged, got '{model}'"

    def test_known_provider_unaffected(self, mock_config_with_custom_provider):
        """Known providers should not be affected by this fix."""
        with patch("hermes_cli.web_server.load_config", return_value=mock_config_with_custom_provider):
            provider, model = _normalize_main_model_assignment(
                "openrouter", "anthropic/claude-3-5-sonnet"
            )

            # openrouter is in _KNOWN_PROVIDER_NAMES, should work normally
            assert provider == "openrouter", f"Expected 'openrouter', got '{provider}'"
            assert model == "anthropic/claude-3-5-sonnet", f"Expected model unchanged, got '{model}'"

    def test_custom_provider_without_slash(self, mock_config_with_custom_provider):
        """Custom provider with model without slash should also be preserved."""
        with patch("hermes_cli.web_server.load_config", return_value=mock_config_with_custom_provider):
            provider, model = _normalize_main_model_assignment(
                "litellm-proxy", "deepseek-v4-pro"
            )

            # Should preserve the custom provider
            assert provider == "litellm-proxy", f"Expected 'litellm-proxy', got '{provider}'"
            assert model == "deepseek-v4-pro", f"Expected model unchanged, got '{model}'"

    def test_empty_config_continues_with_fallback(self):
        """When config load fails, should continue with existing fallback logic.

        Note: 'anthropic' is in _KNOWN_PROVIDER_NAMES, so we use 'unknown-provider' which is NOT.
        """
        with patch("hermes_cli.web_server.load_config", side_effect=Exception("Config read failed")):
            # Even with empty config, the original logic should still work
            provider, model = _normalize_main_model_assignment(
                "unknown-provider", "unknown-provider/model"
            )

            # Should fallback to openrouter (current provider is empty)
            assert provider == "openrouter", f"Expected 'openrouter', got '{provider}'"