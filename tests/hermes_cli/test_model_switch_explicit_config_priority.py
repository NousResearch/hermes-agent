"""Tests for Issue #39753: Config priority - OpenRouter catalog overrides explicit custom provider.

These tests ensure that when a user explicitly configures model.provider and model.base_url,
their explicit config is respected and not silently overridden by OpenRouter catalog detection.

Regression test for: https://github.com/NousResearch/hermes-agent/issues/39753
"""

import pytest
import logging
from hermes_cli.model_switch import switch_model


class TestExplicitConfigPriority:
    """Tests that explicit provider/base_url config is respected over OpenRouter catalog."""

    def test_explicit_provider_config_skips_openrouter_detection(self, monkeypatch):
        """When user has explicit provider + base_url, skip OpenRouter catalog detection.
        
        Regression test for Issue #39753:
        When a user configures:
          model:
            provider: minimax-cn
            base_url: https://api.minimaxi.com/v1
        
        Switching to a model that exists in OpenRouter catalog should NOT
        override their explicit config and route to OpenRouter.
        """
        # Mock detect_provider_for_model to return OpenRouter match
        # This simulates the model existing in OpenRouter catalog
        monkeypatch.setattr(
            "hermes_cli.models.detect_provider_for_model",
            lambda model, provider: ("openrouter", f"minimax/{model}") if model == "MiniMax-M2.5" else None
        )
        
        # Mock other dependencies
        monkeypatch.setattr("hermes_cli.models.fetch_openrouter_models", lambda: [])
        monkeypatch.setattr("agent.models_dev.fetch_models_dev", lambda: {})
        monkeypatch.setattr("hermes_cli.providers.HERMES_OVERLAYS", {})
        
        # User has explicit config
        result = switch_model(
            raw_input="MiniMax-M2.5",
            current_provider="minimax-cn",
            current_model="MiniMax-M2.7",
            current_base_url="https://api.minimaxi.com/v1",
            current_api_key="test-key",
            is_global=False,
            explicit_provider="",
            user_providers={},
            custom_providers=[],
        )
        
        # Should succeed
        assert result.success is True
        # Should keep the user's explicit provider
        assert result.target_provider == "minimax-cn"
        # Should keep the requested model name (not remapped to OpenRouter slug)
        assert result.new_model == "MiniMax-M2.5"

    def test_no_explicit_config_allows_openrouter_detection(self, monkeypatch):
        """When user has no explicit config, OpenRouter detection should still work.
        
        This ensures we don't break the default auto-detection behavior.
        """
        # Mock detect_provider_for_model to return OpenRouter match
        monkeypatch.setattr(
            "hermes_cli.models.detect_provider_for_model",
            lambda model, provider: ("openrouter", f"deepseek/{model}") if model == "deepseek-chat" else None
        )
        
        # Mock other dependencies
        monkeypatch.setattr("hermes_cli.models.fetch_openrouter_models", lambda: [])
        monkeypatch.setattr("agent.models_dev.fetch_models_dev", lambda: {})
        monkeypatch.setattr("hermes_cli.providers.HERMES_OVERLAYS", {})
        monkeypatch.setattr(
            "hermes_cli.runtime_provider.resolve_runtime_provider",
            lambda **kwargs: {
                "provider": kwargs.get("requested", "openrouter"),
                "api_key": "test-key",
                "base_url": "https://openrouter.ai/api/v1",
                "api_mode": "chat_completions",
            }
        )
        
        # User has no explicit base_url (auto/openrouter)
        result = switch_model(
            raw_input="deepseek-chat",
            current_provider="openrouter",
            current_model="gpt-4",
            current_base_url="",
            current_api_key="",
            is_global=False,
            explicit_provider="",
            user_providers={},
            custom_providers=[],
        )
        
        # OpenRouter detection should work when no explicit config
        assert result.target_provider == "openrouter"

    def test_explicit_openrouter_config_allows_detection(self, monkeypatch):
        """When user explicitly sets provider=openrouter, detection should still work.
        
        This ensures users who explicitly want OpenRouter still get catalog detection.
        """
        # Mock detect_provider_for_model to return OpenRouter match
        monkeypatch.setattr(
            "hermes_cli.models.detect_provider_for_model",
            lambda model, provider: ("openrouter", f"anthropic/{model}") if model == "claude-opus" else None
        )
        
        # Mock other dependencies
        monkeypatch.setattr("hermes_cli.models.fetch_openrouter_models", lambda: [])
        monkeypatch.setattr("agent.models_dev.fetch_models_dev", lambda: {})
        monkeypatch.setattr("hermes_cli.providers.HERMES_OVERLAYS", {})
        monkeypatch.setattr(
            "hermes_cli.runtime_provider.resolve_runtime_provider",
            lambda **kwargs: {
                "provider": "openrouter",
                "api_key": "test-key",
                "base_url": "https://openrouter.ai/api/v1",
                "api_mode": "chat_completions",
            }
        )
        
        # User explicitly sets provider=openrouter (no base_url override)
        result = switch_model(
            raw_input="claude-opus",
            current_provider="openrouter",
            current_model="gpt-4",
            current_base_url="",
            current_api_key="test-key",
            is_global=False,
            explicit_provider="",
            user_providers={},
            custom_providers=[],
        )
        
        # Should allow OpenRouter detection
        assert result.target_provider == "openrouter"

    def test_custom_local_provider_not_affected(self, monkeypatch):
        """Custom/local providers should continue to work as before."""
        monkeypatch.setattr("hermes_cli.models.fetch_openrouter_models", lambda: [])
        monkeypatch.setattr("agent.models_dev.fetch_models_dev", lambda: {})
        monkeypatch.setattr("hermes_cli.providers.HERMES_OVERLAYS", {})
        monkeypatch.setattr(
            "hermes_cli.runtime_provider.resolve_runtime_provider",
            lambda **kwargs: {
                "provider": "custom",
                "api_key": "",
                "base_url": "http://localhost:11434/v1",
                "api_mode": "chat_completions",
            }
        )
        
        # Local provider should work
        result = switch_model(
            raw_input="llama3",
            current_provider="custom",
            current_model="",
            current_base_url="http://localhost:11434/v1",
            current_api_key="",
            is_global=False,
            explicit_provider="",
            user_providers={},
            custom_providers=[],
        )
        
        # Custom/local provider should keep its provider (not switched to OpenRouter)
        assert result.target_provider == "custom"

    def test_provider_switching_logs_info(self, monkeypatch, caplog):
        """When provider switches due to OpenRouter match, it should be logged (Option C)."""
        # Set log level to capture INFO messages
        caplog.set_level(logging.INFO, logger="hermes_cli.model_switch")
        
        # Mock detect_provider_for_model to return OpenRouter match
        monkeypatch.setattr(
            "hermes_cli.models.detect_provider_for_model",
            lambda model, provider: ("openrouter", f"vendor/{model}") if model == "some-model" else None
        )
        
        # Mock other dependencies
        monkeypatch.setattr("hermes_cli.models.fetch_openrouter_models", lambda: [])
        monkeypatch.setattr("agent.models_dev.fetch_models_dev", lambda: {})
        monkeypatch.setattr("hermes_cli.providers.HERMES_OVERLAYS", {})
        monkeypatch.setattr(
            "hermes_cli.runtime_provider.resolve_runtime_provider",
            lambda **kwargs: {
                "provider": "openrouter",
                "api_key": "test-key",
                "base_url": "https://openrouter.ai/api/v1",
                "api_mode": "chat_completions",
            }
        )
        
        # User on openrouter without explicit base_url
        result = switch_model(
            raw_input="some-model",
            current_provider="openrouter",
            current_model="gpt-4",
            current_base_url="",
            current_api_key="",
            is_global=False,
            explicit_provider="",
            user_providers={},
            custom_providers=[],
        )
        
        # Provider should have switched
        assert result.target_provider == "openrouter"
        # Log message should indicate the switch
        assert "matched OpenRouter catalog" in caplog.text
        assert "Switching provider" in caplog.text

    def test_explicit_config_logs_debug_message(self, monkeypatch, caplog):
        """When explicit config is detected, debug log should show skipping OpenRouter."""
        # Set log level to capture DEBUG messages
        caplog.set_level(logging.DEBUG, logger="hermes_cli.model_switch")
        
        # Mock other dependencies
        monkeypatch.setattr("hermes_cli.models.fetch_openrouter_models", lambda: [])
        monkeypatch.setattr("agent.models_dev.fetch_models_dev", lambda: {})
        monkeypatch.setattr("hermes_cli.providers.HERMES_OVERLAYS", {})
        
        # User has explicit config
        result = switch_model(
            raw_input="MiniMax-M2.5",
            current_provider="minimax-cn",
            current_model="MiniMax-M2.7",
            current_base_url="https://api.minimaxi.com/v1",
            current_api_key="test-key",
            is_global=False,
            explicit_provider="",
            user_providers={},
            custom_providers=[],
        )
        
        # Should succeed
        assert result.success is True
        # Debug log should indicate skipping
        assert "Skipping OpenRouter catalog detection" in caplog.text
        assert "explicit provider config found" in caplog.text
