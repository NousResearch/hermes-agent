"""Tests for surfacing free OpenRouter models in the /model picker."""

import os
from unittest.mock import patch

from hermes_cli.model_switch import list_authenticated_providers, switch_model


@patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"}, clear=True)
def test_free_openrouter_group_appears_without_fake_provider():
    with patch("agent.models_dev.fetch_models_dev", return_value={}), \
         patch("hermes_cli.models.fetch_openrouter_free_models", return_value=["openrouter/free", "meta-llama/model:free"]):
        providers = list_authenticated_providers(current_provider="openrouter", max_models=50)

    free_group = next((p for p in providers if p.get("source") == "openrouter-free"), None)
    assert free_group is not None
    assert free_group["slug"] == "openrouter"
    assert free_group["name"] == "OpenRouter free models"
    assert free_group["models"] == ["openrouter/free", "meta-llama/model:free"]
    assert "free" not in {p["slug"] for p in providers}


@patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"}, clear=True)
def test_free_openrouter_group_omitted_when_discovery_fails_or_empty():
    with patch("agent.models_dev.fetch_models_dev", return_value={}), \
         patch("hermes_cli.models.fetch_openrouter_free_models", return_value=[]):
        providers = list_authenticated_providers(current_provider="openrouter", max_models=50)

    assert not any(p.get("source") == "openrouter-free" for p in providers)


def test_selecting_free_openrouter_model_resolves_to_openrouter_provider():
    with patch("hermes_cli.model_switch.resolve_alias", return_value=None), \
         patch("hermes_cli.model_switch.list_provider_models", return_value=[]), \
         patch("hermes_cli.runtime_provider.resolve_runtime_provider",
               return_value={"api_key": "test-key", "base_url": "https://openrouter.ai/api/v1", "api_mode": "chat_completions"}), \
         patch("hermes_cli.models.validate_requested_model", return_value={"accepted": True, "persist": True, "recognized": True, "message": None}), \
         patch("hermes_cli.model_switch.get_model_info", return_value=None), \
         patch("hermes_cli.model_switch.get_model_capabilities", return_value=None), \
         patch("hermes_cli.models.detect_provider_for_model", return_value=None):
        result = switch_model(
            raw_input="meta-llama/model:free",
            current_provider="openai-codex",
            current_model="gpt-5.3-codex",
            explicit_provider="openrouter",
        )

    assert result.success, result.error_message
    assert result.target_provider == "openrouter"
    assert result.new_model == "meta-llama/model:free"
