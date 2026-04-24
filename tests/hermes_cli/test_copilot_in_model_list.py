"""Tests for GitHub Copilot entries shown in the /model picker."""

import os
from unittest.mock import patch

from hermes_cli.model_switch import list_authenticated_providers


@patch.dict(os.environ, {"GH_TOKEN": "test-key"}, clear=False)
def test_copilot_picker_keeps_curated_copilot_models_when_live_catalog_unavailable():
    with patch("agent.models_dev.fetch_models_dev", return_value={}), \
         patch("hermes_cli.models._resolve_copilot_catalog_api_key", return_value="gh-token"), \
         patch("hermes_cli.models._fetch_github_models", return_value=None):
        providers = list_authenticated_providers(current_provider="openrouter", max_models=50)

    copilot = next((p for p in providers if p["slug"] == "copilot"), None)

    assert copilot is not None
    assert "gpt-5.4" in copilot["models"]
    assert "claude-sonnet-4.6" in copilot["models"]
    assert "claude-sonnet-4" in copilot["models"]
    assert "claude-sonnet-4.5" in copilot["models"]
    assert "claude-haiku-4.5" in copilot["models"]
    assert "gemini-3.1-pro-preview" in copilot["models"]
    assert "claude-opus-4.6" not in copilot["models"]


@patch.dict(os.environ, {"GH_TOKEN": "test-key"}, clear=False)
def test_copilot_picker_uses_live_catalog_when_available():
    live_models = ["gpt-5.4", "claude-sonnet-4.6", "gemini-3.1-pro-preview"]

    with patch("agent.models_dev.fetch_models_dev", return_value={}), \
         patch("hermes_cli.models._resolve_copilot_catalog_api_key", return_value="gh-token"), \
         patch("hermes_cli.models._fetch_github_models", return_value=live_models):
        providers = list_authenticated_providers(current_provider="openrouter", max_models=50)

    copilot = next((p for p in providers if p["slug"] == "copilot"), None)

    assert copilot is not None
    assert copilot["models"] == live_models
    assert copilot["total_models"] == len(live_models)
