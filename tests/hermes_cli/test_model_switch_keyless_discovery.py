"""Tests for keyless endpoint live model discovery in /model picker.

Regression test: Section 3 (providers:) previously gated live /v1/models
discovery on a non-empty api_key, skipping keyless endpoints like LM Studio
or llama.cpp. Section 4 (custom_providers:) already handled this correctly
by probing when no pre-configured models exist.
"""

import pytest
from hermes_cli.model_switch import list_authenticated_providers


def test_keyless_provider_probes_live_models_when_no_configured_models(monkeypatch):
    """Keyless providers (no api_key) should still probe /v1/models when no
    pre-configured models exist.

    This matches Section 4's behavior: ``bool(api_url) and
    (bool(api_key) or not models)``.
    """
    monkeypatch.setattr("agent.models_dev.fetch_models_dev", lambda: {})
    monkeypatch.setattr("hermes_cli.providers.HERMES_OVERLAYS", {})

    # Simulate a keyless provider (LM Studio, llama.cpp, etc.)
    user_providers = {
        "lmstudio": {
            "name": "LM Studio",
            "api": "http://localhost:1234/v1",
            # No api_key, no models, no default_model
        },
    }

    live_models = ["qwen3-8b", "llama-3-8b", "mistral-7b"]

    def fake_fetch_api_models(api_key, api_url, timeout=5.0, api_mode=None):
        """Simulate /v1/models returning a list for a keyless endpoint."""
        if api_url and "localhost" in api_url:
            return live_models
        return None

    monkeypatch.setattr(
        "hermes_cli.models.fetch_api_models", fake_fetch_api_models
    )

    providers = list_authenticated_providers(
        current_provider="lmstudio",
        user_providers=user_providers,
        custom_providers=[],
        max_models=50,
    )

    user_prov = next(
        (p for p in providers if p.get("is_user_defined") and p["slug"] == "lmstudio"),
        None,
    )

    assert user_prov is not None, "lmstudio provider should appear in results"
    assert user_prov["total_models"] == len(live_models), (
        f"Expected {len(live_models)} live models, got {user_prov['total_models']}"
    )
    for m in live_models:
        assert m in user_prov["models"], f"Live model '{m}' missing from picker"


def test_keyless_provider_does_not_override_configured_models(monkeypatch):
    """When a keyless provider already has pre-configured models, do NOT
    override them with live discovery (consistent with Section 4 logic:
    ``not grp['models']`` must be true for keyless probing).
    """
    monkeypatch.setattr("agent.models_dev.fetch_models_dev", lambda: {})
    monkeypatch.setattr("hermes_cli.providers.HERMES_OVERLAYS", {})

    user_providers = {
        "local-llm": {
            "name": "Local LLM",
            "api": "http://localhost:8080/v1",
            "default_model": "my-custom-model",
            # No api_key, but has a configured model
        },
    }

    def fake_fetch_api_models(api_key, api_url, timeout=5.0, api_mode=None):
        # Should NOT be called since models_list is non-empty
        return ["should-not-appear"]

    monkeypatch.setattr(
        "hermes_cli.models.fetch_api_models", fake_fetch_api_models
    )

    providers = list_authenticated_providers(
        current_provider="local-llm",
        user_providers=user_providers,
        custom_providers=[],
        max_models=50,
    )

    user_prov = next(
        (p for p in providers if p.get("is_user_defined") and p["slug"] == "local-llm"),
        None,
    )

    assert user_prov is not None
    # Should show the configured model, not the live-discovered ones
    assert "my-custom-model" in user_prov["models"]
    assert "should-not-appear" not in user_prov["models"]


def test_keyed_provider_still_probes_live_models(monkeypatch):
    """Providers WITH api_key should continue to probe live models as before
    (no regression in the existing behavior).
    """
    monkeypatch.setattr("agent.models_dev.fetch_models_dev", lambda: {})
    monkeypatch.setattr("hermes_cli.providers.HERMES_OVERLAYS", {})

    user_providers = {
        "openai-custom": {
            "name": "Custom OpenAI",
            "api": "https://api.example.com/v1",
            "api_key": "sk-test-key",
        },
    }

    live_models = ["gpt-4o", "gpt-4o-mini"]

    def fake_fetch_api_models(api_key, api_url, timeout=5.0, api_mode=None):
        if api_key == "sk-test-key":
            return live_models
        return None

    monkeypatch.setattr(
        "hermes_cli.models.fetch_api_models", fake_fetch_api_models
    )

    providers = list_authenticated_providers(
        current_provider="openai-custom",
        user_providers=user_providers,
        custom_providers=[],
        max_models=50,
    )

    user_prov = next(
        (p for p in providers if p.get("is_user_defined") and p["slug"] == "openai-custom"),
        None,
    )

    assert user_prov is not None
    assert user_prov["total_models"] == len(live_models)
    for m in live_models:
        assert m in user_prov["models"]
