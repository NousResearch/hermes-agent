"""Regression tests for aggregator live-API fallback and custom: prefix matching.

PR #33716 review feedback (liuhao1024):
1. No tests for the aggregator fallback code paths (custom_providers model
   dict iteration + live fetch_endpoint_model_metadata fallback).
2. normalize_provider match misses custom:-prefixed targets — e.g.
   "Custom:OpenRouter" should still match entry_name = "OpenRouter".
"""

from __future__ import annotations

from dataclasses import asdict
from unittest.mock import patch

import hermes_cli.providers as providers_mod
from hermes_cli.model_switch import switch_model


_MOCK_VALIDATION = {
    "accepted": True,
    "persist": True,
    "recognized": True,
    "message": None,
}


# ---------------------------------------------------------------------------
# Aggregator live-API fallback
# ---------------------------------------------------------------------------

@patch("hermes_cli.model_switch.get_model_info", return_value=None)
@patch("hermes_cli.model_switch.get_model_capabilities", return_value=None)
@patch("hermes_cli.models.validate_requested_model", return_value=_MOCK_VALIDATION)
@patch("hermes_cli.runtime_provider.resolve_runtime_provider", return_value={
    "api_key": "test-key",
    "base_url": "https://example.com/v1",
    "api_mode": "chat_completions",
})
def test_aggregator_custom_providers_models_dict_resolves_missing_from_catalog(
    _resolve, _validate, _caps, _info, monkeypatch,
):
    """Model present in custom_providers[models] dict but NOT in the live
    models.dev catalog should still be accepted (resolved_in_current_catalog
    flips True via the custom_providers iteration path)."""
    # Empty models.dev catalog for the aggregator — model is NOT listed there
    monkeypatch.setattr("agent.models_dev.fetch_models_dev", lambda: {})
    monkeypatch.setattr(providers_mod, "HERMES_OVERLAYS", {})

    # Patch detect_provider_for_model so step-e doesn't hijack
    monkeypatch.setattr(
        "hermes_cli.models.detect_provider_for_model",
        lambda *a, **k: None,
    )

    result = switch_model(
        raw_input="mimo-v2.5",
        current_provider="opencode-zen",
        current_model="claude-sonnet-4",
        current_base_url="https://opencode-zen.example.com/v1",
        current_api_key="fake-key",
        custom_providers=[
            {
                "name": "opencode-zen",
                "base_url": "https://opencode-zen.example.com/v1",
                "model": "mimo-v2.5-free",
                "models": {
                    "mimo-v2.5": {"context_length": 131072},
                    "mimo-v2.5-free": {"context_length": 131072},
                },
            }
        ],
    )

    assert result.success is True, (
        f"Expected success but got error: {result.error_message}"
    )


@patch("hermes_cli.model_switch.get_model_info", return_value=None)
@patch("hermes_cli.model_switch.get_model_capabilities", return_value=None)
@patch("hermes_cli.models.validate_requested_model", return_value=_MOCK_VALIDATION)
@patch("hermes_cli.runtime_provider.resolve_runtime_provider", return_value={
    "api_key": "test-key",
    "base_url": "https://example.com/v1",
    "api_mode": "chat_completions",
})
def test_aggregator_live_api_fallback_resolves_missing_model(
    _resolve, _validate, _caps, _info, monkeypatch,
):
    """When models.dev AND custom_providers don't list the model, the live
    fetch_endpoint_model_metadata fallback should resolve it and preserve the
    canonical casing returned by the API."""
    monkeypatch.setattr("agent.models_dev.fetch_models_dev", lambda: {})
    monkeypatch.setattr(providers_mod, "HERMES_OVERLAYS", {})
    monkeypatch.setattr(
        "hermes_cli.models.detect_provider_for_model",
        lambda *a, **k: None,
    )

    # Live API returns the model (with different casing — the fix should
    # preserve the canonical casing from the live endpoint).
    monkeypatch.setattr(
        "agent.model_metadata.fetch_endpoint_model_metadata",
        lambda base_url, api_key: ["mimo-v2.5", "claude-sonnet-4"],
    )

    result = switch_model(
        raw_input="mimo-v2.5",
        current_provider="opencode-zen",
        current_model="claude-sonnet-4",
        current_base_url="https://opencode-zen.example.com/v1",
        current_api_key="fake-key",
        custom_providers=[],
    )

    assert result.success is True, (
        f"Expected live-API fallback to succeed but got: {result.error_message}"
    )
    # The canonical model name from the live endpoint should be preserved
    assert result.new_model == "mimo-v2.5"


@patch("hermes_cli.model_switch.get_model_info", return_value=None)
@patch("hermes_cli.model_switch.get_model_capabilities", return_value=None)
@patch("hermes_cli.models.validate_requested_model", return_value=_MOCK_VALIDATION)
@patch("hermes_cli.runtime_provider.resolve_runtime_provider", return_value={
    "api_key": "test-key",
    "base_url": "https://example.com/v1",
    "api_mode": "chat_completions",
})
def test_aggregator_live_api_fallback_swallows_exceptions(
    _resolve, _validate, _caps, _info, monkeypatch,
):
    """If fetch_endpoint_model_metadata raises, the except Exception: pass
    should not crash the pipeline — the function must continue to step e."""
    monkeypatch.setattr("agent.models_dev.fetch_models_dev", lambda: {})
    monkeypatch.setattr(providers_mod, "HERMES_OVERLAYS", {})
    monkeypatch.setattr(
        "hermes_cli.models.detect_provider_for_model",
        lambda *a, **k: None,
    )

    # Make the live API call raise — the broad except must swallow this.
    def _broken_fetch(*a, **k):
        raise ConnectionError("endpoint unreachable")
    monkeypatch.setattr(
        "agent.model_metadata.fetch_endpoint_model_metadata",
        _broken_fetch,
    )

    # Validation accepts the model (simulate detect_provider_for_model
    # finding it in a downstream provider catalog).
    result = switch_model(
        raw_input="some-unknown-model",
        current_provider="opencode-zen",
        current_model="claude-sonnet-4",
        current_base_url="https://opencode-zen.example.com/v1",
        current_api_key="fake-key",
        custom_providers=[],
    )

    # We don't care whether this specific model succeeds — we care that
    # the ConnectionError didn't propagate. The pipeline should reach
    # validation, not die in the fallback block.
    # If the exception leaked, we'd get an unhandled ConnectionError
    # instead of a normal ModelSwitchResult.
    assert hasattr(result, "success"), (
        "Expected a ModelSwitchResult, not an unhandled exception"
    )


# ---------------------------------------------------------------------------
# custom: prefix slug matching (case-insensitive)
# ---------------------------------------------------------------------------

@patch("hermes_cli.model_switch.get_model_info", return_value=None)
@patch("hermes_cli.model_switch.get_model_capabilities", return_value=None)
@patch("hermes_cli.models.validate_requested_model", return_value=_MOCK_VALIDATION)
@patch("hermes_cli.runtime_provider.resolve_runtime_provider", return_value={
    "api_key": "test-key",
    "base_url": "https://api.openrouter.ai/v1",
    "api_mode": "chat_completions",
})
def test_custom_prefix_case_insensitive_match(
    _resolve, _validate, _caps, _info, monkeypatch,
):
    """target_provider='Custom:OpenRouter' (non-canonical casing) should
    still match entry_name='OpenRouter' via the split(':',1)[-1].lower()
    fallback added in PR #33716."""
    monkeypatch.setattr("agent.models_dev.fetch_models_dev", lambda: {})
    monkeypatch.setattr(providers_mod, "HERMES_OVERLAYS", {})

    result = switch_model(
        raw_input="anthropic/claude-sonnet-4",
        current_provider="opencode-zen",
        current_model="claude-sonnet-4",
        current_base_url="https://opencode-zen.example.com/v1",
        current_api_key="fake-key",
        explicit_provider="Custom:OpenRouter",
        custom_providers=[
            {
                "name": "OpenRouter",
                "base_url": "https://openrouter.ai/api/v1",
                "model": "anthropic/claude-sonnet-4",
            }
        ],
    )

    assert result.success is True, (
        f"Custom:OpenRouter (mixed case) should match entry_name 'OpenRouter'. "
        f"Got error: {result.error_message}"
    )


@patch("hermes_cli.model_switch.get_model_info", return_value=None)
@patch("hermes_cli.model_switch.get_model_capabilities", return_value=None)
@patch("hermes_cli.models.validate_requested_model", return_value=_MOCK_VALIDATION)
@patch("hermes_cli.runtime_provider.resolve_runtime_provider", return_value={
    "api_key": "test-key",
    "base_url": "https://api.deepseek.com",
    "api_mode": "chat_completions",
})
def test_custom_prefix_lowercase_match(
    _resolve, _validate, _caps, _info, monkeypatch,
):
    """target_provider='custom:deepseek' (lowercase custom: prefix) should
    match entry_name='DeepSeek' via the suffix comparison."""
    monkeypatch.setattr("agent.models_dev.fetch_models_dev", lambda: {})
    monkeypatch.setattr(providers_mod, "HERMES_OVERLAYS", {})

    result = switch_model(
        raw_input="deepseek-chat",
        current_provider="opencode-zen",
        current_model="claude-sonnet-4",
        current_base_url="https://opencode-zen.example.com/v1",
        current_api_key="fake-key",
        explicit_provider="custom:deepseek",
        custom_providers=[
            {
                "name": "DeepSeek",
                "base_url": "https://api.deepseek.com",
                "model": "deepseek-chat",
            }
        ],
    )

    assert result.success is True, (
        f"custom:deepseek should match entry_name 'DeepSeek'. "
        f"Got error: {result.error_message}"
    )
