"""Regression tests for #45006: typed `/model <name>` resolution must route a
model declared in user/custom provider config to that provider instead of
leaving it on the current provider and soft-accepting it.

Repro: with the current provider set to ``openai-codex``, typing
``/model qwen3.5-4b`` (a model the user declares under ``providers.<slug>`` or
``custom_providers``) showed ``Provider: OpenAI Codex`` — because typed
detection only consulted static catalogs / OpenRouter, never the user's
configured provider model lists, so the name stayed on Codex and was
soft-accepted as an unknown hidden Codex model.

The fix adds an exact-match configured-provider detection step in
``switch_model`` that runs before ``detect_provider_for_model`` and before
common-path validation.  These tests pin its precedence rules and prove the
deliberately-supported Codex hidden-model soft-accept (#16172 / #19729) is left
intact when nothing in config matches.

Hermetic: the model-resolution chain is fully mocked (no network), mirroring
``tests/hermes_cli/test_user_providers_model_switch.py``.
"""

from unittest.mock import patch

from hermes_cli.model_switch import switch_model

_ACCEPTED = {"accepted": True, "persist": True, "recognized": True, "message": None}
_REJECTED = {"accepted": False, "persist": False, "recognized": False, "message": "not found"}
# What validate_requested_model returns for an unknown id on openai-codex: it
# soft-accepts with a "may be a hidden model" note (#16172 / #19729).
_CODEX_SOFT_ACCEPT = {
    "accepted": True,
    "persist": True,
    "recognized": False,
    "message": (
        "Note: `gpt-5.9-codex-hidden` was not found in the OpenAI Codex model "
        "listing. It may still work if your account has access to a newer or "
        "hidden model ID."
    ),
}


def _run_switch(
    *,
    raw_input,
    current_provider,
    user_providers=None,
    custom_providers=None,
    validation=_ACCEPTED,
    current_model="old-model",
    current_base_url="",
    runtime_resolution=None,
):
    """Drive ``switch_model`` with the resolution chain mocked out.

    Every external lookup that would otherwise hit catalogs/network is patched:
    alias resolution, aggregator catalog, ``detect_provider_for_model`` (so step
    e is a no-op and cannot accidentally reroute), validation, credential
    resolution, normalization, and model metadata.  This isolates the new
    configured-provider detection step.
    """
    with patch("hermes_cli.model_switch.resolve_alias", return_value=None), \
         patch("hermes_cli.model_switch.list_provider_models", return_value=[]), \
         patch("hermes_cli.model_switch.normalize_model_for_provider", side_effect=lambda model, provider: model), \
         patch("hermes_cli.models_validate.validate_requested_model", return_value=validation), \
         patch("hermes_cli.models.detect_provider_for_model", return_value=None), \
         patch("hermes_cli.model_switch.get_model_info", return_value=None), \
         patch("hermes_cli.model_switch.get_model_capabilities", return_value=None), \
         patch(
             "hermes_cli.runtime_provider.resolve_runtime_provider",
             return_value=runtime_resolution or {
                 "api_key": "***",
                 "base_url": current_base_url or "http://resolved/v1",
                 "api_mode": "",
             },
         ):
        return switch_model(
            raw_input=raw_input,
            current_provider=current_provider,
            current_model=current_model,
            current_base_url=current_base_url,
            user_providers=user_providers or {},
            custom_providers=custom_providers or [],
        )


def test_duplicate_aliases_of_one_configured_provider_preserve_active_identity():
    token_source = lambda: "opaque-token"
    provider_config = {
        "name": "Databricks Unity AI Gateway",
        "provider_key": "databricks",
        "base_url": "https://workspace.invalid/serving-endpoints",
        "models": {
            "inventory-model": {
                "api_mode": "anthropic_messages",
                "base_url": "https://workspace.invalid/anthropic",
            }
        },
    }
    result = _run_switch(
        raw_input="inventory-model",
        current_provider="custom:databricks",
        current_model="old-model",
        current_base_url="https://workspace.invalid/old",
        user_providers={"databricks": provider_config},
        custom_providers=[provider_config],
        runtime_resolution={
            "provider": "custom",
            "requested_provider": "custom:databricks",
            "api_key": token_source,
            "base_url": "https://workspace.invalid/anthropic",
            "api_mode": "anthropic_messages",
        },
    )

    assert result.success is True, result.error_message
    assert result.target_provider == "custom:databricks"
    assert result.requested_provider == "custom:databricks"
    assert result.runtime_provider == "custom"
    assert result.api_key is token_source
    assert result.base_url == "https://workspace.invalid/anthropic"
    assert result.api_mode == "anthropic_messages"


def test_distinct_configured_providers_with_same_model_remain_ambiguous():
    result = _run_switch(
        raw_input="shared-model",
        current_provider="openai-codex",
        user_providers={
            "provider-one": {"base_url": "https://one.invalid/v1", "models": ["shared-model"]},
            "provider-two": {"base_url": "https://two.invalid/v1", "models": ["shared-model"]},
        },
    )

    assert result.success is False
    assert "multiple configured providers" in result.error_message
    assert "provider-one" in result.error_message
    assert "provider-two" in result.error_message


def test_catalog_only_model_routes_through_its_configured_provider():
    result = _run_switch(
        raw_input="catalog-model",
        current_provider="openai-codex",
        user_providers={
            "enterprise-gateway": {
                "base_url": "https://gateway.invalid/v1",
                "models": ["verified-model"],
                "catalog_models": ["verified-model", "catalog-model"],
            },
        },
        validation=_REJECTED,
    )

    assert result.success is True, result.error_message
    assert result.target_provider == "enterprise-gateway"
    assert result.new_model == "catalog-model"




def test_default_model_only_declaration_routes():
    """A model declared ONLY via `default_model` (not in `models`) still routes
    to that configured provider (#45006 — default_model is a declaring field)."""
    user_providers = {
        "local-ollama": {
            "name": "Local Ollama",
            "base_url": "http://localhost:11434/v1",
            "default_model": "qwen3.5-4b",
        }
    }
    result = _run_switch(
        raw_input="qwen3.5-4b",
        current_provider="openai-codex",
        current_model="gpt-5.4",
        user_providers=user_providers,
    )
    assert result.success is True, result.error_message
    assert result.target_provider == "local-ollama"
    assert result.new_model == "qwen3.5-4b"




def test_xai_oauth_soft_accept_preserved_when_no_match():
    """The xai-oauth hidden-model soft-accept (sibling of openai-codex) is also
    a no-op when config declares no matching model."""
    user_providers = {
        "local-ollama": {"base_url": "http://x/v1", "models": ["some-other-model"]},
    }
    result = _run_switch(
        raw_input="grok-hidden-preview",
        current_provider="xai-oauth",
        current_model="grok-4",
        user_providers=user_providers,
        validation=_CODEX_SOFT_ACCEPT,
    )
    assert result.success is True, result.error_message
    assert result.target_provider == "xai-oauth"
