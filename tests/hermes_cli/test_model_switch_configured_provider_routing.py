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
         patch("hermes_cli.models.validate_requested_model", return_value=validation), \
         patch("hermes_cli.models.detect_provider_for_model", return_value=None), \
         patch("hermes_cli.model_switch.get_model_info", return_value=None), \
         patch("hermes_cli.model_switch.get_model_capabilities", return_value=None), \
         patch(
             "hermes_cli.runtime_provider.resolve_runtime_provider",
             return_value={
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


def test_compatibility_dedup_prevents_duplicate_match():
    """A custom_providers entry that carries a ``provider_key`` (i.e. was
    converted from a new-style ``providers.<slug>`` block by
    ``get_compatible_custom_providers()``) must not cause the same model
    to be reported as declared by both the bare slug *and* the
    ``custom:<slug>`` form."""
    user_providers = {
        "openai-codex": {
            "name": "OpenAI Codex",
            "base_url": "http://codex/v1",
            "models": ["gpt-5.9-codex-hidden"],
        }
    }
    # This is what get_compatible_custom_providers() produces:
    # a custom_providers entry with ``provider_key`` pointing back to the
    # original slug.  Without the dedup the model would match both
    # ``openai-codex`` and ``custom:openai-codex``.
    custom_providers = [
        {
            "name": "openai-codex",
            "provider_key": "openai-codex",
            "base_url": "http://codex/v1",
            "models": ["gpt-5.9-codex-hidden"],
        }
    ]
    result = _run_switch(
        raw_input="gpt-5.9-codex-hidden",
        current_provider="claude-codex",
        current_model="claude-sonnet-4",
        user_providers=user_providers,
        custom_providers=custom_providers,
    )
    assert result.success is True, result.error_message
    assert result.target_provider == "openai-codex"


def test_compatibility_dedup_with_mixed_case_slug():
    """Mixed-case provider slugs (e.g. ``providers: {OAI: ...}``) must
    also be deduplicated.  ``OAI`` and ``custom:OAI`` should not both
    appear as matches."""
    user_providers = {
        "OAI": {
            "name": "OpenAI",
            "base_url": "http://oai/v1",
            "models": ["gpt-5.9-codex-hidden"],
        }
    }
    custom_providers = [
        {
            "name": "OAI",
            "provider_key": "OAI",
            "base_url": "http://oai/v1",
            "models": ["gpt-5.9-codex-hidden"],
        }
    ]
    result = _run_switch(
        raw_input="gpt-5.9-codex-hidden",
        current_provider="claude-codex",
        current_model="claude-sonnet-4",
        user_providers=user_providers,
        custom_providers=custom_providers,
    )
    assert result.success is True, result.error_message
    assert result.target_provider == "OAI"
