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

import pytest

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
         patch("hermes_cli.models_validate.validate_requested_model", return_value=validation), \
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



def test_compat_projection_of_same_provider_is_not_ambiguous():
    """The gateway/TUI/CLI pass ``providers:`` AND ``get_compatible_custom_providers()``, which re-lists
    each ``providers.<slug>`` row as ``custom:<name>``. One configured endpoint must route, not be
    rejected as 'declared by multiple configured providers' (#112788)."""
    from hermes_cli.config import get_compatible_custom_providers

    user_providers = {"relay": {"name": "relay", "api": "https://relay.example/v1",
                                "key_env": "RELAY_KEY", "default_model": "claude-opus-4-7"}}
    result = _run_switch(
        raw_input="claude-opus-4-7", current_provider="openrouter", user_providers=user_providers,
        custom_providers=get_compatible_custom_providers({"providers": user_providers}))
    assert result.success is True, result.error_message
    assert result.target_provider == "relay"


def test_distinct_legacy_endpoint_with_same_model_stays_ambiguous():
    """Control for #112788: a hand-written ``custom_providers:`` row (no provider_key) that declares
    the same model on a different endpoint is still a genuinely separate candidate."""
    from hermes_cli.config import get_compatible_custom_providers

    user_providers = {"relay": {"name": "relay", "api": "https://relay.example/v1",
                                "key_env": "RELAY_KEY", "default_model": "claude-opus-4-7"}}
    cfg = {"providers": user_providers, "custom_providers": [
        {"name": "backup-relay", "base_url": "https://backup.example/v1", "key_env": "BACKUP_KEY",
         "model": "claude-opus-4-7"}]}
    result = _run_switch(
        raw_input="claude-opus-4-7", current_provider="openrouter", user_providers=user_providers,
        custom_providers=get_compatible_custom_providers(cfg))
    assert result.success is False
    assert "multiple configured providers" in (result.error_message or "")
    assert "custom:backup-relay" in result.error_message and "relay" in result.error_message


_RELAY = {"name": "relay", "api": "https://relay.example/v1", "key_env": "RELAY_KEY",
          "default_model": "claude-opus-4-7"}
_LEGACY_RELAY = {"name": "relay", "base_url": "https://relay.example/v1", "key_env": "RELAY_KEY",
                 "model": "claude-opus-4-7"}


def test_legacy_duplicate_of_same_endpoint_collapses_by_identity():
    """A hand-migrated config that kept the same endpoint under ``providers.relay`` AND as a legacy
    ``custom_providers`` row (same name, endpoint, credential, protocol) is one provider: ``/model``
    routes to ``relay`` instead of calling it ambiguous (#112788)."""
    from hermes_cli.config import get_compatible_custom_providers

    user_providers = {"relay": _RELAY}
    cfg = {"providers": user_providers, "custom_providers": [_LEGACY_RELAY]}
    result = _run_switch(
        raw_input="claude-opus-4-7", current_provider="openrouter", user_providers=user_providers,
        custom_providers=get_compatible_custom_providers(cfg))
    assert result.success is True, result.error_message
    assert result.target_provider == "relay"


@pytest.mark.parametrize("delta", [
    {"key_env": "OTHER_KEY"}, {"api_mode": "anthropic_messages"}, {"base_url": "https://backup.example/v1"},
], ids=["credential", "protocol", "endpoint"])
def test_same_named_legacy_row_with_different_identity_stays_ambiguous(delta):
    """Identity collapse is exact: a legacy row sharing the display name but differing in credential,
    wire protocol or endpoint is still a second candidate (#112788 acceptance criterion)."""
    from hermes_cli.config import get_compatible_custom_providers

    user_providers = {"relay": _RELAY}
    cfg = {"providers": user_providers, "custom_providers": [{**_LEGACY_RELAY, **delta}]}
    result = _run_switch(
        raw_input="claude-opus-4-7", current_provider="openrouter", user_providers=user_providers,
        custom_providers=get_compatible_custom_providers(cfg))
    assert result.success is False
    assert "multiple configured providers" in (result.error_message or "")


def test_session_on_projection_slug_keeps_its_slug():
    """A session whose current provider is the compat projection slug ``custom:relay`` switching to
    a model ``providers.relay`` declares stays on ``custom:relay`` — same provider, no flip (#112788)."""
    from hermes_cli.config import get_compatible_custom_providers

    user_providers = {"relay": _RELAY}
    result = _run_switch(
        raw_input="claude-opus-4-7", current_provider="custom:relay", user_providers=user_providers,
        custom_providers=get_compatible_custom_providers({"providers": user_providers}))
    assert result.success is True, result.error_message
    assert result.target_provider == "custom:relay"


def test_raw_list_provider_key_pointing_elsewhere_stays_ambiguous():
    """Raw-list fallback (callers pass ``cfg['custom_providers']`` verbatim when the compat view
    fails): a hand-written entry whose ``provider_key`` names a ``providers`` slug but points at a
    different endpoint is a distinct candidate, not silently hidden (#112788)."""
    user_providers = {"relay": _RELAY}
    raw = [{"name": "relay", "provider_key": "relay", "base_url": "https://backup.example/v1",
            "key_env": "BACKUP_KEY", "model": "claude-opus-4-7"}]
    result = _run_switch(
        raw_input="claude-opus-4-7", current_provider="openrouter", user_providers=user_providers,
        custom_providers=raw)
    assert result.success is False
    assert "multiple configured providers" in (result.error_message or "")


def test_legacy_duplicate_keeps_its_own_declared_models():
    """Folding an identity-equal legacy row into ``providers.relay`` must not drop the models only
    that row declares: ``/model gpt-5.4-mini`` still routes to the shared endpoint instead of
    falling through to the current provider (#112788 review follow-up)."""
    from hermes_cli.config import get_compatible_custom_providers

    user_providers = {"relay": _RELAY}
    cfg = {"providers": user_providers,
           "custom_providers": [{**_LEGACY_RELAY, "models": ["gpt-5.4", "gpt-5.4-mini"]}]}
    result = _run_switch(
        raw_input="gpt-5.4-mini", current_provider="openrouter", user_providers=user_providers,
        custom_providers=get_compatible_custom_providers(cfg))
    assert result.success is True, result.error_message
    assert result.target_provider == "relay"
    assert result.new_model == "gpt-5.4-mini"


def test_raw_list_provider_key_with_different_credential_stays_ambiguous():
    """Raw-list fallback: a ``provider_key: relay`` stamp on the same endpoint but a DIFFERENT
    credential is not the row's projection — credential identity differs, ambiguity is preserved."""
    user_providers = {"relay": _RELAY}
    raw = [{"name": "relay", "provider_key": "relay", "base_url": "https://relay.example/v1",
            "key_env": "OTHER_KEY", "model": "claude-opus-4-7"}]
    result = _run_switch(
        raw_input="claude-opus-4-7", current_provider="openrouter", user_providers=user_providers,
        custom_providers=raw)
    assert result.success is False
    assert "multiple configured providers" in (result.error_message or "")


# ---------------------------------------------------------------------------
# #62240: step-e static-catalog reroute must not select an unconfigured
# provider.  ``detect_provider_for_model`` matches Hermes' built-in
# CURATED_MODELS, so a bare name that also lives there (e.g. ``gpt-5.5`` under
# ``openai-api``) would pick that vendor as the switch target even when the
# user never configured it.  The reroute is now gated on the detected provider
# actually having credentials, so an unconfigured static-catalog match no
# longer wins the target selection.
# ---------------------------------------------------------------------------

def _run_switch_detect(
    *,
    raw_input,
    current_provider,
    detected,
    authed,
    custom_providers=None,
    current_model="old-model",
    current_base_url="https://api.deepseek.com/v1",
):
    """Drive ``switch_model`` with a live step-e ``detect_provider_for_model``
    return and a controlled authenticated-provider set.

    Only step e is exercised here: alias/aggregator/config paths are neutralised
    so ``detect_provider_for_model`` decides the reroute, and
    ``get_authenticated_provider_slugs`` is pinned to ``authed`` to prove the
    credential gate.
    """
    with patch("hermes_cli.model_switch.resolve_alias", return_value=None), \
         patch("hermes_cli.model_switch.list_provider_models", return_value=[]), \
         patch("hermes_cli.model_switch.normalize_model_for_provider", side_effect=lambda model, provider: model), \
         patch("hermes_cli.models_validate.validate_requested_model", return_value=_ACCEPTED), \
         patch("hermes_cli.models.detect_provider_for_model", return_value=detected), \
         patch("hermes_cli.model_switch.get_authenticated_provider_slugs", return_value=authed), \
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
            custom_providers=custom_providers or [],
        )


def test_static_reroute_skipped_when_detected_provider_unauthenticated():
    """The core #62240 repro: ``/model gpt-5.5`` while on deepseek must NOT
    reroute to the built-in ``openai-api`` catalog when the user has no
    openai-api credentials — stay on the current provider instead of inheriting
    a stale base_url."""
    result = _run_switch_detect(
        raw_input="gpt-5.5",
        current_provider="deepseek",
        detected=("openai-api", "gpt-5.5"),
        authed=["deepseek", "custom:lmuai"],
        custom_providers=[{"name": "lmuai", "base_url": "https://api.lmuai.com/v1"}],
    )
    assert result.success is True, result.error_message
    assert result.target_provider == "deepseek"
    assert result.new_model == "gpt-5.5"


def test_static_reroute_allowed_when_detected_provider_authenticated():
    """No regression: when the user IS authenticated for the detected provider,
    the static reroute still happens."""
    result = _run_switch_detect(
        raw_input="gpt-5.5",
        current_provider="deepseek",
        detected=("openai-api", "gpt-5.5"),
        authed=["deepseek", "openai-api"],
    )
    assert result.success is True, result.error_message
    assert result.target_provider == "openai-api"
    assert result.new_model == "gpt-5.5"


def test_static_reroute_allowed_when_authed_set_unknown():
    """When the authenticated set can't be determined (empty), fall back to the
    historical reroute behaviour so the gate never over-blocks."""
    result = _run_switch_detect(
        raw_input="gpt-5.5",
        current_provider="deepseek",
        detected=("openai-api", "gpt-5.5"),
        authed=[],
    )
    assert result.success is True, result.error_message
    assert result.target_provider == "openai-api"
