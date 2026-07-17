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


def test_typed_configured_model_routes_away_from_openai_codex():
    """The core repro: a model declared under ``providers.<slug>`` typed while
    on ``openai-codex`` routes to the configured provider, not Codex."""
    user_providers = {
        "local-ollama": {
            "name": "Local Ollama",
            "base_url": "http://localhost:11434/v1",
            "models": ["qwen3.5-4b", "kimi-k2.5"],
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


def test_typed_configured_model_routes_to_custom_provider():
    """``custom_providers`` entries route to their ``custom:<name>`` slug."""
    custom_providers = [
        {
            "name": "mylocal",
            "base_url": "http://localhost:1234/v1",
            "model": "qwen3.5-4b",
            "models": {"qwen3.5-4b": {}},
        }
    ]
    result = _run_switch(
        raw_input="qwen3.5-4b",
        current_provider="openai-codex",
        current_model="gpt-5.4",
        custom_providers=custom_providers,
    )
    assert result.success is True, result.error_message
    assert result.target_provider == "custom:mylocal"
    assert result.new_model == "qwen3.5-4b"


def test_current_provider_declaring_model_is_not_rerouted():
    """Precedence rule 4: if the current provider declares the model, keep it —
    even when another configured provider also declares the same id (so this
    must NOT trip the ambiguity guard)."""
    user_providers = {
        "local-ollama": {
            "name": "Local Ollama",
            "base_url": "http://localhost:11434/v1",
            "models": ["qwen3.5-4b"],
        },
        "other-relay": {
            "name": "Other Relay",
            "base_url": "http://other/v1",
            "models": ["qwen3.5-4b"],
        },
    }
    result = _run_switch(
        raw_input="qwen3.5-4b",
        current_provider="local-ollama",
        current_model="kimi-k2.5",
        current_base_url="http://localhost:11434/v1",
        user_providers=user_providers,
    )
    assert result.success is True, result.error_message
    assert result.target_provider == "local-ollama"


def test_ambiguous_configured_model_fails_with_provider_hint():
    """Precedence rule 6: when two non-current providers declare the same id and
    neither is current, fail clearly and point at ``--provider`` — never
    silently pick the first match."""
    user_providers = {
        "relay-a": {
            "name": "Relay A",
            "base_url": "http://a/v1",
            "models": ["qwen3.5-4b"],
        },
        "relay-b": {
            "name": "Relay B",
            "base_url": "http://b/v1",
            "models": ["qwen3.5-4b"],
        },
    }
    result = _run_switch(
        raw_input="qwen3.5-4b",
        current_provider="openai-codex",
        current_model="gpt-5.4",
        user_providers=user_providers,
    )
    assert result.success is False
    assert "--provider" in result.error_message
    assert "relay-a" in result.error_message
    assert "relay-b" in result.error_message


def test_configured_model_absent_from_live_models_accepted_after_reroute():
    """End-to-end synergy: after rerouting to the configured provider, a live
    ``/v1/models`` probe that does NOT list the model is still accepted via the
    existing user-config override — proving the reroute lands on the right
    provider for that override to match."""
    user_providers = {
        "local-ollama": {
            "name": "Local Ollama",
            "base_url": "http://localhost:11434/v1",
            "models": {"qwen3.5-4b": {"context_length": 32768}},
        }
    }
    result = _run_switch(
        raw_input="qwen3.5-4b",
        current_provider="openai-codex",
        current_model="gpt-5.4",
        user_providers=user_providers,
        validation=_REJECTED,
    )
    assert result.success is True, result.error_message
    assert result.target_provider == "local-ollama"
    assert result.new_model == "qwen3.5-4b"


def test_no_configured_match_leaves_current_provider_for_soft_accept():
    """The Codex hidden-model soft-accept (#16172 / #19729) is untouched: an
    unknown id with no config match stays on the current provider and is
    soft-accepted exactly as before."""
    result = _run_switch(
        raw_input="gpt-5.9-codex-hidden",
        current_provider="openai-codex",
        current_model="gpt-5.4",
        # Config is present but declares an unrelated model — detection is a no-op.
        user_providers={
            "local-ollama": {
                "base_url": "http://localhost:11434/v1",
                "models": ["qwen3.5-4b"],
            }
        },
        validation=_CODEX_SOFT_ACCEPT,
    )
    assert result.success is True, result.error_message
    assert result.target_provider == "openai-codex"
    assert result.new_model == "gpt-5.9-codex-hidden"


def test_configured_match_is_case_insensitive_and_returns_canonical_spelling():
    """Matching is case-insensitive but the configured spelling wins, so the
    downstream validation/override path sees the canonical id."""
    user_providers = {
        "local-ollama": {
            "base_url": "http://localhost:11434/v1",
            "models": ["Qwen3.5-4B"],
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
    assert result.new_model == "Qwen3.5-4B"


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


def test_malformed_provider_config_does_not_raise():
    """Garbage shapes in provider config must not crash detection — they're
    skipped and the typed name falls through to the soft-accept no-op."""
    user_providers = {
        "bad1": "not-a-dict",            # non-dict cfg
        "bad2": {"models": 12345},        # models as int
        "bad3": {"models": [None, 7, {"noname": "x"}]},  # junk list items
        "bad4": {"model": {"k": object()}},  # dict with non-target keys
    }
    custom_providers = [
        "not-a-dict",                     # non-dict entry
        {"name": ""},                     # empty name
        {"models": ["unrelated-model"]},  # no name key
    ]
    result = _run_switch(
        raw_input="gpt-5.9-codex-hidden",
        current_provider="openai-codex",
        current_model="gpt-5.4",
        user_providers=user_providers,
        custom_providers=custom_providers,
        validation=_CODEX_SOFT_ACCEPT,
    )
    # No match anywhere -> stays on codex, soft-accepted, no exception.
    assert result.success is True, result.error_message
    assert result.target_provider == "openai-codex"


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


# ---------------------------------------------------------------------------
# Self-duplicate provider view dedupe (#45006)
#
# Callers like the gateway /model path pass BOTH ``cfg["providers"]`` (as
# user_providers) AND ``get_compatible_custom_providers(cfg)`` (as
# custom_providers).  The latter re-exposes every ``providers`` dict entry as a
# legacy custom-provider row, so ONE physical endpoint used to count as TWO
# matches ("relay-lb" + "custom:Relay LB East") and the ambiguity guard
# rejected an unambiguous bare model name.
# ---------------------------------------------------------------------------

_LB_PROVIDERS = {
    "relay-lb": {
        "name": "Relay LB East",
        "base_url": "http://relay-east.internal:8080",
        "api_key": "lb-key",
        "api_mode": "anthropic_messages",
        "default_model": "claude-opus-4-8",
        "models": {"claude-fable-5": {}, "claude-opus-4-8": {}},
    }
}


def _gateway_shaped_views(cfg_providers):
    """Build (user_providers, custom_providers) exactly as the gateway /model
    path does: providers dict + its get_compatible_custom_providers() view."""
    from hermes_cli.config import get_compatible_custom_providers

    return cfg_providers, get_compatible_custom_providers(
        {"providers": cfg_providers}
    )


def test_dual_exposed_user_provider_counts_as_one():
    """The D1 repro: a bare model name declared by exactly one ``providers``
    entry that is ALSO re-exposed via get_compatible_custom_providers() must
    switch successfully — not trip the multi-provider guard."""
    user_providers, custom_providers = _gateway_shaped_views(dict(_LB_PROVIDERS))
    # Sanity: the legacy view really is present (dual exposure is load-bearing
    # for legacy custom_providers consumers and must not be silently dropped).
    assert any(
        e.get("name") == "Relay LB East" for e in custom_providers
    ), custom_providers

    result = _run_switch(
        raw_input="claude-fable-5",
        current_provider="openai-codex",
        current_model="gpt-5.4",
        user_providers=user_providers,
        custom_providers=custom_providers,
    )
    assert result.success is True, result.error_message
    assert result.target_provider == "relay-lb"
    assert result.new_model == "claude-fable-5"


def test_dual_exposed_match_resolves_user_provider_endpoint_and_key():
    """After deduping to the ``providers.<slug>`` entry, credentials must come
    from THAT entry (base_url + api_key), i.e. the reroute takes the
    explicit-provider user-config credential path."""
    user_providers, custom_providers = _gateway_shaped_views(dict(_LB_PROVIDERS))

    with patch("hermes_cli.model_switch.resolve_alias", return_value=None), \
         patch("hermes_cli.model_switch.list_provider_models", return_value=[]), \
         patch("hermes_cli.model_switch.normalize_model_for_provider", side_effect=lambda model, provider: model), \
         patch("hermes_cli.models.validate_requested_model", return_value=_ACCEPTED), \
         patch("hermes_cli.models.detect_provider_for_model", return_value=None), \
         patch("hermes_cli.model_switch.get_model_info", return_value=None), \
         patch("hermes_cli.model_switch.get_model_capabilities", return_value=None), \
         patch(
             "hermes_cli.runtime_provider.resolve_runtime_provider",
             # Empty runtime answer -> switch_model must fall back to the
             # user-provider entry's own base_url and api_key.
             return_value={"api_key": "", "base_url": "", "api_mode": ""},
         ):
        result = switch_model(
            raw_input="claude-fable-5",
            current_provider="openai-codex",
            current_model="gpt-5.4",
            current_base_url="",
            user_providers=user_providers,
            custom_providers=custom_providers,
        )
    assert result.success is True, result.error_message
    assert result.target_provider == "relay-lb"
    assert result.base_url == "http://relay-east.internal:8080"
    assert result.api_key == "lb-key"


def test_legacy_view_without_provider_key_dedupes_by_name_and_url():
    """A legacy custom row lacking ``provider_key`` but whose display name AND
    endpoint equal a ``providers`` entry IS that entry — still one match."""
    custom_providers = [
        {
            # Hand-rolled legacy view: no provider_key stamp.
            "name": "Relay LB East",
            "base_url": "http://relay-east.internal:8080/",  # trailing slash on purpose
            "models": {"claude-fable-5": {}},
        }
    ]
    result = _run_switch(
        raw_input="claude-fable-5",
        current_provider="openai-codex",
        current_model="gpt-5.4",
        user_providers=dict(_LB_PROVIDERS),
        custom_providers=custom_providers,
    )
    assert result.success is True, result.error_message
    assert result.target_provider == "relay-lb"


def test_genuine_ambiguity_across_user_and_custom_still_errors():
    """Two DIFFERENT endpoints declaring the same model must still trip the
    guard: dedupe only collapses views of the SAME physical entry."""
    custom_providers = [
        {
            "name": "Other LB",
            "base_url": "http://relay-west.internal:8080",
            "models": {"claude-fable-5": {}},
        }
    ]
    result = _run_switch(
        raw_input="claude-fable-5",
        current_provider="openai-codex",
        current_model="gpt-5.4",
        user_providers=dict(_LB_PROVIDERS),
        custom_providers=custom_providers,
    )
    assert result.success is False
    assert "--provider" in result.error_message
    assert "relay-lb" in result.error_message
    assert "custom:Other LB" in result.error_message


def test_same_display_name_different_endpoint_is_still_ambiguous():
    """A custom entry that merely SHARES a display name with a ``providers``
    entry but points at a different endpoint is NOT the same provider — the
    guard must still fire rather than silently picking one."""
    custom_providers = [
        {
            "name": "Relay LB East",  # name collision, different box
            "base_url": "http://relay-other.internal:8080",
            "models": {"claude-fable-5": {}},
        }
    ]
    result = _run_switch(
        raw_input="claude-fable-5",
        current_provider="openai-codex",
        current_model="gpt-5.4",
        user_providers=dict(_LB_PROVIDERS),
        custom_providers=custom_providers,
    )
    assert result.success is False
    assert "--provider" in result.error_message
    assert "relay-lb" in result.error_message
    assert "custom:Relay LB East" in result.error_message


def test_same_display_name_with_missing_endpoint_url_is_not_collapsed():
    """A custom entry that shares a display name with a ``providers`` entry but
    declares NO endpoint URL has incomplete identity — name equality alone must
    NOT collapse it onto the ``providers`` entry, so the guard still fires."""
    custom_providers = [
        {
            "name": "Relay LB East",  # name collision, endpoint unknown
            "models": {"claude-fable-5": {}},
        }
    ]
    result = _run_switch(
        raw_input="claude-fable-5",
        current_provider="openai-codex",
        current_model="gpt-5.4",
        user_providers=dict(_LB_PROVIDERS),
        custom_providers=custom_providers,
    )
    assert result.success is False
    assert "--provider" in result.error_message
    assert "relay-lb" in result.error_message
    assert "custom:Relay LB East" in result.error_message
