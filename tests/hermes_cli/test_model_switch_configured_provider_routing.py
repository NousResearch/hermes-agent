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
# #62240: step-e static-catalog reroute must not switch onto an unconfigured
# provider.  ``detect_provider_for_model`` matches Hermes' built-in
# CURATED_MODELS, so a bare name that also lives there (e.g. ``gpt-5.5`` under
# ``openai-api``) would silently reroute the session to that vendor even when
# the user only configured a different endpoint for it — leaving the previous
# provider's base_url in place and 401ing.  The reroute is now gated on the
# detected provider actually having credentials.
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
         patch("hermes_cli.models.validate_requested_model", return_value=_ACCEPTED), \
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
