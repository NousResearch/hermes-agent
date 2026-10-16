"""Regression tests for #132117: a bare-name ``/model <name>`` from an
aggregator session must not be hijacked to the aggregator when another
CONFIGURED provider declares that exact model id.

Repro: llama.cpp ``llama-server`` registered under ``providers.`` declares
``qwen3.8-flash``; the profile's main provider is OpenRouter. Typing
``/model qwen3.8-flash`` resolved through the aggregator catalog's bare-part
match (``qwen/qwen3.8-flash``) and kept the session on OpenRouter — a metered,
billed provider — before step d.5 (``_route_configured_provider``, #45006)
could ever run, silently billing turns the user believed were local.

The fix consults ``_configured_provider_matches`` before the step-d catalog
match: a bare name declared by a configured provider other than the current
one falls through to d.5 and routes to its declarer. The catalog match keeps
owning names nobody declares (flat-namespace resellers, #opencode-go), and an
explicit vendor-prefixed slug still resolves through the catalog.

Hermetic: the model-resolution chain is fully mocked (no network), mirroring
``tests/hermes_cli/test_model_switch_configured_provider_routing.py``.
"""

from unittest.mock import patch

from hermes_cli.model_switch import switch_model

_ACCEPTED = {"accepted": True, "persist": True, "recognized": True, "message": None}

# Live OpenRouter-style catalog: vendor-prefixed slugs whose bare parts collide
# with the locally declared model id.
_OPENROUTER_LIVE = [
    "qwen/qwen3.8-flash",
    "deepseek/deepseek-v4.1-flash",
    "meta/llama-5-8b-instruct",
]


def _run_switch(
    *,
    raw_input,
    current_provider,
    user_providers=None,
    custom_providers=None,
    catalog=None,
    current_model="deepseek/deepseek-v4.1-flash",
):
    """Drive ``switch_model`` with the resolution chain mocked out.

    ``catalog`` feeds ``list_provider_models`` so step d sees a live aggregator
    listing without network access. Everything else that could hit
    catalogs/network is patched, isolating the step d vs d.5 ordering."""
    with (
        patch("hermes_cli.model_switch.resolve_alias", return_value=None),
        patch(
            "hermes_cli.model_switch.list_provider_models",
            side_effect=lambda provider: (
                list(catalog or []) if provider == "openrouter" else []
            ),
        ),
        patch(
            "hermes_cli.model_switch.normalize_model_for_provider",
            side_effect=lambda model, provider: model,
        ),
        patch(
            "hermes_cli.models_validate.validate_requested_model",
            return_value=_ACCEPTED,
        ),
        patch("hermes_cli.models.detect_provider_for_model", return_value=None),
        patch("hermes_cli.model_switch.get_model_info", return_value=None),
        patch("hermes_cli.model_switch.get_model_capabilities", return_value=None),
        patch(
            "hermes_cli.runtime_provider.resolve_runtime_provider",
            return_value={
                "api_key": "***",
                "base_url": "http://resolved/v1",
                "api_mode": "",
            },
        ),
    ):
        return switch_model(
            raw_input=raw_input,
            current_provider=current_provider,
            current_model=current_model,
            current_base_url="https://openrouter.ai/api/v1"
            if current_provider == "openrouter"
            else "http://127.0.0.1:8080/v1",
            current_api_key="dummy",
            user_providers=user_providers or {},
            custom_providers=custom_providers or [],
        )


def _llama_server_providers(extra=None):
    """The #132117 repro config: a local llama.cpp server declaring the bare id."""
    providers = {
        "llama-server": {
            "name": "llama-server",
            "api": "http://127.0.0.1:8080/v1",
            "transport": "chat_completions",
            "default_model": "qwen3.8-flash",
            "models": {"qwen3.8-flash": {"context_length": 262144}},
        }
    }
    providers.update(extra or {})
    return providers


def test_bare_name_declared_by_configured_provider_routes_there():
    """The #132117 repro: from an OpenRouter session, the bare id that the local
    provider declares must route to it, not be swallowed by the catalog's
    bare-part match onto the metered aggregator."""
    result = _run_switch(
        raw_input="qwen3.8-flash",
        current_provider="openrouter",
        user_providers=_llama_server_providers(),
        catalog=_OPENROUTER_LIVE,
    )
    assert result.success is True, result.error_message
    assert result.target_provider == "llama-server"
    assert result.new_model == "qwen3.8-flash"
    assert result.provider_changed is True


def test_bare_name_without_configured_owner_keeps_catalog_match():
    """Step d's own purpose stays intact: a bare id nobody declares still
    resolves through the aggregator catalog and stays on the aggregator."""
    result = _run_switch(
        raw_input="llama-5-8b-instruct",
        current_provider="openrouter",
        user_providers=_llama_server_providers(),
        catalog=_OPENROUTER_LIVE,
    )
    assert result.success is True, result.error_message
    assert result.target_provider == "openrouter"
    assert result.new_model == "meta/llama-5-8b-instruct"
    assert result.provider_changed is False


def test_explicit_vendor_slug_still_resolves_through_catalog():
    """An explicit vendor-prefixed slug expresses aggregator intent: it is not a
    bare name, so the configured-provider declaration of the bare part does not
    claim it."""
    result = _run_switch(
        raw_input="qwen/qwen3.8-flash",
        current_provider="openrouter",
        user_providers=_llama_server_providers(),
        catalog=_OPENROUTER_LIVE,
    )
    assert result.success is True, result.error_message
    assert result.target_provider == "openrouter"
    assert result.new_model == "qwen/qwen3.8-flash"


def test_session_on_declaring_provider_unchanged():
    """Starting on the declaring local provider keeps resolving locally — the
    pre-fix behavior that already worked must not regress."""
    result = _run_switch(
        raw_input="qwen3.8-flash",
        current_provider="llama-server",
        user_providers=_llama_server_providers(),
        catalog=_OPENROUTER_LIVE,
        current_model="qwen3.8-flash",
    )
    assert result.success is True, result.error_message
    assert result.target_provider == "llama-server"
    assert result.new_model == "qwen3.8-flash"


def test_two_declarers_fail_closed_with_provider_hint():
    """Two configured providers declaring the same bare id must fail with the
    ``--provider`` disambiguation hint instead of silently picking the
    aggregator (d.5's existing multi-match failure)."""
    result = _run_switch(
        raw_input="qwen3.8-flash",
        current_provider="openrouter",
        user_providers=_llama_server_providers(
            extra={
                "other-server": {
                    "name": "other-server",
                    "api": "http://127.0.0.1:9090/v1",
                    "transport": "chat_completions",
                    "models": {"qwen3.8-flash": {"context_length": 131072}},
                },
            }
        ),
        catalog=_OPENROUTER_LIVE,
    )
    assert result.success is False
    assert "multiple configured providers" in result.error_message
