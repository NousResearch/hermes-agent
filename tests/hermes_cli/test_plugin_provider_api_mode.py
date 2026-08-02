"""Regression tests for #53054: a plugin ``ProviderProfile``'s declared
``api_mode`` must be honored by every API-key runtime-resolution path.

``providers/README.md`` documents the declaration as a resolution input, but
nothing read it: ``get_provider()`` resolves against models.dev +
``HERMES_OVERLAYS`` only, and the ``ProviderProfile → ProviderConfig`` bridge
in ``hermes_cli/auth.py`` drops the field. So the runtime derived the mode
from host/URL detection alone, and any ``anthropic_messages`` plugin whose
endpoint is not URL-self-describing (no ``/anthropic`` suffix — e.g.
Volcengine Ark's ``…/api/coding``) silently degraded to ``chat_completions``
and 404'd on every request.

``_fallback_api_mode`` (33a2f29a63) closed the same latent class for in-tree
overlays by routing all three resolver paths through
``providers.determine_api_mode``; plugin profiles are still invisible to it,
because ``determine_api_mode`` never consulted the plugin registry.

These tests run the REAL chain — plugin file on disk → provider discovery
import → ``register_provider()`` → ``auth._extend_registry_from_provider_plugins()``
→ ``resolve_runtime_provider()`` — rather than hand-mirroring the bridge, so
a regression in any link fails them. All three API-key resolver paths are
covered: pooled, explicit API-key, and the ordinary no-pool route, plus the
precedence guards (persisted config, host detection, base_url override) and
the in-tree overlay lane that must stay untouched.

HERMES_HOME is sandboxed to a per-session tempdir by tests/conftest.py, so
the plugin below is never written into a real Hermes install.
"""

from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest


PLUGIN_NAME = "testgw"
PLUGIN_ALIAS = "testgw-alias"
PLUGIN_ENV_VAR = "TESTGW_API_KEY"
# Anthropic-Messages endpoint that is NOT URL-self-describing: no /anthropic
# suffix, host isn't api.kimi.com — same shape as Ark's
# https://ark.cn-beijing.volces.com/api/coding.
PLUGIN_BASE_URL = "https://gw.example.com/api/coding"

PLUGIN_INIT = f'''
from providers import register_provider
from providers.base import ProviderProfile

register_provider(ProviderProfile(
    name="{PLUGIN_NAME}",
    aliases=("{PLUGIN_ALIAS}",),
    api_mode="anthropic_messages",
    env_vars=("{PLUGIN_ENV_VAR}",),
    base_url="{PLUGIN_BASE_URL}",
    auth_type="api_key",
))
'''

PLUGIN_YAML = f"""name: {PLUGIN_NAME}
kind: model-provider
version: 0.0.1
description: Non-self-describing anthropic_messages test provider (#53054)
"""


def _clear_provider_caches() -> None:
    """Force providers/__init__.py to re-discover on next lookup."""
    import providers as _pkg

    _pkg._REGISTRY.clear()
    _pkg._ALIASES.clear()
    _pkg._discovered = False
    for mod in list(sys.modules.keys()):
        if mod.startswith("plugins.model_providers") or mod.startswith(
            "_hermes_user_provider"
        ):
            del sys.modules[mod]


@pytest.fixture()
def registered_plugin_provider():
    """Install a user model-provider plugin and run the production discovery
    + auth-registration chain against it.

    Yields the bridged ``ProviderConfig``. Cleans the shared registries
    afterwards so other tests never see the fake provider.
    """
    from hermes_constants import get_hermes_home
    from hermes_cli import auth as auth_mod

    plugin_dir = get_hermes_home() / "plugins" / "model-providers" / PLUGIN_NAME
    plugin_dir.mkdir(parents=True, exist_ok=True)
    (plugin_dir / "__init__.py").write_text(PLUGIN_INIT)
    (plugin_dir / "plugin.yaml").write_text(PLUGIN_YAML)

    _clear_provider_caches()
    try:
        # The real registration loop: discovery runs lazily inside
        # list_providers(), then the profile is bridged into
        # PROVIDER_REGISTRY exactly as at import time.
        auth_mod._extend_registry_from_provider_plugins()
        pconfig = auth_mod.PROVIDER_REGISTRY.get(PLUGIN_NAME)
        assert pconfig is not None, (
            "plugin provider was not bridged into PROVIDER_REGISTRY"
        )
        yield pconfig
    finally:
        auth_mod.PROVIDER_REGISTRY.pop(PLUGIN_NAME, None)
        auth_mod.PROVIDER_REGISTRY.pop(PLUGIN_ALIAS, None)
        _clear_provider_caches()


def test_declared_api_mode_reaches_determine_api_mode(registered_plugin_provider):
    """The declaration is visible to the resolver entry point, by name and
    by alias — ``get_provider()`` alone never sees a plugin profile."""
    from hermes_cli.providers import determine_api_mode, get_provider

    assert get_provider(PLUGIN_NAME, allow_network=False) is None
    assert determine_api_mode(PLUGIN_NAME, PLUGIN_BASE_URL) == "anthropic_messages"
    assert determine_api_mode(PLUGIN_ALIAS, PLUGIN_BASE_URL) == "anthropic_messages"


def test_no_pool_route_honors_declared_api_mode(
    registered_plugin_provider, monkeypatch
):
    """Ordinary no-pool API-key route: discovery → registry → resolve."""
    from hermes_cli.runtime_provider import resolve_runtime_provider

    monkeypatch.setenv(PLUGIN_ENV_VAR, "sk-testgw-0123456789")
    runtime = resolve_runtime_provider(requested=PLUGIN_NAME)

    assert runtime["provider"] == PLUGIN_NAME
    assert runtime["api_mode"] == "anthropic_messages"
    assert runtime["base_url"] == PLUGIN_BASE_URL
    assert runtime["api_key"] == "sk-testgw-0123456789"


def test_explicit_api_key_route_honors_declared_api_mode(
    registered_plugin_provider,
):
    """Explicit API-key route (``_resolve_explicit_runtime``)."""
    from hermes_cli.runtime_provider import resolve_runtime_provider

    runtime = resolve_runtime_provider(
        requested=PLUGIN_NAME, explicit_api_key="sk-explicit-0123456789"
    )

    assert runtime["source"] == "explicit"
    assert runtime["api_mode"] == "anthropic_messages"
    assert runtime["base_url"] == PLUGIN_BASE_URL


def test_pooled_route_honors_declared_api_mode(registered_plugin_provider):
    """Pooled-credential route (``_resolve_runtime_from_pool_entry``)."""
    from hermes_cli.runtime_provider import _resolve_runtime_from_pool_entry

    entry = SimpleNamespace(
        runtime_base_url=PLUGIN_BASE_URL,
        base_url=PLUGIN_BASE_URL,
        runtime_api_key="sk-pooled-0123456789",
        access_token="sk-pooled-0123456789",
    )
    runtime = _resolve_runtime_from_pool_entry(
        provider=PLUGIN_NAME,
        entry=entry,
        requested_provider=PLUGIN_NAME,
        model_cfg={},
    )

    assert runtime["api_mode"] == "anthropic_messages"


def test_persisted_config_api_mode_still_wins(registered_plugin_provider):
    """Config precedence is unchanged: an explicit persisted api_mode for the
    same provider beats the profile-declared fallback."""
    from hermes_cli.runtime_provider import _resolve_runtime_from_pool_entry

    entry = SimpleNamespace(
        runtime_base_url=PLUGIN_BASE_URL,
        base_url=PLUGIN_BASE_URL,
        runtime_api_key="sk-pooled-0123456789",
        access_token="sk-pooled-0123456789",
    )
    runtime = _resolve_runtime_from_pool_entry(
        provider=PLUGIN_NAME,
        entry=entry,
        requested_provider=PLUGIN_NAME,
        model_cfg={"provider": PLUGIN_NAME, "api_mode": "chat_completions"},
    )

    assert runtime["api_mode"] == "chat_completions"


def test_host_detection_still_wins_over_declared_profile(
    registered_plugin_provider,
):
    """Host precedence is unchanged: a host that mandates a wire protocol
    beats the profile-declared fallback."""
    from hermes_cli.runtime_provider import _fallback_api_mode

    # testgw declares anthropic_messages, but a direct api.openai.com URL
    # must still resolve to codex_responses.
    assert (
        _fallback_api_mode(PLUGIN_NAME, "https://api.openai.com/v1")
        == "codex_responses"
    )


def test_base_url_override_defers_to_default(registered_plugin_provider):
    """A base_url pointing somewhere other than the profile's own endpoint is
    the stronger signal — the declared mode must not be forced onto it."""
    from hermes_cli.runtime_provider import _fallback_api_mode

    assert (
        _fallback_api_mode(PLUGIN_NAME, "https://gw.example.com/v1")
        == "chat_completions"
    )
    # Trailing-slash differences are not an override.
    assert (
        _fallback_api_mode(PLUGIN_NAME, PLUGIN_BASE_URL + "/")
        == "anthropic_messages"
    )


def test_in_tree_overlays_are_unaffected(registered_plugin_provider):
    """The overlay lane keeps precedence and its existing answers."""
    from hermes_cli.providers import determine_api_mode

    assert determine_api_mode("openai-api", "https://api.openai.com/v1") == (
        "codex_responses"
    )
    assert determine_api_mode("openrouter", "https://openrouter.ai/api/v1") == (
        "chat_completions"
    )
    assert determine_api_mode("nous", "", "anthropic/claude-x") == (
        "anthropic_messages"
    )


def test_unknown_provider_keeps_chat_completions_default():
    """Providers with no profile and no overlay keep the existing default."""
    from hermes_cli.runtime_provider import _fallback_api_mode

    assert (
        _fallback_api_mode("no-such-provider", "https://gw.example.com/v1")
        == "chat_completions"
    )
