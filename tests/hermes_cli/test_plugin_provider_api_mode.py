"""Regression tests for plugin ``ProviderProfile.api_mode`` resolution.

These tests run the real chain: plugin file on disk -> provider discovery ->
auth registry extension -> runtime resolution. They cover every API-key route
and the precedence rules that must remain unchanged.
"""

from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest


PLUGIN_NAME = "testgw"
PLUGIN_ALIAS = "testgw-alias"
PLUGIN_ENV_VAR = "TESTGW_API_KEY"
# Anthropic Messages endpoint that is not URL-self-describing, matching the
# shape of Ark's https://ark.cn-beijing.volces.com/api/coding endpoint.
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
    """Force the provider package to rediscover plugins on next lookup."""
    import providers as provider_package

    provider_package._REGISTRY.clear()
    provider_package._ALIASES.clear()
    provider_package._PROVIDER_LIST_CACHE = None
    provider_package._discovered = False
    for module_name in list(sys.modules):
        if module_name.startswith("plugins.model_providers") or module_name.startswith(
            "_hermes_user_provider"
        ):
            del sys.modules[module_name]


@pytest.fixture(scope="module")
def registered_plugin_provider():
    """Install a provider before auth import, then run the production chain."""
    from hermes_constants import get_hermes_home

    plugin_dir = get_hermes_home() / "plugins" / "model-providers" / PLUGIN_NAME
    plugin_dir.mkdir(parents=True, exist_ok=True)
    (plugin_dir / "__init__.py").write_text(PLUGIN_INIT, encoding="utf-8")
    (plugin_dir / "plugin.yaml").write_text(PLUGIN_YAML, encoding="utf-8")

    _clear_provider_caches()
    auth_module = None
    try:
        # The test file runs in its own process. Creating the plugin before
        # auth's first import exercises its real one-shot registration block,
        # without mirroring the bridge or adding a test-only production hook.
        assert "hermes_cli.auth" not in sys.modules
        from hermes_cli import auth as auth_module

        provider_config = auth_module.PROVIDER_REGISTRY.get(PLUGIN_NAME)
        assert provider_config is not None, (
            "plugin provider was not bridged into PROVIDER_REGISTRY"
        )
        yield provider_config
    finally:
        if auth_module is not None:
            auth_module.PROVIDER_REGISTRY.pop(PLUGIN_NAME, None)
            auth_module.PROVIDER_REGISTRY.pop(PLUGIN_ALIAS, None)
        _clear_provider_caches()


def test_declared_api_mode_reaches_determine_api_mode(registered_plugin_provider):
    """The resolver sees the declaration by canonical name and alias."""
    from hermes_cli.providers import determine_api_mode, get_provider

    provider = get_provider(PLUGIN_NAME, allow_network=False)
    assert provider is not None
    assert provider.source == "plugin-profile"
    assert provider.transport == "anthropic_messages"
    assert determine_api_mode(PLUGIN_NAME, PLUGIN_BASE_URL) == "anthropic_messages"
    assert determine_api_mode(PLUGIN_ALIAS, PLUGIN_BASE_URL) == "anthropic_messages"


def test_no_pool_route_honors_declared_api_mode(
    registered_plugin_provider, monkeypatch
):
    """Ordinary no-pool API-key route honors the plugin declaration."""
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
    """Explicit API-key route honors the plugin declaration."""
    from hermes_cli.runtime_provider import resolve_runtime_provider

    runtime = resolve_runtime_provider(
        requested=PLUGIN_NAME, explicit_api_key="sk-explicit-0123456789"
    )

    assert runtime["source"] == "explicit"
    assert runtime["api_mode"] == "anthropic_messages"
    assert runtime["base_url"] == PLUGIN_BASE_URL


def test_pooled_route_honors_declared_api_mode(registered_plugin_provider):
    """Pooled-credential route honors the plugin declaration."""
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
    """An explicit persisted mode for the same provider keeps precedence."""
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
    """A host-mandated wire protocol beats the profile fallback."""
    from hermes_cli.runtime_provider import _fallback_api_mode

    assert (
        _fallback_api_mode(PLUGIN_NAME, "https://api.openai.com/v1")
        == "codex_responses"
    )


def test_base_url_override_defers_to_default(registered_plugin_provider):
    """A different base URL must not inherit the profile's declared mode."""
    from hermes_cli.runtime_provider import _fallback_api_mode

    assert (
        _fallback_api_mode(PLUGIN_NAME, "https://gw.example.com/v1")
        == "chat_completions"
    )
    assert (
        _fallback_api_mode(PLUGIN_NAME, PLUGIN_BASE_URL + "/")
        == "anthropic_messages"
    )


def test_in_tree_overlays_are_unaffected(registered_plugin_provider):
    """The overlay lane retains precedence and its existing answers."""
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
    """Unknown providers retain the conservative existing default."""
    from hermes_cli.runtime_provider import _fallback_api_mode

    assert (
        _fallback_api_mode("no-such-provider", "https://gw.example.com/v1")
        == "chat_completions"
    )
