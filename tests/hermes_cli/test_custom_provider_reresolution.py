"""A bare ``custom`` re-resolution must not clobber a named custom endpoint's runtime.

Every named custom endpoint resolves to the literal provider ``"custom"`` in a runtime dict: the
entry name is lost (the api_key is deliberately never persisted). The api_server session/provider
refresh paths hand that collapsed value back to ``resolve_runtime_provider`` — so bare ``custom``
is a real runtime input, not only a user-typed provider.

Before the heal, that re-resolution found no named entry for bare ``custom`` and fell through to
the OpenRouter/bare-custom fallback, overwriting a working local ``base_url`` with
``https://openrouter.ai/api/v1`` and dropping the api_key. The turn then failed with "No LLM
provider configured".

The contract: re-resolving a collapsed custom keeps the configured endpoint, and a legacy
``provider: custom`` config with only ``model.base_url`` keeps its bare name and the legacy trust
path (no configured entry is reachable, so nothing is healed and behavior is unchanged).
"""

from __future__ import annotations

import pytest

from hermes_cli import runtime_provider as rp

LOCAL_BASE_URL = "http://10.9.9.9:4000/v1"
LOCAL_KEY = "sk-local-endpoint-key"
LEGACY_BASE_URL = "http://10.9.9.9:5000/v1"
LEGACY_KEY = "sk-legacy-key"


@pytest.fixture
def named_custom_config(monkeypatch):
    """A named ``custom_providers`` entry, reached only via its ``custom:<name>`` identity."""
    config = {
        "model": {"provider": "custom:local", "default": "local-model-1"},
        "custom_providers": [
            {
                "name": "local",
                "base_url": LOCAL_BASE_URL,
                "api_key": LOCAL_KEY,
                "models": {"local-model-1": {"context_length": 262144, "max_tokens": 16384}},
            }
        ],
    }
    monkeypatch.setattr(rp, "load_config", lambda *a, **k: config)
    monkeypatch.setattr("hermes_cli.config.load_config", lambda *a, **k: config)
    monkeypatch.setattr(rp, "_get_model_config", lambda: config["model"])
    return config


@pytest.fixture
def legacy_custom_config(monkeypatch):
    """Legacy shape: bare ``provider: custom`` with the endpoint only on ``model.base_url``.

    No ``custom_providers`` / ``providers`` entry exists, so there is no ``custom:<name>`` identity
    to heal to and the bare name must survive.
    """
    config = {
        "model": {
            "provider": "custom",
            "default": "legacy-model-1",
            "base_url": LEGACY_BASE_URL,
            "api_key": LEGACY_KEY,
        },
    }
    monkeypatch.setattr(rp, "load_config", lambda *a, **k: config)
    monkeypatch.setattr("hermes_cli.config.load_config", lambda *a, **k: config)
    monkeypatch.setattr(rp, "_get_model_config", lambda: config["model"])
    return config


def test_collapsed_custom_reresolution_keeps_the_named_endpoint(named_custom_config):
    """The reported bug: re-resolving the collapsed ``custom`` must keep base_url AND the key."""
    from hermes_cli.runtime_provider import resolve_runtime_provider

    first = resolve_runtime_provider(requested="custom:local")
    collapsed = first["provider"]
    assert collapsed == "custom", "precondition: a named custom's runtime provider is the literal 'custom'"

    again = resolve_runtime_provider(requested=collapsed)

    assert again["base_url"] == LOCAL_BASE_URL, "re-resolution must not fall through to OpenRouter"
    assert again["api_key"] == LOCAL_KEY, "re-resolution must not drop the configured api_key"


def test_collapsed_custom_reresolution_is_stable_across_repeats(named_custom_config):
    """The heal must be idempotent — the api_server path re-resolves on every turn."""
    from hermes_cli.runtime_provider import resolve_runtime_provider

    provider = resolve_runtime_provider()["provider"]
    for _ in range(3):
        runtime = resolve_runtime_provider(requested=provider)
        assert runtime["base_url"] == LOCAL_BASE_URL
        assert runtime["api_key"] == LOCAL_KEY


def test_legacy_bare_custom_config_is_not_healed(legacy_custom_config):
    """A legacy ``provider: custom`` + ``model.base_url`` keeps its bare name and its endpoint.

    Nothing is configured that could claim the bare name, so healing must be a no-op rather than
    inventing an identity the config cannot honor.
    """
    from hermes_cli.runtime_provider import resolve_runtime_provider

    assert rp._heal_collapsed_custom("custom") == "custom"

    runtime = resolve_runtime_provider()
    assert runtime["base_url"] == LEGACY_BASE_URL
    assert runtime["api_key"] == LEGACY_KEY


def test_heal_leaves_non_custom_providers_untouched(named_custom_config):
    """Only bare ``custom`` is ambiguous; every other provider name passes through unchanged."""
    for provider in ("custom:local", "openrouter", "anthropic", "auto", ""):
        assert rp._heal_collapsed_custom(provider) == provider
