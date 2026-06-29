"""Cerebras provider profile wiring.

Asserts the bundled ``plugins/model-providers/cerebras/`` profile registers
with the expected OpenAI-compatible endpoint, API-key env var, and offline
catalog. Registration itself is covered generically by test_plugin_discovery;
this pins the Cerebras-specific values so a typo in the endpoint or env var
fails CI rather than silently routing to the wrong place.
"""

from __future__ import annotations

import sys


def _rediscover() -> None:
    """Force providers/__init__.py to re-scan on the next lookup."""
    import providers as _pkg

    _pkg._REGISTRY.clear()
    _pkg._ALIASES.clear()
    _pkg._discovered = False
    for mod in list(sys.modules.keys()):
        if mod.startswith("plugins.model_providers") or mod.startswith(
            "_hermes_user_provider"
        ):
            del sys.modules[mod]


def test_cerebras_profile_registered() -> None:
    _rediscover()
    from providers import get_provider_profile

    p = get_provider_profile("cerebras")
    assert p is not None, "cerebras profile not discovered"
    assert p.name == "cerebras"
    assert p.base_url == "https://api.cerebras.ai/v1"
    assert p.api_mode == "chat_completions"
    assert p.auth_type == "api_key"
    assert "CEREBRAS_API_KEY" in p.env_vars
    # Offline catalog mirrors models.dev's Cerebras entry; the live
    # /v1/models probe (per-account entitlement) overrides it at runtime.
    assert "gpt-oss-120b" in p.fallback_models
    assert p.default_aux_model in p.fallback_models

    _rediscover()
