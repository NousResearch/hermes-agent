"""Regression tests for #152498 — every-startup "Normalized model" noise.

With a built-in vendor provider (``model.provider: nvidia``) and a vendor-prefixed
``model.default`` (`nvidia/nemotron-…`), the startup route split off the prefix as a route
selector, returned the bare slug, and ``_normalize_model_for_provider`` then re-added the
prefix and printed the "Normalized model…" warning on every startup. When the split's
prefix names the ALREADY-pinned provider, the prefix is vendor namespace, not a route
selector: the route keeps the full id.
"""

import pytest

from hermes_cli import model_switch


def test_same_provider_prefix_keeps_the_full_model_id(monkeypatch):
    """`nvidia/<model>` with the config already on nvidia resolves to the full id.

    Any other reading strips the prefix only for the normalizer to re-add it and
    print the warning, which is the reported noise.
    """
    monkeypatch.setattr(model_switch, "DIRECT_ALIASES", {})
    route = model_switch.resolve_startup_model_route(
        "nvidia/nemotron-3-ultra-550b-a55b",
        current_provider="nvidia",
        user_providers={"nvidia": {"base_url": "https://integrate.api.nvidia.com/v1"}},
    )
    assert route == model_switch.StartupModelRoute(
        "nvidia/nemotron-3-ultra-550b-a55b", "nvidia", ""
    )


def test_switch_intent_to_a_configured_provider_still_strips_the_prefix(monkeypatch):
    """From a different pinned provider the prefix IS a route selector: bare id, unchanged."""
    monkeypatch.setattr(model_switch, "DIRECT_ALIASES", {})
    route = model_switch.resolve_startup_model_route(
        "nvidia/nemotron-3-ultra-550b-a55b",
        current_provider="openai",
        user_providers={"nvidia": {"base_url": "https://integrate.api.nvidia.com/v1"}},
    )
    assert route == model_switch.StartupModelRoute(
        "nemotron-3-ultra-550b-a55b", "nvidia", ""
    )


def test_explicit_provider_choice_does_not_get_the_prefixed_id(monkeypatch):
    """`--provider`/`-m` is a real switch: the split's bare-id travel stays (pre-existing None
    here — the alias/qualified branches own that path; the same-provider keep must not
    widen into explicit choices)."""
    monkeypatch.setattr(model_switch, "DIRECT_ALIASES", {})
    route = model_switch.resolve_startup_model_route(
        "nvidia/nemotron-3-ultra-550b-a55b",
        explicit_provider="nvidia",
        current_provider="nvidia",
        user_providers={"nvidia": {}},
    )
    assert route is None or route.model == "nemotron-3-ultra-550b-a55b"
