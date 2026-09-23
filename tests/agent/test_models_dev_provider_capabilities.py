"""Plugin-declared per-model capabilities (#102115): one declaration on ``ProviderProfile`` feeds every
models.dev consumer — capability lookup, context lookup, the picker's reasoning badge and image routing."""

from copy import deepcopy

import providers
from providers.base import ProviderProfile
from agent import models_dev


def _isolated_registry(monkeypatch, catalog=None):
    monkeypatch.setattr(providers, "_REGISTRY", {})
    monkeypatch.setattr(providers, "_ALIASES", {})
    monkeypatch.setattr(providers, "_discovered", True)
    monkeypatch.setattr(models_dev, "_registry_models", lambda *a, **k: catalog)


def test_declared_capabilities_reach_every_consumer_and_user_override_wins(monkeypatch):
    from agent.image_routing import decide_image_input_mode
    from hermes_cli.inventory import _apply_capabilities

    _isolated_registry(monkeypatch)
    overrides: dict = {}
    monkeypatch.setattr(models_dev, "_load_model_overrides", lambda: overrides)
    monkeypatch.setattr("hermes_cli.inventory._reasoning_catalog_reader", lambda slug: None)
    declaration = {
        "tier-high": {"supports_reasoning": False, "supports_vision": True, "context_window": 64000},
    }
    original = deepcopy(declaration)
    providers.register_provider(ProviderProfile(
        name="fixture-provider", aliases=("fixture-alias",), model_capabilities=declaration))

    for name in ("fixture-provider", "fixture-alias"):
        caps = models_dev.get_model_capabilities(name, "tier-high")
        assert (caps.supports_reasoning, caps.supports_vision, caps.context_window) == (False, True, 64000)
        assert models_dev.lookup_models_dev_context(name, "tier-high") == 64000
        # Negative: an undeclared model keeps the catalog/heuristic path (catalog miss → None).
        assert models_dev.get_model_capabilities(name, "undeclared") is None
        assert models_dev.get_model_capabilities(name, "TIER-HIGH").context_window == caps.context_window

    cfg = {"model": {"provider": "fixture-provider", "default": "tier-high"}}
    assert decide_image_input_mode("fixture-provider", "tier-high", cfg) == "native"
    assert decide_image_input_mode("fixture-provider", "undeclared", cfg) == "text"

    rows = [{"slug": "fixture-provider", "models": ["tier-high", "undeclared"]}]
    _apply_capabilities(rows)
    assert rows[0]["capabilities"]["tier-high"]["reasoning"] is False
    assert rows[0]["capabilities"]["undeclared"]["reasoning"] is True

    overrides["fixture-provider"] = {"tier-high": {"supports_reasoning": True, "context_window": 96000}}
    caps = models_dev.get_model_capabilities("fixture-provider", "tier-high")
    assert (caps.supports_reasoning, caps.supports_vision, caps.context_window) == (True, True, 96000)
    assert models_dev.lookup_models_dev_context("fixture-provider", "tier-high") == 96000
    assert declaration == original


def test_partial_plugin_metadata_preserves_unknowns_and_catalog_fields(monkeypatch):
    catalog = {"known": {"reasoning": True, "tool_call": True,
                         "limit": {"context": 32000, "output": 4000},
                         "modalities": {"input": ["text", "image"]}}}
    original = deepcopy(catalog)
    _isolated_registry(monkeypatch, catalog)
    monkeypatch.setitem(models_dev.PROVIDER_TO_MODELS_DEV, "fixture-provider", "fixture-provider")
    monkeypatch.setattr(models_dev, "_load_model_overrides", lambda: {})
    providers.register_provider(ProviderProfile(name="fixture-provider", model_capabilities={
        "known": {"context_window": 64000},
        "unknown": {"context_window": 48000},
    }))
    known = models_dev.get_model_capabilities("fixture-provider", "known")
    assert (known.context_window, known.max_output_tokens) == (64000, 4000)
    assert known.supports_reasoning is True and known.supports_vision is True
    unknown = models_dev.get_model_capabilities("fixture-provider", "unknown")
    assert unknown.context_window == 48000
    assert unknown.supports_reasoning is None and unknown.supports_vision is None
    assert catalog == original

    # An auth route can share its canonical provider's declaration, but its own
    # registered profile remains authoritative, including deliberate omissions.
    monkeypatch.setitem(models_dev.PROVIDER_TO_MODELS_DEV, "fixture-auth", "fixture-provider")
    assert models_dev.lookup_models_dev_context("fixture-auth", "known") == known.context_window
    providers.register_provider(ProviderProfile(name="fixture-auth", model_capabilities={
        "known": {"context_window": 80000},
    }))
    assert models_dev.lookup_models_dev_context("fixture-auth", "known") == 80000
    assert models_dev.get_model_capabilities("fixture-auth", "unknown") is None
