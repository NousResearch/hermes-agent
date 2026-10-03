"""get_default_model_for_provider() must never silently escalate a native provider's
no-model-selected fallback to its priciest curated entry (#64635 covered aggregator
providers via PREFERRED_SILENT_DEFAULT_MODEL; native providers like anthropic order
their own curated list most-capable-first too, and were left resolving to entry [0]
— observed hitting an account-level monthly spend cap on the flagship)."""

from __future__ import annotations

from hermes_cli.models import get_default_model_for_provider
from hermes_cli.models_catalog_static import (
    _NATIVE_PROVIDER_SILENT_DEFAULT_OVERRIDES,
    _PROVIDER_MODELS,
)


def test_anthropic_silent_default_is_not_the_flagship():
    """The silent default must never be the most-capable (priciest) curated entry."""
    default = get_default_model_for_provider("anthropic")
    flagship = _PROVIDER_MODELS["anthropic"][0]
    assert default != flagship
    assert "fable" not in default.lower()


def test_anthropic_silent_default_is_a_real_catalog_entry():
    """The override must resolve to a model actually present in the curated list —
    a stale/typo'd override would silently fall through to models[0] anyway."""
    default = get_default_model_for_provider("anthropic")
    assert default in _PROVIDER_MODELS["anthropic"]


def test_every_native_override_is_a_real_catalog_entry():
    """Relationship check, not a snapshot: every configured native override must exist
    in that provider's own curated list, whatever the list's current contents are."""
    for provider, override in _NATIVE_PROVIDER_SILENT_DEFAULT_OVERRIDES.items():
        assert override in _PROVIDER_MODELS.get(provider, []), (
            f"native silent-default override {override!r} for {provider!r} "
            "is not in that provider's curated model list"
        )


def test_unaffected_providers_still_use_entry_zero():
    """Providers without a native override (and not in _SILENT_DEFAULT_PROVIDERS) keep
    their previous entry-[0] behavior — this fix must not widen beyond its target."""
    from hermes_cli.models_catalog_static import _SILENT_DEFAULT_PROVIDERS

    for provider, models in _PROVIDER_MODELS.items():
        if provider in _NATIVE_PROVIDER_SILENT_DEFAULT_OVERRIDES or provider in _SILENT_DEFAULT_PROVIDERS or not models:
            continue
        assert get_default_model_for_provider(provider) == models[0]
