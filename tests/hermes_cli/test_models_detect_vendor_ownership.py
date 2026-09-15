"""A first-party session never re-routes its own vendor's model when the live catalog cannot vouch.

Second half of the Astra incident (#97487): the live-catalog guard only helps when the fetch
succeeds. A transient Codex outage, or a static fallback that lags an early-access rollout, left
``/model gpt-6-astra`` on ``openai-codex`` walking the ladder to OpenRouter (which relists every
vendor) and silently rebuilding the session on a metered aggregator.
"""

from __future__ import annotations

import pytest

from hermes_cli import models, models_detect


@pytest.fixture
def ladder_would_hijack(monkeypatch):
    """Live catalog unavailable; OpenRouter lists everything; the user holds an OpenRouter key."""
    monkeypatch.setattr(models, "cached_provider_model_ids", lambda provider, **_: [])
    monkeypatch.setattr(models, "_find_openrouter_slug", lambda name: f"vendor/{name}")
    monkeypatch.setattr(models_detect, "provider_has_credentials", lambda p: p == "openrouter")


@pytest.mark.parametrize("provider,model", [
    ("openai-codex", "gpt-6-astra"),
    ("xai-oauth", "grok-5-preview"),
    ("anthropic", "claude-opus-5-early"),
])
def test_own_vendor_id_stays_when_live_catalog_is_empty(ladder_would_hijack, provider, model):
    assert models.detect_provider_for_model(model, provider) is None


@pytest.mark.parametrize("provider,model", [
    ("openai-codex", "claude-opus-4.7"),   # other vendor's id on a single-vendor provider
    ("bedrock", "deepseek-v4-pro"),        # multi-vendor catalog whose non-deepseek ids the
                                           # classifier cannot place: never "exclusively deepseek"
])
def test_non_owned_id_still_remaps_to_keyed_aggregator(ladder_would_hijack, provider, model):
    assert models.detect_provider_for_model(model, provider) == ("openrouter", f"vendor/{model}")
