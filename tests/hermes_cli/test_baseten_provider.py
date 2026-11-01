"""Focused tests for Baseten Model APIs provider wiring."""

from __future__ import annotations

from hermes_cli.auth import (
    resolve_api_key_provider_credentials,
    resolve_provider,
)
from hermes_cli.model_normalize import normalize_model_for_provider
from hermes_cli.models import (
    normalize_provider,
    provider_model_ids,
)


def test_baseten_aliases_resolve(monkeypatch):
    monkeypatch.setenv("BASETEN_API_KEY", "baseten-test-key")

    for alias in ("baseten", "baseten-ai", "basetenai", "baseten-model-apis", "bt"):
        assert resolve_provider(alias) == "baseten"
        assert normalize_provider(alias) == "baseten"


def test_baseten_provider_registry_and_credentials(monkeypatch):
    monkeypatch.setenv("BASETEN_API_KEY", "baseten-secret")
    monkeypatch.delenv("BASETEN_BASE_URL", raising=False)

    creds = resolve_api_key_provider_credentials("baseten")
    assert creds["provider"] == "baseten"
    assert creds["api_key"] == "baseten-secret"
    assert creds["base_url"] == "https://inference.baseten.co/v1"


def test_baseten_base_url_override_reaches_dedicated_deployments(monkeypatch):
    """A dedicated deployment is a per-model host, so it is addressed by
    overriding the base URL rather than by a catalog id."""
    monkeypatch.setenv("BASETEN_API_KEY", "baseten-secret")
    monkeypatch.setenv(
        "BASETEN_BASE_URL",
        "https://model-abcd1234.api.baseten.co/environments/production/sync/v1",
    )

    creds = resolve_api_key_provider_credentials("baseten")
    assert creds["base_url"] == (
        "https://model-abcd1234.api.baseten.co/environments/production/sync/v1"
    )


def test_baseten_model_catalog_prefers_live_profile_fetch(monkeypatch):
    from providers import get_provider_profile

    profile = get_provider_profile("baseten")
    assert profile is not None
    monkeypatch.setattr(
        "hermes_cli.auth.resolve_api_key_provider_credentials",
        lambda provider_id: {
            "provider": provider_id,
            "api_key": "baseten-live-key",
            "base_url": "https://inference.baseten.co/v1",
            "source": "BASETEN_API_KEY",
        },
    )
    monkeypatch.setattr(
        profile,
        "fetch_models",
        lambda *, api_key=None, base_url=None, timeout=8.0: [
            "zai-org/GLM-5.3",
            "deepseek-ai/DeepSeek-V4-Pro-0813",
            "some-brand-new/Live-Only-Model",
        ],
    )

    # Curated-first merge policy: the profile's fallback_models lead the picker,
    # live-only entries are appended, live duplicates of curated entries deduped.
    result = provider_model_ids("baseten")
    assert result[: len(profile.fallback_models)] == list(profile.fallback_models)
    assert "some-brand-new/Live-Only-Model" in result[len(profile.fallback_models) :]


def test_baseten_model_catalog_falls_back_to_profile_models(monkeypatch):
    from providers import get_provider_profile

    profile = get_provider_profile("baseten")
    assert profile is not None
    monkeypatch.setattr(
        "hermes_cli.auth.resolve_api_key_provider_credentials",
        lambda provider_id: {
            "provider": provider_id,
            "api_key": "baseten-live-key",
            "base_url": "https://inference.baseten.co/v1",
            "source": "BASETEN_API_KEY",
        },
    )
    monkeypatch.setattr(profile, "fetch_models", lambda *, api_key=None, base_url=None, timeout=8.0: None)

    assert provider_model_ids("bt")[: len(profile.fallback_models)] == list(profile.fallback_models)


def test_baseten_transport_emits_top_level_reasoning_effort():
    from agent.transports.chat_completions import ChatCompletionsTransport
    from providers import get_provider_profile

    profile = get_provider_profile("baseten")
    assert profile is not None

    kwargs = ChatCompletionsTransport().build_kwargs(
        model="zai-org/GLM-5.3",
        messages=[{"role": "user", "content": "ping"}],
        tools=None,
        provider_profile=profile,
        reasoning_config={"enabled": True, "effort": "low"},
        base_url="https://inference.baseten.co/v1",
        provider_name="baseten",
    )
    assert kwargs["reasoning_effort"] == "low"
    assert "extra_body" not in kwargs


def test_baseten_model_normalization_strips_only_matching_prefix():
    """``baseten/`` is repaired away; the vendor segment Baseten's catalog
    actually requires must survive untouched."""
    model = "zai-org/GLM-5.3"
    assert normalize_model_for_provider(f"baseten/{model}", "baseten") == model
    assert normalize_model_for_provider(f"bt/{model}", "baseten") == model
    assert normalize_model_for_provider(model, "baseten") == model
    assert normalize_model_for_provider("openai/gpt-oss-120b", "baseten") == "openai/gpt-oss-120b"


def test_baseten_is_in_the_canonical_provider_picker():
    from hermes_cli.models_catalog_static import CANONICAL_PROVIDERS

    entry = next((p for p in CANONICAL_PROVIDERS if p.slug == "baseten"), None)
    assert entry is not None, "baseten must be offered in the `hermes model` picker"
    assert entry.label == "Baseten"


def test_baseten_is_mapped_to_its_models_dev_registry_id():
    """``_models_dev_id`` has no identity fallback — an unmapped slug silently
    yields an empty catalog and no pricing, which is how this shipped broken once."""
    from agent.models_dev import PROVIDER_TO_MODELS_DEV, _models_dev_id

    assert PROVIDER_TO_MODELS_DEV.get("baseten") == "baseten"
    assert _models_dev_id("baseten") == "baseten"


def test_baseten_pricing_comes_from_the_models_dev_registry(monkeypatch):
    """Drives the whole path — slug → models.dev id → registry entry → per-token
    rows — rather than stubbing the lookup that the id mapping feeds."""
    from agent import models_dev
    from hermes_cli import models_pricing

    monkeypatch.setattr(
        models_dev, "fetch_models_dev",
        lambda *a, **kw: {
            "baseten": {
                "name": "Baseten",
                "models": {
                    "zai-org/GLM-5.3": {"cost": {"input": 1.4, "output": 4.4, "cache_read": 0.14}},
                    "no-cost/Model": {},
                },
            },
        },
    )
    monkeypatch.setattr(models_pricing, "_cached_catalog", lambda key: None)
    monkeypatch.setattr(models_pricing, "_cache_catalog", lambda key, value: value)

    pricing = models_pricing.get_pricing_for_provider("baseten", force_refresh=True)
    assert "zai-org/GLM-5.3" in pricing, "baseten pricing must resolve through the registry"
    row = pricing["zai-org/GLM-5.3"]
    assert float(row["prompt"]) > 0 and float(row["completion"]) > 0
    assert "input_cache_read" in row
    # Entries without cost data are dropped rather than priced at zero.
    assert "no-cost/Model" not in pricing
