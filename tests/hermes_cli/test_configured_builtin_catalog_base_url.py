"""Built-in picker catalogs must probe the endpoint selected in ``model`` config."""

from __future__ import annotations

import pytest

from hermes_cli import models
from providers import get_provider_profile


@pytest.mark.parametrize("provider", ["deepseek", "gemini", "zai"])
def test_builtin_catalog_uses_matching_model_base_url(monkeypatch, provider):
    """A configured relay owns both inference and its picker ``/models`` probe."""
    configured_url = f"http://127.0.0.1:9001/{provider}/v1"
    calls: list[tuple[str, str]] = []

    monkeypatch.setattr(
        models,
        "_get_model_config_dict",
        lambda: {"provider": provider, "base_url": configured_url},
    )
    monkeypatch.setattr(
        "hermes_cli.auth.resolve_api_key_provider_credentials",
        lambda provider_id: {
            "provider": provider_id,
            "api_key": "test-key",
            "base_url": "https://vendor.invalid/v1",
        },
    )

    def fetch_models(*, api_key, base_url):
        calls.append((api_key, base_url))
        return ["relay-only-model"]

    profile = get_provider_profile(provider)
    monkeypatch.setattr(profile, "fetch_models", fetch_models)

    assert "relay-only-model" in models.provider_model_ids(provider, force_refresh=True)
    assert calls == [("test-key", configured_url)]


def test_builtin_catalog_keeps_resolved_endpoint_without_matching_model_config(monkeypatch):
    """A stale base URL for another provider must not redirect a catalog probe."""
    calls: list[str] = []
    monkeypatch.setattr(
        models,
        "_get_model_config_dict",
        lambda: {"provider": "gemini", "base_url": "http://127.0.0.1:9001/gemini/v1"},
    )
    monkeypatch.setattr(
        "hermes_cli.auth.resolve_api_key_provider_credentials",
        lambda provider_id: {
            "provider": provider_id,
            "api_key": "test-key",
            "base_url": "https://api.deepseek.com/v1",
        },
    )
    profile = get_provider_profile("deepseek")
    monkeypatch.setattr(
        profile,
        "fetch_models",
        lambda *, api_key, base_url: calls.append(base_url) or ["canonical-model"],
    )

    assert "canonical-model" in models.provider_model_ids("deepseek", force_refresh=True)
    assert calls == ["https://api.deepseek.com/v1"]


@pytest.mark.parametrize("provider", ["gmi", "stepfun"])
def test_simple_builtin_catalog_uses_matching_model_base_url(monkeypatch, provider):
    """Fetcher overrides use the same configured endpoint rule as profile catalogs."""
    configured_url = f"http://127.0.0.1:9001/{provider}/v1"
    calls: list[tuple[str, str]] = []
    monkeypatch.setattr(
        models,
        "_get_model_config_dict",
        lambda: {"provider": provider, "base_url": configured_url},
    )
    monkeypatch.setattr(
        "hermes_cli.auth.resolve_api_key_provider_credentials",
        lambda provider_id: {
            "provider": provider_id,
            "api_key": "test-key",
            "base_url": "https://vendor.invalid/v1",
        },
    )
    monkeypatch.setattr(
        models,
        "fetch_api_models",
        lambda api_key, base_url: calls.append((api_key, base_url)) or ["relay-only-model"],
    )

    assert models.provider_model_ids(provider, force_refresh=True) == ["relay-only-model"]
    assert calls == [("test-key", configured_url)]


def test_builtin_catalog_skips_probe_without_credentials(monkeypatch):
    """Endpoint overrides do not turn a missing-key catalog into a network probe."""
    monkeypatch.setattr(
        models,
        "_get_model_config_dict",
        lambda: {"provider": "deepseek", "base_url": "http://127.0.0.1:9001/deepseek/v1"},
    )
    monkeypatch.setattr(
        "hermes_cli.auth.resolve_api_key_provider_credentials",
        lambda provider_id: {"provider": provider_id, "api_key": "", "base_url": "https://api.deepseek.com/v1"},
    )
    profile = get_provider_profile("deepseek")
    monkeypatch.setattr(
        profile,
        "fetch_models",
        lambda **_: pytest.fail("missing credentials must not probe the configured endpoint"),
    )

    assert models.provider_model_ids("deepseek", force_refresh=True) == list(profile.fallback_models)


def test_builtin_catalog_cache_identity_tracks_configured_base_url(monkeypatch):
    """Changing a relay endpoint cannot reuse the previous endpoint's catalog row."""
    monkeypatch.setattr(
        models,
        "_get_model_config_dict",
        lambda: {"provider": "deepseek", "base_url": "http://127.0.0.1:9001/one/v1"},
    )
    first = models._credential_fingerprint("deepseek")
    monkeypatch.setattr(
        models,
        "_get_model_config_dict",
        lambda: {"provider": "deepseek", "base_url": "http://127.0.0.1:9001/two/v1"},
    )

    assert models._credential_fingerprint("deepseek") != first


def test_deepinfra_catalog_uses_matching_model_base_url(monkeypatch):
    """The tagged DeepInfra catalog keeps its query but targets the configured relay."""
    monkeypatch.setattr(
        models,
        "_get_model_config_dict",
        lambda: {"provider": "deepinfra", "base_url": "http://127.0.0.1:9001/deepinfra/v1"},
    )
    monkeypatch.setattr(
        models,
        "_deepinfra_env",
        lambda key: "test-key" if key == "DEEPINFRA_API_KEY" else "https://vendor.invalid/v1",
    )

    _, url = models._deepinfra_catalog_url()

    assert url.startswith("http://127.0.0.1:9001/deepinfra/v1/models?")
