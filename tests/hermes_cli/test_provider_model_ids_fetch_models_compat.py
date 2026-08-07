"""Compatibility for third-party ProviderProfile.fetch_models overrides.

Hermes always calls ``fetch_models(api_key=..., base_url=...)``. Older plugins
that override with a narrower signature raise TypeError; provider_model_ids
must fall back instead of returning an empty catalog.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest


def _make_profile(*, name: str, fetch_models, fallback=("fallback-model",)):
    return SimpleNamespace(
        name=name,
        auth_type="api_key",
        base_url="https://example.invalid/v1",
        fallback_models=fallback,
        fetch_models=fetch_models,
    )


@pytest.fixture(autouse=True)
def _isolate_hermes_home(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))


def test_provider_model_ids_retries_without_base_url_on_typeerror(monkeypatch):
    from hermes_cli import models as hm

    calls: list[dict] = []

    def fetch_models(*, api_key=None, timeout=8.0):
        calls.append({"api_key": api_key, "timeout": timeout})
        return ["live-model-from-narrow"]

    profile = _make_profile(name="narrow-fetch-plugin", fetch_models=fetch_models)

    monkeypatch.setattr(hm, "normalize_provider", lambda p: "narrow-fetch-plugin")
    monkeypatch.setattr(
        "providers.get_provider_profile",
        lambda name: profile if name == "narrow-fetch-plugin" else None,
    )
    monkeypatch.setattr(
        "hermes_cli.auth.resolve_api_key_provider_credentials",
        lambda name: {
            "api_key": "secret",
            "base_url": "https://override.example/v1",
        },
    )
    monkeypatch.setattr(hm, "_PROVIDER_MODELS", {}, raising=False)
    if hasattr(hm, "_MODELS_DEV_PREFERRED"):
        monkeypatch.setattr(hm, "_MODELS_DEV_PREFERRED", set(), raising=False)

    ids = hm.provider_model_ids("narrow-fetch-plugin", force_refresh=True)

    assert "live-model-from-narrow" in ids
    assert calls
    assert calls[-1]["api_key"] == "secret"
    assert "base_url" not in calls[-1]


def test_provider_model_ids_passes_base_url_when_supported(monkeypatch):
    from hermes_cli import models as hm

    calls: list[dict] = []

    def fetch_models(*, api_key=None, base_url=None, timeout=8.0):
        calls.append({"api_key": api_key, "base_url": base_url, "timeout": timeout})
        return ["live-model-from-full"]

    profile = _make_profile(name="full-fetch-plugin", fetch_models=fetch_models)

    monkeypatch.setattr(hm, "normalize_provider", lambda p: "full-fetch-plugin")
    monkeypatch.setattr(
        "providers.get_provider_profile",
        lambda name: profile if name == "full-fetch-plugin" else None,
    )
    monkeypatch.setattr(
        "hermes_cli.auth.resolve_api_key_provider_credentials",
        lambda name: {
            "api_key": "secret",
            "base_url": "https://override.example/v1",
        },
    )
    monkeypatch.setattr(hm, "_PROVIDER_MODELS", {}, raising=False)
    if hasattr(hm, "_MODELS_DEV_PREFERRED"):
        monkeypatch.setattr(hm, "_MODELS_DEV_PREFERRED", set(), raising=False)

    ids = hm.provider_model_ids("full-fetch-plugin", force_refresh=True)

    assert "live-model-from-full" in ids
    assert calls
    assert calls[-1]["base_url"] == "https://override.example/v1"
