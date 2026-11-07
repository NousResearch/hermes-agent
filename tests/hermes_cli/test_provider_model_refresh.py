"""Provider catalog background refresh unit tests."""

import json
from unittest.mock import patch
import pytest

from hermes_cli.provider_model_refresh import (
    background_refresh_interval,
    refresh_configured_provider_models,
)


def test_background_refresh_interval_bounds():
    assert background_refresh_interval({}) == 0
    assert background_refresh_interval({"provider_models": {}}) == 0
    assert background_refresh_interval({"provider_models": {"background_refresh": None}}) == 0
    assert background_refresh_interval({"provider_models": {"background_refresh": {"interval_seconds": 59, "providers": ["ollama"]}}}) == 0
    assert background_refresh_interval({"provider_models": {"background_refresh": {"interval_seconds": 86401, "providers": ["ollama"]}}}) == 0
    assert background_refresh_interval({"provider_models": {"background_refresh": {"interval_seconds": True, "providers": ["ollama"]}}}) == 0
    assert background_refresh_interval({"provider_models": {"background_refresh": {"interval_seconds": 60, "providers": []}}}) == 0
    assert background_refresh_interval({"provider_models": {"background_refresh": {"interval_seconds": 60, "providers": [""]}}}) == 0
    assert background_refresh_interval({"provider_models": {"background_refresh": {"interval_seconds": 60, "providers": ["ollama"]}}}) == 60


@pytest.fixture
def fake_catalog(tmp_path, monkeypatch):
    cache_path = tmp_path / "provider_models_cache.json"
    rows = {
        "secret-one": ["one-a", "one-b"],
        "secret-two": ["two-a"],
    }
    calls = []

    def fake_fetch(api_key, api_url, native_catalog_provider="custom", preserve_native_models=False, headers=None, timeout=5.0, api_mode=None, cache=True, force_native_refresh=False):
        calls.append((api_key, api_url))
        return list(rows.get(api_key, []))

    monkeypatch.setattr("hermes_cli.models._provider_models_cache_path", lambda: cache_path)
    monkeypatch.setattr("hermes_cli.model_switch_providers._fetch_picker_live_models", fake_fetch)
    monkeypatch.setattr("agent.command_token_source.materialize_probe_api_key", lambda key: key)
    monkeypatch.setenv("FIRST_KEY", "secret-one")
    monkeypatch.setenv("SECOND_KEY", "secret-two")
    return cache_path, rows, calls


def _config(custom_providers, selected=None):
    return {
        "provider_models": {
            "background_refresh": {
                "interval_seconds": 60,
                "providers": list(selected or []),
            },
        },
        "custom_providers": custom_providers,
    }


def test_refresh_configured_providers_empty(fake_catalog):
    cache_path, _, calls = fake_catalog
    summary = refresh_configured_provider_models(_config([], selected=["custom:local-cpa"]))
    assert summary.attempted == 0
    assert summary.failed == 0
    assert not summary.diffs
    assert not calls
    assert not cache_path.exists()


def test_refresh_configured_providers_skips_when_disabled(fake_catalog):
    config = _config([{"name": "Local CPA", "base_url": "https://models.example/v1"}], selected=["custom:local-cpa"])
    config["provider_models"]["background_refresh"]["interval_seconds"] = 0
    summary = refresh_configured_provider_models(config)
    assert summary == (0, 0, ())


def test_refresh_populates_cache_and_detects_catalog_diffs(fake_catalog):
    cache_path, rows, _calls = fake_catalog
    config = _config([
        {"name": "Local CPA", "base_url": "https://models.example/v1", "key_env": "FIRST_KEY"},
    ], selected=["custom:local-cpa"])

    # Initial baseline run: populates cache, diffs list is empty.
    summary1 = refresh_configured_provider_models(config)
    assert summary1.attempted == 1
    assert summary1.failed == 0
    assert not summary1.diffs

    with open(cache_path) as f:
        data = json.load(f)
    assert any("custom:https://models.example/v1" in k for k in data)

    # Second run without upstream catalog changes: 0 diffs.
    summary2 = refresh_configured_provider_models(config)
    assert summary2.attempted == 1
    assert summary2.failed == 0
    assert not summary2.diffs

    # Upstream changes: "one-a" is removed, "one-c" and "one-d" are added.
    rows["secret-one"] = ["one-b", "one-c", "one-d"]
    summary3 = refresh_configured_provider_models(config)
    assert summary3.attempted == 1
    assert summary3.failed == 0
    assert len(summary3.diffs) == 1

    diff = summary3.diffs[0]
    assert diff.slug == "custom:local-cpa"
    assert diff.added == ("one-c", "one-d")
    assert diff.removed == ("one-a",)
    assert diff.has_changes is True
