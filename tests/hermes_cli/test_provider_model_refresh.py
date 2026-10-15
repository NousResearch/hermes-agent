"""Background endpoint refresh behavior and cache identity across configured providers."""

import json

import pytest

from hermes_cli.provider_model_refresh import (
    background_refresh_interval, refresh_configured_provider_models,
)


def _config(entries, *, selected, interval=60, named=None):
    return {
        "provider_models": {"background_refresh": {
            "interval_seconds": interval, "providers": selected,
        }},
        "custom_providers": entries,
        "providers": named or {},
    }


@pytest.fixture
def fake_catalog(tmp_path, monkeypatch):
    from hermes_cli import models

    cache_path = tmp_path / "provider_models_cache.json"
    monkeypatch.setattr(models, "_provider_models_cache_path", lambda: cache_path)
    monkeypatch.setattr("hermes_cli.model_switch._scoped_key_env", lambda key: {
        "FIRST_KEY": "secret-one", "SECOND_KEY": "secret-two",
    }.get(key, ""))
    rows = {
        "secret-one": ["one-a", "one-b"],
        "secret-two": ["two-a"],
    }
    calls = []

    def live(key, url, timeout=5.0, api_mode=None, headers=None):
        calls.append((key, url))
        return rows[key] if key in rows else None

    monkeypatch.setattr(models, "fetch_api_models", live)
    return cache_path, rows, calls


def test_disabled_by_default_and_invalid_settings_do_not_probe(fake_catalog):
    _, _, calls = fake_catalog
    entry = {"name": "Local CPA", "base_url": "https://models.example/v1",
             "key_env": "FIRST_KEY"}
    for interval, selected in [(0, ["custom:local-cpa"]), (59, ["custom:local-cpa"]),
                               (True, ["custom:local-cpa"]), (60, []),
                               (60, [""]), (60, ["custom:local-cpa", 99])]:
        config = _config([entry], selected=selected, interval=interval)
        assert background_refresh_interval(config) == 0
        assert refresh_configured_provider_models(config) == (0, 0)
    assert refresh_configured_provider_models({"custom_providers": [entry]}) == (0, 0)
    assert not calls


def test_opted_in_updates_native_picker_cache_without_extra_network(fake_catalog):
    cache_path, rows, calls = fake_catalog
    config = _config([
        {"name": "Local CPA", "base_url": "https://models.example/v1",
         "key_env": "FIRST_KEY"},
        {"name": "Other", "base_url": "https://unlisted.example/v1",
         "key_env": "SECOND_KEY"},
        {"name": "Pinned", "base_url": "http://127.0.0.1:9001/v1",
         "models": ["pinned"], "discover_models": False},
    ], selected=["custom:local-cpa"])
    assert refresh_configured_provider_models(config) == (1, 0)
    assert calls == [("secret-one", "https://models.example/v1")]
    saved = json.loads(cache_path.read_text())
    assert len(saved) == 1
    key = next(iter(saved))
    assert key.startswith("custom:https://models.example/v1#")
    assert saved[key]["models"] == ["one-a", "one-b"]

    rows["secret-one"] = ["one-a", "one-b", "one-c"]
    assert refresh_configured_provider_models(config) == (1, 0)
    assert json.loads(cache_path.read_text())[key]["models"][-1] == "one-c"
    assert len(calls) == 2  # force-refresh even inside the one-hour TTL


def test_same_endpoint_different_credentials_remain_isolated(fake_catalog):
    cache_path, _, calls = fake_catalog
    config = _config([
        {"name": "Local CPA", "base_url": "https://models.example/v1",
         "key_env": "FIRST_KEY"},
        {"name": "Local New API", "base_url": "https://models.example/v1",
         "key_env": "SECOND_KEY"},
    ], selected=["custom:local-cpa", "custom:local-new-api"])
    assert refresh_configured_provider_models(config) == (2, 0)
    data = json.loads(cache_path.read_text())
    assert len(data) == 2
    assert {tuple(v["models"]) for v in data.values()} == {
        ("one-a", "one-b"), ("two-a",),
    }
    assert len(calls) == 2


def test_failure_retains_last_success_and_next_run_recovers(fake_catalog):
    cache_path, rows, _ = fake_catalog
    config = _config([
        {"name": "Local CPA", "base_url": "https://models.example/v1",
         "key_env": "FIRST_KEY"},
    ], selected=["custom:local-cpa"])
    assert refresh_configured_provider_models(config) == (1, 0)
    previous = json.loads(cache_path.read_text())
    rows.pop("secret-one")
    assert refresh_configured_provider_models(config) == (1, 1)
    assert json.loads(cache_path.read_text()) == previous

    rows["secret-one"] = ["new-model"]
    assert refresh_configured_provider_models(config) == (1, 0)
    assert next(iter(json.loads(cache_path.read_text()).values()))["models"] == ["new-model"]


def test_allowlist_and_disabled_entry_skip_network(fake_catalog):
    _, _, calls = fake_catalog
    config = _config([
        {"name": "Pinned", "base_url": "https://pinned.example/v1",
         "models": ["static-id"]},
        {"name": "Do Not Discover", "base_url": "https://disabled.example/v1",
         "key_env": "FIRST_KEY", "discover_models": False},
        {"name": "Bad key", "base_url": "https://missing.example/v1",
         "key_env": "NOT_PRESENT"},
    ], selected=["custom:pinned", "custom:do-not-discover", "custom:bad-key"],
       named={"disabled": {"base_url": "https://disabled.example/v1",
                           "enabled": False, "key_env": "FIRST_KEY"}})
    config["provider_models"]["background_refresh"]["providers"].append("custom:disabled")
    assert refresh_configured_provider_models(config) == (1, 1)
    assert not calls


def test_named_providers_use_same_configured_identity(fake_catalog):
    cache_path, _, calls = fake_catalog
    config = _config([], selected=["custom:my-proxy"], named={
        "my-proxy": {"name": "Display name", "api": "https://models.example/v1",
                     "key_env": "SECOND_KEY"},
        "unlisted": {"api": "https://no.example/v1", "key_env": "FIRST_KEY"},
    })
    assert refresh_configured_provider_models(config) == (1, 0)
    assert calls == [("secret-two", "https://models.example/v1")]
    assert next(iter(json.loads(cache_path.read_text()).values()))["models"] == ["two-a"]


def test_live_cache_write_preserves_sibling_added_during_probe(tmp_path, monkeypatch):
    from hermes_cli import models

    cache_path = tmp_path / "provider_models_cache.json"
    monkeypatch.setattr(models, "_provider_models_cache_path", lambda: cache_path)

    def live(*_args, **_kwargs):
        models.update_provider_cache_entry("sibling", ["sibling-model"])
        return ["current-model"]

    monkeypatch.setattr(models, "fetch_api_models", live)
    got = models.cached_fetch_api_models(
        "key", "https://models.example/v1", force_refresh=True)
    assert got == ["current-model"]
    stored = json.loads(cache_path.read_text())
    assert "sibling" in stored
    assert len(stored) == 2


def test_legacy_model_suffixed_names_resolve_to_picker_group(fake_catalog):
    cache_path, _, calls = fake_catalog
    config = _config([
        {"name": "Local CPA — model-a", "base_url": "https://models.example/v1",
         "key_env": "FIRST_KEY"},
        {"name": "Local CPA - model-b", "base_url": "https://models.example/v1",
         "key_env": "FIRST_KEY"},
    ], selected=["custom:local-cpa"])
    assert refresh_configured_provider_models(config) == (1, 0)
    assert calls == [("secret-one", "https://models.example/v1")]
    assert len(json.loads(cache_path.read_text())) == 1


def test_forced_native_ollama_bypasses_memory_memo(monkeypatch):
    import io
    import time
    from hermes_cli import models, models_local

    endpoint = "http://127.0.0.1:11434/v1"
    root = models_local._root_for_ollama_native_api(endpoint)
    cache_key = models_local._ollama_probe_cache_key(root, None)
    monkeypatch.setitem(
        models_local._OLLAMA_LOCAL_MODELS_CACHE,
        cache_key, (("old-model",), time.monotonic()),
    )

    calls = []

    def fresh(_request, timeout):
        calls.append(timeout)
        return io.BytesIO(b'{"models": [{"model": "new-model"}]}')

    monkeypatch.setattr(models, "_urlopen_model_catalog_request", fresh)
    assert models_local.fetch_ollama_local_models(endpoint) == ["old-model"]
    assert not calls
    assert models_local.fetch_ollama_local_models(endpoint, force_refresh=True) == ["new-model"]
    assert calls == [2.0]
    assert models_local.fetch_ollama_local_models(endpoint) == ["new-model"]
