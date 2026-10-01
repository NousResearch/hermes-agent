"""Tests for lower-owned remote model catalogue policy and cache lifecycle."""

from __future__ import annotations

import json
import os
import time

from models import catalog_runtime
from models.catalog_manifest import (
    DEFAULT_CATALOG_FALLBACK_URLS,
    catalog_settings,
    curated_model_ids,
    curated_openrouter_models,
    default_model,
    provider_block,
    validate_manifest,
)
from models.catalog_seed import seed_cache_from_checkout


def _manifest(default_id: str = "vendor/primary"):
    return {
        "version": 1,
        "providers": {
            "openrouter": {
                "models": [
                    {
                        "id": "vendor/primary",
                        "description": "recommended",
                        "default": default_id == "vendor/primary",
                    },
                    {
                        "id": "vendor/secondary",
                        "description": "secondary",
                        "default": default_id == "vendor/secondary",
                    },
                ]
            },
            "nous": {"models": [{"id": "nous/model"}]},
        },
    }


def test_manifest_policy_and_settings():
    settings = catalog_settings(
        {
            "ttl_minutes": 5,
            "providers": {"openrouter": {"url": "https://example.test/catalog.json"}},
        }
    )
    block = provider_block(_manifest(), "openrouter")

    assert settings.ttl_seconds == 300
    assert settings.provider_url("openrouter") == "https://example.test/catalog.json"
    assert validate_manifest(_manifest())
    assert not validate_manifest("bad")
    assert not validate_manifest({"version": 2, "providers": {}})
    assert curated_model_ids(block) == ("vendor/primary", "vendor/secondary")
    assert curated_openrouter_models(block)[0] == ("vendor/primary", "recommended")
    assert default_model(block) == "vendor/primary"


def test_legacy_ttl_hours_is_honoured_when_minutes_is_default():
    assert catalog_settings({"ttl_minutes": 20, "ttl_hours": 3}).ttl_seconds == 10800


def test_fallback_chain_uses_primary_then_raw_fallback(monkeypatch):
    settings = catalog_settings({"url": "https://primary.test/catalog.json"})
    calls = []

    def fetch(url, _user_agent, timeout=8.0):
        calls.append(url)
        return None if url == settings.url else _manifest()

    monkeypatch.setattr(catalog_runtime, "_fetch", fetch)

    assert catalog_runtime._fetch_with_fallback(settings, "test-agent") == _manifest()
    assert calls == [settings.url, DEFAULT_CATALOG_FALLBACK_URLS[0]]


def test_fresh_disk_catalog_is_served_without_network(tmp_path, monkeypatch):
    path = tmp_path / "model_catalog.json"
    path.write_text(json.dumps(_manifest()), encoding="utf-8")
    monkeypatch.setattr(
        catalog_runtime,
        "_fetch_with_fallback",
        lambda *_a, **_k: (_ for _ in ()).throw(AssertionError("network fetch")),
    )

    assert catalog_runtime.get_catalog(catalog_settings({"ttl_minutes": 20}), path) == _manifest()


def test_force_refresh_writes_disk_and_replaces_default(tmp_path, monkeypatch):
    path = tmp_path / "model_catalog.json"
    path.write_text(json.dumps(_manifest()), encoding="utf-8")
    replacement = _manifest("vendor/secondary")
    monkeypatch.setattr(catalog_runtime, "_fetch_with_fallback", lambda *_a, **_k: replacement)

    assert catalog_runtime.get_catalog(
        catalog_settings({}), path, force_refresh=True
    ) == replacement
    assert json.loads(path.read_text(encoding="utf-8")) == replacement
    assert catalog_runtime.cached_default_model(path, "openrouter") == "vendor/secondary"


def test_fetch_failure_returns_empty_or_stale_disk(tmp_path, monkeypatch):
    missing = tmp_path / "missing.json"
    monkeypatch.setattr(catalog_runtime, "_fetch_with_fallback", lambda *_a, **_k: None)
    assert catalog_runtime.get_catalog(
        catalog_settings({}), missing, force_refresh=True
    ) == {}

    stale = tmp_path / "stale.json"
    stale.write_text(json.dumps(_manifest()), encoding="utf-8")
    old = time.time() - 30 * 24 * 3600
    os.utime(stale, (old, old))
    assert catalog_runtime.get_catalog(
        catalog_settings({}), stale, force_refresh=True
    ) == _manifest()


def test_stale_disk_is_served_while_refresh_is_scheduled(tmp_path, monkeypatch):
    path = tmp_path / "model_catalog.json"
    path.write_text(json.dumps(_manifest()), encoding="utf-8")
    old = time.time() - 3600
    os.utime(path, (old, old))
    scheduled = []
    monkeypatch.setattr(
        catalog_runtime,
        "_spawn_refresh",
        lambda settings, cache_path, user_agent: scheduled.append(cache_path),
    )

    assert catalog_runtime.get_catalog(
        catalog_settings({"ttl_minutes": 1}), path
    ) == _manifest()
    assert scheduled == [path]


def test_provider_override_takes_precedence(tmp_path, monkeypatch):
    path = tmp_path / "model_catalog.json"
    path.write_text(json.dumps(_manifest()), encoding="utf-8")
    override = {
        "version": 1,
        "providers": {
            "openrouter": {
                "models": [{"id": "override/model", "description": "custom"}]
            }
        },
    }
    settings = catalog_settings(
        {"providers": {"openrouter": {"url": "https://override.test/catalog.json"}}}
    )
    monkeypatch.setattr(catalog_runtime, "_fetch", lambda *_a, **_k: override)

    assert catalog_runtime.curated_openrouter(settings, path) == (
        ("override/model", "custom"),
    )


def test_disabled_catalog_skips_provider_override(tmp_path, monkeypatch):
    settings = catalog_settings(
        {
            "enabled": False,
            "providers": {"openrouter": {"url": "https://override.test/catalog.json"}},
        }
    )
    called = []
    monkeypatch.setattr(catalog_runtime, "_fetch", lambda *_a, **_k: called.append(True))

    assert catalog_runtime.provider_catalog_block(
        settings, tmp_path / "catalog.json", "openrouter"
    ) is None
    assert called == []


def test_cache_is_scoped_by_path(tmp_path):
    a = tmp_path / "a.json"
    b = tmp_path / "b.json"
    a.write_text(json.dumps(_manifest("vendor/primary")), encoding="utf-8")
    b.write_text(json.dumps(_manifest("vendor/secondary")), encoding="utf-8")
    catalog_runtime.reset_cache()

    assert catalog_runtime.cached_default_model(a, "openrouter") == "vendor/primary"
    assert catalog_runtime.cached_default_model(b, "openrouter") == "vendor/secondary"


def test_seed_cache_from_checkout_validates_and_resets(tmp_path):
    project = tmp_path / "repo"
    source = project / "website" / "static" / "api"
    source.mkdir(parents=True)
    manifest = _manifest("vendor/secondary")
    (source / "model-catalog.json").write_text(json.dumps(manifest), encoding="utf-8")
    cache = tmp_path / "home" / "cache" / "model_catalog.json"

    assert seed_cache_from_checkout(project, cache)
    assert catalog_runtime.cached_default_model(cache, "openrouter") == "vendor/secondary"

def test_remembered_chat_catalogue_facts_are_profile_scoped(tmp_path):
    from hermes_constants import set_hermes_home_override, reset_hermes_home_override
    from models.catalog_chat import note_catalog_item, is_known_non_chat_model

    for profile, non_chat in (("a", True), ("b", False), ("a", True)):
        token = set_hermes_home_override(tmp_path / profile)
        try:
            if profile == "a":
                assert note_catalog_item({"id": "opaque-model", "type": "image"})
            assert is_known_non_chat_model("opaque-model") is non_chat
        finally:
            reset_hermes_home_override(token)
