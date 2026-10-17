"""A normal picker open (``cached_only=True``) finds pricing after a backend restart.

#131240: the pricing catalog cache is process-local, so every backend restart left the
first picker open unpriced. The disk persistence must serve the picker's real read
seam — ``get_pricing_for_provider(..., cached_only=True)`` — not only direct
``_cached_catalog`` reads, for every provider the picker prices: a cold process must
find the entry under the key the previous process wrote (the provider-key map is
process-local too, and nous/novita/deepinfra keys are resolved dynamically).
"""

from __future__ import annotations

import time

import pytest

import hermes_cli.models_pricing as mp


@pytest.fixture(autouse=True)
def _isolated_disk_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(mp, "_disk_cache_path", lambda: tmp_path / "pricing-cache.json")
    mp._pricing_cache.clear()
    mp._pricing_cache_retry_after.clear()
    mp._disk_cache_checked.clear()
    mp._pricing_provider_cache_keys.clear()
    yield
    mp._pricing_cache.clear()
    mp._pricing_cache_retry_after.clear()
    mp._disk_cache_checked.clear()
    mp._pricing_provider_cache_keys.clear()


def _restart_process_caches() -> None:
    """Simulate a backend restart: every process-local pricing map goes cold."""
    mp._pricing_cache.clear()
    mp._pricing_cache_retry_after.clear()
    mp._disk_cache_checked.clear()
    mp._pricing_provider_cache_keys.clear()


def test_nous_picker_read_finds_the_persisted_catalog_after_a_restart(monkeypatch):
    """The nous cache key is dynamic (base + credential fingerprint), so the cold
    process must resolve the persisted entry without the process-local key map."""
    base = "https://inference.example.test"
    monkeypatch.setattr(mp, "get_cached_nous_inference_base_url", lambda: base)
    catalog = {"nous/model-a": {"prompt": "0", "completion": "0"}}
    key = base + mp._pricing_auth_fingerprint("sk-test")
    mp._cache_catalog(key, catalog, ttl_seconds=300.0)

    _restart_process_caches()

    pricing = mp.get_pricing_for_provider("nous", cached_only=True)
    assert pricing == catalog, "a normal picker open after a restart must find the persisted nous catalog"


def test_nous_expired_policy_catalog_is_not_resurrected(monkeypatch):
    """The persisted Nous catalog is the org's policy allowlist; an expired entry
    must not answer a restart-warmed picker read."""
    base = "https://inference.example.test"
    monkeypatch.setattr(mp, "get_cached_nous_inference_base_url", lambda: base)
    key = base + mp._pricing_auth_fingerprint("sk-test")
    now = time.time()
    monkeypatch.setattr(time, "time", lambda: now)
    mp._cache_catalog(key, {"nous/revoked": {"prompt": "1"}}, ttl_seconds=300.0)

    _restart_process_caches()
    monkeypatch.setattr(time, "time", lambda: now + 301.0)
    try:
        assert mp.get_pricing_for_provider("nous", cached_only=True) == {}
    finally:
        monkeypatch.undo()


def test_openrouter_picker_read_finds_the_persisted_catalog_after_a_restart():
    catalog = {"openrouter/auto": {"prompt": "0.000001", "completion": "0"}}
    mp._cache_catalog("https://openrouter.ai/api", catalog)

    _restart_process_caches()

    assert mp.get_pricing_for_provider("openrouter", cached_only=True) == catalog


def test_novita_picker_read_finds_the_persisted_catalog_after_a_restart(monkeypatch):
    monkeypatch.setenv("NOVITA_BASE_URL", "https://api.novita.example.test/openai/v1")
    catalog = {"novita/model": {"prompt": "0.000001", "completion": "0"}}
    mp._cache_catalog("https://api.novita.example.test/openai/v1", catalog)

    _restart_process_caches()

    assert mp.get_pricing_for_provider("novita", cached_only=True) == catalog


def test_deepinfra_picker_read_finds_the_persisted_catalog_after_a_restart(monkeypatch):
    """DeepInfra's derived pricing must survive a restart like every other priced
    provider's, instead of depending on the process-local raw catalog cache."""
    monkeypatch.setattr(
        "hermes_cli.models._deepinfra_catalog_url",
        lambda: ("https://api.deepinfra.example.test#anon", "https://api.deepinfra.example.test/models?q"),
        raising=False,
    )
    catalog = {"deepinfra/model": {"prompt": "0.0000001", "completion": "0"}}
    mp._cache_catalog("https://api.deepinfra.example.test#anon", catalog)

    _restart_process_caches()

    assert mp.get_pricing_for_provider("deepinfra", cached_only=True) == catalog
