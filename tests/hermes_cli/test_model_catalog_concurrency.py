"""Bounded model-catalog discovery and cache merge guarantees."""

from __future__ import annotations

import json
import threading

from hermes_cli import models
from hermes_cli.model_switch import _BoundedProviderCatalogLoader


def test_catalog_loader_runs_in_parallel_with_a_fixed_cap() -> None:
    release = threading.Event()
    reached_cap = threading.Event()
    lock = threading.Lock()
    active = 0
    maximum_active = 0

    def fetch(provider: str) -> list[str]:
        nonlocal active, maximum_active
        with lock:
            active += 1
            maximum_active = max(maximum_active, active)
            if active == 4:
                reached_cap.set()
        assert release.wait(timeout=2)
        with lock:
            active -= 1
        return [f"{provider}-model"]

    loader = _BoundedProviderCatalogLoader(fetch, max_workers=4)
    loader.prefetch([f"provider-{index}" for index in range(6)])
    assert reached_cap.wait(timeout=2)
    assert maximum_active == 4

    release.set()
    assert loader.get("provider-5") == ["provider-5-model"]
    loader.close()
    assert maximum_active == 4


def test_concurrent_provider_cache_writes_merge_entries(tmp_path, monkeypatch) -> None:
    cache_path = tmp_path / "provider_models_cache.json"
    barrier = threading.Barrier(2)
    results: dict[str, list[str]] = {}

    monkeypatch.setattr(models, "_provider_models_cache_path", lambda: cache_path)
    monkeypatch.setattr(models, "_credential_fingerprint", lambda provider: provider)

    def fetch(provider: str, *, force_refresh: bool = False) -> list[str]:
        del force_refresh
        barrier.wait(timeout=2)
        return [f"{provider}-model"]

    monkeypatch.setattr(models, "provider_model_ids", fetch)

    threads = [
        threading.Thread(
            target=lambda provider=provider: results.setdefault(
                provider,
                models.cached_provider_model_ids(provider, force_refresh=True),
            )
        )
        for provider in ("anthropic", "openrouter")
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=2)

    assert all(not thread.is_alive() for thread in threads)
    assert results == {
        "anthropic": ["anthropic-model"],
        "openrouter": ["openrouter-model"],
    }
    stored = json.loads(cache_path.read_text(encoding="utf-8"))
    assert set(stored) == {"anthropic", "openrouter"}
