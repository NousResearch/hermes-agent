"""Regression tests for #106184: ``custom_providers`` entries that share one
base_url with distinct credentials must each keep their own cached
``/v1/models`` catalog.

The Desktop picker surfaces (chat model pill, settings auxiliary slots) request
``model.options`` with live probing off, so section 4 of
``list_authenticated_providers`` serves each row from
``cached_fetch_api_models(..., cache_only=True)``. That disk cache used a
URL-only key (``custom:<base_url>``) with the credential fingerprint stored
inside the entry, so with several same-URL entries (gateways issuing
per-tenant keys) only the credential that probed last could re-read its own
catalog; every other row collapsed to an empty model list and the frontend's
``models.length > 0`` filter hid the providers from the UI entirely.
"""

from __future__ import annotations

import time
from unittest.mock import patch

GW_URL = "https://gw.example.com/v1"


class TestSharedUrlCredentialSlots:
    def _isolate_disk(self, monkeypatch, tmp_path):
        import hermes_cli.models as mod

        cache_path = tmp_path / "provider_models_cache.json"
        monkeypatch.setattr(mod, "_provider_models_cache_path", lambda: cache_path)
        return mod

    def test_each_credential_keeps_its_own_slot(self, monkeypatch, tmp_path):
        """Two keys on one base_url must not evict each other: after both
        probe, a cache-only (picker) read must still answer for each key."""
        mod = self._isolate_disk(monkeypatch, tmp_path)

        catalogs = {"sk-a": ["model-a1", "model-a2"], "sk-b": ["model-b1"]}

        def fake_fetch(api_key, base_url, **kwargs):
            return list(catalogs[api_key])

        monkeypatch.setattr(mod, "fetch_api_models", fake_fetch)

        # Both credentials probe (any order) — each writes a cache entry.
        assert mod.cached_fetch_api_models("sk-a", GW_URL) == catalogs["sk-a"]
        assert mod.cached_fetch_api_models("sk-b", GW_URL) == catalogs["sk-b"]

        # Picker opens read cache-only: both must hit their own catalog.
        # Before #106184's fix the second write shadowed the first (single
        # URL-keyed slot), so one of these returned None.
        assert mod.cached_fetch_api_models("sk-a", GW_URL, cache_only=True) == catalogs["sk-a"]
        assert mod.cached_fetch_api_models("sk-b", GW_URL, cache_only=True) == catalogs["sk-b"]

    def test_disk_holds_one_slot_per_credential(self, monkeypatch, tmp_path):
        """The persisted cache must contain two distinct (url, credential)
        keys for the shared endpoint, each with its own fingerprint."""
        mod = self._isolate_disk(monkeypatch, tmp_path)

        monkeypatch.setattr(mod, "fetch_api_models", lambda key, url, **kw: [f"m-{key}"])
        mod.cached_fetch_api_models("sk-a", GW_URL)
        mod.cached_fetch_api_models("sk-b", GW_URL)

        cache = mod._load_provider_models_cache()
        custom_keys = [k for k in cache if k.startswith(f"custom:{GW_URL}")]
        assert len(custom_keys) == 2, f"expected one slot per credential, got {sorted(custom_keys)}"
        fp_a = mod._custom_endpoint_fingerprint("sk-a", None, None)
        fp_b = mod._custom_endpoint_fingerprint("sk-b", None, None)
        assert fp_a != fp_b
        assert cache[f"custom:{GW_URL}:{fp_a}"]["models"] == ["m-sk-a"]
        assert cache[f"custom:{GW_URL}:{fp_b}"]["models"] == ["m-sk-b"]

    def test_cache_only_miss_after_shadowing_is_gone(self, monkeypatch, tmp_path):
        """Pin the exact failing shape from the report: probe with key A, then
        key B, then open the picker twice (cache-only) — A's row must survive
        B's write."""
        mod = self._isolate_disk(monkeypatch, tmp_path)

        state = {"calls": 0}

        def fake_fetch(api_key, base_url, **kwargs):
            state["calls"] += 1
            return [f"catalog-{api_key}"]

        monkeypatch.setattr(mod, "fetch_api_models", fake_fetch)

        assert mod.cached_fetch_api_models("sk-a", GW_URL) == ["catalog-sk-a"]
        assert mod.cached_fetch_api_models("sk-b", GW_URL) == ["catalog-sk-b"]
        assert state["calls"] == 2

        # Both picker opens are cache-only — neither may trigger a live fetch.
        with patch.object(mod, "fetch_api_models") as live:
            assert mod.cached_fetch_api_models("sk-a", GW_URL, cache_only=True) == ["catalog-sk-a"]
            assert mod.cached_fetch_api_models("sk-b", GW_URL, cache_only=True) == ["catalog-sk-b"]
            live.assert_not_called()
        assert state["calls"] == 2

    def test_stale_entry_for_other_credential_is_not_served(self, monkeypatch, tmp_path):
        """Slots are per-credential, but the in-entry fingerprint check stays
        authoritative: an entry whose stored fp disagrees with the caller's
        credentials (e.g. hand-edited cache) is still a miss, never served."""
        mod = self._isolate_disk(monkeypatch, tmp_path)
        cache_path = tmp_path / "provider_models_cache.json"

        fp_a = mod._custom_endpoint_fingerprint("sk-a", None, None)
        stale = {"fp": "someone-else", "at": time.time(), "models": ["not-yours"]}
        mod._save_provider_models_cache({f"custom:{GW_URL}:{fp_a}": stale})
        assert cache_path.exists()

        assert mod.cached_fetch_api_models("sk-a", GW_URL, cache_only=True) is None
