"""Stale-while-revalidate behavior for the model-id disk cache and the
remote model-catalog manifest.

Regression tests for the /model picker stall: when the 1h provider-models
cache TTL (or the catalog manifest TTL) lapsed mid-session, the picker
blocked on 8-9 serial /v1/models round-trips (~2-3s) before rendering.
With SWR, an expired-but-credential-matching entry is served immediately
and refreshed off-thread for the next open.
"""

from __future__ import annotations

import time
from unittest.mock import patch

import pytest


@pytest.fixture(autouse=True)
def _reset_swr_state():
    import hermes_cli.models as models_mod
    with models_mod._swr_refresh_lock:
        models_mod._swr_refresh_inflight.clear()
    yield
    with models_mod._swr_refresh_lock:
        models_mod._swr_refresh_inflight.clear()


class TestProviderModelsSWR:
    def _cache_entry(self, models, age_seconds, fp="fp"):
        return {"fp": fp, "at": time.time() - age_seconds, "models": list(models)}

    def test_fresh_entry_served_without_refresh(self):
        import hermes_cli.models as mod

        cache = {"openrouter": self._cache_entry(["m1"], age_seconds=10)}
        with patch.object(mod, "_load_provider_models_cache", return_value=cache), \
             patch.object(mod, "_credential_fingerprint", return_value="fp"), \
             patch.object(mod, "_spawn_swr_refresh") as spawn, \
             patch.object(mod, "provider_model_ids") as live:
            out = mod.cached_provider_model_ids("openrouter")
        assert out == ["m1"]
        spawn.assert_not_called()
        live.assert_not_called()

    def test_stale_entry_served_immediately_with_background_refresh(self):
        import hermes_cli.models as mod

        # 2h old — beyond the 1h TTL, within the 7d stale-serve window.
        cache = {"openrouter": self._cache_entry(["m1", "m2"], age_seconds=7200)}
        with patch.object(mod, "_load_provider_models_cache", return_value=cache), \
             patch.object(mod, "_credential_fingerprint", return_value="fp"), \
             patch.object(mod, "_spawn_swr_refresh") as spawn, \
             patch.object(mod, "provider_model_ids") as live:
            out = mod.cached_provider_model_ids("openrouter")
        assert out == ["m1", "m2"]  # served stale, no blocking
        spawn.assert_called_once_with("openrouter")
        live.assert_not_called()  # the caller thread never hit the network

    def test_too_old_entry_blocks_on_live_fetch(self):
        import hermes_cli.models as mod

        age = mod._PROVIDER_MODELS_STALE_SERVE_MAX + 60
        cache = {"openrouter": self._cache_entry(["ancient"], age_seconds=age)}
        with patch.object(mod, "_load_provider_models_cache", return_value=cache), \
             patch.object(mod, "_credential_fingerprint", return_value="fp"), \
             patch.object(mod, "_save_provider_models_cache"), \
             patch.object(mod, "_spawn_swr_refresh") as spawn, \
             patch.object(mod, "provider_model_ids", return_value=["fresh"]) as live:
            out = mod.cached_provider_model_ids("openrouter")
        assert out == ["fresh"]
        spawn.assert_not_called()
        live.assert_called_once()

    def test_credential_rotation_still_busts_stale_entry(self):
        import hermes_cli.models as mod

        # Stale entry with a DIFFERENT fingerprint (key rotated) must NOT be
        # served — it reflects the old credentials' catalog.
        cache = {"openrouter": self._cache_entry(["old-key-models"], 7200, fp="old")}
        with patch.object(mod, "_load_provider_models_cache", return_value=cache), \
             patch.object(mod, "_credential_fingerprint", return_value="new"), \
             patch.object(mod, "_save_provider_models_cache"), \
             patch.object(mod, "_spawn_swr_refresh") as spawn, \
             patch.object(mod, "provider_model_ids", return_value=["new-key-models"]):
            out = mod.cached_provider_model_ids("openrouter")
        assert out == ["new-key-models"]
        spawn.assert_not_called()

    def test_force_refresh_bypasses_swr(self):
        import hermes_cli.models as mod

        cache = {"openrouter": self._cache_entry(["m1"], age_seconds=7200)}
        with patch.object(mod, "_load_provider_models_cache", return_value=cache), \
             patch.object(mod, "_credential_fingerprint", return_value="fp"), \
             patch.object(mod, "_save_provider_models_cache"), \
             patch.object(mod, "_spawn_swr_refresh") as spawn, \
             patch.object(mod, "provider_model_ids", return_value=["live"]) as live:
            out = mod.cached_provider_model_ids("openrouter", force_refresh=True)
        assert out == ["live"]
        spawn.assert_not_called()
        live.assert_called_once_with("openrouter", force_refresh=True)

    def test_swr_refresh_dedupes_inflight(self):
        import hermes_cli.models as mod

        started = []

        class FakeThread:
            def __init__(self, target=None, daemon=None, name=None):
                started.append(name)
                self._target = target

            def start(self):
                pass  # never run — keeps the provider marked in-flight

        with patch.object(mod.threading, "Thread", FakeThread):
            mod._spawn_swr_refresh("openrouter")
            mod._spawn_swr_refresh("openrouter")  # deduped
            mod._spawn_swr_refresh("nous")
        assert started == ["model-cache-swr-openrouter", "model-cache-swr-nous"]

    def test_swr_refresh_writes_cache_and_clears_inflight(self):
        import hermes_cli.models as mod

        saved = {}

        def fake_save(data):
            saved.update(data)

        captured = {}

        class InlineThread:
            def __init__(self, target=None, daemon=None, name=None):
                captured["target"] = target

            def start(self):
                captured["target"]()  # run synchronously

        with patch.object(mod.threading, "Thread", InlineThread), \
             patch.object(mod, "provider_model_ids", return_value=["fresh1", "fresh2"]), \
             patch.object(mod, "_credential_fingerprint", return_value="fp"), \
             patch.object(mod, "_load_provider_models_cache", return_value={}), \
             patch.object(mod, "_save_provider_models_cache", side_effect=fake_save):
            mod._spawn_swr_refresh("openrouter")

        assert saved["openrouter"]["models"] == ["fresh1", "fresh2"]
        assert "openrouter" not in mod._swr_refresh_inflight  # cleared on completion


class TestCatalogSWR:
    def test_stale_disk_catalog_served_with_background_refresh(self, tmp_path, monkeypatch):
        import hermes_cli.model_catalog as mc

        manifest = {"version": 1, "providers": {"nous": {"models": [{"id": "hermes-4"}]}}}
        mc.reset_cache()
        with patch.object(mc, "_load_catalog_config", return_value={
                 "enabled": True, "ttl_hours": 1.0, "url": "https://example/cat.json",
                 "providers": {}}), \
             patch.object(mc, "_read_disk_cache", return_value=(manifest, time.time() - 7200)), \
             patch.object(mc, "_spawn_catalog_swr_refresh") as spawn, \
             patch.object(mc, "_fetch_manifest_with_fallback") as fetch:
            out = mc.get_catalog()
        assert out == manifest  # stale copy served without blocking
        spawn.assert_called_once()
        fetch.assert_not_called()

    def test_cold_cache_still_blocks_on_fetch(self, monkeypatch):
        import hermes_cli.model_catalog as mc

        manifest = {"version": 1, "providers": {}}
        mc.reset_cache()
        with patch.object(mc, "_load_catalog_config", return_value={
                 "enabled": True, "ttl_hours": 1.0, "url": "https://example/cat.json",
                 "providers": {}}), \
             patch.object(mc, "_read_disk_cache", return_value=(None, 0.0)), \
             patch.object(mc, "_spawn_catalog_swr_refresh") as spawn, \
             patch.object(mc, "_write_disk_cache"), \
             patch.object(mc, "_fetch_manifest_with_fallback", return_value=manifest) as fetch:
            out = mc.get_catalog()
        assert out == manifest
        fetch.assert_called_once()
        spawn.assert_not_called()


class TestCorruptCacheRowDegradation:
    """A corrupted 'at' in the user-editable provider_models_cache.json must
    degrade cached_provider_model_ids to a cache miss (live fetch), never
    raise through the picker (which has no try/except at its call sites)."""

    @pytest.mark.parametrize("bad_at", ["yesterday", None, True])
    def test_corrupt_at_falls_back_to_live_fetch(self, bad_at):
        import hermes_cli.models as mod

        cache = {"openrouter": {"fp": "fp", "at": bad_at, "models": ["corrupt-row"]}}
        with patch.object(mod, "_load_provider_models_cache", return_value=cache), \
             patch.object(mod, "_credential_fingerprint", return_value="fp"), \
             patch.object(mod, "_save_provider_models_cache"), \
             patch.object(mod, "provider_model_ids", return_value=["live-model"]) as live:
            out = mod.cached_provider_model_ids("openrouter")
        assert out == ["live-model"]
        live.assert_called_once()
