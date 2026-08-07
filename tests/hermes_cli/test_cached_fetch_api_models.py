"""Cache-contract tests for ``cached_fetch_api_models()``.

Custom OpenAI-compatible endpoints (named ``custom_providers`` rows, bare
``provider: custom``, and per-endpoint-map entries) previously called
``fetch_api_models()`` directly with no disk cache, so the current custom
endpoint's ``/v1/models`` got a live HTTP round-trip on literally every
``/model`` open (#72762). ``cached_fetch_api_models()`` gives custom
endpoints the same ``provider_models_cache.json`` TTL cache first-class
providers already get via ``cached_provider_model_ids()``.

These pin the cache contract directly (hit / stale / rotation / refresh /
fallback), separate from ``test_model_switch_custom_providers.py``'s
higher-level picker-shape tests.
"""

from __future__ import annotations

import time
from unittest.mock import patch

import pytest


class TestCachedFetchApiModels:
    def _entry(self, models, age_seconds, fp="fp"):
        return {"fp": fp, "at": time.time() - age_seconds, "models": list(models)}

    def test_fresh_entry_served_without_live_fetch(self):
        import hermes_cli.models as mod

        cache = {"custom:https://gw.example.com/v1": self._entry(["m1", "m2"], age_seconds=10)}
        with patch.object(mod, "_load_provider_models_cache", return_value=cache), \
             patch.object(mod, "_custom_endpoint_fingerprint", return_value="fp"), \
             patch.object(mod, "_save_provider_models_cache") as save, \
             patch.object(mod, "fetch_api_models") as live:
            out = mod.cached_fetch_api_models("sk-key", "https://gw.example.com/v1")
        assert out == ["m1", "m2"]
        live.assert_not_called()
        save.assert_not_called()

    def test_cache_key_normalizes_trailing_slash_and_case(self):
        """A saved entry for the lowercased/rstripped URL must be hit even
        when the caller passes a differently-cased URL with a trailing
        slash — config.yaml entries are not guaranteed to be normalized."""
        import hermes_cli.models as mod

        cache = {"custom:https://gw.example.com/v1": self._entry(["m1"], age_seconds=10)}
        with patch.object(mod, "_load_provider_models_cache", return_value=cache), \
             patch.object(mod, "_custom_endpoint_fingerprint", return_value="fp"), \
             patch.object(mod, "fetch_api_models") as live:
            out = mod.cached_fetch_api_models("sk-key", "HTTPS://GW.example.com/v1/")
        assert out == ["m1"]
        live.assert_not_called()

    def test_expired_entry_triggers_live_fetch_and_is_persisted(self):
        import hermes_cli.models as mod

        # Beyond the stale-serve window, so the wrapper must block on a
        # live fetch (within the window it stale-serves + refreshes off
        # thread — covered in TestSalvageFollowups).
        too_old = mod._PROVIDER_MODELS_STALE_SERVE_MAX + 60
        cache = {"custom:https://gw.example.com/v1": self._entry(["old"], age_seconds=too_old)}
        saved = {}
        with patch.object(mod, "_load_provider_models_cache", return_value=cache), \
             patch.object(mod, "_custom_endpoint_fingerprint", return_value="fp"), \
             patch.object(mod, "_save_provider_models_cache", side_effect=saved.update), \
             patch.object(mod, "fetch_api_models", return_value=["fresh-a", "fresh-b"]) as live:
            out = mod.cached_fetch_api_models(
                "sk-key", "https://gw.example.com/v1", ttl_seconds=3600
            )
        assert out == ["fresh-a", "fresh-b"]
        live.assert_called_once()
        assert saved["custom:https://gw.example.com/v1"]["models"] == ["fresh-a", "fresh-b"]
        assert saved["custom:https://gw.example.com/v1"]["fp"] == "fp"

    def test_rotated_api_key_busts_cache_even_when_fresh(self):
        """A same-age entry with a DIFFERENT fingerprint (key rotated, or
        extra_headers edited) must not be served — it reflects the old
        credentials' catalog."""
        import hermes_cli.models as mod

        cache = {"custom:https://gw.example.com/v1": self._entry(["old-key-models"], 10, fp="old-fp")}
        with patch.object(mod, "_load_provider_models_cache", return_value=cache), \
             patch.object(mod, "_custom_endpoint_fingerprint", return_value="new-fp"), \
             patch.object(mod, "_save_provider_models_cache"), \
             patch.object(mod, "fetch_api_models", return_value=["new-key-models"]) as live:
            out = mod.cached_fetch_api_models("sk-new-key", "https://gw.example.com/v1")
        assert out == ["new-key-models"]
        live.assert_called_once()

    def test_force_refresh_bypasses_fresh_cache(self):
        import hermes_cli.models as mod

        cache = {"custom:https://gw.example.com/v1": self._entry(["stale-but-fresh"], age_seconds=5)}
        with patch.object(mod, "_load_provider_models_cache", return_value=cache), \
             patch.object(mod, "_custom_endpoint_fingerprint", return_value="fp"), \
             patch.object(mod, "_save_provider_models_cache"), \
             patch.object(mod, "fetch_api_models", return_value=["forced-live"]) as live:
            out = mod.cached_fetch_api_models(
                "sk-key", "https://gw.example.com/v1", force_refresh=True
            )
        assert out == ["forced-live"]
        live.assert_called_once()

    def test_live_failure_falls_back_to_stale_same_fingerprint_entry(self):
        """Stale data beats no data when the endpoint is flaky (#72762
        proposed-fix: 'same stale-beats-nothing fallback as
        cached_provider_model_ids')."""
        import hermes_cli.models as mod

        cache = {"custom:https://gw.example.com/v1": self._entry(["last-known-good"], age_seconds=99999, fp="fp")}
        with patch.object(mod, "_load_provider_models_cache", return_value=cache), \
             patch.object(mod, "_custom_endpoint_fingerprint", return_value="fp"), \
             patch.object(mod, "_save_provider_models_cache") as save, \
             patch.object(mod, "fetch_api_models", return_value=None):
            out = mod.cached_fetch_api_models("sk-key", "https://gw.example.com/v1")
        assert out == ["last-known-good"]
        save.assert_not_called()  # nothing new to persist

    def test_live_failure_with_no_matching_entry_returns_live_value(self):
        import hermes_cli.models as mod

        with patch.object(mod, "_load_provider_models_cache", return_value={}), \
             patch.object(mod, "_custom_endpoint_fingerprint", return_value="fp"), \
             patch.object(mod, "_save_provider_models_cache") as save, \
             patch.object(mod, "fetch_api_models", return_value=None):
            out = mod.cached_fetch_api_models("sk-key", "https://gw.example.com/v1")
        assert out is None
        save.assert_not_called()

    def test_empty_live_result_is_not_persisted(self):
        """An empty list from a transient error must never pin an empty
        cache entry over real data on the next open."""
        import hermes_cli.models as mod

        with patch.object(mod, "_load_provider_models_cache", return_value={}), \
             patch.object(mod, "_custom_endpoint_fingerprint", return_value="fp"), \
             patch.object(mod, "_save_provider_models_cache") as save, \
             patch.object(mod, "fetch_api_models", return_value=[]):
            out = mod.cached_fetch_api_models("sk-key", "https://gw.example.com/v1")
        assert out == []
        save.assert_not_called()

    def test_blank_base_url_skips_cache_entirely(self):
        """No base_url means nothing to key the cache on — call straight
        through to fetch_api_models rather than caching under an empty key."""
        import hermes_cli.models as mod

        with patch.object(mod, "_load_provider_models_cache") as load, \
             patch.object(mod, "fetch_api_models", return_value=["x"]) as live:
            out = mod.cached_fetch_api_models("sk-key", "")
        assert out == ["x"]
        live.assert_called_once()
        load.assert_not_called()

    def test_fingerprint_ignores_timeout_but_reacts_to_headers(self):
        """Sanity check on the real (non-mocked) fingerprint helper: it must
        not vary with call-only params like timeout, but must vary with the
        actual credential/header inputs."""
        import hermes_cli.models as mod

        fp_a = mod._custom_endpoint_fingerprint("sk-key", None, {"X-Tenant": "a"})
        fp_b = mod._custom_endpoint_fingerprint("sk-key", None, {"X-Tenant": "b"})
        fp_a_again = mod._custom_endpoint_fingerprint("sk-key", None, {"X-Tenant": "a"})
        assert fp_a != fp_b
        assert fp_a == fp_a_again


class TestCachedFetchApiModelsDiskRoundTrip:
    """End-to-end through the real (per-test-isolated) provider_models_cache.json
    disk file rather than mocked load/save, so a regression in the on-disk
    schema (e.g. a key collision with provider-slug entries) would show up
    here even if the mocked unit tests above stayed green."""

    def test_second_call_within_ttl_hits_disk_cache_no_live_fetch(self, monkeypatch):
        import hermes_cli.models as mod

        calls = []

        def fake_fetch(api_key, base_url, **kwargs):
            calls.append((api_key, base_url))
            return ["disk-cached-model"]

        monkeypatch.setattr(mod, "fetch_api_models", fake_fetch)

        first = mod.cached_fetch_api_models("sk-key", "https://gw.example.com/v1")
        second = mod.cached_fetch_api_models("sk-key", "https://gw.example.com/v1")

        assert first == ["disk-cached-model"]
        assert second == ["disk-cached-model"]
        assert len(calls) == 1, "second open must be served from disk, not a fresh live fetch"

    def test_custom_key_does_not_collide_with_provider_slug_cache(self, monkeypatch):
        """A custom endpoint literally named e.g. 'openrouter' in its
        base_url must not read/write the same cache slot as the first-class
        'openrouter' provider slug used by cached_provider_model_ids()."""
        import hermes_cli.models as mod

        monkeypatch.setattr(
            mod, "fetch_api_models", lambda *a, **k: ["custom-endpoint-model"]
        )
        monkeypatch.setattr(
            mod, "provider_model_ids", lambda *a, **k: ["openrouter-curated-model"]
        )

        mod.cached_fetch_api_models("sk-key", "https://openrouter.ai/v1")
        mod.cached_provider_model_ids("openrouter")

        cache = mod._load_provider_models_cache()
        assert cache["custom:https://openrouter.ai/v1"]["models"] == ["custom-endpoint-model"]
        assert cache["openrouter"]["models"] == ["openrouter-curated-model"]


class TestSalvageFollowups:
    """Follow-up behaviors added while salvaging PR #80740: SWR stale-serve
    parity with cached_provider_model_ids, and corrupt-cache degradation."""

    def _entry(self, models, age_seconds, fp="fp"):
        return {"fp": fp, "at": time.time() - age_seconds, "models": list(models)}

    def test_expired_entry_within_stale_window_is_served_and_refreshed_off_thread(self):
        """TTL-expired (but < stale-serve max) entries must be served
        immediately — the picker never blocks on a live round-trip — while a
        background refresh is spawned for the next open."""
        import hermes_cli.models as mod

        cache = {"custom:https://gw.example.com/v1": self._entry(["stale-ok"], age_seconds=7200)}
        with patch.object(mod, "_load_provider_models_cache", return_value=cache), \
             patch.object(mod, "_custom_endpoint_fingerprint", return_value="fp"), \
             patch.object(mod, "_spawn_swr_refresh") as spawn, \
             patch.object(mod, "fetch_api_models") as live:
            out = mod.cached_fetch_api_models(
                "sk-key", "https://gw.example.com/v1", ttl_seconds=3600
            )
        assert out == ["stale-ok"]
        live.assert_not_called()
        spawn.assert_called_once()
        assert spawn.call_args[0][0] == "custom:https://gw.example.com/v1"

    def test_entry_beyond_stale_window_blocks_on_live_fetch(self):
        import hermes_cli.models as mod

        too_old = mod._PROVIDER_MODELS_STALE_SERVE_MAX + 60
        cache = {"custom:https://gw.example.com/v1": self._entry(["ancient"], age_seconds=too_old)}
        with patch.object(mod, "_load_provider_models_cache", return_value=cache), \
             patch.object(mod, "_custom_endpoint_fingerprint", return_value="fp"), \
             patch.object(mod, "_save_provider_models_cache"), \
             patch.object(mod, "_spawn_swr_refresh") as spawn, \
             patch.object(mod, "fetch_api_models", return_value=["fresh"]) as live:
            out = mod.cached_fetch_api_models("sk-key", "https://gw.example.com/v1")
        assert out == ["fresh"]
        live.assert_called_once()
        spawn.assert_not_called()

    def test_swr_refresh_fn_writes_custom_entry_through_shared_scaffolding(self):
        """The generalized _spawn_swr_refresh(key, refresh_fn) must persist a
        custom-endpoint entry via the shared inflight-dedupe machinery."""
        import threading as _threading

        import hermes_cli.models as mod

        done = _threading.Event()
        saved = {}

        def fake_save(data):
            saved.update(data)
            done.set()

        with patch.object(mod, "_load_provider_models_cache", return_value={}), \
             patch.object(mod, "_save_provider_models_cache", side_effect=fake_save):
            mod._spawn_swr_refresh(
                "custom:https://gw.example.com/v1",
                lambda: {"fp": "fp", "at": time.time(), "models": ["refreshed"]},
            )
            assert done.wait(timeout=5), "background refresh did not complete"
        assert saved["custom:https://gw.example.com/v1"]["models"] == ["refreshed"]
        assert "custom:https://gw.example.com/v1" not in mod._swr_refresh_inflight

    def test_corrupt_at_field_degrades_to_live_fetch_instead_of_raising(self):
        """provider_models_cache.json is user-editable; a corrupted 'at' must
        be a cache miss (live fetch), never an exception out of the wrapper."""
        import hermes_cli.models as mod

        cache = {
            "custom:https://gw.example.com/v1": {
                "fp": "fp", "at": "yesterday", "models": ["corrupt-row"],
            }
        }
        with patch.object(mod, "_load_provider_models_cache", return_value=cache), \
             patch.object(mod, "_custom_endpoint_fingerprint", return_value="fp"), \
             patch.object(mod, "_save_provider_models_cache"), \
             patch.object(mod, "fetch_api_models", return_value=["live-models"]) as live:
            out = mod.cached_fetch_api_models("sk-key", "https://gw.example.com/v1")
        assert out == ["live-models"]
        live.assert_called_once()
