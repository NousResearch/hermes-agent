"""Tests for parallel model-catalog prefetch and thread-safe cache writes.

Regression tests for the serial /v1/models bottleneck: when the 1h disk cache
lapses, ``list_authenticated_providers()`` previously fetched each authed
provider's model list serially. With 10+ providers this stacked to 15-30s of
blocking HTTP round-trips. The parallel prefetch warms stale cache entries
concurrently via ThreadPoolExecutor before the serial picker loop starts.
"""

from __future__ import annotations

import time
from unittest.mock import patch

import pytest

# ---------------------------------------------------------------------------
# Thread-safe cache entry update (hermes_cli/models.py)
# ---------------------------------------------------------------------------

class TestStoreCacheEntry:
    """Verify ``_store_cache_entry`` writes safely under concurrency."""

    def test_writes_new_entry(self, tmp_path, monkeypatch):
        """A new entry is persisted to the cache file."""
        import hermes_cli.models as mod

        cache_path = tmp_path / "provider_models_cache.json"
        monkeypatch.setattr(mod, "_provider_models_cache_path", lambda: cache_path)

        mod._store_cache_entry("openrouter", mod._cache_entry("fp1", ["m1", "m2"]))

        cache = mod._load_provider_models_cache()
        assert "openrouter" in cache
        assert cache["openrouter"]["models"] == ["m1", "m2"]
        assert cache["openrouter"]["fp"] == "fp1"

    def test_does_not_clobber_other_entries(self, tmp_path, monkeypatch):
        """Concurrent writes to different providers don't lose entries."""
        import hermes_cli.models as mod

        cache_path = tmp_path / "provider_models_cache.json"
        monkeypatch.setattr(mod, "_provider_models_cache_path", lambda: cache_path)

        # Seed with one entry
        mod._store_cache_entry("provider_a", mod._cache_entry("fp_a", ["a1"]))

        # Write a second entry
        mod._store_cache_entry("provider_b", mod._cache_entry("fp_b", ["b1"]))

        cache = mod._load_provider_models_cache()
        assert "provider_a" in cache
        assert cache["provider_a"]["models"] == ["a1"]
        assert "provider_b" in cache
        assert cache["provider_b"]["models"] == ["b1"]

    def test_concurrent_writes_no_lost_entries(self, tmp_path, monkeypatch):
        """Multiple threads writing different providers concurrently — all land."""
        import hermes_cli.models as mod
        import concurrent.futures

        cache_path = tmp_path / "provider_models_cache.json"
        monkeypatch.setattr(mod, "_provider_models_cache_path", lambda: cache_path)

        providers = [f"prov_{i}" for i in range(10)]

        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            list(executor.map(
                lambda p: mod._store_cache_entry(p, mod._cache_entry(f"fp_{p}", [f"model_{p}"])),
                providers,
            ))

        cache = mod._load_provider_models_cache()
        for p in providers:
            assert p in cache, f"{p} was lost in concurrent write"
            assert cache[p]["models"] == [f"model_{p}"]

    @pytest.mark.parametrize("cleared", ["openrouter", None])
    def test_clear_is_not_undone_by_a_concurrent_locked_write(self, tmp_path, monkeypatch, cleared):
        """``--refresh`` clears under the same lock: a writer that loaded the cache before the clear
        must not save its snapshot afterwards and bring the cleared row back."""
        import threading

        import hermes_cli.models as mod
        from hermes_cli.models_cache_lock import write_under_cache_lock

        import hermes_cli.models_cache_lock as cache_lock_module
        from tests.lock_contention import ContentionObservedLock, watch_file_lock_contention
        from tests.waiting import wait_until

        cache_path = tmp_path / "provider_models_cache.json"
        monkeypatch.setattr(mod, "_provider_models_cache_path", lambda: cache_path)
        contended = watch_file_lock_contention(monkeypatch)
        # In-process writers meet the thread lock first, other processes the file lock.
        thread_lock = ContentionObservedLock()
        monkeypatch.setattr(cache_lock_module, "_thread_lock", thread_lock)
        mod._save_provider_models_cache({
            "openrouter": mod._cache_entry("fp", ["or/model"]),
            "anthropic": mod._cache_entry("fp", ["claude"]),
        })
        loaded = threading.Event()
        release = threading.Event()

        def slow_write():  # _store_cache_entry's write, paused between its read and its save
            cache = mod._load_provider_models_cache()
            loaded.set()
            release.wait(5)
            cache["anthropic"] = mod._cache_entry("fp", ["claude-new"])
            mod._save_provider_models_cache(cache)

        writer = threading.Thread(target=write_under_cache_lock, args=(cache_path, slow_write))
        writer.start()
        assert loaded.wait(5)
        clearer = threading.Thread(target=mod.clear_provider_models_cache, args=(cleared,))
        clearer.start()
        wait_until(lambda: not clearer.is_alive() or contended.is_set() or thread_lock.contended.is_set(),
                   what="the clear finished or blocked on the cache lock")
        release.set()
        writer.join(5)
        clearer.join(5)

        assert "openrouter" not in mod._load_provider_models_cache()

# ---------------------------------------------------------------------------
# Parallel prefetch (hermes_cli/model_switch.py)
# ---------------------------------------------------------------------------

class TestPrefetchProviderModelsParallel:
    """Verify ``_prefetch_provider_models_parallel`` fetches concurrently."""

    def test_skips_all_fresh_entries(self, monkeypatch):
        """When all cache entries are fresh, no fetch is made."""
        from hermes_cli.model_switch_providers import _prefetch_provider_models_parallel

        fresh_cache = {
            "openrouter": {"fp": "fp", "at": time.time(), "models": ["m1"]},
            "anthropic": {"fp": "fp", "at": time.time(), "models": ["m2"]},
        }

        with patch("hermes_cli.models._load_provider_models_cache", return_value=fresh_cache), \
             patch("hermes_cli.models._credential_fingerprint", return_value="fp"), \
             patch("hermes_cli.models.cached_provider_model_ids") as fetch:
            _prefetch_provider_models_parallel(["openrouter", "anthropic"])

        fetch.assert_not_called()

    def test_fetches_only_stale_entries(self, monkeypatch):
        """Only providers with stale/missing cache entries are fetched."""
        from hermes_cli.model_switch_providers import _prefetch_provider_models_parallel

        cache = {
            "fresh_prov": {"fp": "fp_f", "at": time.time(), "models": ["m1"]},
        }

        fetch_calls = []

        def mock_fetch(slug, force_refresh=False):
            fetch_calls.append(slug)
            return [f"model_{slug}"]

        with patch("hermes_cli.models._load_provider_models_cache", return_value=cache), \
             patch("hermes_cli.models._credential_fingerprint", return_value="fp_f"), \
             patch("hermes_cli.models.cached_provider_model_ids", side_effect=mock_fetch):
            _prefetch_provider_models_parallel(["fresh_prov", "stale_prov"])

        assert "fresh_prov" not in fetch_calls
        assert "stale_prov" in fetch_calls

    def test_fetches_in_parallel(self, monkeypatch):
        """Multiple providers are fetched concurrently, not serially."""
        from hermes_cli.model_switch_providers import _prefetch_provider_models_parallel

        # Track overlap: if serial, no two fetches should overlap in time.
        active = []
        max_concurrent = [0]
        lock = __import__("threading").Lock()

        def mock_fetch(slug, force_refresh=False):
            with lock:
                active.append(slug)
                max_concurrent[0] = max(max_concurrent[0], len(active))
            time.sleep(0.05)  # simulate network latency
            with lock:
                active.remove(slug)
            return [f"model_{slug}"]

        slugs = [f"prov_{i}" for i in range(6)]

        with patch("hermes_cli.models._load_provider_models_cache", return_value={}), \
             patch("hermes_cli.models._credential_fingerprint", return_value="fp"), \
             patch("hermes_cli.models.cached_provider_model_ids", side_effect=mock_fetch):
            _prefetch_provider_models_parallel(slugs)

        assert max_concurrent[0] > 1, "fetches were serial, not parallel"

    def test_swallows_exceptions(self):
        """A failing provider fetch doesn't raise — best-effort."""
        from hermes_cli.model_switch_providers import _prefetch_provider_models_parallel

        def mock_fetch(slug, force_refresh=False):
            raise ConnectionError("simulated network failure")

        with patch("hermes_cli.models._load_provider_models_cache", return_value={}), \
             patch("hermes_cli.models._credential_fingerprint", return_value="fp"), \
             patch("hermes_cli.models.cached_provider_model_ids", side_effect=mock_fetch):
            # Should not raise
            _prefetch_provider_models_parallel(["failing_prov"])

    def test_empty_list_is_noop(self):
        """Empty provider list does nothing."""
        from hermes_cli.model_switch_providers import _prefetch_provider_models_parallel

        with patch("hermes_cli.models.cached_provider_model_ids") as fetch:
            _prefetch_provider_models_parallel([])
        fetch.assert_not_called()

    def test_skips_ttl_expired_entries_the_serial_path_can_still_serve(self):
        """A TTL-expired entry inside the stale-serve window is not prefetched.

        ``cached_provider_model_ids`` returns such an entry from disk right
        away and revalidates on a background thread, so blocking the picker
        on a parallel fetch buys nothing. ``_PROVIDER_MODELS_STALE_SERVE_MAX``
        is far longer than ``_PROVIDER_MODELS_CACHE_TTL``, so this is the
        state every picker open a TTL after the previous one lands in.
        """
        import hermes_cli.models as models_mod
        from hermes_cli.model_switch_providers import _prefetch_provider_models_parallel

        expired = time.time() - models_mod._PROVIDER_MODELS_CACHE_TTL - 60
        cache = {"openrouter": {"fp": "fp", "at": expired, "models": ["m1"]}}

        with patch("hermes_cli.models._load_provider_models_cache", return_value=cache), \
             patch("hermes_cli.models._credential_fingerprint", return_value="fp"), \
             patch("hermes_cli.models.cached_provider_model_ids") as fetch:
            _prefetch_provider_models_parallel(["openrouter"])

        fetch.assert_not_called()

    def test_fetches_curated_fallback_rows_past_their_short_ttl(self):
        """A curated-fallback row is served only for ``_PROVIDER_MODELS_FALLBACK_TTL``
        and never through the stale window, so the serial call blocks on it and the
        parallel prefetch must fetch it."""
        from hermes_cli.model_switch_providers import _prefetch_provider_models_parallel

        cache = {"openrouter": {"fp": "fp", "at": time.time() - 7200, "models": ["m1"],
                                "fallback": True}}
        fetched = []

        def mock_fetch(slug, force_refresh=False):
            fetched.append(slug)
            return ["m1"]

        with patch("hermes_cli.models._load_provider_models_cache", return_value=cache), \
             patch("hermes_cli.models._credential_fingerprint", return_value="fp"), \
             patch("hermes_cli.models.cached_provider_model_ids", side_effect=mock_fetch):
            _prefetch_provider_models_parallel(["openrouter"])

        assert fetched == ["openrouter"]

    def test_slow_worker_does_not_drop_rows_persisted_during_its_fetch(self, tmp_path, monkeypatch):
        """A worker whose fetch outlasts another's must not write back its pre-fetch snapshot."""
        import threading

        import hermes_cli.models as mod
        from hermes_cli.model_switch_providers import _prefetch_provider_models_parallel

        monkeypatch.setattr(mod, "_provider_models_cache_path", lambda: tmp_path / "provider_models_cache.json")
        monkeypatch.setattr(mod, "_credential_fingerprint", lambda provider: "fp")
        fast_persisted = threading.Event()

        def fake_provider_model_ids(slug, force_refresh=False):
            if slug != "openrouter":
                # The slow provider answers only after the fast one fully persisted.
                assert fast_persisted.wait(10)
            return [f"{slug}/model"]

        real_store = mod._store_cache_entry

        def store_then_signal(cache_key, entry):
            real_store(cache_key, entry)
            if cache_key == "openrouter":
                fast_persisted.set()

        monkeypatch.setattr(mod, "provider_model_ids", fake_provider_model_ids)
        # Observe the canonical write: the slow worker's own store must re-read past the fast one's.
        monkeypatch.setattr(mod, "_store_cache_entry", store_then_signal)
        _prefetch_provider_models_parallel(["openrouter", "anthropic"])

        assert sorted(mod._load_provider_models_cache()) == ["anthropic", "openrouter"]

    def test_failed_live_fetch_leaves_the_stale_row_untouched(self, tmp_path, monkeypatch):
        """A prefetch whose live fetch returns nothing serves the old rows but must not re-stamp
        them: the stale row stays stale so the next open retries instead of trusting it for a TTL."""
        import hermes_cli.models as mod
        from hermes_cli.model_switch_providers import _prefetch_provider_models_parallel

        monkeypatch.setattr(mod, "_provider_models_cache_path", lambda: tmp_path / "provider_models_cache.json")
        monkeypatch.setattr(mod, "_credential_fingerprint", lambda provider: "fp")
        monkeypatch.setattr(mod, "provider_model_ids", lambda slug, force_refresh=False: [])
        stale_at = time.time() - mod._PROVIDER_MODELS_STALE_SERVE_MAX - 3600
        mod._save_provider_models_cache({"openrouter": {"fp": "fp", "at": stale_at, "models": ["openrouter/model"]}})

        _prefetch_provider_models_parallel(["openrouter"])

        assert mod._load_provider_models_cache()["openrouter"]["at"] == stale_at
