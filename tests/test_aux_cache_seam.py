"""Seam + behavior regression tests for slice R4-C2 (auxiliary_client.py client
cache extraction into agent/aux_client_cache.py).

The extraction moved lines 6933-7306 of the godfile into
``agent.aux_client_cache``; ``agent.auxiliary_client`` now re-exports every
moved name.  These tests pin the seam: every re-exported name must be
``is``-identical to the module that owns it, and the cache behavior must
survive the move (store/get/evict/shutdown through the seam).
"""

import threading

import pytest

from agent import aux_client_cache as cache_mod
from agent import auxiliary_client as aux

REEXPORTED_NAMES = (
    "_CallableCacheDiscriminator",
    "_runtime_cache_discriminator",
    "_client_cache_key",
    "_store_cached_client",
    "_refresh_nous_auxiliary_client",
    "neuter_async_httpx_del",
    "_force_close_async_httpx",
    "_close_cached_client",
    "shutdown_cached_clients",
    "cleanup_stale_async_clients",
    "_is_openrouter_client",
    "_cached_client_accepts_slash_models",
    "_compat_model",
    "_get_cached_client",
    "_client_cache",
    "_client_cache_lock",
    "_CLIENT_CACHE_MAX_SIZE",
)


@pytest.fixture(autouse=True)
def _clean_cache():
    """Isolate each test from cached clients/state."""
    with aux._client_cache_lock:
        aux._client_cache.clear()
    yield
    with aux._client_cache_lock:
        aux._client_cache.clear()


def test_reexported_names_are_identical():
    """Every re-exported name must resolve is-identical through the seam."""
    for name in REEXPORTED_NAMES:
        assert getattr(aux, name) is getattr(cache_mod, name), name


def test_owned_in_aux_client_cache_module():
    """The moved definitions live in the new module, not the godfile."""
    assert cache_mod._client_cache is aux._client_cache
    assert cache_mod._client_cache_lock is aux._client_cache_lock
    assert cache_mod._CLIENT_CACHE_MAX_SIZE == 64
    assert isinstance(aux._client_cache_lock, type(threading.Lock()))


def test_module_state_shared_single_dict():
    """The godfile and the extracted module share one cache dict."""
    cache_mod._client_cache["shared-sentinel"] = ("x", "m", None)
    try:
        assert aux._client_cache["shared-sentinel"] == ("x", "m", None)
    finally:
        del aux._client_cache["shared-sentinel"]


class TestCacheBehaviorThroughSeam:
    def test_store_then_get_hit(self):
        key = aux._client_cache_key(
            "custom",
            async_mode=False,
            base_url="http://cache-seam.test/v1",
            api_key="seam-key",
            model="seam-model",
        )
        aux._store_cached_client(key, "client-a", "seam-model")
        client, model = aux._get_cached_client(
            "custom",
            "seam-model",
            base_url="http://cache-seam.test/v1",
            api_key="seam-key",
        )
        assert client == "client-a"
        assert model == "seam-model"

    def test_store_evicts_old_client_on_replace(self):
        key = aux._client_cache_key(
            "custom",
            async_mode=False,
            base_url="http://cache-seam.test/v1",
            api_key="seam-key",
            model="seam-model",
        )
        closed = []
        old = type("Old", (), {"close": lambda self: closed.append(True)})()
        aux._store_cached_client(key, old, "seam-model")
        aux._store_cached_client(key, "client-b", "seam-model")
        assert closed == [True]
        assert aux._client_cache[key][0] == "client-b"

    def test_max_size_evicts_oldest(self, monkeypatch):
        """The cache-miss build path enforces the FIFO size cap (#10200 belt)."""
        base = "http://cache-seam.test/v1"
        # Seed the cache to the cap with distinct keys.
        for i in range(aux._CLIENT_CACHE_MAX_SIZE):
            key = aux._client_cache_key(
                "custom",
                async_mode=False,
                base_url=base,
                api_key=f"key-{i}",
                model=f"model-{i}",
            )
            with aux._client_cache_lock:
                aux._client_cache[key] = (f"client-{i}", f"model-{i}", None)
        first_key = next(iter(aux._client_cache))
        assert len(aux._client_cache) == aux._CLIENT_CACHE_MAX_SIZE

        # A new-key miss goes through the build path, which evicts the
        # oldest entry before inserting (FIFO — dict insertion order).
        monkeypatch.setattr(
            "agent.auxiliary_client.resolve_provider_client",
            lambda *a, **kw: ("overflow-client", "overflow-model"),
        )
        client, model = aux._get_cached_client(
            "custom",
            "overflow-model",
            base_url=base,
            api_key="overflow-key",
        )
        assert client == "overflow-client"
        assert len(aux._client_cache) == aux._CLIENT_CACHE_MAX_SIZE
        assert first_key not in aux._client_cache

    def test_shutdown_clears_cache(self):
        key = aux._client_cache_key(
            "custom",
            async_mode=False,
            base_url="http://cache-seam.test/v1",
            api_key="seam-key",
            model="seam-model",
        )
        with aux._client_cache_lock:
            aux._client_cache[key] = ("client-a", "seam-model", None)
        aux.shutdown_cached_clients()
        assert aux._client_cache == {}

    def test_runtime_cache_discriminator_hashes_callables_by_identity(self):
        def _cb():
            return "secret"

        first = aux._runtime_cache_discriminator("api_key", _cb)
        second = aux._runtime_cache_discriminator("api_key", _cb)
        assert first == second
        assert hash(first) == hash(second)
        assert isinstance(first, aux._CallableCacheDiscriminator)
        assert repr(first) == "<callable-api-key>"
