"""Tests for the process-level backend cache in the mem0 provider.

Qdrant local-path storage refuses a second QdrantClient on the same folder
within one process. Long-lived hosts (desktop serve, gateway) re-initialize
memory providers per session, so without a process-level cache every new
session would trip that single-instance guard. These tests pin the caching
behaviour: same config → shared backend, shutdown must not close shared
instances, and distinct configs must not collide.
"""

import pytest

import plugins.memory.mem0 as mem0_plugin
import plugins.memory.mem0._backend as mem0_backend_mod
from plugins.memory.mem0 import Mem0MemoryProvider


class _FakeOSSBackend:
    """Minimal stand-in for OSSBackend with a close() marker."""

    instances = []

    def __init__(self, cfg):
        self.cfg = cfg
        self.closed = False
        _FakeOSSBackend.instances.append(self)

    def close(self):
        self.closed = True


@pytest.fixture(autouse=True)
def _isolate_cache(monkeypatch):
    """Give each test a pristine cache and fake OSS backend construction."""
    monkeypatch.setattr(mem0_plugin, "_BACKEND_CACHE", {})
    _FakeOSSBackend.instances = []
    monkeypatch.setattr(mem0_backend_mod, "OSSBackend", _FakeOSSBackend)


def _provider_with_oss(path):
    """Provider pre-configured for OSS mode with a given vector store path."""
    prov = Mem0MemoryProvider()
    prov._mode = "oss"
    prov._config = {
        "oss": {
            "vector_store": {"provider": "qdrant", "config": {"path": path}},
        }
    }
    return prov


def test_same_config_shares_backend():
    p1 = _provider_with_oss("~/.hermes/mem0_qdrant")
    p2 = _provider_with_oss("~/.hermes/mem0_qdrant")

    b1 = p1._create_backend()[0]
    b2 = p2._create_backend()[0]

    assert b1 is b2, "same OSS config must share one backend per process"
    assert len(mem0_plugin._BACKEND_CACHE) == 1


def test_shutdown_does_not_close_shared_backend():
    p1 = _provider_with_oss("~/.hermes/mem0_qdrant")
    p2 = _provider_with_oss("~/.hermes/mem0_qdrant")
    b1 = p1._create_backend()[0]
    p2._backend = b1

    p2.shutdown()

    assert p2._backend is None, "session shutdown releases the provider's ref"
    assert not b1.closed, "shared backend must survive per-session shutdown"
    assert len(mem0_plugin._BACKEND_CACHE) == 1
    # A third session in the same process still reuses the live instance.
    p3 = _provider_with_oss("~/.hermes/mem0_qdrant")
    assert p3._create_backend()[0] is b1


def test_distinct_configs_get_distinct_backends():
    p1 = _provider_with_oss("~/.hermes/mem0_qdrant")
    p2 = _provider_with_oss("~/.hermes/mem0_qdrant_other")

    b1 = p1._create_backend()[0]
    b2 = p2._create_backend()[0]

    assert b1 is not b2
    assert len(mem0_plugin._BACKEND_CACHE) == 2


def test_close_cached_backends_closes_all():
    p1 = _provider_with_oss("~/.hermes/mem0_qdrant")
    p1._create_backend()
    mem0_plugin._BACKEND_CACHE[("oss", "extra")] = _FakeOSSBackend({})

    mem0_plugin._close_cached_backends()

    assert mem0_plugin._BACKEND_CACHE == {}
    assert all(b.closed for b in _FakeOSSBackend.instances)


def test_normalized_cache_key_collapses_equivalent_paths():
    """Semantically identical vector-store paths must share one backend.

    ``~`` vs expanded home, redundant ``./``, and trailing-slash spellings of
    the same folder must not produce two OSSBackend instances on the same
    Qdrant path (which would recreate the single-instance-lock failure).
    """
    p1 = _provider_with_oss("~/.hermes/mem0_qdrant")
    p2 = _provider_with_oss(
        __import__("os").path.join(
            __import__("os").path.expanduser("~"), ".hermes", ".", "mem0_qdrant"
        )
    )

    b1 = p1._create_backend()[0]
    b2 = p2._create_backend()[0]

    assert b1 is b2, "equivalent path spellings must share one backend"
    assert len(mem0_plugin._BACKEND_CACHE) == 1
