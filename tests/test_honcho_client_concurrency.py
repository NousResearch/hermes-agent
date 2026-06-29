"""Concurrency test for get_honcho_client() — the TOCTOU race fix (#24759).

Proves the Honcho client is constructed exactly once even when many threads
race the first call, by stubbing the SDK constructor and counting invocations.
"""

import sys
import threading
import types

import pytest

from plugins.memory.honcho import client as honcho_client
from plugins.memory.honcho.client import (
    HonchoClientConfig,
    get_honcho_client,
    reset_honcho_client,
)


@pytest.fixture(autouse=True)
def _reset_singleton():
    reset_honcho_client()
    yield
    reset_honcho_client()


def _install_fake_honcho_sdk(monkeypatch, build_count, build_lock):
    """Make `from honcho import Honcho` resolve to a counting fake."""

    class _FakeHoncho:
        def __init__(self, **kwargs):
            with build_lock:
                build_count["n"] += 1
            import time
            time.sleep(0.01)  # widen the race window
            self.kwargs = kwargs

    fake_mod = types.ModuleType("honcho")
    fake_mod.Honcho = _FakeHoncho
    monkeypatch.setitem(sys.modules, "honcho", fake_mod)
    # Skip the lazy-install path entirely.
    monkeypatch.setattr(
        honcho_client, "_resolve_optional_float", lambda *a, **k: None, raising=False
    )


def test_get_honcho_client_builds_once_under_concurrent_first_call(monkeypatch):
    build_count = {"n": 0}
    build_lock = threading.Lock()
    _install_fake_honcho_sdk(monkeypatch, build_count, build_lock)

    config = HonchoClientConfig(
        api_key="test-key",
        workspace_id="ws",
        environment="production",
    )

    barrier = threading.Barrier(20)
    results = []
    results_lock = threading.Lock()

    def worker():
        barrier.wait()
        c = get_honcho_client(config)
        with results_lock:
            results.append(c)

    threads = [threading.Thread(target=worker) for _ in range(20)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert build_count["n"] == 1, "Honcho client must be constructed exactly once"
    assert len(results) == 20
    assert all(r is results[0] for r in results), "all threads share one client"


def test_reset_allows_rebuild(monkeypatch):
    build_count = {"n": 0}
    build_lock = threading.Lock()
    _install_fake_honcho_sdk(monkeypatch, build_count, build_lock)

    config = HonchoClientConfig(
        api_key="test-key", workspace_id="ws", environment="production"
    )

    c1 = get_honcho_client(config)
    assert build_count["n"] == 1
    # Cached: no rebuild.
    assert get_honcho_client(config) is c1
    assert build_count["n"] == 1

    reset_honcho_client()
    c2 = get_honcho_client(config)
    assert build_count["n"] == 2
    assert c2 is not c1


def test_distinct_compartment_configs_get_distinct_cached_clients(monkeypatch):
    """Ops and personal compartments must not share the first-built client."""
    build_count = {"n": 0}
    build_lock = threading.Lock()
    _install_fake_honcho_sdk(monkeypatch, build_count, build_lock)

    ops = HonchoClientConfig(
        host="hermes:ops",
        api_key="ops-key",
        workspace_id="ops-prod",
        base_url="https://honcho.example.test",
        api_key_explicit=True,
    )
    personal = HonchoClientConfig(
        host="hermes:personal",
        api_key="personal-key",
        workspace_id="personal-prod",
        base_url="https://honcho.example.test",
        api_key_explicit=True,
    )

    ops_client = get_honcho_client(ops)
    personal_client = get_honcho_client(personal)

    assert build_count["n"] == 2
    assert personal_client is not ops_client
    assert ops_client.kwargs["workspace_id"] == "ops-prod"
    assert personal_client.kwargs["workspace_id"] == "personal-prod"
    assert get_honcho_client(ops) is ops_client
    assert get_honcho_client(personal) is personal_client


def test_missing_credentials_still_raises_before_build(monkeypatch):
    build_count = {"n": 0}
    build_lock = threading.Lock()
    _install_fake_honcho_sdk(monkeypatch, build_count, build_lock)

    bad = HonchoClientConfig(api_key="", base_url="", workspace_id="ws")
    with pytest.raises(ValueError):
        get_honcho_client(bad)
    assert build_count["n"] == 0
