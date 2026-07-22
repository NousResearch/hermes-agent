"""Per-profile client isolation for get_honcho_client() under multiplexing.

A multiplexed gateway (``gateway.multiplex_profiles``) serves many profiles
from one process: each turn enters ``_profile_runtime_scope``, which points
``get_hermes_home()`` at the routed profile's home, so profile-local
honcho.json files (and their workspaces) resolve per turn. The client cache
must key on that identity — with a single first-config-wins slot, whichever
profile built first pinned its ``workspace_id`` for every other profile's
memory, silently merging tenants into one workspace (#69123).

These tests pin the per-identity cache: distinct workspaces get distinct
clients (explicit configs and ambient HERMES_HOME-override resolution alike),
while same-identity callers keep sharing one build.
"""

import json
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
def _reset_cache():
    reset_honcho_client()
    yield
    reset_honcho_client()


def _install_fake_honcho_sdk(monkeypatch, build_count, build_lock):
    """Make `from honcho import Honcho` resolve to a counting fake."""

    class _FakeHoncho:
        def __init__(self, **kwargs):
            with build_lock:
                build_count["n"] += 1
            self.kwargs = kwargs

    fake_mod = types.ModuleType("honcho")
    fake_mod.Honcho = _FakeHoncho
    monkeypatch.setitem(sys.modules, "honcho", fake_mod)
    # Pin the timeout axis to the default so identity keying is the only
    # variable under test.
    monkeypatch.setattr(
        honcho_client, "_resolve_optional_float", lambda *a, **k: None, raising=False
    )


def test_distinct_workspace_configs_get_distinct_clients(monkeypatch):
    build_count = {"n": 0}
    build_lock = threading.Lock()
    _install_fake_honcho_sdk(monkeypatch, build_count, build_lock)

    config_a = HonchoClientConfig(
        api_key="test-key", workspace_id="brand-a", environment="production"
    )
    config_b = HonchoClientConfig(
        api_key="test-key", workspace_id="brand-b", environment="production"
    )

    client_a = get_honcho_client(config_a)
    client_b = get_honcho_client(config_b)

    assert client_a is not client_b, "distinct workspaces must not share a client"
    assert client_a.kwargs["workspace_id"] == "brand-a"
    assert client_b.kwargs["workspace_id"] == "brand-b"
    assert build_count["n"] == 2

    # Same identity keeps sharing one build.
    assert get_honcho_client(config_a) is client_a
    assert get_honcho_client(config_b) is client_b
    assert build_count["n"] == 2


def test_concurrent_distinct_configs_build_once_each(monkeypatch):
    build_count = {"n": 0}
    build_lock = threading.Lock()
    _install_fake_honcho_sdk(monkeypatch, build_count, build_lock)

    configs = [
        HonchoClientConfig(
            api_key="test-key", workspace_id=f"ws-{i}", environment="production"
        )
        for i in range(2)
    ]

    barrier = threading.Barrier(20)
    results: dict[str, list] = {"ws-0": [], "ws-1": []}
    results_lock = threading.Lock()

    def worker(config):
        barrier.wait()
        c = get_honcho_client(config)
        with results_lock:
            results[config.workspace_id].append(c)

    threads = [
        threading.Thread(target=worker, args=(configs[i % 2],)) for i in range(20)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert build_count["n"] == 2, "one build per identity, even under races"
    for workspace_id, clients in results.items():
        assert len(clients) == 10
        assert all(c is clients[0] for c in clients), (
            f"all callers of {workspace_id} share one client"
        )
        assert clients[0].kwargs["workspace_id"] == workspace_id


def _write_profile_home(tmp_path, name, workspace):
    home = tmp_path / name
    home.mkdir()
    (home / "honcho.json").write_text(
        json.dumps({"workspace": workspace, "apiKey": "test-key"}),
        encoding="utf-8",
    )
    return home


def test_ambient_resolution_follows_hermes_home_override(monkeypatch, tmp_path):
    """The multiplex scenario: config=None resolved under per-turn home overrides.

    ``HonchoSessionManager.honcho`` calls ``get_honcho_client()`` with no
    config on every memory access; under the multiplexer the ambient
    honcho.json is profile-local. Two profiles with different workspaces
    must get two clients — not whichever built first.
    """
    from hermes_constants import reset_hermes_home_override, set_hermes_home_override

    build_count = {"n": 0}
    build_lock = threading.Lock()
    _install_fake_honcho_sdk(monkeypatch, build_count, build_lock)
    monkeypatch.delenv("HERMES_HONCHO_HOST", raising=False)
    monkeypatch.delenv("HONCHO_TIMEOUT", raising=False)

    home_a = _write_profile_home(tmp_path, "profile-a", "brand-a")
    home_b = _write_profile_home(tmp_path, "profile-b", "brand-b")

    def under_home(home):
        token = set_hermes_home_override(home)
        try:
            return get_honcho_client()
        finally:
            reset_hermes_home_override(token)

    client_a = under_home(home_a)
    client_b = under_home(home_b)

    assert client_a is not client_b, "profiles must not share the first-built client"
    assert client_a.kwargs["workspace_id"] == "brand-a"
    assert client_b.kwargs["workspace_id"] == "brand-b"
    assert build_count["n"] == 2

    # Re-entering a profile's scope reuses its cached client.
    assert under_home(home_a) is client_a
    assert under_home(home_b) is client_b
    assert build_count["n"] == 2
