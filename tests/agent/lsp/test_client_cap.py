"""Per-service language-server process cap (t_8e8ec4de, 2026-10-28).

A Kanban worker has no operator anchor (``operator_workspace_roots()`` returns an
empty set when ``HERMES_KANBAN_TASK`` is set), so every checkout it touches is
UNTRUSTED. An untrusted root gets its own client key, so pyright's multi-root
sharing never engages: a worker that wrote into two repos ran TWO pyright
processes (observed 2026-10-28: 436 MB and 483 MB RSS in one worker scope; the
kernel OOM-killed one of them at 13:01 alongside its worker).

The cap is a RESOURCE bound, not a trust relaxation — the trust boundary is
unchanged. When the cap is reached the least-recently-used client is shut down
and respawned on demand.
"""

from __future__ import annotations

import asyncio

import pytest

from agent.lsp.manager import DEFAULT_MAX_CLIENTS, LSPService


class _FakeClient:
    """Minimal stand-in for :class:`agent.lsp.client.LSPClient`."""

    def __init__(self, server_id: str, workspace_root: str) -> None:
        self.server_id = server_id
        self.workspace_root = workspace_root
        self.workspace_folders = [workspace_root]
        self.is_running = True
        self.state = "running"
        self.shutdown_calls = 0

    async def shutdown(self) -> None:
        self.shutdown_calls += 1
        self.is_running = False


def _service(max_clients: int = DEFAULT_MAX_CLIENTS) -> LSPService:
    svc = LSPService(
        enabled=True, wait_mode="document", wait_timeout=5.0,
        install_strategy="manual", idle_timeout=0, max_clients=max_clients,
    )
    return svc


def _add(svc: LSPService, root: str, *, last_used: float) -> _FakeClient:
    client = _FakeClient("pyright", root)
    with svc._state_lock:
        svc._clients[("pyright", root)] = client
        svc._last_used[("pyright", root)] = last_used
    return client


def test_default_cap_is_two():
    assert DEFAULT_MAX_CLIENTS == 2
    assert _service()._max_clients == 2


def test_cap_is_never_below_one():
    assert _service(max_clients=0)._max_clients == 1
    assert _service(max_clients=-5)._max_clients == 1


def test_no_eviction_below_the_cap():
    """A service UNDER the cap is left completely alone.

    The cap counts live language-server processes. With one client and a cap
    of two there is room to spare, so a spawn must not disturb anything.
    """
    svc = _service(max_clients=2)
    a = _add(svc, "/a", last_used=100.0)

    evicted = asyncio.run(svc._evict_for_capacity(keep_key=("pyright", "/c")))

    assert evicted == 0
    assert a.shutdown_calls == 0
    assert len(svc._clients) == 1


def test_evicts_the_least_recently_used_when_over_the_cap():
    svc = _service(max_clients=2)
    oldest = _add(svc, "/a", last_used=100.0)
    newer = _add(svc, "/b", last_used=200.0)

    evicted = asyncio.run(svc._evict_for_capacity(keep_key=("pyright", "/c")))

    assert evicted == 1
    assert oldest.shutdown_calls == 1
    assert newer.shutdown_calls == 0
    assert ("pyright", "/a") not in svc._clients
    assert ("pyright", "/b") in svc._clients


def test_eviction_never_drops_the_client_the_caller_needs():
    svc = _service(max_clients=1)
    keep = _add(svc, "/keep", last_used=100.0)  # oldest, but it is the one in use
    other = _add(svc, "/other", last_used=200.0)

    evicted = asyncio.run(svc._evict_for_capacity(keep_key=("pyright", "/keep")))

    assert evicted == 1
    assert other.shutdown_calls == 1
    assert keep.shutdown_calls == 0
    assert ("pyright", "/keep") in svc._clients


def test_eviction_brings_the_service_back_within_the_cap():
    svc = _service(max_clients=2)
    for i, used in enumerate((100.0, 200.0, 300.0)):
        _add(svc, f"/w{i}", last_used=used)

    asyncio.run(svc._evict_for_capacity(keep_key=("pyright", "/new")))

    assert len(svc._clients) == 2


def test_eviction_forgets_the_last_used_entry():
    svc = _service(max_clients=1)
    _add(svc, "/a", last_used=100.0)

    asyncio.run(svc._evict_for_capacity(keep_key=("pyright", "/b")))

    assert ("pyright", "/a") not in svc._last_used


def test_service_reads_max_clients_from_config(monkeypatch):
    monkeypatch.setattr(
        "hermes_cli.config.load_config_readonly",
        lambda: {"lsp": {"enabled": True, "max_clients": 1}},
    )
    svc = LSPService.create_from_config()
    assert svc is not None
    assert svc._max_clients == 1


def test_service_defaults_max_clients_without_config(monkeypatch):
    monkeypatch.setattr(
        "hermes_cli.config.load_config_readonly",
        lambda: {"lsp": {"enabled": True}},
    )
    svc = LSPService.create_from_config()
    assert svc is not None
    assert svc._max_clients == DEFAULT_MAX_CLIENTS


def test_service_ignores_a_malformed_max_clients(monkeypatch):
    monkeypatch.setattr(
        "hermes_cli.config.load_config_readonly",
        lambda: {"lsp": {"enabled": True, "max_clients": "many"}},
    )
    svc = LSPService.create_from_config()
    assert svc is not None
    assert svc._max_clients == DEFAULT_MAX_CLIENTS
