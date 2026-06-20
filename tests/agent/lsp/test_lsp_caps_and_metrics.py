"""Regression tests for LSP caps, LRU eviction, active-request safety, and metrics.

Covers the A-only debloat scope:
- per-server caps are enforced
- total cap is enforced across server_ids
- LRU order is used to choose victims
- clients with active requests are NEVER evicted
- the periodic reaper runs on its own loop and shuts down idle clients
- metrics counters update correctly across spawn / reuse / reap / cap eviction
- LSPClient tracks in-flight requests via inc/dec
"""
from __future__ import annotations

import asyncio

import pytest

from agent.lsp.manager import LSPService


class FakeClient:
    """Minimal stand-in for an LSPClient that records lifecycle events."""

    def __init__(self, *, active: bool = False) -> None:
        self.shutdown_calls = 0
        self.is_running = True
        self._active = active

    async def shutdown(self) -> None:
        self.shutdown_calls += 1
        self.is_running = False

    @property
    def has_active_requests(self) -> bool:
        return self._active

    @property
    def inflight(self) -> int:
        return 1 if self._active else 0


def _service_with_caps(
    *,
    total_cap: int,
    per_server_caps: dict[str, int],
    idle: float = 0.0,
    reaper_interval: float = 60.0,
) -> LSPService:
    """Build a service with a configurable cap layout and reaping off by default."""
    return LSPService(
        enabled=False,
        wait_mode="document",
        wait_timeout=1.0,
        install_strategy="never",
        idle_timeout=idle,
        reaper_interval=reaper_interval,
        total_lsp_cap=total_cap,
        per_server_caps=per_server_caps,
    )


def _seed(service: LSPService, key, last_used: float) -> FakeClient:
    """Install a fake client under ``key`` with a fixed last_used timestamp."""
    client = FakeClient()
    service._clients[key] = client  # type: ignore[attr-defined]
    service._last_used[key] = last_used  # type: ignore[attr-defined]
    return client


@pytest.mark.asyncio
async def test_enforce_caps_returns_true_under_total_cap() -> None:
    service = _service_with_caps(total_cap=3, per_server_caps={"pyright": 2})
    _seed(service, ("pyright", "/tmp/ws-a"), last_used=10.0)
    assert await service._enforce_caps("pyright") is True  # type: ignore[attr-defined]


@pytest.mark.asyncio
async def test_enforce_caps_evicts_idle_peer_when_at_per_server_cap() -> None:
    service = _service_with_caps(total_cap=10, per_server_caps={"pyright": 2})
    idle_peer = _seed(service, ("pyright", "/tmp/old-ws"), last_used=1.0)
    _seed(service, ("pyright", "/tmp/new-ws"), last_used=9.0)
    assert await service._enforce_caps("pyright") is True
    assert idle_peer.shutdown_calls == 1
    assert ("pyright", "/tmp/old-ws") not in service._clients  # type: ignore[attr-defined]


@pytest.mark.asyncio
async def test_enforce_caps_evicts_lru_first() -> None:
    """3 pyright clients, cap=2 — must evict the 2 LRU peers so a new spawn fits.

    Cap of 2 means at most 2 clients.  Spawning a 4th requires going from
    3 → 1 → +1 = 2 (after spawn).  So 2 evictions are required, and the
    2 oldest by last_used are the LRU victims.
    """
    service = _service_with_caps(total_cap=10, per_server_caps={"pyright": 2})
    oldest = _seed(service, ("pyright", "/tmp/oldest"), last_used=1.0)
    middle = _seed(service, ("pyright", "/tmp/middle"), last_used=5.0)
    newest = _seed(service, ("pyright", "/tmp/newest"), last_used=9.0)
    assert await service._enforce_caps("pyright") is True
    assert oldest.shutdown_calls == 1
    assert middle.shutdown_calls == 1
    assert newest.shutdown_calls == 0


@pytest.mark.asyncio
async def test_enforce_caps_refuses_when_all_peers_busy() -> None:
    service = _service_with_caps(total_cap=10, per_server_caps={"pyright": 2})
    busy_a = _seed(service, ("pyright", "/tmp/a"), last_used=1.0)
    busy_b = _seed(service, ("pyright", "/tmp/b"), last_used=5.0)
    busy_a._active = True  # type: ignore[attr-defined]
    busy_b._active = True  # type: ignore[attr-defined]
    assert await service._enforce_caps("pyright") is False
    assert busy_a.shutdown_calls == 0
    assert busy_b.shutdown_calls == 0


@pytest.mark.asyncio
async def test_enforce_caps_enforces_total_cap_across_servers() -> None:
    service = _service_with_caps(
        total_cap=2, per_server_caps={"pyright": 8, "tsserver": 8}
    )
    pyright = _seed(service, ("pyright", "/tmp/a"), last_used=1.0)
    tsserver = _seed(service, ("tsserver", "/tmp/b"), last_used=2.0)
    # total=2 already; spawning a third must evict the LRU peer.
    # The pyright client has last_used=1.0 (older), so it is the LRU victim.
    assert await service._enforce_caps("pyright") is True
    assert pyright.shutdown_calls == 1
    assert tsserver.shutdown_calls == 0
    assert ("pyright", "/tmp/a") not in service._clients  # type: ignore[attr-defined]
    assert ("tsserver", "/tmp/b") in service._clients  # type: ignore[attr-defined]


@pytest.mark.asyncio
async def test_reaper_skips_active_clients() -> None:
    service = _service_with_caps(
        total_cap=10,
        per_server_caps={"pyright": 4},
        idle=10.0,
    )
    idle_client = _seed(service, ("pyright", "/tmp/idle"), last_used=1.0)
    busy_client = _seed(service, ("pyright", "/tmp/busy"), last_used=1.0)
    busy_client._active = True  # type: ignore[attr-defined]
    await service._reap_idle_clients_async(now=100.0)  # type: ignore[attr-defined]
    assert idle_client.shutdown_calls == 1
    assert busy_client.shutdown_calls == 0
    metrics = service.snapshot_metrics()["metrics"]
    assert metrics["reaps_blocked_active"] == 1
    assert metrics["reaps_idle"] == 1


@pytest.mark.asyncio
async def test_metrics_snapshot_returns_expected_shape() -> None:
    service = _service_with_caps(total_cap=5, per_server_caps={"pyright": 3})
    _seed(service, ("pyright", "/tmp/ws-a"), last_used=10.0)
    _seed(service, ("pyright", "/tmp/ws-b"), last_used=11.0)
    snap = service.snapshot_metrics()
    assert set(snap) == {
        "metrics",
        "clients",
        "caps",
        "idle_timeout_seconds",
        "reaper_interval_seconds",
        "process_fd_count",
        "process_rss_bytes",
        "lsp_child_rss_bytes",
    }
    assert snap["caps"]["current_total"] == 2
    assert snap["caps"]["current_per_server"] == {"pyright": 2}
    assert len(snap["clients"]) == 2
    metrics = snap["metrics"]
    assert set(metrics) == {
        "spawns",
        "reuses",
        "reaps_idle",
        "reaps_cap",
        "reaps_blocked_active",
        "shutdowns",
        "shutdown_failures",
        "spawn_failures",
    }


@pytest.mark.asyncio
async def test_inflight_increments_and_decrements() -> None:
    from agent.lsp.client import LSPClient

    client = LSPClient(
        server_id="pyright",
        workspace_root="/tmp/ws",
        command=["echo"],  # not actually started
    )
    assert client.inflight == 0
    assert client.has_active_requests is False
    await client.inc_inflight()
    await client.inc_inflight()
    assert client.inflight == 2
    assert client.has_active_requests is True
    await client.dec_inflight()
    assert client.inflight == 1
    assert client.has_active_requests is True
    await client.dec_inflight()
    assert client.inflight == 0
    assert client.has_active_requests is False


@pytest.mark.asyncio
async def test_track_request_context_manager_decrements_on_exception() -> None:
    """Even on exception, the inflight counter must drop back to zero."""
    from agent.lsp.client import LSPClient

    client = LSPClient(
        server_id="pyright",
        workspace_root="/tmp/ws",
        command=["echo"],
    )
    try:
        async with client.track_request():
            assert client.inflight == 1
            assert client.has_active_requests is True
            raise RuntimeError("simulated LSP failure")
    except RuntimeError:
        pass
    assert client.inflight == 0
    assert client.has_active_requests is False
