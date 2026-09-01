"""Shutdown fencing for the process-owned hosted-room worker."""

from __future__ import annotations

import asyncio
import threading
from types import SimpleNamespace

import pytest

from gateway.run import GatewayRunner


@pytest.mark.asyncio
async def test_late_hosted_room_recovery_is_stopped_after_shutdown() -> None:
    recovery_started = asyncio.Event()
    release_recovery = asyncio.Event()
    stop_calls: list[float] = []

    async def ensure_worker() -> object:
        recovery_started.set()
        await release_recovery.wait()
        return object()

    async def stop_worker(timeout: float = 5.0) -> bool:
        stop_calls.append(timeout)
        return True

    runner = SimpleNamespace(
        _running=True,
        _ensure_hosted_room_worker=ensure_worker,
        _stop_hosted_room_worker=stop_worker,
    )
    watcher = asyncio.create_task(
        GatewayRunner._hosted_room_worker_watcher(runner, interval=60.0)
    )

    await recovery_started.wait()
    runner._running = False
    release_recovery.set()

    await asyncio.wait_for(watcher, timeout=1.0)
    assert stop_calls == [5.0]


@pytest.mark.asyncio
async def test_cleared_start_gate_blocks_inflight_recovery(monkeypatch) -> None:
    from tui_gateway import methods_groups, server as _server  # noqa: F401

    start_entered = threading.Event()
    release_start = threading.Event()
    starts: list[object] = []

    monkeypatch.setattr(methods_groups, "get_hosted_room_service", lambda: None)

    def start_service(*, start_allowed=None):
        assert start_allowed is not None
        start_entered.set()
        assert release_start.wait(timeout=5.0)
        if not start_allowed.is_set():
            return None
        service = object()
        starts.append(service)
        return service

    monkeypatch.setattr(methods_groups, "start_hosted_room_service", start_service)
    runner = GatewayRunner.__new__(GatewayRunner)
    runner._running = True
    ensure = asyncio.create_task(runner._ensure_hosted_room_worker())

    assert await asyncio.to_thread(start_entered.wait, 5.0)
    runner._hosted_room_start_allowed.clear()
    runner._running = False
    release_start.set()

    assert await asyncio.wait_for(ensure, timeout=5.0) is None
    assert starts == []
