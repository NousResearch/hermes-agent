"""Direct tests for the opt-in bounded LSP client lifecycle."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from agent.lsp import manager
from agent.lsp.manager import LSPService, _ClientEntry, _lifecycle_config


class _FakeClient:
    def __init__(self, *, running: bool = True) -> None:
        self.is_running = running
        self.process_alive = running
        self.shutdown_calls = 0

    async def shutdown(self) -> None:
        self.shutdown_calls += 1
        self.is_running = False
        self.process_alive = False


def _service(*, now: float = 100.0, max_clients: int = 0) -> LSPService:
    service = LSPService(
        enabled=False,
        wait_mode="document",
        wait_timeout=1.0,
        install_strategy="manual",
        lifecycle_enabled=True,
        idle_timeout=10.0,
        sweep_interval=1.0,
        max_clients=max_clients,
        clock=lambda: now,
    )
    service._admission_lock = asyncio.Lock()
    return service


def _entry(
    key: tuple[str, str],
    generation: int,
    *,
    last_used: float,
    client: _FakeClient | None = None,
    leases: int = 0,
) -> _ClientEntry:
    return _ClientEntry(
        key=key,
        generation=generation,
        workspace_root=key[1],
        state="active",
        last_used=last_used,
        client=client or _FakeClient(),
        leases=leases,
    )


@pytest.mark.asyncio
async def test_reaper_retires_idle_generation() -> None:
    service = _service(now=100.0)
    entry = _entry(("pyright", "/repo"), 1, last_used=80.0)
    service._entries[entry.key] = entry

    await service._reap_idle_clients()

    assert entry.client.shutdown_calls == 1
    assert service._entries == {}
    assert service._reap_count == 1


@pytest.mark.asyncio
async def test_reaper_does_not_retire_leased_generation() -> None:
    service = _service(now=100.0)
    entry = _entry(("pyright", "/repo"), 1, last_used=80.0, leases=1)
    service._entries[entry.key] = entry

    await service._reap_idle_clients()

    assert entry.client.shutdown_calls == 0
    assert service._entries[entry.key] is entry
    assert entry.state == "active"


@pytest.mark.asyncio
async def test_capacity_evicts_least_recently_used_generation() -> None:
    service = _service(now=100.0, max_clients=2)
    oldest = _entry(("pyright", "/old"), 1, last_used=10.0)
    middle = _entry(("pyright", "/middle"), 2, last_used=20.0)
    newest = _entry(("pyright", "/new"), 3, last_used=30.0)
    service._entries = {
        oldest.key: oldest,
        middle.key: middle,
        newest.key: newest,
    }

    await service._converge_capacity()

    assert oldest.client.shutdown_calls == 1
    assert set(service._entries) == {middle.key, newest.key}
    assert service._capacity_eviction_count == 1


@pytest.mark.asyncio
async def test_crashed_generation_is_retired_before_replacement(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = _service()
    key = ("pyright", "/repo")
    crashed = _entry(
        key,
        1,
        last_used=90.0,
        client=_FakeClient(running=False),
    )
    service._entries[key] = crashed
    server = SimpleNamespace(server_id="pyright")
    monkeypatch.setattr(
        service,
        "_resolve_target",
        lambda *_args, **_kwargs: (server, key, "/repo"),
    )
    replacement = object()

    async def reserve_spawn(*_args, **_kwargs):
        assert key not in service._entries
        return replacement

    monkeypatch.setattr(service, "_reserve_spawn", reserve_spawn)

    acquired = await service._acquire_lease("/repo/x.py", spawn=True)

    assert acquired is replacement
    assert crashed.client.shutdown_calls == 1


@pytest.mark.asyncio
async def test_shutdown_timeout_cancels_lease_owner_and_drains(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = _service()
    service.begin_shutdown()
    entry = _entry(("pyright", "/repo"), 1, last_used=100.0, leases=1)
    service._entries[entry.key] = entry
    started = asyncio.Event()
    never = asyncio.Event()

    async def lease_owner() -> None:
        task = asyncio.current_task()
        assert task is not None
        entry.lease_tasks.add(task)
        started.set()
        try:
            await never.wait()
        finally:
            with service._state_lock:
                entry.leases -= 1
                entry.lease_tasks.discard(task)

    owner = asyncio.create_task(lease_owner())
    await started.wait()
    monkeypatch.setattr(manager, "SHUTDOWN_LEASE_DRAIN_TIMEOUT", 0.01)

    await service._shutdown_async()

    assert owner.cancelled()
    assert entry.client.shutdown_calls == 1
    assert service.is_closed()


def test_invalid_enabled_lifecycle_keeps_bounded_defaults() -> None:
    enabled, idle, sweep, max_clients, error = _lifecycle_config(
        {
            "lifecycle": {
                "enabled": True,
                "idle_timeout_seconds": "not-a-number",
                "max_clients_per_process": 4,
            }
        }
    )

    assert enabled is True
    assert idle == manager.DEFAULT_IDLE_TIMEOUT
    assert sweep == manager.DEFAULT_SWEEP_INTERVAL
    assert max_clients == 0
    assert "idle_timeout_seconds" in (error or "")
