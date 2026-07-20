"""Race-focused tests for the opt-in bounded LSP client lifecycle."""
from __future__ import annotations

import asyncio
import threading
from types import SimpleNamespace
from typing import cast

import pytest

import agent.lsp.manager as manager_module
from agent.lsp.client import LSPClient
from agent.lsp.manager import (
    LSPService,
    _ClientEntry,
    _ClientLease,
    _lifecycle_config,
)


class FakeClient:
    def __init__(self, server_id: str = "fake", workspace_root: str = "/repo"):
        self.server_id = server_id
        self.workspace_root = workspace_root
        self.state = "running"
        self.running = True
        self.shutdown_calls = 0

    @property
    def is_running(self) -> bool:
        return self.running

    async def shutdown(self) -> None:
        self.shutdown_calls += 1
        await asyncio.sleep(0)
        self.running = False
        self.state = "stopped"

    def diagnostics_for(self, _file_path: str):
        return []


def make_service(*, clock, lifecycle_enabled=True, idle_timeout=100.0, max_clients=0, sweep=3600.0):
    return LSPService(
        enabled=True,
        wait_mode="document",
        wait_timeout=1.0,
        install_strategy="manual",
        lifecycle_enabled=lifecycle_enabled,
        idle_timeout=idle_timeout,
        sweep_interval=sweep,
        max_clients=max_clients,
        clock=clock,
    )


def install_entry(
    service: LSPService,
    key: tuple[str, str],
    *,
    last_used: float,
    leases: int = 0,
) -> tuple[_ClientEntry, FakeClient]:
    client = FakeClient(key[0], key[1])
    with service._state_lock:
        service._next_generation += 1
        entry = _ClientEntry(
            key=key,
            generation=service._next_generation,
            workspace_root=key[1],
            state="active",
            last_used=last_used,
            client=cast(LSPClient, client),
            leases=leases,
        )
        service._entries[key] = entry
    return entry, client


def test_lifecycle_config_is_opt_in_and_strict(caplog):
    assert _lifecycle_config({}) == (False, 7200.0, 60.0, 0, None)
    assert _lifecycle_config(
        {
            "lifecycle": {
                "enabled": True,
                "idle_timeout_seconds": 120,
                "sweep_interval_seconds": 5,
                "max_clients_per_process": 3,
            }
        }
    ) == (True, 120.0, 5.0, 3, None)

    invalid_enabled = _lifecycle_config({"lifecycle": {"enabled": "false"}})
    assert invalid_enabled[:4] == (False, 7200.0, 60.0, 0)
    assert invalid_enabled[4]

    invalid_enabled_policies = [
        {"enabled": True, "idle_timeout_seconds": True},
        {"enabled": True, "idle_timeout_seconds": float("nan")},
        {"enabled": True, "sweep_interval_seconds": 0},
        {"enabled": True, "max_clients_per_process": 1.5},
        {"enabled": True, "max_clients_per_process": 65},
        {"enabled": True, "max_client_per_process": 3},
    ]
    for lifecycle in invalid_enabled_policies:
        parsed = _lifecycle_config({"lifecycle": lifecycle})
        assert parsed[:4] == (True, 7200.0, 60.0, 0)
        assert parsed[4]
    assert "invalid lsp.lifecycle configuration" in caplog.text


def test_default_config_declares_default_off_lifecycle_policy():
    from hermes_cli.config import DEFAULT_CONFIG

    assert DEFAULT_CONFIG["lsp"]["lifecycle"] == {
        "enabled": False,
        "idle_timeout_seconds": 7200,
        "sweep_interval_seconds": 60,
        "max_clients_per_process": 0,
    }


def test_create_from_default_config_wires_process_lifetime_policy(monkeypatch):
    from hermes_cli import config as config_module

    monkeypatch.setattr(config_module, "load_config", lambda: config_module.DEFAULT_CONFIG)
    service = LSPService.create_from_config()
    assert service is not None
    try:
        lifecycle = service.get_status()["lifecycle"]
        assert lifecycle["enabled"] is False
        assert lifecycle["idle_timeout_seconds"] == 7200.0
        assert lifecycle["sweep_interval_seconds"] == 60.0
        assert lifecycle["max_clients_per_process"] == 0
        assert lifecycle["config_error"] is None
        assert lifecycle["reaper_running"] is False
    finally:
        service.shutdown()


def test_create_from_config_wires_lifecycle_settings(monkeypatch):
    from hermes_cli import config as config_module

    monkeypatch.setattr(
        config_module,
        "load_config",
        lambda: {
            "lsp": {
                "enabled": True,
                "lifecycle": {
                    "enabled": True,
                    "idle_timeout_seconds": 1800,
                    "sweep_interval_seconds": 30,
                    "max_clients_per_process": 4,
                },
            }
        },
    )
    service = LSPService.create_from_config()
    assert service is not None
    try:
        lifecycle = service.get_status()["lifecycle"]
        assert lifecycle["enabled"] is True
        assert lifecycle["idle_timeout_seconds"] == 1800.0
        assert lifecycle["sweep_interval_seconds"] == 30.0
        assert lifecycle["max_clients_per_process"] == 4
        assert lifecycle["config_error"] is None
        assert lifecycle["reaper_running"] is True
    finally:
        service.shutdown()


def test_invalid_enabled_policy_stays_bounded_and_surfaces_error(monkeypatch):
    from hermes_cli import config as config_module

    monkeypatch.setattr(
        config_module,
        "load_config",
        lambda: {
            "lsp": {
                "enabled": True,
                "lifecycle": {
                    "enabled": True,
                    "max_clients_per_process": 65,
                },
            }
        },
    )
    service = LSPService.create_from_config()
    assert service is not None
    try:
        lifecycle = service.get_status()["lifecycle"]
        assert lifecycle["enabled"] is True
        assert lifecycle["idle_timeout_seconds"] == 7200.0
        assert lifecycle["sweep_interval_seconds"] == 60.0
        assert lifecycle["max_clients_per_process"] == 0
        assert "between 0 and 64" in lifecycle["config_error"]
        assert lifecycle["reaper_running"] is True
    finally:
        service.shutdown()


def test_feature_disabled_preserves_process_lifetime_retention():
    now = [1000.0]
    service = make_service(
        clock=lambda: now[0], lifecycle_enabled=False, idle_timeout=1.0
    )
    _, client = install_entry(service, ("fake", "/repo"), last_used=0.0)
    try:
        service._loop.run(service._reap_idle_clients(), timeout=2.0)
        assert client.shutdown_calls == 0
        status = service.get_status()
        assert status["lifecycle"]["reaper_running"] is False
        assert status["clients"][0]["state"] == "running"
        assert status["clients"][0]["lifecycle_state"] == "active"
    finally:
        service.shutdown()


def test_service_shutdown_is_idempotent_and_stops_owner_thread():
    service = make_service(clock=lambda: 0.0)
    service.shutdown()
    service.shutdown()
    assert service.is_closed() is True
    assert service._loop._thread is None


def test_synchronous_shutdown_is_rejected_on_owner_loop():
    service = make_service(clock=lambda: 0.0)

    async def scenario():
        with pytest.raises(RuntimeError, match="asynchronous teardown"):
            service.shutdown()

    try:
        service._loop.run(scenario(), timeout=2.0)
        assert service.is_accepting_requests() is True
    finally:
        service.shutdown()


def test_idle_boundary_and_monotonic_clock():
    now = [99.9]
    service = make_service(clock=lambda: now[0], idle_timeout=100.0)
    _, client = install_entry(service, ("fake", "/repo"), last_used=0.0)
    try:
        service._loop.run(service._reap_idle_clients(), timeout=2.0)
        assert client.shutdown_calls == 0
        now[0] = 100.0
        service._loop.run(service._reap_idle_clients(), timeout=2.0)
        assert client.shutdown_calls == 1
        assert service.get_status()["clients"] == []
    finally:
        service.shutdown()


def test_active_lease_blocks_reaping_until_a_later_idle_sweep(monkeypatch):
    now = [100.0]
    service = make_service(clock=lambda: now[0], idle_timeout=50.0)
    key = ("fake", "/repo")
    entry, client = install_entry(service, key, last_used=0.0)
    target = (SimpleNamespace(server_id="fake"), key, "/repo")
    monkeypatch.setattr(service, "_resolve_target", lambda *args, **kwargs: target)

    async def scenario():
        lease = await service._acquire_lease("/repo/x.py", spawn=False)
        assert lease is not None
        assert entry.leases == 1
        await service._reap_idle_clients()
        assert client.shutdown_calls == 0
        lease.release()
        assert entry.leases == 0
        now[0] = 151.0
        await service._reap_idle_clients()

    try:
        service._loop.run(scenario(), timeout=2.0)
        assert client.shutdown_calls == 1
    finally:
        service.shutdown()


def test_shutdown_cancels_lease_owner_and_waits_for_final_release(monkeypatch):
    now = [100.0]
    service = make_service(clock=lambda: now[0])
    key = ("fake", "/repo")
    entry, client = install_entry(service, key, last_used=0.0)
    started = asyncio.Event()
    blocker = asyncio.Event()
    monkeypatch.setattr(manager_module, "SHUTDOWN_LEASE_DRAIN_TIMEOUT", 0.01)

    async def owner():
        task = asyncio.current_task()
        assert task is not None
        with service._state_lock:
            entry.leases = 1
            entry.lease_tasks.add(task)
        lease = _ClientLease(
            service,
            key,
            entry.generation,
            cast(LSPClient, client),
            task,
        )
        started.set()
        try:
            await blocker.wait()
        finally:
            lease.release()

    async def scenario():
        owner_task = asyncio.create_task(owner())
        await started.wait()
        service.begin_shutdown()
        await service._shutdown_async()
        assert owner_task.cancelled()

    try:
        service._loop.run(scenario(), timeout=2.0)
        assert client.shutdown_calls == 1
        assert service.get_status()["clients"] == []
    finally:
        blocker.set()
        assert service._loop.stop() is True


def test_failed_retirement_tombstone_blocks_overlapping_replacement(monkeypatch):
    service = make_service(clock=lambda: 0.0)
    key = ("fake", "/repo")
    entry, _ = install_entry(service, key, last_used=0.0)
    srv = SimpleNamespace(server_id="fake")
    monkeypatch.setattr(
        service, "_resolve_target", lambda *args, **kwargs: (srv, key, "/repo")
    )

    class FailingClient(FakeClient):
        async def shutdown(self) -> None:
            self.shutdown_calls += 1
            raise RuntimeError("process still alive")

    client = FailingClient()
    with service._state_lock:
        entry.client = cast(LSPClient, client)

    async def scenario():
        assert await service._retire_key_async(key, "idle") is True
        with service._state_lock:
            assert service._entries[key] is entry
            assert entry.state == "retiring"
            assert entry.retire_task is None
        assert await service._reserve_spawn(srv, key, "/repo") is None
        assert (
            await asyncio.wait_for(
                service._acquire_lease("/repo/x.py", spawn=True), timeout=0.1
            )
            is None
        )

    try:
        service._loop.run(scenario(), timeout=2.0)
    finally:
        client.running = False
        client.state = "stopped"
        service.shutdown()


def test_generation_bound_release_is_idempotent_and_cannot_touch_replacement():
    now = [10.0]
    service = make_service(clock=lambda: now[0])
    key = ("fake", "/repo")
    old_entry, old_client = install_entry(service, key, last_used=0.0, leases=1)
    old_lease = _ClientLease(
        service, key, old_entry.generation, cast(LSPClient, old_client)
    )
    with service._state_lock:
        service._entries.pop(key)
    replacement, _ = install_entry(service, key, last_used=5.0, leases=1)
    try:
        old_lease.release()
        old_lease.release()
        assert replacement.leases == 1
    finally:
        with service._state_lock:
            replacement.leases = 0
        service.shutdown()


def test_distinct_cold_starts_are_not_globally_serialized(monkeypatch):
    service = make_service(clock=lambda: 0.0, lifecycle_enabled=False)
    both_started = asyncio.Event()
    started = []

    async def fake_spawn(entry, _srv):
        started.append(entry.key)
        if len(started) == 2:
            both_started.set()
        await both_started.wait()
        client = FakeClient(entry.key[0], entry.workspace_root)
        with service._state_lock:
            entry.client = cast(LSPClient, client)
            entry.state = "active"
        return cast(LSPClient, client)

    monkeypatch.setattr(service, "_spawn_entry", fake_spawn)

    async def scenario():
        srv = SimpleNamespace(server_id="fake")
        first = asyncio.create_task(service._reserve_spawn(srv, ("fake", "/a"), "/a"))
        second = asyncio.create_task(service._reserve_spawn(srv, ("fake", "/b"), "/b"))
        leases = await asyncio.wait_for(asyncio.gather(first, second), timeout=1.0)
        assert all(lease is not None for lease in leases)
        for lease in leases:
            assert lease is not None
            lease.release()

    try:
        service._loop.run(scenario(), timeout=2.0)
        assert set(started) == {("fake", "/a"), ("fake", "/b")}
    finally:
        service.shutdown()


def test_capacity_admission_waits_for_inflight_retirement(monkeypatch):
    service = make_service(clock=lambda: 0.0, max_clients=1)
    old_entry, _ = install_entry(service, ("fake", "/old"), last_used=0.0)
    release_retirement = asyncio.Event()
    spawn_started = asyncio.Event()

    async def fake_spawn(entry, _srv):
        spawn_started.set()
        client = FakeClient(entry.key[0], entry.workspace_root)
        with service._state_lock:
            entry.client = cast(LSPClient, client)
            entry.state = "active"
        return cast(LSPClient, client)

    monkeypatch.setattr(service, "_spawn_entry", fake_spawn)

    async def scenario():
        async def retire_old():
            await release_retirement.wait()
            with service._state_lock:
                service._entries.pop(old_entry.key, None)

        with service._state_lock:
            old_entry.state = "retiring"
            old_entry.retire_task = asyncio.create_task(retire_old())
        srv = SimpleNamespace(server_id="fake")
        reservation = asyncio.create_task(
            service._reserve_spawn(srv, ("fake", "/new"), "/new")
        )
        await asyncio.sleep(0)
        assert not spawn_started.is_set()
        release_retirement.set()
        lease = await asyncio.wait_for(reservation, timeout=1.0)
        assert lease is not None
        lease.release()

    try:
        service._loop.run(scenario(), timeout=2.0)
        assert spawn_started.is_set()
    finally:
        release_retirement.set()
        service.shutdown()


def test_capacity_evicts_exactly_the_unleased_lru_generation():
    now = [100.0]
    service = make_service(clock=lambda: now[0], max_clients=3)
    clients = {}
    for index in range(4):
        key = ("fake", f"/repo-{index}")
        _, clients[key] = install_entry(service, key, last_used=float(index))
    try:
        service._loop.run(service._converge_capacity(), timeout=2.0)
        status = service.get_status()
        assert len(status["clients"]) == 3
        assert clients[("fake", "/repo-0")].shutdown_calls == 1
        assert status["lifecycle"]["capacity_evictions"] == 1
    finally:
        service.shutdown()


def test_all_leased_overflow_converges_on_release():
    now = [100.0]
    service = make_service(clock=lambda: now[0], max_clients=3)
    entries = []
    clients = []
    for index in range(4):
        entry, client = install_entry(
            service,
            ("fake", f"/repo-{index}"),
            last_used=float(index),
            leases=1,
        )
        entries.append(entry)
        clients.append(client)

    async def scenario():
        await service._converge_capacity()
        assert len(service._entries) == 4
        lease = _ClientLease(
            service,
            entries[0].key,
            entries[0].generation,
            cast(LSPClient, clients[0]),
        )
        lease.release()
        await asyncio.sleep(0)
        pending = list(service._maintenance_tasks)
        if pending:
            await asyncio.gather(*pending)

    try:
        service._loop.run(scenario(), timeout=2.0)
        assert len(service.get_status()["clients"]) == 3
        assert clients[0].shutdown_calls == 1
    finally:
        with service._state_lock:
            for entry in service._entries.values():
                entry.leases = 0
        service.shutdown()


def test_reaper_scheduler_survives_one_failed_sweep(monkeypatch):
    now = [0.0]
    service = make_service(clock=lambda: now[0], sweep=0.01)
    completed = threading.Event()
    calls = []

    async def flaky_sweep():
        calls.append(len(calls) + 1)
        if len(calls) == 1:
            raise RuntimeError("synthetic sweep failure")
        completed.set()

    monkeypatch.setattr(service, "_reap_idle_clients", flaky_sweep)
    try:
        assert completed.wait(timeout=1.0)
        assert len(calls) >= 2
        assert service.get_status()["lifecycle"]["reaper_running"] is True
    finally:
        service.shutdown()
