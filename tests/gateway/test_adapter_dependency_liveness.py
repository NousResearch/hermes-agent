"""Slow platform preparation must leave the gateway loop and profile scope intact."""

import asyncio
import sqlite3
import threading
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platform_registry import PlatformEntry, PlatformRegistry
from gateway.run import GatewayRunner, _profile_runtime_scope
from hermes_constants import get_hermes_home


def _startup_runner(monkeypatch):
    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig(
        platforms={Platform.FEISHU: PlatformConfig(enabled=True, token="fixture-token")}
    )
    runner._restart_requested = runner._draining = False
    runner._shutdown_event = asyncio.Event()
    runner._stop_task = None
    # Handler wiring follows construction; no dependency/registry/factory step is replaced.
    runner._wire_adapter_handlers = Mock()
    registry = PlatformRegistry()
    monkeypatch.setattr("gateway.platform_registry.platform_registry", registry)
    return runner, registry


@pytest.mark.asyncio
@pytest.mark.parametrize("stage", ["load", "probe", "install"])
@pytest.mark.parametrize("profile", ["default", "secondary"])
async def test_startup_preparation_yields_and_preserves_profile(tmp_path, monkeypatch, stage, profile):
    runner, registry = _startup_runner(monkeypatch)
    home = tmp_path / profile
    home.mkdir()
    (home / "config.yaml").write_text("{}\n", encoding="utf-8")
    loop = asyncio.get_running_loop()
    released = threading.Event()
    observations = []
    responsive = []
    adapter = SimpleNamespace()

    def prepare_step(name):
        observations.append((name, get_hermes_home()))
        if name == stage:
            # Only the gateway event loop can release the simulated blocking SDK work.
            # The timeout makes the unfixed synchronous path fail instead of deadlocking.
            loop.call_soon_threadsafe(released.set)
            responsive.append(released.wait(5))

    def probe():
        prepare_step("probe")
        return False

    def install():
        prepare_step("install")
        return True

    def factory(config):
        observations.append(("factory", get_hermes_home()))
        assert asyncio.get_running_loop() is loop
        return adapter

    def load():
        prepare_step("load")
        registry.register(PlatformEntry(
            name="feishu", label="Fixture", check_fn=probe,
            ensure_deps_fn=install, adapter_factory=factory,
        ))

    with _profile_runtime_scope(home, hydrate_secrets=False):
        registry.register_deferred("feishu", load)
        aborted, enabled, skipped, pending = await runner._start_prefilter_platforms()

    assert responsive == [True], "platform preparation blocked the gateway event loop"
    assert [name for name, _ in observations] == ["load", "probe", "install", "factory"]
    assert all(path == home for _, path in observations)
    assert not aborted and enabled == 1 and skipped == []
    assert pending[0][2] is adapter and adapter.gateway_runner is runner
    runner._wire_adapter_handlers.assert_called_once_with(adapter)


@pytest.mark.asyncio
@pytest.mark.parametrize("raises", [False, True])
async def test_failed_install_does_not_construct_or_fall_back(monkeypatch, raises):
    runner, registry = _startup_runner(monkeypatch)
    factory = Mock()
    builtin = Mock(side_effect=AssertionError("registered platforms must not fall back"))
    monkeypatch.setattr("gateway.run._instantiate_builtin_adapter", builtin)
    loop = asyncio.get_running_loop()
    released = threading.Event()
    responsive = []

    def install():
        loop.call_soon_threadsafe(released.set)
        responsive.append(released.wait(5))
        if raises:
            raise RuntimeError("package download failed")
        return False

    registry.register(PlatformEntry(
        name="feishu", label="Fixture", check_fn=lambda: False,
        ensure_deps_fn=install, adapter_factory=factory,
    ))
    aborted, enabled, skipped, pending = await runner._start_prefilter_platforms()

    assert responsive == [True], "failed installation blocked the gateway event loop"
    assert not aborted and enabled == 1 and skipped == [] and pending == []
    factory.assert_not_called()
    builtin.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize("interrupt", ["shutdown", "cancel"])
async def test_preparation_interrupt_closes_unpublished_adapters(tmp_path, monkeypatch, interrupt):
    runner, registry = _startup_runner(monkeypatch)
    runner.config.platforms = {
        Platform.TELEGRAM: PlatformConfig(enabled=True, token="fixture-token"),
        Platform.FEISHU: PlatformConfig(enabled=True, token="fixture-token"),
    }
    loop = asyncio.get_running_loop()
    released = threading.Event()
    connections = []

    def factory(config):
        connection = sqlite3.connect(tmp_path / f"adapter-{len(connections)}.db")
        connections.append(connection)

        async def disconnect():
            connection.close()

        async def cancel_background_tasks():
            pass

        return SimpleNamespace(disconnect=disconnect, cancel_background_tasks=cancel_background_tasks)

    def stop_preparing():
        if interrupt == "cancel":
            startup.cancel()
        else:
            runner._shutdown_event.set()
        released.set()

    def install():
        loop.call_soon_threadsafe(stop_preparing)
        return released.wait(5)

    registry.register(PlatformEntry(name="telegram", label="Ready", check_fn=lambda: True, adapter_factory=factory))
    registry.register(PlatformEntry(
        name="feishu", label="Preparing", check_fn=lambda: False,
        ensure_deps_fn=install, adapter_factory=factory,
    ))
    startup = asyncio.create_task(runner._start_prefilter_platforms())
    if interrupt == "cancel":
        with pytest.raises(asyncio.CancelledError):
            await startup
    else:
        aborted, _, _, pending = await startup
        assert aborted and pending == []
    try:
        assert len(connections) == (1 if interrupt == "cancel" else 2)
        for connection in connections:
            with pytest.raises(sqlite3.ProgrammingError, match="closed"):
                connection.execute("SELECT 1")
    finally:
        for connection in connections:
            connection.close()
