"""A ``reload_safe`` platform handler is wired again after a sweep that only reloads plugins.

The plugin builds a fresh owner per load and registers its bound method with ``reload_safe=True`` (same
qualified name on every load); its unload callback removes that owner's handler from the native table. A
sweep that reloads only plugins the process already had fires no ``on_plugin_loaded`` event, so nothing
wired the replacement after unload removed the old handler. Both reload entrypoints run against the real
plugin manager, a real Telegram adapter and the real gateway re-wire mixin.
"""

from __future__ import annotations

import asyncio
import os
import threading
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from gateway.config import Platform, PlatformConfig
from gateway.run_plugin_rewire import GatewayPluginRewireMixin, reload_plugins_verb
from hermes_cli.plugins import PluginManager, discover_plugins, get_plugin_manager
from plugins.platforms.telegram.adapter import TelegramAdapter
from tests.waiting import async_wait_until

_PLUGIN = '''
class Owner:
    def __init__(self):
        self.native = None

    def wire(self, native, adapter):
        self.native = native
        native.handlers.append(self)

    def unwire(self):
        if self.native is not None and self in self.native.handlers:
            self.native.handlers.remove(self)


def register(ctx):
    owner = Owner()
    ctx.register_platform_handler("telegram", owner.wire, reload_safe=True)
    ctx.on_unload(owner.unwire)
'''


class _Native:
    def __init__(self):
        self.handlers: list = []


class _Runner(GatewayPluginRewireMixin):
    def __init__(self, adapter):
        self.adapters = {Platform.TELEGRAM: adapter}


@pytest.fixture
def connected():
    """The plugin loaded at boot and wired once by ``connect()`` on a live native client."""
    home = Path(os.environ["HERMES_HOME"])
    plugin = home / "plugins" / "reload_owner"
    plugin.mkdir(parents=True)
    (plugin / "plugin.yaml").write_text("name: reload_owner\nversion: '0.1'\ndescription: t\n")
    (plugin / "__init__.py").write_text(_PLUGIN)
    (home / "config.yaml").write_text("plugins:\n  enabled: [reload_owner]\n")
    discover_plugins()
    native = _Native()
    adapter = TelegramAdapter(PlatformConfig(enabled=True, token="t", extra={}))
    adapter._wire_plugin_handlers(native)
    [first] = native.handlers
    return adapter, native, first


class _CountingLock:
    """The manager's discovery lock, counting the re-wire pass's non-blocking attempts."""

    def __init__(self):
        self._lock = threading.RLock()
        self.tries = 0

    def acquire(self, blocking: bool = True) -> bool:
        self.tries += not blocking
        return self._lock.acquire(blocking)

    def release(self) -> None:
        self._lock.release()

    __enter__ = acquire

    def __exit__(self, *_exc) -> None:
        self.release()


def test_forced_sweep_on_a_worker_thread_wires_the_replacement(connected):
    adapter, native, first = connected
    manager = get_plugin_manager()

    async def scenario():
        _Runner(adapter)._subscribe_plugin_rewire(manager)
        await asyncio.to_thread(manager.discover_and_load, True)
        await async_wait_until(lambda: native.handlers, what="replacement wired")

    asyncio.run(scenario())
    [replacement] = native.handlers
    assert replacement is not first and replacement.native is native


def test_reload_plugins_verb_wires_the_replacement(connected, monkeypatch):
    adapter, native, first = connected
    monkeypatch.setattr("gateway.run_plugin_rewire._reconcile_plugin_mcp_servers", lambda *_a: None)

    async def scenario():
        runner = _Runner(adapter)
        runner._subscribe_plugin_rewire(get_plugin_manager())
        return await asyncio.to_thread(reload_plugins_verb(runner, asyncio.get_running_loop()), {})

    answer = asyncio.run(scenario())
    assert answer["reloaded"] is True and answer["adapters_rewired"] == 1
    [replacement] = native.handlers
    assert replacement is not first and replacement.native is native


def test_rewire_pass_defers_to_a_sweep_in_flight_without_blocking_the_loop():
    manager = PluginManager()
    manager._discovery_lock = lock = _CountingLock()
    adapter = MagicMock()
    held, release = threading.Event(), threading.Event()

    def sweep_in_flight():
        with manager._discovery_lock:
            held.set()
            release.wait(5)

    async def scenario():
        holder = threading.Thread(target=sweep_in_flight)
        holder.start()
        held.wait(5)
        _Runner(adapter)._rewire_after_sweep(asyncio.get_running_loop(), manager)
        await async_wait_until(lambda: lock.tries >= 3, what="the pass retried while the sweep held the lock")
        during = adapter.rewire_plugin_handlers.call_count
        release.set()
        holder.join(5)
        await async_wait_until(lambda: adapter.rewire_plugin_handlers.call_count, what="the deferred pass ran")
        return during, adapter.rewire_plugin_handlers.call_count

    assert asyncio.run(scenario()) == (0, 1)
