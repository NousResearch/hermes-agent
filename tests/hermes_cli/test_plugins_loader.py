import asyncio
import logging
import threading
import types

import pytest

from hermes_cli import plugins_loader


def test_nested_plugin_load_runs_inline_on_deadline_worker(monkeypatch):
    monkeypatch.setattr(plugins_loader, "_resolve_plugin_load_timeout", lambda: 0.3)
    abandoned = []

    class Context:
        def __init__(self, name):
            self.name = name

        def _abandon_load(self):
            abandoned.append(self.name)

    observed_threads = []
    release = threading.Event()

    def outer_load():
        observed_threads.append(threading.current_thread())
        plugins_loader.run_with_load_deadline(
            "done", Context("done"), lambda: observed_threads.append(threading.current_thread()),
        )
        plugins_loader.run_with_load_deadline("hung", Context("hung"), release.wait)

    with pytest.raises(plugins_loader.PluginLoadTimeout):
        plugins_loader.run_with_load_deadline("outer", Context("outer"), outer_load)
    release.set()

    assert len(observed_threads) == 2
    assert observed_threads[0] is observed_threads[1]
    # The outer timeout abandons only contexts still loading, not a nested load that already finished.
    assert abandoned == ["outer", "hung"]


@pytest.mark.asyncio
async def test_default_loader_schedules_background_task_on_host_loop(tmp_path, monkeypatch):
    """register() runs on the deadline worker, but its observer belongs to the caller's loop."""
    from hermes_cli.plugins import PluginManager, PluginManifest

    manager = PluginManager()
    manifest = PluginManifest(name="observer", source="user", path=str(tmp_path))
    host_loop = asyncio.get_running_loop()
    host_thread = threading.current_thread()
    started = asyncio.Event()
    cancelled = asyncio.Event()
    observed = {}

    async def observer():
        observed["loop"] = asyncio.get_running_loop()
        observed["thread"] = threading.current_thread()
        started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            cancelled.set()
            raise

    def register(ctx):
        observed["register_thread"] = threading.current_thread()
        observed["handle"] = ctx.spawn_task(observer(), name="selected-observer")

    monkeypatch.setattr(manager, "_load_directory_module", lambda *args, **kwargs: types.SimpleNamespace(register=register))
    monkeypatch.setattr(manager, "_runs_in_plugin_host", lambda _manifest: False)
    monkeypatch.setattr(plugins_loader, "_resolve_plugin_load_timeout", lambda: 1.0)

    manager._load_plugin(manifest)
    await asyncio.wait_for(started.wait(), timeout=1)

    assert manager._plugins["observer"].enabled is True
    assert observed["register_thread"] is not host_thread
    assert observed["loop"] is host_loop
    assert observed["thread"] is host_thread
    assert manager.unload("observer") is True
    await asyncio.wait_for(cancelled.wait(), timeout=1)


def test_sync_loader_rejects_background_task_and_closes_coroutine(tmp_path, monkeypatch):
    """CLI-style synchronous discovery fails clearly instead of leaking an unscheduled coroutine."""
    from hermes_cli.plugins import PluginManager, PluginManifest

    manager = PluginManager()
    manifest = PluginManifest(name="sync-observer", source="user", path=str(tmp_path))
    created = []

    async def observer():
        return None

    def register(ctx):
        coro = observer()
        created.append(coro)
        ctx.spawn_task(coro, name="sync-observer")

    monkeypatch.setattr(manager, "_load_directory_module", lambda *args, **kwargs: types.SimpleNamespace(register=register))
    monkeypatch.setattr(manager, "_runs_in_plugin_host", lambda _manifest: False)
    monkeypatch.setattr(plugins_loader, "_resolve_plugin_load_timeout", lambda: 1.0)

    manager._load_plugin(manifest)

    loaded = manager._plugins["sync-observer"]
    assert loaded.enabled is False
    assert loaded.error is not None
    assert "running host asyncio loop" in loaded.error
    assert created[0].cr_frame is None


@pytest.mark.asyncio
async def test_supervised_background_task_exception_is_logged(caplog):
    from hermes_cli.plugins import PluginContext, PluginManager, PluginManifest

    manager = PluginManager()
    ctx = PluginContext(PluginManifest(name="boom-observer", source="user"), manager)

    async def boom():
        raise RuntimeError("observer exploded")

    with caplog.at_level(logging.ERROR, logger="hermes_cli.plugins"):
        task = ctx.spawn_task(boom(), name="boom-observer")
        with pytest.raises(RuntimeError, match="observer exploded"):
            await task
        await asyncio.sleep(0)

    assert "background task boom-observer failed: observer exploded" in caplog.text
    assert "boom-observer" not in manager._ownership_ledger
