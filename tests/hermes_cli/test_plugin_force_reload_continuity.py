"""Force re-discovery must not open a gap in live plugin behaviour.

Between unload-all and re-registration a concurrent turn used to see an empty hook table (context hooks
missing, ``pre_tool_call`` guards failing open) and every plugin's supervised task was cancelled even when
the plugin came straight back. Real temporary plugins, real discovery, real ledger. Seams: a wrapper that
runs a check right after the native unload-all returns, and, for the dispatch race, a live hook table
whose lookup can be paused on one thread.
"""

from __future__ import annotations

import asyncio
import threading
from pathlib import Path

import pytest

import hermes_yaml as yaml

_HOOK_PLUGIN = (
    "UNLOADED = []\n"
    "\n"
    "def _context(**kwargs):\n"
    "    return {'context': 'KEEP:' + __name__}\n"
    "\n"
    "def _guard(tool_name=None, **kwargs):\n"
    "    return {'action': 'block', 'message': 'GUARD:' + __name__}\n"
    "\n"
    "def register(ctx):\n"
    "    ctx.register_hook('pre_llm_call', _context)\n"
    "    ctx.register_hook('pre_tool_call', _guard)\n"
    "    ctx.on_unload(lambda: UNLOADED.append(True))\n"
)

_TASK_PLUGIN = (
    "import asyncio\n"
    "context = None\n"
    "\n"
    "def register(ctx):\n"
    "    global context\n"
    "    context = ctx\n"
    "    if RESTART_POLLER:\n"
    "        try:\n"
    "            asyncio.get_running_loop()\n"
    "        except RuntimeError:\n"
    "            return\n"
    "        ctx.spawn_task(asyncio.sleep(3600), name='poller')\n"
    "\n"
    "def start(coro, name='held-work'):\n"
    "    return context.spawn_task(coro, name=name)\n"
)


def _plugin(home: Path, name: str, body: str) -> None:
    plugin_dir = home / "plugins" / name
    plugin_dir.mkdir(parents=True)
    (plugin_dir / "plugin.yaml").write_text(yaml.safe_dump({"name": name, "version": "0.1.0"}))
    (plugin_dir / "__init__.py").write_text(body)


def _enable(home: Path, *names: str) -> None:
    # Inline loads keep register() on the calling thread (the test's event loop for task plugins).
    (home / "config.yaml").write_text(
        yaml.safe_dump({"plugins": {"enabled": list(names), "load_timeout_seconds": 0}}))


@pytest.fixture
def home(tmp_path, monkeypatch):
    import hermes_cli.plugins as plugins_mod
    from hermes_cli.plugins import PluginManager

    hermes_home = tmp_path / "hermes"
    (hermes_home / "plugins").mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    monkeypatch.setattr(plugins_mod, "get_bundled_plugins_dir", lambda: tmp_path / "empty-bundled")
    monkeypatch.setattr(PluginManager, "_scan_entry_points", lambda self: [])
    return hermes_home


def _observe_after_unload(manager, observe):
    """Run ``observe`` right after the real unload-all returns, inside the force sweep."""
    native = manager._unload_scoped
    seen = []

    def wrapper(plugin=None):
        result = native(plugin)
        if plugin is None:
            seen.append(observe())
        return result

    manager._unload_scoped = wrapper
    return seen


def _manager():
    from hermes_cli.plugins import PluginManager
    manager = PluginManager()
    manager.discover_and_load()
    return manager


def test_hooks_stay_dispatchable_between_unload_all_and_reregistration(home):
    _plugin(home, "keep_probe", _HOOK_PLUGIN)
    _enable(home, "keep_probe")
    manager = _manager()
    first_guard = manager._hooks["pre_tool_call"][0]
    first_module = manager._plugins["keep_probe"].module
    assert manager.invoke_hook("pre_llm_call", session_id="s")

    seen = _observe_after_unload(manager, lambda: (
        dict(manager._hooks),
        list(first_module.UNLOADED),
        manager.invoke_hook("pre_llm_call", session_id="s"),
        manager.invoke_hook("pre_tool_call", tool_name="terminal", args={}),
        asyncio.run(manager.ainvoke_hook("pre_tool_call", tool_name="terminal", args={})),
        manager.has_hook("pre_tool_call"),
        manager.iter_hook_callbacks("pre_tool_call"),
    ))
    manager.discover_and_load(force=True)

    (live, unloaded, context, guard, async_guard, has_guard, callbacks), = seen
    assert live == {}, "the native unload-all really ran before the observation"
    # Documented trade-off: the frozen table still serves the previous incarnation after its on_unload ran.
    assert unloaded == [True]
    assert context and context[0]["context"].startswith("KEEP:")
    assert guard and guard[0]["action"] == "block", "pre_tool_call guard must not fail open mid-reload"
    assert async_guard and async_guard[0]["action"] == "block"
    assert has_guard and callbacks == (first_guard,)
    # Commit: dispatch is back on the live table, served by the re-registered module.
    assert manager._hook_dispatch_state[1] is None
    assert len(manager._hooks["pre_tool_call"]) == 1
    assert manager._hooks["pre_tool_call"][0] is not first_guard
    assert manager.iter_hook_callbacks("pre_tool_call") == (manager._hooks["pre_tool_call"][0],)


def _manager_pausing_first_hook_table_read():
    """A real manager whose hook-table reads (``_hooks`` and any ``_hook_dispatch*`` attribute) are plain
    attribute reads, except that the first such read on ``pause_thread`` blocks until ``release`` is set.

    That forces an interleaving the scheduler can produce: a dispatch has already read hook-table state and
    is descheduled before it uses it, while a force sweep starts and unload-all runs.
    """
    from hermes_cli.plugins import PluginManager

    class PausingManager(PluginManager):
        pause_thread: threading.Thread | None = None
        paused = threading.Event()
        release = threading.Event()

        def __getattribute__(self, name):
            value = super().__getattribute__(name)
            cls = type(self)
            if ((name == "_hooks" or name.startswith("_hook_dispatch"))
                    and threading.current_thread() is cls.pause_thread):
                cls.pause_thread = None  # pause once, after the value was read
                cls.paused.set()
                assert cls.release.wait(10), "the paused dispatch was never released"
            return value

    return PausingManager


def test_dispatch_that_read_hook_state_before_the_sweep_started_does_not_fail_open(home):
    _plugin(home, "keep_probe", _HOOK_PLUGIN)
    _enable(home, "keep_probe")
    manager_cls = _manager_pausing_first_hook_table_read()
    manager = manager_cls()
    manager.discover_and_load()
    assert manager.invoke_hook("pre_tool_call", tool_name="terminal", args={})[0]["action"] == "block"

    results = []
    dispatch = threading.Thread(target=lambda: results.append(
        manager.invoke_hook("pre_tool_call", tool_name="terminal", args={})))

    def resume_dispatch_after_unload_all():
        live = dict(manager._hooks)
        manager_cls.release.set()
        dispatch.join(10)
        return live, dispatch.is_alive()

    seen = _observe_after_unload(manager, resume_dispatch_after_unload_all)
    manager_cls.pause_thread = dispatch
    dispatch.start()
    try:
        assert manager_cls.paused.wait(10), "the dispatch never read hook-table state"
        manager.discover_and_load(force=True)  # starts while that dispatch is descheduled mid-lookup
    finally:
        manager_cls.release.set()
        dispatch.join(10)

    (live, still_running), = seen
    assert live == {}, "the native unload-all really ran before the dispatch resumed"
    assert not still_running, "the dispatch finished inside the sweep, before re-registration"
    assert results and results[0] and results[0][0]["action"] == "block", (
        "a dispatch that began just before a force sweep must not fail open after unload-all")


def test_disabled_plugin_hooks_end_when_the_sweep_commits(home):
    _plugin(home, "stays", _HOOK_PLUGIN)
    _plugin(home, "goes", _HOOK_PLUGIN)
    _enable(home, "stays", "goes")
    manager = _manager()
    assert len(manager.invoke_hook("pre_llm_call")) == 2

    _enable(home, "stays")
    seen = _observe_after_unload(manager, lambda: len(manager.invoke_hook("pre_llm_call")))
    manager.discover_and_load(force=True)

    assert seen == [2], "the last committed table serves until the new one is complete"
    after = manager.invoke_hook("pre_llm_call")
    assert len(after) == 1 and "stays" in after[0]["context"]
    assert "goes" not in manager._plugins or not manager._plugins["goes"].enabled


def test_failed_sweep_clears_the_view(home, monkeypatch):
    from hermes_cli.plugins import PluginManager

    _plugin(home, "keep_probe", _HOOK_PLUGIN)
    _enable(home, "keep_probe")
    manager = _manager()

    def boom(self):
        raise RuntimeError("scan failed")

    monkeypatch.setattr(PluginManager, "_discover_and_load_inner", boom)
    with pytest.raises(RuntimeError, match="scan failed"):
        manager.discover_and_load(force=True)
    assert manager._hook_dispatch_state[1] is None
    assert manager.invoke_hook("pre_llm_call") == []  # unchanged failure contract: nothing re-registered


def _task_plugin(home: Path, name: str, *, restart_poller: bool = False) -> None:
    _plugin(home, name, f"RESTART_POLLER = {restart_poller!r}\n" + _TASK_PLUGIN)


def test_supervised_task_survives_force_reload_and_stays_owned(home):
    _task_plugin(home, "worker")
    _enable(home, "worker")

    async def scenario():
        manager = _manager()
        release = asyncio.Event()

        async def body():
            await release.wait()
            return "FINISHED"

        held = manager._plugins["worker"].module.start(body())
        await asyncio.sleep(0)
        manager.discover_and_load(force=True)
        await asyncio.sleep(0)
        assert not held.done(), "force reload must not cancel in-flight work of a plugin that came back"
        kinds = [(r.kind, r.key) for r in manager._ownership_ledger["worker"]]
        assert ("background_task", "held-work") in kinds
        release.set()
        assert await held == "FINISHED"
        await asyncio.sleep(0)
        assert ("background_task", "held-work") not in [
            (r.kind, r.key) for r in manager._ownership_ledger.get("worker", [])]

        # Re-adopted handles stay cancellable by a later targeted unload (disable/uninstall path).
        second = manager._plugins["worker"].module.start(asyncio.Event().wait(), name="second")
        manager.discover_and_load(force=True)
        await asyncio.sleep(0)
        assert not second.done()
        assert manager.unload("worker") is True
        with pytest.raises(asyncio.CancelledError):
            await second

    asyncio.run(scenario())


def test_task_of_plugin_that_does_not_come_back_is_cancelled(home):
    _task_plugin(home, "worker")
    _enable(home, "worker")

    async def scenario():
        manager = _manager()
        held = manager._plugins["worker"].module.start(asyncio.Event().wait())
        await asyncio.sleep(0)
        _enable(home)
        manager.discover_and_load(force=True)
        with pytest.raises(asyncio.CancelledError):
            await held
        assert "worker" not in manager._ownership_ledger

    asyncio.run(scenario())


def test_restarted_same_name_task_supersedes_the_parked_one(home):
    _task_plugin(home, "poller_probe", restart_poller=True)
    _enable(home, "poller_probe")

    async def scenario():
        manager = _manager()
        first = [r for r in manager._ownership_ledger["poller_probe"] if r.kind == "background_task"]
        assert [r.key for r in first] == ["poller"]
        manager.discover_and_load(force=True)
        await asyncio.sleep(0)
        live = [r for r in manager._ownership_ledger["poller_probe"] if r.kind == "background_task" and r.active]
        assert [r.key for r in live] == ["poller"] and live[0] is not first[0]
        assert not first[0].active, "a register()-spawned loop must not run twice after a reload"
        manager.unload()

    asyncio.run(scenario())


def test_plain_unload_all_still_cancels_supervised_tasks(home):
    _task_plugin(home, "worker")
    _enable(home, "worker")

    async def scenario():
        manager = _manager()
        held = manager._plugins["worker"].module.start(asyncio.Event().wait())
        await asyncio.sleep(0)
        manager.unload()
        with pytest.raises(asyncio.CancelledError):
            await held

    asyncio.run(scenario())
