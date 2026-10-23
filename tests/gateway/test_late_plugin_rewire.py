"""#87770 — plugins loaded after the gateway started get their platform handlers wired.

Invariants (each red on base):
* ``PluginManager.on_plugin_loaded`` fires after a discovery pass with an activation summary per plugin
  (what is live now vs deferred), through the real discovery path.
* ``rewire_plugin_handlers()`` is idempotent: a factory runs once per native client, however often
  the loaded event fires (a force re-discovery hands back NEW factory objects for the same plugin).
* Telegram: a late plugin handler is hoisted ahead of core's catch-all ``filters.COMMAND`` handler.
* Slack: a late ``register_slack_action_handler`` registers one Bolt listener, never two.
* The runner subscribes at boot and re-wires every live adapter on the loop.
"""

from __future__ import annotations

import asyncio
import os
import textwrap
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from gateway.config import PlatformConfig, Platform
from gateway.run_plugin_rewire import GatewayPluginRewireMixin
from hermes_cli.plugins import PluginContext, PluginManager, PluginManifest, discover_plugins, get_plugin_manager
from plugins.platforms.telegram.adapter import TelegramAdapter


def _write_plugin(home: Path, name: str = "late_cmd") -> None:
    d = home / "plugins" / name
    d.mkdir(parents=True)
    (d / "plugin.yaml").write_text(f"name: {name}\nversion: '0.1'\ndescription: t\nprovides_tools: [late_tool]\n")
    (d / "__init__.py").write_text(textwrap.dedent('''
        def register(ctx):
            ctx.register_command("late", lambda raw: "hi", description="late")
            ctx.register_platform_handler("telegram", lambda app, adapter: None)
    '''))
    (home / "config.yaml").write_text(f"plugins:\n  enabled: [{name}]\n")


def test_on_plugin_loaded_fires_with_activation_summary_after_a_late_load():
    home = Path(os.environ["HERMES_HOME"])
    manager = get_plugin_manager()
    discover_plugins()  # boot: plugin not on disk yet
    events: list = []
    unsubscribe = manager.on_plugin_loaded(events.append)
    _write_plugin(home)
    discover_plugins(force=True)  # the mid-run load
    late = [e for e in events[-1] if e["name"] == "late_cmd"]
    assert late == [{"name": "late_cmd", "key": "late_cmd",
                     "activated_now": {"gateway_commands": ["late"], "callbacks": ["telegram"]},
                     "deferred": {"tools": ["late_tool"]}}]
    assert len(events) == 1  # only the sweep that added it fires — and only for the newcomer
    discover_plugins(force=True)  # nothing new: no event (the gateway re-wire is not spammed)
    assert len(events) == 1
    unsubscribe()


def _factory_manager(calls: list, plugin: str = "p") -> PluginManager:
    mgr = PluginManager()
    ctx = PluginContext(manifest=PluginManifest(name=plugin, version="0", description=""), manager=mgr)

    def factory(native, adapter):
        calls.append(native)
    ctx.register_platform_handler("telegram", factory)
    return mgr


def test_rewire_is_idempotent_per_native_client():
    calls: list = []
    adapter = TelegramAdapter(PlatformConfig(enabled=True, token="t", extra={}))
    with patch("hermes_cli.plugins.get_plugin_manager", return_value=_factory_manager(calls)):
        adapter.rewire_plugin_handlers()  # before connect wired anything: nothing to do
        assert calls == []
        adapter._wire_plugin_handlers("app-1")
        adapter.rewire_plugin_handlers()
        adapter.rewire_plugin_handlers()
        assert calls == ["app-1"]
        # A force re-discovery re-registers a NEW factory object for the same plugin: still once.
        with patch("hermes_cli.plugins.get_plugin_manager", return_value=_factory_manager(calls)):
            adapter.rewire_plugin_handlers()
        assert calls == ["app-1"]
        adapter._wire_plugin_handlers("app-2")  # rebuilt native client: wired again, once
        assert calls == ["app-1", "app-2"]


class _PtbApp:
    """``Application.add_handler`` semantics that matter: append to ``handlers[group]``; PTB dispatches
    the first matching handler per group in that list order (tests/gateway mocks the real SDK)."""

    def __init__(self):
        self.handlers: dict = {}

    def add_handler(self, handler, group: int = 0):
        self.handlers.setdefault(group, []).append(handler)


def test_telegram_late_plugin_handler_precedes_core_catch_all():
    app = _PtbApp()
    adapter = TelegramAdapter(PlatformConfig(enabled=True, token="t", extra={}))
    plugin_handler = object()
    mgr = PluginManager()
    ctx = PluginContext(manifest=PluginManifest(name="p", version="0", description=""), manager=mgr)
    with patch("hermes_cli.plugins.get_plugin_manager", return_value=mgr):
        adapter._wire_plugin_handlers(app)  # connect(): plugins first (none yet), then core
        adapter._register_handlers(app)
        core = list(app.handlers[0])  # text catch-all, COMMAND catch-all, ..., CallbackQueryHandler
        ctx.register_platform_handler("telegram", lambda a, _adapter: a.add_handler(plugin_handler))
        adapter.rewire_plugin_handlers()
        adapter.rewire_plugin_handlers()
    group0 = app.handlers[0]
    assert group0 == [plugin_handler] + core  # ahead of every core catch-all, core order untouched
    assert app.handlers[99] and plugin_handler not in app.handlers[99]  # the group-99 observer stays alone


def test_slack_late_action_handler_registers_one_listener():
    from plugins.platforms.slack.adapter import SlackAdapter
    adapter = SlackAdapter(PlatformConfig(enabled=True, token="xoxb-t", extra={}))
    adapter._app = MagicMock()
    mgr = PluginManager()
    ctx = PluginContext(manifest=PluginManifest(name="p", version="0", description=""), manager=mgr)
    with patch("hermes_cli.plugins.get_plugin_manager", return_value=mgr):
        adapter._register_plugin_action_handlers()  # connect
        async def cb(ack, body, action):
            pass
        ctx.register_slack_action_handler("late_btn", cb)
        adapter.rewire_plugin_handlers()
        adapter.rewire_plugin_handlers()
    assert [c.args[0] for c in adapter._app.action.call_args_list] == ["late_btn"]


def test_runner_rewires_live_adapters_on_the_loop_when_plugins_load():
    class Runner(GatewayPluginRewireMixin):
        adapters = {Platform.TELEGRAM: MagicMock()}

    _write_plugin(Path(os.environ["HERMES_HOME"]))  # something must actually load for the event to fire
    mgr = PluginManager()

    async def scenario():
        runner = Runner()
        runner._subscribe_plugin_rewire(mgr)
        runner._subscribe_plugin_rewire(mgr)  # a served-profile rescan re-enters: one listener
        await asyncio.to_thread(mgr.discover_and_load, True)  # fires on another thread
        await asyncio.sleep(0)
        return runner.adapters[Platform.TELEGRAM].rewire_plugin_handlers.call_count

    assert asyncio.run(scenario()) == 1


class _ObservedTelegram(TelegramAdapter):
    def __init__(self):
        super().__init__(PlatformConfig(enabled=True, token="t", extra={}))
        self.factory_calls = []
        self.rewire_calls = 0

    def rewire_plugin_handlers(self):
        self.rewire_calls += 1
        super().rewire_plugin_handlers()


@pytest.mark.parametrize("secondary", [False, True])
def test_control_reload_runs_new_factory_of_an_existing_plugin_without_loaded_event(tmp_path, secondary):
    from gateway.run import _profile_runtime_scope
    from gateway.run_plugin_rewire import reload_plugins_verb

    home = tmp_path / "secondary" if secondary else Path(os.environ["HERMES_HOME"])
    profile = "secondary" if secondary else None
    _write_plugin(home)
    events = []
    adapter = _ObservedTelegram()
    other = _ObservedTelegram()
    with _profile_runtime_scope(home, hydrate_secrets=False):
        discover_plugins()
        manager = get_plugin_manager()
        manager.on_plugin_loaded(events.append)
        adapter._wire_plugin_handlers(object())
    (home / "plugins/late_cmd/__init__.py").write_text(textwrap.dedent('''
        from hermes_constants import get_hermes_home
        def added_factory(native, adapter):
            adapter.factory_calls.append(("added_after_reload", str(get_hermes_home())))
        def register(ctx):
            ctx.register_platform_handler("telegram", added_factory)
    '''), encoding="utf-8")

    class Runner(GatewayPluginRewireMixin):
        adapters = {Platform.TELEGRAM: other if secondary else adapter}
        _profile_adapters = {"secondary": {Platform.TELEGRAM: adapter if secondary else other}}
        _served_profile_homes = {"secondary": home} if secondary else {}

    async def scenario():
        runner = Runner()
        runner._subscribe_plugin_rewire(manager, profile, home)
        result = await asyncio.to_thread(reload_plugins_verb(runner, asyncio.get_running_loop()),
                                         {"home": str(home)})
        assert result["reloaded"] is True
        assert result["adapters_rewired"] == 1
        return result

    asyncio.run(scenario())
    assert events == []  # The newcomer-only event protocol is intentionally unchanged.
    assert adapter.factory_calls == [("added_after_reload", str(home))]
    assert other.rewire_calls == 0


def test_control_reload_new_plugin_keeps_event_but_does_not_double_rewire():
    from gateway.run_plugin_rewire import reload_plugins_verb

    home = Path(os.environ["HERMES_HOME"])
    discover_plugins()
    manager = get_plugin_manager()
    events = []
    manager.on_plugin_loaded(events.append)
    adapter = _ObservedTelegram()
    adapter._wire_plugin_handlers(object())
    _write_plugin(home)

    class Runner(GatewayPluginRewireMixin):
        adapters = {Platform.TELEGRAM: adapter}

    async def scenario():
        runner = Runner()
        observer_adapter = _ObservedTelegram()
        observer_runner = Runner()
        observer_runner.adapters = {Platform.TELEGRAM: observer_adapter}
        runner._subscribe_plugin_rewire(manager)
        observer_runner._subscribe_plugin_rewire(manager)
        result = await asyncio.to_thread(reload_plugins_verb(runner, asyncio.get_running_loop()))
        assert result["adapters_rewired"] == 1
        assert adapter.rewire_calls == observer_adapter.rewire_calls == 1
        assert len(events) == 1
        _write_plugin(home, "next_newcomer")
        await asyncio.to_thread(discover_plugins, force=True)
        await asyncio.sleep(0)
        assert adapter.rewire_calls == observer_adapter.rewire_calls == 2

    asyncio.run(scenario())
    assert len(events) == 2


@pytest.mark.parametrize("target", ["secondary", "default"])
@pytest.mark.parametrize("newcomer", [False, True])
def test_control_reload_named_launch_uses_served_owner_and_discovery_home(tmp_path, monkeypatch, target, newcomer):
    import threading
    from gateway.run import _profile_runtime_scope
    from gateway.run_plugin_rewire import reload_plugins_verb
    from hermes_constants import get_hermes_home, hermes_home_key

    homes = {"default": tmp_path / "primary", "secondary": tmp_path / "secondary"}
    launch = homes["secondary"]
    home = homes[target]
    monkeypatch.setenv("HERMES_HOME", str(launch))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    observed = {name: _ObservedTelegram() for name in homes}
    natives = {name: _PtbApp() for name in homes}
    managers = {}
    events = []
    for name, profile_home in homes.items():
        _write_plugin(profile_home)
        with _profile_runtime_scope(profile_home, hydrate_secrets=False):
            discover_plugins()
            managers[name] = get_plugin_manager()
            observed[name]._wire_plugin_handlers(natives[name])
    managers[target].on_plugin_loaded(events.append)
    plugin_name = "new_cmd" if newcomer else "late_cmd"
    if newcomer:
        _write_plugin(home, plugin_name)
    (home / f"plugins/{plugin_name}/__init__.py").write_text(textwrap.dedent('''
        import threading
        from hermes_constants import get_hermes_home
        from hermes_cli.plugins import get_plugin_manager
        def added_factory(native, adapter):
            native.add_handler("added_handler")
            adapter.factory_calls.append((str(get_hermes_home()),
                                         get_plugin_manager().scope_key, threading.get_ident()))
        def register(ctx):
            ctx.register_platform_handler("telegram", added_factory)
    '''), encoding="utf-8")

    class Runner(GatewayPluginRewireMixin):
        _primary_profile_name = "default"
        _served_profile_homes = homes
        adapters = {Platform.TELEGRAM: observed["default"]}
        _profile_adapters = {"secondary": {Platform.TELEGRAM: observed["secondary"]}}

    async def scenario():
        runner = Runner()
        for name, manager in managers.items():
            runner._subscribe_plugin_rewire(manager, None if name == "default" else name, homes[name])
        result = await asyncio.to_thread(reload_plugins_verb(runner, asyncio.get_running_loop()),
                                         {"home": str(home)})
        assert result["reloaded"] is True
        assert result["home"] == str(home)
        assert result["plugins"] == sorted(managers[target]._plugins)
        assert result["adapters_rewired"] == 1
        assert observed[target].factory_calls == [
            (str(home), hermes_home_key(home), threading.get_ident())]

    asyncio.run(scenario())
    other = "default" if target == "secondary" else "secondary"
    assert natives[target].handlers[0] == ["added_handler"]
    assert natives[other].handlers == {}
    assert observed[target].rewire_calls == 1
    assert observed[other].rewire_calls == 0
    assert len(events) == int(newcomer)  # Only new keys notify other subscribers.
    assert get_hermes_home() == launch


def test_control_reload_rejects_unserved_launch_home_when_served_map_is_known(tmp_path, monkeypatch):
    from gateway.run_plugin_rewire import reload_plugins_verb

    launch = tmp_path / "unserved"
    monkeypatch.setenv("HERMES_HOME", str(launch))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)

    class Runner(GatewayPluginRewireMixin):
        _primary_profile_name = "default"
        _served_profile_homes = {"default": tmp_path / "primary"}
        adapters = {Platform.TELEGRAM: _ObservedTelegram()}

    async def scenario():
        runner = Runner()
        result = await asyncio.to_thread(reload_plugins_verb(runner, asyncio.get_running_loop()),
                                         {"home": str(launch)})
        assert result == {"reloaded": False, "error": "home is not served by this gateway",
                          "home": str(launch)}
        assert runner.adapters[Platform.TELEGRAM].rewire_calls == 0

    asyncio.run(scenario())
