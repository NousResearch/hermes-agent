"""Plugin RPC methods and session events work the same under ``plugins.isolation: host`` as in-process,
on a gateway serving two profiles: the plugin belongs to the secondary one."""

import os
import sys

import hermes_yaml as yaml
import pytest

from hermes_cli import plugins as plugins_mod
from hermes_cli.plugins import PluginManager
from tests.gateway.test_session_plugin_rpc import two_profile_runner

PLUGIN = '''
import os, threading
from hermes_cli.plugin_events import emit_plugin_session_event

CTX = []

def _state(params, *, session_id, **kw):
    ctx = CTX[0]
    return {"n": params.get("n"), "session_id": session_id, "pid": os.getpid(),
            "ctx": ctx.emit_to_session(session_id, "pane.changed", {"via": "ctx"}),
            "module": emit_plugin_session_event(ctx.plugin_id, session_id, "pane.changed", {"via": "module"})}

async def _async(params, *, session_id, **kw):
    return {"async": True, "session_id": session_id, "pid": os.getpid()}

def _from_thread(params, *, session_id, **kw):
    out = []
    worker = threading.Thread(target=lambda: out.append(CTX[0].emit_to_session(session_id, "pane.thread", {})))
    worker.start()
    worker.join(10)
    return out

def _not_json(params, *, session_id, **kw):
    try:
        emit_plugin_session_event(CTX[0].plugin_id, session_id, "pane.changed", {"x": float("nan")})
    except ValueError:
        return "ValueError"
    return "sent"

def register(ctx):
    CTX.append(ctx)
    ctx.register_rpc("pane.state", _state)
    ctx.register_rpc("pane.async", _async)
    ctx.register_rpc("pane.thread", _from_thread)
    ctx.register_rpc("pane.nan", _not_json)
'''


@pytest.mark.platforms("any")  # the host is a child process: its env/home resolution is per-OS
@pytest.mark.parametrize("isolation", ["host", "in_process"])
def test_plugin_rpc_and_session_events_in_either_isolation_mode(tmp_path, monkeypatch, isolation):
    from gateway import run_api
    from hermes_cli.plugins_dispatch import resolve_plugin_command_result
    from hermes_constants import reset_hermes_home_override, set_hermes_home_override

    launch, secondary = tmp_path / "launch", tmp_path / "secondary"
    plugin_dir = secondary / "plugins" / "hostrpc"
    plugin_dir.mkdir(parents=True)
    launch.mkdir()
    (plugin_dir / "plugin.yaml").write_text(yaml.safe_dump({"name": "hostrpc", "version": "1.0"}), encoding="utf-8")
    (plugin_dir / "__init__.py").write_text(PLUGIN, encoding="utf-8")
    (secondary / "config.yaml").write_text(yaml.safe_dump(
        {"plugins": {"enabled": ["hostrpc"], "isolation": isolation}}), encoding="utf-8")
    empty_bundled = tmp_path / "bundled"
    empty_bundled.mkdir()
    monkeypatch.setenv("HOME", str(tmp_path / "os-home"))
    monkeypatch.setenv("HERMES_HOME", str(launch))
    monkeypatch.setattr(plugins_mod, "get_bundled_plugins_dir", lambda: empty_bundled)
    runner, frames = two_profile_runner(launch, secondary)
    monkeypatch.setattr(run_api, "_ACTIVE_RUNNER", runner)

    manager = PluginManager(str(secondary))
    manager.discover_and_load()

    def call(name, **params):
        # As the gateway calls it: in the owning profile's scope (``async_owner_scope``).
        token = set_hermes_home_override(secondary)
        try:
            handler = manager._plugin_rpc_methods[f"plugin.hostrpc.{name}"]
            return resolve_plugin_command_result(handler(params, session_id="in-secondary"))
        finally:
            reset_hermes_home_override(token)

    try:
        assert manager._plugins["hostrpc"].error is None
        state = call("pane.state", n=3)
        assert (state["pid"] != os.getpid()) is (isolation == "host")
        assert (state["n"], state["session_id"], state["ctx"], state["module"]) == (3, "in-secondary", True, True)
        assert [f["params"]["payload"]["via"] for f in frames[secondary]] == ["ctx", "module"]
        assert call("pane.async") == {"async": True, "session_id": "in-secondary", "pid": state["pid"]}
        assert call("pane.thread") == [True]
        assert frames[secondary][-1]["params"]["type"] == "plugin.hostrpc.pane.thread"
        assert call("pane.nan") == "ValueError"
        assert frames[launch] == [] and len(frames[secondary]) == 3
        manager.unload("hostrpc")
        assert manager._plugin_rpc_methods == {}
    finally:
        manager.unload()
        host = getattr(manager, "_plugin_host_instance", None)
        if host is not None:
            host.shutdown()
        for name in [m for m in sys.modules if m.endswith("hostrpc")]:
            sys.modules.pop(name, None)


SENDER = '''
import asyncio
from hermes_cli.plugin_events import emit_plugin_session_event

CTX, UNLOADED, LATE = [], [], []

def _own(params, *, session_id, **kw):
    return emit_plugin_session_event(CTX[0].plugin_id, session_id, "pane.own", {})

async def _own_async(params, *, session_id, **kw):
    return emit_plugin_session_event(CTX[0].plugin_id, session_id, "pane.own_async", {})

def _foreign(params, *, session_id, **kw):
    return emit_plugin_session_event("not-loaded", session_id, "pane.changed", {})

def _after_unload(params, *, session_id, **kw):
    async def emit_later():
        while not UNLOADED:
            await asyncio.sleep(0.02)
        LATE.append(emit_plugin_session_event(CTX[0].plugin_id, session_id, "pane.late", {}))
    CTX[0].spawn_task(emit_later())
    return True

def register(ctx):
    CTX.append(ctx)
    ctx.on_unload(lambda: UNLOADED.append(1))
    for name, fn in (("own", _own), ("own_async", _own_async), ("foreign", _foreign), ("later", _after_unload)):
        ctx.register_rpc(f"pane.{name}", fn)
'''


@pytest.mark.platforms("any")  # the host is a child process: its env/home resolution is per-OS
@pytest.mark.parametrize("profiles", ["single", "two"])
def test_hosted_session_event_is_sent_only_as_the_live_plugin_running_it(tmp_path, monkeypatch, caplog, profiles):
    """A hosted plugin's module-level emit goes out under its own id only while it is loaded: a
    foreign ``plugin_id`` and a task still running after unload get False and publish nothing."""
    import time
    from types import SimpleNamespace

    from gateway import run_api
    from gateway.config import Platform
    from gateway.session_authorities import SessionAuthorities
    from hermes_cli.plugins_dispatch import resolve_plugin_command_result
    from hermes_constants import reset_hermes_home_override, set_hermes_home_override
    from tests.gateway.test_session_plugin_rpc import _live

    launch = tmp_path / "launch"
    home = launch if profiles == "single" else tmp_path / "secondary"
    plugin_dir = home / "plugins" / "hostrpc"
    plugin_dir.mkdir(parents=True)
    launch.mkdir(exist_ok=True)
    (plugin_dir / "plugin.yaml").write_text(yaml.safe_dump({"name": "hostrpc", "version": "1.0"}), encoding="utf-8")
    (plugin_dir / "__init__.py").write_text(SENDER, encoding="utf-8")
    (home / "config.yaml").write_text(yaml.safe_dump(
        {"plugins": {"enabled": ["hostrpc"], "isolation": "host"}}), encoding="utf-8")
    empty_bundled = tmp_path / "bundled"
    empty_bundled.mkdir()
    monkeypatch.setenv("HOME", str(tmp_path / "os-home"))
    monkeypatch.setenv("HERMES_HOME", str(launch))
    monkeypatch.setattr(plugins_mod, "get_bundled_plugins_dir", lambda: empty_bundled)
    registry, frames = SessionAuthorities(launch), {}
    for profile in dict.fromkeys((launch, home)):
        live = _live(Platform.TELEGRAM, "sess")
        frames[profile] = []
        live.event_stream.observers.add(frames[profile].append)
        registry.add(profile, SimpleNamespace(profile_id=str(profile), sessions={"sess": live}, agent=lambda ref: None))
    monkeypatch.setattr(run_api, "_ACTIVE_RUNNER", SimpleNamespace(session_authorities=registry))

    manager = PluginManager(str(home))
    manager.discover_and_load()

    def call(name):
        token = set_hermes_home_override(home)
        try:
            handler = manager._plugin_rpc_methods[f"plugin.hostrpc.{name}"]
            return resolve_plugin_command_result(handler({}, session_id="sess"))
        finally:
            reset_hermes_home_override(token)

    def sent():
        return [f["params"]["type"] for f in frames[home]]

    try:
        assert manager._plugins["hostrpc"].error is None
        assert (call("pane.own"), call("pane.own_async")) == (True, True)
        assert sent() == ["plugin.hostrpc.pane.own", "plugin.hostrpc.pane.own_async"]
        caplog.set_level("WARNING", logger="hermes_cli.plugins")
        assert call("pane.foreign") is False
        assert call("pane.later") is True
        manager.unload("hostrpc")
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline and sum("refused" in r.getMessage() for r in caplog.records) < 2:
            time.sleep(0.05)
        refused = [r.getMessage() for r in caplog.records if "refused" in r.getMessage()]
        assert len(refused) == 2 and "'not-loaded'" in refused[0] and "'hostrpc'" in refused[1]
        assert sent() == ["plugin.hostrpc.pane.own", "plugin.hostrpc.pane.own_async"]
        assert all(frames[p] == [] for p in frames if p != home)
    finally:
        manager.unload()
        host = getattr(manager, "_plugin_host_instance", None)
        if host is not None:
            host.shutdown()
        for name in [m for m in sys.modules if m.endswith("hostrpc")]:
            sys.modules.pop(name, None)
