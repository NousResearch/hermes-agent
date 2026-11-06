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
