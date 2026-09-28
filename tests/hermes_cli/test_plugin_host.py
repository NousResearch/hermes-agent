"""plugins.isolation: host — third-party plugins run in a per-profile host process, reached only via ctx."""

import json
import os
import sys
import time

import hermes_yaml as yaml
import pytest

from hermes_cli import plugins as plugins_mod
from hermes_cli.plugins import PluginManager

PROBE_PLUGIN = '''
import json, os
from agent.image_gen_provider import ImageGenProvider

class Painter(ImageGenProvider):
    @property
    def name(self):
        return "hostpainter"
    def generate(self, prompt, aspect_ratio="landscape", **kw):
        return {"success": True, "image": f"{prompt}@{os.getpid()}"}

SEEN = []

def register(ctx):
    ctx.register_tool(name="hostprobe_pid", toolset="hostprobe", schema={"name": "hostprobe_pid",
        "description": "d", "parameters": {"type": "object", "properties": {}}},
        handler=lambda args, **kw: json.dumps({"pid": os.getpid(), "seen": SEEN, "x": args.get("x")}))
    ctx.register_tool(name="hostprobe_nested", toolset="hostprobe", schema={"name": "hostprobe_nested",
        "description": "d", "parameters": {"type": "object", "properties": {}}},
        handler=lambda args, **kw: ctx.dispatch_tool("hostprobe_pid", {"x": "nested"}))
    ctx.register_tool(name="hostprobe_crash", toolset="hostprobe", schema={"name": "hostprobe_crash",
        "description": "d", "parameters": {"type": "object", "properties": {}}},
        handler=lambda args, **kw: os._exit(3))
    ctx.register_hook("post_tool_call", lambda tool_name=None, **kw: SEEN.append(tool_name))
    ctx.register_image_gen_provider(Painter())
'''

PLATFORM_PLUGIN = '''
def register(ctx):
    ctx.register_platform("x", "X", adapter_factory=lambda cfg: None, check_fn=lambda: True)
'''


def _home_with_plugins(tmp_path, monkeypatch, plugins: dict, isolation="host"):
    home = tmp_path / ".hermes"
    home.mkdir(parents=True, exist_ok=True)
    for name, body in plugins.items():
        plugin_dir = home / "plugins" / name
        plugin_dir.mkdir(parents=True)
        (plugin_dir / "plugin.yaml").write_text(yaml.safe_dump({"name": name, "version": "1.0"}), encoding="utf-8")
        (plugin_dir / "__init__.py").write_text(body, encoding="utf-8")
    (home / "config.yaml").write_text(yaml.safe_dump(
        {"plugins": {"enabled": list(plugins), "isolation": isolation}}), encoding="utf-8")
    empty_bundled = tmp_path / "bundled"
    empty_bundled.mkdir()
    monkeypatch.setenv("HOME", str(tmp_path / "os-home"))
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(plugins_mod, "get_bundled_plugins_dir", lambda: empty_bundled)
    return home


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX process semantics (os._exit child)")
def test_plugin_runs_out_of_process_with_ctx_round_trips_and_survives_host_crash(tmp_path, monkeypatch):
    from agent import image_gen_registry
    from agent.image_gen_provider import ImageGenProvider
    from tools.registry import registry

    _home_with_plugins(tmp_path, monkeypatch, {"hostprobe": PROBE_PLUGIN, "platformprobe": PLATFORM_PLUGIN})
    manager = PluginManager()
    manager.discover_and_load()
    try:
        # The plugin's code never entered this interpreter, yet its registrations are ordinary.
        assert not any("hostprobe" in name for name in sys.modules)
        assert manager._plugins["hostprobe"].error is None
        first = json.loads(registry.dispatch("hostprobe_pid", {"x": 1}, scope=manager.scope_key))
        host_pid = first["pid"]
        assert host_pid != os.getpid()
        # A handler calling back into Hermes while Hermes waits on it (nested ctx.dispatch_tool).
        nested = json.loads(registry.dispatch("hostprobe_nested", {}, scope=manager.scope_key))
        assert nested["pid"] == host_pid and nested["x"] == "nested"
        # Hooks fire in the host; provider objects arrive as instances of the ABC Hermes checks.
        manager.invoke_hook("post_tool_call", tool_name="read_file", args={}, result="ok")
        assert "read_file" in json.loads(registry.dispatch("hostprobe_pid", {}, scope=manager.scope_key))["seen"]
        painter = image_gen_registry.get_provider("hostpainter")
        assert isinstance(painter, ImageGenProvider)
        assert painter.generate("cat") == {"success": True, "image": f"cat@{host_pid}"}
        # Live-object ctx surfaces fail the plugin with the boundary's reason, not a crash.
        assert "register_platform" in str(manager._plugins["platformprobe"].error)

        # A plugin killing its host costs one tool error; Hermes keeps running and the host restarts.
        crashed = registry.dispatch("hostprobe_crash", {}, scope=manager.scope_key)
        assert "plugin host" in crashed
        deadline, result = time.monotonic() + 15, ""
        while time.monotonic() < deadline:
            result = registry.dispatch("hostprobe_pid", {}, scope=manager.scope_key)
            if '"pid"' in result:
                break
            time.sleep(0.2)
        assert json.loads(result)["pid"] not in {host_pid, os.getpid()}
    finally:
        manager.unload()
        manager._plugin_host().shutdown()


def test_isolation_host_keeps_every_user_import_path_out_of_process(tmp_path, monkeypatch):
    from hermes_cli.plugin_isolation_audit import audit_plugin_dir
    from plugins import plugin_loader

    home = _home_with_plugins(tmp_path, monkeypatch, {"platformprobe": PLATFORM_PLUGIN, "fine": PROBE_PLUGIN})
    # Category loaders' in-process import refuses user code under host isolation (the backstop
    # behind the memory / context-engine / cron host routes).
    import logging
    assert plugin_loader.load_plugin_module(
        "_hermes_user_x.fine", home / "plugins" / "fine", parents=("_hermes_user_x",),
        logger=logging.getLogger("t"), synthetic_namespace="_hermes_user_x") is None
    assert "_hermes_user_x.fine" not in sys.modules
    # The static audit reads the same boundary table the host enforces.
    assert audit_plugin_dir(home / "plugins" / "fine").verdict == "host"
    blocked = audit_plugin_dir(home / "plugins" / "platformprobe")
    assert blocked.verdict == "in_process" and "register_platform" in blocked.reasons[0]


MODEL_PROVIDER_PLUGIN = '''
import os
from providers import register_provider
from providers.base import ProviderProfile

class HostModel(ProviderProfile):
    def build_extra_body(self, *, session_id=None, **context):
        return {"pid": os.getpid(), "session_id": session_id}

register_provider(HostModel(name="hostmodel", base_url="https://hostmodel.example/v1", env_vars=("HOSTMODEL_KEY",)))
'''


def test_model_provider_profile_data_is_local_and_overrides_run_in_the_host(tmp_path, monkeypatch):
    import providers

    home = _home_with_plugins(tmp_path, monkeypatch, {})
    plugin_dir = home / "plugins" / "model-providers" / "hostmodel"
    plugin_dir.mkdir(parents=True)
    (plugin_dir / "plugin.yaml").write_text("name: hostmodel\nkind: model-provider\n", encoding="utf-8")
    (plugin_dir / "__init__.py").write_text(MODEL_PROVIDER_PLUGIN, encoding="utf-8")
    profile = providers.get_provider_profile("hostmodel")
    try:
        assert isinstance(profile, providers.ProviderProfile)
        # Discovery never started a host: data came from the cached, credential-free extraction.
        assert profile.base_url == "https://hostmodel.example/v1" and tuple(profile.env_vars) == ("HOSTMODEL_KEY",)
        assert list((home / "cache" / "plugin_host" / "model-providers").glob("hostmodel.json"))
        assert not any("hostmodel" in name for name in sys.modules)
        body = profile.build_extra_body(session_id="s1")
        assert body["session_id"] == "s1" and body["pid"] != os.getpid()
    finally:
        host = plugins_mod.get_plugin_manager()._plugin_host_instance
        if host is not None:
            host.shutdown()
