"""plugins.isolation: host — third-party plugins run in a per-profile host process, reached only via ctx."""

import json
import os
import sys
import time
from pathlib import Path
from typing import Any

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


@pytest.mark.platforms("posix")  # os._exit / SIGKILL process semantics
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


@pytest.mark.platforms("any")  # the host is a child process: its env/home resolution is per-OS
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



def test_managed_scope_pins_host_isolation_over_the_profiles_own_config(tmp_path, monkeypatch):
    """The isolated party must not be able to opt out: an operator pin in the managed scope
    (/etc/hermes/config.yaml) wins over a profile config that says in_process."""
    from hermes_cli import managed_scope
    from tools.registry import registry

    _home_with_plugins(tmp_path, monkeypatch, {"hostprobe": PROBE_PLUGIN}, isolation="in_process")
    managed = tmp_path / "managed"
    managed.mkdir()
    (managed / "config.yaml").write_text(yaml.safe_dump({"plugins": {"isolation": "host"}}), encoding="utf-8")
    monkeypatch.setenv("HERMES_MANAGED_DIR", str(managed))
    managed_scope.invalidate_managed_cache()
    manager = PluginManager()
    manager.discover_and_load()
    try:
        assert not any("hostprobe" in name for name in sys.modules)
        result = json.loads(registry.dispatch("hostprobe_pid", {}, scope=manager.scope_key))
        assert result["pid"] != os.getpid()
    finally:
        host = getattr(manager, "_plugin_host_instance", None)
        if host is not None:
            host.shutdown()
        managed_scope.invalidate_managed_cache()

MODEL_PROVIDER_PLUGIN = '''
import os
from providers import register_provider
from providers.base import ProviderProfile

class HostModel(ProviderProfile):
    def build_extra_body(self, *, session_id=None, **context):
        return {"pid": os.getpid(), "session_id": session_id}

register_provider(HostModel(name="hostmodel", base_url="https://hostmodel.example/v1", env_vars=("HOSTMODEL_KEY",)))
'''


@pytest.mark.platforms("any")  # the host is a child process: its env/home resolution is per-OS
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
        assert list((home / "cache" / "plugin_host" / "model-providers").glob("hostmodel-*.json"))
        assert not any("hostmodel" in name for name in sys.modules)
        body = profile.build_extra_body(session_id="s1")
        assert body["session_id"] == "s1" and body["pid"] != os.getpid()
    finally:
        host = getattr(plugins_mod.get_plugin_manager(), "_plugin_host_instance", None)
        if host is not None:
            host.shutdown()


ASYNC_PLUGIN = '''
import asyncio, json
S = {"type": "object", "properties": {}}

def register(ctx):
    async def whoami(args, **kw):  # async plugin code calling back into Hermes
        return json.dumps({"inner": json.loads(ctx.dispatch_tool("session_probe", {})),
                           "acomplete_awaitable": asyncio.iscoroutinefunction(ctx.llm.acomplete)})
    ctx.register_tool(name="async_whoami", toolset="asyncprobe", schema={"name": "async_whoami",
        "description": "d", "parameters": S}, handler=whoami, is_async=True)
'''


@pytest.mark.platforms("any")  # the host is a child process: its env/home resolution is per-OS
def test_async_plugin_code_calls_back_in_the_callers_session(tmp_path, monkeypatch):
    import contextvars
    from tools.registry import registry

    session = contextvars.ContextVar("session", default="<unset>")
    session.set("host-creator")  # whoever first touches the host must not leak into later calls
    registry.register(name="session_probe", toolset="asyncprobe", schema={"name": "session_probe",
                      "description": "d", "parameters": {"type": "object", "properties": {}}},
                      handler=lambda args, **kw: json.dumps({"session": session.get()}))
    _home_with_plugins(tmp_path, monkeypatch, {"asyncprobe": ASYNC_PLUGIN})
    manager = PluginManager()
    manager.discover_and_load()
    try:
        def call_as(name):
            def run():
                session.set(name)
                return json.loads(registry.dispatch("async_whoami", {}, scope=manager.scope_key))
            return contextvars.copy_context().run(run)

        assert call_as("session-B") == {"inner": {"session": "session-B"}, "acomplete_awaitable": True}
        assert call_as("session-C")["inner"] == {"session": "session-C"}
    finally:
        registry.deregister("session_probe")
        manager.unload()
        manager._plugin_host().shutdown()


MEMORY_PLUGIN = '''
import os
from agent.memory_provider import MemoryProvider

class Probe(MemoryProvider):
    level = 0  # a class-level default the provider changes later
    @property
    def name(self): return "memprobe"
    def is_available(self): return True
    def initialize(self, session_id, **kw): pass
    def get_tool_schemas(self): return []
    def whoami(self):
        self.level += 1
        return os.getpid()

def register(ctx):
    ctx.register_memory_provider(Probe())
'''


@pytest.mark.platforms("posix")  # SIGKILL
def test_hosted_memory_provider_stays_live_across_a_host_crash(tmp_path, monkeypatch):
    import signal
    from plugins.memory import load_memory_provider
    from plugins.memory.config_schema import get_provider_config_schema

    home = _home_with_plugins(tmp_path, monkeypatch, {})
    provider_dir = home / "plugins" / "memprobe"
    provider_dir.mkdir(parents=True)
    (provider_dir / "__init__.py").write_text(MEMORY_PLUGIN, encoding="utf-8")
    marker = tmp_path / "schema_pid"
    (provider_dir / "config_schema.py").write_text(
        f"import os, pathlib\npathlib.Path({str(marker)!r}).write_text(str(os.getpid()))\nCONFIG_SCHEMA = None\n",
        encoding="utf-8")
    host = plugins_mod.get_plugin_manager()._plugin_host()
    try:
        provider = load_memory_provider("memprobe")
        first_pid = provider.whoami()
        assert first_pid == host.pid != os.getpid()
        assert provider.level == 1  # read live from the plugin's object, not the class default
        # The provider's schema file is user code too: it runs in the host, never here.
        get_provider_config_schema("memprobe")
        assert int(marker.read_text(encoding="utf-8-sig")) == host.pid

        os.kill(first_pid, signal.SIGKILL)  # windows-footgun: ok — posix-only test (platforms marker)
        deadline = time.monotonic() + 10
        while host.alive and time.monotonic() < deadline:
            time.sleep(0.1)
        # The proxy Hermes already holds reloads the provider into the new host on next use.
        assert provider.whoami() not in {first_pid, os.getpid()}
    finally:
        host.shutdown()


@pytest.mark.platforms("any")  # structured return values cross the child-process wire
def test_hosted_memory_prefetch_preserves_structured_result(tmp_path, monkeypatch):
    from agent.memory_manager import MemoryManager
    from plugins.memory import load_memory_provider
    from hermes_cli.config import atomic_config_write

    home = _home_with_plugins(tmp_path, monkeypatch, {})
    config = yaml.safe_load((home / "config.yaml").read_text(encoding="utf-8"))
    config["memory"] = {"prefetch_spill_enabled": True}
    atomic_config_write(home / "config.yaml", config)
    provider_dir = home / "plugins" / "memprobe"
    provider_dir.mkdir(parents=True)
    (provider_dir / "__init__.py").write_text(
        '''
from agent.memory_provider import MemoryObservation, MemoryPrefetchResult, MemoryProvider

class Probe(MemoryProvider):
    @property
    def name(self): return "memprobe"
    def is_available(self): return True
    def initialize(self, session_id, **kwargs): pass
    def get_tool_schemas(self): return []
    def prefetch(self, query, *, session_id=""):
        return MemoryPrefetchResult(
            context=f"host context: {query}:{session_id}" + chr(10) + "x" * 10_100,
            observations=(MemoryObservation(
                source_kind="recall",
                schema="memprobe.recall",
                version=1,
                payload={"query": query},
            ),),
        )
''',
        encoding="utf-8",
    )

    memory_manager = MemoryManager()
    host = plugins_mod.get_plugin_manager()._plugin_host()
    try:
        provider = load_memory_provider("memprobe", register_skills=False)
        assert provider is not None
        memory_manager.add_provider(provider)

        result = memory_manager.prefetch_all_result(
            "question", session_id="session-a"
        )

        original_context = "host context: question:session-a\n" + "x" * 10_100
        assert result.context != original_context
        marker = "full content saved to "
        assert marker in result.context
        spill_path = Path(result.context.split(marker, 1)[1].split("]", 1)[0])
        assert spill_path.is_relative_to(home)
        assert spill_path.read_text(encoding="utf-8") == original_context + "\n"
        assert len(result.observations) == 1
        observation = result.observations[0]
        assert observation.source_kind == "recall"
        assert observation.schema == "memprobe.recall"
        assert observation.version == 1
        assert observation.provider == "memprobe"
        assert observation.payload == {"query": "question"}
    finally:
        memory_manager.shutdown_all()
        host.shutdown()


def test_plugin_host_wire_bounds_memory_observations_before_encoding():
    from agent.memory_provider import (
        MAX_MEMORY_OBSERVATION_BYTES,
        MAX_MEMORY_OBSERVATION_INSPECTED_CANDIDATES,
        MemoryObservation,
        MemoryPrefetchResult,
    )
    from hermes_cli.plugin_host_wire import Opaque, decode, encode

    observations = tuple(
        MemoryObservation(
            "recall",
            "memprobe.recall",
            1,
            "x" * (MAX_MEMORY_OBSERVATION_BYTES * 32) if index == 0 else {"i": index},
        )
        for index in range(MAX_MEMORY_OBSERVATION_INSPECTED_CANDIDATES * 4)
    )

    wire = encode(MemoryPrefetchResult(context="usable context", observations=observations))
    result = decode(wire)

    assert result.context == "usable context"
    assert len(result.observations) == MAX_MEMORY_OBSERVATION_INSPECTED_CANDIDATES + 1
    assert isinstance(result.observations[0].payload, Opaque)
    assert len(json.dumps(wire, separators=(",", ":"))) < 100_000


def test_plugin_host_wire_bounds_raw_dict_prefetch_before_generic_encoding():
    from agent.memory_provider import (
        MAX_MEMORY_OBSERVATION_BATCH_BYTES,
        MAX_MEMORY_OBSERVATION_INSPECTED_CANDIDATES,
        MemoryProvider,
        MemoryPrefetchResult,
    )
    from hermes_cli.plugin_host_child import HostRuntime
    from hermes_cli.plugin_host_wire import decode

    candidate = {
        "source_kind": "recall",
        "schema": "memprobe.recall",
        "version": 1,
        "payload": {"kept": True},
        "provider": "",
        **{f"extra-{index}": index for index in range(2_000)},
    }
    raw_result = {
        "context": "usable context",
        "observations": [candidate]
        + [
            {
                "source_kind": "recall",
                "schema": "memprobe.recall",
                "version": 1,
                "payload": {"index": index},
            }
            for index in range(1_000)
        ],
    }

    class Provider(MemoryProvider):
        @property
        def name(self):
            return "fixture"

        def is_available(self):
            return True

        def initialize(self, session_id, **kwargs):
            pass

        def get_tool_schemas(self):
            return []

        def prefetch(self, query="", *, session_id="") -> Any:
            return raw_result

    runtime = HostRuntime.__new__(HostRuntime)
    runtime.refs = {1: Provider()}
    runtime.owners = {1: "fixture"}
    wire = runtime.op_obj_invoke({"ref": 1, "method": "prefetch"})
    result = decode(wire)

    assert type(result) is MemoryPrefetchResult
    assert result.context == "usable context"
    assert len(result.observations) <= MAX_MEMORY_OBSERVATION_INSPECTED_CANDIDATES + 1
    assert result.observations[0].payload == {"kept": True}
    assert len(json.dumps(wire, separators=(",", ":")).encode()) < (
        MAX_MEMORY_OBSERVATION_BATCH_BYTES + 4_096
    )


def test_plugin_host_wire_enforces_aggregate_observation_bytes():
    from agent.memory_provider import (
        MAX_MEMORY_OBSERVATION_BATCH_BYTES,
        MAX_MEMORY_OBSERVATION_INSPECTED_CANDIDATES,
        MemoryObservation,
        MemoryPrefetchResult,
    )
    from hermes_cli.plugin_host_wire import encode

    result = MemoryPrefetchResult(
        context="context stays intact",
        observations=tuple(
            MemoryObservation("recall", "memprobe.recall", 1, {"text": "x" * 3_500})
            for _ in range(MAX_MEMORY_OBSERVATION_INSPECTED_CANDIDATES + 1)
        ),
    )

    wire = encode(result)

    assert wire["fields"]["context"] == "context stays intact"
    assert len(json.dumps(wire, separators=(",", ":")).encode()) <= (
        MAX_MEMORY_OBSERVATION_BATCH_BYTES + 2_048
    )


def test_plugin_host_wire_does_not_run_tuple_subclass_iteration():
    from agent.memory_provider import MemoryObservation, MemoryPrefetchResult
    from hermes_cli.plugin_host_wire import encode

    class IteratorBomb(tuple):
        def __iter__(self):
            raise AssertionError("the host wire must slice tuple storage directly")

    result = MemoryPrefetchResult(
        context="context",
        observations=IteratorBomb((MemoryObservation("recall", "fixture", 1, {}),)),
    )

    assert encode(result)


def test_plugin_host_wire_preserves_immutable_memory_observer_payload():
    from agent.memory_provider import MemoryObservation, _freeze_memory_observation_payload
    from hermes_cli.plugin_host_wire import decode, encode

    payload, _ = _freeze_memory_observation_payload({"nested": [{"x": 1}]})
    trusted = (
        MemoryObservation("recall", "fixture", 1, payload, provider="memprobe"),
    )
    decoded = decode(encode({"observations": trusted}))["observations"]

    assert type(decoded) is tuple
    assert type(decoded[0]) is MemoryObservation
    assert type(decoded[0].payload["nested"]) is tuple
    with pytest.raises(TypeError):
        decoded[0].payload["nested"][0]["x"] = 2


def test_unload_profile_manager_does_not_hold_registry_lock_during_teardown(
    tmp_path, monkeypatch
):
    import threading
    from types import SimpleNamespace
    from hermes_cli import plugins
    from hermes_cli.plugins_lifecycle import unload_plugin_manager_for_home

    home = (tmp_path / "profile").resolve()
    lock_available_during_unload = []

    def unload():
        acquired = threading.Event()

        def probe_lock():
            if plugins._plugin_managers_lock.acquire(timeout=2.0):
                acquired.set()
                plugins._plugin_managers_lock.release()

        thread = threading.Thread(target=probe_lock)
        thread.start()
        lock_available_during_unload.append(acquired.wait(timeout=2.0))
        thread.join(timeout=2.0)

    manager = SimpleNamespace(home_path=home, unload=unload)
    monkeypatch.setattr(plugins, "_plugin_managers_by_home", {home: manager})
    monkeypatch.setattr(plugins, "_plugin_manager", manager)
    monkeypatch.setattr(plugins, "_clear_plugin_submodules", lambda _manager: None)

    assert unload_plugin_manager_for_home(home)
    assert lock_available_during_unload == [True]


def test_unload_profile_manager_stops_its_plugin_host_after_disposal(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from hermes_cli import plugins
    from hermes_cli.plugins_lifecycle import unload_plugin_manager_for_home

    home = (tmp_path / "profile").resolve()
    order = []
    host = SimpleNamespace(shutdown=lambda: order.append("host"))
    manager = SimpleNamespace(
        home_path=home,
        _plugin_host_instance=host,
        unload=lambda: order.append("manager"),
    )
    monkeypatch.setattr(plugins, "_plugin_managers_by_home", {home: manager})
    monkeypatch.setattr(plugins, "_plugin_manager", manager)
    monkeypatch.setattr(plugins, "_clear_plugin_submodules", lambda _manager: None)

    assert unload_plugin_manager_for_home(home)
    assert order == ["manager", "host"]


def test_unload_profile_manager_stops_live_plugin_host_process(tmp_path, monkeypatch):
    from hermes_cli.plugins_lifecycle import unload_plugin_manager_for_home

    home = _home_with_plugins(tmp_path, monkeypatch, {"hostprobe": PROBE_PLUGIN})
    manager = PluginManager(scope_key=str(home))
    manager.discover_and_load()
    host = manager._plugin_host_instance
    process = host._proc
    assert process is not None and process.poll() is None

    monkeypatch.setattr(plugins_mod, "_plugin_managers_by_home", {home.resolve(): manager})
    monkeypatch.setattr(plugins_mod, "_plugin_manager", manager)

    try:
        assert unload_plugin_manager_for_home(home)
        assert process.poll() is not None
    finally:
        if process.poll() is None:
            host.shutdown()


def test_profile_manager_retires_observers_before_clearing_plugin_modules(
    tmp_path, monkeypatch
):
    import threading

    from agent import plugin_stream_hooks as psh
    from hermes_cli import plugins
    from hermes_cli.plugins_lifecycle import unload_plugin_manager_for_home

    home = (tmp_path / "profile").resolve()
    monkeypatch.setenv("HERMES_HOME", str(home))
    manager = PluginManager(scope_key=str(home))
    manager._discovered = True
    monkeypatch.setattr(plugins, "_plugin_managers_by_home", {home: manager})
    monkeypatch.setattr(plugins, "_plugin_manager", manager)

    running = threading.Event()
    release_running = threading.Event()
    state_closed = threading.Event()
    pending_ran_after_close = threading.Event()

    def observer(event_id):
        if event_id == "running":
            running.set()
            release_running.wait(timeout=5.0)
        elif event_id == "pending" and state_closed.is_set():
            pending_ran_after_close.set()

    plugins.PluginContext(
        plugins.PluginManifest(name="teardown-observer"), manager
    ).register_hook("memory_prefetch", observer)

    assert psh.enqueue_plugin_observer_hook("memory_prefetch", event_id="running")
    assert running.wait(timeout=5.0)
    dispatcher = psh._dispatchers_for("memory_prefetch")[0]
    assert psh.enqueue_plugin_observer_hook("memory_prefetch", event_id="pending")

    def clear_modules(_manager):
        state_closed.set()
        release_running.set()
        dispatcher.events.join()

    monkeypatch.setattr(plugins, "_clear_plugin_submodules", clear_modules)
    try:
        assert unload_plugin_manager_for_home(home)
        assert not pending_ran_after_close.is_set()
    finally:
        release_running.set()
        psh.shutdown_plugin_observer_dispatcher(timeout=5.0)
