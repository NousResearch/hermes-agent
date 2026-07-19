"""Tests for the bundled observability/nemo_relay plugin."""

from __future__ import annotations

import asyncio
import contextvars
import gc
import importlib
import json
import sys
import warnings
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from hermes_cli import lifecycle, plugins as plugin_api
from hermes_cli.observability import relay_runtime, relay_shared_metrics
from hermes_cli.plugins import PluginManager


REPO_ROOT = Path(__file__).resolve().parents[2]
PLUGIN_DIR = REPO_ROOT / "plugins" / "observability" / "nemo_relay"


class _FakeNemoRelay:
    def __init__(self):
        self.events = []
        self._callbacks = {}
        self._llm_starts = {}
        self._scope_serial = 0
        self._scope_context = contextvars.ContextVar(
            "fake_nemo_relay_scope", default=None
        )
        self.ScopeType = SimpleNamespace(Agent="agent", Function="function")
        self.scope = SimpleNamespace(
            push=self._scope_push,
            pop=self._scope_pop,
            event=self._scope_event,
        )
        self.llm = SimpleNamespace(
            call=self._llm_call,
            call_end=self._llm_call_end,
            execute=self._llm_execute,
        )
        self.tools = SimpleNamespace(
            call=self._tool_call,
            call_end=self._tool_call_end,
            execute=self._tool_execute,
            request_intercepts=self._tool_request_intercepts,
        )
        self.plugin = SimpleNamespace(
            initialize=self._plugin_initialize,
            clear=self._plugin_clear,
            activate_dynamic_plugins=self._plugin_activate_dynamic,
        )
        self.subscribers = SimpleNamespace(
            register=self._register_subscriber,
            deregister=self._deregister_subscriber,
            flush=self._flush_subscribers,
        )
        self.LLMRequest = _FakeLLMRequest
        self.AtofExporterConfig = _FakeAtofExporterConfig
        self.AtofExporterMode = SimpleNamespace(Append="append", Overwrite="overwrite")
        self.AtofExporter = self._make_atof_exporter
        self.AtifExporter = self._make_atif_exporter
        self.get_scope_stack = self._get_scope_stack

    def _scope_push(self, name, scope_type, **kwargs):
        self._scope_serial += 1
        handle = ("scope", name, self._scope_serial)
        self._scope_context.set(handle)
        self.events.append(("scope.push", name, scope_type, kwargs))
        return handle

    def _scope_pop(self, handle, **kwargs):
        self.events.append(("scope.pop", handle, kwargs))

    def _scope_event(self, name, **kwargs):
        self.events.append(("scope.event", name, kwargs))

    def _get_scope_stack(self):
        current = self._scope_context.get()
        self.events.append(("scope.sync", current))
        return current

    def _llm_call(self, name, request, **kwargs):
        handle = ("llm", name)
        self._llm_starts[handle] = kwargs
        self.events.append(("llm.call", name, request.content, kwargs))
        return handle

    def _llm_call_end(self, handle, response, **kwargs):
        self.events.append(("llm.call_end", handle, response, kwargs))
        start = self._llm_starts.pop(handle, {})
        event = SimpleNamespace(
            kind="scope",
            category="llm",
            name=handle[1],
            scope_category="end",
            category_profile={"model_name": start.get("model_name")},
            metadata={
                **(start.get("metadata") or {}),
                **(kwargs.get("metadata") or {}),
                "otel.status_code": "OK",
            },
            data=response,
        )
        for callback in list(self._callbacks.values()):
            callback(event)

    def _llm_execute(self, name, request, func, **kwargs):
        self.events.append(("llm.execute.start", name, request.content, kwargs))
        handle = self._llm_call(name, request, **kwargs)
        result = func(_FakeLLMRequest(request.headers, {"intercepted": True, **request.content}))
        self._llm_call_end(
            handle,
            result,
            **{key: value for key, value in kwargs.items() if key != "handle"},
        )
        self.events.append(("llm.execute.end", name, result, kwargs))
        return result

    def _tool_call(self, name, args, **kwargs):
        handle = ("tool", name)
        self.events.append(("tool.call", name, args, kwargs))
        return handle

    def _tool_call_end(self, handle, result, **kwargs):
        self.events.append(("tool.call_end", handle, result, kwargs))

    def _tool_execute(self, name, args, func, **kwargs):
        self.events.append(("tool.execute.start", name, args, kwargs))
        handle = self._tool_call(name, args, **kwargs)
        result = func(args)
        self._tool_call_end(
            handle,
            result,
            **{key: value for key, value in kwargs.items() if key != "handle"},
        )
        self.events.append(("tool.execute.end", name, result, kwargs))
        return result

    def _tool_request_intercepts(self, name, args):
        self.events.append(("tool.request_intercepts", name, args))
        return {"intercepted": True, **args}

    def _make_atof_exporter(self, config):
        return _FakeAtofExporter(self.events, config)

    def _make_atif_exporter(self, session_id, agent_name, agent_version, **kwargs):
        return _FakeAtifExporter(self.events, session_id, agent_name, agent_version, kwargs)

    async def _plugin_initialize(self, config):
        self.events.append(("plugin.initialize", config))
        return {"diagnostics": []}

    async def _plugin_clear(self):
        self.events.append(("plugin.clear",))

    async def _plugin_activate_dynamic(self, config, dynamic_plugins):
        self.events.append(("plugin.activate_dynamic", config, dynamic_plugins))
        return _FakePluginActivation(self.events)

    def _register_subscriber(self, name, callback):
        self._callbacks[name] = callback
        self.events.append(("subscribers.register", name))

    def _deregister_subscriber(self, name):
        self._callbacks.pop(name, None)
        self.events.append(("subscribers.deregister", name))

    def _flush_subscribers(self):
        self.events.append(("subscribers.flush",))


class _FakePluginActivation:
    def __init__(self, events):
        self.events = events
        self.report = {"diagnostics": []}

    async def close(self):
        self.events.append(("plugin.activation.close",))


class _FakeLLMRequest:
    def __init__(self, headers, content):
        self.headers = headers
        self.content = content


class _FakeAtofExporterConfig:
    def __init__(self):
        self.output_directory = ""
        self.filename = "events.jsonl"
        self.mode = "append"


class _FakeAtofExporter:
    def __init__(self, events, config):
        self.events = events
        self.config = config

    def register(self, name):
        self.events.append(("atof.register", name, self.config.output_directory, self.config.filename))

    def deregister(self, name):
        self.events.append(("atof.deregister", name, self.config.output_directory, self.config.filename))
        return True


class _FakeAtifExporter:
    def __init__(self, events, session_id, agent_name, agent_version, kwargs):
        self.events = events
        self.session_id = session_id
        self.agent_name = agent_name
        self.agent_version = agent_version
        self.kwargs = kwargs

    def register(self, name):
        self.events.append(("atif.register", name, self.session_id))

    def deregister(self, name):
        self.events.append(("atif.deregister", name, self.session_id))
        return True

    def export_json(self):
        self.events.append(("atif.export", self.session_id))
        return json.dumps({"session_id": self.session_id, "agent_name": self.agent_name})


def _fresh_plugin(monkeypatch, fake):
    existing = sys.modules.get("plugins.observability.nemo_relay")
    if existing is not None:
        existing.reset_for_tests()
    relay_shared_metrics._reset_for_tests()
    relay_runtime._reset_for_tests()
    monkeypatch.setattr(relay_runtime, "_load_nemo_relay", lambda: fake)
    monkeypatch.setitem(sys.modules, "nemo_relay", fake)
    sys.modules.pop("plugins.observability.nemo_relay", None)
    plugin = importlib.import_module("plugins.observability.nemo_relay")
    plugin.reset_for_tests()
    return plugin


def _enable_dynamic_plugin(tmp_path, monkeypatch) -> Path:
    plugins_toml = tmp_path / "plugins.toml"
    plugins_toml.write_text(
        f"""
version = 1

[[dynamic_plugins]]
plugin_id = "fixture"
kind = "rust_dynamic"
manifest_ref = "{(tmp_path / "fixture" / "relay-plugin.toml").as_posix()}"

[dynamic_plugins.config]
mode = "test"
""",
        encoding="utf-8",
    )
    monkeypatch.setenv("HERMES_NEMO_RELAY_PLUGINS_TOML", str(plugins_toml))
    return plugins_toml


def test_manifest_fields():
    data = yaml.safe_load((PLUGIN_DIR / "plugin.yaml").read_text())
    assert data["name"] == "nemo_relay"
    assert set(data["hooks"]) == {
        "on_session_start",
        "on_session_end",
        "on_session_finalize",
        "on_session_reset",
        "pre_llm_call",
        "post_llm_call",
        "pre_approval_request",
        "post_approval_response",
        "subagent_start",
        "subagent_stop",
    }


def test_nemo_relay_plugin_is_discoverable_as_bundled_plugin(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes_test"))

    manager = PluginManager()
    manager.discover_and_load()

    loaded = manager._plugins["observability/nemo_relay"]
    assert loaded.manifest.name == "nemo_relay"
    assert loaded.manifest.source == "bundled"
    assert not loaded.enabled


def test_nemo_relay_plugin_uses_nemo_relay_runtime(monkeypatch):
    fake_relay = _FakeNemoRelay()
    plugin = _fresh_plugin(monkeypatch, fake_relay)

    plugin.on_session_start(session_id="s1")

    assert any(event[0] == "scope.push" for event in fake_relay.events)


def test_nemo_relay_plugin_exports_core_managed_llm_and_tool_events(
    tmp_path,
    monkeypatch,
):
    from agent import relay_llm, relay_tools

    fake = _FakeNemoRelay()
    plugin = _fresh_plugin(monkeypatch, fake)
    monkeypatch.setenv("HERMES_NEMO_RELAY_ATOF_ENABLED", "1")
    monkeypatch.setenv("HERMES_NEMO_RELAY_ATOF_OUTPUT_DIRECTORY", str(tmp_path / "atof"))
    monkeypatch.setenv("HERMES_NEMO_RELAY_ATIF_ENABLED", "1")
    monkeypatch.setenv("HERMES_NEMO_RELAY_ATIF_OUTPUT_DIRECTORY", str(tmp_path / "atif"))

    base = {
        "session_id": "s1",
        "task_id": "t1",
        "turn_id": "turn-1",
        "telemetry_schema_version": "hermes.observer.v1",
    }
    plugin.on_session_start(**base, model="demo-model", platform="cli")
    coordinator = relay_runtime.SESSION_COORDINATOR
    lease = coordinator.acquire_conversation(
        profile_key=relay_runtime.current_profile_key(),
        session_id="s1",
        platform="cli",
    )
    turn = coordinator.begin_turn(
        lease,
        turn_id="turn-1",
        task_id="t1",
    )
    relay_llm.execute(
        {"messages": [{"role": "user", "content": "hi"}]},
        lambda request: {
            "assistant_message": {"role": "assistant", "content": "hello"},
            "request": request,
        },
        session_id="s1",
        name="openai",
        model_name="demo-model",
        metadata={"api_request_id": "api-1", "api_mode": "custom"},
    )
    relay_tools.execute(
        "read_file",
        {"path": "x"},
        lambda _args: {"ok": True},
        session_id="s1",
        metadata={"tool_call_id": "tool-1"},
    )
    coordinator.end_turn(turn, outcome="success")
    coordinator.release_conversation(lease)
    plugin.on_session_end(**base, completed=True, interrupted=False)
    plugin.on_session_finalize(**base, reason="shutdown")

    event_names = [event[0] for event in fake.events]
    assert "atof.register" in event_names
    assert "atif.register" in event_names
    assert event_names.count("llm.call") == 1
    assert event_names.count("llm.call_end") == 1
    assert event_names.count("tool.call") == 1
    assert event_names.count("tool.call_end") == 1
    assert "scope.pop" in event_names
    assert (tmp_path / "atif" / "hermes-atif-s1.json").exists()


def test_shared_metrics_and_rich_plugin_share_one_core_session(
    tmp_path,
    monkeypatch,
):
    from agent import relay_llm

    fake = _FakeNemoRelay()
    hermes_home = tmp_path / "hermes-home"
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    monkeypatch.setenv("HERMES_NEMO_RELAY_ATIF_ENABLED", "1")
    monkeypatch.setenv(
        "HERMES_NEMO_RELAY_ATIF_OUTPUT_DIRECTORY", str(tmp_path / "atif")
    )
    monkeypatch.setattr(
        "hermes_cli.config.load_config_readonly",
        lambda: {"telemetry": {"shared_metrics": {"enabled": True}}},
    )
    plugin = _fresh_plugin(monkeypatch, fake)
    manager = PluginManager()

    class _Context:
        def register_hook(self, name, callback):
            manager._hooks.setdefault(name, []).append(callback)

    plugin.register(_Context())
    monkeypatch.setattr(plugin_api, "_plugin_manager", manager)

    event = {
        "session_id": "s1",
        "task_id": "t1",
        "api_request_id": "api-1",
        "provider": "anthropic",
        "model": "claude-sonnet",
        "platform": "cli",
    }
    coordinator = relay_runtime.SESSION_COORDINATOR
    lease = coordinator.acquire_conversation(
        profile_key=relay_runtime.current_profile_key(),
        session_id="s1",
        platform="cli",
        model=event["model"],
    )
    lifecycle.invoke_hook("on_session_start", **event)
    turn = coordinator.begin_turn(
        lease,
        turn_id="turn-1",
        task_id="t1",
    )
    lifecycle.invoke_hook(
        "pre_api_request",
        **event,
        request={"body": {"messages": [{"role": "user", "content": "hi"}]}},
    )
    relay_llm.execute(
        {"messages": [{"role": "user", "content": "hi"}]},
        lambda _request: {
            "assistant_message": {"role": "assistant", "content": "hello"}
        },
        session_id="s1",
        name="anthropic",
        model_name="claude-sonnet",
        metadata={"api_request_id": "api-1", "api_mode": "custom"},
    )
    lifecycle.invoke_hook(
        "post_api_request",
        **event,
        response={"assistant_message": {"role": "assistant", "content": "hello"}},
    )
    coordinator.end_turn(turn, outcome="success")
    coordinator.release_conversation(lease)
    lifecycle.finalize_session(session_id="s1")

    session_pushes = [
        item
        for item in fake.events
        if item[0] == "scope.push" and item[1] == relay_runtime.SESSION_SCOPE
    ]
    assert len(session_pushes) == 1
    register_metrics = next(
        index
        for index, item in enumerate(fake.events)
        if item[0] == "subscribers.register"
        and item[1].startswith("hermes.nemo_relay.shared_metrics.")
    )
    register_atif = next(
        index for index, item in enumerate(fake.events) if item[0] == "atif.register"
    )
    open_session = fake.events.index(session_pushes[0])
    assert register_metrics < register_atif < open_session

    plugin_runtime = plugin._get_runtime()
    assert plugin_runtime is not None
    assert not plugin_runtime.sessions
    assert relay_runtime.get_session_handle("s1") is None
    packages = list(
        (hermes_home / "telemetry" / "shared_metrics" / "outbox").glob("*.json")
    )
    assert len(packages) == 1
    package = json.loads(packages[0].read_text(encoding="utf-8"))
    assert package["metrics"][0]["name"] == "hermes.model_call.count"
    assert package["metrics"][0]["value"] == 1
    assert (tmp_path / "atif" / "hermes-atif-s1.json").exists()


def test_nemo_relay_plugin_emits_approval_marks(monkeypatch):
    fake = _FakeNemoRelay()
    plugin = _fresh_plugin(monkeypatch, fake)

    plugin.on_pre_approval_request(session_id="s1", approval_id="approval-1", tool_name="shell")
    plugin.on_post_approval_response(session_id="s1", approval_id="approval-1", approved=True)

    mark_names = [event[1] for event in fake.events if event[0] == "scope.event"]
    assert "hermes.approval.request" in mark_names
    assert "hermes.approval.response" in mark_names


def test_nemo_relay_plugin_metadata_promotes_trajectory_and_subagent_ids(monkeypatch):
    fake = _FakeNemoRelay()
    plugin = _fresh_plugin(monkeypatch, fake)

    plugin.on_pre_llm_call(
        session_id="parent-session",
        task_id="task-1",
        turn_id="turn-1",
        telemetry_schema_version="hermes.observer.v1",
    )
    plugin.on_subagent_start(
        parent_session_id="parent-session",
        parent_turn_id="turn-1",
        parent_subagent_id="parent-sa",
        child_session_id="child-session",
        child_subagent_id="child-sa",
        child_role="leaf",
        telemetry_schema_version="hermes.observer.v1",
    )
    plugin.on_subagent_stop(
        parent_session_id="parent-session",
        parent_turn_id="turn-1",
        child_session_id="child-session",
        child_role="leaf",
        child_status="completed",
        telemetry_schema_version="hermes.observer.v1",
    )

    turn_mark = next(event for event in fake.events if event[0] == "scope.event" and event[1] == "hermes.turn.start")
    turn_metadata = turn_mark[2]["metadata"]
    assert turn_metadata["session_id"] == "parent-session"
    assert turn_metadata["trajectory_id"] == "parent-session"

    start_mark = next(event for event in fake.events if event[0] == "scope.event" and event[1] == "hermes.subagent.start")
    start_metadata = start_mark[2]["metadata"]
    assert start_metadata["parent_session_id"] == "parent-session"
    assert start_metadata["parent_trajectory_id"] == "parent-session"
    assert start_metadata["child_session_id"] == "child-session"
    assert start_metadata["child_trajectory_id"] == "child-session"
    assert start_metadata["child_subagent_id"] == "child-sa"
    assert start_metadata["child_role"] == "leaf"

    stop_mark = next(event for event in fake.events if event[0] == "scope.event" and event[1] == "hermes.subagent.stop")
    assert stop_mark[2]["metadata"]["child_status"] == "completed"


def test_nemo_relay_plugin_reuses_core_parented_child_scope_for_embedded_atif(monkeypatch):
    fake = _FakeNemoRelay()
    plugin = _fresh_plugin(monkeypatch, fake)

    plugin.on_session_start(session_id="parent-session")
    plugin.on_subagent_start(
        parent_session_id="parent-session",
        parent_turn_id="turn-1",
        child_session_id="child-session",
        child_subagent_id="child-sa",
        child_role="leaf",
        telemetry_schema_version="hermes.observer.v1",
    )
    runtime = plugin._get_runtime()
    assert runtime is not None
    assert runtime.host.get_session("child-session") is None
    runtime.host.register_subagent(
        {
            "parent_session_id": "parent-session",
            "child_session_id": "child-session",
        }
    )
    plugin.on_session_start(session_id="child-session")

    session_pushes = [
        event
        for event in fake.events
        if event[0] == "scope.push" and event[1] == relay_runtime.SESSION_SCOPE
    ]
    assert len(session_pushes) == 2
    child_push = session_pushes[1]
    child_kwargs = child_push[3]
    assert child_kwargs["handle"] == runtime.sessions["parent-session"].handle
    assert child_kwargs["metadata"]["nemo_relay_scope_role"] == "subagent"
    assert "session_id" not in child_kwargs["metadata"]
    assert "subagent_id" not in child_kwargs["metadata"]
    assert runtime.sessions["child-session"].parent_session_id == "parent-session"

    runtime.host.unregister_subagent({"child_session_id": "child-session"})
    assert runtime.host.get_session("child-session") is None
    child_close_index = max(
        index
        for index, event in enumerate(fake.events)
        if event[0] == "scope.pop" and event[1] == runtime.sessions["child-session"].handle
    )
    plugin.on_subagent_stop(
        parent_session_id="parent-session",
        child_session_id="child-session",
        child_status="completed",
    )

    assert "child-session" not in runtime.sessions
    assert runtime.host.get_session("child-session") is None
    stop_mark_index = next(
        index
        for index, event in enumerate(fake.events)
        if event[0] == "scope.event" and event[1] == "hermes.subagent.stop"
    )
    assert child_close_index < stop_mark_index


def test_nemo_relay_plugin_skips_embedded_child_atif_file_by_default(tmp_path, monkeypatch):
    fake = _FakeNemoRelay()
    plugin = _fresh_plugin(monkeypatch, fake)
    monkeypatch.setenv("HERMES_NEMO_RELAY_ATIF_ENABLED", "1")
    monkeypatch.setenv("HERMES_NEMO_RELAY_ATIF_OUTPUT_DIRECTORY", str(tmp_path / "atif"))

    plugin.on_session_start(session_id="parent-session")
    plugin.on_subagent_start(
        parent_session_id="parent-session",
        child_session_id="child-session",
        child_subagent_id="child-sa",
    )
    plugin.on_session_start(session_id="child-session")
    plugin.on_session_end(session_id="child-session")
    plugin.on_session_finalize(session_id="child-session")
    plugin.on_session_end(session_id="parent-session")
    plugin.on_session_finalize(session_id="parent-session")

    assert (tmp_path / "atif" / "hermes-atif-parent-session.json").exists()
    assert not (tmp_path / "atif" / "hermes-atif-child-session.json").exists()


def test_nemo_relay_plugin_can_write_embedded_child_atif_file_in_all_mode(tmp_path, monkeypatch):
    fake = _FakeNemoRelay()
    plugin = _fresh_plugin(monkeypatch, fake)
    monkeypatch.setenv("HERMES_NEMO_RELAY_ATIF_ENABLED", "1")
    monkeypatch.setenv("HERMES_NEMO_RELAY_ATIF_OUTPUT_DIRECTORY", str(tmp_path / "atif"))
    monkeypatch.setenv("HERMES_NEMO_RELAY_ATIF_SUBAGENT_EXPORT_MODE", "all")

    plugin.on_session_start(session_id="parent-session")
    plugin.on_subagent_start(
        parent_session_id="parent-session",
        child_session_id="child-session",
        child_subagent_id="child-sa",
    )
    plugin.on_session_start(session_id="child-session")
    plugin.on_session_end(session_id="child-session")
    plugin.on_session_finalize(session_id="child-session")
    plugin.on_session_end(session_id="parent-session")
    plugin.on_session_finalize(session_id="parent-session")

    assert (tmp_path / "atif" / "hermes-atif-parent-session.json").exists()
    assert (tmp_path / "atif" / "hermes-atif-child-session.json").exists()


def test_nemo_relay_plugin_can_initialize_plugins_toml(tmp_path, monkeypatch):
    fake = _FakeNemoRelay()
    plugin = _fresh_plugin(monkeypatch, fake)
    plugins_toml = tmp_path / "plugins.toml"
    atof_dir = tmp_path / "exports" / "events"
    atif_dir = tmp_path / "exports" / "trajectories"
    plugins_toml.write_text(
        f"""
version = 1

[[components]]
kind = "observability"
enabled = true

[components.config.atof]
enabled = true
output_directory = "{atof_dir}"

[components.config.atif]
enabled = true
output_directory = "{atif_dir}"
""",
        encoding="utf-8",
    )
    monkeypatch.setenv("HERMES_NEMO_RELAY_PLUGINS_TOML", str(plugins_toml))

    plugin.on_session_start(session_id="s1")

    assert any(event[0] == "plugin.initialize" for event in fake.events)
    assert not any(event[0] == "atof.register" for event in fake.events)
    assert atof_dir.is_dir()
    assert atif_dir.is_dir()


def test_nemo_relay_plugin_clears_plugins_toml_on_final_session_finalize_and_reinitializes(tmp_path, monkeypatch):
    fake = _FakeNemoRelay()
    plugin = _fresh_plugin(monkeypatch, fake)
    plugins_toml = tmp_path / "plugins.toml"
    plugins_toml.write_text(
        """
version = 1

[[components]]
kind = "observability"
enabled = true
""",
        encoding="utf-8",
    )
    monkeypatch.setenv("HERMES_NEMO_RELAY_PLUGINS_TOML", str(plugins_toml))

    plugin.on_session_start(session_id="s1")
    plugin.on_session_finalize(session_id="s1", reason="shutdown")
    plugin.on_session_start(session_id="s2")

    event_names = [event[0] for event in fake.events]
    assert event_names.count("plugin.initialize") == 2
    assert event_names.count("plugin.clear") == 1


def test_nemo_relay_plugin_activates_and_owns_dynamic_plugins(tmp_path, monkeypatch):
    from agent import relay_llm, relay_tools

    fake = _FakeNemoRelay()
    plugin = _fresh_plugin(monkeypatch, fake)
    _enable_dynamic_plugin(tmp_path, monkeypatch)
    monkeypatch.setenv("HERMES_NEMO_RELAY_ATIF_ENABLED", "1")
    monkeypatch.setenv("HERMES_NEMO_RELAY_ATIF_OUTPUT_DIRECTORY", str(tmp_path / "atif"))

    plugin.on_session_start(session_id="s1")
    runtime = plugin._get_runtime()
    assert runtime is not None
    assert runtime._plugin_activation is not None
    coordinator = relay_runtime.SESSION_COORDINATOR
    lease = coordinator.acquire_conversation(
        profile_key=relay_runtime.current_profile_key(),
        session_id="s1",
        platform="cli",
    )
    turn = coordinator.begin_turn(lease, turn_id="turn-1", task_id="task-1")
    llm_result = relay_llm.execute(
        {"messages": []},
        lambda request: {"request": request},
        session_id="s1",
        name="openai",
        model_name="fixture",
        metadata={"api_mode": "custom", "api_request_id": "api-1"},
    )
    tool_args = relay_runtime.apply_tool_request_intercepts(
        session_id="s1",
        tool_name="fixture-tool",
        args={"value": 1},
    )
    tool_result, final_args = relay_tools.execute(
        "fixture-tool",
        tool_args,
        lambda args: {"args": args},
        session_id="s1",
        metadata={"tool_call_id": "tool-1"},
    )
    coordinator.end_turn(turn, outcome="success")
    coordinator.release_conversation(lease)
    assert llm_result["request"]["intercepted"] is True
    assert tool_result["args"]["intercepted"] is True
    assert final_args["intercepted"] is True
    relay_runtime.SESSION_COORDINATOR.finalize_conversation(
        profile_key=relay_runtime.current_profile_key(),
        session_id="s1",
    )
    plugin.on_session_finalize(session_id="s1", reason="shutdown")
    assert runtime._plugin_activation is not None
    assert not any(event[0] == "plugin.activation.close" for event in fake.events)
    plugin.on_session_start(session_id="s2")
    plugin.on_session_finalize(session_id="s2", reason="shutdown")
    assert sum(event[0] == "plugin.activate_dynamic" for event in fake.events) == 1

    activation = next(event for event in fake.events if event[0] == "plugin.activate_dynamic")
    assert "dynamic_plugins" not in activation[1]
    assert activation[2] == [
        {
            "plugin_id": "fixture",
            "kind": "rust_dynamic",
            "manifest_ref": str(tmp_path / "fixture" / "relay-plugin.toml"),
            "config": {"mode": "test"},
        }
    ]
    event_names = [event[0] for event in fake.events]
    assert "plugin.clear" not in event_names
    assert event_names.index("subscribers.flush") < event_names.index("atif.export")
    assert event_names.index("atif.export") < event_names.index("atif.deregister")

    runtime.shutdown()
    event_names = [event[0] for event in fake.events]
    assert event_names.index("atif.deregister") < event_names.index("plugin.activation.close")


def test_nemo_relay_rejects_gateway_dynamic_config_with_actionable_diagnostic(
    tmp_path, monkeypatch, caplog
):
    fake = _FakeNemoRelay()
    plugin = _fresh_plugin(monkeypatch, fake)
    plugins_toml = tmp_path / "plugins.toml"
    plugins_toml.write_text(
        """
version = 1

[[plugins.dynamic]]
manifest = "plugins/fixture/relay-plugin.toml"

[plugins.dynamic.config]
mode = "test"
""",
        encoding="utf-8",
    )
    monkeypatch.setenv("HERMES_NEMO_RELAY_PLUGINS_TOML", str(plugins_toml))

    with caplog.at_level("ERROR"):
        plugin.on_session_start(session_id="s1")

    assert not any(event[0] == "plugin.activate_dynamic" for event in fake.events)
    initialize = next(event for event in fake.events if event[0] == "plugin.initialize")
    assert initialize[1] == {"version": 1}
    assert "does not expose the CLI lifecycle resolver" in caplog.text
    assert "Use Hermes-owned [[dynamic_plugins]]" in caplog.text


def test_nemo_relay_explicit_dynamic_paths_resolve_from_plugins_toml(tmp_path, monkeypatch):
    fake = _FakeNemoRelay()
    plugin = _fresh_plugin(monkeypatch, fake)
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    plugins_toml = config_dir / "plugins.toml"
    plugins_toml.write_text(
        """
version = 1

[[dynamic_plugins]]
plugin_id = "worker-fixture"
kind = "worker"
manifest_ref = "../plugins/worker/relay-plugin.toml"
environment_ref = "../environments/worker-fixture"
""",
        encoding="utf-8",
    )
    monkeypatch.setenv("HERMES_NEMO_RELAY_PLUGINS_TOML", str(plugins_toml))

    plugin.on_session_start(session_id="s1")

    activation = next(event for event in fake.events if event[0] == "plugin.activate_dynamic")
    assert activation[2] == [
        {
            "plugin_id": "worker-fixture",
            "kind": "worker",
            "manifest_ref": str(tmp_path / "plugins" / "worker" / "relay-plugin.toml"),
            "environment_ref": str(tmp_path / "environments" / "worker-fixture"),
            "config": {},
        }
    ]
    runtime = plugin._get_runtime()
    assert runtime is not None
    runtime.shutdown()


def test_relay_tool_request_rewrite_precedes_hermes_authorization_boundary(
    tmp_path,
    monkeypatch,
):
    from hermes_cli.middleware import apply_tool_request_middleware

    fake = _FakeNemoRelay()
    plugin = _fresh_plugin(monkeypatch, fake)
    _enable_dynamic_plugin(tmp_path, monkeypatch)
    plugin.on_session_start(session_id="s1")

    result = apply_tool_request_middleware(
        "fixture-tool",
        {"value": 1},
        session_id="s1",
        tool_call_id="tool-1",
    )

    assert result.payload == {"intercepted": True, "value": 1}
    assert result.trace[0] == {"source": "nemo_relay"}


def test_nemo_relay_plugin_activates_without_duplicate_execution_hooks(
    tmp_path, monkeypatch
):
    fake = _FakeNemoRelay()
    plugin = _fresh_plugin(monkeypatch, fake)
    _enable_dynamic_plugin(tmp_path, monkeypatch)
    registered_hooks = []

    class _Context:
        def register_hook(self, name, callback):
            del callback
            registered_hooks.append(name)

        def register_middleware(self, name, callback):
            del callback
            fake.events.append(("hermes.register_middleware", name))

    plugin.register(_Context())

    event_names = [event[0] for event in fake.events]
    assert "plugin.activate_dynamic" in event_names
    assert "hermes.register_middleware" not in event_names
    assert not {
        "pre_api_request",
        "post_api_request",
        "api_request_error",
        "pre_tool_call",
        "post_tool_call",
    }.intersection(registered_hooks)
    runtime = plugin._get_runtime()
    assert runtime is not None
    runtime.shutdown()


def test_nemo_relay_plugin_degrades_to_static_config_on_relay_0_5(
    tmp_path, monkeypatch, caplog
):
    fake = _FakeNemoRelay()
    delattr(fake.plugin, "activate_dynamic_plugins")
    plugin = _fresh_plugin(monkeypatch, fake)
    _enable_dynamic_plugin(tmp_path, monkeypatch)

    with caplog.at_level("WARNING"):
        plugin.on_session_start(session_id="s1")

    initialize = next(event for event in fake.events if event[0] == "plugin.initialize")
    assert "dynamic_plugins" not in initialize[1]
    assert not any(event[0] == "plugin.activate_dynamic" for event in fake.events)
    assert "available in NeMo Relay 0.6+" in caplog.text


def test_nemo_relay_plugin_rejects_invalid_dynamic_specs(tmp_path, monkeypatch, caplog):
    fake = _FakeNemoRelay()
    plugin = _fresh_plugin(monkeypatch, fake)
    plugins_toml = tmp_path / "plugins.toml"
    plugins_toml.write_text(
        """
version = 1
dynamic_plugins = [{ kind = "rust_dynamic", manifest_ref = "missing-id" }]
""",
        encoding="utf-8",
    )
    monkeypatch.setenv("HERMES_NEMO_RELAY_PLUGINS_TOML", str(plugins_toml))

    with caplog.at_level("WARNING"):
        plugin.on_session_start(session_id="s1")

    assert not any(event[0] == "plugin.activate_dynamic" for event in fake.events)
    assert "plugin_id is required" in caplog.text


def test_nemo_relay_plugin_rejects_entire_mixed_valid_invalid_dynamic_request(
    tmp_path, monkeypatch, caplog
):
    fake = _FakeNemoRelay()
    plugin = _fresh_plugin(monkeypatch, fake)
    plugins_toml = tmp_path / "plugins.toml"
    plugins_toml.write_text(
        f"""
version = 1

[[dynamic_plugins]]
plugin_id = "valid-fixture"
kind = "rust_dynamic"
manifest_ref = "{(tmp_path / "valid" / "relay-plugin.toml").as_posix()}"

[[dynamic_plugins]]
kind = "worker"
manifest_ref = "{(tmp_path / "invalid" / "relay-plugin.toml").as_posix()}"
""",
        encoding="utf-8",
    )
    monkeypatch.setenv("HERMES_NEMO_RELAY_PLUGINS_TOML", str(plugins_toml))

    with caplog.at_level("WARNING"):
        plugin.on_session_start(session_id="s1")

    assert not any(event[0] == "plugin.activate_dynamic" for event in fake.events)
    initialize = next(event for event in fake.events if event[0] == "plugin.initialize")
    assert "dynamic_plugins" not in initialize[1]
    assert "no dynamic plugins will be activated" in caplog.text


def test_nemo_relay_plugin_registers_shutdown_after_dynamic_retry(tmp_path, monkeypatch):
    fake = _FakeNemoRelay()
    activation_attempts = 0

    async def _flaky_activate(config, dynamic_plugins):
        nonlocal activation_attempts
        activation_attempts += 1
        fake.events.append(
            ("plugin.activate_dynamic.attempt", activation_attempts, config, dynamic_plugins)
        )
        if activation_attempts == 1:
            raise RuntimeError("temporary activation failure")
        return _FakePluginActivation(fake.events)

    fake.plugin.activate_dynamic_plugins = _flaky_activate
    plugin = _fresh_plugin(monkeypatch, fake)
    _enable_dynamic_plugin(tmp_path, monkeypatch)

    plugin.on_session_start(session_id="s1")
    plugin.on_session_finalize(session_id="s1")
    plugin.on_session_start(session_id="s2")

    runtime = plugin._get_runtime()
    assert runtime is not None
    assert runtime._plugin_activation is not None
    assert runtime._shutdown_registered is True
    assert activation_attempts == 2
    runtime.shutdown()
    assert any(event[0] == "plugin.activation.close" for event in fake.events)


def test_nemo_relay_plugin_attempts_activation_close_after_subscriber_flush_failure(
    tmp_path, monkeypatch, caplog
):
    fake = _FakeNemoRelay()

    def _failing_flush():
        fake.events.append(("subscribers.flush.failed",))
        raise RuntimeError("flush boom")

    fake.subscribers.flush = _failing_flush
    plugin = _fresh_plugin(monkeypatch, fake)
    _enable_dynamic_plugin(tmp_path, monkeypatch)
    plugin.on_session_start(session_id="s1")
    runtime = plugin._get_runtime()
    assert runtime is not None

    with caplog.at_level("WARNING"):
        runtime.shutdown()

    event_names = [event[0] for event in fake.events]
    assert event_names.count("subscribers.flush.failed") == 2
    flush_indices = [
        index for index, name in enumerate(event_names) if name == "subscribers.flush.failed"
    ]
    assert max(flush_indices) < event_names.index("plugin.activation.close")
    assert runtime._plugin_activation is None
    assert "subscriber flush failed: flush boom" in caplog.text


def test_nemo_relay_plugin_continues_shutdown_after_atif_export_failure(
    tmp_path, monkeypatch, caplog
):
    fake = _FakeNemoRelay()

    class _FailingAtifExporter(_FakeAtifExporter):
        def export_json(self):
            self.events.append(("atif.export.failed", self.session_id))
            raise OSError("disk full")

    fake.AtifExporter = lambda session_id, agent_name, agent_version, **kwargs: (
        _FailingAtifExporter(fake.events, session_id, agent_name, agent_version, kwargs)
    )
    plugin = _fresh_plugin(monkeypatch, fake)
    _enable_dynamic_plugin(tmp_path, monkeypatch)
    monkeypatch.setenv("HERMES_NEMO_RELAY_ATIF_ENABLED", "1")
    monkeypatch.setenv("HERMES_NEMO_RELAY_ATIF_OUTPUT_DIRECTORY", str(tmp_path / "atif"))
    plugin.on_session_start(session_id="s1")
    runtime = plugin._get_runtime()
    assert runtime is not None

    with caplog.at_level("WARNING"):
        runtime.shutdown()

    event_names = [event[0] for event in fake.events]
    assert event_names.index("atif.export.failed") < event_names.index("atif.deregister")
    assert event_names.index("atif.deregister") < event_names.index("plugin.activation.close")
    assert runtime._plugin_activation is None
    assert "ATIF export failed: disk full" in caplog.text


def test_nemo_relay_plugin_keeps_plugins_toml_active_while_other_sessions_remain(tmp_path, monkeypatch):
    fake = _FakeNemoRelay()
    plugin = _fresh_plugin(monkeypatch, fake)
    plugins_toml = tmp_path / "plugins.toml"
    plugins_toml.write_text(
        """
version = 1

[[components]]
kind = "observability"
enabled = true
""",
        encoding="utf-8",
    )
    monkeypatch.setenv("HERMES_NEMO_RELAY_PLUGINS_TOML", str(plugins_toml))

    plugin.on_session_start(session_id="parent")
    plugin.on_session_start(session_id="child")
    plugin.on_session_finalize(session_id="child", reason="shutdown")
    plugin.on_session_finalize(session_id="parent", reason="shutdown")

    event_names = [event[0] for event in fake.events]
    assert event_names.count("plugin.initialize") == 1
    assert event_names.count("plugin.clear") == 1


def test_nemo_relay_plugin_reinitializes_plugins_toml_inside_active_event_loop(tmp_path, monkeypatch):
    fake = _FakeNemoRelay()
    plugin = _fresh_plugin(monkeypatch, fake)
    plugins_toml = tmp_path / "plugins.toml"
    plugins_toml.write_text(
        """
version = 1

[[components]]
kind = "observability"
enabled = true
""",
        encoding="utf-8",
    )
    monkeypatch.setenv("HERMES_NEMO_RELAY_PLUGINS_TOML", str(plugins_toml))

    async def _drive() -> None:
        plugin.on_session_start(session_id="s1")
        plugin.on_session_finalize(session_id="s1", reason="shutdown")
        plugin.on_session_start(session_id="s2")
        await asyncio.sleep(0)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        asyncio.run(_drive())
        gc.collect()

    assert not any("was never awaited" in str(w.message) for w in caught)
    runtime = plugin._get_runtime()
    assert runtime is not None
    assert runtime._plugin_config_initialized is True
    scope_push_names = [event[1] for event in fake.events if event[0] == "scope.push"]
    assert relay_runtime.SESSION_SCOPE in scope_push_names


def test_nemo_relay_plugin_retries_plugins_toml_after_clear_failure(tmp_path, monkeypatch):
    fake = _FakeNemoRelay()
    initialize_calls = 0

    async def _counting_initialize(config):
        nonlocal initialize_calls
        initialize_calls += 1
        fake.events.append(("plugin.initialize.attempt", initialize_calls, config))
        return {"diagnostics": []}

    async def _failing_clear():
        fake.events.append(("plugin.clear.failed",))
        raise RuntimeError("boom")

    fake.plugin.initialize = _counting_initialize
    fake.plugin.clear = _failing_clear
    plugin = _fresh_plugin(monkeypatch, fake)
    plugins_toml = tmp_path / "plugins.toml"
    plugins_toml.write_text(
        """
version = 1

[[components]]
kind = "observability"
enabled = true
""",
        encoding="utf-8",
    )
    monkeypatch.setenv("HERMES_NEMO_RELAY_PLUGINS_TOML", str(plugins_toml))

    plugin.on_session_start(session_id="s1")
    plugin.on_session_finalize(session_id="s1", reason="shutdown")
    plugin.on_session_start(session_id="s2")

    event_names = [event[0] for event in fake.events]
    assert event_names.count("plugin.initialize.attempt") == 2
    assert event_names.count("plugin.clear.failed") == 1
    scope_push_names = [event[1] for event in fake.events if event[0] == "scope.push"]
    assert relay_runtime.SESSION_SCOPE in scope_push_names


def test_nemo_relay_plugin_disables_direct_atif_when_plugins_toml_owns_atif(tmp_path, monkeypatch):
    fake = _FakeNemoRelay()
    plugin = _fresh_plugin(monkeypatch, fake)
    plugins_toml = tmp_path / "plugins.toml"
    plugins_toml.write_text(
        f"""
version = 1

[[components]]
kind = "observability"
enabled = true

[components.config.atif]
enabled = true
output_directory = "{(tmp_path / "managed-atif").as_posix()}"
""",
        encoding="utf-8",
    )
    monkeypatch.setenv("HERMES_NEMO_RELAY_PLUGINS_TOML", str(plugins_toml))
    monkeypatch.setenv("HERMES_NEMO_RELAY_ATIF_ENABLED", "1")
    monkeypatch.setenv("HERMES_NEMO_RELAY_ATIF_OUTPUT_DIRECTORY", str(tmp_path / "direct-atif"))

    plugin.on_session_start(session_id="s1")
    plugin.on_session_finalize(session_id="s1", reason="shutdown")

    event_names = [event[0] for event in fake.events]
    assert "plugin.initialize" in event_names
    assert "plugin.clear" in event_names
    assert "atif.register" not in event_names
    assert not (tmp_path / "direct-atif" / "hermes-atif-s1.json").exists()


def test_nemo_relay_plugin_keeps_direct_atif_when_plugins_toml_init_fails(tmp_path, monkeypatch):
    fake = _FakeNemoRelay()

    async def _failing_initialize(config):
        fake.events.append(("plugin.initialize.failed", config))
        raise RuntimeError("boom")

    fake.plugin.initialize = _failing_initialize
    plugin = _fresh_plugin(monkeypatch, fake)
    plugins_toml = tmp_path / "plugins.toml"
    plugins_toml.write_text(
        f"""
version = 1

[[components]]
kind = "observability"
enabled = true

[components.config.atif]
enabled = true
output_directory = "{(tmp_path / "managed-atif").as_posix()}"
""",
        encoding="utf-8",
    )
    monkeypatch.setenv("HERMES_NEMO_RELAY_PLUGINS_TOML", str(plugins_toml))
    monkeypatch.setenv("HERMES_NEMO_RELAY_ATIF_ENABLED", "1")
    monkeypatch.setenv("HERMES_NEMO_RELAY_ATIF_OUTPUT_DIRECTORY", str(tmp_path / "direct-atif"))

    plugin.on_session_start(session_id="s1")
    plugin.on_session_finalize(session_id="s1", reason="shutdown")

    event_names = [event[0] for event in fake.events]
    assert "plugin.initialize.failed" in event_names
    assert "plugin.clear" not in event_names
    assert "atif.register" in event_names
    assert (tmp_path / "direct-atif" / "hermes-atif-s1.json").exists()


def test_nemo_relay_plugin_retries_plugins_toml_after_fallback_only_session_and_clears_direct_atof(
    tmp_path,
    monkeypatch,
):
    fake = _FakeNemoRelay()
    initialize_calls = 0

    async def _flaky_initialize(config):
        nonlocal initialize_calls
        initialize_calls += 1
        fake.events.append(("plugin.initialize.attempt", initialize_calls, config))
        if initialize_calls == 1:
            raise RuntimeError("boom")
        return {"diagnostics": []}

    fake.plugin.initialize = _flaky_initialize
    plugin = _fresh_plugin(monkeypatch, fake)
    plugins_toml = tmp_path / "plugins.toml"
    plugins_toml.write_text(
        f"""
version = 1

[[components]]
kind = "observability"
enabled = true

[components.config.atof]
enabled = true
output_directory = "{(tmp_path / "managed-atof").as_posix()}"
""",
        encoding="utf-8",
    )
    monkeypatch.setenv("HERMES_NEMO_RELAY_PLUGINS_TOML", str(plugins_toml))
    monkeypatch.setenv("HERMES_NEMO_RELAY_ATOF_ENABLED", "1")
    monkeypatch.setenv("HERMES_NEMO_RELAY_ATOF_OUTPUT_DIRECTORY", str(tmp_path / "direct-atof"))

    plugin.on_session_start(session_id="s1")
    plugin.on_session_finalize(session_id="s1", reason="shutdown")
    plugin.on_session_start(session_id="s2")

    runtime = plugin._get_runtime()
    assert runtime is not None
    assert runtime._plugin_config_initialized is True
    event_names = [event[0] for event in fake.events]
    assert event_names.count("plugin.initialize.attempt") == 2
    assert event_names.count("atof.register") == 1
    assert event_names.count("atof.deregister") == 1
