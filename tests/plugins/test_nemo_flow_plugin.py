"""Tests for the bundled observability/nemo_flow plugin."""

from __future__ import annotations

import builtins
import importlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
PLUGIN_DIR = REPO_ROOT / "plugins" / "observability" / "nemo_flow"


class _FakeNemoFlow:
    def __init__(self):
        self.events = []
        self.ScopeType = SimpleNamespace(Agent="agent")
        self.scope = SimpleNamespace(
            push=self._scope_push,
            pop=self._scope_pop,
            event=self._scope_event,
        )
        self.llm = SimpleNamespace(call=self._llm_call, call_end=self._llm_call_end)
        self.tools = SimpleNamespace(call=self._tool_call, call_end=self._tool_call_end)
        self.plugin = SimpleNamespace(initialize=self._plugin_initialize)
        self.LLMRequest = _FakeLLMRequest
        self.AtofExporterConfig = _FakeAtofExporterConfig
        self.AtofExporterMode = SimpleNamespace(Append="append", Overwrite="overwrite")
        self.AtofExporter = self._make_atof_exporter
        self.AtifExporter = self._make_atif_exporter

    def _scope_push(self, name, scope_type, **kwargs):
        handle = ("scope", name)
        self.events.append(("scope.push", name, scope_type, kwargs))
        return handle

    def _scope_pop(self, handle, **kwargs):
        self.events.append(("scope.pop", handle, kwargs))

    def _scope_event(self, name, **kwargs):
        self.events.append(("scope.event", name, kwargs))

    def _llm_call(self, name, request, **kwargs):
        handle = ("llm", name)
        self.events.append(("llm.call", name, request.content, kwargs))
        return handle

    def _llm_call_end(self, handle, response, **kwargs):
        self.events.append(("llm.call_end", handle, response, kwargs))

    def _tool_call(self, name, args, **kwargs):
        handle = ("tool", name)
        self.events.append(("tool.call", name, args, kwargs))
        return handle

    def _tool_call_end(self, handle, result, **kwargs):
        self.events.append(("tool.call_end", handle, result, kwargs))

    def _make_atof_exporter(self, config):
        return _FakeAtofExporter(self.events, config)

    def _make_atif_exporter(self, session_id, agent_name, agent_version, **kwargs):
        return _FakeAtifExporter(self.events, session_id, agent_name, agent_version, kwargs)

    async def _plugin_initialize(self, config):
        self.events.append(("plugin.initialize", config))
        return {"diagnostics": []}


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
        return json.dumps({"session_id": self.session_id, "agent_name": self.agent_name})


def _fresh_plugin(monkeypatch, fake):
    monkeypatch.setitem(sys.modules, "nemo_flow", fake)
    sys.modules.pop("plugins.observability.nemo_flow", None)
    plugin = importlib.import_module("plugins.observability.nemo_flow")
    plugin.reset_for_tests()
    return plugin


def test_manifest_fields():
    data = yaml.safe_load((PLUGIN_DIR / "plugin.yaml").read_text())
    assert data["name"] == "nemo_flow"
    assert "pre_api_request" in data["hooks"]
    assert "api_request_error" in data["hooks"]
    assert "subagent_start" in data["hooks"]


def test_nemo_flow_plugin_emits_llm_tool_and_exports_atif(tmp_path, monkeypatch):
    fake = _FakeNemoFlow()
    plugin = _fresh_plugin(monkeypatch, fake)
    monkeypatch.setenv("HERMES_NEMO_FLOW_ATOF_ENABLED", "1")
    monkeypatch.setenv("HERMES_NEMO_FLOW_ATOF_OUTPUT_DIRECTORY", str(tmp_path / "atof"))
    monkeypatch.setenv("HERMES_NEMO_FLOW_ATIF_ENABLED", "1")
    monkeypatch.setenv("HERMES_NEMO_FLOW_ATIF_OUTPUT_DIRECTORY", str(tmp_path / "atif"))

    base = {
        "session_id": "s1",
        "task_id": "t1",
        "turn_id": "turn-1",
        "telemetry_schema_version": "hermes.observer.v1",
    }
    plugin.on_session_start(**base, model="demo-model", platform="cli")
    plugin.on_pre_api_request(
        **base,
        api_request_id="api-1",
        provider="openai",
        model="demo-model",
        request={"method": "POST", "body": {"messages": [{"role": "user", "content": "hi"}]}},
    )
    plugin.on_post_api_request(
        **base,
        api_request_id="api-1",
        response={"assistant_message": {"role": "assistant", "content": "hello"}},
    )
    plugin.on_pre_tool_call(**base, tool_name="read_file", tool_call_id="tool-1", args={"path": "x"})
    plugin.on_post_tool_call(**base, tool_name="read_file", tool_call_id="tool-1", result='{"ok": true}', status="ok")
    plugin.on_session_end(**base, completed=True, interrupted=False)
    plugin.on_session_finalize(**base, reason="shutdown")

    event_names = [event[0] for event in fake.events]
    assert "atof.register" in event_names
    assert "atif.register" in event_names
    assert "llm.call" in event_names
    assert "llm.call_end" in event_names
    assert "tool.call" in event_names
    assert "tool.call_end" in event_names
    assert "scope.pop" in event_names
    assert (tmp_path / "atif" / "hermes-atif-s1.json").exists()


def test_nemo_flow_plugin_metadata_promotes_trajectory_and_subagent_ids(monkeypatch):
    fake = _FakeNemoFlow()
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


def test_nemo_flow_plugin_can_initialize_plugins_toml(tmp_path, monkeypatch):
    fake = _FakeNemoFlow()
    plugin = _fresh_plugin(monkeypatch, fake)
    plugins_toml = tmp_path / "plugins.toml"
    plugins_toml.write_text(
        """
version = 1

[[components]]
kind = "observability"
enabled = true

[components.config.atof]
enabled = true
output_directory = "events"
""",
        encoding="utf-8",
    )
    monkeypatch.setenv("HERMES_NEMO_FLOW_PLUGINS_TOML", str(plugins_toml))

    plugin.on_session_start(session_id="s1")

    assert any(event[0] == "plugin.initialize" for event in fake.events)
    assert not any(event[0] == "atof.register" for event in fake.events)


def test_nemo_flow_plugin_noops_without_dependency(monkeypatch):
    monkeypatch.delitem(sys.modules, "nemo_flow", raising=False)
    sys.modules.pop("plugins.observability.nemo_flow", None)
    plugin = importlib.import_module("plugins.observability.nemo_flow")
    plugin.reset_for_tests()

    real_import = builtins.__import__

    def blocked_import(name, *args, **kwargs):
        if name == "nemo_flow":
            raise ModuleNotFoundError("No module named 'nemo_flow'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", blocked_import)

    plugin.on_pre_api_request(session_id="s1", api_request_id="api-1")
    plugin.on_post_api_request(session_id="s1", api_request_id="api-1")
