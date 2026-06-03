"""Tests for the bundled observability/nemo_relay plugin."""

from __future__ import annotations

import builtins
import importlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import yaml

from hermes_cli.plugins import PluginManager


REPO_ROOT = Path(__file__).resolve().parents[2]
PLUGIN_DIR = REPO_ROOT / "plugins" / "observability" / "nemo_relay"


class _FakeNemoRelay:
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
    monkeypatch.setitem(sys.modules, "nemo_relay", fake)
    sys.modules.pop("plugins.observability.nemo_relay", None)
    plugin = importlib.import_module("plugins.observability.nemo_relay")
    plugin.reset_for_tests()
    return plugin


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
        "pre_api_request",
        "post_api_request",
        "api_request_error",
        "pre_tool_call",
        "post_tool_call",
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


def test_nemo_relay_plugin_emits_llm_tool_and_exports_atif(tmp_path, monkeypatch):
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


def test_nemo_relay_plugin_closes_api_span_on_error(monkeypatch):
    fake = _FakeNemoRelay()
    plugin = _fresh_plugin(monkeypatch, fake)
    base = {
        "session_id": "s1",
        "task_id": "t1",
        "turn_id": "turn-1",
        "telemetry_schema_version": "hermes.observer.v1",
    }

    plugin.on_pre_api_request(
        **base,
        api_request_id="api-err",
        provider="openai",
        model="demo-model",
        request={"body": {"messages": [{"role": "user", "content": "hi"}]}},
    )
    plugin.on_api_request_error(
        **base,
        api_request_id="api-err",
        error={"type": "RateLimitError", "message": "rate limited"},
        retryable=True,
        reason="rate_limit",
    )

    call_end = next(event for event in fake.events if event[0] == "llm.call_end")
    assert call_end[1] == ("llm", "openai")
    assert call_end[2] == {"error": {"type": "RateLimitError", "message": "rate limited"}}
    assert call_end[3]["data"]["reason"] == "rate_limit"
    assert not plugin._get_runtime().sessions["s1"].llm_spans


def test_nemo_relay_plugin_emits_approval_marks(monkeypatch):
    fake = _FakeNemoRelay()
    plugin = _fresh_plugin(monkeypatch, fake)

    plugin.on_pre_approval_request(session_id="s1", approval_id="approval-1", tool_name="shell")
    plugin.on_post_approval_response(session_id="s1", approval_id="approval-1", approved=True)

    mark_names = [event[1] for event in fake.events if event[0] == "scope.event"]
    assert "hermes.approval.request" in mark_names
    assert "hermes.approval.response" in mark_names


def test_nemo_relay_plugin_emits_unmatched_fallback_marks(monkeypatch):
    fake = _FakeNemoRelay()
    plugin = _fresh_plugin(monkeypatch, fake)

    plugin.on_post_api_request(session_id="s1", api_request_id="missing-api", response={"ok": True})
    plugin.on_api_request_error(
        session_id="s1",
        api_request_id="missing-api",
        error={"type": "TimeoutError", "message": "timed out"},
    )
    plugin.on_post_tool_call(session_id="s1", tool_call_id="missing-tool", result={"ok": True})

    mark_names = [event[1] for event in fake.events if event[0] == "scope.event"]
    assert "hermes.api.response.unmatched" in mark_names
    assert "hermes.api.error" in mark_names
    assert "hermes.tool.response.unmatched" in mark_names


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


def test_nemo_relay_plugin_reparents_child_session_scope_for_embedded_atif(monkeypatch):
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
    plugin.on_session_start(session_id="child-session")

    child_push = next(
        event
        for event in fake.events
        if event[0] == "scope.push" and event[1] == "hermes-session-child-session"
    )
    child_kwargs = child_push[3]
    assert child_kwargs["handle"] == ("scope", "hermes-session-parent-session")
    assert child_kwargs["metadata"]["session_id"] == "child-session"
    assert child_kwargs["metadata"]["trajectory_id"] == "child-session"
    assert child_kwargs["metadata"]["nemo_relay_scope_role"] == "subagent"
    assert child_kwargs["metadata"]["subagent_id"] == "child-sa"
    assert child_kwargs["metadata"]["parent_session_id"] == "parent-session"


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


def test_nemo_relay_plugin_noops_without_dependency(monkeypatch):
    monkeypatch.delitem(sys.modules, "nemo_relay", raising=False)
    sys.modules.pop("plugins.observability.nemo_relay", None)
    plugin = importlib.import_module("plugins.observability.nemo_relay")
    plugin.reset_for_tests()

    real_import = builtins.__import__

    def blocked_import(name, *args, **kwargs):
        if name == "nemo_relay":
            raise ModuleNotFoundError(f"No module named {name!r}")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", blocked_import)

    plugin.on_pre_api_request(session_id="s1", api_request_id="api-1")
    plugin.on_post_api_request(session_id="s1", api_request_id="api-1")
