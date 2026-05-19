from __future__ import annotations

import importlib
import json
import sys
from types import SimpleNamespace

import pytest

from hermes_cli.plugins import PluginManager


class _FakeHandle:
    def __init__(self, kind: str, name: str) -> None:
        self.kind = kind
        self.name = name


class _FakeNemoFlow:
    def __init__(self) -> None:
        self.scope_pushes = []
        self.scope_pops = []
        self.events = []
        self.llm_calls = []
        self.llm_ends = []
        self.tool_calls = []
        self.tool_ends = []
        self.atif_exporters = []
        self.atof_exporters = []
        self.ScopeType = SimpleNamespace(Agent="agent", Function="function")
        self.AtofExporterMode = SimpleNamespace(Append="append", Overwrite="overwrite")
        self.scope = SimpleNamespace(
            push=self._scope_push, pop=self._scope_pop, event=self._event
        )
        self.llm = SimpleNamespace(call=self._llm_call, call_end=self._llm_call_end)
        self.tools = SimpleNamespace(call=self._tool_call, call_end=self._tool_call_end)
        self.LLMRequest = _FakeLLMRequest
        self.AtofExporterConfig = _FakeAtofExporterConfig
        self.AtofExporter = self._make_atof_exporter
        self.AtifExporter = self._make_atif_exporter
        self.plugin = _FakePlugin()

    def _scope_push(self, name, scope_type, **kwargs):
        handle = _FakeHandle("scope", name)
        self.scope_pushes.append((name, scope_type, kwargs, handle))
        return handle

    def _scope_pop(self, handle, **kwargs):
        self.scope_pops.append((handle, kwargs))

    def _event(self, name, **kwargs):
        self.events.append((name, kwargs))

    def _llm_call(self, name, request, **kwargs):
        handle = _FakeHandle("llm", name)
        self.llm_calls.append((name, request, kwargs, handle))
        return handle

    def _llm_call_end(self, handle, response, **kwargs):
        self.llm_ends.append((handle, response, kwargs))

    def _tool_call(self, name, args, **kwargs):
        handle = _FakeHandle("tool", name)
        self.tool_calls.append((name, args, kwargs, handle))
        return handle

    def _tool_call_end(self, handle, result, **kwargs):
        self.tool_ends.append((handle, result, kwargs))

    def _make_atof_exporter(self, config):
        exporter = _FakeAtofExporter(config)
        self.atof_exporters.append(exporter)
        return exporter

    def _make_atif_exporter(self, *args, **kwargs):
        exporter = _FakeAtifExporter(*args, **kwargs)
        self.atif_exporters.append(exporter)
        return exporter


class _FakeLLMRequest:
    def __init__(self, headers, content) -> None:
        self.headers = headers
        self.content = content


class _FakeSubscription:
    def __init__(self, name) -> None:
        self.name = name
        self.deregistered = False

    def deregister(self):
        self.deregistered = True
        return True

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.deregister()


class _FakeTelemetryV1:
    EVENT_SCHEMA_VERSION = "nemo_flow.telemetry.v1"

    def __init__(self) -> None:
        self.registered = []

    def register_observer(self, name, callback, *, error_policy="log"):
        subscription = _FakeSubscription(name)
        self.registered.append((name, callback, error_policy, subscription))
        return subscription


class _FakePlugin:
    def __init__(self) -> None:
        self.initialized = []
        self.clear_count = 0

    async def initialize(self, config):
        self.initialized.append(config)
        return {"diagnostics": []}

    def clear(self):
        self.clear_count += 1


class _FakeAtofExporterConfig:
    def __init__(self) -> None:
        self.output_directory = ""
        self.filename = "nemo-flow-atof.jsonl"
        self.mode = "append"


class _FakeAtofExporter:
    def __init__(self, config) -> None:
        self.config = config
        self.registered = []
        self.flushed = False

    def register(self, name):
        self.registered.append(name)

    def force_flush(self):
        self.flushed = True


class _FakeAtifExporter:
    def __init__(self, session_id, agent_name, agent_version, **kwargs) -> None:
        self.session_id = session_id
        self.agent_name = agent_name
        self.agent_version = agent_version
        self.kwargs = kwargs
        self.registered = []
        self.deregistered = []

    def register(self, name):
        self.registered.append(name)

    def deregister(self, name):
        self.deregistered.append(name)
        return True

    def export_json(self):
        return json.dumps({
            "session_id": self.session_id,
            "agent_name": self.agent_name,
        })


class _FakeLangfuseContext:
    def __init__(self, observation) -> None:
        self.observation = observation

    def __enter__(self):
        return self.observation

    def __exit__(self, exc_type, exc, tb) -> None:
        return None


class _FakeLangfuseObservation:
    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs
        self.children = []
        self.trace_io = {}
        self.updates = []
        self.ended = False

    def set_trace_io(self, **kwargs):
        self.trace_io.update(kwargs)

    def update(self, **kwargs):
        self.updates.append(kwargs)

    def end(self):
        self.ended = True

    def start_observation(self, **kwargs):
        child = _FakeLangfuseObservation(**kwargs)
        self.children.append(child)
        return child


class _FakeLangfuseClient:
    def __init__(self) -> None:
        self.roots = []
        self.flush_count = 0

    def create_trace_id(self, seed):
        return f"trace-{seed}"

    def start_as_current_observation(self, **kwargs):
        observation = _FakeLangfuseObservation(**kwargs)
        self.roots.append(observation)
        return _FakeLangfuseContext(observation)

    def flush(self):
        self.flush_count += 1


@pytest.fixture
def fake_nemo(monkeypatch):
    import hermes_cli.config as config_mod
    import hermes_cli.nemo_flow_telemetry as telemetry

    fake = _FakeNemoFlow()
    monkeypatch.setitem(sys.modules, "nemo_flow", fake)
    monkeypatch.setenv("HERMES_NEMO_FLOW_TELEMETRY", "1")
    monkeypatch.setattr(
        config_mod,
        "load_config",
        lambda: {"telemetry": {"nemo_flow": {"enabled": True}}},
    )
    telemetry.reset_for_tests()
    yield fake
    telemetry.reset_for_tests()


def test_translates_api_and_tool_hooks_to_nemo_flow_spans(fake_nemo):
    from hermes_cli.nemo_flow_telemetry import record_hook

    base = {
        "session_id": "session-1",
        "task_id": "task-1",
        "turn_id": "turn-1",
        "telemetry_schema_version": "hermes.observer.v1",
    }
    record_hook(
        "pre_api_request",
        {
            **base,
            "api_request_id": "api-1",
            "provider": "test-provider",
            "model": "test-model",
            "base_url": "https://example.test/v1",
            "api_mode": "chat",
            "api_call_count": 3,
            "started_at": 100.0,
            "request": {
                "body": {
                    "messages": [{"role": "user", "content": "hi"}],
                    "api_key": "secret",
                }
            },
        },
    )
    record_hook(
        "post_api_request",
        {
            **base,
            "api_request_id": "api-1",
            "ended_at": 101.0,
            "api_duration": 1.0,
            "response": {"assistant_message": {"content": "hello"}},
            "finish_reason": "stop",
            "response_model": "test-model",
            "usage": {"input_tokens": 1, "output_tokens": 2},
            "assistant_content_chars": 5,
            "assistant_tool_call_count": 0,
        },
    )
    record_hook(
        "pre_tool_call",
        {
            **base,
            "tool_name": "web_search",
            "tool_call_id": "tool-1",
            "args": {"query": "hermes"},
        },
    )
    record_hook(
        "post_tool_call",
        {
            **base,
            "tool_name": "web_search",
            "tool_call_id": "tool-1",
            "result": '{"results":["ok"]}',
            "status": "ok",
        },
    )

    assert [push[1] for push in fake_nemo.scope_pushes[:2]] == ["agent", "function"]
    assert fake_nemo.llm_calls[0][0] == "test-provider"
    assert fake_nemo.llm_calls[0][1].content["api_key"] == "<redacted>"
    assert fake_nemo.llm_ends[0][1]["assistant_message"]["content"] == "hello"
    api_end_metadata = fake_nemo.llm_ends[0][2]["metadata"]
    assert api_end_metadata["usage"] == {"input_tokens": 1, "output_tokens": 2}
    assert api_end_metadata["finish_reason"] == "stop"
    assert api_end_metadata["response_model"] == "test-model"
    assert api_end_metadata["api_duration"] == 1.0
    assert api_end_metadata["assistant_content_chars"] == 5
    assert api_end_metadata["assistant_tool_call_count"] == 0
    assert fake_nemo.llm_calls[0][2]["metadata"]["base_url"] == (
        "https://example.test/v1"
    )
    assert fake_nemo.llm_calls[0][2]["metadata"]["api_mode"] == "chat"
    assert fake_nemo.llm_calls[0][2]["metadata"]["api_call_count"] == 3
    assert fake_nemo.tool_calls[0][0] == "web_search"
    assert fake_nemo.tool_ends[0][1] == {"results": ["ok"]}


def test_blocked_tool_closes_started_tool_span(fake_nemo):
    from hermes_cli.nemo_flow_telemetry import record_hook

    payload = {
        "session_id": "session-1",
        "turn_id": "turn-1",
        "tool_name": "terminal",
        "tool_call_id": "tool-blocked",
        "args": {"command": "rm -rf /tmp/nope"},
        "telemetry_schema_version": "hermes.observer.v1",
    }
    record_hook("pre_tool_call", payload)
    record_hook(
        "post_tool_call",
        {
            **payload,
            "result": '{"error":"Blocked"}',
            "status": "blocked",
            "error_type": "ToolBlocked",
            "error_message": "Blocked",
        },
    )

    assert len(fake_nemo.tool_calls) == 1
    assert len(fake_nemo.tool_ends) == 1
    assert fake_nemo.tool_ends[0][2]["metadata"]["status"] == "blocked"


def test_plugin_dispatcher_keeps_hook_return_values_unchanged(fake_nemo):
    manager = PluginManager()

    def blocker(**_kwargs):
        return {"action": "block", "message": "policy"}

    manager._hooks["pre_tool_call"] = [blocker]
    results = manager.invoke_hook(
        "pre_tool_call",
        session_id="s1",
        turn_id="t1",
        tool_name="terminal",
        args={"command": "echo hi"},
        telemetry_schema_version="hermes.observer.v1",
    )

    assert results == [{"action": "block", "message": "policy"}]
    assert fake_nemo.tool_calls[0][0] == "terminal"


def test_public_telemetry_facade_uses_stable_nemo_flow_api(fake_nemo):
    import hermes_cli.nemo_flow_telemetry as bridge
    import hermes_cli.telemetry as telemetry

    fake_nemo.telemetry_v1 = _FakeTelemetryV1()
    bridge.reset_for_tests()

    events = []
    assert telemetry.is_enabled()
    assert telemetry.is_available()

    handle = telemetry.register_observer(
        "test-observer",
        events.append,
        error_policy="ignore",
    )

    assert handle.name == "test-observer"
    assert fake_nemo.telemetry_v1.registered[0][:3] == (
        "test-observer",
        events.append,
        "ignore",
    )
    assert handle.deregister()


def test_public_telemetry_facade_noops_without_stable_nemo_flow_api(fake_nemo):
    import hermes_cli.nemo_flow_telemetry as bridge
    import hermes_cli.telemetry as telemetry

    bridge.reset_for_tests()

    assert telemetry.is_enabled()
    assert not telemetry.is_available()
    handle = telemetry.register_observer("test-observer", lambda _event: None)

    assert handle.name == "test-observer"
    assert not handle.deregister()


def test_plugins_toml_initializes_nemo_flow_observability_plugin(
    fake_nemo,
    monkeypatch,
    tmp_path,
):
    import hermes_cli.nemo_flow_telemetry as bridge
    from hermes_cli.nemo_flow_telemetry import record_hook

    plugins_toml = tmp_path / "plugins.toml"
    output_dir = tmp_path / "nf-logs"
    plugins_toml.write_text(
        f"""
version = 1

[[components]]
kind = "observability"
enabled = true

[components.config]
version = 1

[components.config.atof]
enabled = true
output_directory = "{output_dir}"
filename = "events.jsonl"
mode = "overwrite"

[components.config.atif]
enabled = true
output_directory = "{output_dir}"
filename_template = "trajectory-{{session_id}}.json"
""",
        encoding="utf-8",
    )
    monkeypatch.setenv("HERMES_NEMO_FLOW_PLUGINS_TOML", str(plugins_toml))
    monkeypatch.setenv("HERMES_NEMO_FLOW_ATIF_DIR", str(tmp_path / "direct-atif"))
    bridge.reset_for_tests()

    record_hook(
        "on_session_start",
        {
            "session_id": "session-plugins-toml",
            "telemetry_schema_version": "hermes.observer.v1",
        },
    )

    assert len(fake_nemo.plugin.initialized) == 1
    config = fake_nemo.plugin.initialized[0]
    assert config["version"] == 1
    assert config["components"][0]["kind"] == "observability"
    assert config["components"][0]["config"]["atof"]["filename"] == "events.jsonl"
    assert config["components"][0]["config"]["atif"]["filename_template"] == (
        "trajectory-{session_id}.json"
    )
    assert fake_nemo.atof_exporters == []
    assert fake_nemo.atif_exporters == []

    bridge.reset_for_tests()
    assert fake_nemo.plugin.clear_count == 1


def test_default_plugins_toml_initializes_without_suppressing_manual_atof(
    fake_nemo,
    monkeypatch,
    tmp_path,
):
    import hermes_cli.nemo_flow_telemetry as bridge
    from hermes_cli.nemo_flow_telemetry import record_hook

    monkeypatch.setenv("HERMES_NEMO_FLOW_ATOF_DIR", str(tmp_path))
    bridge.reset_for_tests()

    record_hook(
        "on_session_start",
        {
            "session_id": "session-default-plugins-toml",
            "telemetry_schema_version": "hermes.observer.v1",
        },
    )

    assert len(fake_nemo.plugin.initialized) == 1
    config = fake_nemo.plugin.initialized[0]
    assert config["components"][0]["kind"] == "observability"
    assert config["components"][0]["config"]["atof"]["enabled"] is False
    assert config["components"][0]["config"]["atif"]["enabled"] is False
    assert len(fake_nemo.atof_exporters) == 1
    assert fake_nemo.atof_exporters[0].config.output_directory == str(tmp_path)

    bridge.reset_for_tests()
    assert fake_nemo.plugin.clear_count == 1


def test_default_plugins_toml_and_langfuse_plugin_run_together(fake_nemo, monkeypatch):
    import hermes_cli.nemo_flow_telemetry as bridge

    sys.modules.pop("plugins.observability.langfuse", None)
    langfuse_plugin = importlib.import_module("plugins.observability.langfuse")
    fake_langfuse = _FakeLangfuseClient()
    monkeypatch.setattr(langfuse_plugin, "_get_langfuse", lambda: fake_langfuse)
    bridge.reset_for_tests()

    manager = PluginManager()
    manager._hooks["pre_api_request"] = [langfuse_plugin.on_pre_llm_request]
    manager._hooks["post_api_request"] = [langfuse_plugin.on_post_llm_call]
    manager._hooks["pre_tool_call"] = [langfuse_plugin.on_pre_tool_call]
    manager._hooks["post_tool_call"] = [langfuse_plugin.on_post_tool_call]

    base = {
        "session_id": "session-langfuse",
        "task_id": "task-langfuse",
        "turn_id": "turn-langfuse",
        "platform": "cli",
        "provider": "openai",
        "model": "gpt-test",
        "base_url": "https://example.test/v1",
        "api_mode": "chat",
        "telemetry_schema_version": "hermes.observer.v1",
    }
    messages = [{"role": "user", "content": "read README.md"}]

    manager.invoke_hook(
        "pre_api_request",
        **base,
        api_request_id="api-1",
        api_call_count=1,
        messages=messages,
        request={"body": {"messages": messages}},
    )
    manager.invoke_hook(
        "post_api_request",
        **base,
        api_request_id="api-1",
        api_call_count=1,
        finish_reason="tool_calls",
        usage={"input_tokens": 5, "output_tokens": 1},
        assistant_content_chars=0,
        assistant_tool_call_count=1,
        response={"assistant_message": {"tool_calls": [{"id": "tool-1"}]}},
    )
    manager.invoke_hook(
        "pre_tool_call",
        **base,
        tool_name="read_file",
        tool_call_id="tool-1",
        args={"path": "README.md"},
    )
    manager.invoke_hook(
        "post_tool_call",
        **base,
        tool_name="read_file",
        tool_call_id="tool-1",
        args={"path": "README.md"},
        result='{"content":"1|ok","total_lines":1,"file_size":2,'
        '"is_binary":false,"is_image":false}',
        status="ok",
    )

    followup_messages = [
        *messages,
        {"role": "tool", "tool_call_id": "tool-1", "content": "1|ok"},
    ]
    manager.invoke_hook(
        "pre_api_request",
        **base,
        api_request_id="api-2",
        api_call_count=2,
        messages=followup_messages,
        request={"body": {"messages": followup_messages}},
    )
    manager.invoke_hook(
        "post_api_request",
        **base,
        api_request_id="api-2",
        api_call_count=2,
        finish_reason="stop",
        usage={"input_tokens": 6, "output_tokens": 3},
        assistant_content_chars=9,
        assistant_tool_call_count=0,
        response={"assistant_message": {"content": "done"}},
    )

    assert len(fake_nemo.plugin.initialized) == 1
    assert fake_nemo.plugin.initialized[0]["components"][0]["kind"] == "observability"
    assert [call[0] for call in fake_nemo.llm_calls] == ["openai", "openai"]
    assert fake_nemo.tool_calls[0][0] == "read_file"

    assert len(fake_langfuse.roots) == 1
    root = fake_langfuse.roots[0]
    child_types = [child.kwargs["as_type"] for child in root.children]
    assert child_types == ["generation", "tool", "generation"]
    assert all(child.ended for child in root.children)
    assert root.ended
    assert fake_langfuse.flush_count == 1


def test_atif_exporter_writes_on_session_finalize(monkeypatch, tmp_path):
    import hermes_cli.config as config_mod
    import hermes_cli.nemo_flow_telemetry as telemetry

    fake = _FakeNemoFlow()
    monkeypatch.setitem(sys.modules, "nemo_flow", fake)
    monkeypatch.setenv("HERMES_NEMO_FLOW_TELEMETRY", "1")
    monkeypatch.setenv("HERMES_NEMO_FLOW_ATIF_DIR", str(tmp_path))
    monkeypatch.setattr(
        config_mod,
        "load_config",
        lambda: {"telemetry": {"nemo_flow": {"enabled": True}}},
    )
    telemetry.reset_for_tests()

    telemetry.record_hook(
        "on_session_start",
        {
            "session_id": "session/with spaces",
            "platform": "cli",
            "telemetry_schema_version": "hermes.observer.v1",
        },
    )
    telemetry.record_hook(
        "on_session_finalize",
        {
            "session_id": "session/with spaces",
            "reason": "exit",
            "telemetry_schema_version": "hermes.observer.v1",
        },
    )

    output = tmp_path / "hermes-atif-session-with-spaces.json"
    assert json.loads(output.read_text(encoding="utf-8")) == {
        "session_id": "session/with spaces",
        "agent_name": "Hermes Agent",
    }
    assert fake.atif_exporters[0].deregistered
    telemetry.reset_for_tests()
