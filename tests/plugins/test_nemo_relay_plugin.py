"""Tests for the bundled observability/nemo_relay plugin."""

from __future__ import annotations

import asyncio
import builtins
import gc
import importlib
import json
import sys
import warnings
from pathlib import Path
from types import SimpleNamespace

import pytest
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
        self.llm = SimpleNamespace(
            call=self._llm_call,
            call_end=self._llm_call_end,
            execute=self._llm_execute,
        )
        self.tools = SimpleNamespace(
            call=self._tool_call,
            call_end=self._tool_call_end,
            execute=self._tool_execute,
        )
        self.plugin = SimpleNamespace(initialize=self._plugin_initialize, clear=self._plugin_clear)
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

    def _llm_execute(self, name, request, func, **kwargs):
        self.events.append(("llm.execute.start", name, request.content, kwargs))
        result = func(_FakeLLMRequest(request.headers, {"intercepted": True, **request.content}))
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
        result = func({"intercepted": True, **args})
        self.events.append(("tool.execute.end", name, result, kwargs))
        return result

    def _make_atof_exporter(self, config):
        return _FakeAtofExporter(self.events, config)

    def _make_atif_exporter(self, session_id, agent_name, agent_version, **kwargs):
        return _FakeAtifExporter(self.events, session_id, agent_name, agent_version, kwargs)

    async def _plugin_initialize(self, config):
        self.events.append(("plugin.initialize", config))
        return {"diagnostics": []}

    async def _plugin_clear(self):
        self.events.append(("plugin.clear",))


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
        return json.dumps({"session_id": self.session_id, "agent_name": self.agent_name})


def _fresh_plugin(monkeypatch, fake):
    monkeypatch.setitem(sys.modules, "nemo_relay", fake)
    sys.modules.pop("plugins.observability.nemo_relay", None)
    plugin = importlib.import_module("plugins.observability.nemo_relay")
    plugin.reset_for_tests()
    return plugin


def _wrapped_downstream_error(original):
    class _DownstreamExecutionError(Exception):
        def __init__(self, original):
            super().__init__(str(original))
            self.original = original

    return _DownstreamExecutionError(original)


def _enable_adaptive_plugin(tmp_path, monkeypatch) -> None:
    plugins_toml = tmp_path / "plugins.toml"
    plugins_toml.write_text(
        """
version = 1

[[components]]
kind = "adaptive"
enabled = true

[components.config.tool_parallelism]
mode = "observe_only"
""",
        encoding="utf-8",
    )
    monkeypatch.setenv("HERMES_NEMO_RELAY_PLUGINS_TOML", str(plugins_toml))


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
    tool_end = next(event for event in fake.events if event[0] == "tool.call_end")
    assert tool_end[3]["data"]["status"] == "ok"
    assert "error_kind" not in tool_end[3]["data"]
    assert "hermes.tool.error_kind" not in tool_end[3]["data"]
    assert "scope.pop" in event_names
    assert (tmp_path / "atif" / "hermes-atif-s1.json").exists()


def test_nemo_relay_plugin_exports_command_metadata(monkeypatch):
    fake = _FakeNemoRelay()
    plugin = _fresh_plugin(monkeypatch, fake)

    plugin.on_session_start(session_id="s1")
    plugin.on_pre_tool_call(session_id="s1", tool_name="terminal", tool_call_id="tool-1", args={"command": "SECRET=*** pytest"})
    plugin.on_post_tool_call(
        session_id="s1",
        tool_name="terminal",
        tool_call_id="tool-1",
        result={"ok": True},
        status="ok",
        command_class="test",
        timeout_seconds=120,
        background=True,
        notify_on_complete=True,
        pty=False,
        wait_kind="background_wait",
    )

    tool_end = next(event for event in fake.events if event[0] == "tool.call_end")
    data = tool_end[3]["data"]
    assert data["hermes.tool.command_class"] == "test"
    assert data["hermes.tool.timeout_seconds"] == 120
    assert data["hermes.tool.background"] is True
    assert data["hermes.tool.notify_on_complete"] is True
    assert data["hermes.tool.pty"] is False
    assert data["hermes.tool.wait_kind"] == "background_wait"
    assert "SECRET" not in repr({k: v for k, v in data.items() if k.startswith("hermes.tool.")})


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
    assert "hermes-session-s2" in scope_push_names


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
    assert "hermes-session-s2" in scope_push_names


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


def test_nemo_relay_adaptive_llm_execution_middleware_preserves_raw_response(tmp_path, monkeypatch):
    fake = _FakeNemoRelay()
    plugin = _fresh_plugin(monkeypatch, fake)
    plugins_toml = tmp_path / "plugins.toml"
    plugins_toml.write_text(
        """
version = 1

[[components]]
kind = "adaptive"
enabled = true

[components.config.tool_parallelism]
mode = "observe_only"
""",
        encoding="utf-8",
    )
    monkeypatch.setenv("HERMES_NEMO_RELAY_PLUGINS_TOML", str(plugins_toml))

    seen_request = {}
    raw_choice = SimpleNamespace(
        message=SimpleNamespace(
            role="assistant",
            content=None,
            tool_calls=[
                SimpleNamespace(
                    id="tool-1",
                    type="function",
                    function=SimpleNamespace(name="terminal", arguments='{"command":"pwd"}'),
                )
            ],
            reasoning_content="need a tool",
        ),
        finish_reason="tool_calls",
    )

    def next_call(request):
        seen_request.update(request)
        return SimpleNamespace(
            id="resp-1",
            model="demo-model",
            choices=[raw_choice],
            usage=SimpleNamespace(prompt_tokens=3, completion_tokens=5, total_tokens=8),
        )

    response = plugin.on_llm_execution_middleware(
        session_id="s1",
        task_id="t1",
        turn_id="turn-1",
        api_request_id="api-1",
        provider="anthropic",
        model="demo-model",
        api_call_count=1,
        request={"messages": [{"role": "user", "content": "hi"}]},
        next_call=next_call,
    )

    assert response.model == "demo-model"
    assert response.choices == [raw_choice]
    assert seen_request["intercepted"] is True
    execute_start = next(event for event in fake.events if event[0] == "llm.execute.start")
    assert execute_start[3]["data"]["mode"] == "observe_only"
    execute_end = next(event for event in fake.events if event[0] == "llm.execute.end")
    assert execute_end[2] == {
        "model": "demo-model",
        "assistant_message": {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "tool-1",
                    "type": "function",
                    "function": {"name": "terminal", "arguments": '{"command":"pwd"}'},
                }
            ],
            "reasoning_content": "need a tool",
        },
        "finish_reason": "tool_calls",
        "usage": {"prompt_tokens": 3, "completion_tokens": 5, "total_tokens": 8},
    }


def test_nemo_relay_adaptive_llm_execution_preserves_downstream_error(tmp_path, monkeypatch):
    fake = _FakeNemoRelay()

    def native_like_execute(name, request, func, **kwargs):
        fake.events.append(("llm.execute.start", name, request.content, kwargs))
        try:
            return func(_FakeLLMRequest(request.headers, {"intercepted": True, **request.content}))
        except Exception as exc:
            raise RuntimeError(f"internal error: {type(exc).__name__}: {exc}") from None

    fake.llm.execute = native_like_execute
    plugin = _fresh_plugin(monkeypatch, fake)
    _enable_adaptive_plugin(tmp_path, monkeypatch)

    class ProviderAuthError(Exception):
        status_code = 403

    provider_error = ProviderAuthError("provider auth failed")

    def next_call(request):
        raise _wrapped_downstream_error(provider_error)

    with pytest.raises(ProviderAuthError) as caught:
        plugin.on_llm_execution_middleware(
            session_id="s1",
            provider="anthropic",
            model="demo-model",
            request={"messages": [{"role": "user", "content": "hi"}]},
            next_call=next_call,
        )

    assert caught.value is provider_error
    assert caught.value.status_code == 403


def test_nemo_relay_adaptive_llm_execution_preserves_downstream_error_with_relay_suffix(
    tmp_path, monkeypatch
):
    # Guards the startswith (vs exact ==) match in _is_relay_wrapped_callback_error:
    # Relay re-wraps the callback failure with its canonical prefix but APPENDS a
    # trailing suffix. Exact equality would miss this and surface Relay's wrapper;
    # prefix matching must still recover the original downstream error.
    fake = _FakeNemoRelay()

    def native_like_execute(name, request, func, **kwargs):
        try:
            return func(_FakeLLMRequest(request.headers, {"intercepted": True, **request.content}))
        except Exception as exc:
            raise RuntimeError(f"internal error: {type(exc).__name__}: {exc} (retried 3x)") from None

    fake.llm.execute = native_like_execute
    plugin = _fresh_plugin(monkeypatch, fake)
    _enable_adaptive_plugin(tmp_path, monkeypatch)

    class ProviderAuthError(Exception):
        status_code = 403

    provider_error = ProviderAuthError("provider auth failed")

    def next_call(request):
        raise _wrapped_downstream_error(provider_error)

    with pytest.raises(ProviderAuthError) as caught:
        plugin.on_llm_execution_middleware(
            session_id="s1",
            provider="anthropic",
            model="demo-model",
            request={"messages": [{"role": "user", "content": "hi"}]},
            next_call=next_call,
        )

    assert caught.value is provider_error
    assert caught.value.status_code == 403


def test_nemo_relay_adaptive_llm_execution_keeps_unrelated_internal_error(tmp_path, monkeypatch):
    fake = _FakeNemoRelay()

    relay_error = RuntimeError("internal error: relay setup failed")

    def internal_error_execute(name, request, func, **kwargs):
        raise relay_error

    fake.llm.execute = internal_error_execute
    plugin = _fresh_plugin(monkeypatch, fake)
    _enable_adaptive_plugin(tmp_path, monkeypatch)

    with pytest.raises(RuntimeError) as caught:
        plugin.on_llm_execution_middleware(
            session_id="s1",
            provider="anthropic",
            model="demo-model",
            request={"messages": [{"role": "user", "content": "hi"}]},
            next_call=lambda request: {"raw": request},
        )

    assert caught.value is relay_error


def test_nemo_relay_adaptive_llm_execution_keeps_wrapped_relay_error_after_downstream_failure(
    tmp_path, monkeypatch
):
    fake = _FakeNemoRelay()
    relay_error = RuntimeError("internal error: RuntimeError: relay policy blocked after downstream")

    def translated_execute(name, request, func, **kwargs):
        try:
            return func(_FakeLLMRequest(request.headers, {"intercepted": True, **request.content}))
        except Exception:
            raise relay_error

    fake.llm.execute = translated_execute
    plugin = _fresh_plugin(monkeypatch, fake)
    _enable_adaptive_plugin(tmp_path, monkeypatch)

    def next_call(request):
        raise _wrapped_downstream_error(RuntimeError("provider failed"))

    with pytest.raises(RuntimeError) as caught:
        plugin.on_llm_execution_middleware(
            session_id="s1",
            provider="anthropic",
            model="demo-model",
            request={"messages": [{"role": "user", "content": "hi"}]},
            next_call=next_call,
        )

    assert caught.value is relay_error


def test_nemo_relay_adaptive_llm_execution_keeps_relay_translated_error(tmp_path, monkeypatch):
    fake = _FakeNemoRelay()

    class RelayPolicyError(Exception):
        pass

    relay_error = RelayPolicyError("relay policy blocked")

    def translated_execute(name, request, func, **kwargs):
        try:
            return func(_FakeLLMRequest(request.headers, {"intercepted": True, **request.content}))
        except Exception:
            raise relay_error

    fake.llm.execute = translated_execute
    plugin = _fresh_plugin(monkeypatch, fake)
    _enable_adaptive_plugin(tmp_path, monkeypatch)

    provider_error = RuntimeError("provider failed")

    def next_call(request):
        raise _wrapped_downstream_error(provider_error)

    with pytest.raises(RelayPolicyError) as caught:
        plugin.on_llm_execution_middleware(
            session_id="s1",
            provider="anthropic",
            model="demo-model",
            request={"messages": [{"role": "user", "content": "hi"}]},
            next_call=next_call,
        )

    assert caught.value is relay_error


def test_nemo_relay_downstream_unwrap_matches_real_middleware_wrapper_shape(monkeypatch):
    # Regression guard against core/plugin drift. The synthetic tests above model
    # the downstream-error wrapper with a local class, so they keep passing even
    # if core middleware renames its private ``_DownstreamExecutionError`` or drops
    # ``.original`` -- the exact shape the plugin matches by name at
    # ``_original_downstream_error``. Capture the wrapper the REAL
    # ``hermes_cli.middleware._run_execution_chain`` hands to a middleware
    # callback's ``next_call`` and assert the plugin's detector unwraps it to the
    # original exception. If core middleware changes the wrapper shape, this fails
    # here instead of silently defeating the unwrap in production.
    from hermes_cli import middleware

    from plugins.observability.nemo_relay import _original_downstream_error

    class ProviderError(Exception):
        status_code = 403

    provider_error = ProviderError("provider auth failed")
    captured: dict[str, Exception] = {}

    def terminal_call(payload):
        raise provider_error

    def capturing_callback(**kwargs):
        next_call = kwargs["next_call"]
        try:
            return next_call(kwargs.get("request"))
        except Exception as exc:
            captured["wrapper"] = exc
            # Surface the original so the chain unwinds without re-wrapping noise.
            raise _original_downstream_error(exc) from None

    with pytest.raises(ProviderError) as caught:
        middleware._run_execution_chain(
            "llm",
            [capturing_callback],
            terminal_call,
            request={"messages": []},
        )

    wrapper = captured["wrapper"]
    # The wrapper the plugin sees must match what _original_downstream_error keys on.
    assert wrapper.__class__.__name__ == "_DownstreamExecutionError"
    assert isinstance(getattr(wrapper, "original", None), BaseException)
    assert _original_downstream_error(wrapper) is provider_error
    assert caught.value is provider_error
    assert caught.value.status_code == 403


def _adaptive_llm_execute_mode(tmp_path, monkeypatch, plugins_toml_text: str) -> str:
    fake = _FakeNemoRelay()
    plugin = _fresh_plugin(monkeypatch, fake)
    plugins_toml = tmp_path / "plugins.toml"
    plugins_toml.write_text(plugins_toml_text, encoding="utf-8")
    monkeypatch.setenv("HERMES_NEMO_RELAY_PLUGINS_TOML", str(plugins_toml))

    plugin.on_llm_execution_middleware(
        session_id="s1",
        provider="anthropic",
        model="demo-model",
        request={"messages": [{"role": "user", "content": "hi"}]},
        next_call=lambda request: {"raw": request},
    )

    execute_start = next(event for event in fake.events if event[0] == "llm.execute.start")
    return execute_start[3]["data"]["mode"]


def test_nemo_relay_adaptive_llm_execution_middleware_defaults_to_observe_only_when_mode_is_unset(
    tmp_path, monkeypatch
):
    mode = _adaptive_llm_execute_mode(
        tmp_path,
        monkeypatch,
        """
version = 1

[[components]]
kind = "adaptive"
enabled = true

[components.config]
version = 1
""",
    )
    assert mode == "observe_only"


def test_nemo_relay_adaptive_llm_execution_middleware_accepts_legacy_top_level_mode(tmp_path, monkeypatch):
    mode = _adaptive_llm_execute_mode(
        tmp_path,
        monkeypatch,
        """
version = 1

[[components]]
kind = "adaptive"
enabled = true

[components.config]
mode = "route"
""",
    )
    assert mode == "route"


def test_nemo_relay_adaptive_llm_execution_middleware_prefers_tool_parallelism_mode(tmp_path, monkeypatch):
    mode = _adaptive_llm_execute_mode(
        tmp_path,
        monkeypatch,
        """
version = 1

[[components]]
kind = "adaptive"
enabled = true

[components.config]
mode = "route"

[components.config.tool_parallelism]
mode = "schedule"
""",
    )
    assert mode == "schedule"


def test_nemo_relay_llm_execution_middleware_calls_through_without_adaptive(monkeypatch):
    fake = _FakeNemoRelay()
    plugin = _fresh_plugin(monkeypatch, fake)

    response = plugin.on_llm_execution_middleware(
        session_id="s1",
        provider="anthropic",
        model="demo-model",
        request={"messages": []},
        next_call=lambda request: {"raw": request},
    )

    assert response == {"raw": {"messages": []}}
    assert not any(event[0] == "llm.execute.start" for event in fake.events)


def test_nemo_relay_adaptive_tool_execution_middleware_preserves_raw_response(tmp_path, monkeypatch):
    fake = _FakeNemoRelay()
    plugin = _fresh_plugin(monkeypatch, fake)
    plugins_toml = tmp_path / "plugins.toml"
    plugins_toml.write_text(
        """
version = 1

[[components]]
kind = "adaptive"
enabled = true

[components.config.tool_parallelism]
mode = "observe_only"
""",
        encoding="utf-8",
    )
    monkeypatch.setenv("HERMES_NEMO_RELAY_PLUGINS_TOML", str(plugins_toml))

    seen_args = {}

    def next_call(args):
        seen_args.update(args)
        return {"raw": True, "args": args}

    response = plugin.on_tool_execution_middleware(
        session_id="s1",
        task_id="t1",
        turn_id="turn-1",
        api_request_id="api-1",
        tool_name="terminal",
        tool_call_id="tool-1",
        args={"command": "pwd"},
        next_call=next_call,
    )

    assert response == {"raw": True, "args": {"command": "pwd", "intercepted": True}}
    assert seen_args["intercepted"] is True
    execute_start = next(event for event in fake.events if event[0] == "tool.execute.start")
    assert execute_start[3]["data"]["mode"] == "observe_only"
    assert execute_start[3]["data"]["tool_call_id"] == "tool-1"


def test_nemo_relay_adaptive_tool_execution_preserves_downstream_error(tmp_path, monkeypatch):
    fake = _FakeNemoRelay()

    def native_like_execute(name, args, func, **kwargs):
        fake.events.append(("tool.execute.start", name, args, kwargs))
        try:
            return func({"intercepted": True, **args})
        except Exception as exc:
            raise RuntimeError(f"internal error: {type(exc).__name__}: {exc}") from None

    fake.tools.execute = native_like_execute
    plugin = _fresh_plugin(monkeypatch, fake)
    _enable_adaptive_plugin(tmp_path, monkeypatch)

    class ToolAuthError(Exception):
        status_code = 403

    tool_error = ToolAuthError("tool auth failed")

    def next_call(args):
        raise _wrapped_downstream_error(tool_error)

    with pytest.raises(ToolAuthError) as caught:
        plugin.on_tool_execution_middleware(
            session_id="s1",
            tool_name="terminal",
            args={"command": "pwd"},
            next_call=next_call,
        )

    assert caught.value is tool_error
    assert caught.value.status_code == 403


def test_nemo_relay_adaptive_tool_execution_keeps_unrelated_internal_error(tmp_path, monkeypatch):
    fake = _FakeNemoRelay()

    relay_error = RuntimeError("internal error: relay setup failed")

    def internal_error_execute(name, args, func, **kwargs):
        raise relay_error

    fake.tools.execute = internal_error_execute
    plugin = _fresh_plugin(monkeypatch, fake)
    _enable_adaptive_plugin(tmp_path, monkeypatch)

    with pytest.raises(RuntimeError) as caught:
        plugin.on_tool_execution_middleware(
            session_id="s1",
            tool_name="terminal",
            args={"command": "pwd"},
            next_call=lambda args: {"raw": args},
        )

    assert caught.value is relay_error


def test_nemo_relay_adaptive_tool_execution_keeps_wrapped_relay_error_after_downstream_failure(
    tmp_path, monkeypatch
):
    fake = _FakeNemoRelay()
    relay_error = RuntimeError("internal error: RuntimeError: relay policy blocked after downstream")

    def translated_execute(name, args, func, **kwargs):
        try:
            return func({"intercepted": True, **args})
        except Exception:
            raise relay_error

    fake.tools.execute = translated_execute
    plugin = _fresh_plugin(monkeypatch, fake)
    _enable_adaptive_plugin(tmp_path, monkeypatch)

    def next_call(args):
        raise _wrapped_downstream_error(RuntimeError("tool failed"))

    with pytest.raises(RuntimeError) as caught:
        plugin.on_tool_execution_middleware(
            session_id="s1",
            tool_name="terminal",
            args={"command": "pwd"},
            next_call=next_call,
        )

    assert caught.value is relay_error


def test_nemo_relay_adaptive_tool_execution_keeps_relay_translated_error(tmp_path, monkeypatch):
    fake = _FakeNemoRelay()

    class RelayPolicyError(Exception):
        pass

    relay_error = RelayPolicyError("relay policy blocked")

    def translated_execute(name, args, func, **kwargs):
        try:
            return func({"intercepted": True, **args})
        except Exception:
            raise relay_error

    fake.tools.execute = translated_execute
    plugin = _fresh_plugin(monkeypatch, fake)
    _enable_adaptive_plugin(tmp_path, monkeypatch)

    tool_error = RuntimeError("tool failed")

    def next_call(args):
        raise _wrapped_downstream_error(tool_error)

    with pytest.raises(RelayPolicyError) as caught:
        plugin.on_tool_execution_middleware(
            session_id="s1",
            tool_name="terminal",
            args={"command": "pwd"},
            next_call=next_call,
        )

    assert caught.value is relay_error


def test_nemo_relay_tool_execution_middleware_calls_through_without_adaptive(monkeypatch):
    fake = _FakeNemoRelay()
    plugin = _fresh_plugin(monkeypatch, fake)

    response = plugin.on_tool_execution_middleware(
        session_id="s1",
        tool_name="terminal",
        args={"command": "pwd"},
        next_call=lambda args: {"raw": args},
    )

    assert response == {"raw": {"command": "pwd"}}
    assert not any(event[0] == "tool.execute.start" for event in fake.events)


def test_nemo_relay_adaptive_execution_skips_duplicate_observer_spans(tmp_path, monkeypatch):
    fake = _FakeNemoRelay()
    plugin = _fresh_plugin(monkeypatch, fake)
    plugins_toml = tmp_path / "plugins.toml"
    plugins_toml.write_text(
        """
version = 1

[[components]]
kind = "adaptive"
enabled = true

[components.config.tool_parallelism]
mode = "observe_only"
""",
        encoding="utf-8",
    )
    monkeypatch.setenv("HERMES_NEMO_RELAY_PLUGINS_TOML", str(plugins_toml))

    base = {
        "session_id": "s1",
        "task_id": "t1",
        "turn_id": "turn-1",
        "api_request_id": "api-1",
    }
    plugin.on_pre_api_request(
        **base,
        provider="anthropic",
        model="demo-model",
        request={"body": {"messages": [{"role": "user", "content": "hi"}]}},
    )
    plugin.on_post_api_request(**base, response={"ok": True})
    plugin.on_pre_tool_call(**base, tool_name="terminal", tool_call_id="tool-1", args={"command": "pwd"})
    plugin.on_post_tool_call(**base, tool_name="terminal", tool_call_id="tool-1", result={"ok": True})

    plugin.on_llm_execution_middleware(
        **base,
        provider="anthropic",
        model="demo-model",
        request={"messages": [{"role": "user", "content": "hi"}]},
        next_call=lambda request: {"raw": request},
    )
    plugin.on_tool_execution_middleware(
        **base,
        tool_name="terminal",
        tool_call_id="tool-1",
        args={"command": "pwd"},
        next_call=lambda args: {"raw": args},
    )

    event_names = [event[0] for event in fake.events]
    assert "llm.call" not in event_names
    assert "llm.call_end" not in event_names
    assert "tool.call" not in event_names
    assert "tool.call_end" not in event_names
    assert "llm.execute.start" in event_names
    assert "tool.execute.start" in event_names


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
