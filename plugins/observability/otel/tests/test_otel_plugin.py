"""Tests for the bundled observability/otel plugin."""

from __future__ import annotations

import importlib
import sys
from types import SimpleNamespace

import pytest

from plugins.observability.otel.config import load_config


HOOKS = {
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
    "subagent_start",
    "subagent_stop",
    "pre_approval_request",
    "post_approval_response",
}


def _fresh_plugin():
    sys.modules.pop("plugins.observability.otel", None)
    return importlib.import_module("plugins.observability.otel")


def _in_memory_runtime():
    pytest.importorskip("opentelemetry.sdk")
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

    exporter = InMemorySpanExporter()
    provider = TracerProvider(resource=Resource.create({"service.name": "test-hermes"}))
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    return SimpleNamespace(
        provider=provider,
        tracer=provider.get_tracer("test.hermes.otel"),
        exporter=exporter,
    )


def test_registers_all_observer_hooks():
    plugin = _fresh_plugin()
    registered = {}
    ctx = SimpleNamespace(
        register_hook=lambda name, callback: registered.setdefault(name, callback)
    )

    plugin.register(ctx)

    assert set(registered) == HOOKS
    assert all(callable(callback) for callback in registered.values())


def test_config_defaults_disabled():
    config = load_config({})

    assert config.enabled is False
    assert config.protocol == "http"
    assert config.service_name == "hermes-agent"
    assert config.endpoint == ""
    assert config.headers == {}


def test_config_parses_environment_values():
    config = load_config(
        {
            "HERMES_OTEL_ENABLED": "1",
            "HERMES_OTEL_ENDPOINT": "http://collector:4318/v1/traces",
            "HERMES_OTEL_PROTOCOL": "grpc",
            "HERMES_OTEL_SERVICE_NAME": "hermes-test",
            "HERMES_OTEL_HEADERS": '{"authorization":"Bearer test","tenant":7}',
        }
    )

    assert config.enabled is True
    assert config.endpoint == "http://collector:4318/v1/traces"
    assert config.protocol == "grpc"
    assert config.service_name == "hermes-test"
    assert config.headers == {"authorization": "Bearer test", "tenant": "7"}


def test_config_invalid_values_fail_closed():
    config = load_config(
        {
            "HERMES_OTEL_ENABLED": "maybe",
            "HERMES_OTEL_PROTOCOL": "udp",
            "HERMES_OTEL_HEADERS": "not-json",
        }
    )

    assert config.enabled is False
    assert config.protocol == "http"
    assert config.headers == {}


def test_hooks_create_correlated_privacy_safe_spans(monkeypatch):
    runtime = _in_memory_runtime()
    plugin = _fresh_plugin()
    monkeypatch.setenv("HERMES_OTEL_ENABLED", "1")
    monkeypatch.setattr(plugin, "create_runtime", lambda config: runtime)

    base = {
        "session_id": "session-1",
        "task_id": "task-1",
        "turn_id": "turn-1",
        "telemetry_schema_version": "hermes.observer.v1",
    }
    plugin.on_session_start(**base, model="test-model", platform="cli")
    plugin.on_pre_llm_call(
        **base,
        model="test-model",
        platform="cli",
        user_message="private user text",
        conversation_history=[{"content": "private history"}],
        is_first_turn=True,
    )
    plugin.on_pre_api_request(
        **base,
        api_request_id="api-1",
        model="test-model",
        provider="openai",
        api_mode="chat_completions",
        message_count=2,
        tool_count=1,
        approx_input_tokens=17,
        request_char_count=420,
        request={"body": {"messages": [{"content": "secret"}]}},
    )
    plugin.on_post_api_request(
        **base,
        api_request_id="api-1",
        response_model="test-model-v2",
        finish_reason="stop",
        api_duration=0.25,
        usage={"input_tokens": 20, "output_tokens": 5},
        assistant_content_chars=12,
        assistant_tool_call_count=0,
        response={"body": {"content": "private response"}},
    )
    plugin.on_pre_tool_call(
        **base,
        tool_call_id="tool-1",
        tool_name="read_file",
        args={"path": "/secret/path"},
    )
    plugin.on_post_tool_call(
        **base,
        tool_call_id="tool-1",
        tool_name="read_file",
        args={"path": "/secret/path"},
        result="private file contents",
        duration_ms=15,
        status="ok",
    )
    plugin.on_post_llm_call(
        **base,
        assistant_response="private final answer",
        conversation_history=[{"content": "private history"}],
    )
    plugin.on_session_end(**base, completed=True, interrupted=False)

    spans = {span.name: span for span in runtime.exporter.get_finished_spans()}
    assert set(spans) >= {
        "hermes.session",
        "hermes.turn",
        "hermes.llm_request",
        "hermes.tool.read_file",
    }
    assert spans["hermes.turn"].parent.span_id == spans["hermes.session"].context.span_id
    assert spans["hermes.llm_request"].parent.span_id == spans["hermes.turn"].context.span_id
    assert spans["hermes.tool.read_file"].parent.span_id == spans["hermes.turn"].context.span_id
    assert spans["hermes.llm_request"].attributes["gen_ai.request.model"] == "test-model"
    assert spans["hermes.llm_request"].attributes["gen_ai.provider.name"] == "openai"
    assert spans["hermes.llm_request"].attributes["gen_ai.usage.input_tokens"] == 20
    assert spans["hermes.llm_request"].attributes["hermes.api.duration_ms"] == 250
    assert spans["hermes.tool.read_file"].attributes["hermes.tool.status"] == "ok"

    serialized_attributes = repr(
        [dict(span.attributes) for span in runtime.exporter.get_finished_spans()]
    )
    for secret in (
        "private user text",
        "private history",
        "private response",
        "private file contents",
        "/secret/path",
    ):
        assert secret not in serialized_attributes


def test_subagent_and_approval_spans_use_safe_summaries(monkeypatch):
    runtime = _in_memory_runtime()
    plugin = _fresh_plugin()
    monkeypatch.setenv("HERMES_OTEL_ENABLED", "1")
    monkeypatch.setattr(plugin, "create_runtime", lambda config: runtime)

    plugin.on_session_start(session_id="parent", model="m", platform="cli")
    plugin.on_pre_llm_call(session_id="parent", turn_id="turn", user_message="x")
    plugin.on_subagent_start(
        parent_session_id="parent",
        parent_turn_id="turn",
        child_session_id="child",
        child_subagent_id="sub-1",
        child_role="researcher",
        child_goal="sensitive delegated goal",
    )
    plugin.on_subagent_stop(
        parent_session_id="parent",
        parent_turn_id="turn",
        child_session_id="child",
        child_subagent_id="sub-1",
        child_role="researcher",
        child_status="completed",
        child_summary="sensitive child result",
        duration_ms=50,
    )
    plugin.on_pre_approval_request(
        session_id="parent",
        session_key="parent",
        command="curl -H 'Authorization: secret' private.example",
        description="sensitive command description",
        pattern_keys=["network"],
        surface="cli",
    )
    plugin.on_post_approval_response(
        session_id="parent",
        session_key="parent",
        choice="deny",
    )
    plugin.on_post_llm_call(
        session_id="parent",
        turn_id="turn",
        assistant_response="done",
    )
    plugin.on_session_end(session_id="parent", completed=True, interrupted=False)

    spans = {span.name: span for span in runtime.exporter.get_finished_spans()}
    subagent = spans["hermes.subagent"]
    approval = spans["hermes.approval"]
    assert subagent.attributes["hermes.subagent.role"] == "researcher"
    assert subagent.attributes["hermes.subagent.goal_length"] == 24
    assert len(subagent.attributes["hermes.subagent.goal_id"]) == 16
    assert approval.attributes["hermes.approval.decision"] == "deny"
    assert approval.attributes["hermes.approval.command_length"] > 0
    assert len(approval.attributes["hermes.approval.command_id"]) == 16
    assert subagent.parent.span_id == spans["hermes.turn"].context.span_id
    serialized = repr([dict(subagent.attributes), dict(approval.attributes)])
    assert "sensitive delegated goal" not in serialized
    assert "sensitive child result" not in serialized
    assert "Authorization: secret" not in serialized


def test_finalize_shuts_down_provider(monkeypatch):
    plugin = _fresh_plugin()
    calls = []
    runtime = SimpleNamespace(
        tracer=SimpleNamespace(),
        provider=SimpleNamespace(shutdown=lambda: calls.append("shutdown")),
    )
    builder = SimpleNamespace(finalize_session=lambda kwargs: calls.append(kwargs["session_id"]))
    monkeypatch.setattr(plugin, "_RUNTIME", runtime)
    monkeypatch.setattr(plugin, "_BUILDER", builder)

    plugin.on_session_finalize(session_id="session-1")

    assert calls == ["session-1", "shutdown"]
    assert plugin._RUNTIME is None
    assert plugin._BUILDER is None


def test_hook_errors_fail_open(monkeypatch):
    plugin = _fresh_plugin()
    monkeypatch.setattr(
        plugin,
        "_get_builder",
        lambda: SimpleNamespace(start_tool=lambda kwargs: (_ for _ in ()).throw(RuntimeError("boom"))),
    )

    plugin.on_pre_tool_call(tool_call_id="tool-1", tool_name="shell")


def test_initialization_errors_fail_open(monkeypatch):
    plugin = _fresh_plugin()
    monkeypatch.setenv("HERMES_OTEL_ENABLED", "1")
    monkeypatch.setattr(
        plugin,
        "create_runtime",
        lambda config: (_ for _ in ()).throw(RuntimeError("missing SDK")),
    )

    plugin.on_session_start(session_id="session-1")
    plugin.on_session_start(session_id="session-1")

    assert plugin._RUNTIME is plugin._INIT_FAILED
