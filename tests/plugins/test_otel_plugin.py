"""Tests for the bundled observability/otel plugin."""
from __future__ import annotations

import importlib
import sys
from pathlib import Path

import yaml


PLUGIN_DIR = Path(__file__).resolve().parents[2] / "plugins" / "observability" / "otel"


class _FakeExporter:
    def __init__(self) -> None:
        self.spans = []

    def export(self, spans):
        self.spans.extend(spans)


def _fresh_plugin(monkeypatch):
    sys.modules.pop("plugins.observability.otel", None)
    plugin = importlib.import_module("plugins.observability.otel")
    plugin._reset_for_tests()
    fake = _FakeExporter()
    monkeypatch.setattr(plugin, "_Exporter", lambda cfg: fake)
    return plugin, fake


def _attrs(span):
    out = {}
    for item in span["attributes"]:
        value = item["value"]
        out[item["key"]] = next(iter(value.values()))
    return out


def test_manifest_declares_phase_1_hooks():
    data = yaml.safe_load((PLUGIN_DIR / "plugin.yaml").read_text())
    assert data["name"] == "otel"
    assert set(data["hooks"]) == {
        "on_session_start",
        "on_session_end",
        "on_session_finalize",
        "on_session_reset",
        "post_api_request",
        "api_request_error",
        "post_tool_call",
    }


def test_api_request_exports_gen_ai_span_with_continuity_attrs(monkeypatch):
    plugin, fake = _fresh_plugin(monkeypatch)
    monkeypatch.setenv("HERMES_CRON_JOB_ID", "job-123")
    base = {
        "session_id": "s1",
        "parent_session_id": "s0",
        "turn_id": "turn-1",
        "api_request_id": "api-1",
        "provider": "openai",
        "model": "gpt-4o-mini",
        "api_call_count": 1,
    }

    plugin.on_session_start(**base, platform="cron")
    plugin.post_api_request(
        **base,
        resume_from="s0",
        response_model="gpt-4o-mini-2024-07-18",
        finish_reason="stop",
        api_duration=0.125,
        usage={"prompt_tokens": 11, "completion_tokens": 7, "total_tokens": 18},
    )

    assert len(fake.spans) == 1
    span = fake.spans[0]
    assert span["name"] == "gen_ai.chat openai"
    assert span["kind"] == plugin.KIND_CLIENT
    assert span["status"] == {"code": plugin.STATUS_OK, "message": ""}
    attrs = _attrs(span)
    assert attrs["gen_ai.operation.name"] == "chat"
    assert attrs["gen_ai.system"] == "openai"
    assert attrs["gen_ai.request.model"] == "gpt-4o-mini"
    assert attrs["gen_ai.response.model"] == "gpt-4o-mini-2024-07-18"
    assert attrs["gen_ai.usage.input_tokens"] == "11"
    assert attrs["gen_ai.usage.output_tokens"] == "7"
    assert attrs["gen_ai.usage.total_tokens"] == "18"
    assert attrs["gen_ai.session.id"] == "s1"
    assert attrs["gen_ai.agent.execution.id"]
    assert attrs["hermes.resume_from"] == "s0"
    assert attrs["hermes.cron.job.id"] == "job-123"
    assert attrs["hermes.api_request_id"] == "api-1"


def test_tool_call_exports_tool_span_with_call_id(monkeypatch):
    plugin, fake = _fresh_plugin(monkeypatch)

    plugin.post_tool_call(
        session_id="s1",
        turn_id="turn-1",
        tool_call_id="tool-1",
        tool_name="terminal",
        args={"command": "pwd"},
        result='{"output":"/tmp"}',
        status="ok",
        duration_ms=12.5,
    )

    span = fake.spans[0]
    assert span["name"] == "hermes.tool terminal"
    attrs = _attrs(span)
    assert attrs["gen_ai.tool.name"] == "terminal"
    assert attrs["gen_ai.tool.call.id"] == "tool-1"
    assert attrs["gen_ai.session.id"] == "s1"
    assert attrs["gen_ai.agent.execution.id"]
    assert attrs["hermes.tool.status"] == "ok"
    assert attrs["hermes.tool.duration_ms"] == 12.5
    assert attrs["hermes.tool.args.size"]
    assert attrs["hermes.tool.result.size"]


def test_session_finalize_exports_session_span(monkeypatch):
    plugin, fake = _fresh_plugin(monkeypatch)

    plugin.on_session_start(session_id="s1", platform="discord", model="m", provider="p")
    plugin.on_session_finalize(session_id="s1", reason="shutdown")

    span = fake.spans[0]
    assert span["name"] == "hermes.session"
    attrs = _attrs(span)
    assert attrs["gen_ai.session.id"] == "s1"
    assert attrs["gen_ai.agent.execution.id"]
    assert attrs["hermes.finalize.reason"] == "shutdown"


def test_continuity_attrs_propagate_to_subsequent_spans(monkeypatch):
    plugin, fake = _fresh_plugin(monkeypatch)
    plugin.on_session_start(session_id="s1", parent_session_id="s0", resume_from="r0", platform="cli")
    plugin.post_tool_call(
        session_id="s1",
        tool_call_id="tc-1",
        tool_name="terminal",
        args={"cmd": "ls"},
        result="ok",
        status="ok",
        duration_ms=4,
    )
    plugin.post_api_request(
        session_id="s1",
        turn_id="turn-1",
        api_request_id="api-1",
        provider="openai",
        model="gpt",
        resume_from="r0",
        usage={"prompt_tokens": 1, "completion_tokens": 1},
        finish_reason="stop",
    )

    tool_attrs = _attrs(fake.spans[0])
    api_attrs = _attrs(fake.spans[1])
    # gen_ai.session.id and gen_ai.agent.execution.id are seeded by
    # on_session_start and carried through to all subsequent spans.
    assert tool_attrs["gen_ai.session.id"] == "s1"
    assert tool_attrs["gen_ai.agent.execution.id"]
    # post_api_request explicitly receives resume_from.
    assert api_attrs["gen_ai.session.id"] == "s1"
    assert api_attrs["gen_ai.agent.execution.id"]
    assert api_attrs["hermes.resume_from"] == "r0"


def test_api_request_error_emits_error_status(monkeypatch):
    plugin, fake = _fresh_plugin(monkeypatch)
    plugin.api_request_error(
        session_id="s1",
        provider="openai",
        model="gpt",
        error={"type": "RateLimitError", "message": "rate limited"},
        reason="rate_limit",
        retryable=True,
        retry_count=1,
    )

    span = fake.spans[0]
    assert span["name"] == "gen_ai.chat openai"
    assert span["status"] == {"code": plugin.STATUS_ERROR, "message": "rate limited"}
    attrs = _attrs(span)
    assert attrs["error.type"] == "RateLimitError"
    assert attrs["hermes.retryable"] is True


def test_exporter_warns_when_export_fails_and_when_backoff_drops(monkeypatch, caplog):
    plugin = importlib.import_module("plugins.observability.otel")
    monkeypatch.setenv("HERMES_OTEL_ENABLED", "1")
    monkeypatch.setenv("HERMES_OTEL_ENDPOINT", "http://127.0.0.1:1/v1/traces")
    exporter = plugin._Exporter({"endpoint": "http://127.0.0.1:1/v1/traces", "timeout_seconds": 0.1, "retry_interval_seconds": 60})
    span = {"traceId": "1" * 32, "spanId": "2" * 16, "name": "test", "startTimeUnixNano": "1", "endTimeUnixNano": "2", "attributes": [], "status": {"code": 1}}

    with caplog.at_level("WARNING", logger=plugin.__name__):
        exporter.export([span])
        exporter.export([span])

    messages = [record.getMessage() for record in caplog.records]
    assert any("OTEL export failed; dropping 1 span(s)" in message for message in messages)
    assert any("OTEL export skipped during retry backoff; dropping 1 span(s)" in message for message in messages)