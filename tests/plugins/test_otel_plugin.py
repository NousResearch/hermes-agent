"""Tests for the bundled observability/otel plugin.

These exercise the span-mapping core, the dashboard log-record emitter, and the
local-model cost estimator with in-memory OTel exporters, so they run with only
``opentelemetry-sdk`` installed (no Hermes runtime, no network). They assert
that Hermes turn / LLM-call / tool-call events become the expected ``gen_ai.*``
GenAI-convention **spans** *and* the dashboard-shaped OTLP **log records**, that
local-model cost is estimated from model size, and that the plugin's
``register(ctx)`` + hook surface matches the real Hermes plugin contract.
"""
from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest
import yaml
from opentelemetry._logs import get_logger, set_logger_provider
from opentelemetry.sdk._logs import LoggerProvider
from opentelemetry.sdk._logs.export import (
    InMemoryLogRecordExporter,
    SimpleLogRecordProcessor,
)
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)

from plugins.observability.otel.cost import (
    estimate_cost_usd,
    extract_param_count_billions,
)
from plugins.observability.otel.emitter import (
    OP_CHAT,
    OP_EXECUTE_TOOL,
    OP_INVOKE_AGENT,
    OtelGenAIEmitter,
)
from plugins.observability.otel.log_emitter import (
    EVENT_API_RESPONSE,
    EVENT_SESSION_START,
    EVENT_TOOL_RESULT,
    OtelLogEmitter,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
PLUGIN_DIR = REPO_ROOT / "plugins" / "observability" / "otel"


# ---------------------------------------------------------------------------
# Manifest + layout
# ---------------------------------------------------------------------------


class TestManifest:
    def test_plugin_directory_exists(self):
        assert PLUGIN_DIR.is_dir()
        assert (PLUGIN_DIR / "plugin.yaml").exists()
        assert (PLUGIN_DIR / "__init__.py").exists()

    def test_manifest_fields(self):
        data = yaml.safe_load((PLUGIN_DIR / "plugin.yaml").read_text(encoding="utf-8"))
        assert data["name"] == "otel"
        assert data["version"]
        # Hooks the plugin implements — must match register() exactly.
        assert set(data["hooks"]) == {
            "on_session_start",
            "on_session_end",
            "on_session_finalize",
            "on_session_reset",
            "pre_llm_call",
            "transform_llm_output",
            "post_api_request",
            "post_tool_call",
        }


# ---------------------------------------------------------------------------
# Emitter: pure span-mapping core, no Hermes import.
# ---------------------------------------------------------------------------


@pytest.fixture()
def exporter_and_tracer():
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    tracer = provider.get_tracer("test")
    return exporter, tracer


@pytest.fixture()
def log_exporter_and_logger():
    """In-memory OTLP *log* pipeline — keeps the log-record tests hermetic.

    ``SimpleLogRecordProcessor`` flushes synchronously, so emitted records are
    immediately readable via ``get_finished_logs()`` with no network and no
    batch-timer wait.
    """
    exporter = InMemoryLogRecordExporter()
    provider = LoggerProvider(resource=Resource.create({"service.name": "test"}))
    provider.add_log_record_processor(SimpleLogRecordProcessor(exporter))
    otel_logger = get_logger("test", logger_provider=provider)
    return exporter, otel_logger


def _by_name(spans):
    return {s.name: s for s in spans}


def _log_attrs(records):
    """Map ``event.name`` → attributes dict for the finished log records."""
    out: dict[str, dict] = {}
    for r in records:
        attrs = dict(r.log_record.attributes or {})
        name = attrs.get("event.name")
        if name is not None:
            out[name] = attrs
    return out


class TestEmitter:
    def test_turn_span_carries_session_and_agent(self, exporter_and_tracer):
        exporter, tracer = exporter_and_tracer
        emitter = OtelGenAIEmitter(tracer, capture_content=False)

        with emitter.turn_span(session_id="sess-123", model="hermes-llama-70b"):
            pass

        spans = _by_name(exporter.get_finished_spans())
        assert OP_INVOKE_AGENT in spans
        attrs = spans[OP_INVOKE_AGENT].attributes
        assert attrs["gen_ai.operation.name"] == OP_INVOKE_AGENT
        assert attrs["gen_ai.system"] == "hermes"
        assert attrs["gen_ai.agent.name"] == "hermes"
        assert attrs["gen_ai.conversation.id"] == "sess-123"
        assert attrs["session.id"] == "sess-123"
        assert attrs["gen_ai.request.model"] == "hermes-llama-70b"

    def test_chat_span_carries_tokens_cost_and_finish_reason(self, exporter_and_tracer):
        exporter, tracer = exporter_and_tracer
        emitter = OtelGenAIEmitter(tracer)

        with emitter.turn_span(session_id="sess-1"):
            emitter.record_llm_call(
                request_model="hermes-llama-70b",
                input_tokens=100,
                output_tokens=42,
                cost_usd=0.0031,
                finish_reasons=["stop"],
                ttft_ms=180.0,
            )

        spans = _by_name(exporter.get_finished_spans())
        assert OP_CHAT in spans
        attrs = spans[OP_CHAT].attributes
        assert attrs["gen_ai.operation.name"] == OP_CHAT
        assert attrs["gen_ai.usage.input_tokens"] == 100
        assert attrs["gen_ai.usage.output_tokens"] == 42
        assert attrs["cost_usd"] == pytest.approx(0.0031)
        assert list(attrs["gen_ai.response.finish_reasons"]) == ["stop"]
        assert attrs["copilot_chat.time_to_first_token"] == pytest.approx(180.0)

    def test_chat_span_nested_under_turn(self, exporter_and_tracer):
        exporter, tracer = exporter_and_tracer
        emitter = OtelGenAIEmitter(tracer)

        with emitter.turn_span(session_id="sess-1", model="m"):
            emitter.record_llm_call(request_model="m", input_tokens=1, output_tokens=1)

        spans = _by_name(exporter.get_finished_spans())
        chat = spans[OP_CHAT]
        invoke = spans[OP_INVOKE_AGENT]
        assert chat.parent is not None
        assert chat.parent.span_id == invoke.context.span_id
        assert chat.context.trace_id == invoke.context.trace_id

    def test_execute_tool_span_carries_tool_name_and_type(self, exporter_and_tracer):
        exporter, tracer = exporter_and_tracer
        emitter = OtelGenAIEmitter(tracer)

        emitter.record_tool_call(tool_name="apply_patch", tool_type="function")

        spans = _by_name(exporter.get_finished_spans())
        assert OP_EXECUTE_TOOL in spans
        attrs = spans[OP_EXECUTE_TOOL].attributes
        assert attrs["gen_ai.operation.name"] == OP_EXECUTE_TOOL
        assert attrs["gen_ai.tool.name"] == "apply_patch"
        assert attrs["gen_ai.tool.type"] == "function"

    def test_tool_error_is_recorded_not_raised(self, exporter_and_tracer):
        exporter, tracer = exporter_and_tracer
        emitter = OtelGenAIEmitter(tracer)

        # A failing tool call must not propagate out of observability.
        emitter.record_tool_call(tool_name="run_tests", error=RuntimeError("boom"))

        spans = _by_name(exporter.get_finished_spans())
        attrs = spans[OP_EXECUTE_TOOL].attributes
        assert attrs["error.type"] == "RuntimeError"

    def test_content_not_captured_by_default(self, exporter_and_tracer):
        exporter, tracer = exporter_and_tracer
        emitter = OtelGenAIEmitter(tracer, capture_content=False)

        with emitter.turn_span(session_id="s", user_prompt="secret prompt"):
            emitter.record_llm_call(request_model="m", response_text="secret response")

        spans = _by_name(exporter.get_finished_spans())
        assert "gen_ai.prompt" not in spans[OP_INVOKE_AGENT].attributes
        assert "gen_ai.completion" not in spans[OP_CHAT].attributes

    def test_content_captured_when_enabled(self, exporter_and_tracer):
        exporter, tracer = exporter_and_tracer
        emitter = OtelGenAIEmitter(tracer, capture_content=True)

        with emitter.turn_span(session_id="s", user_prompt="hello"):
            emitter.record_llm_call(request_model="m", response_text="world")

        spans = _by_name(exporter.get_finished_spans())
        assert spans[OP_INVOKE_AGENT].attributes["gen_ai.prompt"] == "hello"
        assert spans[OP_CHAT].attributes["gen_ai.completion"] == "world"


# ---------------------------------------------------------------------------
# Cost estimator: local-model size → price-tier, mirrors genai-otel-instrument.
# ---------------------------------------------------------------------------


class TestCostEstimator:
    @pytest.mark.parametrize(
        ("model", "expected"),
        [
            ("llama3.1:8b", 8.0),
            ("qwen3:0.6b", 0.6),
            ("smollm2:360m", 0.36),
            ("llama3.3:70b-instruct", 70.0),
            ("phi3:14b", 14.0),
        ],
    )
    def test_extract_param_count_from_suffix(self, model, expected):
        assert extract_param_count_billions(model) == pytest.approx(expected)

    def test_extract_param_count_from_hf_name_map(self):
        assert extract_param_count_billions("gpt2") == pytest.approx(0.124)
        assert extract_param_count_billions("t5-small") == pytest.approx(0.06)

    def test_extract_param_count_unknown_returns_none(self):
        assert extract_param_count_billions("some-mystery-model") is None
        assert extract_param_count_billions("") is None

    def test_cost_formula_matches_size_tier(self):
        # 8B → 1–10B tier: (0.0003 prompt, 0.0006 completion) per 1K tokens.
        # 1000 in + 1000 out → 0.0003 + 0.0006 = 0.0009.
        assert estimate_cost_usd("llama3.1:8b", 1000, 1000) == pytest.approx(0.0009)

    def test_cost_tier_boundaries(self):
        # < 1B tier (0.0001, 0.0002): 0.6B model.
        assert estimate_cost_usd("qwen3:0.6b", 1000, 1000) == pytest.approx(0.0003)
        # 80B+ XLARGE tier (0.0012, 0.0012): 120B model.
        assert estimate_cost_usd("bigmodel:120b", 1000, 1000) == pytest.approx(0.0024)

    def test_cost_none_when_size_unknown(self):
        assert estimate_cost_usd("mystery-model", 100, 100) is None

    def test_cost_none_when_no_tokens(self):
        assert estimate_cost_usd("llama3.1:8b", 0, 0) is None
        assert estimate_cost_usd("llama3.1:8b", None, None) is None

    def test_cost_handles_one_sided_tokens(self):
        # output-only is valid (completion price only).
        assert estimate_cost_usd("llama3.1:8b", 0, 1000) == pytest.approx(0.0006)


# ---------------------------------------------------------------------------
# Log emitter: dashboard-shaped OTLP log records, in-memory log exporter.
# ---------------------------------------------------------------------------


class TestLogEmitter:
    def test_disabled_when_logger_is_none(self):
        emitter = OtelLogEmitter(None)
        assert emitter.enabled is False
        # Every method is a safe no-op when there is no logger.
        emitter.session_start(session_id="s")
        emitter.api_response(session_id="s", request_model="m")
        emitter.tool_result(session_id="s", tool_name="t")

    def test_session_start_record_shape(self, log_exporter_and_logger):
        exporter, otel_logger = log_exporter_and_logger
        emitter = OtelLogEmitter(otel_logger, agent_name="hermes")

        emitter.session_start(session_id="sess-1", model="llama3.1:8b")

        attrs = _log_attrs(exporter.get_finished_logs())
        assert EVENT_SESSION_START in attrs
        rec = attrs[EVENT_SESSION_START]
        assert rec["event.name"] == EVENT_SESSION_START
        assert rec["session.id"] == "sess-1"
        assert rec["model"] == "llama3.1:8b"
        assert rec["agent_type"] == "hermes"
        assert rec["gen_ai.system"] == "hermes"
        assert rec["event.sequence"] == 1

    def test_api_response_record_carries_tokens_and_cost(self, log_exporter_and_logger):
        exporter, otel_logger = log_exporter_and_logger
        emitter = OtelLogEmitter(otel_logger)

        emitter.api_response(
            session_id="sess-1",
            request_model="llama3.1:8b",
            response_model="llama3.1:8b-instruct",
            input_tokens=120,
            output_tokens=34,
            cost_usd=0.0009,
            finish_reasons=["stop", "length"],
            ttft_ms=210.0,
        )

        rec = _log_attrs(exporter.get_finished_logs())[EVENT_API_RESPONSE]
        # response_model wins for the by_model breakdown.
        assert rec["model"] == "llama3.1:8b-instruct"
        assert rec["input_tokens"] == 120
        assert rec["output_tokens"] == 34
        assert rec["cost_usd"] == pytest.approx(0.0009)
        assert rec["finish_reason"] == "stop"  # first only
        assert rec["ttft_ms"] == pytest.approx(210.0)

    def test_tool_result_success_record(self, log_exporter_and_logger):
        exporter, otel_logger = log_exporter_and_logger
        emitter = OtelLogEmitter(otel_logger)

        emitter.tool_result(session_id="sess-1", tool_name="read_file", tool_type="function")

        rec = _log_attrs(exporter.get_finished_logs())[EVENT_TOOL_RESULT]
        assert rec["tool_name"] == "read_file"
        assert rec["tool_type"] == "function"
        assert rec["status_code"] == "success"
        assert rec["decision_type"] == "executed"
        assert rec["decision_source"] == "agent"
        assert "error" not in rec

    def test_tool_result_error_record(self, log_exporter_and_logger):
        exporter, otel_logger = log_exporter_and_logger
        emitter = OtelLogEmitter(otel_logger)

        emitter.tool_result(
            session_id="sess-1",
            tool_name="run_tests",
            error=RuntimeError("boom"),
        )

        rec = _log_attrs(exporter.get_finished_logs())[EVENT_TOOL_RESULT]
        assert rec["status_code"] == "error"
        assert rec["decision_type"] == "error"
        assert rec["error_type"] == "RuntimeError"
        assert "boom" in rec["error"]

    def test_content_not_captured_by_default(self, log_exporter_and_logger):
        exporter, otel_logger = log_exporter_and_logger
        emitter = OtelLogEmitter(otel_logger, capture_content=False)

        emitter.session_start(session_id="s", user_prompt="secret prompt")
        emitter.tool_result(
            session_id="s",
            tool_name="t",
            arguments={"path": "secret.txt"},
            result="secret output",
        )

        attrs = _log_attrs(exporter.get_finished_logs())
        assert "prompt" not in attrs[EVENT_SESSION_START]
        assert "tool_input" not in attrs[EVENT_TOOL_RESULT]
        assert "tool_output" not in attrs[EVENT_TOOL_RESULT]

    def test_content_captured_when_enabled(self, log_exporter_and_logger):
        exporter, otel_logger = log_exporter_and_logger
        emitter = OtelLogEmitter(otel_logger, capture_content=True)

        emitter.session_start(session_id="s", user_prompt="hello there")
        emitter.tool_result(
            session_id="s",
            tool_name="t",
            arguments={"path": "a.txt"},
            result="done",
        )

        attrs = _log_attrs(exporter.get_finished_logs())
        assert attrs[EVENT_SESSION_START]["prompt"] == "hello there"
        assert "a.txt" in attrs[EVENT_TOOL_RESULT]["tool_input"]
        assert "done" in attrs[EVENT_TOOL_RESULT]["tool_output"]

    def test_emit_never_raises_on_bad_logger(self):
        class _BoomLogger:
            def emit(self, *_a, **_k):
                raise RuntimeError("transport down")

        emitter = OtelLogEmitter(_BoomLogger())
        # Must be swallowed — observability never breaks the agent.
        emitter.session_start(session_id="s")
        emitter.api_response(session_id="s", request_model="m")
        emitter.tool_result(session_id="s", tool_name="t")


# ---------------------------------------------------------------------------
# Plugin adapter: drives the emitter from the real Hermes hook surface
# (register(ctx) + **kwargs hooks). Uses a fake ctx that records hooks.
# ---------------------------------------------------------------------------


class _FakeCtx:
    """Minimal stand-in for the Hermes plugin context."""

    def __init__(self):
        self.hooks: dict[str, object] = {}

    def register_hook(self, name, fn):
        self.hooks[name] = fn


def _fresh_plugin():
    """Import the plugin module fresh (clears any cached emitter).

    Returns ``(plugin_module, provider_module)`` so callers can monkeypatch
    ``provider.build_tracer`` on the live module object (robust to the
    package being re-imported between tests).
    """
    mod_name = "plugins.observability.otel"
    sys.modules.pop(mod_name, None)
    mod = importlib.import_module(mod_name)
    provider = importlib.import_module(mod_name + ".provider")
    mod.reset_for_tests()
    return mod, provider


def _in_memory_logger():
    """Build an OTel logger backed by an in-memory exporter (no network).

    Used to monkeypatch ``provider.build_logger`` in the adapter tests so the
    dashboard log emitter never tries to reach a real OTLP endpoint.
    """
    exporter = InMemoryLogRecordExporter()
    provider = LoggerProvider(resource=Resource.create({"service.name": "test"}))
    provider.add_log_record_processor(SimpleLogRecordProcessor(exporter))
    return exporter, get_logger("test", logger_provider=provider)


class TestPluginAdapter:
    def test_register_wires_the_documented_hooks(self):
        plugin, _provider = _fresh_plugin()
        ctx = _FakeCtx()
        plugin.register(ctx)
        assert set(ctx.hooks) == {
            "on_session_start",
            "on_session_end",
            "on_session_finalize",
            "on_session_reset",
            "pre_llm_call",
            "transform_llm_output",
            "post_api_request",
            "post_tool_call",
        }

    def test_prompt_and_response_stamped_for_eval(self, exporter_and_tracer, monkeypatch):
        """The prompt arrives on ``pre_llm_call`` and the response on
        ``transform_llm_output`` (``on_session_start`` carries only the session
        id and ``post_api_request`` only token/finish metadata). Both must be
        stamped on the open ``invoke_agent`` span so prompt-side eval (PII /
        injection / restricted-topics) AND response-side eval (toxicity /
        hallucination / output PII) can score the turn."""
        monkeypatch.setenv("HERMES_OTEL_CAPTURE_CONTENT", "true")
        exporter, tracer = exporter_and_tracer
        plugin, provider = _fresh_plugin()
        monkeypatch.setattr(provider, "build_tracer", lambda **_: tracer)
        _log_exporter, otel_logger = _in_memory_logger()
        monkeypatch.setattr(provider, "build_logger", lambda **_: otel_logger)

        ctx = _FakeCtx()
        plugin.register(ctx)
        ctx.hooks["on_session_start"](session_id="sess-p", model="hermes-x")
        ctx.hooks["pre_llm_call"](
            session_id="sess-p",
            user_message="email the report to john@example.com",
        )
        # transform_llm_output is a read-only observer: returns None (output
        # unchanged) but lets us capture the response for eval.
        result = ctx.hooks["transform_llm_output"](
            session_id="sess-p",
            response_text="Sure — sending it to john@example.com now.",
        )
        assert result is None  # must never alter the model output
        ctx.hooks["on_session_end"](session_id="sess-p")

        attrs = _by_name(exporter.get_finished_spans())[OP_INVOKE_AGENT].attributes
        assert attrs["gen_ai.prompt"] == "email the report to john@example.com"
        assert attrs["gen_ai.completion"] == "Sure — sending it to john@example.com now."

    def test_prompt_and_response_gated_by_capture_content(
        self, exporter_and_tracer, monkeypatch
    ):
        """With content capture OFF neither prompt nor response is attached."""
        monkeypatch.setenv("HERMES_OTEL_CAPTURE_CONTENT", "false")
        exporter, tracer = exporter_and_tracer
        plugin, provider = _fresh_plugin()
        monkeypatch.setattr(provider, "build_tracer", lambda **_: tracer)
        _log_exporter, otel_logger = _in_memory_logger()
        monkeypatch.setattr(provider, "build_logger", lambda **_: otel_logger)

        ctx = _FakeCtx()
        plugin.register(ctx)
        ctx.hooks["on_session_start"](session_id="sess-q", model="m")
        ctx.hooks["pre_llm_call"](session_id="sess-q", user_message="secret prompt")
        assert ctx.hooks["transform_llm_output"](
            session_id="sess-q", response_text="secret response"
        ) is None
        ctx.hooks["on_session_end"](session_id="sess-q")

        attrs = _by_name(exporter.get_finished_spans())[OP_INVOKE_AGENT].attributes
        assert "gen_ai.prompt" not in attrs
        assert "gen_ai.completion" not in attrs

    def test_hooks_map_hermes_events_end_to_end(self, exporter_and_tracer, monkeypatch):
        exporter, tracer = exporter_and_tracer
        plugin, provider = _fresh_plugin()
        # Inject our in-memory tracer + logger instead of the network-bound ones.
        monkeypatch.setattr(provider, "build_tracer", lambda **_: tracer)
        log_exporter, otel_logger = _in_memory_logger()
        monkeypatch.setattr(provider, "build_logger", lambda **_: otel_logger)

        ctx = _FakeCtx()
        plugin.register(ctx)

        ctx.hooks["on_session_start"](session_id="sess-9", model="hermes-x")
        # The per-turn invoke_agent span opens on pre_llm_call and closes on
        # transform_llm_output (one span per turn).
        ctx.hooks["pre_llm_call"](session_id="sess-9", model="hermes-x", user_message="hi")
        ctx.hooks["post_api_request"](
            session_id="sess-9",
            model="hermes-x",
            usage={"input_tokens": 50, "output_tokens": 10},
            cost_usd=0.0009,
            finish_reason="stop",
        )
        ctx.hooks["post_tool_call"](
            session_id="sess-9", tool_name="read_file", args={"path": "a.txt"}
        )
        ctx.hooks["transform_llm_output"](session_id="sess-9", response_text="done")
        ctx.hooks["on_session_end"](session_id="sess-9")

        names = sorted(s.name for s in exporter.get_finished_spans())
        assert names == [OP_CHAT, OP_EXECUTE_TOOL, OP_INVOKE_AGENT]

        spans = _by_name(exporter.get_finished_spans())
        assert spans[OP_INVOKE_AGENT].attributes["gen_ai.conversation.id"] == "sess-9"
        assert spans[OP_CHAT].attributes["gen_ai.usage.input_tokens"] == 50
        assert list(spans[OP_CHAT].attributes["gen_ai.response.finish_reasons"]) == ["stop"]
        assert spans[OP_EXECUTE_TOOL].attributes["gen_ai.tool.name"] == "read_file"

        # The richer scope ALSO emits dashboard log records for the same events.
        log_attrs = _log_attrs(log_exporter.get_finished_logs())
        assert set(log_attrs) == {
            EVENT_SESSION_START,
            EVENT_API_RESPONSE,
            EVENT_TOOL_RESULT,
        }
        assert log_attrs[EVENT_API_RESPONSE]["input_tokens"] == 50
        assert log_attrs[EVENT_API_RESPONSE]["cost_usd"] == pytest.approx(0.0009)
        assert log_attrs[EVENT_TOOL_RESULT]["tool_name"] == "read_file"
        # session.id is propagated onto the per-call records via the active turn.
        assert log_attrs[EVENT_API_RESPONSE]["session.id"] == "sess-9"

    def test_post_api_request_estimates_cost_for_local_model(
        self, exporter_and_tracer, monkeypatch
    ):
        """When the agent reports no price, cost is estimated from model size."""
        exporter, tracer = exporter_and_tracer
        plugin, provider = _fresh_plugin()
        monkeypatch.setattr(provider, "build_tracer", lambda **_: tracer)
        log_exporter, otel_logger = _in_memory_logger()
        monkeypatch.setattr(provider, "build_logger", lambda **_: otel_logger)

        ctx = _FakeCtx()
        plugin.register(ctx)

        ctx.hooks["on_session_start"](session_id="s", model="llama3.1:8b")
        ctx.hooks["post_api_request"](
            session_id="s",
            model="llama3.1:8b",
            usage={"input_tokens": 1000, "output_tokens": 1000},
            # no cost_usd → estimated from the 8B size tier (0.0003 + 0.0006).
        )

        spans = _by_name(exporter.get_finished_spans())
        assert spans[OP_CHAT].attributes["cost_usd"] == pytest.approx(0.0009)
        log_attrs = _log_attrs(log_exporter.get_finished_logs())
        assert log_attrs[EVENT_API_RESPONSE]["cost_usd"] == pytest.approx(0.0009)

    def test_hooks_survive_malformed_event(self, exporter_and_tracer, monkeypatch):
        """A bad event must never raise out of a hook (observability safety)."""
        exporter, tracer = exporter_and_tracer
        plugin, provider = _fresh_plugin()
        monkeypatch.setattr(provider, "build_tracer", lambda **_: tracer)
        _log_exporter, otel_logger = _in_memory_logger()
        monkeypatch.setattr(provider, "build_logger", lambda **_: otel_logger)
        ctx = _FakeCtx()
        plugin.register(ctx)

        # Missing everything — swallowed and logged, not raised.
        ctx.hooks["post_api_request"]()
        ctx.hooks["post_tool_call"]()  # tool_name falls back to "unknown"

        spans = _by_name(exporter.get_finished_spans())
        assert spans[OP_EXECUTE_TOOL].attributes["gen_ai.tool.name"] == "unknown"

    def test_tool_error_status_recorded(self, exporter_and_tracer, monkeypatch):
        exporter, tracer = exporter_and_tracer
        plugin, provider = _fresh_plugin()
        monkeypatch.setattr(provider, "build_tracer", lambda **_: tracer)
        _log_exporter, otel_logger = _in_memory_logger()
        monkeypatch.setattr(provider, "build_logger", lambda **_: otel_logger)
        ctx = _FakeCtx()
        plugin.register(ctx)

        ctx.hooks["post_tool_call"](tool_name="terminal", status="error")

        spans = _by_name(exporter.get_finished_spans())
        assert spans[OP_EXECUTE_TOOL].attributes["error.type"] == "error"
