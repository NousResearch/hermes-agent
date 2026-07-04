"""OTLP exporter tests.

Skip cleanly when the optional OTel SDK isn't installed. When it is, verify:
  * event -> OTel span attribute mapping
  * headers_env resolves the value from the named environment variable, not config
  * a failing or slow subscriber never breaks the emitter hot path
  * is_enabled / is_available gating
"""

from __future__ import annotations

import sqlite3
import time

import pytest

import hermes_state
from agent.telemetry import otlp_exporter as OE
from agent.telemetry.emitter import TelemetryEmitter
from agent.telemetry.events import ModelCallEvent, RunEvent, ToolCallEvent

otel = pytest.importorskip("opentelemetry.sdk.trace", reason="otlp extra not installed")


def _in_memory_provider():
    """A TracerProvider with an in-memory span exporter (no network)."""
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
    provider = TracerProvider()
    mem = InMemorySpanExporter()
    provider.add_span_processor(SimpleSpanProcessor(mem))
    return provider, mem


def test_event_maps_to_span_with_real_attrs():
    provider, mem = _in_memory_provider()
    batch = [
        {"event": "run", "entrypoint": "gateway", "platform": "telegram",
         "end_reason": "completed", "model_call_count": 2, "tool_call_count": 3},
        {"event": "model_call", "provider": "anthropic", "model": "claude-opus-4",
         "input_tokens": 5000, "output_tokens": 800},
        {"event": "tool_call", "tool_name": "web_search", "result_class": "ok"},
    ]
    n = OE.export_batch(provider, batch)
    assert n == 3
    spans = mem.get_finished_spans()
    names = {s.name for s in spans}
    assert names == {"hermes.run", "hermes.model_call", "hermes.tool_call"}
    # real values present as span attributes
    run = [s for s in spans if s.name == "hermes.run"][0]
    assert run.attributes["hermes.entrypoint"] == "gateway"
    assert run.attributes["hermes.platform"] == "telegram"
    model = [s for s in spans if s.name == "hermes.model_call"][0]
    assert model.attributes["hermes.model"] == "claude-opus-4"
    assert model.attributes["hermes.provider"] == "anthropic"
    tool = [s for s in spans if s.name == "hermes.tool_call"][0]
    assert tool.attributes["hermes.tool_name"] == "web_search"


def test_headers_resolve_from_env_not_value(monkeypatch):
    monkeypatch.setenv("MY_OTLP_TOKEN", "supersecretvalue")
    resolved = OE._resolve_headers({"Authorization": "MY_OTLP_TOKEN"})
    assert resolved == {"Authorization": "supersecretvalue"}
    # missing env var -> skipped, not crashed
    assert OE._resolve_headers({"X": "NOPE_NOT_SET"}) == {}


def test_is_enabled_requires_endpoint_and_flag():
    assert OE.is_enabled({"telemetry": {"export": {"otlp": {"enabled": True, "endpoint": "http://x"}}}}) is True
    assert OE.is_enabled({"telemetry": {"export": {"otlp": {"enabled": True}}}}) is False
    assert OE.is_enabled({"telemetry": {"export": {"otlp": {"enabled": False, "endpoint": "http://x"}}}}) is False
    assert OE.is_enabled({}) is False


def test_require_sdk_routes_through_lazy_install(monkeypatch):
    # _require_sdk(auto_install=True) should call lazy_deps.ensure('export.otlp').
    import tools.lazy_deps as ld
    calls = []
    monkeypatch.setattr(ld, "ensure", lambda feature, **kw: calls.append((feature, kw)))
    OE._require_sdk(auto_install=True, prompt=False)
    assert calls == [("export.otlp", {"prompt": False})]


def test_is_available_does_not_install(monkeypatch):
    # A pure availability check must NEVER trigger an install.
    import tools.lazy_deps as ld
    calls = []
    monkeypatch.setattr(ld, "ensure", lambda *a, **k: calls.append(a))
    OE.is_available()
    assert calls == []


def test_export_otlp_feature_specs_match_pyproject():
    # The LAZY_DEPS entry must track the [otlp] extra in pyproject.toml.
    from tools.lazy_deps import feature_specs
    import pathlib, re
    specs = set(feature_specs("export.otlp"))
    pyproject = pathlib.Path(__file__).resolve().parents[2] / "pyproject.toml"
    text = pyproject.read_text(encoding="utf-8")
    m = re.search(r"^otlp\s*=\s*\[([^\]]*)\]", text, re.MULTILINE)
    assert m, "otlp extra not found in pyproject.toml"
    extra = set(re.findall(r'"([^"]+)"', m.group(1)))
    assert specs == extra, f"LAZY_DEPS {specs} != pyproject extra {extra}"


def test_streamer_subscription_receives_events(tmp_path, monkeypatch):
    # Wire an OTLPStreamer-like subscriber via the in-memory provider.
    provider, mem = _in_memory_provider()

    db = tmp_path / "state.db"
    conn = sqlite3.connect(db); conn.executescript(hermes_state.SCHEMA_SQL); conn.close()
    em = TelemetryEmitter(events_path=tmp_path / "t" / "e.jsonl", db_path=db)

    def subscriber(batch):
        OE.export_batch(provider, batch)

    em.subscribe(subscriber)
    em.emit(RunEvent(run_id="r1", trace_id="t1", entrypoint="cli", end_reason="completed"))
    em.emit(ModelCallEvent(span_id="m1", run_id="r1", provider="anthropic", model="claude-opus-4"))
    em.flush()
    em.close()
    spans = mem.get_finished_spans()
    assert {s.name for s in spans} == {"hermes.run", "hermes.model_call"}


def test_failing_subscriber_never_breaks_hot_path(tmp_path):
    db = tmp_path / "state.db"
    conn = sqlite3.connect(db); conn.executescript(hermes_state.SCHEMA_SQL); conn.close()
    em = TelemetryEmitter(events_path=tmp_path / "t" / "e.jsonl", db_path=db)

    def bad_subscriber(batch):
        time.sleep(0.2)
        raise RuntimeError("OTLP collector down")

    em.subscribe(bad_subscriber)
    start = time.monotonic()
    for i in range(30):
        em.emit(ModelCallEvent(span_id=f"s{i}", run_id="r1", input_tokens=1))
    elapsed = time.monotonic() - start
    # emit() returns immediately regardless of the broken subscriber
    assert elapsed < 1.0
    em.flush()
    # durable writes still happened despite the subscriber raising
    conn = sqlite3.connect(db)
    assert conn.execute("SELECT COUNT(*) FROM tel_model_calls").fetchone()[0] == 30
    conn.close()
    em.close()


def test_export_once_reads_db_and_returns_count(tmp_path, monkeypatch):
    db = tmp_path / "state.db"
    conn = sqlite3.connect(db); conn.executescript(hermes_state.SCHEMA_SQL); conn.close()
    em = TelemetryEmitter(events_path=tmp_path / "t" / "e.jsonl", db_path=db)
    em.emit(RunEvent(run_id="r1", trace_id="t1", entrypoint="cli", end_reason="completed",
                     start_ns=time.time_ns(), end_ns=time.time_ns()))
    em.emit(ToolCallEvent(span_id="w1", run_id="r1", tool_name="web_search", result_class="ok"))
    em.flush(); em.close()

    # Patch the provider builder to an in-memory one (no network). The processor
    # stand-in only needs force_flush(); provider.shutdown() works on the real one.
    provider, mem = _in_memory_provider()

    class _Proc:
        def force_flush(self, *a, **k):
            return True

    monkeypatch.setattr(OE, "_make_provider", lambda config: (provider, _Proc()))
    n = OE.export_once({"telemetry": {"export": {"otlp": {"enabled": True, "endpoint": "http://x"}}}}, db_path=db)
    assert n == 2
    assert len(mem.get_finished_spans()) == 2
