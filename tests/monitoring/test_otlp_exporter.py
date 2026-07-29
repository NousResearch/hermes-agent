"""OTLP exporter tests: config resolution, span mapping, streaming subscriber.

No SQLite involved — monitoring is an egress path, so the exporter consumes
emitter batches directly. Uses the in-memory OTel span exporter; skipped when
the optional otlp extra is not installed.
"""

from __future__ import annotations

import pytest

otel = pytest.importorskip("opentelemetry.sdk.trace", reason="otlp extra not installed")

import agent.monitoring.otlp_exporter as OE
from agent.monitoring.emitter import MonitoringEmitter


def _mem_provider():
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    return provider, exporter


def test_gateway_health_event_maps_to_span_with_attrs():
    provider, mem = _mem_provider()
    n = OE.export_batch(provider, [{
        "event": "gateway_health", "name": "gateway.lifecycle",
        "old_state": "starting", "new_state": "running",
        "active_agents": 2, "pid": 4242,
    }])
    assert n == 1
    spans = mem.get_finished_spans()
    assert spans[0].name == "hermes.gateway_health"
    attrs = dict(spans[0].attributes or {})
    assert attrs["hermes.old_state"] == "starting"
    assert attrs["hermes.new_state"] == "running"
    assert attrs["hermes.active_agents"] == 2


def test_gateway_diagnostic_event_drops_arbitrary_message_content():
    provider, mem = _mem_provider()
    OE.export_batch(provider, [{
        "event": "gateway_diagnostic", "name": "platform.fatal",
        "subsystem": "platform.slack", "error_class": "auth_failed",
        "redacted_message": "Unauthorized user: acct_7f3a (Alice Smith)",
        "severity": "error",
    }])
    attrs = dict(mem.get_finished_spans()[0].attributes or {})
    assert attrs["hermes.error_class"] == "auth_failed"
    assert "hermes.redacted_message" not in attrs
    assert "acct_7f3a" not in str(attrs)
    assert "Alice Smith" not in str(attrs)


def test_unknown_event_kind_exports_no_attrs_beyond_kind():
    provider, mem = _mem_provider()
    OE.export_batch(provider, [{"event": "model_call", "provider": "anthropic",
                                "model": "claude-opus-4"}])
    attrs = dict(mem.get_finished_spans()[0].attributes or {})
    # Non-monitoring event kinds carry no attribute mapping on this plane.
    assert attrs == {"hermes.event": "model_call"}


def test_headers_resolve_from_env_not_value(monkeypatch):
    monkeypatch.setenv("DD_KEY_ENV", "secret-value")
    resolved = OE._resolve_headers({"DD-API-KEY": "DD_KEY_ENV", "X-Missing": "NOPE_ENV"})
    assert resolved == {"DD-API-KEY": "secret-value"}


def test_is_enabled_requires_endpoint_and_flag():
    assert OE.is_enabled({"monitoring": {"export": {"otlp": {"enabled": True, "endpoint": "http://x"}}}})
    assert not OE.is_enabled({"monitoring": {"export": {"otlp": {"enabled": True}}}})
    assert not OE.is_enabled({"monitoring": {"export": {"otlp": {"endpoint": "http://x"}}}})
    assert not OE.is_enabled({})


def test_trace_resource_includes_stable_hashed_instance():
    attrs = OE._resource_attributes(
        {"monitoring": {"install_id": "private-install-id"}}
    )

    assert attrs["service.name"] == "hermes-gateway"
    assert attrs["service.instance.id"].startswith("sha256:")
    assert len(attrs["service.instance.id"]) == len("sha256:") + 24
    assert "private-install-id" not in str(attrs)
    assert attrs["telemetry.scope"] == "gateway_monitoring"


def test_export_otlp_feature_specs_match_pyproject():
    from tools.lazy_deps import LAZY_DEPS
    import re
    from pathlib import Path

    specs = set(LAZY_DEPS["export.otlp"])
    pyproject = Path(__file__).resolve().parents[2] / "pyproject.toml"
    m = re.search(r'^otlp = \[(.*?)\]', pyproject.read_text(), re.M | re.S)
    assert m, "otlp extra missing from pyproject.toml"
    extra = set(re.findall(r'"([^"]+)"', m.group(1)))
    assert specs == extra


def test_streamer_receives_events_and_respects_filter(monkeypatch):
    provider, mem = _mem_provider()
    monkeypatch.setattr(OE, "_make_provider", lambda cfg: (provider, None))
    streamer = OE.OTLPStreamer(
        {}, event_filter=lambda ev: ev.get("event") == "gateway_health")

    em = MonitoringEmitter()
    em.subscribe(streamer)
    em.emit({"event": "gateway_health", "name": "gateway.health_snapshot"})
    em.emit({"event": "model_call", "provider": "anthropic"})  # filtered out
    em.flush()
    em.close()

    spans = mem.get_finished_spans()
    assert [s.name for s in spans] == ["hermes.gateway_health"]
    assert streamer.exported == 1


def test_failing_streamer_never_breaks_emitter(monkeypatch):
    def boom(cfg):
        raise RuntimeError("no provider")

    em = MonitoringEmitter()

    def bad_subscriber(batch):
        raise RuntimeError("export down")

    seen: list = []
    em.subscribe(bad_subscriber)
    em.subscribe(lambda batch: seen.extend(batch))
    em.emit({"event": "gateway_health", "name": "gateway.lifecycle"})
    em.flush()
    em.close()
    assert len(seen) == 1
