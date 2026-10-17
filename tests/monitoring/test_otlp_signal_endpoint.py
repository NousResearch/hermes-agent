"""OTLP/HTTP per-signal endpoint derivation. No OpenTelemetry SDK needed: the exporter
constructor is faked, so these run without the optional ``otlp`` extra."""

from __future__ import annotations

import pytest

import agent.monitoring.otlp_exporter as OE


@pytest.mark.parametrize("signal", ["traces", "metrics", "logs"])
def test_bare_collector_url_gets_the_signal_path(signal):
    # A bare collector URL must get /v1/<signal>, not 404 against the collector root.
    assert OE._signal_endpoint("http://127.0.0.1:4318", signal) == f"http://127.0.0.1:4318/v1/{signal}"
    assert OE._signal_endpoint("http://127.0.0.1:4318/", signal) == f"http://127.0.0.1:4318/v1/{signal}"


def test_known_signal_paths_swap_to_the_target_signal():
    assert OE._signal_endpoint("http://127.0.0.1:4318/v1/traces", "metrics") == "http://127.0.0.1:4318/v1/metrics"
    assert OE._signal_endpoint("http://127.0.0.1:4318/v1/metrics", "logs") == "http://127.0.0.1:4318/v1/logs"
    assert OE._signal_endpoint("http://127.0.0.1:4318/v1/logs", "traces") == "http://127.0.0.1:4318/v1/traces"
    assert OE._signal_endpoint("http://127.0.0.1:4318/v1/traces", "traces") == "http://127.0.0.1:4318/v1/traces"


def test_custom_vendor_path_passes_through():
    assert OE._signal_endpoint("https://otlp.example.com/api/v2/otlp", "traces") == "https://otlp.example.com/api/v2/otlp"


def test_span_exporter_receives_the_traces_path_for_a_bare_endpoint(monkeypatch):
    captured = {}

    class FakeSpanExporter:
        def __init__(self, endpoint, headers):
            captured["endpoint"] = endpoint

    monkeypatch.setattr(OE, "_require_sdk", lambda *a, **k: {"OTLPSpanExporter": FakeSpanExporter})
    OE.build_exporter({"monitoring": {"export": {"otlp": {"endpoint": "http://127.0.0.1:4318"}}}})
    assert captured["endpoint"] == "http://127.0.0.1:4318/v1/traces"
