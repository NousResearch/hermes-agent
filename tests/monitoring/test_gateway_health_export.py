from __future__ import annotations


























def test_otlp_attrs_redact_strings_and_never_export_profile():
    from agent.monitoring.otlp_exporter import _span_attrs

    attrs = _span_attrs({
        "event": "gateway_health",
        "name": "gateway.lifecycle",
        "profile": "user@example.com",
        "exit_reason": "Bearer top-secret-token for user@example.com",
    })

    assert "hermes.profile" not in attrs
    assert "top-secret-token" not in str(attrs)
    assert "user@example.com" not in str(attrs)


def test_resource_attributes_are_allowlisted_and_sanitized():
    from agent.monitoring.otlp_exporter import _safe_resource_attributes

    attrs = _safe_resource_attributes({
        "service.name": "hermes-gateway",
        "service.instance.id": "install-1",
        "deployment.environment.name": "staging",
        "user.email": "user@example.com",
        "authorization": "Bearer top-secret-token",
        "custom.request.id": "unbounded",
    })

    assert attrs == {
        "service.name": "hermes-gateway",
        "service.instance.id": attrs["service.instance.id"],
        "deployment.environment.name": "staging",
    }
    assert attrs["service.instance.id"].startswith("sha256:")
    assert "install-1" not in attrs["service.instance.id"]






def test_diagnostic_log_attributes_are_allowlisted_redacted_and_profile_free():
    from agent.monitoring.gateway_health_export import _diagnostic_log_attributes

    attrs = _diagnostic_log_attributes({
        "event": "gateway_diagnostic",
        "name": "platform.fatal",
        "subsystem": "platform.slack",
        "profile": "user@example.com",
        "error_code": "Bearer top-secret-token",
        "custom": "must-not-egress",
    })

    assert "hermes.profile" not in attrs
    assert "hermes.custom" not in attrs
    assert "top-secret-token" not in str(attrs)




















def test_gateway_health_metrics_reach_loopback_collector():
    """Exercise the real SDK loader, metric provider and OTLP transport together."""
    import threading
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    import pytest

    pytest.importorskip("opentelemetry.sdk.metrics", reason="otlp extra not installed")
    from opentelemetry.proto.collector.metrics.v1.metrics_service_pb2 import ExportMetricsServiceRequest
    from agent.monitoring.gateway_health_export import start_gateway_health_export

    received = []
    delivered = threading.Event()

    class Collector(BaseHTTPRequestHandler):
        def do_POST(self):
            payload = self.rfile.read(int(self.headers["Content-Length"]))
            received.append((self.path, ExportMetricsServiceRequest.FromString(payload)))
            self.send_response(200)
            self.send_header("Content-Length", "0")
            self.end_headers()
            delivered.set()

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Collector)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    runtime = None
    try:
        runtime = start_gateway_health_export({"monitoring": {
            "install_id": "test-install",
            "export": {"otlp": {"enabled": True, "endpoint": f"http://127.0.0.1:{server.server_port}/v1/traces"}},
            "gateway_health_export": {
                "enabled": True, "metrics_enabled": True, "diagnostic_events_enabled": False,
            },
        }})
        assert runtime.enabled, runtime.reason
        assert runtime.metric_provider.force_flush(timeout_millis=10000)
        assert delivered.wait(timeout=10)
        assert all(path == "/v1/metrics" for path, _ in received)
        names = {metric.name for _, request in received
                 for resource in request.resource_metrics
                 for scope in resource.scope_metrics for metric in scope.metrics}
        assert "hermes.gateway.up" in names
    finally:
        if runtime is not None:
            runtime.shutdown()
        server.shutdown()
        server.server_close()
        thread.join(timeout=10)


def test_start_gateway_health_export_reports_otlp_unavailable_when_sdk_missing(monkeypatch):
    import agent.monitoring.gateway_health_export as gw
    from agent.monitoring.otlp_exporter import OTLPUnavailable

    config = {
        "monitoring": {
            "export": {"otlp": {"enabled": True, "endpoint": "http://127.0.0.1:4318"}},
            "gateway_health_export": {"enabled": True, "metrics_enabled": True, "diagnostic_events_enabled": False},
        }
    }

    def no_sdk(*args, **kwargs):
        raise OTLPUnavailable("missing")

    monkeypatch.setattr(gw.otlp_exporter, "_require_sdk", no_sdk)

    runtime = gw.start_gateway_health_export(config)

    assert runtime.enabled is False
    assert runtime.reason == "otlp_unavailable"


def test_install_id_persists_across_calls(tmp_path, monkeypatch):
    """A minted install id must survive restarts (service.instance.id continuity)."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    (tmp_path / "config.yaml").write_text("{}\n")

    import hermes_cli.config as cfg_mod
    from agent.monitoring.policy import ensure_install_id

    first = ensure_install_id(cfg_mod.load_config())
    assert first and first != "unknown"
    # Persisted: a fresh load (simulating a new gateway process) returns the same id.
    second = ensure_install_id(cfg_mod.load_config())
    assert second == first
    assert first in (tmp_path / "config.yaml").read_text()


