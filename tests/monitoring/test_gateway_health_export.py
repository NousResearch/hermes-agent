from __future__ import annotations

import pytest


























def test_otlp_attrs_redact_strings_and_never_export_profile():
    from agent.monitoring.otlp_exporter import _span_attrs

    attrs = _span_attrs({
        "event": "gateway_health",
        "name": "gateway.lifecycle",
        "profile": "user@example.com",
        "exit_reason": "Bearer top-secret-token-0123456789 for user@example.com",
    })

    assert "hermes.profile" not in attrs
    assert "top-secret-token-0123456789" not in str(attrs)
    assert "user@example.com" not in str(attrs)


def test_resource_attributes_are_allowlisted_and_sanitized():
    from agent.monitoring.otlp_exporter import _safe_resource_attributes

    attrs = _safe_resource_attributes({
        "service.name": "hermes-gateway",
        "service.instance.id": "install-1",
        "deployment.environment.name": "staging",
        "user.email": "user@example.com",
        "authorization": "Bearer top-secret-token-0123456789",
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
        "error_code": "Bearer top-secret-token-0123456789",
        "custom": "must-not-egress",
    })

    assert "hermes.profile" not in attrs
    assert "hermes.custom" not in attrs
    assert "top-secret-token-0123456789" not in str(attrs)




















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


def test_managed_process_events_and_gauges_reach_loopback_collector(tmp_path, monkeypatch):
    """A reap recorded by the owner-death supervisor after its owner died sits in the spool of the home it
    belongs to (a served profile here); the gateway exports it as a span with only allowlisted attributes,
    alongside the screen and owned-group gauges."""
    import threading
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    import pytest

    pytest.importorskip("opentelemetry.sdk.metrics", reason="otlp extra not installed")
    from opentelemetry.proto.collector.metrics.v1.metrics_service_pb2 import ExportMetricsServiceRequest
    from opentelemetry.proto.collector.trace.v1.trace_service_pb2 import ExportTraceServiceRequest
    from agent.monitoring import emitter, managed_processes
    from agent.monitoring.gateway_health_export import start_gateway_health_export

    own, served = tmp_path / "own", tmp_path / "served"
    managed_processes.set_spooling([served], True)
    import time
    reaped_at = (time.time_ns() // 10**9 - 3600) * 10**9  # an hour ago, whole seconds as the shell writes it
    (managed_processes.spool_dir(served) / f"{reaped_at}-7.json").write_text(
        '{"event":"managed_process","subsystem":"owned_children","action":"orphan_reap","outcome":"killed",'
        f'"actor":"supervisor","count":3,"ts_ns":{reaped_at}}}\n', encoding="utf-8")
    received = {"/v1/traces": [], "/v1/metrics": []}
    parsers = {"/v1/traces": ExportTraceServiceRequest, "/v1/metrics": ExportMetricsServiceRequest}

    class Collector(BaseHTTPRequestHandler):
        def do_POST(self):
            payload = self.rfile.read(int(self.headers["Content-Length"]))
            if self.path in parsers:
                received[self.path].append(parsers[self.path].FromString(payload))
            self.send_response(200)
            self.send_header("Content-Length", "0")
            self.end_headers()

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
            "gateway_health_export": {"enabled": True, "metrics_enabled": True, "diagnostic_events_enabled": True},
        }}, homes=lambda: [own, served])
        assert runtime.enabled, runtime.reason
        assert managed_processes.spool_dir(own).is_dir(), "the gateway's own home must spool too"
        emitter.get_emitter().flush(timeout=5)
        runtime.streamer._provider.force_flush(timeout_millis=10000)
        assert runtime.metric_provider.force_flush(timeout_millis=10000)
        spans = [span for request in received["/v1/traces"] for resource in request.resource_spans
                 for scope in resource.scope_spans for span in scope.spans if span.name == "hermes.managed_process"]
        assert len(spans) == 1, "the spooled reap was not exported exactly once"
        assert spans[0].start_time_unix_nano == reaped_at, "exported at drain time, not when it happened"
        attrs = {kv.key: kv.value for kv in spans[0].attributes}
        assert {k: (v.string_value or v.int_value) for k, v in attrs.items()} == {
            "hermes.event": "managed_process", "hermes.subsystem": "owned_children", "hermes.action": "orphan_reap",
            "hermes.outcome": "killed", "hermes.actor": "supervisor", "hermes.count": 3}
        assert list(managed_processes.spool_dir(served).iterdir()) == []
        names = {metric.name for request in received["/v1/metrics"] for resource in request.resource_metrics
                 for scope in resource.scope_metrics for metric in scope.metrics}
        assert {"hermes.bot_desktop.screens.running", "hermes.bot_desktop.screens.human_held",
                "hermes.owned_children.groups"} <= names
    finally:
        if runtime is not None:
            runtime.shutdown()
        server.shutdown()
        server.server_close()
        thread.join(timeout=10)


@pytest.mark.parametrize("why", ["export off", "diagnostics off", "sdk missing"])
def test_a_gateway_that_does_not_drain_removes_the_spool_so_nothing_accumulates(tmp_path, monkeypatch, why):
    import agent.monitoring.gateway_health_export as gw
    from agent.monitoring import managed_processes
    from agent.monitoring.otlp_exporter import OTLPUnavailable

    managed_processes.set_spooling([tmp_path], True)
    (managed_processes.spool_dir(tmp_path) / "1-1.json").write_text("{}", encoding="utf-8")
    on = {"export": {"otlp": {"enabled": True, "endpoint": "http://127.0.0.1:9"}},
          "gateway_health_export": {"enabled": True, "metrics_enabled": False,
                                    "diagnostic_events_enabled": why != "diagnostics off"}}
    if why == "sdk missing":
        monkeypatch.setattr(gw.otlp_exporter, "_require_sdk", lambda *a, **k: (_ for _ in ()).throw(OTLPUnavailable("x")))
    runtime = gw.start_gateway_health_export({"monitoring": {} if why == "export off" else on}, homes=lambda: [tmp_path])
    try:
        assert not runtime.spooling
        assert not managed_processes.spool_dir(tmp_path).exists()
    finally:
        runtime.shutdown()


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




def test_the_gateway_reports_its_own_home_and_every_profile_it_serves_now(monkeypatch, tmp_path):
    """The served set changes while the gateway runs: the export reads it at each pass, not once at start."""
    from types import SimpleNamespace

    import agent.monitoring.gateway_health_export as gw
    import hermes_constants
    from gateway.run_startup import GatewayStartupMixin

    seen = {}
    monkeypatch.setattr(gw, "start_gateway_health_export", lambda config, *, homes: seen.setdefault("homes", homes))
    monkeypatch.setattr(hermes_constants, "get_routing_process_hermes_home", lambda: tmp_path / "own")
    runner = SimpleNamespace(_served_profile_homes=None)
    GatewayStartupMixin._start_gateway_health_export(runner)
    assert list(seen["homes"]()) == [tmp_path / "own"]
    runner._served_profile_homes = {"work": tmp_path / "work"}
    assert list(seen["homes"]()) == [tmp_path / "own", tmp_path / "work"]


def test_a_profile_served_after_startup_loses_its_spool_when_this_gateway_does_not_drain(tmp_path):
    """Startup removes spools before secondary profiles are served; each later served set must be swept too,
    unless this gateway drains spools (then it keeps them)."""
    from types import SimpleNamespace

    from agent.monitoring import managed_processes
    from gateway.run_profile_reconcile import GatewayProfileReconcileMixin

    work = tmp_path / "work"
    managed_processes.set_spooling([work], True)
    runner = SimpleNamespace(_served_profile_signatures={}, _gateway_health_export_runtime=SimpleNamespace(spooling=True))
    GatewayProfileReconcileMixin._note_served_profiles(runner, [("work", work)])
    assert managed_processes.spool_dir(work).is_dir(), "a draining gateway removed a spool it drains"
    runner._gateway_health_export_runtime = None  # torn down: its spools wait for the restart
    GatewayProfileReconcileMixin._note_served_profiles(runner, [("work", work)])
    assert managed_processes.spool_dir(work).is_dir(), "a served-set change during teardown removed the spools"
    runner._gateway_health_export_runtime = SimpleNamespace(spooling=False)
    GatewayProfileReconcileMixin._note_served_profiles(runner, [("work", work)])
    assert not managed_processes.spool_dir(work).exists()


def test_a_config_load_that_raises_turns_export_off_and_removes_undrained_spools(tmp_path, monkeypatch):
    """A start that raised would leave the runtime unset, and served-set changes keep spools then. (An unreadable
    file does not get here: load_config returns its last-known-good settings, and those decide export.)"""
    from types import SimpleNamespace

    from hermes_cli import config
    import hermes_constants
    from agent.monitoring import managed_processes
    from gateway.run_startup import GatewayStartupMixin

    managed_processes.set_spooling([tmp_path], True)
    # e.g. HERMES_HOME cannot be created; a broken file never raises (last-known-good settings instead)
    monkeypatch.setattr(config, "load_config", lambda: (_ for _ in ()).throw(PermissionError(13, "denied")))
    monkeypatch.setattr(hermes_constants, "get_routing_process_hermes_home", lambda: tmp_path)
    runtime = GatewayStartupMixin._start_gateway_health_export(SimpleNamespace(_served_profile_homes=None))
    assert runtime.enabled is False
    assert not managed_processes.spool_dir(tmp_path).exists()
