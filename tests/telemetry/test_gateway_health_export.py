from __future__ import annotations

import logging


def test_default_config_keeps_gateway_health_export_disabled():
    from hermes_cli.config import DEFAULT_CONFIG

    cfg = DEFAULT_CONFIG["telemetry"]["gateway_health_export"]

    assert cfg["enabled"] is False
    assert cfg["metrics_enabled"] is True
    assert cfg["diagnostic_events_enabled"] is True
    assert cfg["warning_error_events_enabled"] is True
    assert cfg["export_interval_seconds"] == 60
    assert cfg["logs_export_interval_seconds"] == 5
    assert cfg["redaction"]["enabled"] is True
    assert cfg["redaction"]["include_raw_stack"] is False


def test_gateway_health_snapshot_maps_runtime_status_to_low_cardinality_metrics():
    from agent.telemetry.gateway_health import build_gateway_health_snapshot

    runtime = {
        "gateway_state": "running",
        "pid": 1234,
        "active_agents": "2",
        "restart_requested": False,
        "platforms": {
            "slack": {"state": "running"},
            "telegram": {
                "state": "fatal",
                "error_code": "auth_failed",
                "error_message": "token xoxb-secret rejected for user 123",
            },
        },
    }

    snapshot = build_gateway_health_snapshot(
        runtime,
        gateway_running=True,
        profile="default",
        install_id="install-1",
        version="2026.7.test",
        supervision_mode="manual",
    )

    metric_names = {m.name for m in snapshot.metrics}
    assert {
        "hermes.gateway.up",
        "hermes.gateway.active_agents",
        "hermes.gateway.busy",
        "hermes.gateway.drainable",
        "hermes.gateway.restart_requested",
        "hermes.platform.up",
        "hermes.platform.degraded",
    } <= metric_names

    active = next(m for m in snapshot.metrics if m.name == "hermes.gateway.active_agents")
    assert active.value == 2
    assert active.attributes == {
        "hermes.profile": "default",
        "service.instance.id": "install-1",
        "service.version": "2026.7.test",
        "hermes.supervision_mode": "manual",
    }

    busy = next(m for m in snapshot.metrics if m.name == "hermes.gateway.busy")
    drainable = next(m for m in snapshot.metrics if m.name == "hermes.gateway.drainable")
    assert busy.value == 1
    assert drainable.value == 1

    degraded = next(
        m for m in snapshot.metrics
        if m.name == "hermes.platform.degraded" and m.attributes["hermes.platform"] == "telegram"
    )
    assert degraded.value == 1
    assert degraded.attributes["hermes.error_code"] == "auth_failed"
    assert all("secret" not in str(v).lower() for v in degraded.attributes.values())


def test_gateway_health_snapshot_emits_content_free_diagnostic_event():
    from agent.telemetry.gateway_health import build_gateway_health_snapshot

    snapshot = build_gateway_health_snapshot(
        {
            "gateway_state": "running",
            "active_agents": 1,
            "platforms": {
                "slack": {"state": "fatal", "error_code": "auth_failed", "error_message": "Bearer sk-live-secret"},
            },
        },
        gateway_running=True,
        profile="default",
        install_id="install-1",
        version="v-test",
        supervision_mode="container",
    )

    events = [event.to_dict() for event in snapshot.events]
    health = next(e for e in events if e["event"] == "gateway_health")
    platform = next(e for e in events if e["event"] == "gateway_diagnostic" and e["name"] == "platform.fatal")

    assert health["gateway_state"] == "running"
    assert health["active_agents"] == 1
    assert health["gateway_busy"] is True
    assert health["gateway_drainable"] is True
    assert health["fatal_platform_count"] == 1
    assert platform["platform"] == "slack"
    assert platform["error_code"] == "auth_failed"
    assert "secret" not in platform["redacted_message"].lower()
    assert "Bearer" not in platform["redacted_message"]


def test_gateway_diagnostic_log_handler_redacts_and_filters(caplog):
    from agent.telemetry import emitter
    from agent.telemetry.gateway_health import GatewayDiagnosticLogHandler

    captured = []

    class DummyEmitter:
        def emit(self, event):
            captured.append(event.to_dict())

    old = emitter.get_emitter
    emitter.get_emitter = lambda: DummyEmitter()  # type: ignore[assignment]
    try:
        handler = GatewayDiagnosticLogHandler(profile="default", version="v-test")
        logger = logging.getLogger("gateway.platforms.slack")
        logger.setLevel(logging.DEBUG)
        logger.addHandler(handler)
        try:
            logger.info("ignore info token sk-live-secret")
            logger.warning("Slack token sk-live-secret failed for user@example.com")
        finally:
            logger.removeHandler(handler)
    finally:
        emitter.get_emitter = old  # type: ignore[assignment]

    assert len(captured) == 1
    event = captured[0]
    assert event["event"] == "gateway_diagnostic"
    assert event["name"] == "gateway.log.warning"
    assert event["subsystem"] == "platform.slack"
    assert event["error_class"] == "auth_failed"
    assert "***" not in event["redacted_message"]
    assert "user@example.com" not in event["redacted_message"]


def test_runtime_status_transition_emits_lifecycle_and_platform_events(monkeypatch):
    from agent.telemetry import emitter
    from agent.telemetry.gateway_health import emit_runtime_status_transition

    captured = []

    class DummyEmitter:
        def emit(self, event):
            captured.append(event.to_dict())

    old = emitter.emit
    monkeypatch.setattr(emitter, "emit", lambda event: captured.append(event.to_dict()))

    previous = {"gateway_state": "starting", "platforms": {"slack": {"state": "running"}}}
    current = {
        "gateway_state": "running",
        "pid": 123,
        "active_agents": 1,
        "platforms": {
            "slack": {
                "state": "fatal",
                "error_code": "auth_failed",
                "error_message": "Bearer *** failed",
            }
        },
    }

    emit_runtime_status_transition(previous, current)

    names = [e["name"] for e in captured]
    assert "gateway.lifecycle" in names
    assert "platform.state_change" in names
    assert "platform.fatal" in names
    lifecycle = next(e for e in captured if e["name"] == "gateway.lifecycle")
    assert lifecycle["old_state"] == "starting"
    assert lifecycle["new_state"] == "running"
    platform = next(e for e in captured if e["name"] == "platform.state_change")
    assert platform["old_state"] == "running"
    assert platform["new_state"] == "fatal"
    assert platform["error_code"] == "auth_failed"
    assert "Bearer" not in platform["redacted_message"]


def test_runtime_status_transition_emits_startup_failed_and_exit():
    from agent.telemetry.gateway_health import emit_runtime_status_transition
    from agent.telemetry import emitter

    captured = []
    old = emitter.emit
    emitter.emit = lambda event: captured.append(event.to_dict())  # type: ignore[assignment]
    try:
        emit_runtime_status_transition({"gateway_state": "starting"}, {"gateway_state": "startup_failed", "exit_reason": "startup token ***"})
        emit_runtime_status_transition({"gateway_state": "running"}, {"gateway_state": "stopped", "exit_reason": "shutdown", "restart_requested": True})
    finally:
        emitter.emit = old  # type: ignore[assignment]

    names = [e["name"] for e in captured]
    assert "gateway.startup_failed" in names
    assert "gateway.exit" in names
    failed = next(e for e in captured if e["name"] == "gateway.startup_failed")
    assert "***" not in failed["redacted_message"]
    exit_event = next(e for e in captured if e["name"] == "gateway.exit")
    assert exit_event["restart_requested"] is True


def test_otlp_attrs_include_gateway_transition_fields():
    from agent.telemetry.otlp_exporter import _span_attrs

    attrs = _span_attrs({
        "event": "gateway_health",
        "name": "gateway.lifecycle",
        "old_state": "starting",
        "new_state": "running",
        "exit_reason": "restart",
        "restart_requested": True,
    })

    assert attrs["hermes.old_state"] == "starting"
    assert attrs["hermes.new_state"] == "running"
    assert attrs["hermes.exit_reason"] == "restart"
    assert attrs["hermes.restart_requested"] is True


def test_gateway_health_export_start_is_fail_open_when_otlp_missing(monkeypatch):
    from agent.telemetry import gateway_health_export
    from agent.telemetry.gateway_health_export import GatewayHealthExportRuntime

    monkeypatch.setattr(gateway_health_export, "_require_metrics_sdk", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("missing sdk")))

    runtime = gateway_health_export.start_gateway_health_export({
        "telemetry": {
            "gateway_health_export": {"enabled": True},
            "export": {"otlp": {"enabled": True, "endpoint": "http://collector:4317"}},
        }
    })

    assert isinstance(runtime, GatewayHealthExportRuntime)
    assert runtime.enabled is False
    assert runtime.reason == "otlp_unavailable"


def test_gateway_health_export_streams_only_gateway_events(monkeypatch):
    from agent.telemetry import gateway_health_export

    captured = {}

    def fake_start_streaming(config, *, event_filter=None):
        captured["filter"] = event_filter
        return object()

    monkeypatch.setattr(gateway_health_export, "_start_metric_provider", lambda *a, **k: None)
    monkeypatch.setattr(gateway_health_export, "_require_metrics_sdk", lambda *a, **k: {})
    monkeypatch.setattr(gateway_health_export, "_attach_log_handler", lambda *a, **k: None)
    monkeypatch.setattr(gateway_health_export, "_emit_snapshot_events", lambda *a, **k: None)
    monkeypatch.setattr(gateway_health_export, "_start_snapshot_thread", lambda *a, **k: None)
    from agent.telemetry import otlp_exporter
    monkeypatch.setattr(otlp_exporter, "start_streaming", fake_start_streaming)

    runtime = gateway_health_export.start_gateway_health_export({
        "telemetry": {
            "gateway_health_export": {"enabled": True, "metrics_enabled": False},
            "export": {"otlp": {"enabled": True, "endpoint": "http://collector:4318/v1/traces"}},
        }
    })

    assert runtime.enabled is True
    event_filter = captured["filter"]
    assert event_filter({"event": "gateway_health"}) is True
    assert event_filter({"event": "gateway_diagnostic"}) is False
    assert event_filter({"event": "run"}) is False
    assert event_filter({"event": "model_call"}) is False
    assert event_filter({"event": "tool_call"}) is False


def test_gateway_health_export_metric_failure_does_not_start_streamer(monkeypatch):
    from agent.telemetry import gateway_health_export, otlp_exporter

    started = []
    monkeypatch.setattr(gateway_health_export, "_require_metrics_sdk", lambda *a, **k: {})
    monkeypatch.setattr(gateway_health_export, "_start_metric_provider", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")))
    monkeypatch.setattr(otlp_exporter, "start_streaming", lambda *a, **k: started.append(True))

    runtime = gateway_health_export.start_gateway_health_export({
        "telemetry": {
            "gateway_health_export": {"enabled": True},
            "export": {"otlp": {"enabled": True, "endpoint": "http://collector:4318/v1/traces"}},
        }
    })

    assert runtime.enabled is False
    assert runtime.reason == "metrics_start_failed"
    assert started == []


def test_otlp_streamer_shutdown_unsubscribes(monkeypatch):
    from agent.telemetry import emitter
    from agent.telemetry.otlp_exporter import OTLPStreamer

    class Dummy:
        def force_flush(self):
            pass
        def shutdown(self):
            pass

    e = emitter.get_emitter()
    streamer = OTLPStreamer.__new__(OTLPStreamer)
    streamer._processor = Dummy()
    streamer._provider = Dummy()
    streamer._event_filter = None
    streamer.exported = 0
    e.subscribe(streamer)
    assert streamer in e._subscribers

    streamer.shutdown()

    assert streamer not in e._subscribers


def test_gateway_diagnostic_log_handler_never_raises_on_malformed_record():
    from agent.telemetry.gateway_health import GatewayDiagnosticLogHandler

    handler = GatewayDiagnosticLogHandler(profile="default", version="v-test")
    record = logging.LogRecord(
        "gateway.platforms.slack",
        logging.WARNING,
        __file__,
        1,
        "broken %s %s",
        ("one",),
        None,
    )

    handler.emit(record)
