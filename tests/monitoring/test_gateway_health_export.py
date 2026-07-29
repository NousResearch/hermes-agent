from __future__ import annotations

import logging

import pytest


def test_gateway_diagnostic_event_preserves_positional_error_class():
    from agent.monitoring.events import GatewayDiagnosticEvent

    event = GatewayDiagnosticEvent("gateway.log.warning", "gateway", "auth_failed")

    assert event.error_class == "auth_failed"
    assert event.source_logger is None




def test_gateway_health_snapshot_maps_runtime_status_to_low_cardinality_metrics():
    from agent.monitoring.gateway_health import build_gateway_health_snapshot

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
        "service.instance.id": active.attributes["service.instance.id"],
        "service.version": "2026.7.test",
        "hermes.supervision_mode": "manual",
    }
    assert active.attributes["service.instance.id"].startswith("sha256:")
    assert "install-1" not in active.attributes["service.instance.id"]

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






def test_gateway_diagnostic_log_handler_never_carries_rendered_message(caplog):
    from agent.monitoring import emitter
    from agent.monitoring.gateway_health import GatewayDiagnosticLogHandler

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
            logger.warning(
                "Unauthorized user: acct_7f3a (Alice Smith) on slack; "
                "token «redacted:sk-…»"
            )
        finally:
            logger.removeHandler(handler)
    finally:
        emitter.get_emitter = old  # type: ignore[assignment]

    assert len(captured) == 1
    event = captured[0]
    assert event["event"] == "gateway_diagnostic"
    assert event["name"] == "gateway.log.warning"
    assert event["subsystem"] == "platform.slack"
    assert event["source_logger"] == "gateway.platforms.slack"
    assert event["error_class"] == "auth_failed"
    assert "redacted_message" not in event
    assert "acct_7f3a" not in str(event)
    assert "Alice Smith" not in str(event)












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
    from agent.monitoring.gateway_health_export import _safe_resource_attributes

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


def test_instance_id_hash_is_stable_and_distinguishes_instances():
    from agent.monitoring.gateway_health import _safe_instance_id

    first = _safe_instance_id("install-1")
    repeat = _safe_instance_id("install-1")
    second = _safe_instance_id("install-2")

    assert first == repeat
    assert first != second
    assert first.startswith("sha256:")
    assert "install-1" not in first




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




def test_gateway_health_export_start_is_fail_open_when_otlp_missing(monkeypatch):
    from agent.monitoring import gateway_health_export
    from agent.monitoring.gateway_health_export import GatewayHealthExportRuntime

    monkeypatch.setattr(gateway_health_export, "_require_metrics_sdk", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("missing sdk")))

    runtime = gateway_health_export.start_gateway_health_export({
        "monitoring": {
            "gateway_health_export": {"enabled": True},
            "export": {"otlp": {"enabled": True, "endpoint": "http://collector:4317"}},
        }
    })

    assert isinstance(runtime, GatewayHealthExportRuntime)
    assert runtime.enabled is False
    assert runtime.reason == "otlp_unavailable"






def test_gateway_health_export_metric_failure_does_not_start_streamer(monkeypatch):
    from agent.monitoring import gateway_health_export, otlp_exporter

    started = []
    monkeypatch.setattr(gateway_health_export, "_require_metrics_sdk", lambda *a, **k: {})
    monkeypatch.setattr(gateway_health_export, "_start_metric_provider", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")))
    monkeypatch.setattr(otlp_exporter, "start_streaming", lambda *a, **k: started.append(True))

    runtime = gateway_health_export.start_gateway_health_export({
        "monitoring": {
            "gateway_health_export": {"enabled": True},
            "export": {"otlp": {"enabled": True, "endpoint": "http://collector:4318/v1/traces"}},
        }
    })

    assert runtime.enabled is False
    assert runtime.reason == "metrics_start_failed"
    assert started == []








def test_gateway_diagnostic_log_handler_never_raises_on_malformed_record():
    from agent.monitoring.gateway_health import GatewayDiagnosticLogHandler

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


