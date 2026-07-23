"""Gateway Health & Diagnostics OTLP export runtime.

This exporter emits operator-owned gateway service-health metrics plus
narrow redacted diagnostic events. It is deliberately in-process and fail-open so
it works under systemd, launchd, s6, containers, tmux, nohup, or a simple shell
without a sidecar/watchdog dependency.
"""

from __future__ import annotations

import logging
import os
import re
import threading
from dataclasses import dataclass
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

_DEFAULT_DIAGNOSTIC_SCOPE = "hermes.gateway.diagnostics"

_RESOURCE_ATTRIBUTE_KEYS = frozenset({
    "service.name",
    "service.namespace",
    "service.version",
    "service.instance.id",
    "deployment.environment.name",
    "cloud.provider",
    "cloud.platform",
    "cloud.region",
    "telemetry.scope",
})
_DIAGNOSTIC_ATTRIBUTE_KEYS = frozenset({
    "name",
    "subsystem",
    "error_class",
    "error_code",
    "platform",
    "old_state",
    "new_state",
    "version",
    "severity",
})
_SAFE_RESOURCE_VALUE = re.compile(r"^[A-Za-z0-9._:/-]{1,128}$")


def _redact_string(raw: Any, *, limit: int = 500) -> str:
    try:
        from agent.monitoring.redaction import redact_for_export
        return (redact_for_export(str(raw or "")) or "[redacted]")[:limit]
    except Exception:
        return "[redaction-unavailable]"


def _safe_resource_attributes(raw: Any) -> Dict[str, str]:
    """Allowlist bounded resource labels and reject values changed by redaction."""
    attrs: Dict[str, str] = {}
    if not isinstance(raw, dict):
        return attrs
    for key, value in raw.items():
        key = str(key)
        if key not in _RESOURCE_ATTRIBUTE_KEYS or value is None:
            continue
        if key == "service.instance.id":
            from agent.monitoring.gateway_health import _safe_instance_id
            attrs[key] = _safe_instance_id(value)
            continue
        text = str(value)
        if not _SAFE_RESOURCE_VALUE.fullmatch(text):
            continue
        if _redact_string(text, limit=128) != text:
            continue
        attrs[key] = text
    return attrs


def _runtime_resource_attributes(
    config: Dict[str, Any], *, telemetry_scope: str
) -> Dict[str, str]:
    """Build the safe OTLP resource shared by metrics and diagnostic logs."""
    gh = _gateway_health_config(config)
    attrs = _safe_resource_attributes(gh.get("resource_attributes"))
    from agent.monitoring.gateway_health import _safe_instance_id

    attrs["service.name"] = "hermes-gateway"
    attrs["service.instance.id"] = _safe_instance_id(_install_id(config))
    attrs["telemetry.scope"] = telemetry_scope
    return attrs


def _diagnostic_log_attributes(event: Dict[str, Any]) -> Dict[str, Any]:
    attrs: Dict[str, Any] = {}
    for key in _DIAGNOSTIC_ATTRIBUTE_KEYS:
        value = event.get(key)
        if value is None:
            continue
        attrs[f"hermes.{key}"] = _redact_string(value) if isinstance(value, str) else value
    return attrs


@dataclass(slots=True)
class GatewayHealthExportRuntime:
    enabled: bool
    reason: str = "disabled"
    streamer: Any = None
    metric_provider: Any = None
    log_handler: Any = None
    log_streamer: Any = None
    thread: Optional[threading.Thread] = None
    stop_event: Optional[threading.Event] = None

    def shutdown(self) -> None:
        if self.stop_event is not None:
            self.stop_event.set()
        if self.thread is not None:
            self.thread.join(timeout=0.25)
        if self.log_handler is not None:
            try:
                logging.getLogger().removeHandler(self.log_handler)
            except Exception:
                pass

        # All producers above are now stopped. Drain queued and in-flight
        # events before detaching subscribers so the terminal lifecycle event
        # cannot race exporter shutdown. The barrier is bounded and fail-open.
        try:
            from agent.monitoring.emitter import get_emitter
            emitter = get_emitter()
            emitter.flush(timeout=1.0)
            if self.streamer is not None:
                emitter.unsubscribe(self.streamer)
            if self.log_streamer is not None:
                emitter.unsubscribe(self.log_streamer)
        except Exception:
            pass

        # Network flush/close runs under one bounded daemon-thread deadline and
        # can never delay gateway teardown indefinitely.
        closeables = [
            item for item in (self.streamer, self.log_streamer, self.metric_provider)
            if item is not None
        ]

        def _close() -> None:
            for item in closeables:
                try:
                    item.shutdown()
                except Exception:
                    pass

        if closeables:
            worker = threading.Thread(
                target=_close,
                name="hermes-gateway-health-export-shutdown",
                daemon=True,
            )
            worker.start()
            worker.join(timeout=2.0)

        self.streamer = None
        self.log_streamer = None
        self.metric_provider = None
        self.thread = None
        self.stop_event = None


def _gateway_health_config(config: Dict[str, Any]) -> Dict[str, Any]:
    mon = (config or {}).get("monitoring") or {}
    return mon.get("gateway_health_export") or {}


def _otlp_config(config: Dict[str, Any]) -> Dict[str, Any]:
    mon = (config or {}).get("monitoring") or {}
    export = mon.get("export") or {}
    return export.get("otlp") or {}


def _enabled(config: Dict[str, Any]) -> bool:
    gh = _gateway_health_config(config)
    otlp = _otlp_config(config)
    return bool(gh.get("enabled") and otlp.get("enabled") and otlp.get("endpoint"))


def _require_metrics_sdk(*, auto_install: bool = True, prompt: bool = False) -> Dict[str, Any]:
    if auto_install:
        try:
            from tools.lazy_deps import ensure as _lazy_ensure
            _lazy_ensure("export.otlp", prompt=prompt)
        except Exception:
            pass
    try:
        from opentelemetry.exporter.otlp.proto.http._log_exporter import OTLPLogExporter
        from opentelemetry.exporter.otlp.proto.http.metric_exporter import OTLPMetricExporter
        from opentelemetry.metrics import Observation
        from opentelemetry.trace import INVALID_SPAN_ID, INVALID_TRACE_ID, TraceFlags
        from opentelemetry._logs import LogRecord
        from opentelemetry._logs.severity import SeverityNumber
        from opentelemetry.sdk._logs import LoggerProvider
        from opentelemetry.sdk._logs.export import BatchLogRecordProcessor
        from opentelemetry.sdk.metrics import MeterProvider
        from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
        from opentelemetry.sdk.resources import Resource
        return {
            "OTLPLogExporter": OTLPLogExporter,
            "OTLPMetricExporter": OTLPMetricExporter,
            "Observation": Observation,
            "LogRecord": LogRecord,
            "LoggerProvider": LoggerProvider,
            "INVALID_SPAN_ID": INVALID_SPAN_ID,
            "INVALID_TRACE_ID": INVALID_TRACE_ID,
            "TraceFlags": TraceFlags,
            "SeverityNumber": SeverityNumber,
            "BatchLogRecordProcessor": BatchLogRecordProcessor,
            "MeterProvider": MeterProvider,
            "PeriodicExportingMetricReader": PeriodicExportingMetricReader,
            "Resource": Resource,
        }
    except Exception as exc:
        raise RuntimeError(f"OTLP metrics SDK unavailable: {exc}") from exc


def _resolve_headers(headers_env: Optional[Dict[str, str]]) -> Dict[str, str]:
    resolved: Dict[str, str] = {}
    for header_name, env_name in (headers_env or {}).items():
        val = os.environ.get(str(env_name))
        if val:
            resolved[str(header_name)] = val
    return resolved


def _metric_endpoint(endpoint: str) -> str:
    if endpoint.endswith("/v1/traces"):
        return endpoint[: -len("/v1/traces")] + "/v1/metrics"
    return endpoint


def _logs_endpoint(endpoint: str) -> str:
    if endpoint.endswith("/v1/traces"):
        return endpoint[: -len("/v1/traces")] + "/v1/logs"
    if endpoint.endswith("/v1/metrics"):
        return endpoint[: -len("/v1/metrics")] + "/v1/logs"
    return endpoint


def _version() -> str:
    try:
        from hermes_cli import __version__
        return str(__version__)
    except Exception:
        return "unknown"


def _profile() -> str:
    try:
        from hermes_cli.profiles import get_active_profile_name
        return str(get_active_profile_name() or "default")
    except Exception:
        return "default"


def _install_id(config: Dict[str, Any]) -> str:
    try:
        from agent.monitoring.policy import ensure_install_id
        return str(ensure_install_id(config))
    except Exception:
        return "unknown"


def _supervision_mode() -> str:
    if os.environ.get("INVOCATION_ID"):
        return "systemd"
    if os.environ.get("S6_CMD_ARG0") or os.environ.get("S6_VERSION"):
        return "s6"
    if os.environ.get("container") or os.path.exists("/.dockerenv"):
        return "container"
    if os.environ.get("LAUNCHD_SOCKET"):
        return "launchd"
    return "manual"


def _read_runtime_snapshot(config: Dict[str, Any]):
    from agent.monitoring.gateway_health import build_gateway_health_snapshot
    try:
        from gateway.status import read_runtime_status
        runtime = read_runtime_status() or {}
    except Exception:
        runtime = {}
    return build_gateway_health_snapshot(
        runtime,
        gateway_running=True,
        profile=_profile(),
        install_id=_install_id(config),
        version=_version(),
        supervision_mode=_supervision_mode(),
    )


def _emit_snapshot_events(config: Dict[str, Any]) -> None:
    gh = _gateway_health_config(config)
    if not gh.get("diagnostic_events_enabled", True):
        return
    try:
        from agent.monitoring import emitter
        snapshot = _read_runtime_snapshot(config)
        for event in snapshot.events:
            emitter.emit(event)
    except Exception:
        logger.debug("gateway health snapshot emit failed", exc_info=True)


def _start_metric_provider(config: Dict[str, Any], sdk: Dict[str, Any]) -> Any:
    gh = _gateway_health_config(config)
    if not gh.get("metrics_enabled", True):
        return None
    otlp = _otlp_config(config)
    endpoint = _metric_endpoint(str(otlp.get("endpoint")))
    headers = _resolve_headers(otlp.get("headers_env"))
    exporter = sdk["OTLPMetricExporter"](endpoint=endpoint, headers=headers or None)
    interval_ms = max(5, int(gh.get("export_interval_seconds", 60))) * 1000
    reader = sdk["PeriodicExportingMetricReader"](exporter, export_interval_millis=interval_ms)
    resource_attrs = _runtime_resource_attributes(
        config, telemetry_scope="gateway_health"
    )
    provider = sdk["MeterProvider"](
        metric_readers=[reader],
        resource=sdk["Resource"].create(resource_attrs),
    )
    meter = provider.get_meter("hermes.gateway.health")
    Observation = sdk["Observation"]

    metric_names = [
        "hermes.gateway.up",
        "hermes.gateway.state",
        "hermes.gateway.active_agents",
        "hermes.gateway.busy",
        "hermes.gateway.drainable",
        "hermes.gateway.restart_requested",
        "hermes.platform.up",
        "hermes.platform.degraded",
    ]

    def callback(name: str):
        def _cb(_options=None):
            try:
                snapshot = _read_runtime_snapshot(config)
                return [Observation(m.value, m.attributes) for m in snapshot.metrics if m.name == name]
            except Exception:
                logger.debug("gateway metric callback failed", exc_info=True)
                return []
        return _cb

    for metric_name in metric_names:
        meter.create_observable_gauge(metric_name, callbacks=[callback(metric_name)])
    return provider


def _severity_number(sdk: Dict[str, Any], severity: Any) -> Any:
    SeverityNumber = sdk["SeverityNumber"]
    sev = str(severity or "warning").lower()
    if sev in {"critical", "fatal"}:
        return SeverityNumber.FATAL
    if sev == "error":
        return SeverityNumber.ERROR
    if sev in {"info", "information"}:
        return SeverityNumber.INFO
    if sev == "debug":
        return SeverityNumber.DEBUG
    return SeverityNumber.WARN


class GatewayDiagnosticLogStreamer:
    """Emitter subscriber that sends gateway diagnostic events as OTLP logs."""

    def __init__(self, config: Dict[str, Any], sdk: Dict[str, Any]):
        otlp = _otlp_config(config)
        headers = _resolve_headers(otlp.get("headers_env"))
        endpoint = _logs_endpoint(str(otlp.get("endpoint")))
        resource_attrs = _runtime_resource_attributes(
            config, telemetry_scope="gateway_diagnostics"
        )
        self._provider = sdk["LoggerProvider"](resource=sdk["Resource"].create(resource_attrs))
        self._processor = sdk["BatchLogRecordProcessor"](
            sdk["OTLPLogExporter"](endpoint=endpoint, headers=headers or None)
        )
        self._provider.add_log_record_processor(self._processor)
        self._logger = self._provider.get_logger(_DEFAULT_DIAGNOSTIC_SCOPE)
        self._LogRecord = sdk["LogRecord"]
        self._sdk = sdk
        self.exported = 0

    def __call__(self, batch: list[Dict[str, Any]]) -> None:
        from agent.monitoring.gateway_health import source_logger_for_export

        for ev in batch:
            if ev.get("event") != "gateway_diagnostic":
                continue
            attrs = _diagnostic_log_attributes(ev)
            # Preserve the source-controlled Python logger as the OTel
            # instrumentation scope. This adds precise code attribution without
            # turning a fluid module layout into a maintained subsystem enum.
            # Rendered messages stay out because they may contain arbitrary IDs,
            # names, paths, or configured strings. A future, separately gated
            # ``diagnostic_detail: redacted_message`` mode may add best-effort
            # free text when an observability plane defines that privacy policy.
            source_logger = source_logger_for_export(ev.get("source_logger"))
            otel_logger = (
                self._provider.get_logger(source_logger)
                if source_logger is not None
                else self._logger
            )
            body = "gateway diagnostic"
            record = self._LogRecord(
                timestamp=ev.get("ts_ns"),
                trace_id=self._sdk["INVALID_TRACE_ID"],
                span_id=self._sdk["INVALID_SPAN_ID"],
                trace_flags=self._sdk["TraceFlags"].DEFAULT,
                severity_text=str(ev.get("severity") or "warning").upper(),
                severity_number=_severity_number(self._sdk, ev.get("severity")),
                body=_redact_string(body),
                attributes=attrs,
            )
            otel_logger.emit(record)
            self.exported += 1

    def shutdown(self) -> None:
        try:
            from agent.monitoring.emitter import get_emitter
            get_emitter().unsubscribe(self)
        except Exception:
            pass
        try:
            self._processor.force_flush()
            self._provider.shutdown()
        except Exception:
            pass


def _start_diagnostic_log_streamer(config: Dict[str, Any], sdk: Dict[str, Any]) -> GatewayDiagnosticLogStreamer:
    from agent.monitoring.emitter import get_emitter
    streamer = GatewayDiagnosticLogStreamer(config, sdk)
    get_emitter().subscribe(streamer)
    return streamer


def _start_snapshot_thread(config: Dict[str, Any], stop_event: threading.Event) -> threading.Thread:
    interval = max(5, int(_gateway_health_config(config).get("logs_export_interval_seconds", 5)))

    def _run() -> None:
        while not stop_event.wait(interval):
            _emit_snapshot_events(config)

    thread = threading.Thread(target=_run, name="hermes-gateway-health-export", daemon=True)
    thread.start()
    return thread


def _attach_log_handler(config: Dict[str, Any]) -> Any:
    gh = _gateway_health_config(config)
    if not gh.get("diagnostic_events_enabled", True) or not gh.get("warning_error_events_enabled", True):
        return None
    from agent.monitoring.gateway_health import GatewayDiagnosticLogHandler
    handler = GatewayDiagnosticLogHandler(profile=_profile(), version=_version())
    root = logging.getLogger()
    if handler not in root.handlers:
        root.addHandler(handler)
    return handler


def _gateway_health_event(ev: Dict[str, Any]) -> bool:
    return ev.get("event") == "gateway_health"


def start_gateway_health_export(config: Dict[str, Any]) -> GatewayHealthExportRuntime:
    """Start P0 gateway health export if configured. Never raises."""
    if not _enabled(config):
        return GatewayHealthExportRuntime(enabled=False, reason="disabled")
    gh = _gateway_health_config(config)
    runtime = GatewayHealthExportRuntime(enabled=True, reason="enabled")
    sdk: Optional[Dict[str, Any]] = None

    if gh.get("metrics_enabled", True) or gh.get("diagnostic_events_enabled", True):
        try:
            sdk = _require_metrics_sdk(prompt=False)
        except Exception:
            logger.warning(
                "monitoring.gateway_health_export.enabled but OTLP SDK is unavailable; "
                "install 'hermes-agent[otlp]'",
                exc_info=True,
            )
            return GatewayHealthExportRuntime(enabled=False, reason="otlp_unavailable")

    if gh.get("metrics_enabled", True) and sdk is not None:
        try:
            runtime.metric_provider = _start_metric_provider(config, sdk)
        except Exception:
            logger.warning("gateway health OTLP metrics failed to start", exc_info=True)
            runtime.shutdown()
            return GatewayHealthExportRuntime(enabled=False, reason="metrics_start_failed")

    if gh.get("diagnostic_events_enabled", True) and sdk is not None:
        try:
            from agent.monitoring import otlp_exporter
            runtime.streamer = otlp_exporter.start_streaming(config, event_filter=_gateway_health_event)
            if runtime.streamer is None:
                raise RuntimeError("gateway health span streamer did not start")
            runtime.log_streamer = _start_diagnostic_log_streamer(config, sdk)
        except Exception:
            logger.debug("gateway diagnostic OTLP export failed to start", exc_info=True)
            runtime.shutdown()
            return GatewayHealthExportRuntime(enabled=False, reason="diagnostics_start_failed")

    try:
        runtime.log_handler = _attach_log_handler(config)
    except Exception:
        logger.debug("gateway diagnostic log handler failed to attach", exc_info=True)
    if gh.get("diagnostic_events_enabled", True):
        try:
            _emit_snapshot_events(config)
            runtime.stop_event = threading.Event()
            runtime.thread = _start_snapshot_thread(config, runtime.stop_event)
        except Exception:
            logger.debug("gateway health snapshot thread failed to start", exc_info=True)
    return runtime


__all__ = [
    "GatewayHealthExportRuntime",
    "start_gateway_health_export",
]
