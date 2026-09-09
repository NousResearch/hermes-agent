"""Gateway Health & Diagnostics OTLP export runtime.

Emits operator-owned gateway service-health metrics plus narrow redacted diagnostic events.
Deliberately in-process and fail-open so it works under systemd, launchd, s6, containers,
tmux, nohup, or a plain shell without a sidecar/watchdog dependency.
"""

from __future__ import annotations

import importlib
import logging
import os
import threading
from contextlib import suppress
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional

from agent.monitoring import emitter, otlp_exporter
from agent.monitoring.gateway_health import (
    GatewayDiagnosticLogHandler,
    GatewayMetric,
    _safe_profile as _profile,
    _safe_version as _version,
    build_gateway_health_snapshot,
    source_logger_for_export,
)
from agent.monitoring.otlp_exporter import (
    EmitterStreamer,
    _allowlisted_attrs,
    _install_id,
    _monitoring_section,
    _otlp_config,
    _resolve_headers,
    _runtime_resource_attributes,
    _signal_endpoint,
)
from agent.monitoring.redaction import redact_bounded

logger = logging.getLogger(__name__)

_METRICS_SDK = (
    "OTLPLogExporter", "OTLPMetricExporter", "Observation", "LogRecord", "LoggerProvider",
    "INVALID_SPAN_ID", "INVALID_TRACE_ID", "TraceFlags", "SeverityNumber",
    "BatchLogRecordProcessor", "MeterProvider", "PeriodicExportingMetricReader", "Resource",
)
# Every gauge the runtime snapshot can emit MUST be listed here or it is silently dropped.
_OBSERVABLE_METRIC_NAMES = (
    "hermes.gateway.up", "hermes.gateway.state", "hermes.gateway.active_agents", "hermes.gateway.busy",
    "hermes.gateway.drainable", "hermes.gateway.restart_requested", "hermes.gateway.background_work",
    "hermes.gateway.background_delegations", "hermes.platform.up", "hermes.platform.degraded",
    "hermes.cron.scheduler.heartbeat_age_seconds", "hermes.cron.scheduler.last_success_age_seconds",
    "hermes.cron.scheduler.catch_up_occurrences", "hermes.cron.jobs.enabled", "hermes.cron.jobs.running",
    "hermes.cron.jobs.overdue",
)


def _diagnostic_log_attributes(event: Dict[str, Any]) -> Dict[str, Any]:
    # Same allowlist as the span mapping: profile/install_id/ts never egress as attributes.
    return _allowlisted_attrs(event, otlp_exporter._KEEP_BY_KIND["gateway_diagnostic"])


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
            with suppress(Exception):
                logging.getLogger().removeHandler(self.log_handler)
        # Producers are stopped; drain queued/in-flight events BEFORE detaching subscribers
        # so the terminal lifecycle event cannot race exporter shutdown. Bounded, fail-open.
        subscribers = [item for item in (self.streamer, self.log_streamer) if item is not None]
        with suppress(Exception):
            bus = emitter.get_emitter()
            bus.flush(timeout=1.0)
            for sub in subscribers:
                bus.unsubscribe(sub)
        # Network flush/close runs under one bounded daemon-thread deadline so it can
        # never delay gateway teardown indefinitely.
        closeables = [item for item in (self.streamer, self.log_streamer, self.metric_provider) if item is not None]

        def _close() -> None:
            for item in closeables:
                with suppress(Exception):
                    item.shutdown()

        if closeables:
            worker = threading.Thread(target=_close, name="hermes-gateway-health-export-shutdown", daemon=True)
            worker.start()
            worker.join(timeout=2.0)
        self.streamer = self.log_streamer = self.metric_provider = self.thread = self.stop_event = None


def _gateway_health_config(config: Dict[str, Any]) -> Dict[str, Any]:
    return _monitoring_section(config, "gateway_health_export")


def _enabled(config: Dict[str, Any]) -> bool:
    return bool(_gateway_health_config(config).get("enabled") and otlp_exporter.is_enabled(config))


def _exporter_kwargs(config: Dict[str, Any], signal: str) -> Dict[str, Any]:
    otlp = _otlp_config(config)
    return {"endpoint": _signal_endpoint(str(otlp.get("endpoint")), signal), "headers": _resolve_headers(otlp.get("headers_env")) or None}


# Ordered detection: systemd > s6 > container > launchd > manual (first match wins).
_SUPERVISION_DETECTORS: tuple[tuple[str, Callable[[], Any]], ...] = (
    ("systemd", lambda: os.environ.get("INVOCATION_ID")),
    ("s6", lambda: os.environ.get("S6_CMD_ARG0") or os.environ.get("S6_VERSION")),
    ("container", lambda: os.environ.get("container") or os.path.exists("/.dockerenv")),
    ("launchd", lambda: os.environ.get("LAUNCHD_SOCKET")),
)


def _supervision_mode() -> str:
    return next((mode for mode, detect in _SUPERVISION_DETECTORS if detect()), "manual")


def _read_gateway_snapshot(config: Dict[str, Any]):
    try:
        from gateway.status import read_runtime_status
        runtime = read_runtime_status() or {}
    except Exception:
        runtime = {}
    return build_gateway_health_snapshot(
        runtime, gateway_running=True, profile=_profile(), install_id=_install_id(config), version=_version(),
        supervision_mode=_supervision_mode(),
    )


def _read_cron_snapshot():
    from agent.monitoring.cron_health import build_cron_health_snapshot
    return build_cron_health_snapshot()


def _count(failure_msg: str, module: str, read: Callable[[Any], Any]) -> int:
    """Best-effort non-negative count read from a lazily imported module; 0 when it can't be imported/read."""
    try:
        return max(0, int(read(importlib.import_module(module))))
    except Exception:
        logger.debug(failure_msg, exc_info=True)
        return 0


def _read_background_work_count() -> int:
    """Live background/subagent work that ``active_agents`` (foreground turns + in-flight cron + API
    runs) deliberately does NOT include: backgrounded ``delegate_task`` subagents,
    ``terminal(background=true)`` processes, kanban workers.  TASK-granular: a fan-out batch of N
    contributes N (real concurrent load), unlike the pool's one-slot-per-batch accounting."""
    return _count("background-work async-delegation count failed", "tools.async_delegation", lambda m: m.active_task_count()) + _count(
        "background-work process-registry count failed", "tools.process_registry", lambda m: m.process_registry.count_running()
    )


def _read_background_delegations_count() -> int:
    """Live async delegation UNITS (pool slots): a batch counts ONE regardless of fan-out width, so
    operators see slot pressure (vs ``max_concurrent_children``) alongside ``background_work``."""
    return _count("background-delegations count failed", "tools.async_delegation", lambda m: m.active_count())


def _read_runtime_snapshot(config: Dict[str, Any]):
    gateway_snapshot = _read_gateway_snapshot(config)
    # Background/subagent work is appended to the gateway snapshot so it rides the same base
    # resource attributes (service.instance.id etc.).
    try:
        base = dict(gateway_snapshot.metrics[0].attributes) if gateway_snapshot.metrics else {}
        for name, read in (
            ("hermes.gateway.background_work", _read_background_work_count),
            ("hermes.gateway.background_delegations", _read_background_delegations_count),
        ):
            gateway_snapshot.metrics.append(GatewayMetric(name=name, value=read(), attributes=base))
    except Exception as exc:
        logger.warning("background-work snapshot unavailable; metric not exported (error_type=%s)", type(exc).__name__)
        logger.debug("background-work snapshot traceback", exc_info=True)
    try:
        gateway_snapshot.metrics.extend(_read_cron_snapshot().metrics)
    except Exception as exc:
        # Cron telemetry silently dropping out is a release-relevant regression: WARN with only
        # the exception *type* (the message could carry paths); exc_info stays on DEBUG.
        logger.warning("cron health snapshot unavailable; cron telemetry not exported (error_type=%s)", type(exc).__name__)
        logger.debug("cron health snapshot traceback", exc_info=True)
    return gateway_snapshot


def _emit_snapshot_events(config: Dict[str, Any]) -> None:
    if not _gateway_health_config(config).get("diagnostic_events_enabled", True):
        return
    try:
        for event in _read_runtime_snapshot(config).events:
            emitter.emit(event)
    except Exception:
        logger.debug("gateway health snapshot emit failed", exc_info=True)


def _start_metric_provider(config: Dict[str, Any], sdk: Dict[str, Any]) -> Any:
    exporter = sdk["OTLPMetricExporter"](**_exporter_kwargs(config, "metrics"))
    interval_ms = max(5, int(_gateway_health_config(config).get("export_interval_seconds", 60))) * 1000
    reader = sdk["PeriodicExportingMetricReader"](exporter, export_interval_millis=interval_ms)
    resource = sdk["Resource"].create(_runtime_resource_attributes(config, telemetry_scope="gateway_health"))
    provider = sdk["MeterProvider"](metric_readers=[reader], resource=resource)
    meter = provider.get_meter("hermes.gateway.health")
    Observation = sdk["Observation"]

    def callback(name: str):
        def _cb(_options=None):
            try:
                return [Observation(m.value, m.attributes) for m in _read_runtime_snapshot(config).metrics if m.name == name]
            except Exception:
                logger.debug("gateway metric callback failed", exc_info=True)
                return []

        return _cb

    for metric_name in _OBSERVABLE_METRIC_NAMES:
        meter.create_observable_gauge(metric_name, callbacks=[callback(metric_name)])
    return provider


_SEVERITY_NAMES = {"critical": "FATAL", "fatal": "FATAL", "error": "ERROR", "info": "INFO", "information": "INFO", "debug": "DEBUG"}


def _severity_number(sdk: Dict[str, Any], severity: Any) -> Any:
    return getattr(sdk["SeverityNumber"], _SEVERITY_NAMES.get(str(severity or "warning").lower(), "WARN"))


class GatewayDiagnosticLogStreamer(EmitterStreamer):
    """Emitter subscriber that sends gateway diagnostic events as OTLP logs."""

    def __init__(self, config: Dict[str, Any], sdk: Dict[str, Any]):
        resource = sdk["Resource"].create(_runtime_resource_attributes(config, telemetry_scope="gateway_diagnostics"))
        self._provider = sdk["LoggerProvider"](resource=resource)
        self._processor = sdk["BatchLogRecordProcessor"](sdk["OTLPLogExporter"](**_exporter_kwargs(config, "logs")))
        self._provider.add_log_record_processor(self._processor)
        self._logger = self._provider.get_logger("hermes.gateway.diagnostics")
        self._sdk = sdk
        self.exported = 0

    def __call__(self, batch: list[Dict[str, Any]]) -> None:
        sdk = self._sdk
        for ev in batch:
            if ev.get("event") != "gateway_diagnostic":
                continue
            # The source-controlled Python logger becomes the OTel instrumentation scope:
            # precise code attribution without maintaining a subsystem enum. Rendered
            # messages stay out (they may carry IDs, names, paths, configured strings).
            source_logger = source_logger_for_export(ev.get("source_logger"))
            otel_logger = self._provider.get_logger(source_logger) if source_logger is not None else self._logger
            otel_logger.emit(sdk["LogRecord"](
                timestamp=ev.get("ts_ns"), trace_id=sdk["INVALID_TRACE_ID"], span_id=sdk["INVALID_SPAN_ID"],
                trace_flags=sdk["TraceFlags"].DEFAULT, severity_text=str(ev.get("severity") or "warning").upper(),
                severity_number=_severity_number(sdk, ev.get("severity")), body=redact_bounded("gateway diagnostic"),
                attributes=_diagnostic_log_attributes(ev),
            ))
            self.exported += 1


def _gateway_health_event(ev: Dict[str, Any]) -> bool:
    return ev.get("event") in {"gateway_health", "cron_execution"}


def _fail(runtime: GatewayHealthExportRuntime, log: Callable[..., None], msg: str, reason: str) -> GatewayHealthExportRuntime:
    log(msg, exc_info=True)
    runtime.shutdown()
    return GatewayHealthExportRuntime(enabled=False, reason=reason)


def start_gateway_health_export(config: Dict[str, Any]) -> GatewayHealthExportRuntime:
    """Start P0 gateway health export if configured. Never raises."""
    if not _enabled(config):
        return GatewayHealthExportRuntime(enabled=False, reason="disabled")
    gh = _gateway_health_config(config)
    metrics_on = gh.get("metrics_enabled", True)
    diagnostics_on = gh.get("diagnostic_events_enabled", True)
    runtime = GatewayHealthExportRuntime(enabled=True, reason="enabled")
    sdk: Optional[Dict[str, Any]] = None
    if metrics_on or diagnostics_on:
        try:
            sdk = otlp_exporter._require_sdk(_METRICS_SDK, auto_install=True, prompt=False)
        except Exception:
            logger.warning("monitoring.gateway_health_export.enabled but OTLP SDK is unavailable; install 'hermes-agent[otlp]'", exc_info=True)
            return GatewayHealthExportRuntime(enabled=False, reason="otlp_unavailable")
    if metrics_on and sdk is not None:
        try:
            runtime.metric_provider = _start_metric_provider(config, sdk)
        except Exception:
            return _fail(runtime, logger.warning, "gateway health OTLP metrics failed to start", "metrics_start_failed")
    if diagnostics_on and sdk is not None:
        try:
            runtime.streamer = otlp_exporter.start_streaming(config, event_filter=_gateway_health_event)
            if runtime.streamer is None:
                raise RuntimeError("gateway health span streamer did not start")
            log_streamer = GatewayDiagnosticLogStreamer(config, sdk)
            emitter.get_emitter().subscribe(log_streamer)
            runtime.log_streamer = log_streamer
        except Exception:
            return _fail(runtime, logger.debug, "gateway diagnostic OTLP export failed to start", "diagnostics_start_failed")
    try:
        if diagnostics_on and gh.get("warning_error_events_enabled", True):
            handler = GatewayDiagnosticLogHandler(profile=_profile(), version=_version())
            root = logging.getLogger()
            if handler not in root.handlers:
                root.addHandler(handler)
            runtime.log_handler = handler
    except Exception:
        logger.debug("gateway diagnostic log handler failed to attach", exc_info=True)
    if diagnostics_on:
        try:
            _emit_snapshot_events(config)
            stop_event = runtime.stop_event = threading.Event()
            interval = max(5, int(gh.get("logs_export_interval_seconds", 5)))

            def _run() -> None:
                while not stop_event.wait(interval):
                    _emit_snapshot_events(config)

            thread = threading.Thread(target=_run, name="hermes-gateway-health-export", daemon=True)
            thread.start()
            runtime.thread = thread
        except Exception:
            logger.debug("gateway health snapshot thread failed to start", exc_info=True)
    return runtime


__all__ = ["GatewayHealthExportRuntime", "start_gateway_health_export"]


# ---- BEGIN PLUGIN-COMPAT (revert-scheduled; see COMPAT_MANIFEST.md) ----
# Names external plugins imported from this module before the Sep 2026 decomposition.
# Internal code MUST NOT use these (scripts/check_compat_pointers.py fails CI if it does).
# The whole block is removed by reverting the commit that added it.
import re  # noqa: F401,E402
# ---- END PLUGIN-COMPAT ----
