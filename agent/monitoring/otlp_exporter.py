"""Export monitoring events to an OpenTelemetry Collector over OTLP/HTTP.

Maps gateway monitoring events to OTel spans for the operator-configured ``monitoring.export.otlp``
endpoint (no default destination ships) and hosts the OTLP plumbing shared with
``gateway_health_export``.  The OTel SDK is an optional extra (``hermes-agent[otlp]``) imported
lazily; ``headers_env`` values are read at export time and never logged or stored.  The streaming
subscriber runs fail-isolated on the emitter thread; ``event_filter`` keeps other planes off it.
"""

from __future__ import annotations

import importlib
import logging
import os
import re
from contextlib import suppress
from typing import Any, Callable, Dict, Iterable, List, Optional

from agent.monitoring.gateway_health import _safe_instance_id
from agent.monitoring.redaction import redact_bounded

logger = logging.getLogger(__name__)


class OTLPUnavailable(RuntimeError):
    """Raised when the optional OpenTelemetry SDK isn't installed."""


# ── SDK loading ──────────────────────────────────────────────────────────────
_SDK_MODULES: Dict[str, tuple[str, ...]] = {
    "opentelemetry.sdk.trace": ("TracerProvider",),
    "opentelemetry.sdk.trace.export": ("BatchSpanProcessor",),
    "opentelemetry.sdk.resources": ("Resource",),
    "opentelemetry.exporter.otlp.proto.http.trace_exporter": ("OTLPSpanExporter",),
    "opentelemetry.exporter.otlp.proto.http._log_exporter": ("OTLPLogExporter",),
    "opentelemetry.exporter.otlp.proto.http.metric_exporter": ("OTLPMetricExporter",),
    "opentelemetry.trace": ("SpanKind", "INVALID_SPAN_ID", "INVALID_TRACE_ID", "TraceFlags"),
    "opentelemetry.metrics": ("Observation",),
    "opentelemetry._logs": ("LogRecord",),
    "opentelemetry._logs.severity": ("SeverityNumber",),
    "opentelemetry.sdk._logs": ("LoggerProvider",),
    "opentelemetry.sdk._logs.export": ("BatchLogRecordProcessor",),
    "opentelemetry.sdk.metrics": ("MeterProvider",),
    "opentelemetry.sdk.metrics.export": ("PeriodicExportingMetricReader",),
}
_SDK_SYMBOLS: Dict[str, str] = {name: module for module, names in _SDK_MODULES.items() for name in names}
_SPAN_SDK = ("TracerProvider", "BatchSpanProcessor", "Resource", "OTLPSpanExporter", "SpanKind")


def _require_sdk(names: Iterable[str] = _SPAN_SDK, *, auto_install: bool = True, prompt: bool = True) -> Dict[str, Any]:
    """Import the named OTel SDK symbols, lazily installing the extra on first use.

    Routes through tools.lazy_deps (feature 'export.otlp') — gated by security.allow_lazy_installs
    and TTY-prompted (``prompt=False`` from non-interactive contexts).  Any lazy-install failure
    falls through to the import attempt, which raises OTLPUnavailable with a manual install hint.
    """
    if auto_install:
        with suppress(Exception):
            from tools.lazy_deps import ensure as _lazy_ensure
            _lazy_ensure("export.otlp", prompt=prompt)
    try:
        return {name: getattr(importlib.import_module(_SDK_SYMBOLS[name]), name) for name in names}
    except Exception as e:  # ImportError or partial install
        raise OTLPUnavailable(
            "OTLP export requires the optional dependency. Install with:\n"
            "    pip install 'hermes-agent[otlp]'\n"
            f"(import error: {e})"
        )


# ── config + connection plumbing ─────────────────────────────────────────────
def _monitoring_section(config: Dict[str, Any], *path: str) -> Dict[str, Any]:
    node: Any = (config or {}).get("monitoring") or {}
    for key in path:
        node = node.get(key) or {}
    return node


def _otlp_config(config: Dict[str, Any]) -> Dict[str, Any]:
    return _monitoring_section(config, "export", "otlp")


def _resolve_headers(headers_env: Optional[Dict[str, str]]) -> Dict[str, str]:
    """Resolve {header_name: ENV_VAR_NAME} -> {header_name: value}; missing vars skipped."""
    resolved: Dict[str, str] = {}
    for header_name, env_name in (headers_env or {}).items():
        val = os.environ.get(str(env_name))
        if val:
            resolved[str(header_name)] = val
        else:
            logger.debug("OTLP header %s: env var %s not set; skipping", header_name, env_name)
    return resolved


_SIGNAL_SUFFIXES = ("/v1/traces", "/v1/metrics")


def _signal_endpoint(endpoint: str, signal: str) -> str:
    """Rewrite a traces/metrics OTLP path to ``/v1/<signal>``; other paths pass through."""
    target = f"/v1/{signal}"
    for suffix in _SIGNAL_SUFFIXES:
        if suffix != target and endpoint.endswith(suffix):
            return endpoint[: -len(suffix)] + target
    return endpoint


_RESOURCE_ATTRIBUTE_KEYS = frozenset({
    "service.name", "service.namespace", "service.version", "service.instance.id",
    "deployment.environment.name", "cloud.provider", "cloud.platform", "cloud.region", "telemetry.scope",
})
_SAFE_RESOURCE_VALUE = re.compile(r"^[A-Za-z0-9._:/-]{1,128}$")


def _install_id(config: Dict[str, Any]) -> str:
    try:
        from agent.monitoring.policy import ensure_install_id
        return str(ensure_install_id(config))
    except Exception:
        return "unknown"


def _safe_resource_attributes(raw: Any) -> Dict[str, str]:
    """Allowlist bounded resource labels and reject values changed by redaction."""
    attrs: Dict[str, str] = {}
    for key, value in (raw.items() if isinstance(raw, dict) else ()):
        key = str(key)
        if key not in _RESOURCE_ATTRIBUTE_KEYS or value is None:
            continue
        text = str(value)
        if key == "service.instance.id":
            attrs[key] = _safe_instance_id(value)
        elif _SAFE_RESOURCE_VALUE.fullmatch(text) and redact_bounded(text, limit=128) == text:
            attrs[key] = text
    return attrs


def _runtime_resource_attributes(config: Dict[str, Any], *, telemetry_scope: str) -> Dict[str, str]:
    """Build the safe OTLP resource shared by spans, metrics and diagnostic logs."""
    attrs = _safe_resource_attributes(_monitoring_section(config, "gateway_health_export").get("resource_attributes"))
    attrs["service.name"] = "hermes-gateway"
    attrs["service.instance.id"] = _safe_instance_id(_install_id(config))
    attrs["telemetry.scope"] = telemetry_scope
    return attrs


def build_exporter(config: Dict[str, Any]):
    """Construct an OTLP span exporter from config. Raises OTLPUnavailable if no SDK."""
    sdk = _require_sdk()
    otlp = _otlp_config(config)
    endpoint = otlp.get("endpoint")
    if not endpoint:
        raise ValueError("monitoring.export.otlp.endpoint is not set")
    return sdk["OTLPSpanExporter"](endpoint=endpoint, headers=_resolve_headers(otlp.get("headers_env")) or None)


def _resource_attributes(config: Dict[str, Any]) -> Dict[str, str]:
    return _runtime_resource_attributes(config, telemetry_scope="gateway_monitoring")


def _make_provider(config: Dict[str, Any]):
    sdk = _require_sdk()
    provider = sdk["TracerProvider"](resource=sdk["Resource"].create(_resource_attributes(config)))
    processor = sdk["BatchSpanProcessor"](build_exporter(config))
    provider.add_span_processor(processor)
    return provider, processor


# ── event -> span attribute mapping ──────────────────────────────────────────
# Per-kind attribute allowlists: everything else (profile, install_id, ...) never egresses.
_KEEP_BY_KIND: Dict[str, tuple[str, ...]] = {
    "gateway_health": (
        "name", "gateway_state", "old_state", "new_state", "exit_reason", "restart_requested", "active_agents",
        "gateway_busy", "gateway_drainable", "platform_count", "fatal_platform_count", "version", "supervision_mode", "pid",
    ),
    "gateway_diagnostic": ("name", "subsystem", "error_class", "error_code", "platform", "old_state", "new_state", "version", "severity"),
    "cron_execution": ("status", "job_key", "source", "duration_ms", "delivery_outcome", "error_class"),
}


def _allowlisted_attrs(ev: Dict[str, Any], keys: Iterable[str]) -> Dict[str, Any]:
    """``hermes.<key>`` attributes for present keys; string values are redacted and bounded."""
    attrs: Dict[str, Any] = {}
    for col in keys:
        v = ev.get(col)
        if v is not None:
            attrs[f"hermes.{col}"] = redact_bounded(v) if isinstance(v, str) else v
    return attrs


def _span_attrs(ev: Dict[str, Any]) -> Dict[str, Any]:
    """Span attributes for a monitoring event (content-free by construction)."""
    kind = ev.get("event")
    attrs: Dict[str, Any] = {"hermes.event": kind or "unknown"}
    attrs.update(_allowlisted_attrs(ev, _KEEP_BY_KIND.get(kind, ())))  # type: ignore[arg-type]
    return attrs


def export_batch(provider, batch: List[Dict[str, Any]]) -> int:
    """Map a batch of events to OTel spans. Returns spans created."""
    tracer = provider.get_tracer("hermes.monitoring")
    n = 0
    for ev in batch:
        try:
            tracer.start_span(f"hermes.{ev.get('event', 'event')}", attributes=_span_attrs(ev)).end()
            n += 1
        except Exception:
            logger.debug("OTLP span map failed", exc_info=True)
    return n


# ── continuous streaming subscribers ─────────────────────────────────────────
class EmitterStreamer:
    """Base for emitter subscribers owning an OTel provider + batch processor.
    Register with ``emitter.subscribe(streamer)``. Fail-isolated by the emitter."""
    _provider: Any
    _processor: Any
    exported: int = 0

    def shutdown(self) -> None:
        with suppress(Exception):
            from agent.monitoring.emitter import get_emitter
            get_emitter().unsubscribe(self)
        with suppress(Exception):
            self._processor.force_flush()
            self._provider.shutdown()


class OTLPStreamer(EmitterStreamer):
    """A live subscriber that pushes each emitter batch to OTLP as spans."""

    def __init__(self, config: Dict[str, Any], *, event_filter: Optional[Callable[[Dict[str, Any]], bool]] = None):
        self._provider, self._processor = _make_provider(config)
        self._event_filter = event_filter
        self.exported = 0

    def __call__(self, batch: List[Dict[str, Any]]) -> None:
        if self._event_filter is not None:
            batch = [ev for ev in batch if self._event_filter(ev)]
        if not batch:
            return
        self.exported += export_batch(self._provider, batch)


def is_available() -> bool:
    """True when the OTel SDK is already importable (pure check, no auto-install)."""
    try:
        _require_sdk(auto_install=False)
        return True
    except OTLPUnavailable:
        return False


def is_enabled(config: Dict[str, Any]) -> bool:
    otlp = _otlp_config(config)
    return bool(otlp.get("enabled") and otlp.get("endpoint"))


def start_streaming(
    config: Dict[str, Any], *, event_filter: Optional[Callable[[Dict[str, Any]], bool]] = None,
) -> Optional[OTLPStreamer]:
    """If OTLP is enabled, attach a streamer to the singleton emitter.

    ``event_filter`` scopes the exporter to its plane.  Startup is non-interactive: a
    configured-but-missing SDK is lazily installed once (prompt=False, gated by
    security.allow_lazy_installs); if it still can't load, log and no-op — never raise into startup.
    """
    if not is_enabled(config):
        return None
    try:
        _require_sdk(prompt=False)
    except OTLPUnavailable:
        logger.warning("monitoring.export.otlp.enabled but the OTel SDK could not "
                       "be installed/imported; install 'hermes-agent[otlp]'")
        return None
    from agent.monitoring.emitter import get_emitter
    streamer = OTLPStreamer(config, event_filter=event_filter)
    get_emitter().subscribe(streamer)
    return streamer


__all__ = [
    "OTLPUnavailable", "OTLPStreamer", "build_exporter", "export_batch", "is_available", "is_enabled", "start_streaming",
]
