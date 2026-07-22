"""Export monitoring events to an OpenTelemetry Collector over OTLP/HTTP.

Maps gateway monitoring events to OTel spans and sends them to the endpoint
configured under ``monitoring.export.otlp``. Lets an operator stream Hermes
gateway health into their own observability stack (OTEL Collector, DataDog,
and similar).

Notes:
  * The destination is operator-configured; this module only sends to that
    endpoint. No default destination ships.
  * ``opentelemetry-sdk`` + ``opentelemetry-exporter-otlp-proto-http`` are an
    optional extra (``pip install hermes-agent[otlp]``), imported lazily so the
    dependency is only required when OTLP export is actually used.
  * ``headers_env`` maps a header name to an environment variable name; values
    are read from the environment at export time and never logged or stored.
  * The continuous subscriber runs in the emitter's dispatcher thread and is
    fail-isolated, so an export error cannot affect the gateway.

Only monitoring events (gateway_health / gateway_diagnostic) exist on this
plane; the ``event_filter`` seam is kept so future planes sharing the emitter
cannot silently ride along on this exporter.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


class OTLPUnavailable(RuntimeError):
    """Raised when the optional OpenTelemetry SDK isn't installed."""


def _require_sdk(*, auto_install: bool = True, prompt: bool = True):
    """Import the OTel SDK, lazily installing it on first use if needed.

    Routes through tools.lazy_deps (feature 'export.otlp') so a missing SDK
    triggers the standard venv install flow — same as every other optional
    backend — gated by security.allow_lazy_installs and TTY-prompted. Falls back
    to OTLPUnavailable (with a manual install hint) when the SDK can't be made
    importable (lazy installs disabled, install failed, or auto_install=False).

    ``auto_install``: attempt the lazy install when missing (default True).
    ``prompt``: ask before installing when interactive (default True); pass
    False from non-interactive contexts like the continuous streamer.
    """
    if auto_install:
        try:
            from tools.lazy_deps import ensure as _lazy_ensure
            _lazy_ensure("export.otlp", prompt=prompt)
        except ImportError:
            pass  # lazy_deps unavailable — fall through to the import attempt
        except Exception:
            # FeatureUnavailable (lazy installs disabled / declined / failed) —
            # fall through; the import below raises OTLPUnavailable with the hint.
            pass
    try:
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
            OTLPSpanExporter,
        )
        from opentelemetry.trace import SpanKind
        return {
            "TracerProvider": TracerProvider,
            "BatchSpanProcessor": BatchSpanProcessor,
            "Resource": Resource,
            "OTLPSpanExporter": OTLPSpanExporter,
            "SpanKind": SpanKind,
        }
    except Exception as e:  # ImportError or partial install
        raise OTLPUnavailable(
            "OTLP export requires the optional dependency. Install with:\n"
            "    pip install 'hermes-agent[otlp]'\n"
            f"(import error: {e})"
        )


def _resolve_headers(headers_env: Optional[Dict[str, str]]) -> Dict[str, str]:
    """Resolve {header_name: ENV_VAR_NAME} -> {header_name: value} from env.

    The config stores environment variable names, not secret values; values are
    read from the environment here. Missing variables are skipped (and noted at
    debug level without the value).
    """
    resolved: Dict[str, str] = {}
    for header_name, env_name in (headers_env or {}).items():
        val = os.environ.get(str(env_name))
        if val:
            resolved[str(header_name)] = val
        else:
            logger.debug("OTLP header %s: env var %s not set; skipping",
                         header_name, env_name)
    return resolved


def _otlp_config(config: Dict[str, Any]) -> Dict[str, Any]:
    mon = (config or {}).get("monitoring") or {}
    export = mon.get("export") or {}
    return export.get("otlp") or {}


def build_exporter(config: Dict[str, Any]):
    """Construct an OTLP span exporter from config. Raises OTLPUnavailable if no SDK."""
    sdk = _require_sdk()
    otlp = _otlp_config(config)
    endpoint = otlp.get("endpoint")
    if not endpoint:
        raise ValueError("monitoring.export.otlp.endpoint is not set")
    headers = _resolve_headers(otlp.get("headers_env"))
    return sdk["OTLPSpanExporter"](endpoint=endpoint, headers=headers or None)


def _make_provider(config: Dict[str, Any]):
    sdk = _require_sdk()
    resource = sdk["Resource"].create({
        "service.name": "hermes-gateway",
        "telemetry.scope": "gateway_monitoring",
    })
    provider = sdk["TracerProvider"](resource=resource)
    processor = sdk["BatchSpanProcessor"](build_exporter(config))
    provider.add_span_processor(processor)
    return provider, processor


# ── event -> span attribute mapping ──────────────────────────────────────────
def _span_attrs(ev: Dict[str, Any]) -> Dict[str, Any]:
    """Span attributes for a monitoring event (content-free by construction)."""
    kind = ev.get("event")
    attrs: Dict[str, Any] = {"hermes.event": kind or "unknown"}
    keep_by_kind = {
        "gateway_health": ("name", "gateway_state", "old_state", "new_state",
                           "exit_reason", "restart_requested", "active_agents",
                           "gateway_busy", "gateway_drainable", "platform_count",
                           "fatal_platform_count", "version",
                           "supervision_mode", "pid"),
        "gateway_diagnostic": ("name", "subsystem", "error_class", "error_code",
                               "redacted_message", "platform", "old_state", "new_state",
                               "version", "severity"),
    }
    for col in keep_by_kind.get(kind, ()):  # type: ignore[arg-type]
        v = ev.get(col)
        if v is not None:
            if isinstance(v, str):
                try:
                    from agent.monitoring.redaction import redact_for_export
                    v = (redact_for_export(v) or "[redacted]")[:500]
                except Exception:
                    v = "[redaction-unavailable]"
            attrs[f"hermes.{col}"] = v
    return attrs


def export_batch(provider, batch: List[Dict[str, Any]]) -> int:
    """Map a batch of events to OTel spans. Returns spans created."""
    tracer = provider.get_tracer("hermes.monitoring")
    n = 0
    for ev in batch:
        try:
            name = f"hermes.{ev.get('event', 'event')}"
            span = tracer.start_span(name, attributes=_span_attrs(ev))
            span.end()
            n += 1
        except Exception:
            logger.debug("OTLP span map failed", exc_info=True)
    return n


# ── continuous streaming subscriber ─────────────────────────────────────────
class OTLPStreamer:
    """A live subscriber that pushes each emitter batch to OTLP as it lands.

    Register with ``emitter.subscribe(streamer)``. Fail-isolated by the emitter.
    """

    def __init__(
        self,
        config: Dict[str, Any],
        *,
        event_filter: Optional[Callable[[Dict[str, Any]], bool]] = None,
    ):
        self._provider, self._processor = _make_provider(config)
        self._event_filter = event_filter
        self.exported = 0

    def __call__(self, batch: List[Dict[str, Any]]) -> None:
        if self._event_filter is not None:
            batch = [ev for ev in batch if self._event_filter(ev)]
        if not batch:
            return
        self.exported += export_batch(self._provider, batch)

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


def is_available() -> bool:
    """True when the OTel SDK is already importable. Does NOT auto-install —
    this is a pure check (e.g. for status display)."""
    try:
        _require_sdk(auto_install=False)
        return True
    except OTLPUnavailable:
        return False


def is_enabled(config: Dict[str, Any]) -> bool:
    otlp = _otlp_config(config)
    return bool(otlp.get("enabled") and otlp.get("endpoint"))


def start_streaming(
    config: Dict[str, Any],
    *,
    event_filter: Optional[Callable[[Dict[str, Any]], bool]] = None,
) -> Optional[OTLPStreamer]:
    """If OTLP is enabled, attach a streamer to the singleton emitter.

    ``event_filter`` scopes the exporter to its plane, e.g. gateway-health
    export, so enabling one plane cannot silently export unrelated events.

    Non-interactive context (startup): attempts a lazy install with prompt=False
    so a configured-but-missing SDK is installed once (gated by
    security.allow_lazy_installs), then streams. If it still can't load, logs and
    no-ops — never blocks or raises into startup.
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
    "OTLPUnavailable",
    "OTLPStreamer",
    "build_exporter",
    "export_batch",
    "is_available",
    "is_enabled",
    "start_streaming",
]
