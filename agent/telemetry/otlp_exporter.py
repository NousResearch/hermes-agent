"""Export telemetry to an OpenTelemetry Collector over OTLP/HTTP.

Maps the local tel_* events to OTel spans and sends them to the endpoint configured
under ``telemetry.export.otlp``. Lets an operator stream Hermes telemetry into their
own observability stack.

Notes:
  * The destination is operator-configured; this module only sends to that endpoint.
    It does not import or interact with any aggregate-metrics path.
  * ``opentelemetry-sdk`` + ``opentelemetry-exporter-otlp-proto-http`` are an optional
    extra (``pip install hermes-agent[otlp]``), imported lazily so the dependency is
    only required when OTLP export is actually used.
  * ``headers_env`` maps a header name to an environment variable name; values are read
    from the environment at export time and never logged or stored.
  * The continuous subscriber runs in the emitter's writer thread after durable writes
    and is fail-isolated, so an export error cannot affect a run.

Each event is exported as a span carrying its recorded attributes (provider, model,
tokens, duration, etc.). The timing/parent linkage captured in tel_spans
(trace_id/span_id/parent_span_id/start_ns/end_ns) is not yet reconstructed into OTel
SpanContexts here, so spans currently arrive as independent records rather than a
connected trace tree; building the connected-trace projection is tracked separately.

Spans carry structural telemetry by default. Message content is included only when
trajectories is enabled, and always passes through the export redaction pipeline.
"""

from __future__ import annotations

import logging
import os
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional

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

    The config stores environment variable names, not secret values; values are read
    from the environment here. Missing variables are skipped (and noted at debug level
    without the value).
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
    tel = (config or {}).get("telemetry") or {}
    export = tel.get("export") or {}
    return export.get("otlp") or {}


def build_exporter(config: Dict[str, Any]):
    """Construct an OTLP span exporter from config. Raises OTLPUnavailable if no SDK."""
    sdk = _require_sdk()
    otlp = _otlp_config(config)
    endpoint = otlp.get("endpoint")
    if not endpoint:
        raise ValueError("telemetry.export.otlp.endpoint is not set")
    headers = _resolve_headers(otlp.get("headers_env"))
    return sdk["OTLPSpanExporter"](endpoint=endpoint, headers=headers or None)


def _make_provider(config: Dict[str, Any]):
    sdk = _require_sdk()
    resource = sdk["Resource"].create({
        "service.name": "hermes-agent",
        "telemetry.scope": "local",  # never aggregate metrics
    })
    provider = sdk["TracerProvider"](resource=resource)
    processor = sdk["BatchSpanProcessor"](build_exporter(config))
    provider.add_span_processor(processor)
    return provider, processor


# ── event -> span attribute mapping (real values) ───────────────────────────
def _span_attrs(ev: Dict[str, Any]) -> Dict[str, Any]:
    """Span attributes for an event — the real recorded values (local telemetry)."""
    kind = ev.get("event")
    attrs: Dict[str, Any] = {"hermes.event": kind or "unknown"}
    keep_by_kind = {
        "run": ("entrypoint", "platform", "end_reason",
                "model_call_count", "tool_call_count", "error_count",
                "estimated_cost_usd", "cost_status"),
        "span": ("trace_id", "run_id", "parent_span_id", "name", "kind",
                 "start_ns", "end_ns", "status"),
        "model_call": ("provider", "model", "base_url",
                       "input_tokens", "output_tokens", "cache_read_tokens",
                       "cache_write_tokens", "reasoning_tokens", "latency_ms",
                       "ttft_ms", "end_reason"),
        "tool_call": ("tool_name", "backend", "duration_ms", "result_class"),
        "error": ("error_class", "subsystem", "recovery"),
    }
    for col in keep_by_kind.get(kind, ()):  # type: ignore[arg-type]
        v = ev.get(col)
        if v is not None:
            attrs[f"hermes.{col}"] = v
    return attrs


def export_batch(provider, batch: List[Dict[str, Any]]) -> int:
    """Map a batch of events to OTel spans. Returns spans created."""
    tracer = provider.get_tracer("hermes.telemetry")
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


# ── one-shot drain (export current local rows) ──────────────────────────────
def export_once(
    config: Dict[str, Any],
    *,
    db_path: Optional[Path] = None,
    since_ns: Optional[int] = None,
) -> int:
    """Drain the local tel_* tables to the configured OTLP endpoint once."""
    provider, processor = _make_provider(config)
    try:
        rows = _read_events(db_path, since_ns)
        total = export_batch(provider, rows)
        processor.force_flush()
        return total
    finally:
        try:
            provider.shutdown()
        except Exception:
            pass


def _read_events(db_path: Optional[Path], since_ns: Optional[int]) -> List[Dict[str, Any]]:
    if db_path is None:
        from hermes_constants import get_hermes_home
        db_path = get_hermes_home() / "state.db"
    c = sqlite3.connect(str(db_path), timeout=5.0)
    c.row_factory = sqlite3.Row
    out: List[Dict[str, Any]] = []
    try:
        table_event = {
            "tel_runs": "run", "tel_spans": "span",
            "tel_model_calls": "model_call",
            "tel_tool_calls": "tool_call", "tel_error_events": "error",
        }
        for table, evkind in table_event.items():
            where = ""
            if table == "tel_runs" and since_ns:
                where = f" WHERE start_ns >= {int(since_ns)}"
            for r in c.execute(f"SELECT * FROM {table}{where}").fetchall():
                d = dict(r)
                d["event"] = evkind
                out.append(d)
    finally:
        c.close()
    return out


# ── continuous streaming subscriber ─────────────────────────────────────────
class OTLPStreamer:
    """A live subscriber that pushes each emitter batch to OTLP as it lands.

    Register with ``emitter.subscribe(streamer)``. Fail-isolated by the emitter.
    """

    def __init__(self, config: Dict[str, Any]):
        self._provider, self._processor = _make_provider(config)
        self.exported = 0

    def __call__(self, batch: List[Dict[str, Any]]) -> None:
        self.exported += export_batch(self._provider, batch)

    def shutdown(self) -> None:
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


def start_streaming(config: Dict[str, Any]) -> Optional[OTLPStreamer]:
    """If OTLP is enabled, attach a streamer to the singleton emitter.

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
        logger.warning("telemetry.export.otlp.enabled but the OTel SDK could not "
                       "be installed/imported; install 'hermes-agent[otlp]'")
        return None
    from agent.telemetry.emitter import get_emitter
    streamer = OTLPStreamer(config)
    get_emitter().subscribe(streamer)
    return streamer


__all__ = [
    "OTLPUnavailable",
    "OTLPStreamer",
    "build_exporter",
    "export_once",
    "export_batch",
    "is_available",
    "is_enabled",
    "start_streaming",
]
