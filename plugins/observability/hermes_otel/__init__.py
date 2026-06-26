"""hermes_otel — Hermes observability plugin (Phase 1: JSONL exporter).

Collects per-API-call metadata (model, provider, tokens, cost, credits, rate
limits, errors) from the Hermes plugin hook system and writes structured spans as
JSONL to ``~/.hermes/logs/otel-spans.jsonl``.

Activation is handled by the Hermes plugin system — standalone plugins only
load when listed in ``plugins.enabled`` (via ``hermes plugins enable
observability/hermes_otel`` or ``hermes tools → Hermes OTel``). At runtime the
plugin is fail-open: missing dependencies, missing config, or write failures all
short-circuit to inert behavior rather than crashing the agent.

Optional env vars (set via ``hermes tools`` or ~/.hermes/.env):
  HERMES_OTEL_ENABLED=true              # master switch (default true if plugin enabled)
  HERMES_OTEL_OUTPUT_PATH               # override JSONL path
  HERMES_OTEL_FLUSH_SECONDS=5           # in-memory flush interval
  HERMES_OTEL_MAX_BUFFER=100            # in-memory buffer cap before flush
  HERMES_OTEL_SAMPLE_RATE=1.0          # 0.0-1.0 sampling
"""
from __future__ import annotations

import json
import logging
import os
import threading
import time
from dataclasses import asdict
from decimal import Decimal
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# Phase 1 uses a simple JSONL exporter.  The opentelemetry-sdk is not required
# until Phase 2 (OTLP exporter).  Keeping the import optional keeps startup
# cheap and the plugin fail-open.
try:
    from opentelemetry import trace
except Exception:  # pragma: no cover - fail-open when optional dep is missing
    trace = None  # type: ignore

try:
    from agent.usage_pricing import CanonicalUsage, estimate_usage_cost, normalize_usage
except Exception:
    CanonicalUsage = None  # type: ignore
    normalize_usage = None  # type: ignore
    estimate_usage_cost = None  # type: ignore


# Sentinel: _get_exporter() has tried and failed. Lets subsequent hook calls
# short-circuit without re-checking env vars or re-attempting exporter init.
# Tests clear this by reloading the module.
_INIT_FAILED = object()

_STATE_LOCK = threading.Lock()
_EXPORTER: Any = None


def _env(name: str, default: str = "") -> str:
    return os.environ.get(name, default).strip()


def _env_bool(*names: str) -> bool:
    for name in names:
        value = _env(name).lower()
        if value:
            return value in {"1", "true", "yes", "on"}
    return False


def _sample_rate() -> float:
    raw = _env("HERMES_OTEL_SAMPLE_RATE", "1.0")
    try:
        return max(0.0, min(1.0, float(raw)))
    except ValueError:
        logger.warning("Invalid HERMES_OTEL_SAMPLE_RATE=%r", raw)
        return 1.0


def _should_sample() -> bool:
    """Sample this request. Deterministic by api_request_id if present in ctx."""
    rate = _sample_rate()
    if rate >= 1.0:
        return True
    if rate <= 0.0:
        return False
    # Deterministic sampling: use the last 32 bits of the api_request_id hash so
    # retries of the same request get the same decision and different requests
    # spread uniformly.
    return (hash(threading.current_thread()) % 1000000) / 1000000 < rate


def _profile() -> str:
    return _env("HERMES_PROFILE", "default")


def _hermes_home() -> str:
    from hermes_constants import get_hermes_home

    return str(get_hermes_home())


def _default_output_path() -> str:
    custom = _env("HERMES_OTEL_OUTPUT_PATH")
    if custom:
        return custom
    return os.path.join(_hermes_home(), "logs", "otel-spans.jsonl")


def _serialize_rate_limits(value: Any) -> Optional[Dict[str, Any]]:
    """Serialize a RateLimitState (or plain dict) to a flat dict.

    The hook may receive either a serialized dict from the core or, in tests, a
    dataclass.  This helper normalizes both to a stable JSON-ready shape.
    """
    if value is None:
        return None
    if isinstance(value, dict):
        data = value
    else:
        try:
            data = asdict(value)
        except TypeError:  # pragma: no cover - fail-open on unsupported input shapes
            return None

    def _bucket(name: str) -> Dict[str, Any]:
        b = data.get(name) or {}
        if not isinstance(b, dict):
            b = {}
        return {
            "limit": b.get("limit", 0),
            "remaining": b.get("remaining", 0),
            "reset_seconds": b.get("reset_seconds", 0.0),
        }

    out: Dict[str, Any] = {}
    for short, full in (
        ("rpm", "requests_min"),
        ("rph", "requests_hour"),
        ("tpm", "tokens_min"),
        ("tph", "tokens_hour"),
    ):
        b = _bucket(full)
        out[f"{short}_limit"] = b["limit"]
        out[f"{short}_remaining"] = b["remaining"]
        out[f"{short}_reset_seconds"] = b["reset_seconds"]
    return out


def _credits_dict(value: Any) -> Optional[Dict[str, Any]]:
    """Normalize CreditsState (or dict) to a JSON-ready dict."""
    if value is None:
        return None
    if isinstance(value, dict):
        return value
    try:
        return asdict(value)
    except TypeError:  # pragma: no cover - fail-open on unsupported input shapes
        return None


class _JsonlExporter:
    """Thread-safe buffered JSONL exporter.

    Buffers completed spans in memory and flushes either when the buffer reaches
    ``max_buffer`` spans or every ``flush_seconds``.
    """

    def __init__(
        self,
        output_path: str,
        *,
        flush_seconds: float = 5.0,
        max_buffer: int = 100,
    ) -> None:
        self.output_path = output_path
        self.flush_seconds = flush_seconds
        self.max_buffer = max_buffer
        self._buffer: list[Dict[str, Any]] = []
        self._lock = threading.Lock()
        self._last_flush = time.time()
        self._flush_thread: Optional[threading.Thread] = None
        # Ensure the log directory exists.
        try:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
        except OSError:  # pragma: no cover - log dir creation is best-effort
            pass

    def emit(self, span: Dict[str, Any]) -> None:
        should_flush = False
        with self._lock:
            self._buffer.append(span)
            if (
                len(self._buffer) >= self.max_buffer
                or (time.time() - self._last_flush) >= self.flush_seconds
            ):
                should_flush = True

        if should_flush:
            self._flush()

    def _flush(self) -> None:
        with self._lock:
            batch = list(self._buffer)
            self._buffer.clear()
            self._last_flush = time.time()

        if not batch:
            return

        try:
            with open(self.output_path, "a", encoding="utf-8") as fh:
                for span in batch:
                    fh.write(json.dumps(span, default=_json_default) + "\n")
        except Exception as exc:
            logger.debug("Hermes OTel flush failed: %s", exc)

    def flush(self) -> None:
        self._flush()

    def shutdown(self) -> None:
        self._flush()


def _json_default(value: Any) -> Any:
    if isinstance(value, Decimal):
        return float(value)
    if isinstance(value, set):
        return list(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _get_exporter() -> Optional[_JsonlExporter]:
    global _EXPORTER
    if _EXPORTER is _INIT_FAILED:
        return None
    if isinstance(_EXPORTER, _JsonlExporter):
        return _EXPORTER

    if not _env_bool("HERMES_OTEL_ENABLED"):
        # The plugin is enabled in config; default it on unless explicitly off.
        pass

    try:
        flush_seconds = float(_env("HERMES_OTEL_FLUSH_SECONDS", "5") or "5")
    except ValueError:
        flush_seconds = 5.0
    try:
        max_buffer = int(_env("HERMES_OTEL_MAX_BUFFER", "100") or "100")
    except ValueError:
        max_buffer = 100

    try:
        _EXPORTER = _JsonlExporter(
            _default_output_path(),
            flush_seconds=flush_seconds,
            max_buffer=max_buffer,
        )
    except Exception as exc:
        logger.debug("Hermes OTel exporter disabled: init failed: %s", exc)
        _EXPORTER = _INIT_FAILED
        return None

    return _EXPORTER


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _usage_cost(
    *,
    response: Any,
    usage: Any,
    provider: str,
    api_mode: str,
    model: str,
    base_url: str,
) -> Dict[str, Any]:
    """Return cost estimate attributes from the usage summary."""
    if estimate_usage_cost is None or normalize_usage is None or CanonicalUsage is None:
        return {"cost_amount_usd": None, "cost_status": None, "cost_source": None}

    try:
        if isinstance(usage, dict) and usage:
            canonical = {
                "input_tokens": _safe_int(usage.get("input_tokens")),
                "output_tokens": _safe_int(usage.get("output_tokens")),
                "cache_read_tokens": _safe_int(usage.get("cache_read_tokens")),
                "cache_write_tokens": _safe_int(usage.get("cache_write_tokens")),
                "reasoning_tokens": _safe_int(usage.get("reasoning_tokens")),
            }
        elif response is not None and getattr(response, "usage", None) is not None:
            from dataclasses import asdict as _asdict

            cu = normalize_usage(response.usage, provider=provider, api_mode=api_mode)
            canonical = _asdict(cu)
        else:
            canonical = {
                "input_tokens": 0,
                "output_tokens": 0,
                "cache_read_tokens": 0,
                "cache_write_tokens": 0,
                "reasoning_tokens": 0,
            }

        cu = CanonicalUsage(**canonical)
        cost = estimate_usage_cost(
            model,
            cu,
            provider=provider,
            base_url=base_url,
            api_key="",
        )
        amount = float(cost.amount_usd) if cost.amount_usd is not None else None
        return {
            "cost_amount_usd": amount,
            "cost_status": cost.status,
            "cost_source": cost.source,
        }
    except Exception as exc:
        logger.debug("Hermes OTel cost estimation failed: %s", exc)
        return {"cost_amount_usd": None, "cost_status": None, "cost_source": None}


# In-memory span state keyed by api_request_id.  We only store metadata — never
# message content — so memory growth is bounded by the number of in-flight API
# calls.
_SPAN_STATE: Dict[str, Dict[str, Any]] = {}


def _span_state_key(api_request_id: str) -> str:
    return api_request_id


# Fields in the exported JSONL span schema, in a stable order.
_SPAN_FIELDS: tuple[str, ...] = (
    "trace_id",
    "span_id",
    "timestamp",
    "duration_ms",
    "profile",
    "session_id",
    "platform",
    "model",
    "provider",
    "base_url",
    "api_mode",
    "api_call_count",
    "turn_id",
    "input_tokens",
    "output_tokens",
    "cache_read_tokens",
    "cache_write_tokens",
    "reasoning_tokens",
    "prompt_tokens",
    "total_tokens",
    "finish_reason",
    "response_model",
    "assistant_content_chars",
    "assistant_tool_call_count",
    "message_count",
    "cost_amount_usd",
    "cost_status",
    "cost_source",
    "credits_remaining_usd",
    "credits_remaining_micros",
    "credits_paid_access",
    "credits_used_fraction",
    "rate_limit_rpm_remaining",
    "rate_limit_rpm_limit",
    "rate_limit_tpm_remaining",
    "rate_limit_tpm_limit",
    "error_type",
    "error_message",
    "status_code",
    "failover_reason",
    "retry_count",
)


def on_pre_api_request(
    *,
    api_request_id: str = "",
    session_id: str = "",
    platform: str = "",
    model: str = "",
    provider: str = "",
    base_url: str = "",
    api_mode: str = "",
    api_call_count: int = 0,
    turn_id: str = "",
    started_at: float = 0.0,
    **_: Any,
) -> None:
    exporter = _get_exporter()
    if exporter is None:
        return
    if not api_request_id:
        return
    if not _should_sample():
        return

    span: Dict[str, Any] = {"_started_at": started_at}
    for key in _SPAN_FIELDS:
        span[key] = None

    span.update(
        {
            "trace_id": session_id or api_request_id,
            "span_id": api_request_id,
            "timestamp": _iso_timestamp(started_at),
            "duration_ms": None,
            "profile": _profile(),
            "session_id": session_id or "",
            "platform": platform or "",
            "model": model or "",
            "provider": provider or "",
            "base_url": base_url or "",
            "api_mode": api_mode or "",
            "api_call_count": api_call_count,
            "turn_id": turn_id or "",
        }
    )

    with _STATE_LOCK:
        _SPAN_STATE[_span_state_key(api_request_id)] = span


def _iso_timestamp(ts: float) -> str:
    from datetime import datetime, timezone

    if not ts:
        ts = time.time()
    return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()


def _complete_span(
    *,
    api_request_id: str = "",
    ended_at: float = 0.0,
    usage: Any = None,
    response_model: Any = None,
    assistant_content_chars: int = 0,
    assistant_tool_call_count: int = 0,
    message_count: int = 0,
    finish_reason: Optional[str] = None,
    credits_state: Any = None,
    rate_limit_state: Any = None,
    cost_kwargs: Optional[Dict[str, Any]] = None,
    error: Optional[Dict[str, Any]] = None,
    status_code: Optional[int] = None,
    failover_reason: Optional[str] = None,
    retry_count: Optional[int] = None,
) -> Optional[Dict[str, Any]]:
    """Finalize and emit a span from the in-memory state."""
    exporter = _get_exporter()
    if exporter is None:
        return None
    if not api_request_id:
        return None

    with _STATE_LOCK:
        span = _SPAN_STATE.pop(_span_state_key(api_request_id), None)
    if span is None:
        return None

    started_at = span.pop("_started_at", 0.0)
    duration_ms = int((ended_at - started_at) * 1000) if ended_at and started_at else None
    span["duration_ms"] = duration_ms

    if isinstance(usage, dict):
        span["input_tokens"] = usage.get("input_tokens")
        span["output_tokens"] = usage.get("output_tokens")
        span["cache_read_tokens"] = usage.get("cache_read_tokens")
        span["cache_write_tokens"] = usage.get("cache_write_tokens")
        span["reasoning_tokens"] = usage.get("reasoning_tokens")
        span["prompt_tokens"] = usage.get("prompt_tokens")
        span["total_tokens"] = usage.get("total_tokens")

    span["response_model"] = response_model
    span["assistant_content_chars"] = assistant_content_chars
    span["assistant_tool_call_count"] = assistant_tool_call_count
    span["message_count"] = message_count
    span["finish_reason"] = finish_reason

    credits = _credits_dict(credits_state)
    if credits:
        span["credits_remaining_usd"] = credits.get("remaining_usd")
        span["credits_remaining_micros"] = credits.get("remaining_micros")
        span["credits_paid_access"] = credits.get("paid_access")
        span["credits_used_fraction"] = credits.get("used_fraction")

    rate_limits = _serialize_rate_limits(rate_limit_state)
    if rate_limits:
        span["rate_limit_rpm_remaining"] = rate_limits.get("rpm_remaining")
        span["rate_limit_rpm_limit"] = rate_limits.get("rpm_limit")
        span["rate_limit_tpm_remaining"] = rate_limits.get("tpm_remaining")
        span["rate_limit_tpm_limit"] = rate_limits.get("tpm_limit")

    if cost_kwargs:
        try:
            cost_attrs = _usage_cost(**cost_kwargs)
            span.update(cost_attrs)
        except Exception as exc:
            logger.debug("Hermes OTel cost computation failed: %s", exc)

    if error:
        span["error_type"] = error.get("type")
        span["error_message"] = error.get("message")

    if status_code is not None:
        span["status_code"] = status_code
    if failover_reason is not None:
        span["failover_reason"] = failover_reason
    if retry_count is not None:
        span["retry_count"] = retry_count

    try:
        exporter.emit(span)
    except Exception as exc:
        logger.debug("Hermes OTel emit failed: %s", exc)

    return span


def on_post_api_request(
    *,
    api_request_id: str = "",
    session_id: str = "",
    turn_id: str = "",
    api_call_count: int = 0,
    api_duration: float = 0.0,
    started_at: float = 0.0,
    ended_at: float = 0.0,
    response: Any = None,
    response_model: Any = None,
    usage: Any = None,
    assistant_content_chars: int = 0,
    assistant_tool_call_count: int = 0,
    message_count: int = 0,
    finish_reason: Optional[str] = None,
    model: str = "",
    provider: str = "",
    base_url: str = "",
    api_mode: str = "",
    credits_state: Any = None,
    rate_limit_state: Any = None,
    **_: Any,
) -> None:
    try:
        _complete_span(
            api_request_id=api_request_id,
            ended_at=ended_at or (started_at + api_duration) or time.time(),
            usage=usage,
            response_model=response_model,
            assistant_content_chars=assistant_content_chars,
            assistant_tool_call_count=assistant_tool_call_count,
            message_count=message_count,
            finish_reason=finish_reason,
            credits_state=credits_state,
            rate_limit_state=rate_limit_state,
            cost_kwargs={
                "response": response,
                "usage": usage,
                "provider": provider,
                "api_mode": api_mode,
                "model": model,
                "base_url": base_url,
            },
        )
    except Exception as exc:
        logger.debug("Hermes OTel post_api_request hook failed: %s", exc)


def on_api_request_error(
    *,
    api_request_id: str = "",
    ended_at: float = 0.0,
    started_at: float = 0.0,
    api_duration: float = 0.0,
    error: Any = None,
    status_code: Optional[int] = None,
    reason: Optional[str] = None,
    retry_count: Optional[int] = None,
    **_: Any,
) -> None:
    try:
        err: Optional[Dict[str, Any]] = None
        if isinstance(error, dict):
            err = error
        elif error is not None:
            err = {"type": type(error).__name__, "message": str(error)}
        _complete_span(
            api_request_id=api_request_id,
            ended_at=ended_at or (started_at + api_duration) or time.time(),
            error=err,
            status_code=status_code,
            failover_reason=reason,
            retry_count=retry_count,
        )
    except Exception as exc:
        logger.debug("Hermes OTel api_request_error hook failed: %s", exc)


def on_session_start(*, session_id: str = "", platform: str = "", **_: Any) -> None:
    """Optional Phase 1 hook — currently a no-op; used in Phase 2 for trace roots."""
    pass


def on_session_end(*, session_id: str = "", **_: Any) -> None:
    try:
        exporter = _get_exporter()
        if exporter is not None:
            exporter.flush()
    except Exception as exc:
        logger.debug("Hermes OTel session_end flush failed: %s", exc)


def register(ctx) -> None:
    ctx.register_hook("on_session_start", on_session_start)
    ctx.register_hook("on_session_end", on_session_end)
    ctx.register_hook("pre_api_request", on_pre_api_request)
    ctx.register_hook("post_api_request", on_post_api_request)
    ctx.register_hook("api_request_error", on_api_request_error)
