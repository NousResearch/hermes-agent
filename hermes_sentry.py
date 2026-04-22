"""Centralized Sentry bootstrap and event capture helpers for Hermes.

Sentry is optional and entirely environment-driven. When no DSN is present,
the helpers become cheap no-ops so the rest of Hermes startup and cron
execution remain unchanged.
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import atexit
import threading
from typing import Any, Generator, Mapping

from agent.redact import redact_sensitive_text

logger = logging.getLogger(__name__)

try:
    import sentry_sdk
    from sentry_sdk.integrations.logging import LoggingIntegration
    from sentry_sdk.integrations.threading import ThreadingIntegration
except ImportError:  # pragma: no cover - exercised by runtime packaging
    sentry_sdk = None
    LoggingIntegration = None
    ThreadingIntegration = None


_INIT_LOCK = threading.Lock()
_SENTRY_INITIALIZED = False


def _dsn() -> str:
    return (
        os.getenv("SENTRY_DSN", "").strip()
        or os.getenv("HERMES_SENTRY_DSN", "").strip()
    )


def _coerce_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, default=str, sort_keys=True)
    except (TypeError, ValueError):
        return str(value)


def sanitize_observability_text(value: Any, *, limit: int = 4000) -> str | None:
    """Redact secrets and neutralize newlines before attaching text to telemetry."""
    if value is None:
        return None

    rendered = redact_sensitive_text(_coerce_text(value))
    rendered = rendered.replace("\r", "\\r").replace("\n", "\\n")
    if len(rendered) > limit:
        return rendered[: limit - 3] + "..."
    return rendered


def _sanitize_event_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _sanitize_event_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_sanitize_event_value(item) for item in value]
    if isinstance(value, str):
        return sanitize_observability_text(value)
    return value


def _resolve_environment() -> str:
    return (
        os.getenv("HERMES_SENTRY_ENVIRONMENT", "").strip()
        or os.getenv("SENTRY_ENVIRONMENT", "").strip()
        or "development"
    )


def _resolve_release() -> str | None:
    configured = (
        os.getenv("HERMES_SENTRY_RELEASE", "").strip()
        or os.getenv("SENTRY_RELEASE", "").strip()
    )
    if configured:
        return configured

    try:
        from hermes_cli import __version__

        return f"hermes-agent@{__version__}"
    except Exception:
        return None


def _resolve_traces_sample_rate() -> float:
    raw = (
        os.getenv("HERMES_SENTRY_TRACES_SAMPLE_RATE", "").strip()
        or os.getenv("SENTRY_TRACES_SAMPLE_RATE", "").strip()
    )
    if not raw:
        return 0.0
    try:
        rate = float(raw)
    except ValueError:
        logger.warning("Invalid Sentry traces sample rate %r; disabling traces", raw)
        return 0.0
    return min(max(rate, 0.0), 1.0)


def _before_send(event: dict, hint: dict) -> dict:
    del hint
    return _sanitize_event_value(event)


def _set_common_tags(component: str) -> None:
    if sentry_sdk is None:
        return

    sentry_sdk.set_tag("component", component)

    agent_name = os.getenv("AGENT_NAME", "").strip()
    if agent_name:
        sentry_sdk.set_tag("agent_name", agent_name)

    hostname = os.getenv("HOSTNAME", "").strip()
    if hostname:
        sentry_sdk.set_tag("hostname", hostname)


def init_sentry(component: str) -> bool:
    """Initialize Sentry once for the current process when a DSN is configured."""
    global _SENTRY_INITIALIZED

    dsn = _dsn()
    if not dsn:
        return False
    if sentry_sdk is None:
        logger.warning("Sentry DSN configured but sentry-sdk is not installed")
        return False

    os.environ.setdefault("HERMES_SENTRY_COMPONENT", component)

    with _INIT_LOCK:
        if _SENTRY_INITIALIZED:
            _set_common_tags(component)
            return True

        logging_integration = LoggingIntegration(
            level=logging.INFO,
            event_level=None,
        )
        sentry_sdk.init(
            dsn=dsn,
            environment=_resolve_environment(),
            release=_resolve_release(),
            send_default_pii=False,
            traces_sample_rate=_resolve_traces_sample_rate(),
            max_breadcrumbs=50,
            before_send=_before_send,
            integrations=[
                logging_integration,
                ThreadingIntegration(propagate_hub=True),
            ],
        )
        _set_common_tags(component)
        _SENTRY_INITIALIZED = True
        atexit.register(lambda: sentry_sdk.flush(timeout=2.0))
        logger.info("Sentry initialized for component=%s", component)
        return True


def capture_exception(
    error: BaseException,
    *,
    tags: Mapping[str, Any] | None = None,
    extras: Mapping[str, Any] | None = None,
    flush_timeout: float = 2.0,
) -> bool:
    """Capture a handled exception with redacted contextual metadata."""
    if sentry_sdk is None:
        return False
    if not _SENTRY_INITIALIZED and not init_sentry(os.getenv("HERMES_SENTRY_COMPONENT", "hermes")):
        return False

    with sentry_sdk.push_scope() as scope:
        for key, value in (tags or {}).items():
            scope.set_tag(str(key), sanitize_observability_text(value, limit=200) or "")
        for key, value in (extras or {}).items():
            scope.set_extra(str(key), _sanitize_event_value(value))
        sentry_sdk.capture_exception(error)

    sentry_sdk.flush(flush_timeout)
    return True


# ── Tracing helpers ──────────────────────────────────────────────────────

@contextlib.contextmanager
def start_transaction(
    *,
    op: str,
    name: str,
    tags: Mapping[str, str] | None = None,
    trace_parent: str | None = None,
    baggage: str | None = None,
) -> Generator[Any, None, None]:
    """Start a Sentry transaction (trace root).

    Yields the transaction object (or ``None`` when Sentry is unavailable)
    so callers can attach child spans or set additional data.

    When *trace_parent* (a ``sentry-trace`` header value) is provided, the
    transaction is created as a child of the remote trace — enabling
    distributed trace linking across subprocess boundaries.

    Usage::

        with start_transaction(op="hermes.chat", name="conversation-turn") as txn:
            ...  # txn may be None
    """
    if sentry_sdk is None or not _SENTRY_INITIALIZED:
        yield None
        return

    # Support distributed trace continuation — if a parent sentry-trace header
    # is provided, create the transaction as a child of that remote trace.
    continuation_kwargs: dict[str, Any] = {}
    if trace_parent:
        try:
            parent = sentry_sdk.continue_trace(
                environ_or_headers={"sentry-trace": trace_parent, **({"baggage": baggage} if baggage else {})},
                op=op,
                name=name,
            )
            if parent is not None:
                # continue_trace returns a Transaction — use it directly
                continuation_kwargs["transaction"] = parent
        except Exception:
            pass  # fall back to a fresh root transaction

    if "transaction" in continuation_kwargs:
        txn = continuation_kwargs["transaction"]
        with sentry_sdk.start_transaction(txn):
            for key, value in (tags or {}).items():
                txn.set_tag(str(key), str(value)[:200])
            yield txn
    else:
        with sentry_sdk.start_transaction(op=op, name=name) as txn:
            for key, value in (tags or {}).items():
                txn.set_tag(str(key), str(value)[:200])
            yield txn


def get_trace_headers() -> dict[str, str]:
    """Return the current span's ``sentry-trace`` and ``baggage`` headers.

    These should be injected into outbound HTTP requests or subprocess env
    vars so the downstream service can continue the same distributed trace.

    Returns an empty dict when Sentry is unavailable or no transaction is active.
    """
    if sentry_sdk is None or not _SENTRY_INITIALIZED:
        return {}

    scope = sentry_sdk.get_current_scope()
    span = scope.span if scope else None
    if span is None:
        return {}

    headers: dict[str, str] = {}
    try:
        trace_value = span.to_traceparent()
        if trace_value:
            headers["sentry-trace"] = trace_value
    except Exception:
        pass
    try:
        baggage_value = span.to_baggage()
        if baggage_value:
            headers["baggage"] = str(baggage_value)
    except Exception:
        pass
    return headers


@contextlib.contextmanager
def start_span(
    *,
    op: str,
    description: str | None = None,
    tags: Mapping[str, str] | None = None,
) -> Generator[Any, None, None]:
    """Start a Sentry child span on the current transaction.

    Safe to call even when there is no active transaction — yields ``None``
    and becomes a no-op.

    Usage::

        with start_span(op="hermes.tool", description="terminal") as span:
            ...  # span may be None
    """
    if sentry_sdk is None or not _SENTRY_INITIALIZED:
        yield None
        return

    with sentry_sdk.start_span(op=op, description=description) as span:
        for key, value in (tags or {}).items():
            span.set_tag(str(key), str(value)[:200])
        yield span