"""
Structured Logging for Hermes-Agent.

Provides:
- HermesJSONFormatter: JSON-formatted log output
- HermesStructuredLogHandler: Rotating file handler that outputs structured JSON
- StructuredLoggerAdapter: LoggerAdapter that injects session_id / trace_id
- EventBus integration: emits structured log events to the analytics EventBus

Follows the OS (Observability) design principle — all log records carry
session_id and trace_id for correlation.

Usage:
    from agent.hermes.structured_logging import (
        get_structured_logger,
        HermesStructuredLogHandler,
        HermesJSONFormatter,
    )

    logger = get_structured_logger("my_component", session_id="sess-abc", trace_id="trace-123")
    logger.info("hello", extra={"tool": "bash"})
"""

import json
import logging
import threading
import uuid
from datetime import datetime, timezone
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any, Dict, Optional

from agent.hermes.analytics import Event, EventType

# Global context: thread-safe session/trace IDs
class _LogContext:
    """Thread-safe log context holding session_id and trace_id per thread."""

    def __init__(self) -> None:
        self._local = threading.local()

    def set(self, session_id: str = "", trace_id: str = "") -> None:
        self._local.session_id = session_id
        self._local.trace_id = trace_id or self._generate_trace_id()

    def get_session_id(self) -> str:
        return getattr(self._local, "session_id", "")

    def get_trace_id(self) -> str:
        return getattr(self._local, "trace_id", "")

    def _generate_trace_id(self) -> str:
        return uuid.uuid4().hex[:16]

    def clear(self) -> None:
        self._local.session_id = ""
        self._local.trace_id = ""


_log_context = _LogContext()


def set_log_context(session_id: str = "", trace_id: str = "") -> None:
    """Set the current thread's log context (session_id + trace_id)."""
    _log_context.set(session_id=session_id, trace_id=trace_id)


def get_log_context() -> tuple[str, str]:
    """Return (session_id, trace_id) for the current thread."""
    return _log_context.get_session_id(), _log_context.get_trace_id()


def clear_log_context() -> None:
    """Clear the current thread's log context."""
    _log_context.clear()


# ---------------------------------------------------------------------------
# JSON Formatter
# ---------------------------------------------------------------------------

class HermesJSONFormatter(logging.Formatter):
    """
    JSON log formatter for Hermes-Agent.

    Each log record is serialized to a single JSON line:
        {
          "timestamp": "2026-04-11T10:23:45.123Z",
          "level": "INFO",
          "logger": "my.component",
          "message": "hello world",
          "session_id": "sess-abc",
          "trace_id": "trace-123",
          "module": "my.component",
          "function": "run",
          "line": 42,
          "extra": { ... }
        }

    Optionally accepts a redaction function to scrub secrets.
    """

    def __init__(
        self,
        *,
        redaction_fn: Optional[callable] = None,
        include_extra: bool = True,
    ) -> None:
        super().__init__()
        self._redact = redaction_fn or (lambda s: s)
        self._include_extra = include_extra

    def format(self, record: logging.LogRecord) -> str:
        session_id = getattr(record, "session_id", "") or _log_context.get_session_id()
        trace_id = getattr(record, "trace_id", "") or _log_context.get_trace_id()

        payload: Dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": self._redact(str(record.getMessage())),
            "session_id": session_id,
            "trace_id": trace_id,
        }

        if record.filename:
            payload["module"] = record.filename
        if record.funcName:
            payload["function"] = record.funcName
        if record.lineno and record.lineno > 0:
            payload["line"] = record.lineno

        if self._include_extra:
            extra = {
                k: v for k, v in record.__dict__.items()
                if k not in logging.LogRecord(
                    "", 0, "", 0, "", (), None
               ).__dict__ and not k.startswith("_")
            }
            if extra:
                payload["extra"] = {k: self._redact(str(v)) for k, v in extra.items()}

        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)

        try:
            return json.dumps(payload, default=str)
        except (TypeError, ValueError):
            payload["message"] = self._redact("<unserializable message>")
            return json.dumps(payload, default=str)


# ---------------------------------------------------------------------------
# Structured Log Handler
# ---------------------------------------------------------------------------

class HermesStructuredLogHandler(RotatingFileHandler):
    """
    Rotating file handler that outputs structured JSON log lines.

    Thread-safe via RotatingFileHandler's built-in locking.
    Integrates with the analytics EventBus to emit log events.
    """

    def __init__(
        self,
        filename: str | Path,
        *,
        event_bus: Optional["EventBus"] = None,
        session_id: str = "",
        trace_id: str = "",
        max_bytes: int = 5 * 1024 * 1024,
        backup_count: int = 3,
        encoding: str = "utf-8",
    ) -> None:
        super().__init__(
            str(filename),
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding=encoding,
        )
        self._event_bus = event_bus
        self._handler_session_id = session_id
        self._handler_trace_id = trace_id
        self.setFormatter(HermesJSONFormatter())

    def emit(self, record: logging.LogRecord) -> None:
        """Emit a structured JSON log line and optionally to EventBus."""
        try:
            super().emit(record)
        except Exception:
            self.handleError(record)

        # Emit to EventBus if configured
        if self._event_bus is not None:
            try:
                session_id = getattr(record, "session_id", "") or self._handler_session_id
                trace_id = getattr(record, "trace_id", "") or self._handler_trace_id

                event = Event(
                    type=EventType.TOOL_RESULT,
                    payload={
                        "logger": record.name,
                        "level": record.levelname,
                        "message": record.getMessage(),
                        "module": record.filename or "",
                        "function": record.funcName or "",
                        "line": record.lineno or 0,
                    },
                    session_id=session_id,
                )
                self._event_bus.emit(event)
            except Exception:
                pass  # fire-and-forget


# ---------------------------------------------------------------------------
# Structured Logger Adapter
# ---------------------------------------------------------------------------

class StructuredLoggerAdapter(logging.LoggerAdapter):
    """
    LoggerAdapter that injects session_id and trace_id into every log record.

    The session_id and trace_id are sourced from the current thread's log context
    (see set_log_context / get_log_context), or can be overridden per-logger
    via the ``extra`` kwarg passed to ``log()`` / ``debug()`` / etc.

    Example::

        set_log_context(session_id="sess-abc", trace_id="trace-xyz")
        logger = get_structured_logger(__name__)
        logger.info("hello")   # record.session_id="sess-abc", record.trace_id="trace-xyz"

        # Override per-call:
        logger.info("custom trace", extra={"trace_id": "custom-trace"})
    """

    def process(self, msg: str, kwargs: Dict[str, Any]) -> tuple[str, Dict[str, Any]]:
        extra = kwargs.get("extra", {})

        # Per-call overrides take precedence over thread context
        session_id = extra.get("session_id") or _log_context.get_session_id()
        trace_id = extra.get("trace_id") or _log_context.get_trace_id()

        # Merge into a fresh extra dict so we don't mutate caller's dict
        merged_extra = {**extra, "session_id": session_id, "trace_id": trace_id}
        kwargs["extra"] = merged_extra
        return msg, kwargs


def get_structured_logger(
    name: str,
    *,
    session_id: str = "",
    trace_id: str = "",
) -> StructuredLoggerAdapter:
    """
    Create a StructuredLoggerAdapter for *name*.

    The adapter injects session_id / trace_id into every log record.
    Thread-safe — each call to log() reads the current thread context.

    Args:
        name: Logger name (typically ``__name__``)
        session_id: Optional session ID override (thread context is used if empty)
        trace_id: Optional trace ID override (auto-generated if empty)

    Returns:
        A StructuredLoggerAdapter that wraps ``logging.getLogger(name)``.
    """
    if session_id:
        _log_context.set(session_id=session_id, trace_id=trace_id)

    base = logging.getLogger(name)
    return StructuredLoggerAdapter(base, {})


# ---------------------------------------------------------------------------
# Setup helper
# ---------------------------------------------------------------------------

def setup_structured_logging(
    log_dir: Path,
    *,
    event_bus: Optional["EventBus"] = None,
    session_id: str = "",
    trace_id: str = "",
    log_level: int = logging.INFO,
    max_bytes: int = 5 * 1024 * 1024,
    backup_count: int = 3,
) -> None:
    """
    Add a structured JSON log handler to the root logger.

    Logs are written to ``log_dir / "structured.jsonl"`` in JSON lines format.

    If *event_bus* is provided, log events are also emitted to it.

    Args:
        log_dir: Directory for the log file
        event_bus: Optional EventBus instance for analytics integration
        session_id: Default session ID for this handler
        trace_id: Default trace ID for this handler
        log_level: Minimum log level (default: INFO)
        max_bytes: Max file size before rotation
        backup_count: Number of backup files to keep
    """
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    handler = HermesStructuredLogHandler(
        log_dir / "structured.jsonl",
        event_bus=event_bus,
        session_id=session_id,
        trace_id=trace_id,
        max_bytes=max_bytes,
        backup_count=backup_count,
    )
    handler.setLevel(log_level)

    root = logging.getLogger()
    root.addHandler(handler)

    if root.level == logging.NOTSET or root.level > log_level:
        root.setLevel(log_level)
