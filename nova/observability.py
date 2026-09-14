"""Operational logging — what NOVA did, for whoever is debugging it at 3am.

Distinct from the audit log, and the distinction is not pedantry. The audit log is a
*governance* record: it answers "what changed that a model can see, and who authorised it",
it is deliberately sparse, and it is sealed and verified. This is the other thing — the
boring, high-volume account of what the process was doing, which is useless for compliance
and indispensable for diagnosis.

Before this, NOVA had one logger, in the control server. Everything else — applying a
bundle, materializing an agent, ingesting a corpus, submitting work — was silent. On a
customer's host that means an operator investigating a slow apply has the audit log, which
records only the model-visible outcome, and nothing about the run that produced it.

**JSON lines, to stderr, off by default.** JSON because the destination is a log shipper —
CloudWatch, journald, a sidecar — and a human reading it directly is the rarer case that
``--log-format=text`` serves. Off by default because a CLI that prints structured logs over
its own output is hostile to the person running it interactively.

**The correlation id is the join.** It is the same id the audit log stamps on every event of
one operator action, so an operational trace and a governance record can be lined up
without guessing from timestamps.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from contextvars import ContextVar
from typing import Any, Optional

LOGGER_NAME = "nova"

#: Set for the duration of one operator action, so every line it produces carries the same
#: id as the audit events it writes.
_CORRELATION: ContextVar[str] = ContextVar("nova_correlation_id", default="")

#: Caller fields travel under one key rather than being splatted into ``extra``.
#:
#: ``logging.makeRecord`` raises ``KeyError`` when ``extra`` names an attribute a
#: ``LogRecord`` already has — and the collisions are ordinary words a caller would
#: naturally use: ``created``, ``message``, ``module``, ``name``, ``args``, ``filename``.
#: An apply logging ``created=2`` is not an exotic case; it is the obvious one, and it
#: crashed. Nesting makes the collision impossible rather than remembering a deny-list.
_FIELDS = "nova_fields"

#: Keys the log envelope owns. A caller field colliding with one is renamed, not dropped.
_ENVELOPE = frozenset({"ts", "level", "logger", "message", "correlation_id", "error"})


def set_correlation_id(value: str) -> None:
    _CORRELATION.set(value or "")


def correlation_id() -> str:
    return _CORRELATION.get()


class JsonFormatter(logging.Formatter):
    """One JSON object per line, with the caller's structured fields merged in.

    Never raises. A formatter that can fail takes the logs down at exactly the moment
    someone needs them, so an unserialisable value is stringified rather than propagated.
    """

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(record.created))
            + f".{int(record.msecs):03d}Z",
            "level": record.levelname.lower(),
            "logger": record.name,
            "message": record.getMessage(),
        }
        cid = getattr(record, "correlation_id", "") or correlation_id()
        if cid:
            payload["correlation_id"] = cid
        for key, value in (getattr(record, _FIELDS, None) or {}).items():
            # The envelope wins. A caller field named `message` or `level` would otherwise
            # silently replace the log line's own — losing the thing the reader came for,
            # with nothing to indicate it happened. Renamed rather than dropped, because
            # the caller passed it for a reason.
            safe = f"field_{key}" if key in _ENVELOPE else key
            payload[safe] = value if _serialisable(value) else repr(value)
        if record.exc_info:
            payload["error"] = self.formatException(record.exc_info)
        try:
            return json.dumps(payload, sort_keys=True, default=str)
        except (TypeError, ValueError):
            return json.dumps(
                {"ts": payload["ts"], "level": "error", "logger": record.name,
                 "message": "a log record could not be serialised"}
            )


def _serialisable(value: Any) -> bool:
    return isinstance(value, (str, int, float, bool, type(None), list, dict, tuple))


class TextFormatter(logging.Formatter):
    """For a human watching a terminal. The same fields, laid out to be read."""

    def format(self, record: logging.LogRecord) -> str:
        cid = getattr(record, "correlation_id", "") or correlation_id()
        extras = " ".join(
            f"{key}={value}"
            for key, value in sorted((getattr(record, _FIELDS, None) or {}).items())
        )
        parts = [f"{record.levelname.lower():7}", record.getMessage()]
        if extras:
            parts.append(extras)
        if cid:
            parts.append(f"[{cid[:8]}]")
        return "  ".join(parts)


def configure(level: str = "", fmt: str = "") -> logging.Logger:
    """Set up NOVA's logger. Reads ``NOVA_LOG_LEVEL`` and ``NOVA_LOG_FORMAT`` when unset.

    Idempotent, and never attaches a second handler: called from both the CLI and a long
    running control plane, and duplicate handlers mean duplicate lines, which is how a log
    stops being trusted.

    Logging goes to **stderr**, always, so a machine-readable log can never interleave with
    the JSON a caller asked for on stdout.
    """
    level = (level or os.environ.get("NOVA_LOG_LEVEL", "")).upper()
    fmt = (fmt or os.environ.get("NOVA_LOG_FORMAT", "json")).lower()

    logger = logging.getLogger(LOGGER_NAME)
    for handler in list(logger.handlers):
        logger.removeHandler(handler)

    if not level or level in ("OFF", "NONE"):
        # Silent by default. A CLI that prints structured logs over its own output is
        # hostile to the person running it, and this must be opt-in.
        logger.addHandler(logging.NullHandler())
        logger.setLevel(logging.CRITICAL + 1)
        return logger

    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(TextFormatter() if fmt == "text" else JsonFormatter())
    logger.addHandler(handler)
    logger.setLevel(getattr(logging, level, logging.INFO))
    logger.propagate = False
    return logger


def get(name: str = "") -> logging.Logger:
    """A NOVA logger. ``get("apply")`` is ``nova.apply``."""
    return logging.getLogger(f"{LOGGER_NAME}.{name}" if name else LOGGER_NAME)


def log(logger: logging.Logger, level: int, message: str, **fields: Any) -> None:
    """Emit ``message`` with structured ``fields``, collision-free.

    The one supported way to attach fields. Passing them straight to ``extra`` is what
    raised on ``created``, so nothing in NOVA should do it directly.
    """
    logger.log(level, message, extra={_FIELDS: dict(fields)})


class operation:
    """Log the start, end and duration of one operation, and never swallow its failure.

    A context manager because the duration and the outcome are the parts worth having and
    the parts a caller most often forgets to record. An exception is logged and re-raised —
    this observes, it does not handle.
    """

    def __init__(self, name: str, **fields: Any) -> None:
        self.name = name
        self.fields = fields
        self._started = 0.0

    def __enter__(self) -> "operation":
        self._started = time.monotonic()
        get(self.name).info("%s started", self.name, extra={_FIELDS: dict(self.fields)})
        return self

    def add(self, **fields: Any) -> None:
        """Record something discovered while the operation ran."""
        self.fields.update(fields)

    def __exit__(self, exc_type, exc, traceback) -> bool:
        elapsed = round((time.monotonic() - self._started) * 1000)
        fields = {**self.fields, "duration_ms": elapsed}
        logger = get(self.name)
        if exc is not None:
            logger.error(
                "%s failed",
                self.name,
                extra={_FIELDS: {**fields, "error_type": exc_type.__name__}},
            )
        else:
            logger.info("%s finished", self.name, extra={_FIELDS: fields})
        return False
