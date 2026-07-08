"""Minimal local telemetry/event sink for Hermes Agent.

This module intentionally starts small: privacy-preserving JSONL events under
``$HERMES_HOME/ops/events/YYYY-MM-DD.jsonl``.  It is designed so future OTLP
exporters can tail or translate the same schema without requiring a collector
for local installs.
"""

from __future__ import annotations

import hashlib
import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from hermes_constants import get_hermes_home

logger = logging.getLogger(__name__)

SCHEMA_VERSION = 1
_MAX_STRING_LEN = 240
_SENSITIVE_KEY_FRAGMENTS = (
    "api_key",
    "apikey",
    "authorization",
    "completion",
    "content",
    "cookie",
    "password",
    "prompt",
    "secret",
    "token",
)


def stable_hash(value: Any) -> str | None:
    """Return a stable short SHA-256 hash for IDs without storing raw values."""
    if value is None:
        return None
    text = str(value)
    if not text:
        return None
    digest = hashlib.sha256(f"hermes-telemetry-v1:{text}".encode("utf-8")).hexdigest()
    return digest[:16]


def error_fingerprint(error: Any) -> str | None:
    """Hash an error string/object so telemetry can group failures safely."""
    if error is None:
        return None
    return stable_hash(f"{type(error).__name__}:{str(error)}")


def events_dir(hermes_home: str | Path | None = None) -> Path:
    """Resolve the JSONL event directory for the active Hermes profile."""
    home = Path(hermes_home) if hermes_home is not None else get_hermes_home()
    return home / "ops" / "events"


def event_path(ts: datetime | None = None, hermes_home: str | Path | None = None) -> Path:
    """Return the daily JSONL path for ``ts`` in the active Hermes profile."""
    when = ts or datetime.now(UTC)
    return events_dir(hermes_home) / f"{when.date().isoformat()}.jsonl"


def _safe_value(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, str):
        return value if len(value) <= _MAX_STRING_LEN else value[:_MAX_STRING_LEN] + "…"
    if isinstance(value, dict):
        safe: dict[str, Any] = {}
        for key, item in value.items():
            key_text = str(key)
            key_lc = key_text.lower()
            if any(fragment in key_lc for fragment in _SENSITIVE_KEY_FRAGMENTS):
                safe[key_text] = "[REDACTED]"
            else:
                safe[key_text] = _safe_value(item)
        return safe
    if isinstance(value, (list, tuple, set)):
        return [_safe_value(item) for item in list(value)[:50]]
    return str(value)[:_MAX_STRING_LEN]


def build_event(
    event_type: str,
    payload: dict[str, Any] | None = None,
    *,
    status: str = "ok",
    source: str = "hermes",
    ts: datetime | None = None,
) -> dict[str, Any]:
    """Build a privacy-preserving telemetry event dictionary."""
    when = ts or datetime.now(UTC)
    return {
        "schema_version": SCHEMA_VERSION,
        "event_id": str(uuid4()),
        "timestamp": when.isoformat().replace("+00:00", "Z"),
        "event_type": str(event_type),
        "source": str(source),
        "status": str(status),
        "payload": _safe_value(payload or {}),
    }


def emit_event(
    event_type: str,
    payload: dict[str, Any] | None = None,
    *,
    status: str = "ok",
    source: str = "hermes",
    ts: datetime | None = None,
    hermes_home: str | Path | None = None,
) -> Path:
    """Append one JSONL event and return the file path.

    This function may raise filesystem/serialization errors. Runtime callers
    that must never fail because of telemetry should use ``safe_emit_event``.
    """
    event = build_event(event_type, payload, status=status, source=source, ts=ts)
    path = event_path(ts, hermes_home)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(event, ensure_ascii=False, separators=(",", ":")) + "\n")
    return path


def safe_emit_event(
    event_type: str,
    payload: dict[str, Any] | None = None,
    *,
    status: str = "ok",
    source: str = "hermes",
    ts: datetime | None = None,
    hermes_home: str | Path | None = None,
) -> Path | None:
    """Best-effort event emission for hot paths.

    Telemetry must never break cron, gateway, tools, or model calls.  Failures
    are logged at debug level and otherwise ignored.
    """
    try:
        return emit_event(
            event_type,
            payload,
            status=status,
            source=source,
            ts=ts,
            hermes_home=hermes_home,
        )
    except Exception as exc:  # pragma: no cover - defensive by design
        logger.debug("Failed to emit Hermes telemetry event %s: %s", event_type, exc)
        return None
