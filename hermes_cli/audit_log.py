"""Private, redacted JSONL audit logging for Hermes security decisions."""

from __future__ import annotations

import hashlib
import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from hermes_constants import get_hermes_home
from hermes_cli.private_artifacts import private_text_writer

logger = logging.getLogger(__name__)

AUDIT_SCHEMA_VERSION = 1
AUDIT_REDACTION_VERSION = 1
MAX_AUDIT_TEXT_CHARS = 400

_SECRET_KEY_PARTS = (
    "api_key",
    "apikey",
    "authorization",
    "auth_token",
    "bearer",
    "client_secret",
    "credential",
    "key_material",
    "password",
    "private_key",
    "raw_secret",
    "refresh_token",
    "secret",
    "token",
)
_PRIVATE_PAYLOAD_KEYS = {
    "messages",
    "prompt",
    "raw_command_output",
    "raw_env",
    "request_body",
    "response_body",
}
_IDENTIFIER_KEYS = {
    "actor_id",
    "chat_id",
    "session_id",
    "session_key",
    "thread_id",
    "user_id",
}


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


def _format_timestamp(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _audit_root(audit_dir: str | Path | None = None) -> Path:
    if audit_dir is not None:
        return Path(audit_dir).expanduser()
    override = os.getenv("HERMES_AUDIT_DIR", "").strip()
    if override:
        return Path(override).expanduser()
    return get_hermes_home() / "audit"


def _hash_identifier(value: Any) -> dict[str, Any]:
    text = "" if value is None else str(value)
    if not text:
        return {"present": False}
    digest = hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()
    return {
        "present": True,
        "sha256_12": digest[:12],
        "length": len(text),
    }


def _truncate_text(value: str) -> str:
    if len(value) <= MAX_AUDIT_TEXT_CHARS:
        return value
    return f"{value[:MAX_AUDIT_TEXT_CHARS]}...[truncated]"


def _is_secret_key(key: str) -> bool:
    normalized = key.lower().replace("-", "_")
    return any(part in normalized for part in _SECRET_KEY_PARTS)


def _redact_text(value: Any) -> str:
    from agent.redact import redact_sensitive_text

    text = "" if value is None else str(value)
    return _truncate_text(redact_sensitive_text(text, force=True))


def redact_audit_value(value: Any, *, key: str | None = None) -> Any:
    """Return a JSON-safe value for the audit ledger.

    The audit ledger is an evidence trail, not a data lake. Keep strings short,
    force secret redaction regardless of global logging settings, hash private
    identifiers, and drop known raw payload fields.
    """
    normalized_key = (key or "").lower()
    if normalized_key in _PRIVATE_PAYLOAD_KEYS:
        return "[REDACTED_PAYLOAD]"
    if normalized_key in _IDENTIFIER_KEYS:
        return _hash_identifier(value)
    if key and _is_secret_key(key):
        return "[REDACTED_SECRET]"

    if isinstance(value, dict):
        return {
            str(item_key): redact_audit_value(item_value, key=str(item_key))
            for item_key, item_value in value.items()
        }
    if isinstance(value, (list, tuple, set)):
        return [redact_audit_value(item) for item in value]
    if isinstance(value, (str, bytes)):
        if isinstance(value, bytes):
            value = value.decode("utf-8", errors="replace")
        return _redact_text(value)
    if value is None or isinstance(value, (bool, int, float)):
        return value
    return _redact_text(value)


def append_audit_event(
    event: dict[str, Any],
    *,
    audit_dir: str | Path | None = None,
    now: datetime | None = None,
) -> Path:
    """Append one redacted audit event to the private daily JSONL ledger."""
    if not isinstance(event, dict):
        raise TypeError("audit event must be a dict")

    timestamp = now or _now_utc()
    payload = redact_audit_value(dict(event))
    payload.setdefault("schema_version", AUDIT_SCHEMA_VERSION)
    payload.setdefault("timestamp", _format_timestamp(timestamp))
    payload.setdefault("redaction", {
        "status": "redacted",
        "version": AUDIT_REDACTION_VERSION,
    })

    root = _audit_root(audit_dir)
    target = root / f"{timestamp.astimezone(timezone.utc).date().isoformat()}.jsonl"
    line = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    with private_text_writer(target, append=True) as handle:
        handle.write(line)
        handle.write("\n")
    return target


def safe_append_audit_event(event: dict[str, Any], **kwargs: Any) -> Path | None:
    """Best-effort audit append that never changes the caller's control flow."""
    try:
        return append_audit_event(event, **kwargs)
    except Exception as exc:  # pragma: no cover - defensive logging only
        logger.debug("Audit event write failed: %s", exc, exc_info=True)
        return None


def record_approval_audit_event(
    *,
    event_type: str = "approval.decision",
    decision: str,
    command: str | None = None,
    description: str | None = None,
    pattern_key: str | None = None,
    pattern_keys: list[str] | None = None,
    session_key: str | None = None,
    surface: str | None = None,
    risk_tier: str = "R3",
    approval_scope: str | None = None,
    status: str | None = None,
    reason: str | None = None,
    choice: str | None = None,
    extra: dict[str, Any] | None = None,
) -> Path | None:
    """Record a structured approval/confirmation decision event."""
    event: dict[str, Any] = {
        "event_type": event_type,
        "category": "approval",
        "decision": decision,
        "surface": surface or "unknown",
        "risk_tier": risk_tier,
        "status": status or decision,
        "redaction_status": "redacted",
    }
    if command is not None:
        event["command"] = {"preview": command}
    if description is not None:
        event["description"] = description
    if pattern_key is not None:
        event["pattern_key"] = pattern_key
    if pattern_keys is not None:
        event["pattern_keys"] = list(pattern_keys)
    if session_key is not None:
        event["session_key"] = session_key
    if approval_scope is not None:
        event["approval_scope"] = approval_scope
    if choice is not None:
        event["choice"] = choice
    if reason is not None:
        event["reason"] = reason
    if extra:
        event["extra"] = dict(extra)
    return safe_append_audit_event(event)
