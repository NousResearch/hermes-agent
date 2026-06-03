"""Audit helpers for the Volt V2 tool adapter proof."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Mapping


AUDIT_SCHEMA_VERSION = 1
REQUIRED_AUDIT_KEYS = {
    "schema_version",
    "ts",
    "session_id",
    "task_id",
    "tool_call_id",
    "tool_name",
    "mode",
    "decision",
    "allowlisted",
    "duration_ms",
    "args_shape",
    "result_chars",
    "reason",
}
SENSITIVE_KEY_PARTS = (
    "api_key",
    "apikey",
    "authorization",
    "credential",
    "password",
    "secret",
    "token",
)
PATH_KEY_PARTS = ("path", "file", "dir", "root")
MAX_SHAPE_TEXT = 96


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def hash_value(value: Any) -> str:
    raw = str(value).encode("utf-8", errors="replace")
    return "sha256:" + hashlib.sha256(raw).hexdigest()[:16]


def redact_args(args: Mapping[str, Any] | None) -> Dict[str, Any]:
    """Return a JSON-safe argument shape without raw secrets or private paths."""
    if not isinstance(args, Mapping):
        return {}
    return {str(key): _redact_value(str(key), value) for key, value in args.items()}


def result_char_count(result: Any) -> int:
    if result is None:
        return 0
    if isinstance(result, str):
        return len(result)
    try:
        return len(json.dumps(result, ensure_ascii=False, default=str))
    except Exception:
        return len(str(result))


def build_audit_event(
    *,
    tool_name: str,
    args: Mapping[str, Any] | None,
    result: Any = None,
    mode: str,
    decision: str,
    allowlisted: bool,
    reason: str = "ok",
    task_id: str = "",
    session_id: str = "",
    tool_call_id: str = "",
    duration_ms: int | float | None = None,
) -> Dict[str, Any]:
    event = {
        "schema_version": AUDIT_SCHEMA_VERSION,
        "ts": utc_now_iso(),
        "session_id": session_id or "",
        "task_id": task_id or "",
        "tool_call_id": tool_call_id or "",
        "tool_name": tool_name or "",
        "mode": mode,
        "decision": decision,
        "allowlisted": bool(allowlisted),
        "duration_ms": _safe_duration(duration_ms),
        "args_shape": redact_args(args),
        "result_chars": result_char_count(result),
        "reason": reason,
    }
    validate_audit_event(event)
    return event


def validate_audit_event(event: Mapping[str, Any]) -> None:
    missing = REQUIRED_AUDIT_KEYS.difference(event.keys())
    if missing:
        raise ValueError(f"audit event missing required keys: {sorted(missing)}")
    if event.get("schema_version") != AUDIT_SCHEMA_VERSION:
        raise ValueError("unsupported Volt V2 adapter audit schema_version")
    if not isinstance(event.get("args_shape"), Mapping):
        raise ValueError("audit event args_shape must be an object")
    if not isinstance(event.get("allowlisted"), bool):
        raise ValueError("audit event allowlisted must be boolean")
    if not isinstance(event.get("duration_ms"), int):
        raise ValueError("audit event duration_ms must be integer")
    if not isinstance(event.get("result_chars"), int):
        raise ValueError("audit event result_chars must be integer")


def write_audit_event(audit_path: str | Path, event: Mapping[str, Any]) -> None:
    validate_audit_event(event)
    path = Path(audit_path).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(dict(event), ensure_ascii=False, sort_keys=True, default=str))
        fh.write("\n")


def read_jsonl_events(path: str | Path) -> list[dict[str, Any]]:
    p = Path(path)
    if not p.exists():
        return []
    events: list[dict[str, Any]] = []
    for line_number, line in enumerate(p.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        event = json.loads(line)
        validate_audit_event(event)
        events.append(event)
    return events


def _safe_duration(duration_ms: int | float | None) -> int:
    if duration_ms is None:
        return 0
    try:
        duration = int(duration_ms)
    except Exception:
        return 0
    return max(duration, 0)


def _redact_value(key: str, value: Any) -> Any:
    lowered = key.lower()
    if any(part in lowered for part in SENSITIVE_KEY_PARTS):
        return "<redacted>"
    if any(part in lowered for part in PATH_KEY_PARTS):
        return hash_value(value)
    if isinstance(value, Mapping):
        return {str(k): _redact_value(str(k), v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_redact_sequence_item(item) for item in value]
    if isinstance(value, (bool, int, float)) or value is None:
        return value
    text = str(value)
    if len(text) > MAX_SHAPE_TEXT:
        return f"<str:{len(text)} chars>"
    return text


def _redact_sequence_item(item: Any) -> Any:
    if isinstance(item, Mapping):
        return {str(k): _redact_value(str(k), v) for k, v in item.items()}
    if isinstance(item, (list, tuple, set)):
        return [_redact_sequence_item(v) for v in item]
    if isinstance(item, (bool, int, float)) or item is None:
        return item
    text = str(item)
    if "/" in text or "\\" in text:
        return hash_value(text)
    if len(text) > MAX_SHAPE_TEXT:
        return f"<str:{len(text)} chars>"
    return text
