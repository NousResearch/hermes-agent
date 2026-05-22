"""Audit helpers for the Volt V2 tool adapter proof."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping


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
        "ts": utc_now_iso(),
        "session_id": session_id or "",
        "task_id": task_id or "",
        "tool_call_id": tool_call_id or "",
        "tool_name": tool_name or "",
        "mode": mode,
        "decision": decision,
        "allowlisted": bool(allowlisted),
        "duration_ms": duration_ms if duration_ms is not None else 0,
        "args_shape": redact_args(args),
        "result_chars": result_char_count(result),
        "reason": reason,
    }
    return event


def write_audit_event(audit_path: str | Path, event: Mapping[str, Any]) -> None:
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
    for line in p.read_text(encoding="utf-8").splitlines():
        if line.strip():
            events.append(json.loads(line))
    return events


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
