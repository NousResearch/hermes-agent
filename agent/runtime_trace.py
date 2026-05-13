"""Safe runtime trace logging for Hermes agent routing decisions.

This module is intentionally tiny and best-effort: observability must never
change whether an agent run succeeds.  Events are JSONL so they can be tailed,
filtered, and exposed through tools without a database migration.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from hermes_constants import get_hermes_home

_TRACE_FILE_NAME = "runtime-trace.jsonl"
_REDACTED = "[REDACTED]"
_SECRET_KEY_PARTS = (
    "api_key",
    "apikey",
    "authorization",
    "auth_token",
    "password",
    "secret",
    "token",
)


def _trace_path() -> Path:
    return get_hermes_home() / "logs" / _TRACE_FILE_NAME


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _is_secret_key(key: str) -> bool:
    lowered = key.lower().replace("-", "_")
    return any(part in lowered for part in _SECRET_KEY_PARTS)


def _sanitize(value: Any) -> Any:
    if isinstance(value, dict):
        sanitized: dict[str, Any] = {}
        for key, item in value.items():
            key_str = str(key)
            sanitized[key_str] = _REDACTED if _is_secret_key(key_str) else _sanitize(item)
        return sanitized
    if isinstance(value, list):
        return [_sanitize(item) for item in value]
    if isinstance(value, tuple):
        return [_sanitize(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def emit_runtime_event(
    event: str,
    *,
    session_id: str | None = None,
    task_id: str | None = None,
    **data: Any,
) -> None:
    """Append a runtime trace event.

    This function is best-effort by design.  It catches all exceptions so
    debug logging cannot break the production runtime path.
    """
    try:
        payload = {
            "ts": _utc_now_iso(),
            "session_id": session_id or "default",
            "task_id": task_id or "",
            "event": str(event),
            "data": _sanitize(data),
        }
        path = _trace_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")
    except Exception:
        return


def _agent_name_from_event(event: dict[str, Any]) -> str | None:
    data = event.get("data")
    if not isinstance(data, dict):
        return None
    agent = data.get("agent")
    if isinstance(agent, dict) and isinstance(agent.get("name"), str):
        return agent["name"]
    if isinstance(data.get("agent_name"), str):
        return data["agent_name"]
    return None


def read_runtime_events(
    *,
    session_id: str | None = None,
    limit: int = 50,
    agent_name: str | None = None,
) -> list[dict[str, Any]]:
    """Read recent runtime trace events, newest last in the returned list."""
    try:
        max_events = max(1, min(int(limit), 500))
    except (TypeError, ValueError):
        max_events = 50

    path = _trace_path()
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except FileNotFoundError:
        return []
    except Exception:
        return []

    matched: list[dict[str, Any]] = []
    for line in reversed(lines):
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(event, dict):
            continue
        if session_id and event.get("session_id") != session_id:
            continue
        if agent_name and _agent_name_from_event(event) != agent_name:
            continue
        matched.append(event)
        if len(matched) >= max_events:
            break
    return list(reversed(matched))
