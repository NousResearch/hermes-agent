"""Durable context handoff snapshots for compression recovery.

The handoff file is a best-effort local safety rail: before lossy compression,
or when compression aborts/fails, Hermes records enough prompt-safe state for a
fresh session to verify where work left off instead of trusting a possibly
truncated compression summary.
"""

from __future__ import annotations

import json
import logging
import os
import re
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from hermes_constants import get_hermes_home

logger = logging.getLogger(__name__)

_MAX_TAIL_MESSAGES = 24
_MAX_CONTENT_CHARS = 4_000
_SECRET_ASSIGNMENT_RE = re.compile(
    r"(?i)\b([A-Z0-9_]*(?:API[_-]?KEY|TOKEN|SECRET|PASSWORD|PASS|CREDENTIAL|PRIVATE[_-]?KEY)[A-Z0-9_]*)\s*[:=]\s*([^\s\n]+)"
)
_BEARER_RE = re.compile(r"(?i)\b(bearer\s+)[A-Za-z0-9._\-]{12,}")


@dataclass(frozen=True)
class ContextHandoffResult:
    """Paths written by :func:`write_context_handoff`."""

    json_path: Path
    markdown_path: Path


def _redact_text(value: str) -> str:
    """Return text with obvious credential assignments redacted."""
    redacted = _SECRET_ASSIGNMENT_RE.sub(lambda m: f"{m.group(1)}=[REDACTED]", value)
    return _BEARER_RE.sub(lambda m: f"{m.group(1)}[REDACTED]", redacted)


def _json_safe(value: Any) -> Any:
    """Coerce arbitrary message payloads into JSON-safe, redacted values."""
    if isinstance(value, str):
        text = _redact_text(value)
        if len(text) > _MAX_CONTENT_CHARS:
            return text[:_MAX_CONTENT_CHARS] + "…[truncated]"
        return text
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value[:50]]
    if isinstance(value, (int, float, bool)) or value is None:
        return value
    return _redact_text(str(value))


def _message_tail(messages: list[dict[str, Any]] | list[Any]) -> list[dict[str, Any]]:
    """Return the redacted tail of the current conversation messages."""
    tail: list[dict[str, Any]] = []
    for raw in list(messages or [])[-_MAX_TAIL_MESSAGES:]:
        if isinstance(raw, dict):
            role = str(raw.get("role") or "unknown")
            content = _json_safe(raw.get("content"))
            item: dict[str, Any] = {"role": role, "content": content}
            if raw.get("name"):
                item["name"] = _json_safe(raw.get("name"))
            if raw.get("tool_call_id"):
                item["tool_call_id"] = _json_safe(raw.get("tool_call_id"))
            tail.append(item)
        else:
            tail.append({"role": "unknown", "content": _json_safe(raw)})
    return tail


def _todo_snapshot(agent: Any) -> str:
    """Read the current todo snapshot if the agent exposes one."""
    store = getattr(agent, "_todo_store", None)
    if store is None or not hasattr(store, "format_for_injection"):
        return ""
    try:
        return _redact_text(str(store.format_for_injection() or ""))
    except Exception:
        return ""


def _atomic_write(path: Path, content: str) -> None:
    """Atomically replace ``path`` with ``content``."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        Path(tmp_name).replace(path)
    except Exception:
        try:
            Path(tmp_name).unlink(missing_ok=True)
        finally:
            raise


def _render_markdown(payload: dict[str, Any]) -> str:
    """Render a compact human-readable handoff note."""
    lines = [
        "# Hermes Context Handoff",
        "",
        f"- Reason: `{payload.get('reason') or ''}`",
        f"- Session: `{payload.get('session_id') or ''}`",
        f"- Created: `{payload.get('created_at') or ''}`",
        f"- Model: `{payload.get('model') or ''}`",
        f"- Platform: `{payload.get('platform') or ''}`",
    ]
    if payload.get("approx_tokens") is not None:
        lines.append(f"- Approx tokens: `{payload['approx_tokens']}`")
    if payload.get("focus_topic"):
        lines.append(f"- Focus: `{payload['focus_topic']}`")
    if payload.get("error"):
        lines.append(f"- Error: `{payload['error']}`")
    if payload.get("todo_snapshot"):
        lines.extend([
            "",
            "## Todo snapshot",
            "",
            "```",
            payload["todo_snapshot"],
            "```",
        ])
    lines.extend(["", "## Recent messages", ""])
    for msg in payload.get("messages_tail", []):
        role = msg.get("role", "unknown")
        content = msg.get("content", "")
        if not isinstance(content, str):
            content = json.dumps(content, ensure_ascii=False)
        lines.extend([f"### {role}", "", content, ""])
    lines.extend([
        "## Resume discipline",
        "",
        "新会话读取本文件后，先真实校验 git / PR / CI / 部署状态，再继续执行；不要只相信压缩摘要。",
        "",
    ])
    return "\n".join(lines)


def write_context_handoff(
    agent: Any,
    messages: list[dict[str, Any]] | list[Any],
    *,
    reason: str,
    approx_tokens: int | None = None,
    focus_topic: str | None = None,
    error: str | None = None,
) -> ContextHandoffResult | None:
    """Persist a profile-scoped context handoff snapshot.

    The function is intentionally best-effort and returns ``None`` on failure so
    compression behavior never regresses because the local safety file could not
    be written.
    """
    try:
        handoff_dir = get_hermes_home() / "handoffs"
        json_path = handoff_dir / "latest.json"
        markdown_path = handoff_dir / "latest.md"
        payload: dict[str, Any] = {
            "schema_version": 1,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "reason": reason,
            "session_id": str(getattr(agent, "session_id", "") or ""),
            "model": str(getattr(agent, "model", "") or ""),
            "provider": str(getattr(agent, "provider", "") or ""),
            "platform": str(getattr(agent, "platform", "") or ""),
            "gateway_session_key": str(
                getattr(agent, "_gateway_session_key", "") or ""
            ),
            "turn_id": str(getattr(agent, "_current_turn_id", "") or ""),
            "approx_tokens": approx_tokens,
            "focus_topic": _redact_text(focus_topic or "") or None,
            "error": _redact_text(error or "") or None,
            "todo_snapshot": _todo_snapshot(agent),
            "messages_tail": _message_tail(messages),
        }
        json_content = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True)
        _atomic_write(json_path, json_content + "\n")
        _atomic_write(markdown_path, _render_markdown(payload))
        logger.info(
            "context handoff written: session=%s reason=%s path=%s",
            payload["session_id"] or "none",
            reason,
            json_path,
        )
        return ContextHandoffResult(json_path=json_path, markdown_path=markdown_path)
    except Exception as exc:
        logger.warning("context handoff write failed: %s", exc)
        return None
