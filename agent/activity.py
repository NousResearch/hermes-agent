"""Small runtime activity helpers for gateway visibility surfaces.

Presentation-only state. Nothing here mutates conversation history, tools, or
system prompts; it only helps gateway heartbeats explain what the running turn is
currently doing.
"""

from __future__ import annotations

import re
import time
from typing import Any, Iterable, Optional

from agent.display import (
    build_tool_label,
    redact_tool_args_for_display,
)
from agent.redact import redact_sensitive_text

_TOOL_PREVIEW_MAX = 140
_TODO_CONTENT_MAX = 100
_CONCURRENT_TOOL_PREVIEW_MAX = 80
_ACTION_HISTORY_MAX = 3
_SECRET_FLAG_RE = re.compile(
    r"(?ix)"
    r"(?P<prefix>(?:^|\s)"
    r"(?:--?(?:password|passwd|pass|token|secret|api[-_]?key|auth[-_]?token|access[-_]?token|refresh[-_]?token|client[-_]?secret|private[-_]?key))"
    r"(?:\s+|=))"
    r"(?P<quote>['\"]?)"
    r"(?P<value>[^\s'\"]+)"
    r"(?P=quote)"
)
_MYSQL_SHORT_PASSWORD_RE = re.compile(r"(?i)(?P<prefix>(?:^|\s)-p)(?P<value>\S+)")
_SECRET_ASSIGN_RE = re.compile(
    r"(?ix)"
    r"(?P<prefix>\b(?:[A-Za-z0-9_.-]*(?:password|passwd|token|secret|api[-_.]?key|auth)[A-Za-z0-9_.-]*|MYSQL_PWD|PGPASSWORD)\s*=\s*)"
    r"(?P<quote>['\"]?)"
    r"(?P<value>[^\s&;'\"]+)"
    r"(?P=quote)"
)


def _oneline(value: Any) -> str:
    return " ".join(str(value or "").split())


def _truncate(value: Any, max_len: int) -> str:
    text = _oneline(value)
    if max_len <= 0 or len(text) <= max_len:
        return text
    if max_len <= 1:
        return text[:max_len]
    return text[: max_len - 1].rstrip() + "…"


def _redact_preview_text(text: str) -> str:
    """Redact secrets from gateway-visible activity labels.

    ``redact_tool_args_for_display`` intentionally preserves most normal typed
    text for CLI/debug surfaces. The long-running gateway heartbeat is more
    public: it edits a chat bubble even when detailed tool-progress is disabled,
    so it gets a stricter final scrub.
    """
    redacted = redact_sensitive_text(text, force=True)
    redacted = _SECRET_FLAG_RE.sub(lambda m: f"{m.group('prefix')}***", redacted)
    redacted = _MYSQL_SHORT_PASSWORD_RE.sub(lambda m: f"{m.group('prefix')}***", redacted)
    redacted = _SECRET_ASSIGN_RE.sub(lambda m: f"{m.group('prefix')}***", redacted)
    return redacted


def _normalise_duration(duration: Any) -> float:
    try:
        return max(0.0, float(duration or 0.0))
    except (TypeError, ValueError):
        return 0.0


def _bounded_history(agent: Any) -> list[dict[str, Any]]:
    history = getattr(agent, "_recent_tool_activity", None)
    if not isinstance(history, list):
        history = []
        try:
            agent._recent_tool_activity = history
        except Exception:
            return []
    return history


def _append_completed_action(agent: Any, item: dict[str, Any]) -> None:
    history = _bounded_history(agent)
    history.append(item)
    del history[:-_ACTION_HISTORY_MAX]


def tool_activity_label(tool_name: str, args: Optional[dict], *, max_len: int = _TOOL_PREVIEW_MAX) -> Optional[str]:
    """Return a redacted, one-line, human tool activity label.

    Mirrors the gateway's tool-progress preview logic but keeps it tiny enough
    for a repeatedly-edited heartbeat bubble.
    """
    if not tool_name or not isinstance(args, dict):
        return None
    try:
        display_args = redact_tool_args_for_display(tool_name, args) or args
        label = build_tool_label(tool_name, display_args, max_len=max_len)
    except Exception:
        return None
    label = _redact_preview_text(label) if label else ""
    label = _truncate(label, max_len) if label else ""
    return label or None


def mark_tool_started(agent: Any, tool_name: str, args: Optional[dict]) -> None:
    """Stamp the active tool + redacted preview on the agent."""
    try:
        agent._current_tool = str(tool_name or "tool")
        agent._current_tool_preview = tool_activity_label(tool_name, args)
        agent._current_tool_started_at = time.time()
    except Exception:
        pass


def mark_concurrent_tools_started(agent: Any, calls: Iterable[tuple[str, dict]]) -> None:
    """Stamp a compact summary for a concurrent tool batch."""
    try:
        call_list = [(str(name or "tool"), args or {}) for name, args in calls]
        if not call_list:
            return
        names = [name for name, _ in call_list]
        labels: list[str] = []
        for name, args in call_list[:3]:
            label = tool_activity_label(name, args, max_len=_CONCURRENT_TOOL_PREVIEW_MAX)
            labels.append(label or name)
        if len(call_list) > 3:
            labels.append(f"+{len(call_list) - 3} more")
        agent._current_tool = ", ".join(names)
        agent._current_tool_preview = "; ".join(labels)
        agent._current_tool_started_at = time.time()
    except Exception:
        pass


def mark_tool_completed(agent: Any, tool_name: str, duration: Any, *, is_error: bool = False) -> None:
    """Stamp the last completed tool, append history, and clear active-tool metadata."""
    dur = _normalise_duration(duration)
    try:
        completed_at = time.time()
        completed = {
            "name": str(tool_name or "tool"),
            "label": str(tool_name or "tool"),
            "duration": dur,
            "is_error": bool(is_error),
            "state": "failed" if is_error else "done",
            "completed_at": completed_at,
        }
        agent._last_completed_tool = {
            "name": completed["name"],
            "duration": completed["duration"],
            "is_error": completed["is_error"],
            "completed_at": completed_at,
        }
        _append_completed_action(agent, completed)
        agent._current_tool = None
        agent._current_tool_preview = None
        agent._current_tool_started_at = None
    except Exception:
        pass


def reset_turn_activity(agent: Any) -> None:
    """Clear per-turn activity metadata on a cached agent."""
    try:
        agent._current_tool = None
        agent._current_tool_preview = None
        agent._current_tool_started_at = None
        agent._last_completed_tool = None
        agent._recent_tool_activity = []
    except Exception:
        pass


def current_tool_elapsed(agent: Any, *, now: Optional[float] = None) -> Optional[float]:
    started = getattr(agent, "_current_tool_started_at", None)
    if not isinstance(started, (int, float)):
        return None
    end = time.time() if now is None else now
    elapsed = end - float(started)
    return elapsed if elapsed >= 0 else 0.0


def tool_activity_history(agent: Any, *, now: Optional[float] = None) -> list[dict[str, Any]]:
    """Return up to 3 recent tool actions, including the current running one."""
    items: list[dict[str, Any]] = []
    for item in _bounded_history(agent)[-_ACTION_HISTORY_MAX:]:
        if isinstance(item, dict):
            copied = dict(item)
            copied["label"] = _truncate(_redact_preview_text(str(copied.get("label") or copied.get("name") or "tool")), _TOOL_PREVIEW_MAX)
            copied["duration"] = _normalise_duration(copied.get("duration"))
            items.append(copied)

    elapsed = current_tool_elapsed(agent, now=now)
    current_label = getattr(agent, "_current_tool_preview", None) or getattr(agent, "_current_tool", None)
    if current_label and isinstance(elapsed, (int, float)):
        items.append({
            "name": str(getattr(agent, "_current_tool", None) or "tool"),
            "label": _truncate(_redact_preview_text(str(current_label)), _TOOL_PREVIEW_MAX),
            "duration": elapsed,
            "state": "running",
            "is_error": False,
        })

    return items[-_ACTION_HISTORY_MAX:]


def todo_activity_snapshot(store: Any) -> Optional[dict[str, Any]]:
    """Return the current/next todo item for heartbeat display."""
    if store is None:
        return None
    try:
        items = store.read_with_timing() if hasattr(store, "read_with_timing") else store.read()
    except Exception:
        return None
    if not isinstance(items, list):
        return None

    selected = None
    for item in items:
        if isinstance(item, dict) and item.get("status") == "in_progress":
            selected = item
            break
    if selected is None:
        for item in items:
            if isinstance(item, dict) and item.get("status") == "pending":
                selected = item
                break
    if not selected:
        return None

    raw_content = selected.get("content") or selected.get("id") or "todo"
    content = _truncate(_redact_preview_text(str(raw_content)), _TODO_CONTENT_MAX)
    if not content:
        return None
    elapsed = selected.get("elapsed_seconds")
    return {
        "content": content,
        "status": str(selected.get("status") or "pending"),
        "elapsed_seconds": elapsed if isinstance(elapsed, (int, float)) else None,
    }
