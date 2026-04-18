"""Runtime observability helpers for direct gateway shortcuts."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any


_MAX_RECENT_DIRECT_SHORTCUTS = 20
_MAX_RECENT_DIRECT_SHORTCUT_SUMMARY = 5


def _truncate_preview(value: Any, *, limit: int = 120) -> str:
    text = " ".join(str(value or "").strip().split())
    if len(text) <= limit:
        return text
    return text[: max(limit - 3, 0)].rstrip() + "..."


def _ensure_recent_direct_shortcuts(runner: Any) -> list[dict[str, Any]]:
    recent = getattr(runner, "_recent_direct_shortcuts", None)
    if isinstance(recent, list):
        return recent
    recent = []
    setattr(runner, "_recent_direct_shortcuts", recent)
    return recent


def record_direct_shortcut_trace(
    runner: Any,
    event: Any,
    *,
    matched_handler: str,
    attempted_handlers: list[str] | tuple[str, ...],
    response: Any,
) -> None:
    source = getattr(event, "source", None)
    entry = {
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "matched_handler": str(matched_handler or "").strip(),
        "attempt_count": len([item for item in attempted_handlers if str(item or "").strip()]),
        "attempted_handlers": [str(item).strip() for item in attempted_handlers if str(item or "").strip()],
        "platform": (
            getattr(getattr(source, "platform", None), "value", None)
            or str(getattr(source, "platform", "") or "").strip()
        ),
        "chat_type": str(getattr(source, "chat_type", "") or "").strip(),
        "chat_id": str(getattr(source, "chat_id", "") or "").strip(),
        "text_preview": _truncate_preview(getattr(event, "text", "")),
        "response_preview": _truncate_preview(response),
    }
    recent = _ensure_recent_direct_shortcuts(runner)
    recent.append(entry)
    overflow = len(recent) - _MAX_RECENT_DIRECT_SHORTCUTS
    if overflow > 0:
        del recent[:overflow]


def build_direct_shortcut_runtime_summary(runner: Any) -> dict[str, Any]:
    recent = _ensure_recent_direct_shortcuts(runner)
    ordered = list(reversed(recent[-_MAX_RECENT_DIRECT_SHORTCUT_SUMMARY:]))
    return {
        "recent_count": len(recent),
        "recent": ordered,
    }
