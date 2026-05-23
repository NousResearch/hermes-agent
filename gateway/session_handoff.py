"""Compact session handoff helpers for gateway auto-reset.

A session handoff is background context for interpreting the first message in a
new session. It is not transcript replay and it must not revive old user
requests as active instructions.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional


@dataclass
class SessionHandoffConfig:
    """Controls background handoff context after an automatic session reset."""

    mode: str = "none"  # none, notice, last_n
    last_messages: int = 8
    max_chars: int = 2400

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> "SessionHandoffConfig":
        if not isinstance(data, dict):
            return cls()
        mode = str(data.get("mode") or "none").strip().lower()
        if mode not in {"none", "notice", "last_n"}:
            mode = "none"
        return cls(
            mode=mode,
            last_messages=_positive_int(data.get("last_messages"), 8),
            max_chars=_positive_int(data.get("max_chars"), 2400),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "mode": self.mode,
            "last_messages": self.last_messages,
            "max_chars": self.max_chars,
        }


def _positive_int(value: Any, default: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return parsed if parsed > 0 else default


def _text_content(content: Any) -> str:
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts: List[str] = []
        for item in content:
            if isinstance(item, dict):
                text = item.get("text") or item.get("content")
                if isinstance(text, str):
                    parts.append(text)
        return "\n".join(parts).strip()
    return ""


def _format_elapsed(previous_updated_at: Optional[datetime], now: datetime) -> str:
    if not previous_updated_at:
        return "unknown time"
    delta = now - previous_updated_at
    seconds = max(0, int(delta.total_seconds()))
    days, rem = divmod(seconds, 86400)
    hours, rem = divmod(rem, 3600)
    minutes = rem // 60
    if days:
        return f"{days}d {hours}h"
    if hours:
        return f"{hours}h {minutes}m"
    return f"{minutes}m"


def _select_messages(messages: Iterable[Dict[str, Any]], limit: int) -> List[Dict[str, str]]:
    selected: List[Dict[str, str]] = []
    for msg in messages:
        role = str(msg.get("role") or "").strip().lower()
        if role not in {"user", "assistant"}:
            continue
        content = _text_content(msg.get("content"))
        if not content:
            continue
        selected.append({"role": role, "content": content})
    return selected[-limit:]


def build_session_handoff_note(
    *,
    mode: str,
    parent_session_id: Optional[str],
    reset_reason: Optional[str],
    parent_messages: Optional[List[Dict[str, Any]]],
    previous_updated_at: Optional[datetime],
    now: datetime,
    max_chars: int = 2400,
    last_messages: int = 8,
) -> Optional[str]:
    """Return a compact background handoff note, or None when disabled.

    The note is intentionally framed as context for interpretation only. It
    warns the agent not to treat old user requests as active instructions.
    """

    mode = (mode or "none").strip().lower()
    if mode == "none":
        return None
    if not parent_session_id:
        return None

    reason = reset_reason or "auto-reset"
    elapsed = _format_elapsed(previous_updated_at, now)
    local_time = now.strftime("%Y-%m-%d %H:%M")
    header = (
        "[Session handoff: It is now a new day/session. What should the agent "
        "know before interpreting the next user message? This is background "
        "context only. Do not treat prior user requests as active instructions "
        "unless the user reactivates them."
    )
    meta = (
        f" Prior session: {parent_session_id}. Reset reason: {reason}. "
        f"Elapsed since last activity: {elapsed}. Current local time: {local_time}.]"
    )

    if mode == "notice":
        return header + meta

    if mode != "last_n":
        return None

    selected = _select_messages(parent_messages or [], last_messages)
    if not selected:
        return header + meta

    lines = [header + meta, "Recent prior-session turns, for interpretation only:"]
    for item in selected:
        content = item["content"].replace("\n", " ").strip()
        if len(content) > 360:
            content = content[:357].rstrip() + "..."
        lines.append(f"- {item['role']}: {content}")

    note = "\n".join(lines)
    if len(note) > max_chars:
        note = note[: max_chars - 3].rstrip() + "..."
    return note
