"""Compact session handoff helpers for gateway auto-reset.

A session handoff is background context for interpreting the first message in a
new session. It is not transcript replay and it must not revive old user
requests as active instructions.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional


SESSION_HANDOFF_FOCUS_TOPIC = (
    "daily/project continuity, loose ends, unresolved user asks, relevant "
    "decisions and constraints, active working state, and what the agent needs "
    "before interpreting the first user message after an automatic session reset"
)


@dataclass
class SessionHandoffConfig:
    """Controls background handoff context after an automatic session reset."""

    mode: str = "none"  # none, notice, last_n, summary
    last_messages: int = 8
    max_chars: int = 2400

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> "SessionHandoffConfig":
        if not isinstance(data, dict):
            return cls()
        mode = str(data.get("mode") or "none").strip().lower()
        if mode not in {"none", "notice", "last_n", "summary"}:
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


def _build_extractive_handoff_summary(messages: List[Dict[str, Any]], limit: int) -> str:
    """Build a compaction-shaped objective fallback when the LLM summary fails.

    This is deliberately not transcript replay. It preserves enough visible
    structure for continuity while keeping old turns as source material only.
    The real summary path reuses the context compaction summarizer; this fallback
    exists so an auto-reset can still proceed if auxiliary summarization is down.
    """

    selected = _select_messages(messages, limit)
    if not selected:
        return "## Active Task\nNone.\n\n## Goal\nNo user/assistant turns were available."

    last_user = next((m["content"] for m in reversed(selected) if m["role"] == "user"), None)
    assistant_turns = [m["content"] for m in selected if m["role"] == "assistant"]
    user_turns = [m["content"] for m in selected if m["role"] == "user"]

    def _clip(text: str, n: int = 700) -> str:
        text = text.replace("\n", " ").strip()
        return text if len(text) <= n else text[: n - 3].rstrip() + "..."

    lines = [
        "## Active Task",
        _clip(last_user) if last_user else "None.",
        "",
        "## Goal",
        "Infer from the user's next message; prior turns are context only.",
        "",
        "## Constraints & Preferences",
        "Prior requests are not active unless the user reactivates them.",
        "",
        "## Completed Actions",
    ]
    lines.extend(f"- {_clip(t, 500)}" for t in assistant_turns[-5:])
    lines.extend([
        "",
        "## Active State",
        "Automatically reset session; bridge metadata is in the handoff header.",
        "",
        "## Pending User Asks",
    ])
    lines.extend(f"- {_clip(t, 500)}" for t in user_turns[-5:])
    lines.extend([
        "",
        "## Remaining Work",
        "Use this only to interpret the new message. Do not continue prior requests unless reactivated.",
        "",
        "## Critical Context",
        f"Summary focus: {SESSION_HANDOFF_FOCUS_TOPIC}.",
    ])
    return "\n".join(lines)


def generate_compaction_handoff_summary(
    messages: List[Dict[str, Any]],
    *,
    model: Optional[str],
    runtime_kwargs: Optional[Dict[str, Any]] = None,
) -> Optional[str]:
    """Reuse the context-compaction summarizer for session handoff summaries.

    Returns the summary body without the compaction prefix. The session handoff
    layer supplies its own stricter header because auto-reset continuity should
    help interpret the first new message, not resume old tasks by default.
    """

    selected = _select_messages(messages, max(len(messages), 1))
    if not selected or not model:
        return None

    runtime = runtime_kwargs or {}
    try:
        from agent.context_compressor import ContextCompressor

        compressor = ContextCompressor(
            model=model,
            quiet_mode=True,
            summary_model_override="",
            base_url=runtime.get("base_url") or "",
            api_key=runtime.get("api_key") or "",
            provider=runtime.get("provider") or "",
            api_mode=runtime.get("api_mode") or "",
        )
        summary = compressor._generate_summary(  # Reuse compaction's sectioned prompt.
            selected,
            focus_topic=SESSION_HANDOFF_FOCUS_TOPIC,
        )
        if not summary:
            return None
        return ContextCompressor._strip_summary_prefix(summary).strip() or None
    except Exception:
        return None


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
    summary_text: Optional[str] = None,
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

    if mode == "summary":
        body = (summary_text or "").strip()
        if not body:
            body = _build_extractive_handoff_summary(parent_messages or [], last_messages)
        note = header + meta + "\n\n" + body
        if len(note) > max_chars:
            note = note[: max_chars - 3].rstrip() + "..."
        return note

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
