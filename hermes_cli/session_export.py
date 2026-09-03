"""Shared renderers for session export commands.

The CLI, dashboard, and slash-command surfaces all deal with the same
session-shaped data: a session dict with a ``messages`` list. Keep filtering
and human-readable rendering here so each surface only has to load sessions
and write bytes.
"""

from __future__ import annotations

from datetime import datetime, timezone
from html import escape as html_escape
import json
from typing import Any, Dict, Iterable, Iterator, List, Literal, Optional, Tuple


ExportFormat = Literal["jsonl", "markdown"]
ExportOnly = Literal["user-prompts"]


def normalize_export_format(fmt: str) -> ExportFormat:
    """Return the canonical export format name."""
    value = (fmt or "jsonl").strip().lower()
    if value == "md":
        value = "markdown"
    if value not in {"jsonl", "markdown"}:
        raise ValueError(f"Unsupported session export format: {fmt}")
    return value  # type: ignore[return-value]


def normalize_export_only(only: Optional[str]) -> Optional[ExportOnly]:
    """Return the canonical export filter name."""
    if only is None:
        return None
    value = only.strip().lower()
    if value in {"user", "prompts", "user-prompts", "user_prompts"}:
        return "user-prompts"
    raise ValueError(f"Unsupported session export filter: {only}")


def render_sessions_export(
    sessions: Iterable[Dict[str, Any]],
    *,
    fmt: str = "jsonl",
    only: Optional[str] = None,
) -> str:
    """Render exported sessions in a stable, reusable format.

    ``fmt=jsonl`` with no filter intentionally preserves the legacy shape:
    one full session object per line. ``only=user-prompts`` switches the unit
    of export to one prompt record per line so the output is easy to pipe into
    review, memory-ingestion, or prompt-library tooling.
    """
    session_list = list(sessions)
    export_format = normalize_export_format(fmt)
    export_only = normalize_export_only(only)

    if export_format == "jsonl":
        return _render_jsonl(session_list, only=export_only)
    return _render_markdown(session_list, only=export_only)


def export_record_count(
    sessions: Iterable[Dict[str, Any]], *, only: Optional[str] = None
) -> Tuple[int, str]:
    """Return ``(count, noun)`` for status messages after an export."""
    session_list = list(sessions)
    export_only = normalize_export_only(only)
    if export_only == "user-prompts":
        return sum(1 for _ in iter_user_prompt_records(session_list)), "prompt"
    return len(session_list), "session"


def iter_user_prompt_records(
    sessions: Iterable[Dict[str, Any]]
) -> Iterator[Dict[str, Any]]:
    """Yield one normalized record for each user-authored prompt."""
    for session in sessions:
        session_id = str(session.get("id") or session.get("session_id") or "")
        index = 0
        for message in _messages(session):
            if message.get("role") != "user":
                continue
            index += 1
            record: Dict[str, Any] = {
                "session_id": session_id,
                "index": index,
                "created_at": _format_timestamp(message.get("timestamp")),
                "role": "user",
                "text": _message_text(message.get("content")),
            }
            message_id = message.get("id")
            if message_id is not None:
                record["message_id"] = message_id
            event_id = message.get("platform_message_id") or message.get("event_id")
            if event_id:
                record["event_id"] = event_id
            yield record


def _render_jsonl(
    sessions: List[Dict[str, Any]], *, only: Optional[ExportOnly]
) -> str:
    if only == "user-prompts":
        rows = iter_user_prompt_records(sessions)
    else:
        rows = iter(sessions)
    lines = [json.dumps(row, ensure_ascii=False) for row in rows]
    return ("\n".join(lines) + "\n") if lines else ""


def _render_markdown(
    sessions: List[Dict[str, Any]], *, only: Optional[ExportOnly]
) -> str:
    if only == "user-prompts":
        return _render_user_prompts_markdown(sessions)
    return _render_full_markdown(sessions)


def _render_user_prompts_markdown(sessions: List[Dict[str, Any]]) -> str:
    lines: List[str] = []
    if len(sessions) == 1:
        session = sessions[0]
        lines.append(f"# User prompts for session {_heading_text(_session_id(session))}")
        lines.extend(_session_metadata_lines(session))
        lines.append("")
        _append_prompt_records(lines, session, heading_level=2)
    else:
        lines.append("# User prompts export")
        lines.append("")
        for session in sessions:
            lines.append(f"## Session {_heading_text(_session_id(session))}")
            lines.extend(_session_metadata_lines(session))
            lines.append("")
            _append_prompt_records(lines, session, heading_level=3)
    if not sessions:
        lines.append("_No user prompts found._")
        lines.append("")
    return _finish_markdown(lines)


def _append_prompt_records(
    lines: List[str], session: Dict[str, Any], *, heading_level: int
) -> None:
    prompts = list(iter_user_prompt_records([session]))
    if not prompts:
        lines.append("_No user prompts found._")
        lines.append("")
        return
    marker = "#" * heading_level
    for prompt in prompts:
        timestamp = prompt.get("created_at") or "timestamp unavailable"
        lines.append(f"{marker} {prompt['index']}. {timestamp}")
        message_id = prompt.get("message_id")
        if message_id is not None:
            lines.append(f"Message ID: `{message_id}`")
            lines.append("")
        lines.append(str(prompt.get("text") or ""))
        lines.append("")


def _render_full_markdown(sessions: List[Dict[str, Any]]) -> str:
    lines: List[str] = []
    if len(sessions) == 1:
        session = sessions[0]
        lines.append(f"# Session: {_heading_text(_session_title_or_id(session))}")
        lines.extend(_session_metadata_lines(session))
        lines.append("")
        _append_session_messages(lines, session, heading_level=2)
    else:
        lines.append("# Hermes sessions export")
        lines.append("")
        for session in sessions:
            lines.append(f"## Session: {_heading_text(_session_title_or_id(session))}")
            lines.extend(_session_metadata_lines(session))
            lines.append("")
            _append_session_messages(lines, session, heading_level=3)
    return _finish_markdown(lines)


def _append_session_messages(
    lines: List[str], session: Dict[str, Any], *, heading_level: int
) -> None:
    marker = "#" * heading_level
    visible_messages = [
        message for message in _messages(session) if message.get("role") != "system"
    ]
    if not visible_messages:
        lines.append("_No messages found._")
        lines.append("")
        return

    for message in visible_messages:
        role = str(message.get("role") or "unknown")
        timestamp = _format_timestamp(message.get("timestamp"))
        suffix = f" - {timestamp}" if timestamp else ""
        if role == "tool":
            tool_name = str(message.get("tool_name") or message.get("name") or "tool")
            lines.append(f"{marker} Tool: {_heading_text(tool_name)}{suffix}")
            lines.append("")
            lines.append(f"<details><summary>{html_escape(tool_name)}</summary>")
            lines.append("")
            lines.append(_fenced_text(_message_text(message.get("content"))))
            lines.append("")
            lines.append("</details>")
            lines.append("")
            continue

        label = {
            "user": "User",
            "assistant": "Assistant",
        }.get(role, role.title())
        lines.append(f"{marker} {label}{suffix}")
        lines.append("")
        lines.append(_message_text(message.get("content")))
        lines.append("")


def _messages(session: Dict[str, Any]) -> List[Dict[str, Any]]:
    messages = session.get("messages") or []
    return [message for message in messages if isinstance(message, dict)]


def _message_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = [_content_part_text(part) for part in content]
        return "\n".join(part for part in parts if part)
    if isinstance(content, dict):
        for key in ("text", "content"):
            value = content.get(key)
            if isinstance(value, str):
                return value
        return json.dumps(content, ensure_ascii=False, sort_keys=True)
    return str(content)


def _content_part_text(part: Any) -> str:
    if isinstance(part, str):
        return part
    if isinstance(part, dict):
        for key in ("text", "content"):
            value = part.get(key)
            if isinstance(value, str):
                return value
        return json.dumps(part, ensure_ascii=False, sort_keys=True)
    return str(part)


def _format_timestamp(value: Any) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return (
            datetime.fromtimestamp(float(value), tz=timezone.utc)
            .isoformat(timespec="seconds")
            .replace("+00:00", "Z")
        )
    if isinstance(value, datetime):
        dt = value if value.tzinfo else value.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc).isoformat(timespec="seconds").replace(
            "+00:00", "Z"
        )
    return str(value)


def _session_metadata_lines(session: Dict[str, Any]) -> List[str]:
    lines: List[str] = [f"- Session ID: `{_session_id(session)}`"]
    source = session.get("source")
    if source:
        lines.append(f"- Source: `{source}`")
    model = session.get("model")
    if model:
        lines.append(f"- Model: `{model}`")
    title = session.get("title")
    if title:
        lines.append(f"- Title: {_inline_text(str(title))}")
    started = _format_timestamp(session.get("started_at"))
    if started:
        lines.append(f"- Started: {started}")
    message_count = session.get("message_count")
    if message_count is not None:
        lines.append(f"- Messages: {message_count}")
    return lines


def _session_id(session: Dict[str, Any]) -> str:
    return str(session.get("id") or session.get("session_id") or "unknown")


def _session_title_or_id(session: Dict[str, Any]) -> str:
    title = str(session.get("title") or "").strip()
    return title or _session_id(session)


def _heading_text(value: str) -> str:
    return " ".join(str(value).splitlines()).strip() or "unknown"


def _inline_text(value: str) -> str:
    return " ".join(value.splitlines()).strip()


def _fenced_text(text: str, *, language: str = "text") -> str:
    fence = "```"
    while fence in text:
        fence += "`"
    return f"{fence}{language}\n{text}\n{fence}"


def _finish_markdown(lines: List[str]) -> str:
    while lines and lines[-1] == "":
        lines.pop()
    return "\n".join(lines) + "\n"
