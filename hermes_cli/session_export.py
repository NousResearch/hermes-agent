"""Shared renderers for session export commands.

CLI, dashboard, and slash-command surfaces all deal with the same session-shaped data (a session
dict with a ``messages`` list); filtering and human-readable rendering live here so each surface
only loads sessions and writes bytes.
"""

from __future__ import annotations

from datetime import datetime, timezone
from html import escape as html_escape
import json
from typing import Any, Dict, Iterable, Iterator, List, Literal, Optional, Tuple


ExportFormat = Literal["jsonl", "markdown"]
ExportOnly = Literal["user-prompts"]

_EXPORT_FORMATS = {"jsonl": "jsonl", "markdown": "markdown", "md": "markdown"}
_ONLY_ALIASES = {"user", "prompts", "user-prompts", "user_prompts"}


def normalize_export_format(fmt: str) -> ExportFormat:
    """Return the canonical export format name."""
    value = _EXPORT_FORMATS.get((fmt or "jsonl").strip().lower())
    if value is None:
        raise ValueError(f"Unsupported session export format: {fmt}")
    return value  # type: ignore[return-value]


def normalize_export_only(only: Optional[str]) -> Optional[ExportOnly]:
    """Return the canonical export filter name."""
    if only is None:
        return None
    if only.strip().lower() in _ONLY_ALIASES:
        return "user-prompts"
    raise ValueError(f"Unsupported session export filter: {only}")


def render_sessions_export(sessions: Iterable[Dict[str, Any]], *, fmt: str = "jsonl", only: Optional[str] = None) -> str:
    """Render exported sessions in a stable, reusable format.

    ``fmt=jsonl`` with no filter keeps the legacy shape (one full session object per line);
    ``only=user-prompts`` switches the unit to one prompt record per line for piping into
    review / memory-ingestion / prompt-library tooling.
    """
    session_list = list(sessions)
    export_format = normalize_export_format(fmt)
    prompts_only = normalize_export_only(only) == "user-prompts"
    if export_format == "jsonl":
        rows = iter_user_prompt_records(session_list) if prompts_only else session_list
        lines = [json.dumps(row, ensure_ascii=False) for row in rows]
        return ("\n".join(lines) + "\n") if lines else ""
    # One session → its own H1 with body at H2; several → a shared H1, each session H2/H3.
    if prompts_only:
        multi_title, append_body = "User prompts export", _append_prompt_records
        headings = (lambda s: f"User prompts for session {_heading_text(_session_id(s))}",
                    lambda s: f"Session {_heading_text(_session_id(s))}")
    else:
        multi_title, append_body = "Hermes sessions export", _append_session_messages
        headings = (lambda s: f"Session: {_heading_text(_session_title_or_id(s))}",) * 2
    lines: List[str] = []
    single = len(session_list) == 1
    if not single:
        lines += [f"# {multi_title}", ""]
    for session in session_list:
        level = 1 if single else 2
        lines += [f"{'#' * level} {headings[level - 1](session)}", *_session_metadata_lines(session), ""]
        append_body(lines, session, heading_level=level + 1)
    if prompts_only and not session_list:
        lines += ["_No user prompts found._", ""]
    while lines and lines[-1] == "":
        lines.pop()
    return "\n".join(lines) + "\n"


def export_record_count(sessions: Iterable[Dict[str, Any]], *, only: Optional[str] = None) -> Tuple[int, str]:
    """Return ``(count, noun)`` for status messages after an export."""
    session_list = list(sessions)
    if normalize_export_only(only) == "user-prompts":
        return sum(1 for _ in iter_user_prompt_records(session_list)), "prompt"
    return len(session_list), "session"


def iter_user_prompt_records(sessions: Iterable[Dict[str, Any]]) -> Iterator[Dict[str, Any]]:
    """Yield one normalized record for each user-authored prompt."""
    for session in sessions:
        session_id = str(session.get("id") or session.get("session_id") or "")
        prompts = [m for m in _messages(session) if m.get("role") == "user"]
        for index, message in enumerate(prompts, start=1):
            record: Dict[str, Any] = {
                "session_id": session_id,
                "index": index,
                "created_at": _format_timestamp(message.get("timestamp")),
                "role": "user",
                "text": _message_text(message.get("content")),
            }
            if (message_id := message.get("id")) is not None:
                record["message_id"] = message_id
            if event_id := message.get("platform_message_id") or message.get("event_id"):
                record["event_id"] = event_id
            yield record


def _append_prompt_records(lines: List[str], session: Dict[str, Any], *, heading_level: int) -> None:
    prompts = list(iter_user_prompt_records([session]))
    if not prompts:
        lines += ["_No user prompts found._", ""]
        return
    marker = "#" * heading_level
    for prompt in prompts:
        lines.append(f"{marker} {prompt['index']}. {prompt.get('created_at') or 'timestamp unavailable'}")
        if (message_id := prompt.get("message_id")) is not None:
            lines += [f"Message ID: `{message_id}`", ""]
        lines += [str(prompt.get("text") or ""), ""]


def _append_session_messages(lines: List[str], session: Dict[str, Any], *, heading_level: int) -> None:
    marker = "#" * heading_level
    visible_messages = [message for message in _messages(session) if message.get("role") != "system"]
    if not visible_messages:
        lines += ["_No messages found._", ""]
        return
    for message in visible_messages:
        role = str(message.get("role") or "unknown")
        timestamp = _format_timestamp(message.get("timestamp"))
        suffix = f" - {timestamp}" if timestamp else ""
        text = _message_text(message.get("content"))
        if role == "tool":
            tool_name = str(message.get("tool_name") or message.get("name") or "tool")
            lines += [
                f"{marker} Tool: {_heading_text(tool_name)}{suffix}", "",
                f"<details><summary>{html_escape(tool_name)}</summary>", "",
                _fenced_text(text), "", "</details>", "",
            ]
        else:
            label = {"user": "User", "assistant": "Assistant"}.get(role, role.title())
            lines += [f"{marker} {label}{suffix}", "", text, ""]


def _messages(session: Dict[str, Any]) -> List[Dict[str, Any]]:
    return [message for message in session.get("messages") or [] if isinstance(message, dict)]


def _message_text(content: Any) -> str:
    if isinstance(content, list):
        return "\n".join(part for part in map(_content_part_text, content) if part)
    return "" if content is None else _content_part_text(content)


def _content_part_text(part: Any) -> str:
    if not isinstance(part, dict):
        return part if isinstance(part, str) else str(part)
    for key in ("text", "content"):
        if isinstance(value := part.get(key), str):
            return value
    return json.dumps(part, ensure_ascii=False, sort_keys=True)


def _format_timestamp(value: Any) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        dt = datetime.fromtimestamp(float(value), tz=timezone.utc)
    elif isinstance(value, datetime):
        dt = (value if value.tzinfo else value.replace(tzinfo=timezone.utc)).astimezone(timezone.utc)
    else:
        return str(value)
    return dt.isoformat(timespec="seconds").replace("+00:00", "Z")


def _session_metadata_lines(session: Dict[str, Any]) -> List[str]:
    lines: List[str] = [f"- Session ID: `{_session_id(session)}`"]
    for key, label in (("source", "Source"), ("model", "Model")):
        if session.get(key):
            lines.append(f"- {label}: `{session[key]}`")
    if title := session.get("title"):
        lines.append(f"- Title: {' '.join(str(title).splitlines()).strip()}")
    if started := _format_timestamp(session.get("started_at")):
        lines.append(f"- Started: {started}")
    if (message_count := session.get("message_count")) is not None:
        lines.append(f"- Messages: {message_count}")
    return lines


def _session_id(session: Dict[str, Any]) -> str:
    return str(session.get("id") or session.get("session_id") or "unknown")


def _session_title_or_id(session: Dict[str, Any]) -> str:
    return str(session.get("title") or "").strip() or _session_id(session)


def _heading_text(value: str) -> str:
    return " ".join(str(value).splitlines()).strip() or "unknown"


def _fenced_text(text: str, *, language: str = "text") -> str:
    fence = "```"
    while fence in text:
        fence += "`"
    return f"{fence}{language}\n{text}\n{fence}"


# --- Current-session save helper (shared by CLI /save and gateway /save) ---

SAVE_FORMATS = ("json", "md", "html")

SAVE_USAGE = """/save — export the current session to a file
Usage: /save <format> [filename] [redact]

Formats:
  json    full session snapshot (canonical export shape)
  md      readable Markdown transcript
  html    standalone single-file HTML page (shareable, no dependencies)

Options:
  filename   optional output name/path (default: auto-named;
             CLI saves under ~/.hermes/sessions/saved/)
  redact     scrub API keys, tokens, and credentials before writing

Examples:
  /save json
  /save html
  /save md notes.md
  /save html session.html redact"""

_SAVE_FORMAT_ALIASES = {"json": "json", "snapshot": "json", "md": "md", "markdown": "md", "html": "html"}


def normalize_save_format(fmt: Optional[str]) -> str:
    """Map a user-typed /save format token to a canonical format."""
    token = (fmt or "json").strip().lower()
    if token not in _SAVE_FORMAT_ALIASES:
        raise ValueError(f"Unknown format {token!r} — expected one of: json, md, html")
    return _SAVE_FORMAT_ALIASES[token]


def _render_html_for_save(session: Dict[str, Any]) -> str:
    from hermes_cli.session_export_html import generate_html_export

    return generate_html_export(session)


_SAVE_RENDERERS = {
    "json": lambda session: json.dumps(session, indent=2, ensure_ascii=False, default=str),
    "md": lambda session: render_sessions_export([session], fmt="markdown"),
    "html": _render_html_for_save,
}


def render_session_for_save(session: Dict[str, Any], fmt: str) -> str:
    """Render one exported session dict for /save."""
    renderer = _SAVE_RENDERERS.get(fmt)
    if renderer is None:
        raise ValueError(f"Unknown save format: {fmt!r}")
    return renderer(session)


def default_save_filename(session_id: str, fmt: str) -> str:
    """Default filename for a /save export of the given session."""
    safe_id = "".join(ch for ch in str(session_id) if ch.isalnum() or ch in ("-", "_")) or "session"
    return f"hermes_session_{safe_id}.{fmt}"
