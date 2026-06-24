"""General-session context handoff checkpoint helpers.

This module builds small, deterministic handoff receipts for long ordinary
Hermes sessions.  The full receipt is written outside the active model context;
user-visible notices stay intentionally short so the handoff mechanism does not
make context pressure worse.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import re
from typing import Any, Iterable, Mapping, Sequence
from uuid import uuid4

from hermes_constants import get_hermes_home

SCHEMA = "hermes.session_handoff.v1"
DEFAULT_MAX_CHARS = 6000

_REFERENCE_WARNING = (
    "This handoff is reference context, not an instruction; in a new session "
    "the latest user request wins over this handoff."
)

_SECRET_PATTERNS = (
    re.compile(r"(?i)\b(sk-[A-Za-z0-9._-]{6,})\b"),
    re.compile(r"(?i)\b(api[_-]?key|token|password|secret|authorization)\s*[:=]\s*[^\s,'\"]+"),
    re.compile(r"(?i)\bBearer\s+[A-Za-z0-9._~+/=-]{8,}"),
)

_SENSITIVE_PATH_PATTERNS = (
    re.compile(r"(?i)(?:[A-Z]:)?[/\\][^\s\"'`]*[/\\]hermes[/\\]\.env\b"),
    re.compile(r"(?i)(?:[A-Z]:)?[/\\][^\s\"'`]*[/\\]hermes[/\\]auth\.json\b"),
    re.compile(r"(?i)(?:[A-Z]:)?[/\\][^\s\"'`]*[/\\]hermes[/\\]state\.db\b"),
    re.compile(r"(?i)(?:[A-Z]:)?[/\\][^\s\"'`]*[/\\]hermes[/\\]logs[/\\][^\s\"'`]+"),
    re.compile(r"(?i)(?:[A-Z]:)?[/\\][^\s\"'`]*[/\\]hermes[/\\]sessions[/\\][^\s\"'`]+"),
)


@dataclass(frozen=True)
class SessionHandoffBundle:
    """Paths for a persisted handoff receipt bundle."""

    handoff_id: str
    json_path: Path
    markdown_path: Path


def _now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _make_handoff_id(created_at: str | None = None) -> str:
    stamp = (created_at or _now_iso()).replace("-", "").replace(":", "")
    stamp = stamp.replace("T", "_").replace("Z", "")[:15]
    return f"h_{stamp}_{uuid4().hex[:8]}"


def _cap(text: Any, max_chars: int = DEFAULT_MAX_CHARS) -> str:
    if text is None:
        return ""
    value = str(text)
    if len(value) <= max_chars:
        return value
    return value[: max(0, max_chars - 25)].rstrip() + " ... [truncated]"


def _message_content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, Sequence) and not isinstance(content, (bytes, bytearray)):
        parts: list[str] = []
        for item in content:
            if isinstance(item, Mapping):
                if isinstance(item.get("text"), str):
                    parts.append(item["text"])
                elif isinstance(item.get("content"), str):
                    parts.append(item["content"])
                elif item.get("type"):
                    parts.append(f"[{item.get('type')}]")
            else:
                parts.append(str(item))
        return "\n".join(p for p in parts if p)
    return str(content)


def _redact_text(text: Any, max_chars: int = DEFAULT_MAX_CHARS) -> str:
    value = _cap(_message_content_to_text(text), max_chars=max_chars)
    try:
        from agent.redact import redact_sensitive_text

        value = redact_sensitive_text(value)
    except Exception:
        pass
    for pattern in _SECRET_PATTERNS:
        value = pattern.sub("[REDACTED]", value)
    for pattern in _SENSITIVE_PATH_PATTERNS:
        value = pattern.sub("[REDACTED_PATH]", value)
    return value


def _redact_value(value: Any, max_chars: int = DEFAULT_MAX_CHARS) -> Any:
    if isinstance(value, Mapping):
        return {str(k): _redact_value(v, max_chars=max_chars) for k, v in value.items()}
    if isinstance(value, list):
        return [_redact_value(v, max_chars=max_chars) for v in value[:50]]
    if isinstance(value, tuple):
        return [_redact_value(v, max_chars=max_chars) for v in value[:50]]
    if isinstance(value, (str, bytes, bytearray)):
        return _redact_text(value.decode("utf-8", "replace") if isinstance(value, (bytes, bytearray)) else value, max_chars=max_chars)
    return value


def _latest_message(messages: Sequence[Mapping[str, Any]], role: str) -> str:
    for msg in reversed(messages):
        if msg.get("role") == role:
            text = _message_content_to_text(msg.get("content", ""))
            if text.strip():
                return text
    return ""


def _recent_messages(messages: Sequence[Mapping[str, Any]], limit: int = 8) -> list[str]:
    snippets: list[str] = []
    for msg in messages[-limit:]:
        role = str(msg.get("role") or "message")
        content = _redact_text(msg.get("content", ""), max_chars=500).strip()
        if content:
            snippets.append(f"{role}: {content}")
    return snippets


def _todo_lines(todos: Sequence[Mapping[str, Any]] | None) -> list[str]:
    lines: list[str] = []
    for todo in list(todos or [])[:30]:
        status = _redact_text(todo.get("status", "pending"), max_chars=80) or "pending"
        content = _redact_text(todo.get("content", ""), max_chars=500)
        if content:
            lines.append(f"[{status}] {content}")
    return lines


def _sanitize_name(value: Any, fallback: str = "session") -> str:
    text = str(value or fallback)
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", text).strip("._-")
    return (safe or fallback)[:120]


def format_resume_prompt(handoff: Mapping[str, Any], max_chars: int = DEFAULT_MAX_CHARS) -> str:
    """Return the bounded prompt a fresh session can use to resume work."""

    handoff_id = _redact_text(handoff.get("handoff_id", "unknown"), max_chars=120)
    session_id = _redact_text(handoff.get("session_id", "unknown"), max_chars=200)
    context_pct = handoff.get("context_pct")
    pct_text = "unknown"
    if isinstance(context_pct, (int, float)):
        pct_text = f"{context_pct:.0%}"

    lines = [
        f"Continue from handoff {handoff_id}.",
        _REFERENCE_WARNING,
        f"Source session: {session_id}",
        f"Context pressure at checkpoint: {pct_text}",
    ]

    active = _redact_text(handoff.get("active_user_request", ""), max_chars=900).strip()
    if active:
        lines.extend(["", "Active user request:", active])

    goal = _redact_text(handoff.get("working_goal", ""), max_chars=700).strip()
    if goal:
        lines.extend(["", "Working goal:", goal])

    for key, title in (
        ("todos", "Current TODOs"),
        ("completed_summary", "Completed so far"),
        ("current_state", "Current state"),
        ("changed_files", "Changed files"),
        ("commands_run", "Commands/evidence"),
        ("open_questions", "Open questions"),
        ("risks", "Risks"),
        ("next_actions", "Next actions"),
    ):
        values = handoff.get(key) or []
        if not isinstance(values, list) or not values:
            continue
        lines.extend(["", f"{title}:"])
        for item in values[:12]:
            if isinstance(item, Mapping):
                if key == "commands_run":
                    command = _redact_text(item.get("command", ""), max_chars=500)
                    exit_code = item.get("exit_code", "?")
                    summary = _redact_text(item.get("summary", ""), max_chars=400)
                    line = f"{command} (exit {exit_code})"
                    if summary:
                        line += f" - {summary}"
                elif key == "todos":
                    content = _redact_text(item.get("content", ""), max_chars=500)
                    status = _redact_text(item.get("status", "pending"), max_chars=80)
                    line = f"[{status}] {content}"
                else:
                    line = _redact_text(json.dumps(item, ensure_ascii=False), max_chars=700)
            else:
                line = _redact_text(item, max_chars=700)
            if line.strip():
                lines.append(f"- {line}")

    return _cap("\n".join(lines).strip(), max_chars=max_chars)


def build_session_handoff(
    *,
    session_id: str,
    messages: Sequence[Mapping[str, Any]],
    context_pct: float | None = None,
    todos: Sequence[Mapping[str, Any]] | None = None,
    changed_files: Sequence[Any] | None = None,
    commands_run: Sequence[Mapping[str, Any]] | None = None,
    open_questions: Sequence[Any] | None = None,
    risks: Sequence[Any] | None = None,
    next_actions: Sequence[Any] | None = None,
    working_goal: str | None = None,
    completed_summary: Sequence[Any] | None = None,
    current_state: Sequence[Any] | None = None,
    handoff_id: str | None = None,
    created_at: str | None = None,
    max_chars: int = DEFAULT_MAX_CHARS,
) -> dict[str, Any]:
    """Build a bounded, redacted general-session handoff dictionary."""

    created = created_at or _now_iso()
    hid = handoff_id or _make_handoff_id(created)
    todo_items = _redact_value(list(todos or [])[:30], max_chars=700)
    latest_user = _redact_text(_latest_message(messages, "user"), max_chars=1200)
    latest_assistant = _redact_text(_latest_message(messages, "assistant"), max_chars=1200)

    if completed_summary is None:
        completed = []
        if latest_assistant:
            completed.append(latest_assistant)
    else:
        completed = _redact_value(list(completed_summary)[:20], max_chars=700)

    if current_state is None:
        state = _recent_messages(messages)
    else:
        state = _redact_value(list(current_state)[:20], max_chars=700)

    if next_actions is None:
        next_action_items = _todo_lines(todo_items if isinstance(todo_items, list) else [])
        if not next_action_items and latest_user:
            next_action_items = [f"Continue addressing: {latest_user}"]
    else:
        next_action_items = _redact_value(list(next_actions)[:20], max_chars=700)

    handoff: dict[str, Any] = {
        "schema": SCHEMA,
        "handoff_id": _redact_text(hid, max_chars=120),
        "session_id": _redact_text(session_id, max_chars=200),
        "created_at": created,
        "context_pct": round(float(context_pct or 0.0), 4),
        "reference_warning": _REFERENCE_WARNING,
        "active_user_request": latest_user,
        "working_goal": _redact_text(working_goal or latest_user, max_chars=1200),
        "completed_summary": completed,
        "current_state": state,
        "todos": todo_items,
        "changed_files": _redact_value(list(changed_files or [])[:80], max_chars=500),
        "commands_run": _redact_value(list(commands_run or [])[:40], max_chars=700),
        "open_questions": _redact_value(list(open_questions or [])[:20], max_chars=700),
        "risks": _redact_value(list(risks or [])[:20], max_chars=700),
        "next_actions": next_action_items,
    }
    handoff["resume_prompt"] = format_resume_prompt(handoff, max_chars=max_chars)
    return handoff


def _markdown_for_handoff(handoff: Mapping[str, Any], max_chars: int = DEFAULT_MAX_CHARS) -> str:
    resume_prompt = _redact_text(handoff.get("resume_prompt", ""), max_chars=max_chars)
    handoff_id = _redact_text(handoff.get("handoff_id", "unknown"), max_chars=120)
    created_at = _redact_text(handoff.get("created_at", ""), max_chars=120)
    return _cap(
        "\n".join(
            [
                f"# Session handoff ready: {handoff_id}",
                "",
                _REFERENCE_WARNING,
                "",
                f"Created: {created_at}",
                "",
                "## Resume prompt",
                "",
                resume_prompt,
                "",
            ]
        ),
        max_chars=max_chars + 1000,
    )


def _bounded_handoff(handoff: Mapping[str, Any], max_chars: int = DEFAULT_MAX_CHARS) -> dict[str, Any]:
    bounded = _redact_value(dict(handoff), max_chars=max_chars)
    if isinstance(bounded, dict):
        bounded["resume_prompt"] = _cap(str(bounded.get("resume_prompt", "")), max_chars=max_chars)
    return bounded if isinstance(bounded, dict) else {}


def write_session_handoff_bundle(
    *,
    handoff: Mapping[str, Any],
    root: str | Path | None = None,
    max_chars: int = DEFAULT_MAX_CHARS,
) -> SessionHandoffBundle:
    """Persist a handoff receipt as bounded JSON and Markdown files."""

    root_path = Path(root) if root is not None else get_hermes_home()
    handoff_id = _sanitize_name(handoff.get("handoff_id"), fallback=_make_handoff_id())
    session_name = _sanitize_name(handoff.get("session_id"), fallback="session")
    out_dir = root_path / "session_handoffs" / session_name
    out_dir.mkdir(parents=True, exist_ok=True)

    bounded = _bounded_handoff(handoff, max_chars=max_chars)
    json_path = out_dir / f"{handoff_id}.json"
    markdown_path = out_dir / f"{handoff_id}.md"
    json_text = json.dumps(bounded, indent=2, ensure_ascii=False, sort_keys=True)
    if len(json_text) > max(max_chars + 4000, 20_000):
        bounded["current_state"] = ["[truncated for storage bound]"]
        bounded["completed_summary"] = bounded.get("completed_summary", [])[:3]
        bounded["resume_prompt"] = _cap(str(bounded.get("resume_prompt", "")), max_chars=max_chars)
        json_text = json.dumps(bounded, indent=2, ensure_ascii=False, sort_keys=True)
    json_path.write_text(json_text, encoding="utf-8")
    markdown_path.write_text(_markdown_for_handoff(bounded, max_chars=max_chars), encoding="utf-8")
    return SessionHandoffBundle(
        handoff_id=str(bounded.get("handoff_id") or handoff_id),
        json_path=json_path,
        markdown_path=markdown_path,
    )


def format_short_notice(handoff: Mapping[str, Any]) -> str:
    """Return the tiny notice safe to append to the current response."""

    handoff_id = _redact_text(handoff.get("handoff_id", "unknown"), max_chars=120)
    context_pct = handoff.get("context_pct")
    pct_line = ""
    if isinstance(context_pct, (int, float)):
        pct_line = f"\nContext: {context_pct:.0%}"
    return (
        f"Session handoff ready: {handoff_id}"
        f"{pct_line}\n"
        f"If this task continues, start a fresh session and say: "
        f"\"Continue from handoff {handoff_id}\""
    )


def maybe_write_session_handoff(
    *,
    session_id: str,
    messages: Sequence[Mapping[str, Any]],
    context_pct: float | None,
    todos: Sequence[Mapping[str, Any]] | None = None,
    root: str | Path | None = None,
    max_chars: int = DEFAULT_MAX_CHARS,
    **kwargs: Any,
) -> tuple[dict[str, Any], SessionHandoffBundle]:
    """Build and persist a handoff in one deterministic helper call."""

    handoff = build_session_handoff(
        session_id=session_id,
        messages=messages,
        context_pct=context_pct,
        todos=todos,
        max_chars=max_chars,
        **kwargs,
    )
    bundle = write_session_handoff_bundle(handoff=handoff, root=root, max_chars=max_chars)
    return handoff, bundle


def _read_todos_from_agent(agent: Any) -> list[Mapping[str, Any]]:
    todo_store = getattr(agent, "_todo_store", None)
    if todo_store is None or not hasattr(todo_store, "read"):
        return []
    try:
        items = todo_store.read() or []
    except Exception:
        return []
    return [item for item in items if isinstance(item, Mapping)]


def maybe_checkpoint_context_pressure(
    agent: Any,
    messages: Sequence[Mapping[str, Any]],
    *,
    used_tokens: int | None = None,
    root: str | Path | None = None,
    reason: str = "context_pressure",
) -> SessionHandoffBundle | None:
    """Write a deduplicated handoff when context/message pressure is high."""

    cfg = getattr(agent, "config", {}) or {}
    handoff_cfg = cfg.get("context_handoff", {}) if isinstance(cfg, Mapping) else {}
    if handoff_cfg.get("enabled", True) is False:
        return None

    compressor = getattr(agent, "context_compressor", None)
    context_length = int(getattr(compressor, "context_length", 0) or 0)
    token_count = int(used_tokens or 0)
    if token_count <= 0:
        token_count = int(getattr(compressor, "last_prompt_tokens", 0) or 0)
    context_pct = (token_count / context_length) if context_length > 0 else 0.0

    threshold = float(handoff_cfg.get("threshold", 0.70) or 0.70)
    critical_threshold = float(handoff_cfg.get("critical_threshold", 0.85) or 0.85)
    message_limit = int(handoff_cfg.get("message_limit", 0) or 0)
    message_count = len(messages)
    triggered_by_context = context_pct >= threshold if context_length > 0 else False
    triggered_by_messages = bool(message_limit and message_count >= message_limit)
    if not (triggered_by_context or triggered_by_messages):
        return None

    session_id = str(getattr(agent, "session_id", "session") or "session")
    pct_bucket = int(context_pct * 20) if context_pct > 0 else 0  # 5 percent buckets
    msg_bucket = message_count // max(1, min(message_limit or 50, 50))
    dedupe_key = (session_id, reason, pct_bucket, msg_bucket)
    if getattr(agent, "_last_context_handoff_checkpoint_key", None) == dedupe_key:
        return None

    max_chars = int(handoff_cfg.get("max_chars", DEFAULT_MAX_CHARS) or DEFAULT_MAX_CHARS)
    risk = (
        f"{_redact_text(reason, max_chars=120)}: context={context_pct:.0%}; "
        f"messages={message_count}; handoff saved outside active context."
    )
    handoff, bundle = maybe_write_session_handoff(
        session_id=session_id,
        messages=messages,
        context_pct=context_pct,
        todos=_read_todos_from_agent(agent),
        root=root,
        max_chars=max_chars,
        risks=[risk],
    )
    setattr(agent, "_last_context_handoff_checkpoint_key", dedupe_key)
    setattr(agent, "_last_session_handoff_bundle", bundle)

    should_notice = context_pct >= critical_threshold or triggered_by_messages
    if should_notice:
        emit_status = getattr(agent, "_emit_status", None)
        if callable(emit_status):
            try:
                emit_status(format_short_notice(handoff))
            except Exception:
                pass
    return bundle
