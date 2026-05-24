"""Lightweight attribution/action ledger helpers.

The ledger is intentionally smaller than a full transcript: it records
redacted tool-action evidence that can answer "what did this agent do?" for
one session lineage without treating git history, summaries, or broad search
hits as attribution proof.
"""

from __future__ import annotations

import hashlib
import json
import time
from datetime import datetime
from typing import Any, Iterable, Mapping, Optional

from agent.redact import redact_sensitive_text

_READ_TOOLS = {
    "read_file",
    "search_files",
    "browser_snapshot",
    "browser_vision",
    "browser_get_images",
    "vision_analyze",
    "session_search",
    "skill_view",
    "skills_list",
}
_WRITE_TOOLS = {
    "write_file",
    "patch",
    "skill_manage",
    "memory",
    "todo",
    "cronjob",
}
_MESSAGE_TOOLS = {"send_message", "text_to_speech"}
_NETWORK_TOOLS = {
    "browser_navigate",
    "browser_click",
    "browser_type",
    "browser_press",
    "browser_scroll",
    "browser_console",
    "image_generate",
    "delegate_task",
}
_GIT_PREFIXES = ("git ", "gh ")
_TERMINAL_WRITE_HINTS = (
    " > ",
    " >> ",
    " tee ",
    " rm ",
    " mv ",
    " cp ",
    " chmod ",
    " chown ",
    " mkdir ",
    " touch ",
    " apply_patch",
)
_TERMINAL_NETWORK_HINTS = ("curl ", "wget ", "ssh ", "scp ", "rsync ")


def _redact_text(value: str) -> str:
    return redact_sensitive_text(str(value or ""), force=True)


def _compact(value: str, limit: int = 360) -> str:
    value = " ".join(str(value or "").split())
    if len(value) <= limit:
        return value
    return value[: limit - 1] + "…"


def _json_summary(value: Any, limit: int = 360) -> str:
    try:
        text = json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
    except Exception:
        text = str(value)
    return _compact(_redact_text(text), limit=limit)


def classify_side_effect(tool_name: str, args: Mapping[str, Any] | None = None) -> str:
    """Return a coarse side-effect class for a tool call."""
    name = str(tool_name or "")
    args = args or {}
    if name in _READ_TOOLS:
        return "read"
    if name in _WRITE_TOOLS:
        return "write"
    if name in _MESSAGE_TOOLS:
        return "message"
    if name in _NETWORK_TOOLS or name.startswith("browser_"):
        return "network"
    if name == "terminal":
        command = str(args.get("command") or "").strip()
        lowered = f" {command.lower()} "
        if command.lower().startswith(_GIT_PREFIXES) or " gh " in lowered:
            return "git"
        if any(hint in lowered for hint in _TERMINAL_NETWORK_HINTS):
            return "network"
        if any(hint in lowered for hint in _TERMINAL_WRITE_HINTS):
            return "write"
        return "process"
    if name == "execute_code":
        return "process"
    return "other"


def summarize_tool_call(tool_name: str, args: Mapping[str, Any] | None = None) -> str:
    """Return a short, secret-redacted human summary of the tool call."""
    name = str(tool_name or "unknown")
    args = args or {}
    if name == "terminal":
        command = _compact(_redact_text(str(args.get("command") or "")), 280)
        workdir = args.get("workdir")
        suffix = f" (workdir={_redact_text(workdir)})" if workdir else ""
        return _compact(f"terminal: {command}{suffix}", 360)
    if name == "execute_code":
        code = _compact(_redact_text(str(args.get("code") or "")), 260)
        return f"execute_code: {code}"
    if name in {"read_file", "write_file"}:
        path = _redact_text(str(args.get("path") or ""))
        return _compact(f"{name}: {path}", 360)
    if name == "patch":
        mode = args.get("mode") or "replace"
        path = _redact_text(str(args.get("path") or ""))
        if not path and mode == "patch":
            path = "multi-file patch"
        return _compact(f"patch[{mode}]: {path}", 360)
    if name == "search_files":
        pattern = _redact_text(str(args.get("pattern") or ""))
        path = _redact_text(str(args.get("path") or "."))
        return _compact(f"search_files: {pattern} in {path}", 360)
    if name == "send_message":
        target = _redact_text(str(args.get("target") or ""))
        action = args.get("action") or "send"
        return _compact(f"send_message[{action}]: {target}", 360)
    if name == "delegate_task":
        tasks = args.get("tasks")
        if isinstance(tasks, list) and tasks:
            return f"delegate_task: {len(tasks)} task(s)"
        goal = _redact_text(str(args.get("goal") or ""))
        return _compact(f"delegate_task: {goal}", 360)
    if name == "skill_manage":
        action = args.get("action") or ""
        skill = args.get("name") or ""
        return _compact(f"skill_manage[{action}]: {skill}", 360)
    return _compact(f"{name}: {_json_summary(args)}", 360)


def output_digest(result: Any) -> str:
    """Return a non-secret digest for a tool result."""
    if result is None:
        text = ""
    elif isinstance(result, str):
        text = result
    else:
        try:
            text = json.dumps(result, ensure_ascii=False, sort_keys=True, default=str)
        except Exception:
            text = str(result)
    redacted = _redact_text(text)
    digest = hashlib.sha256(redacted.encode("utf-8", errors="replace")).hexdigest()[:16]
    return f"sha256:{digest}; chars={len(redacted)}"


def _error_preview(result: Any, limit: int = 240) -> Optional[str]:
    if result is None:
        return None
    if isinstance(result, str):
        text = result
    else:
        try:
            text = json.dumps(result, ensure_ascii=False, default=str)
        except Exception:
            text = str(result)
    return _compact(_redact_text(text), limit)


def record_tool_event(
    agent: Any,
    tool_name: str,
    args: Mapping[str, Any] | None,
    *,
    status: str,
    tool_call_id: str | None = None,
    started_at: float | None = None,
    ended_at: float | None = None,
    result: Any = None,
) -> Optional[int]:
    """Best-effort append of a redacted attribution event for a tool call."""
    db = getattr(agent, "_session_db", None)
    session_id = getattr(agent, "session_id", None)
    if not db or not session_id:
        return None
    try:
        ensure = getattr(agent, "_ensure_db_session", None)
        if callable(ensure):
            ensure()
    except Exception:
        pass
    try:
        now = time.time()
        started = float(started_at) if started_at is not None else now
        ended = float(ended_at) if ended_at is not None else (now if status != "started" else None)
        duration_ms = None
        if ended is not None and started is not None:
            duration_ms = max(0, int((ended - started) * 1000))
        failed_like = status in {"failed", "blocked", "cancelled"}
        return db.append_attribution_event(
            session_id=session_id,
            source=getattr(agent, "platform", None),
            user_id=getattr(agent, "_user_id", None),
            chat_id=getattr(agent, "_chat_id", None),
            thread_id=getattr(agent, "_thread_id", None),
            platform_message_id=getattr(agent, "_turn_platform_message_id", None),
            turn_id=getattr(agent, "_turn_id", None),
            tool_call_id=tool_call_id,
            tool_name=tool_name,
            action_summary=summarize_tool_call(tool_name, args),
            args_summary=_json_summary(args or {}),
            side_effect_class=classify_side_effect(tool_name, args),
            status=status,
            started_at=started,
            ended_at=ended,
            duration_ms=duration_ms,
            output_digest=output_digest(result) if result is not None else None,
            error_preview=_error_preview(result) if failed_like else None,
        )
    except Exception:
        return None


def _fmt_time(ts: Any) -> str:
    try:
        return datetime.fromtimestamp(float(ts)).strftime("%H:%M:%S")
    except Exception:
        return "??:??:??"


def _dedupe_started_events(events: Iterable[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    events = list(events or [])
    terminal_keys: set[str] = set()
    for event in events:
        status = str(event.get("status") or "")
        key = str(event.get("tool_call_id") or event.get("id") or "")
        if key and status in {"completed", "failed", "blocked", "cancelled"}:
            terminal_keys.add(key)
    result = []
    for event in events:
        key = str(event.get("tool_call_id") or event.get("id") or "")
        if event.get("status") == "started" and key in terminal_keys:
            continue
        result.append(event)
    return result


def _line(event: Mapping[str, Any]) -> str:
    eid = event.get("id", "?")
    session_id = str(event.get("session_id") or "?")
    sid_short = session_id[:18] + ("…" if len(session_id) > 18 else "")
    side = event.get("side_effect_class") or "other"
    tool = event.get("tool_name") or "tool"
    summary = event.get("action_summary") or tool
    when = _fmt_time(event.get("started_at"))
    extra = ""
    if event.get("error_preview"):
        extra = f" — {event.get('error_preview')}"
    return f"- {when} `{tool}` [{side}] {summary}（event#{eid}, session={sid_short}）{extra}"


def format_attribution_report(
    events: Iterable[Mapping[str, Any]],
    *,
    session_id: str,
    lineage_id: str | None = None,
    chat_id: str | None = None,
    limit: int = 100,
) -> str:
    """Format a strict Chinese attribution report from ledger events."""
    events = _dedupe_started_events(list(events or []))[:limit]
    completed = [e for e in events if e.get("status") == "completed"]
    failed = [e for e in events if e.get("status") in {"failed", "blocked", "cancelled"}]
    started = [e for e in events if e.get("status") == "started"]

    lines = [
        "## 我做了什么（attribution ledger）",
        f"当前 session: {session_id or 'unknown'}",
        f"lineage: {lineage_id or session_id or 'unknown'}",
    ]
    if chat_id:
        lines.append(f"chat: {chat_id}")
    lines.append("")

    if not events:
        lines.append("未发现可归因证据：当前 session/lineage ledger 没有匹配工具事件。")
        lines.append("")

    lines.append("### 已确认完成")
    lines.extend(_line(e) for e in completed) if completed else lines.append("- 无可确认项")
    lines.append("")
    lines.append("### 失败/被阻止")
    lines.extend(_line(e) for e in failed) if failed else lines.append("- 无可确认项")
    lines.append("")
    lines.append("### 仅看到开始、无完成记录")
    lines.extend(_line(e) for e in started) if started else lines.append("- 无可确认项")
    lines.append("")
    lines.append("不计入：git history、branch state、memory、session_search broad hits、compaction summary。")
    return "\n".join(lines)


def format_what_did_you_do_for_session(
    db: Any,
    *,
    session_id: str,
    chat_id: str | None = None,
    source: str | None = None,
    limit: int = 100,
) -> str:
    """Query the ledger for a session lineage and format the report."""
    lineage_id = None
    if db is not None and session_id:
        try:
            lineage_id = db.get_session_lineage_id(session_id)
        except Exception:
            lineage_id = session_id
    events = []
    if db is not None and session_id:
        try:
            events = db.get_attribution_events(
                session_id=session_id,
                include_lineage=True,
                chat_id=chat_id,
                source=source,
                limit=limit,
            )
        except Exception:
            events = []
    return format_attribution_report(
        events,
        session_id=session_id,
        lineage_id=lineage_id or session_id,
        chat_id=chat_id,
        limit=limit,
    )
