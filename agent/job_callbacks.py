from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from hermes_constants import get_hermes_home


def _completion_text(envelope: dict[str, Any]) -> str:
    final_output = (envelope.get("final_output") or "").strip()
    if final_output:
        return final_output
    summary = (envelope.get("summary") or "").strip()
    if summary:
        return summary
    error = (envelope.get("error") or "").strip()
    if error:
        return error
    return ""


def _sessions_dir() -> Path:
    return get_hermes_home() / "sessions"


def _load_session_entry(session_id: str) -> dict[str, Any] | None:
    sessions_file = _sessions_dir() / "sessions.json"
    if not sessions_file.exists():
        return None
    try:
        data = json.loads(sessions_file.read_text(encoding="utf-8"))
    except Exception:
        return None
    for entry in data.values():
        if isinstance(entry, dict) and entry.get("session_id") == session_id:
            return entry
    return None


def _append_transcript_message(session_id: str, message: dict[str, Any]) -> None:
    sessions_dir = _sessions_dir()
    sessions_dir.mkdir(parents=True, exist_ok=True)
    transcript_path = sessions_dir / f"{session_id}.jsonl"
    with open(transcript_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(message, ensure_ascii=False) + "\n")
    try:
        from hermes_state import SessionDB

        db = SessionDB()
        try:
            db.ensure_session(session_id, source="callback")
            db.append_message(
                session_id=session_id,
                role=message.get("role", "assistant"),
                content=message.get("content"),
                tool_name=message.get("tool_name"),
                tool_calls=message.get("tool_calls"),
                tool_call_id=message.get("tool_call_id"),
            )
        finally:
            db.close()
    except Exception:
        pass


def deliver_completion(
    envelope: dict[str, Any],
    *,
    adapters=None,
    loop=None,
) -> str | None:
    callback = envelope.get("callback") or {"type": "none"}
    callback_type = str(callback.get("type") or "none").lower()
    if callback_type in {"", "none"}:
        return None
    if callback_type == "platform":
        target = callback.get("target") or {}
        platform = target.get("platform")
        chat_id = target.get("chat_id")
        thread_id = target.get("thread_id")
        if not platform or not chat_id:
            return "platform callback missing target platform/chat_id"

        deliver = f"{platform}:{chat_id}"
        if thread_id not in (None, ""):
            deliver = f"{deliver}:{thread_id}"

        job = {
            "id": envelope.get("task_id") or "queued-task",
            "name": envelope.get("kind") or "queued-task",
            "deliver": deliver,
            "origin": {
                "platform": platform,
                "chat_id": str(chat_id),
                "thread_id": None if thread_id in (None, "") else str(thread_id),
            },
        }

        from cron.scheduler import _deliver_result

        return _deliver_result(job, _completion_text(envelope), adapters=adapters, loop=loop)
    if callback_type == "session":
        session_id = callback.get("session_id")
        if not session_id:
            return "session callback missing session_id"
        message = {
            "role": "assistant",
            "content": _completion_text(envelope),
            "tool_name": None,
            "tool_calls": None,
            "tool_call_id": None,
        }
        _append_transcript_message(session_id, message)

        session_entry = _load_session_entry(session_id)
        origin = (session_entry or {}).get("origin") or {}
        platform = origin.get("platform")
        chat_id = origin.get("chat_id")
        if not platform or not chat_id or platform == "local":
            return None

        deliver = f"{platform}:{chat_id}"
        thread_id = origin.get("thread_id")
        if thread_id not in (None, ""):
            deliver = f"{deliver}:{thread_id}"
        job = {
            "id": envelope.get("task_id") or session_id,
            "name": envelope.get("kind") or "delegation",
            "deliver": deliver,
            "origin": {
                "platform": platform,
                "chat_id": str(chat_id),
                "thread_id": None if thread_id in (None, "") else str(thread_id),
            },
        }
        from cron.scheduler import _deliver_result

        return _deliver_result(job, message["content"], adapters=adapters, loop=loop)
    return f"unsupported callback type '{callback_type}'"
