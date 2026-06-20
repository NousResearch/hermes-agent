"""Lightweight in-memory runtime status for UI statusbars.

This module is intentionally small and process-local. Producers update it
when runtime events happen; renderers read snapshots. Snapshot creation must
stay cheap: no transcript parsing, no disk IO, no CLI/TUI imports.
"""

from __future__ import annotations

from copy import deepcopy
from threading import RLock
import time
from typing import Any, Dict, List, Optional

_MAX_RECENT = 8
_LOCK = RLock()
_SESSIONS: Dict[str, Dict[str, Any]] = {}


def _sid(session_id: Optional[str]) -> str:
    return str(session_id or "")


def _now() -> float:
    return time.time()


def _coerce_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return default


def _default_state() -> Dict[str, Any]:
    return {
        "phase": "idle",
        "run_mode": "idle",
        "target": "",
        "main_agent": "main",
        "recent_tools": [],
        "recent_tool": None,
        "recent_skills": [],
        "recent_skill": None,
        "active_subagent": None,
        "background_tasks": {"running": 0},
        "task": None,
        "wait": {"reason": "none", "since": None},
    }


def _state(session_id: Optional[str]) -> Dict[str, Any]:
    sid = _sid(session_id)
    state = _SESSIONS.get(sid)
    if state is None:
        state = _default_state()
        _SESSIONS[sid] = state
    return state


def _append_capped(items: List[Dict[str, Any]], item: Dict[str, Any]) -> None:
    items.append(item)
    if len(items) > _MAX_RECENT:
        del items[: len(items) - _MAX_RECENT]


def clear_session(session_id: Optional[str]) -> None:
    """Drop runtime status for a session."""
    with _LOCK:
        _SESSIONS.pop(_sid(session_id), None)


def record_phase(
    session_id: Optional[str],
    *,
    phase: str = "",
    run_mode: str = "",
    target: str = "",
    main_agent: str = "",
) -> None:
    """Record high-level runtime phase fields."""
    with _LOCK:
        state = _state(session_id)
        if phase:
            state["phase"] = str(phase)
        if run_mode:
            state["run_mode"] = str(run_mode)
        if target is not None:
            state["target"] = str(target)
        if main_agent:
            state["main_agent"] = str(main_agent)


def record_tool_started(
    session_id: Optional[str],
    tool_name: str,
    *,
    preview: str = "",
    tool_call_id: str = "",
) -> None:
    """Record a running tool as the most recent tool."""
    if not tool_name:
        return
    ts = _now()
    item = {
        "name": str(tool_name),
        "status": "running",
        "preview": str(preview or ""),
        "tool_call_id": str(tool_call_id or ""),
        "started_at": ts,
        "ended_at": None,
        "duration_ms": None,
        "error_message": "",
    }
    with _LOCK:
        state = _state(session_id)
        _append_capped(state["recent_tools"], item)
        state["recent_tool"] = item
        state["phase"] = "tool"
        if state.get("run_mode") in {"", "idle", "agent"}:
            state["run_mode"] = "agent"
        state["wait"] = {"reason": f"tool:{tool_name}", "since": ts}


def record_tool_completed(
    session_id: Optional[str],
    tool_name: str,
    *,
    status: str = "ok",
    duration_ms: int = 0,
    error_message: str = "",
    tool_call_id: str = "",
) -> None:
    """Record a completed tool and its final status."""
    if not tool_name:
        return
    normalized_status = str(status or "ok")
    if normalized_status not in {"running", "ok", "error", "blocked"}:
        normalized_status = "error" if normalized_status.lower() in {"failed", "failure"} else "ok"

    ts = _now()
    duration = _coerce_int(duration_ms, 0)
    call_id = str(tool_call_id or "")
    with _LOCK:
        state = _state(session_id)
        target = None
        if call_id:
            for item in reversed(state["recent_tools"]):
                if item.get("tool_call_id") == call_id:
                    target = item
                    break
        recent_tool = state.get("recent_tool")
        if target is None and isinstance(recent_tool, dict) and recent_tool.get("name") == tool_name:
            target = recent_tool
        if target is None:
            target = {
                "name": str(tool_name),
                "preview": "",
                "tool_call_id": call_id,
                "started_at": None,
            }
            _append_capped(state["recent_tools"], target)

        target.update(
            {
                "name": str(tool_name),
                "status": normalized_status,
                "ended_at": ts,
                "duration_ms": duration,
                "error_message": str(error_message or ""),
            }
        )
        target.setdefault("preview", "")
        target.setdefault("started_at", None)
        target.setdefault("tool_call_id", call_id)
        state["recent_tool"] = target
        state["phase"] = "running"
        if state.get("run_mode") in {"", "idle", "agent"}:
            state["run_mode"] = "agent"
        state["wait"] = {"reason": "none", "since": None}


def record_skill(session_id: Optional[str], skill_name: str, *, event: str = "use") -> None:
    """Record a recent skill event."""
    if not skill_name:
        return
    item = {
        "name": str(skill_name),
        "event": str(event or "use"),
        "ts": _now(),
    }
    with _LOCK:
        state = _state(session_id)
        _append_capped(state["recent_skills"], item)
        state["recent_skill"] = item


def summarize_active_subagents(records: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Return a compact statusbar summary for active subagent records."""
    candidates = [r for r in records if isinstance(r, dict)]
    if not candidates:
        return None

    def _rank(record: Dict[str, Any]) -> tuple[int, float]:
        running = 1 if str(record.get("status") or "") == "running" else 0
        has_tool = 1 if record.get("last_tool") else 0
        started = float(record.get("started_at") or 0)
        return (running * 2 + has_tool, started)

    picked = max(candidates, key=_rank)
    sid = str(picked.get("subagent_id") or picked.get("id") or "")
    goal = str(picked.get("goal") or sid or "subagent").strip()
    label = goal[:48].rstrip() + ("…" if len(goal) > 48 else "")
    return {
        "id": sid,
        "label": label,
        "status": str(picked.get("status") or "running"),
        "last_tool": str(picked.get("last_tool") or ""),
        "tool_count": _coerce_int(picked.get("tool_count"), 0),
    }


def record_active_subagents(session_id: Optional[str], records: List[Dict[str, Any]]) -> None:
    """Record a summarized active subagent for a session."""
    with _LOCK:
        _state(session_id)["active_subagent"] = summarize_active_subagents(records)


def record_background_process_count(session_id: Optional[str], running: int) -> None:
    """Record the current process-registry running count for statusbars."""
    with _LOCK:
        _state(session_id)["background_tasks"] = {"running": max(0, _coerce_int(running, 0))}


def _activity_phase_and_wait(summary: Dict[str, Any]) -> tuple[str, str]:
    current_tool = str(summary.get("current_tool") or "").strip()
    if current_tool:
        return "tool", f"tool:{current_tool}"

    desc = str(summary.get("last_activity_desc") or "").lower()
    if "retry backoff" in desc or "rate limited" in desc or "retrying in" in desc:
        return "waiting", "retry_backoff"
    if "starting api call" in desc or "api call" in desc or "model" in desc:
        return "thinking", "model"
    return "running", "none"


def record_activity_summary(session_id: Optional[str], summary: Dict[str, Any]) -> None:
    """Record lightweight phase/wait/task fields from AIAgent activity summary."""
    if not isinstance(summary, dict) or not summary:
        return

    interesting_keys = {"current_tool", "last_activity_desc", "budget_used", "budget_max"}
    if not any(key in summary for key in interesting_keys):
        return

    phase, wait_reason = _activity_phase_and_wait(summary)
    with _LOCK:
        state = _state(session_id)
        state["phase"] = phase
        if state.get("run_mode") in {"", "idle", "agent"}:
            state["run_mode"] = "agent"
        state["wait"] = {"reason": wait_reason, "since": None if wait_reason == "none" else _now()}

    budget_max = _coerce_int(summary.get("budget_max"), 0)
    if budget_max > 0:
        completed = min(max(_coerce_int(summary.get("budget_used"), 0), 0), budget_max)
        record_task_summary(
            session_id,
            {
                "total": budget_max,
                "completed": completed,
                "in_progress": 0,
                "pending": max(budget_max - completed, 0),
                "cancelled": 0,
            },
        )


def record_task_summary(session_id: Optional[str], summary: Dict[str, Any]) -> None:
    """Record normalized todo/task progress."""
    if not isinstance(summary, dict):
        return
    task = {
        "total": _coerce_int(summary.get("total"), 0),
        "completed": _coerce_int(summary.get("completed"), 0),
        "in_progress": _coerce_int(summary.get("in_progress"), 0),
        "pending": _coerce_int(summary.get("pending"), 0),
        "cancelled": _coerce_int(summary.get("cancelled"), 0),
    }
    with _LOCK:
        _state(session_id)["task"] = task


def record_wait(session_id: Optional[str], *, reason: str = "none") -> None:
    """Record the current wait/blocker reason."""
    normalized = str(reason or "none")
    wait = {"reason": normalized, "since": None if normalized == "none" else _now()}
    with _LOCK:
        state = _state(session_id)
        state["wait"] = wait
        state["phase"] = "running" if normalized == "none" else "waiting"
        if state.get("run_mode") in {"", "idle", "agent"}:
            state["run_mode"] = "agent"


def snapshot(session_id: Optional[str]) -> Dict[str, Any]:
    """Return a JSON-serializable copy of a session runtime snapshot."""
    with _LOCK:
        return deepcopy(_state(session_id))
