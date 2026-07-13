"""Agent Dashboard — ASCII formatter + live view for tracer data.

Reads from state.db tables (agent_traces, task_checkpoints, agent_messages,
agent_blackboard, sessions) and renders a compact terminal view.

Usage:
    from agent.dashboard import render_snapshot, render_live
    print(render_snapshot(session_id))
"""

from __future__ import annotations

import datetime as _dt
import os
import time
from typing import Any, Dict, List, Optional


# ── ASCII bar helper ───────────────────────────────────────────────────


def _bar(pct: float, width: int = 20) -> str:
    """Return a filled-bar like `████████░░░░░░░░░░ 40%`."""
    pct = max(0.0, min(1.0, pct))
    filled = int(pct * width)
    empty = width - filled
    return f"{'█' * filled}{'░' * empty} {pct * 100:.0f}%"


# ── Data collection ────────────────────────────────────────────────────


def _collect_snapshot(session_id: str) -> Dict[str, Any]:
    """Pull all relevant data for *session_id* from state.db."""
    try:
        from hermes_state import SessionDB
        db = SessionDB()

        def _q(sql, params=()):
            rows = db._execute_write(lambda c: list(c.execute(sql, params)))
            return [tuple(r) for r in rows]

        def _scalar(sql, params=()):
            r = _q(sql, params)
            return int(r[0][0] or 0) if r else 0

        # Session row
        sess = _q("SELECT id, source, started_at, model FROM sessions WHERE id = ?", (session_id,))
        session_info = {}
        if sess:
            sid, source, started_at, model = sess[0]
            session_info = {"id": sid, "source": source, "started_at": started_at, "model": model}

        # Checkpoint
        cp = db.load_task_checkpoint(session_id) or {}

        # Stats
        total_tools = _scalar("SELECT COUNT(*) FROM agent_traces WHERE session_id = ?", (session_id,))
        succeeded = _scalar("SELECT COUNT(*) FROM agent_traces WHERE session_id = ? AND success = 1", (session_id,))
        failed = _q(
            "SELECT error_class, COUNT(*) FROM agent_traces "
            "WHERE session_id = ? AND success = 0 GROUP BY error_class",
            (session_id,),
        )
        by_tool = _q(
            "SELECT tool_name, COUNT(*) FROM agent_traces "
            "WHERE session_id = ? GROUP BY tool_name ORDER BY 2 DESC",
            (session_id,),
        )
        avg_dur = _scalar("SELECT AVG(duration_ms) FROM agent_traces WHERE session_id = ?", (session_id,))

        # Mailbox
        unread = _scalar(
            "SELECT COUNT(*) FROM agent_messages "
            "WHERE read_at IS NULL AND (to_id = ? OR to_id = '*')",
            (f"session:{session_id}",),
        )

        return {
            "session": session_info,
            "checkpoint": cp,
            "total_tools": total_tools,
            "succeeded": succeeded,
            "failed_total": total_tools - succeeded,
            "errors_by_class": {r[0] or "unknown": r[1] for r in failed},
            "calls_by_tool": {r[0]: r[1] for r in by_tool},
            "avg_duration_ms": round(float(avg_dur or 0), 1),
            "unread_mailbox": unread,
        }
    except Exception as exc:
        return {"error": str(exc)}


# ── Section renderers ──────────────────────────────────────────────────


def _render_header(session_id: str) -> List[str]:
    now = _dt.datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
    return [
        "┌" + "─" * 62 + "┐",
        f"│ HERMES AGENT STATUS{' ' * 32}{now} │",
        "├" + "─" * 62 + "┤",
    ]


def _render_section_session(session_id: str, data: Dict[str, Any]) -> List[str]:
    s = data.get("session", {})
    cp = data.get("checkpoint", {})
    goal = (cp.get("task_goal") or "")[:40]
    phase = cp.get("current_phase") or "—"
    budget_remaining = int(cp.get("iteration_budget_remaining") or 0)
    resume_count = int(cp.get("resume_count") or 0)

    # budget_remaining is what's LEFT; budget_used/max = consumed
    budget_max = budget_remaining + resume_count or 50
    budget_used = max(0, budget_max - budget_remaining)
    budget_pct = budget_used / max(budget_max, 1)

    health_pct = (data.get("succeeded", 0) or 0) / max(data.get("total_tools", 1), 1)
    model = (s.get("model") or "—")[:12]

    return [
        f"│ Session:    {session_id[:30]:30s}  model: {model:12s} │",
        f"│ Goal:       {goal:54s}│",
        f"│ Phase:      {phase:54s}│",
        f"│ Budget:     {_bar(budget_pct):30s}   │",
        f"│ Health:     {_bar(health_pct):30s}   │",
        "├" + "─" * 62 + "┤",
    ]


def _render_section_tools(data: Dict[str, Any]) -> List[str]:
    total = data.get("total_tools", 0)
    succ = data.get("succeeded", 0)
    failed = data.get("failed_total", 0)
    by_tool = data.get("calls_by_tool", {})

    top_tools = sorted(by_tool.items(), key=lambda kv: -kv[1])[:5]
    tools_str = ", ".join(f"{n}:{c}" for n, c in top_tools)[:50]

    err_lines = []
    for cls, cnt in data.get("errors_by_class", {}).items():
        err_lines.append(f"│     • {cls:20s} {cnt:>3d}{' ' * 36}│")

    if total:
        first_line = f"│   total:      {total:4d}   ({tools_str:50s})│"
    else:
        first_line = "│   total:      0   " + " " * 45 + "│"

    body = [
        "│ TOOL CALLS" + " " * 53 + "│",
        first_line,
        f"│   succeeded:  {succ:4d}{' ' * 50}│",
        f"│   failed:     {failed:4d}{' ' * 50}│",
    ] + err_lines + [
        f"│   avg duration: {data.get('avg_duration_ms', 0):.0f}ms{' ' * 41}│",
        "├" + "─" * 62 + "┤",
    ]
    return body


def _render_section_mailbox(data: Dict[str, Any]) -> List[str]:
    unread = data.get("unread_mailbox", 0)
    return [
        "│ MAILBOX" + " " * 55 + "│",
        f"│   unread: {unread:5d}{' ' * 49}│",
        "└" + "─" * 62 + "┘",
    ]


# ── Public API ─────────────────────────────────────────────────────────


def render_snapshot(session_id: str) -> str:
    """Render a single dashboard snapshot for *session_id*. Returns string."""
    data = _collect_snapshot(session_id)

    if "error" in data:
        return f"[dashboard unavailable: {data['error']}]"

    lines = [
        *_render_header(session_id),
        *_render_section_session(session_id, data),
        *_render_section_tools(data),
        *_render_section_mailbox(data),
        f"  Last refresh: {_dt.datetime.utcnow().strftime('%H:%M:%S')} UTC",
    ]
    return "\n".join(lines)


def render_live(
    session_id: str,
    refresh_seconds: float = 2.0,
    max_iters: Optional[int] = None,
) -> None:
    """Continuously render dashboard until interrupted.

    Press Ctrl+C to stop. Pass *max_iters* for non-interactive runs.
    """
    i = 0
    while max_iters is None or i < max_iters:
        if os.name == "nt":
            os.system("cls")
        else:
            os.system("clear")
        print(render_snapshot(session_id))
        i += 1
        if max_iters is not None and i >= max_iters:
            return
        time.sleep(refresh_seconds)