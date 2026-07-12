#!/usr/bin/env python3
"""Agent Live Dashboard — ASCII real-time health monitor.

Usage:  python -m agent.dashboard [--interval 1]
        hermes status --live    (future CLI integration)

Displays a live-refreshing dashboard of agent health:
  - Tool execution stats (success rate, recent events)
  - Task progress (phase, budget, completed steps)
  - Multi-agent state (mailbox, blackboard, active agents)
  - Error breakdown (by class)

Reads from state.db + execution_tracer ring buffer.
No new dependencies — uses only stdlib + hermes internals.
"""

from __future__ import annotations

import json
import os
import sys
import time
from datetime import datetime

# ── Terminal helpers ────────────────────────────────────────────────────


def _clear() -> None:
    sys.stdout.write("\033[2J\033[H")


def _bar(pct: float, width: int = 20) -> str:
    filled = round(pct * width)
    blocks = ["█", "▓", "▒", "░", " "]
    bar = ""
    for i in range(width):
        idx = min(3, max(0, round((i - filled) * 4))) if i >= filled else 0
        bar += blocks[idx]
    return bar


def _color(status: str) -> str:
    if status.startswith("✓"):
        return f"\033[32m{status}\033[0m"
    if status.startswith("✗"):
        return f"\033[31m{status}\033[0m"
    if status.startswith("↻"):
        return f"\033[33m{status}\033[0m"
    return status


# ── Data gatherers ──────────────────────────────────────────────────────


def _get_tracer_stats(session_id: str) -> dict:
    try:
        from agent.execution_tracer import get_session_stats, get_recent_events
        stats = get_session_stats(session_id)
        events = get_recent_events(session_id, limit=10)
        return {"stats": stats, "events": events}
    except Exception:
        return {"stats": {}, "events": []}


def _get_task_state(session_id: str) -> dict | None:
    try:
        from hermes_state import SessionDB
        db = SessionDB()
        return db.load_task_checkpoint(session_id)
    except Exception:
        return None


def _get_mailbox_state(session_id: str) -> list:
    try:
        from hermes_state import SessionDB
        db = SessionDB()
        return db.read_agent_mailbox(f"session:{session_id}", mark_read=False)
    except Exception:
        return []


def _get_active_sessions() -> list:
    try:
        from hermes_state import SessionDB
        db = SessionDB()

        def _do(conn):
            rows = conn.execute(
                "SELECT id, source, started_at FROM sessions "
                "WHERE started_at > ? ORDER BY started_at DESC LIMIT 10",
                (time.time() - 86400,),
            ).fetchall()
            return [{"id": r[0], "source": r[1], "at": r[2]} for r in rows]

        return db._execute_write(_do)
    except Exception:
        return []


# ── Dashboard renderer ──────────────────────────────────────────────────


def render(session_id: str = "") -> str:
    """Render a single-frame dashboard as a string."""
    if not session_id:
        sessions = _get_active_sessions()
        if sessions:
            session_id = sessions[0]["id"]

    now = datetime.now().strftime("%H:%M:%S")
    trace = _get_tracer_stats(session_id)
    task = _get_task_state(session_id)
    mailbox = _get_mailbox_state(session_id)
    stats = trace.get("stats", {})
    events = trace.get("events", [])

    # ── Header ──
    lines = []
    lines.append("┌" + "─" * 56 + "┐")
    title = f"│ HERMES AGENT STATUS  —  {now:8s}  session: {session_id[:16]:16s} │"
    lines.append(title[:59] + "│")

    # ── Health ──
    rate = stats.get("success_rate", 0)
    health_pct = int(rate * 100)
    lines.append("├" + "─" * 56 + "┤")
    lines.append(f"│ Health: {_bar(rate)} {health_pct:3d}%                        │")

    # ── Tool stats ──
    total = stats.get("total_tools", 0)
    ok = stats.get("succeeded", 0)
    fail = stats.get("failed", 0)
    recov = stats.get("recovered", 0)
    lines.append(f"│ Tools:  {ok:4d} succeeded, {fail:2d} failed, {recov:2d} recovered  {' ' * 17}│")

    # ── Budget ──
    if task:
        budget = task.get("iteration_budget_remaining", 0)
        phase = task.get("current_phase", "—")
        completed = len(task.get("completed_tool_calls", []))
        lines.append(f"│ Budget: {budget:3d} turns left · Phase: {phase[:30]:30s} │")
        lines.append(f"│ Done:   {completed:3d} steps completed                   {' ' * 20}│")
    else:
        lines.append(f"│ {'(no active task)':56s} │")

    # ── Mailbox ──
    unread = len(mailbox)
    lines.append(f"│ Mailbox: {unread:3d} unread messages                        {' ' * 21}│")

    # ── Error breakdown ──
    lines.append("├" + "─" * 56 + "┤")
    breakdown = stats.get("error_breakdown", {})
    if breakdown:
        lines.append("│ Error breakdown:                                       │")
        for cls, cnt in sorted(breakdown.items(), key=lambda x: -x[1])[:5]:
            cls_short = cls[:22]
            lines.append(f"│   {cls_short:22s}  {cnt:3d}                                │")
    else:
        lines.append("│ No errors recorded                                     │")

    # ── Recent events ──
    lines.append("├" + "─" * 56 + "┤")
    lines.append("│ Recent events:                                         │")
    if events:
        for e in events[:5]:
            icon = "✓" if e["success"] else ("↻" if e.get("recovery_action") else "✗")
            tool = e["tool"][:12]
            summary = e.get("result_summary", "") or e.get("error_class", "")[:22]
            age = max(0, int(time.time() - e.get("at", time.time())))
            age_str = f"{age}s ago" if age < 120 else f"{age//60}m ago"
            lines.append(f"│  {_color(icon)} {age_str:8s}  {tool:12s}  {summary[:22]:22s} │")
    else:
        lines.append("│  (no events yet)                                       │")

    lines.append("└" + "─" * 56 + "┘")
    lines.append(f"  Refresh: Ctrl+C to exit · {now}")
    return "\n".join(lines)


# ── Live loop ───────────────────────────────────────────────────────────


def live(session_id: str = "", interval: float = 2.0) -> None:
    """Run live dashboard with periodic refresh."""
    import signal

    running = True

    def _stop(sig, frame):
        nonlocal running
        running = False

    signal.signal(signal.SIGINT, _stop)

    try:
        while running:
            _clear()
            print(render(session_id))
            sys.stdout.flush()
            time.sleep(interval)
    except KeyboardInterrupt:
        pass
    finally:
        _clear()
        print("[dashboard exited]")


# ── CLI entry point ────────────────────────────────────────────────────


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Hermes Agent Live Dashboard")
    ap.add_argument("--session", "-s", default="", help="Session ID to monitor")
    ap.add_argument("--interval", "-i", type=float, default=2.0, help="Refresh interval (seconds)")
    ap.add_argument("--once", action="store_true", help="Print one frame and exit")
    args = ap.parse_args()

    if args.once:
        print(render(args.session))
    else:
        live(args.session, interval=args.interval)