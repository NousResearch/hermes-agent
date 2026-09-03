"""``hermes inbox`` — one view of everything in flight and needing attention.

Inspired by Cursor's Inbox (changelog Jul 29 2026): a single surface showing
"what's in progress, what needs your attention, and what finished" across
their agents. Hermes already tracks all of that state, but scattered across
four stores with no unified read surface:

- background processes  → ``~/.hermes/processes.json`` (ProcessRegistry
  checkpoint; PIDs re-validated here the same way recovery does)
- async delegations     → ``state.db``'s ``async_delegations`` table
  (running / stalled / undelivered / dropped results)
- cron jobs             → ``cron/jobs.json`` (failures, pauses, next runs)
- open chat surfaces    → ``runtime/active_sessions.json`` leases

This module aggregates them read-only. It never mutates any store, never
imports the agent core, and works whether or not a gateway is running.

Layout follows the footprint ladder: a CLI command, not a model tool — the
agent can run ``hermes inbox --json`` through its terminal when asked.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from hermes_constants import get_hermes_home

# ---------------------------------------------------------------------------
# Collectors — each returns a list of plain dicts and NEVER raises.
# ---------------------------------------------------------------------------


def _pid_alive(pid: Any) -> bool:
    try:
        pid = int(pid)
    except (TypeError, ValueError):
        return False
    if pid <= 0:
        return False
    try:
        import psutil

        return psutil.pid_exists(pid)
    except Exception:
        return False


def collect_processes(home: Optional[Path] = None) -> List[Dict[str, Any]]:
    """Background processes from the ProcessRegistry checkpoint.

    The checkpoint only holds not-yet-exited sessions at write time, but the
    owning process may have died since (or the child may have exited without
    a checkpoint rewrite), so each host PID is re-validated.
    """
    home = home or get_hermes_home()
    path = home / "processes.json"
    out: List[Dict[str, Any]] = []
    try:
        entries = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return out
    if not isinstance(entries, list):
        return out
    now = time.time()
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        pid = entry.get("pid")
        scope = entry.get("pid_scope", "host")
        alive = _pid_alive(pid) if scope == "host" else None  # None = unknown
        started = entry.get("started_at")
        age = None
        if isinstance(started, (int, float)):
            age = max(0.0, now - float(started))
        out.append(
            {
                "kind": "process",
                "id": entry.get("session_id", "?"),
                "command": str(entry.get("command", ""))[:120],
                "pid": pid,
                "alive": alive,
                "age_seconds": age,
                "notify_on_complete": bool(entry.get("notify_on_complete")),
                "cwd": entry.get("cwd"),
            }
        )
    return out


def collect_delegations(
    home: Optional[Path] = None, limit: int = 20
) -> List[Dict[str, Any]]:
    """Async delegations from state.db — running, plus recent non-delivered.

    Reads sqlite directly (read-only URI) so a missing table or a locked
    database degrades to an empty section instead of an error.
    """
    home = home or get_hermes_home()
    db = home / "state.db"
    out: List[Dict[str, Any]] = []
    if not db.exists():
        return out
    import sqlite3

    # Terminal 'dropped' results older than this are historical noise, not
    # actionable — hide them from the inbox (they remain queryable in the DB).
    dropped_cutoff = time.time() - 7 * 86400

    try:
        conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True, timeout=5)
    except sqlite3.Error:
        return out
    try:
        rows = conn.execute(
            """SELECT delegation_id, state, dispatched_at, completed_at,
                      delivery_state, delivery_attempts, task_json
               FROM async_delegations
               WHERE state IN ('running', 'finalizing', 'stalling')
                  OR delivery_state = 'pending'
                  OR (delivery_state = 'dropped'
                      AND COALESCE(completed_at, dispatched_at) >= ?)
               ORDER BY CASE WHEN state IN ('running','finalizing','stalling')
                             THEN 0 ELSE 1 END,
                        COALESCE(completed_at, dispatched_at) DESC
               LIMIT ?""",
            (dropped_cutoff, limit),
        ).fetchall()
    except sqlite3.Error:
        return out
    finally:
        conn.close()
    for (
        delegation_id,
        state,
        dispatched_at,
        completed_at,
        delivery_state,
        delivery_attempts,
        task_json,
    ) in rows:
        goal = ""
        try:
            task = json.loads(task_json or "{}")
            goal = str(task.get("goal") or "")[:100]
        except ValueError:
            pass
        out.append(
            {
                "kind": "delegation",
                "id": delegation_id,
                "state": state,
                "goal": goal,
                "dispatched_at": dispatched_at,
                "completed_at": completed_at,
                "delivery_state": delivery_state,
                "delivery_attempts": delivery_attempts,
            }
        )
    return out


def collect_cron(home: Optional[Path] = None) -> List[Dict[str, Any]]:
    """Cron jobs needing attention (failed / paused) + the next few due."""
    out: List[Dict[str, Any]] = []
    try:
        from cron.jobs import load_jobs

        jobs = load_jobs()
    except Exception:
        return out
    for job in jobs:
        if not isinstance(job, dict):
            continue
        needs_attention = bool(job.get("last_error")) or (
            job.get("last_status") not in (None, "", "success", "ok")
        )
        paused = job.get("state") == "paused" or not job.get("enabled", True)
        out.append(
            {
                "kind": "cron",
                "id": job.get("id", "?"),
                "name": job.get("name", "?"),
                "enabled": bool(job.get("enabled", True)),
                "paused": paused,
                "paused_reason": job.get("paused_reason"),
                "last_status": job.get("last_status"),
                "last_error": (str(job.get("last_error"))[:160]
                               if job.get("last_error") else None),
                "last_run_at": job.get("last_run_at"),
                "next_run_at": job.get("next_run_at"),
                "needs_attention": needs_attention,
            }
        )
    return out


def collect_active_sessions(home: Optional[Path] = None) -> List[Dict[str, Any]]:
    """Open chat surfaces from the active-session lease file."""
    home = home or get_hermes_home()
    path = home / "runtime" / "active_sessions.json"
    out: List[Dict[str, Any]] = []
    try:
        entries = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return out
    if not isinstance(entries, list):
        return out
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        if not _pid_alive(entry.get("pid")):
            continue
        out.append(
            {
                "kind": "session",
                "surface": entry.get("surface") or entry.get("source") or "?",
                "pid": entry.get("pid"),
                "started_at": entry.get("started_at") or entry.get("acquired_at"),
                "label": entry.get("label") or entry.get("session_id") or "",
            }
        )
    return out


# ---------------------------------------------------------------------------
# Assembly + rendering
# ---------------------------------------------------------------------------


def build_inbox(home: Optional[Path] = None) -> Dict[str, Any]:
    """Assemble the full inbox snapshot as JSON-serializable data."""
    home = home or get_hermes_home()
    processes = collect_processes(home)
    delegations = collect_delegations(home)
    cron_jobs = collect_cron(home)
    sessions = collect_active_sessions(home)

    attention: List[Dict[str, Any]] = []
    for d in delegations:
        if d["state"] in ("stalling", "stalled", "unknown") or d[
            "delivery_state"
        ] == "dropped":
            attention.append(d)
    for j in cron_jobs:
        if j["needs_attention"] or (j["paused"] and j.get("paused_reason")):
            attention.append(j)
    for p in processes:
        # Checkpointed as running but the PID is gone and nobody rewrote the
        # checkpoint — likely an orphaned entry from a crashed owner.
        if p["alive"] is False:
            attention.append(p)

    in_progress: List[Dict[str, Any]] = []
    in_progress.extend(p for p in processes if p["alive"] is not False)
    in_progress.extend(
        d for d in delegations if d["state"] in ("running", "finalizing")
    )

    finished: List[Dict[str, Any]] = [
        d
        for d in delegations
        if d["state"] not in ("running", "finalizing", "stalling")
        and d["delivery_state"] == "pending"
    ]

    return {
        "generated_at": time.time(),
        "attention": attention,
        "in_progress": in_progress,
        "finished_undelivered": finished,
        "cron": cron_jobs,
        "sessions": sessions,
    }


def _fmt_age(seconds: Optional[float]) -> str:
    if not isinstance(seconds, (int, float)):
        return "?"
    s = int(max(0, seconds))
    if s < 60:
        return f"{s}s"
    m, s = divmod(s, 60)
    if m < 60:
        return f"{m}m"
    h, m = divmod(m, 60)
    if h < 24:
        return f"{h}h{m:02d}m"
    d, h = divmod(h, 24)
    return f"{d}d{h}h"


def _describe(item: Dict[str, Any], now: float) -> str:
    kind = item.get("kind")
    if kind == "process":
        age = _fmt_age(item.get("age_seconds"))
        state = (
            "running" if item.get("alive")
            else "GONE (stale checkpoint)" if item.get("alive") is False
            else "sandbox"
        )
        return f"process {item['id']}  [{state}, {age}]  {item['command']}"
    if kind == "delegation":
        when = item.get("completed_at") or item.get("dispatched_at")
        age = _fmt_age(now - when) if isinstance(when, (int, float)) else "?"
        bits = f"delegation {item['id']}  [{item['state']}, {age} ago]"
        if item.get("delivery_state") == "dropped":
            bits += "  (result DROPPED after failed deliveries)"
        elif item.get("delivery_state") == "pending" and item["state"] not in (
            "running",
            "finalizing",
        ):
            bits += "  (result awaiting delivery)"
        if item.get("goal"):
            bits += f"  {item['goal']}"
        return bits
    if kind == "cron":
        line = f"cron '{item['name']}'"
        if item.get("last_error"):
            line += f"  last run FAILED: {item['last_error']}"
        elif item.get("paused"):
            line += f"  paused ({item.get('paused_reason') or 'manually'})"
        elif item.get("last_status") not in (None, "", "success", "ok"):
            line += f"  last status: {item['last_status']}"
        return line
    if kind == "session":
        return (
            f"session [{item.get('surface')}] pid={item.get('pid')}"
            f"  {item.get('label') or ''}".rstrip()
        )
    return json.dumps(item)


def render_inbox(data: Dict[str, Any]) -> str:
    """Human-readable inbox text."""
    now = data.get("generated_at") or time.time()
    lines: List[str] = [""]

    attention = data.get("attention") or []
    lines.append(f"  ◆ Needs attention ({len(attention)})")
    if attention:
        for item in attention:
            lines.append(f"    ⚠ {_describe(item, now)}")
    else:
        lines.append("    nothing — all clear")
    lines.append("")

    in_progress = data.get("in_progress") or []
    lines.append(f"  ◆ In progress ({len(in_progress)})")
    if in_progress:
        for item in in_progress:
            lines.append(f"    ◐ {_describe(item, now)}")
    else:
        lines.append("    nothing running in the background")
    lines.append("")

    finished = data.get("finished_undelivered") or []
    if finished:
        lines.append(f"  ◆ Finished, result not yet delivered ({len(finished)})")
        for item in finished:
            lines.append(f"    ✔ {_describe(item, now)}")
        lines.append("")

    cron_jobs = data.get("cron") or []
    active = [j for j in cron_jobs if j.get("enabled") and not j.get("paused")]
    if active:
        upcoming = sorted(
            (j for j in active if j.get("next_run_at")),
            key=lambda j: str(j.get("next_run_at")),
        )[:5]
        lines.append(f"  ◆ Scheduled ({len(active)} active job(s))")
        for j in upcoming:
            lines.append(f"    ⏱ {j['name']}  next: {j['next_run_at']}")
        lines.append("")

    sessions = data.get("sessions") or []
    if sessions:
        lines.append(f"  ◆ Open surfaces ({len(sessions)})")
        for s in sessions:
            lines.append(f"    ◎ {_describe(s, now)}")
        lines.append("")

    return "\n".join(lines)


def cmd_inbox(args) -> int:
    """Entry point for ``hermes inbox``."""
    data = build_inbox()
    if getattr(args, "json", False):
        print(json.dumps(data, indent=2, default=str))
        return 0
    print(render_inbox(data))
    return 0
