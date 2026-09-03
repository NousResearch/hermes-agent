"""Read-only kanban reliability auditor — the ``hermes kanban doctor`` backend.

This module never writes to any board. It is the dry-run harness for the
three dispatch-reliability fixes and the standing surface for stalled work:

  1. Unroutable assignees — ready/review tasks assigned to a name that is
     neither a Hermes profile nor a known pull lane. These sit forever and
     used to be misreported as "correctly idle" (the 55h fable failure).
  2. Dispatcher heartbeat — is the single dispatcher healthy, HUNG (flock
     held but heartbeat stale), or absent? Detects the hang that systemd's
     death-only restart cannot.
  3. Stalled tasks — ready tasks untouched past N days, and todos gated
     behind parents that are dead/blocked (a dependency deadlock that never
     resolves). Surface for human triage; NEVER auto-deleted.

Every check returns structured data plus a human-readable line set, so the
CLI can emit either text or JSON, and the gateway telemetry / a monitoring
pass can consume the same logic.
"""

from __future__ import annotations

import os
import sqlite3
import time
from pathlib import Path
from typing import Any, Optional


# ---------------------------------------------------------------------------
# Read-only board access
# ---------------------------------------------------------------------------

def _ro_connect(db_path: Path) -> sqlite3.Connection:
    """Open a STRICTLY read-only connection to a board DB.

    Uses SQLite ``mode=ro`` URI so the doctor can never migrate, create, or
    write a board — even accidentally. Raises ``sqlite3.OperationalError``
    if the file is missing (caller skips that board).
    """
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=5.0)
    conn.row_factory = sqlite3.Row
    return conn


def _iter_board_dbs() -> "list[tuple[str, Path]]":
    """Return ``(slug, db_path)`` for every board on disk. Read-only."""
    from hermes_cli import kanban_db as _kb

    out: list[tuple[str, Path]] = []
    try:
        boards = _kb.list_boards(include_archived=False)
    except Exception:
        boards = []
    seen: set[str] = set()
    if not boards:
        try:
            boards = [_kb.read_board_metadata(_kb.DEFAULT_BOARD)]
        except Exception:
            boards = []
    for b in boards:
        slug = b.get("slug") or _kb.DEFAULT_BOARD
        try:
            path = Path(b.get("db_path")) if b.get("db_path") else _kb.kanban_db_path(slug)
            resolved = str(path.expanduser().resolve())
        except Exception:
            continue
        if resolved in seen:
            continue
        seen.add(resolved)
        out.append((slug, Path(resolved)))
    return out


# ---------------------------------------------------------------------------
# 1. Unroutable / aging-pull-lane assignees
# ---------------------------------------------------------------------------

def audit_assignees(pull_lane_stale_hours: float = 6.0) -> "dict[str, Any]":
    """Find ready/review tasks that no one can pick up, across all boards.

    Two buckets:
      ``unroutable`` — assignee is neither a profile nor a known pull lane.
                       Always a finding (STUCK).
      ``pull_lane_aging`` — assignee is a known pull lane, but the task has
                       gone unclaimed past ``pull_lane_stale_hours``. Not a
                       dispatcher failure, but a human/terminal owes it a
                       pull (this is what the fable cards needed).
    """
    from hermes_cli import kanban_db as _kb

    unroutable: list[dict] = []
    pull_lane_aging: list[dict] = []
    now = int(time.time())
    for slug, db_path in _iter_board_dbs():
        try:
            conn = _ro_connect(db_path)
        except Exception:
            continue
        try:
            rows = conn.execute(
                "SELECT id, assignee, title, created_at, status FROM tasks "
                "WHERE status IN ('ready', 'review') AND assignee IS NOT NULL "
                "    AND assignee != '' AND claim_lock IS NULL "
                "ORDER BY created_at ASC"
            ).fetchall()
        except Exception:
            conn.close()
            continue
        for r in rows:
            cls = _kb.classify_assignee(r["assignee"])
            age_h = (now - int(r["created_at"])) / 3600.0
            entry = {
                "board": slug, "id": r["id"], "assignee": r["assignee"],
                "status": r["status"], "age_hours": round(age_h, 1),
                "title": (r["title"] or "")[:70],
            }
            if cls == "unroutable":
                unroutable.append(entry)
            elif cls == "pull_lane" and age_h >= pull_lane_stale_hours:
                pull_lane_aging.append(entry)
        conn.close()
    return {
        "unroutable": unroutable,
        "pull_lane_aging": pull_lane_aging,
        "ok": not unroutable,
    }


# ---------------------------------------------------------------------------
# 2. Dispatcher heartbeat / hang detection
# ---------------------------------------------------------------------------

def _flock_holder_pid(lock_path: Path) -> Optional[int]:
    """Best-effort: pid currently holding the dispatcher flock, via ``fuser``.

    Returns ``None`` if fuser is unavailable or nothing holds it. Pure
    read-only introspection.
    """
    import subprocess
    try:
        out = subprocess.run(
            ["fuser", str(lock_path)],
            capture_output=True, text=True, timeout=5,
        )
    except Exception:
        return None
    pids = (out.stdout + " " + out.stderr).split()
    for tok in pids:
        if tok.isdigit():
            return int(tok)
    return None


def audit_heartbeat(stale_seconds: float = 0.0) -> "dict[str, Any]":
    """Classify the dispatcher's liveness from its heartbeat + flock.

    States:
      ``healthy``   — heartbeat fresh (< stale threshold).
      ``hung``      — heartbeat stale but the holder pid is still alive:
                      dispatch is STOPPED fleet-wide and systemd won't fix
                      it (death-only). THE case this fix exists for.
      ``dead``      — heartbeat stale and holder pid gone (flock should be
                      free; a standby / restart recovers automatically).
      ``no_heartbeat`` — no heartbeat file. Either no dispatcher is running,
                      OR the running dispatcher predates this fix (old code
                      holds the flock but writes no heartbeat). The flock
                      holder pid (if any) disambiguates.
    ``stale_seconds`` defaults to 5×interval (read from config; min 60s).
    """
    from hermes_cli import kanban_db as _kb

    lock_path = _kb.kanban_home() / "kanban" / ".dispatcher.lock"
    holder_pid = _flock_holder_pid(lock_path)

    if stale_seconds <= 0:
        interval = 60.0
        try:
            from hermes_cli.config import load_config
            cfg = load_config()
            kcfg = cfg.get("kanban", {}) if isinstance(cfg, dict) else {}
            interval = float(kcfg.get("dispatch_interval_seconds", 60) or 60)
        except Exception:
            interval = 60.0
        stale_seconds = max(interval * 5.0, 60.0)

    hb = _kb.read_dispatcher_heartbeat()
    if hb is None:
        state = "no_heartbeat"
        detail = (
            "no heartbeat file. If a dispatcher IS running it predates this "
            "fix (arms on next gateway restart); if not, nothing is dispatching."
        )
        if holder_pid:
            detail = (
                f"flock held by pid {holder_pid} but it writes NO heartbeat — "
                "old-code dispatcher; arms after this branch lands + gateway cycle."
            )
        return {
            "state": state, "detail": detail, "flock_holder_pid": holder_pid,
            "heartbeat": None, "stale_seconds_threshold": stale_seconds,
            "ok": True,  # pre-arm state is expected, not a failure
        }

    age = hb.get("age_seconds")
    alive = hb.get("pid_alive")
    if age is not None and age < stale_seconds:
        state, ok, detail = "healthy", True, (
            f"heartbeat {age}s old (< {int(stale_seconds)}s threshold), "
            f"pid {hb.get('pid')} on {hb.get('host')}."
        )
    elif alive is False:
        state, ok, detail = "dead", True, (
            f"heartbeat {age}s stale and holder pid {hb.get('pid')} is gone — "
            "flock should be free; a standby/restart recovers automatically."
        )
    else:
        state, ok, detail = "hung", False, (
            f"heartbeat {age}s stale (> {int(stale_seconds)}s) but holder pid "
            f"{hb.get('pid')} is STILL ALIVE — dispatcher HUNG, dispatch stopped "
            "fleet-wide. systemd will not fix this (death-only). Kill the pid."
        )
    return {
        "state": state, "detail": detail, "flock_holder_pid": holder_pid,
        "heartbeat": hb, "stale_seconds_threshold": stale_seconds, "ok": ok,
    }


# ---------------------------------------------------------------------------
# 3. Stalled-task surface (no mutation, ever)
# ---------------------------------------------------------------------------

def audit_stale_tasks(
    ready_days: float = 2.0, todo_days: float = 7.0,
) -> "dict[str, Any]":
    """Surface tasks that have silently stalled. Read-only; never deletes.

      ``stale_ready`` — ready+assigned+unclaimed tasks older than
                        ``ready_days`` (a real profile that never spawned,
                        or an aging queue).
      ``deadlocked_todos`` — todos gated behind ≥1 parent that is ``blocked``
                        or itself a stalled ``todo`` — a dependency deadlock
                        that ``recompute_ready`` can never resolve (it only
                        promotes when ALL parents are done/archived).
    Each deadlocked todo carries the offending parents + their statuses so a
    human can decide: unblock the parent, cut the link, or archive the todo.
    """
    from hermes_cli import kanban_db as _kb

    now = int(time.time())
    stale_ready: list[dict] = []
    deadlocked_todos: list[dict] = []
    for slug, db_path in _iter_board_dbs():
        try:
            conn = _ro_connect(db_path)
        except Exception:
            continue
        try:
            # stale ready (routable assignees only — unroutable is its own
            # finding in audit_assignees; don't double-count)
            for r in conn.execute(
                "SELECT id, assignee, title, created_at FROM tasks "
                "WHERE status='ready' AND assignee IS NOT NULL AND assignee!='' "
                "    AND claim_lock IS NULL ORDER BY created_at ASC"
            ).fetchall():
                age_d = (now - int(r["created_at"])) / 86400.0
                if age_d < ready_days:
                    continue
                # Only PROFILE assignees belong here: a real spawnable task
                # that aged without spawning (profile health / queue depth).
                # Unroutable → audit_assignees; pull lanes → pull_lane_aging.
                # Keeps the three buckets disjoint (no double-counting).
                if _kb.classify_assignee(r["assignee"]) != "profile":
                    continue
                stale_ready.append({
                    "board": slug, "id": r["id"], "assignee": r["assignee"],
                    "age_days": round(age_d, 1), "title": (r["title"] or "")[:70],
                })
            # dependency-deadlocked todos
            todo_rows = conn.execute(
                "SELECT id, assignee, title, created_at FROM tasks WHERE status='todo'"
            ).fetchall()
            for r in todo_rows:
                age_d = (now - int(r["created_at"])) / 86400.0
                if age_d < todo_days:
                    continue
                parents = conn.execute(
                    "SELECT p.id AS pid, p.status AS pstatus FROM task_links l "
                    "JOIN tasks p ON p.id = l.parent_id WHERE l.child_id = ?",
                    (r["id"],),
                ).fetchall()
                blockers = [
                    {"parent": p["pid"], "parent_status": p["pstatus"]}
                    for p in parents
                    if p["pstatus"] not in ("done", "archived")
                ]
                # A todo with a live blocker whose status is blocked/todo is a
                # deadlock: it will never satisfy "all parents done".
                dead = [
                    b for b in blockers
                    if b["parent_status"] in ("blocked", "todo")
                ]
                if dead:
                    deadlocked_todos.append({
                        "board": slug, "id": r["id"], "assignee": r["assignee"],
                        "age_days": round(age_d, 1), "title": (r["title"] or "")[:70],
                        "blocking_parents": blockers,
                    })
        except Exception:
            pass
        finally:
            conn.close()
    return {
        "stale_ready": stale_ready,
        "deadlocked_todos": deadlocked_todos,
        "ok": not stale_ready and not deadlocked_todos,
    }


# ---------------------------------------------------------------------------
# Aggregate
# ---------------------------------------------------------------------------

def run_doctor(
    *,
    ready_days: float = 2.0,
    todo_days: float = 7.0,
    pull_lane_stale_hours: float = 6.0,
    stale_seconds: float = 0.0,
) -> "dict[str, Any]":
    """Run all three audits and return one structured report (read-only)."""
    assignees = audit_assignees(pull_lane_stale_hours=pull_lane_stale_hours)
    heartbeat = audit_heartbeat(stale_seconds=stale_seconds)
    stale = audit_stale_tasks(ready_days=ready_days, todo_days=todo_days)
    ok = assignees["ok"] and heartbeat["ok"] and stale["ok"]
    return {
        "ok": ok,
        "generated_at": int(time.time()),
        "assignees": assignees,
        "heartbeat": heartbeat,
        "stale_tasks": stale,
    }


def render_text(report: "dict[str, Any]") -> str:
    """Render a doctor report as human-readable text."""
    L: list[str] = []
    L.append("== kanban doctor (read-only) ==")

    hb = report["heartbeat"]
    icon = {"healthy": "OK", "hung": "⚠ HUNG", "dead": "·", "no_heartbeat": "·"}.get(hb["state"], "?")
    L.append(f"[dispatcher] {icon} state={hb['state']}: {hb['detail']}")

    a = report["assignees"]
    if a["unroutable"]:
        L.append(f"[assignees] ⚠ {len(a['unroutable'])} UNROUTABLE ready/review task(s) — will sit forever:")
        for e in a["unroutable"]:
            L.append(f"    {e['id']}  assignee={e['assignee']}  age={e['age_hours']}h  {e['title']}")
    else:
        L.append("[assignees] OK — no unroutable ready/review tasks")
    if a["pull_lane_aging"]:
        L.append(f"[pull-lanes] {len(a['pull_lane_aging'])} pull-lane task(s) unclaimed past threshold (a terminal owes a pull):")
        for e in a["pull_lane_aging"]:
            L.append(f"    {e['id']}  lane={e['assignee']}  age={e['age_hours']}h  {e['title']}")

    s = report["stale_tasks"]
    if s["stale_ready"]:
        L.append(f"[stale-ready] ⚠ {len(s['stale_ready'])} ready task(s) unspawned past threshold:")
        for e in s["stale_ready"]:
            L.append(f"    {e['id']}  assignee={e['assignee']}  age={e['age_days']}d  {e['title']}")
    else:
        L.append("[stale-ready] OK — no stale ready tasks")
    if s["deadlocked_todos"]:
        L.append(f"[deadlocked-todos] ⚠ {len(s['deadlocked_todos'])} todo(s) gated behind dead/blocked parents (surface for triage — NOT auto-deleted):")
        for e in s["deadlocked_todos"]:
            ps = ", ".join(f"{b['parent']}={b['parent_status']}" for b in e["blocking_parents"])
            L.append(f"    {e['id']}  assignee={e['assignee']}  age={e['age_days']}d  blocked_by[{ps}]  {e['title']}")
    else:
        L.append("[deadlocked-todos] OK — no dependency-deadlocked todos")

    L.append("== " + ("HEALTHY ==" if report["ok"] else "⚠ FINDINGS ABOVE (surface only; no task was modified) =="))
    return "\n".join(L)
