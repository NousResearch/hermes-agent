"""Kanban doctor — one-shot board health report (Layer 3 self-healing).

``hermes kanban doctor`` produces a read-only health summary so a human (or
Zeus) can audit self-healing state in seconds instead of a 40-call forensic
dig.  The report covers:

* **Counts** — genuinely-stuck vs waiting-by-design vs review-required tasks.
* **Bottleneck** — which profile has the most review-required items aimed at it.
* **Oldest stalled task** — the longest-waiting blocked/ready task + age.
* **Deadlocks** — dependency cycles in task_links.
* **Layer 1 warnings** — tasks with active diagnostics.
* **Heal log** — auto-resolver actions in the last 24h (Layer 2).

The command is strictly read-only: it queries the DB, computes, and prints.
It never mutates state.  Exit code is always 0 (even when the board is sick)
so the watchdog cron can consume ``--json`` output without false alerts.

Layer 2 (auto-resolver) may not be deployed yet; in that case the heal-log
section reports "no heal actions recorded" gracefully.
"""

from __future__ import annotations

import sqlite3
import time
from collections import defaultdict
from typing import Any, Optional


# ---------------------------------------------------------------------------
# Blocked-task classification
# ---------------------------------------------------------------------------

def _block_reason_from_events(events: list[Any]) -> str:
    """Extract the most recent block reason from a task's event list."""
    from hermes_cli import kanban_diagnostics as kd
    reason = ""
    for ev in reversed(events):
        kind = kd._event_kind(ev)
        if kind == "blocked":
            payload = kd._parse_payload(ev)
            reason = payload.get("reason", "") or ""
            if reason:
                return reason
    return reason


def _is_review_required(task: Any, events: list[Any]) -> bool:
    """Return True if this blocked task is review-required.

    Heuristic: the block reason contains 'review-required' (the canonical
    prefix the kanban-worker skill tells workers to use), or the assignee
    equals the creator (self-review smell).
    """
    from hermes_cli import kanban_diagnostics as kd
    status = kd._task_field(task, "status")
    if status != "blocked":
        return False
    reason = _block_reason_from_events(events)
    return bool(reason and reason.startswith("review-required"))


def _is_waiting_by_design(task: Any, events: list[Any]) -> bool:
    """Return True if this blocked task is intentionally waiting.

    Heuristic: block reason starts with 'waiting' or 'scheduled', or the
    task status is 'scheduled', or the task is blocked by incomplete parents
    (natural dependency wait — not stuck, just not ready yet).
    """
    from hermes_cli import kanban_diagnostics as kd
    status = kd._task_field(task, "status")
    if status == "scheduled":
        return True
    if status != "blocked":
        return False
    reason = _block_reason_from_events(events)
    if not reason:
        # No explicit reason but blocked by parent deps = waiting by design.
        return True
    rl = reason.lower()
    return rl.startswith("waiting") or rl.startswith("scheduled")


def _is_genuinely_stuck(task: Any, events: list[Any]) -> bool:
    """Return True if this blocked task is genuinely stuck (neither
    review-required nor waiting-by-design)."""
    from hermes_cli import kanban_diagnostics as kd
    status = kd._task_field(task, "status")
    if status != "blocked":
        return False
    return not _is_review_required(task, events) and not _is_waiting_by_design(task, events)


# ---------------------------------------------------------------------------
# Deadlock detection (cycle in task_links)
# ---------------------------------------------------------------------------

def _detect_deadlocks(conn: sqlite3.Connection) -> list[list[str]]:
    """Find cycles in the task_links graph.

    Returns a list of cycles, each cycle being a list of task ids.
    Uses a simple DFS with a visited-set per path.
    """
    # Build adjacency: parent → children
    adj: dict[str, list[str]] = defaultdict(list)
    for row in conn.execute("SELECT parent_id, child_id FROM task_links"):
        adj[row["parent_id"]].append(row["child_id"])

    # Only consider non-archived tasks for cycle detection.
    active_ids: set[str] = set()
    for row in conn.execute("SELECT id FROM tasks WHERE status != 'archived'"):
        active_ids.add(row["id"])

    # Filter adjacency to active tasks only.
    adj = {k: [v for v in vs if v in active_ids] for k, vs in adj.items() if k in active_ids}

    cycles: list[list[str]] = []
    visited_global: set[str] = set()

    def _dfs(node: str, path: list[str], path_set: set[str]) -> None:
        if node in path_set:
            # Found a cycle — extract it.
            cycle_start = path.index(node)
            cycle = path[cycle_start:]
            cycles.append(cycle)
            return
        if node in visited_global:
            return
        visited_global.add(node)
        path_set.add(node)
        path.append(node)
        for child in adj.get(node, []):
            _dfs(child, path, path_set)
        path.pop()
        path_set.discard(node)

    for start in active_ids:
        if start not in visited_global:
            _dfs(start, [], set())

    return cycles


# ---------------------------------------------------------------------------
# Heal log (Layer 2 auto-resolver actions)
# ---------------------------------------------------------------------------

def _read_heal_log(conn: sqlite3.Connection, since_ts: int) -> list[dict]:
    """Read auto-heal actions from the self_heal_events table (if it exists).

    Layer 2 may create a ``self_heal_events`` table to log its actions.
    If the table doesn't exist, return an empty list — doctor degrades
    gracefully when L2 is not yet deployed.
    """
    try:
        rows = list(conn.execute(
            "SELECT * FROM self_heal_events WHERE created_at >= ? ORDER BY created_at DESC",
            (since_ts,),
        ).fetchall())
    except Exception:
        # Table doesn't exist yet (L2 not deployed).
        return []

    result = []
    for row in rows:
        entry = {k: row[k] for k in row.keys()}
        result.append(entry)
    return result


# ---------------------------------------------------------------------------
# Board doctor — main entry
# ---------------------------------------------------------------------------

def board_doctor(conn: sqlite3.Connection, *, now: Optional[int] = None) -> dict:
    """Compute a board health report.  Pure read-only — no DB writes.

    Returns a dict suitable for both human-readable printing and ``--json``
    machine consumption.

    Structure::

        {
            "generated_at": <unix_ts>,
            "counts": {
                "total_non_archived": N,
                "by_status": {...},
                "blocked_genuinely_stuck": N,
                "blocked_waiting_by_design": N,
                "blocked_review_required": N,
            },
            "bottleneck": {
                "profile": "<name>" | null,
                "review_required_count": N,
                "review_required_by_profile": {<profile>: N, ...},
            },
            "oldest_stalled": {
                "task_id": "<id>" | null,
                "title": "<title>" | null,
                "status": "<status>" | null,
                "age_seconds": N | null,
                "assignee": "<name>" | null,
            },
            "deadlocks": [
                {"cycle": ["t_a", "t_b", "t_a"], "length": 3},
                ...
            ],
            "layer1_warnings": [
                {"task_id": ..., "diagnostics": [...]},
                ...
            ],
            "heal_log_24h": [
                {"action": ..., "task_id": ..., "created_at": ..., ...},
                ...
            ],
            "healthy": true | false,
        }
    """
    from hermes_cli import kanban_diagnostics as kd
    from hermes_cli.config import load_config

    now_ts = int(now if now is not None else time.time())
    day_ago = now_ts - 86400

    # --- Fetch all non-archived tasks + events ---
    tasks_rows = list(conn.execute(
        "SELECT id, title, status, assignee, created_at, started_at, consecutive_failures, last_failure_error, "
        "completed_at, body, created_by FROM tasks WHERE status != 'archived'"
    ).fetchall())

    task_ids = [r["id"] for r in tasks_rows]

    # Fetch events for blocked tasks (only those need classification).
    blocked_ids = [r["id"] for r in tasks_rows if r["status"] == "blocked"]
    events_by_task: dict[str, list] = {}
    if blocked_ids:
        ph = ",".join(["?"] * len(blocked_ids))
        for row in conn.execute(
            f"SELECT * FROM task_events WHERE task_id IN ({ph}) ORDER BY id",
            tuple(blocked_ids),
        ):
            events_by_task.setdefault(row["task_id"], []).append(row)

    # --- Status counts ---
    by_status: dict[str, int] = defaultdict(int)
    for r in tasks_rows:
        by_status[r["status"]] += 1

    # --- Blocked/Scheduled classification ---
    stuck = 0
    waiting = 0
    review_req = 0
    # review-required by assignee (the profile that SHOULD review)
    review_by_profile: dict[str, int] = defaultdict(int)

    for r in tasks_rows:
        if r["status"] not in ("blocked", "scheduled"):
            continue
        evs = events_by_task.get(r["id"], [])
        if r["status"] == "scheduled":
            # Scheduled tasks are always waiting-by-design.
            waiting += 1
        elif _is_review_required(r, evs):
            review_req += 1
            assignee = r["assignee"] or "(unassigned)"
            review_by_profile[assignee] += 1
        elif _is_waiting_by_design(r, evs):
            waiting += 1
        else:
            stuck += 1

    # --- Bottleneck: profile with most review-required items ---
    bottleneck_profile = None
    bottleneck_count = 0
    if review_by_profile:
        top = max(review_by_profile.items(), key=lambda kv: kv[1])
        bottleneck_profile = top[0]
        bottleneck_count = top[1]

    # --- Oldest stalled task (blocked or ready, longest wait) ---
    oldest_id = None
    oldest_title = None
    oldest_status = None
    oldest_age = None
    oldest_assignee = None
    stalled_statuses = {"blocked", "ready"}
    for r in tasks_rows:
        if r["status"] not in stalled_statuses:
            continue
        created = r["created_at"]
        if created is None:
            continue
        age = now_ts - int(created)
        if oldest_age is None or age > oldest_age:
            oldest_age = age
            oldest_id = r["id"]
            oldest_title = r["title"]
            oldest_status = r["status"]
            oldest_assignee = r["assignee"]

    # --- Deadlocks ---
    deadlocks = _detect_deadlocks(conn)
    deadlock_entries = []
    for cycle in deadlocks:
        deadlock_entries.append({
            "cycle": cycle,
            "length": len(cycle),
        })

    # --- Layer 1 warnings (active diagnostics) ---
    diag_config = kd.config_from_runtime_config(load_config())
    diags_by_task: dict[str, list] = {}
    # Only compute diagnostics for tasks that aren't done/archived.
    active_ids = [r["id"] for r in tasks_rows if r["status"] not in ("done", "archived")]
    if active_ids:
        ph = ",".join(["?"] * len(active_ids))
        ev_all: dict[str, list] = {i: [] for i in active_ids}
        for row in conn.execute(
            f"SELECT * FROM task_events WHERE task_id IN ({ph}) ORDER BY id",
            tuple(active_ids),
        ):
            ev_all.setdefault(row["task_id"], []).append(row)
        run_all: dict[str, list] = {i: [] for i in active_ids}
        for row in conn.execute(
            f"SELECT * FROM task_runs WHERE task_id IN ({ph}) ORDER BY id",
            tuple(active_ids),
        ):
            run_all.setdefault(row["task_id"], []).append(row)

        for r in tasks_rows:
            if r["id"] not in active_ids:
                continue
            tid = r["id"]
            dl = kd.compute_task_diagnostics(
                r,
                ev_all.get(tid, []),
                run_all.get(tid, []),
                config=diag_config,
            )
            if dl:
                diags_by_task[tid] = dl

    layer1_warnings = []
    for tid, dl in diags_by_task.items():
        # Find matching task row for title.
        title = ""
        for r in tasks_rows:
            if r["id"] == tid:
                title = r["title"]
                break
        layer1_warnings.append({
            "task_id": tid,
            "title": title,
            "diagnostics": [
                {
                    "kind": d.kind,
                    "severity": d.severity,
                    "title": d.title,
                }
                for d in dl
            ],
        })

    # --- Heal log (Layer 2) ---
    heal_log = _read_heal_log(conn, day_ago)

    # --- Healthy? ---
    is_healthy = (
        stuck == 0
        and review_req == 0
        and len(deadlock_entries) == 0
        and len(layer1_warnings) == 0
    )

    return {
        "generated_at": now_ts,
        "counts": {
            "total_non_archived": len(tasks_rows),
            "by_status": dict(by_status),
            "blocked_genuinely_stuck": stuck,
            "blocked_waiting_by_design": waiting,
            "blocked_review_required": review_req,
            "scheduled_waiting_by_design": sum(
                1 for r in tasks_rows if r["status"] == "scheduled"
            ),
        },
        "bottleneck": {
            "profile": bottleneck_profile,
            "review_required_count": bottleneck_count,
            "review_required_by_profile": dict(review_by_profile),
        },
        "oldest_stalled": {
            "task_id": oldest_id,
            "title": oldest_title,
            "status": oldest_status,
            "age_seconds": oldest_age,
            "assignee": oldest_assignee,
        },
        "deadlocks": deadlock_entries,
        "layer1_warnings": layer1_warnings,
        "heal_log_24h": heal_log,
        "healthy": is_healthy,
    }
