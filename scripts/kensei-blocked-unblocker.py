#!/usr/bin/env python3
"""KENSEI blocked-task unblocker: data collector.

Runs as a no_agent=True script that collects all blocked tasks across all
boards and outputs structured JSON for the LLM cron to consume.

Output: JSON object with scanned_at, scanned_at_iso, total_blocked, and the
routing lists routine_unblock[], escalate[], human_owned_skipped[] (each task
carries board, id, title, assignee, age_seconds, age_hours, priority,
consecutive_failures, max_retries, body_snippet, status_reason).

Silent (``{"wakeAgent": false}``, exit 0) when nothing is actionable.

Import-safe: importing this module performs no scanning and no I/O; all
executable work lives in :func:`main` behind ``if __name__ == "__main__"``.
"""

import json
import os
import sqlite3
import sys
import time
from pathlib import Path

HERMES = Path(os.environ.get("HERMES_HOME") or os.path.expanduser("~/.hermes"))
ESCALATION_AGE = int(os.environ.get("KENSEI_UNBLOCKER_AGE", "3600"))
MAX_TASKS = int(os.environ.get("KENSEI_UNBLOCKER_MAX", "50"))
# Only tasks that have failed / re-blocked this many times warrant a live
# escalation; everything else surfaces in the daily briefing instead.
STUCK_FAILURES = int(os.environ.get("KENSEI_UNBLOCKER_STUCK_FAILURES", "3"))
# Never re-ping the same stuck task more often than this.
DEDUP_HOURS = float(os.environ.get("KENSEI_UNBLOCKER_DEDUP_HOURS", "12"))
STATE_TTL_DAYS = 7
STATE_FILE = HERMES / "state" / "blocked-unblocker-dedup.json"

# P13 isolation: when --dry-run is passed, the dedup state file is never
# written. Read paths (state load, board scans, classification) run
# unchanged so the structured output is still computed and emitted.
_DRY_RUN = False


def discover_boards() -> dict[str, Path]:
    """Dynamically discover all kanban DBs with board identity."""
    dbs: dict[str, Path] = {}
    root = HERMES / "kanban.db"
    if root.exists():
        dbs["default"] = root
    boards_dir = HERMES / "kanban" / "boards"
    if boards_dir.exists():
        for board_dir in sorted(boards_dir.iterdir()):
            db = board_dir / "kanban.db"
            if db.exists():
                dbs[board_dir.name] = db
    return dbs


def _effective_updated(row):
    """The timestamp age is computed from: updated_at, else created_at."""
    return row["updated_at"] if row["updated_at"] is not None else row["created_at"]


def _classify_row(row, *, now: int, max_tasks: int, blocked_tasks_len: int):
    """Classify one blocked-task row into a collection action.

    Returns ``(action, payload)`` where action is one of:

    - ``"stop"`` — MAX_TASKS cap reached; caller stops collecting.
    - ``"skip_too_young"`` — row is younger than ESCALATION_AGE (or has
      no usable timestamp) and is dropped.
    - ``"skip_human_owned"`` — escalation_target is set: decision-needed /
      awaiting a person. The unblocker must not auto-process these
      (t_3bfbe7f4); they are surfaced for visibility only. payload holds
      the routing fields ``escalation_target`` and ``block_kind``.
    - ``"include"`` — auto-processable blocked task.
    """
    if blocked_tasks_len >= max_tasks:
        return "stop", None
    updated = _effective_updated(row)
    if updated is None:
        return "skip_too_young", None
    if now - updated < ESCALATION_AGE:
        return "skip_too_young", None
    escalation_target = row["escalation_target"]
    if escalation_target:
        return "skip_human_owned", {
            "escalation_target": escalation_target,
            "block_kind": row["block_kind"] or "",
        }
    return "include", None


def _collect_blocked_tasks(boards: dict[str, Path], now: int):
    """Scan all boards for actionable blocked tasks.

    Returns ``(blocked_tasks, errors, boards_scanned, boards_skipped)``.
    """
    blocked_tasks: list[dict] = []
    errors: list[dict] = []
    boards_scanned = 0
    boards_skipped = 0

    for slug, db_path in sorted(boards.items()):
        boards_scanned += 1
        if not db_path.exists():
            boards_skipped += 1
            continue
        try:
            conn = sqlite3.connect(
                f"file:{db_path}?mode=ro&timeout=5000", uri=True, timeout=5
            )
            conn.row_factory = sqlite3.Row
            cur = conn.cursor()
            cur.execute("""
                SELECT id, title, assignee, priority, status_reason,
                       created_at, updated_at,
                       consecutive_failures, max_retries,
                       escalation_target, block_kind,
                       substr(body, 1, 500) as body_snippet
                FROM tasks
                WHERE status = 'blocked'
                ORDER BY priority DESC, updated_at ASC
            """)
            rows = cur.fetchall()
            conn.close()
        except Exception as e:
            errors.append({"board": slug, "error": str(e)[:200]})
            boards_skipped += 1
            continue

        for row in rows:
            action, routing = _classify_row(
                row,
                now=now,
                max_tasks=MAX_TASKS,
                blocked_tasks_len=len(blocked_tasks),
            )
            if action == "stop":
                break
            if action == "skip_too_young":
                continue
            age = now - _effective_updated(row)
            task = {
                "board": slug,
                "id": row["id"],
                "title": row["title"],
                "assignee": row["assignee"] or "unassigned",
                "priority": row["priority"] or 0,
                "age_seconds": age,
                "age_hours": round(age / 3600.0, 1),
                "consecutive_failures": row["consecutive_failures"] or 0,
                "max_retries": row["max_retries"],
                "status_reason": row["status_reason"] or "",
                "body_snippet": row["body_snippet"] or "",
            }
            if action == "skip_human_owned":
                task.update(routing)
                task["human_owned"] = True
            else:
                task["human_owned"] = False
            blocked_tasks.append(task)

    return blocked_tasks, errors, boards_scanned, boards_skipped


def _state_ts(v) -> int:
    """Coerce a stored state value to an epoch int, tolerating a hand-edited file."""
    try:
        return int(v)
    except (TypeError, ValueError):
        return 0


def _load_state() -> dict:
    try:
        return json.loads(STATE_FILE.read_text())
    except Exception:
        return {}


def _save_state(state: dict) -> None:
    if _DRY_RUN:
        return
    # Atomic write so two overlapping runs cannot leave a truncated file.
    try:
        STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
        tmp = STATE_FILE.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(state))
        os.replace(tmp, STATE_FILE)
    except Exception as e:
        print(json.dumps({"state_save_error": str(e)[:200]}), file=sys.stderr)


def main() -> int:
    global _DRY_RUN
    if "--dry-run" in sys.argv:
        _DRY_RUN = True
    now = int(time.time())
    blocked_tasks, errors, boards_scanned, boards_skipped = _collect_blocked_tasks(
        discover_boards(), now
    )

    # Log diagnostic info to stderr for cron capture
    diag = {
        "boards_scanned": boards_scanned,
        "boards_skipped": boards_skipped,
        "blocked_found": len(blocked_tasks),
        "max_tasks": MAX_TASKS,
        "truncated": len(blocked_tasks) >= MAX_TASKS,
        "errors": errors,
    }
    print(json.dumps(diag), file=sys.stderr)

    # Split blocked tasks into intents:
    #   routine: failed fewer than STUCK_FAILURES times, auto-recoverable, unblock SILENTLY.
    #   stuck:   failed STUCK_FAILURES+ times, auto-repair has not held, escalate.
    #   human_owned: escalation_target set — decision-needed / awaiting a person.
    #                NEVER auto-unblock or auto-escalate these (t_3bfbe7f4). They are
    #                reported for visibility only; the triage flow handles them separately.
    human_owned = [t for t in blocked_tasks if t.get("human_owned")]
    auto_tasks = [t for t in blocked_tasks if not t.get("human_owned")]
    stuck = [t for t in auto_tasks if t["consecutive_failures"] >= STUCK_FAILURES]
    routine = [t for t in auto_tasks if t["consecutive_failures"] < STUCK_FAILURES]

    # 12h dedup on escalations only: never re-ping the same stuck task within
    # DEDUP_HOURS. Routine unblocks are silent so they need no dedup. JSON object
    # keys are always strings, so key off str(id) on both read and write.
    state = _load_state()
    cutoff = now - int(DEDUP_HOURS * 3600)
    escalate = [t for t in stuck if _state_ts(state.get(str(t["id"]), 0)) < cutoff]

    if not routine and not escalate:
        # Nothing to unblock and nothing new to escalate: do not wake the agent.
        print(json.dumps({"wakeAgent": False}))
        return 0

    # Record escalation times for the deduped tasks and prune stale state entries.
    if escalate:
        ttl_cutoff = now - STATE_TTL_DAYS * 86400
        state = {k: v for k, v in state.items() if _state_ts(v) >= ttl_cutoff}
        for t in escalate:
            state[str(t["id"])] = now
        _save_state(state)

    print(json.dumps({
        "wakeAgent": True,
        "scanned_at": now,
        "scanned_at_iso": time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime(now)),
        "total_blocked": len(blocked_tasks),
        "routine_unblock": routine,
        "escalate": escalate,
        "human_owned_skipped": human_owned,
    }, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
