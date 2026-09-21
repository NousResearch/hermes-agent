#!/usr/bin/env python3
"""Kanban reconciliation — detect and auto-fix terminal-status drift.

Scans all board DBs (root + per-board + profile overlays) for:
1. Tasks marked done/completed with zero successful runs → reopen to triage
2. Tasks marked done/completed with ALL runs crashed/blocked → reopen to triage
3. Profile overlay status inconsistency → flag (no auto-fix, architectural)

Designed as a no_agent=True cron. Silent when clean. Alert when drift found
and fixed.

Schema: kanban_db.py — tasks.status, task_runs.outcome, task_events.kind
"""
from __future__ import annotations
import glob
import json
import os
import sqlite3
import sys
import time
from pathlib import Path

# Cross-process write lock — prevents WAL checkpoint races
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
try:
    from kanban_write_lock import write_lock
except ImportError:
    write_lock = None

# P13 isolation: HERMES derives from HERMES_HOME (env-overridable) so
# local disposable runs never touch /home/kensei/.hermes.
HERMES = Path(os.environ.get("HERMES_HOME", os.path.expanduser("~/.hermes")))
DRIFT_STATE = HERMES / "governance" / "reconcile-drift-state.json"

# P13 isolation: when --dry-run is passed, every write path (reopen_task
# SQLite UPDATE + event INSERT, save_state drift file) is suppressed.
# Read paths (board_dbs, scan_board, detect_profile_drift, load_state)
# run unchanged so drift is still detected and reported.
_DRY_RUN = False
SUCCESS_OUTCOMES = {"completed", "done"}
TERMINAL_STATUSES = {"done", "completed", "archived"}
# Review-flow tasks may legitimately close without a successful run
# (the review itself is the deliverable, recorded as a comment).
REVIEW_PREFIXES = ("review:", "[review]", "clean up:", "investigate:", "audit:")

# Manual closure signal: a non-empty status_reason that wasn't written by
# this script means a human deliberately closed the task with an explanation.
# These are NOT drift — they're housekeeping. Only skip if the reason is
# human-authored (not an auto-reopen stamp from a prior reconcile run).
RECONCILE_REASON_PREFIX = "Auto-reopened by kanban-reconcile"

# Circuit breaker: if a single run would auto-reopen more than this many
# tasks, something is systemically wrong (bad heuristic, schema change,
# bulk import) and mass-reopening would create a churn storm. Above the
# cap we reopen NOTHING and alert for human review instead. Override via
# the HERMES_RECONCILE_MAX_REOPEN env var.
try:
    MAX_REOPEN_PER_RUN = int(os.environ.get("HERMES_RECONCILE_MAX_REOPEN", "10"))
except (TypeError, ValueError):
    MAX_REOPEN_PER_RUN = 10


def now_ts() -> int:
    return int(time.time())


def board_dbs() -> dict[str, Path]:
    """Discover all kanban DBs with board identity."""
    dbs: dict[str, Path] = {}
    # Root DB
    root = HERMES / "kanban.db"
    if root.exists():
        dbs["default"] = root
    # Per-board DBs
    boards_dir = HERMES / "kanban" / "boards"
    if boards_dir.exists():
        for board_dir in sorted(boards_dir.iterdir()):
            db = board_dir / "kanban.db"
            if db.exists():
                dbs[board_dir.name] = db
    # Profile overlay DBs
    profiles_dir = HERMES / "profiles"
    if profiles_dir.exists():
        for profile_dir in sorted(profiles_dir.iterdir()):
            if not profile_dir.is_dir():
                continue
            for kanban_db in profile_dir.glob("kanban/boards/*/kanban.db"):
                board_name = kanban_db.parent.name
                key = f"profile:{profile_dir.name}/{board_name}"
                dbs[key] = kanban_db
            # Profile-level kanban.db (flat)
            flat_db = profile_dir / "kanban.db"
            if flat_db.exists():
                key = f"profile:{profile_dir.name}"
                dbs[key] = flat_db
    return dbs


def get_runs(conn: sqlite3.Connection, task_id: str) -> list[dict]:
    rows = conn.execute(
        "SELECT id, outcome, status, error, started_at, ended_at "
        "FROM task_runs WHERE task_id = ? ORDER BY started_at",
        (task_id,),
    ).fetchall()
    return [dict(r) for r in rows]


def get_events(conn: sqlite3.Connection, task_id: str) -> list[dict]:
    rows = conn.execute(
        "SELECT kind, payload, created_at FROM task_events "
        "WHERE task_id = ? ORDER BY created_at",
        (task_id,),
    ).fetchall()
    return [dict(r) for r in rows]


def successful_runs(runs: list[dict]) -> list[dict]:
    return [r for r in runs if (r.get("outcome") or r.get("status")) in SUCCESS_OUTCOMES]


def all_runs_failed(runs: list[dict]) -> bool:
    if not runs:
        return False
    return all(
        (r.get("outcome") or r.get("status")) not in SUCCESS_OUTCOMES
        for r in runs
    )


def is_review_task(title: str) -> bool:
    t = (title or "").lower().strip()
    return any(t.startswith(p) for p in REVIEW_PREFIXES)


def is_manual_closure(status_reason: str | None) -> bool:
    """A non-empty status_reason not from this script = human housekeeping."""
    if not status_reason or not status_reason.strip():
        return False
    return not status_reason.startswith(RECONCILE_REASON_PREFIX)


def is_deliberate_close(conn: sqlite3.Connection, task: dict) -> bool:
    """A terminal task is NOT drift when its event stream shows a legitimate
    closure signal. The system closes tasks through several paths that never
    set status_reason, so classification must read the event history:
      - 'completed' event carrying denji_review_signal=true (review-flow close)
      - a human-authored comment (author='default' = Sahil's session)
      - a 'decomposed' event (parent split into children IS the completion)
      - status_reason from a human (existing check)
    task_runs success is only one of several legitimate close paths.
    """
    if is_manual_closure(task.get("status_reason")):
        return True
    for ev in get_events(conn, task["id"]):
        kind = ev["kind"]
        if kind == "completed":
            try:
                payload = json.loads(ev["payload"] or "{}")
            except Exception:
                payload = {}
            if payload.get("denji_review_signal") is True:
                return True
        elif kind == "commented":
            try:
                payload = json.loads(ev["payload"] or "{}")
            except Exception:
                payload = {}
            if payload.get("author") == "default":
                return True
        elif kind == "decomposed":
            return True
    return False


def reopen_task(conn: sqlite3.Connection, task_id: str, reason: str) -> bool:
    """Reopen a drifted task: set status=triage, log event."""
    if _DRY_RUN:
        return False  # dry-run: suppress the UPDATE + event INSERT
    now = now_ts()
    cur = conn.execute(
        "UPDATE tasks SET status = 'triage', status_reason = ?, "
        "claim_lock = NULL, claim_expires = NULL, worker_pid = NULL, "
        "current_run_id = NULL "
        "WHERE id = ? AND status IN ('done', 'completed')",
        (reason[:500], task_id),
    )
    if cur.rowcount == 0:
        return False
    # Log the reconciliation event
    conn.execute(
        "INSERT INTO task_events (task_id, kind, payload, created_at) "
        "VALUES (?, 'reopened', ?, ?)",
        (task_id, json.dumps({"reason": reason, "source": "kanban-reconcile"}), now),
    )
    conn.commit()
    return True


def scan_board(name: str, db_path: Path) -> list[dict]:
    """Scan a single board for drift tasks."""
    findings = []
    try:
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        conn.row_factory = sqlite3.Row
    except Exception:
        return findings

    tables = {r[0] for r in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'"
    ).fetchall()}
    if "tasks" not in tables or "task_runs" not in tables:
        conn.close()
        return findings

    rows = conn.execute(
        "SELECT id, title, status, assignee, result, status_reason FROM tasks "
        "WHERE status IN ('done', 'completed')"
    ).fetchall()

    for row in rows:
        task = dict(row)
        tid = task["id"]
        title = task.get("title") or ""
        status = task.get("status") or ""

        if is_review_task(title):
            continue

        # Manual closures (human housekeeping with an explanation) are
        # NOT drift. Skip them so the circuit breaker doesn't trip on
        # legitimate bulk cleanup sessions. A task is deliberate when the
        # event stream shows a review signal, human comment, decomposition,
        # or a human status_reason — not just when status_reason is set.
        if is_deliberate_close(conn, task):
            continue

        runs = get_runs(conn, tid)
        completed = successful_runs(runs)

        # Drift class 1: terminal but zero successful runs
        if not completed and runs:
            findings.append({
                "task_id": tid,
                "title": title,
                "board": name,
                "status": status,
                "assignee": task.get("assignee"),
                "kind": "terminal_no_success",
                "run_count": len(runs),
                "run_outcomes": [(r.get("outcome") or r.get("status")) for r in runs],
            })

        # Drift class 2: terminal but zero runs at all (manual CLI complete?)
        if not runs and status in ("done", "completed"):
            findings.append({
                "task_id": tid,
                "title": title,
                "board": name,
                "status": status,
                "assignee": task.get("assignee"),
                "kind": "terminal_no_runs",
                "run_count": 0,
                "run_outcomes": [],
            })

    conn.close()
    return findings


def detect_profile_drift() -> list[dict]:
    """Detect profile overlay DBs that have tasks in terminal status
    while the same task in the base board is in a non-terminal status."""
    findings = []
    base_tasks: dict[str, str] = {}  # task_id -> status

    # Collect base board statuses
    for name, path in board_dbs().items():
        if name.startswith("profile:"):
            continue
        try:
            conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
            tables = {r[0] for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()}
            if "tasks" not in tables:
                conn.close()
                continue
            for row in conn.execute("SELECT id, status FROM tasks"):
                base_tasks[row[0]] = row[1]
            conn.close()
        except Exception:
            continue

    # Check profile overlays
    for name, path in board_dbs().items():
        if not name.startswith("profile:"):
            continue
        try:
            conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
            tables = {r[0] for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()}
            if "tasks" not in tables:
                conn.close()
                continue
            for row in conn.execute(
                "SELECT id, title, status FROM tasks "
                "WHERE status IN ('done', 'completed', 'archived')"
            ):
                tid, title, prof_status = row[0], row[1], row[2]
                base_status = base_tasks.get(tid)
                if base_status and base_status not in ("done", "completed", "archived"):
                    findings.append({
                        "task_id": tid,
                        "title": title,
                        "board": name,
                        "status": prof_status,
                        "base_status": base_status,
                        "kind": "profile_overlay_drift",
                    })
            conn.close()
        except Exception:
            continue

    return findings


def load_state() -> dict:
    try:
        return json.loads(DRIFT_STATE.read_text())
    except Exception:
        return {"last_run": 0, "drift_keys": []}


def save_state(state: dict) -> None:
    if _DRY_RUN:
        return
    DRIFT_STATE.parent.mkdir(parents=True, exist_ok=True)
    tmp = DRIFT_STATE.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, indent=2, sort_keys=True))
    tmp.replace(DRIFT_STATE)


def main() -> str:
    """Main entry point. Returns alert text (empty string = clean)."""
    global _DRY_RUN
    if "--dry-run" in sys.argv:
        _DRY_RUN = True
    dbs = board_dbs()
    all_findings: list[dict] = []
    fixed: list[dict] = []

    # Phase 1: Detect and fix terminal drift on base boards
    for name, path in dbs.items():
        if name.startswith("profile:"):
            continue  # Handle profiles separately
        findings = scan_board(name, path)
        all_findings.extend(findings)

    # Phase 2: Auto-fix drifted tasks (reopen to triage)
    breaker_tripped = False
    if all_findings:
        # Open a writable connection to each affected board
        fixable = [f for f in all_findings if f["kind"] in ("terminal_no_success", "terminal_no_runs")]
        # Circuit breaker: refuse to mass-reopen. A run that wants to reopen
        # more than MAX_REOPEN_PER_RUN tasks is almost certainly a systemic
        # fault, not genuine drift — reopening them all would inject a churn
        # storm. Reopen nothing this run; alert for human review instead.
        if len(fixable) > MAX_REOPEN_PER_RUN:
            breaker_tripped = True
            fixable = []
        by_board: dict[str, list[dict]] = {}
        for f in fixable:
            by_board.setdefault(f["board"], []).append(f)

        for board_name, board_findings in by_board.items():
            db_path = dbs.get(board_name)
            if not db_path:
                continue
            try:
                conn = sqlite3.connect(str(db_path))
                for f in board_findings:
                    reason = (
                        f"Auto-reopened by kanban-reconcile: {f['kind']}. "
                        f"Runs: {f['run_count']}, outcomes: {f['run_outcomes']}"
                    )
                    if write_lock is not None:
                        with write_lock(conn):
                            if reopen_task(conn, f["task_id"], reason):
                                fixed.append(f)
                    else:
                        if reopen_task(conn, f["task_id"], reason):
                            fixed.append(f)
                conn.close()
            except Exception:
                continue

    # Phase 3: Detect profile overlay drift (report only)
    profile_findings = detect_profile_drift()
    all_findings.extend(profile_findings)

    # Build delta state
    state = load_state()
    current_keys = sorted(f["task_id"] + ":" + f["kind"] for f in all_findings)
    delta = current_keys != state.get("drift_keys", [])

    save_state({
        "last_run": now_ts(),
        "drift_keys": current_keys,
        "fixed_count": len(fixed),
    })

    # Build output
    if not all_findings:
        return ""  # silent = healthy

    lines = []

    if breaker_tripped:
        fixable_count = len([
            f for f in all_findings
            if f["kind"] in ("terminal_no_success", "terminal_no_runs")
        ])
        lines.append(
            f"🛑 **Kanban reconciliation circuit breaker TRIPPED**: "
            f"{fixable_count} tasks looked drifted (cap {MAX_REOPEN_PER_RUN}). "
            f"Auto-reopen SKIPPED to avoid a churn storm. Investigate the "
            f"cause (bad heuristic, schema change, bulk import) and reopen "
            f"manually if genuine."
        )

    if fixed:
        lines.append(f"🔧 **Kanban reconciliation: {len(fixed)} task(s) reopened to triage**")
        for f in fixed:
            lines.append(
                f"  • `{f['task_id']}` [{f['board']}] — "
                f"{f['run_count']} runs, outcomes: {f['run_outcomes']}. "
                f"{f['title'][:60]}"
            )

    unfixed = [f for f in all_findings if f not in fixed and f["kind"] != "profile_overlay_drift"]
    if unfixed:
        lines.append(f"\n⚠️ **{len(unfixed)} drift finding(s) not auto-fixed:**")
        for f in unfixed:
            lines.append(
                f"  • `{f['task_id']}` [{f['board']}] {f['kind']} — "
                f"{f['title'][:60]}"
            )

    if profile_findings:
        lines.append(
            f"\nℹ️ **{len(profile_findings)} profile overlay drift(s)** — "
            f"profile DBs show terminal status but base board is non-terminal. "
            f"Cosmetic (profile DBs are stale snapshots)."
        )

    return "\n".join(lines)


if __name__ == "__main__":
    output = main()
    if output:
        print(output)
