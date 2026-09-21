#!/usr/bin/env python3
"""
kanban-reclaim-zombies.py — Hermes Kanban zombie-run reclaimer.

Conservative, idempotent, read-only-ish maintenance:
  1. Reclaims task_runs stuck in status='running' whose worker PID no longer
     exists (/proc/<pid> gone). The task already finished; the run row was
     never flipped to terminal. We re-check liveness at write time, so a live
     worker can NEVER be reclaimed.
  2. Clears expired claim_lock rows on TERMINAL tasks only (done/archived/
     completed) — an expired lock on a terminal task is dead weight.
  3. Nulls current_run_id on terminal tasks whose pointed run was reclaimed,
     keeping the denormalised pointer honest.

Scans all canonical board DBs by default. Use --db PATH to restrict to one
board (e.g. an immediate one-shot on the ops board).

Exit code: 0 always (watchdog must not fail loudly); errors go to stderr.
"""
import os
import sys
import glob
import sqlite3

# P13 isolation: HERMES_HOME parameterises the kanban root so a
# disposable run never touches /home/kensei/.hermes. When --db is passed
# the explicit path wins regardless.
_HERMES_HOME = os.environ.get("HERMES_HOME", "/home/kensei/.hermes")
CANONICAL_DBS = [os.path.join(_HERMES_HOME, "kanban.db")] + sorted(
    glob.glob(os.path.join(_HERMES_HOME, "kanban/boards/*/kanban.db"))
)

TERMINAL = ("done", "archived", "completed")


def proc_alive(pid):
    try:
        return os.path.exists(f"/proc/{int(pid)}")
    except (ValueError, TypeError):
        return False


def reclaim(db_path, dry_run=False):
    if not os.path.exists(db_path):
        return
    conn = sqlite3.connect(db_path, timeout=10)
    conn.execute("PRAGMA busy_timeout=10000")
    cur = conn.cursor()

    cur.execute(
        "SELECT id, worker_pid FROM task_runs "
        "WHERE status='running' AND worker_pid IS NOT NULL"
    )
    rows = cur.fetchall()
    dead = [(rid, pid) for (rid, pid) in rows if not proc_alive(pid)]

    reclaimed = 0
    reclaimed_ids = []
    for rid, pid in dead:
        if dry_run:
            continue
        cur.execute(
            "UPDATE task_runs SET status='reclaimed', outcome='reclaimed', "
            "ended_at=COALESCE(last_heartbeat_at, started_at), worker_pid=NULL "
            "WHERE id=?",
            (rid,),
        )
        reclaimed += 1
        reclaimed_ids.append(rid)

    # Clear expired claim locks on terminal tasks only.
    cur.execute(
        "SELECT id FROM tasks WHERE claim_lock IS NOT NULL "
        "AND claim_expires IS NOT NULL AND claim_expires < unixepoch('now') "
        "AND status IN ('done','archived','completed')"
    )
    lock_ids = [r[0] for r in cur.fetchall()]
    cleared = 0
    for tid in lock_ids:
        if dry_run:
            continue
        cur.execute(
            "UPDATE tasks SET claim_lock=NULL, claim_expires=NULL WHERE id=?",
            (tid,),
        )
        cleared += 1

    # Honour the denormalised pointer on terminal tasks.
    if reclaimed_ids and not dry_run:
        ph = ",".join("?" * len(reclaimed_ids))
        cur.execute(
            f"UPDATE tasks SET current_run_id=NULL "
            f"WHERE current_run_id IN ({ph}) AND status IN ('done','archived','completed')",
            reclaimed_ids,
        )

    if not dry_run:
        conn.commit()
    conn.close()

    reclaim_n = len(dead) if dry_run else reclaimed
    clear_n = len(lock_ids) if dry_run else cleared
    if dead or lock_ids:
        verb = "WOULD" if dry_run else "did"
        print(
            f"{db_path}: {verb} reclaim {reclaim_n} zombie run(s) "
            f"[{', '.join(str(i) for i,_ in dead)}], "
            f"clear {clear_n} expired lock(s) "
            f"[{', '.join(lock_ids)}]; {len(rows)-len(dead)} live running left"
        )
    # NB: when nothing to do we emit NOTHING — watchdog pattern: empty
    # stdout = silent, so the cron stays quiet between real events.


if __name__ == "__main__":
    dry = "--dry-run" in sys.argv
    single = None
    for a in sys.argv[1:]:
        if a.startswith("--db="):
            single = a.split("=", 1)[1]
        elif a == "--db" and len(sys.argv) > sys.argv.index(a) + 1:
            single = sys.argv[sys.argv.index(a) + 1]
    targets = [single] if single else CANONICAL_DBS
    for db in targets:
        try:
            reclaim(db, dry_run=dry)
        except Exception as e:  # noqa: BLE001
            print(f"{db}: ERROR {e}", file=sys.stderr)
