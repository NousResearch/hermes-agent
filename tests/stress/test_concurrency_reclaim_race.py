"""Target the reclaim race specifically.

Workers claim tasks with a 1s TTL but sleep 2s before completing. The
reclaimer runs every 200ms. Scenario: worker claims, reclaimer expires
the claim mid-work, worker tries to complete AFTER its run has been
reclaimed.

Expected behavior (per design): the worker's complete_task should
either succeed on the reclaimed-and-re-claimed-by-another-worker case
(no, it should refuse — the claim was invalidated), OR succeed by
grace (we "forgive" a late complete from the original worker if no
one else picked it up).

Actually looking at complete_task: it doesn't check claim_lock. It just
transitions from 'running' -> 'done'. So if the reclaimer moved it back
to 'ready', the late worker's complete_task will fail (CAS on
status='running' fails). This is the CORRECT behavior.

Invariant being tested: race between worker.complete and
dispatcher.reclaim must not produce a double-run-close or other
inconsistency.
"""

import json
import multiprocessing as mp
import os
import random
import sqlite3
import sys
import tempfile
import time
from pathlib import Path

NUM_WORKERS = 5
NUM_TASKS = 50
TTL = 1
WORK_DURATION_S = 2.0  # longer than TTL => reclaimer wins
WT = str(Path(__file__).resolve().parents[2])


def worker_loop(worker_id: int, hermes_home: str, result_file: str) -> None:
    os.environ["HERMES_HOME"] = hermes_home
    os.environ["HOME"] = hermes_home
    sys.path.insert(0, WT)
    from hermes_cli import kanban_db as kb

    events = []
    start = time.monotonic()
    idle = 0

    while time.monotonic() - start < 40:
        conn = kb.connect()
        try:
            row = conn.execute(
                "SELECT id FROM tasks WHERE status='ready' AND claim_lock IS NULL LIMIT 1"
            ).fetchone()
            if row is None:
                idle += 1
                if idle > 30:
                    break
                time.sleep(0.05)
                continue
            idle = 0
            tid = row["id"]
            try:
                claimed = kb.claim_task(conn, tid, claimer=f"worker-{worker_id}",
                                        ttl_seconds=TTL)
            except sqlite3.OperationalError as e:
                events.append({"kind": "sqlite_err", "op": "claim", "err": str(e)[:100]})
                continue
            if claimed is None:
                events.append({"kind": "lost_claim", "task": tid})
                continue
            run = kb.latest_run(conn, tid)
            events.append({"kind": "claimed", "task": tid, "worker": worker_id,
                           "run_id": run.id})

            # Sleep longer than TTL so reclaimer has a chance to intervene
            time.sleep(WORK_DURATION_S + random.uniform(-0.3, 0.3))

            try:
                ok = kb.complete_task(
                    conn, tid,
                    result=f"by worker-{worker_id}",
                    summary=f"worker-{worker_id} finished",
                )
                events.append({"kind": "complete_ok" if ok else "complete_refused",
                               "task": tid, "worker": worker_id, "run_id": run.id})
            except sqlite3.OperationalError as e:
                events.append({"kind": "sqlite_err", "op": "complete", "err": str(e)[:100]})
        finally:
            conn.close()

    with open(result_file, "w") as f:
        json.dump(events, f)


def reclaimer_loop(hermes_home: str, result_file: str) -> None:
    os.environ["HERMES_HOME"] = hermes_home
    os.environ["HOME"] = hermes_home
    sys.path.insert(0, WT)
    from hermes_cli import kanban_db as kb

    events = []
    start = time.monotonic()
    while time.monotonic() - start < 42:
        conn = kb.connect()
        try:
            try:
                n = kb.release_stale_claims(conn)
                if n:
                    events.append({"kind": "reclaimed", "count": n,
                                   "t": time.monotonic() - start})
            except sqlite3.OperationalError as e:
                events.append({"kind": "sqlite_err", "err": str(e)[:100]})
        finally:
            conn.close()
        time.sleep(0.2)
    with open(result_file, "w") as f:
        json.dump(events, f)


def main():
    home = tempfile.mkdtemp(prefix="hermes_reclaim_race_")
    os.environ["HERMES_HOME"] = home
    os.environ["HOME"] = home
    sys.path.insert(0, WT)
    from hermes_cli import kanban_db as kb

    kb.init_db()
    conn = kb.connect()
    for i in range(NUM_TASKS):
        kb.create_task(conn, title=f"t{i}", assignee="shared",
                       tenant="reclaim-race")
    conn.close()
    print(f"Seeded {NUM_TASKS} tasks. TTL={TTL}s, work_duration={WORK_DURATION_S}s")
    print(f"(worker work > TTL guarantees reclaims)")

    ctx = mp.get_context("spawn")
    worker_results = [f"/tmp/rc_worker_{i}.json" for i in range(NUM_WORKERS)]
    reclaim_result = "/tmp/rc_reclaim.json"
    procs = []
    for i in range(NUM_WORKERS):
        p = ctx.Process(target=worker_loop, args=(i, home, worker_results[i]))
        p.start()
        procs.append(p)
    r = ctx.Process(target=reclaimer_loop, args=(home, reclaim_result))
    r.start()
    procs.append(r)

    for p in procs:
        p.join(timeout=60)
        if p.is_alive():
            p.terminate()
            p.join()

    # Aggregate.
    all_events = []
    for f in worker_results:
        if os.path.isfile(f):
            with open(f) as fh:
                all_events.extend(json.load(fh))
    reclaim_events = []
    if os.path.isfile(reclaim_result):
        with open(reclaim_result) as fh:
            reclaim_events = json.load(fh)

    op_counts = {}
    for e in all_events:
        op_counts[e["kind"]] = op_counts.get(e["kind"], 0) + 1
    total_reclaims = sum(e.get("count", 0) for e in reclaim_events)
    print(f"\nReclaimer fired {len(reclaim_events)} times, total tasks reclaimed: {total_reclaims}")
    print("Worker events:")
    for k in sorted(op_counts):
        print(f"  {k:<25} {op_counts[k]}")

    # Invariant checks
    failures = []
    conn = kb.connect()
    try:
        # Any task stuck with current_run_id pointing at a closed run?
        bad = conn.execute("""
            SELECT t.id, t.status, t.current_run_id, r.ended_at, r.outcome
            FROM tasks t
            JOIN task_runs r ON r.id = t.current_run_id
            WHERE r.ended_at IS NOT NULL
        """).fetchall()
        for row in bad:
            failures.append(
                f"INVARIANT VIOLATION: task {row['id']} status={row['status']} "
                f"current_run_id={row['current_run_id']} but run ended "
                f"outcome={row['outcome']}"
            )
        # Every run with NULL ended_at should still have the task pointing at it
        orphans = conn.execute("""
            SELECT r.id, r.task_id
            FROM task_runs r
            LEFT JOIN tasks t ON t.current_run_id = r.id
            WHERE r.ended_at IS NULL AND t.id IS NULL
        """).fetchall()
        for row in orphans:
            failures.append(f"ORPHAN OPEN RUN: run {row['id']} on task {row['task_id']}")
        # Event counts
        claim_evts = conn.execute(
            "SELECT COUNT(*) FROM task_events WHERE kind='claimed'").fetchone()[0]
        reclaim_evts = conn.execute(
            "SELECT COUNT(*) FROM task_events WHERE kind='reclaimed'").fetchone()[0]
        comp_evts = conn.execute(
            "SELECT COUNT(*) FROM task_events WHERE kind='completed'").fetchone()[0]
        print(f"\nDB event counts: claimed={claim_evts} reclaimed={reclaim_evts} completed={comp_evts}")
        # Every reclaimed run must have ended_at set
        unended_reclaims = conn.execute(
            "SELECT COUNT(*) FROM task_runs WHERE outcome='reclaimed' AND ended_at IS NULL"
        ).fetchone()[0]
        if unended_reclaims:
            failures.append(f"UNENDED RECLAIMED RUNS: {unended_reclaims}")
        # Count of completed runs
        comp_runs = conn.execute(
            "SELECT COUNT(*) FROM task_runs WHERE outcome='completed'"
        ).fetchone()[0]
        reclaim_runs = conn.execute(
            "SELECT COUNT(*) FROM task_runs WHERE outcome='reclaimed'"
        ).fetchone()[0]
        print(f"DB run outcomes: completed={comp_runs} reclaimed={reclaim_runs}")
    finally:
        conn.close()

    if reclaim_runs == 0:
        failures.append("NO RECLAIMS HAPPENED — test didn't stress what it was supposed to")

    if failures:
        print(f"\nFAILURES ({len(failures)}):")
        for f in failures[:20]:
            print(f"  {f}")
        sys.exit(1)
    else:
        print("\n✔ RECLAIM RACE INVARIANTS HELD")


if __name__ == "__main__":
    main()
