"""Multi-process concurrency stress test for the Kanban kernel.

5 worker processes race for claims on a shared DB with 100 tasks. Each
worker loops: claim -> simulate work -> complete. Asserts the invariants
that make the system worth building:

  - No task claimed by two workers simultaneously
  - No task completed twice
  - Every claim produces exactly one run row
  - Every completion closes exactly one run row
  - Zero SQLite locking errors that escape the retry layer
  - Total run count == total claim events == total completed events

This test is the primary justification for WAL + CAS-based claim. If it
passes, the architecture holds. If it fails, we have a real bug to fix
before anyone runs this in anger.
"""

import json
import multiprocessing as mp
import os
import random
import sqlite3
import subprocess
import sys
import tempfile
import time
from pathlib import Path


NUM_WORKERS = 5
NUM_TASKS = 100
WORKER_TIMEOUT_S = 60
WT = str(Path(__file__).resolve().parents[2])


def worker_loop(worker_id: int, hermes_home: str, result_file: str) -> None:
    """One worker's inner loop. Runs in a fresh Python process.

    Tries to claim a ready task, marks it done with a per-worker summary,
    repeats until the ready pool is empty. Records every claim + complete
    into its own JSON result file for later aggregation.
    """
    os.environ["HERMES_HOME"] = hermes_home
    os.environ["HOME"] = hermes_home
    sys.path.insert(0, WT)

    from hermes_cli import kanban_db as kb

    events = []
    empty_polls = 0
    start = time.monotonic()

    while time.monotonic() - start < WORKER_TIMEOUT_S:
        conn = kb.connect()
        try:
            # Find any ready task (non-deterministic order intentional — we
            # want workers to race on popular assignees).
            row = conn.execute(
                "SELECT id FROM tasks WHERE status = 'ready' "
                "AND claim_lock IS NULL LIMIT 1"
            ).fetchone()
            if row is None:
                empty_polls += 1
                if empty_polls > 20:
                    break  # queue empty long enough, stop
                time.sleep(0.01)
                continue
            empty_polls = 0

            tid = row["id"]
            try:
                claimed = kb.claim_task(
                    conn, tid, claimer=f"worker-{worker_id}",
                )
            except sqlite3.OperationalError as e:
                events.append({"kind": "sqlite_err_on_claim", "task": tid, "err": str(e)})
                continue
            if claimed is None:
                # Someone else beat us — expected contention, not an error.
                events.append({"kind": "lost_claim_race", "task": tid})
                continue

            run = kb.latest_run(conn, tid)
            events.append({
                "kind": "claimed",
                "task": tid,
                "worker": worker_id,
                "run_id": run.id,
                "t": time.monotonic() - start,
            })

            # Simulate short, variable work
            time.sleep(random.uniform(0.001, 0.05))

            try:
                kb.complete_task(
                    conn, tid,
                    result=f"done by worker-{worker_id}",
                    summary=f"worker-{worker_id} finished task {tid}",
                    metadata={"worker_id": worker_id, "run_id": run.id},
                )
            except sqlite3.OperationalError as e:
                events.append({"kind": "sqlite_err_on_complete", "task": tid, "err": str(e)})
                continue
            events.append({
                "kind": "completed",
                "task": tid,
                "worker": worker_id,
                "run_id": run.id,
                "t": time.monotonic() - start,
            })
        finally:
            conn.close()

    with open(result_file, "w") as f:
        json.dump(events, f)


def main():
    home = tempfile.mkdtemp(prefix="hermes_concurrency_")
    print(f"HERMES_HOME = {home}")

    # Seed.
    os.environ["HERMES_HOME"] = home
    os.environ["HOME"] = home
    sys.path.insert(0, WT)
    from hermes_cli import kanban_db as kb

    kb.init_db()
    conn = kb.connect()
    tids = []
    for i in range(NUM_TASKS):
        tid = kb.create_task(
            conn, title=f"task #{i}", assignee="shared",
            tenant="concurrency-test",
        )
        tids.append(tid)
    conn.close()
    print(f"Seeded {NUM_TASKS} tasks.")

    # Spawn workers.
    ctx = mp.get_context("spawn")
    result_files = [f"/tmp/concurrency_worker_{i}.json" for i in range(NUM_WORKERS)]
    procs = []
    start = time.monotonic()
    for i in range(NUM_WORKERS):
        p = ctx.Process(target=worker_loop, args=(i, home, result_files[i]))
        p.start()
        procs.append(p)

    for p in procs:
        p.join(timeout=WORKER_TIMEOUT_S + 30)
        if p.is_alive():
            p.terminate()
            p.join()

    elapsed = time.monotonic() - start
    print(f"All workers done in {elapsed:.1f}s")

    # Aggregate worker events.
    all_events = []
    for i, f in enumerate(result_files):
        if not os.path.isfile(f):
            print(f"  WORKER {i} produced no result file — died?")
            continue
        with open(f) as fh:
            events = json.load(fh)
        all_events.extend(events)

    # ============ INVARIANT CHECKS ============
    print()
    print("=" * 60)
    print("INVARIANT CHECKS")
    print("=" * 60)

    failures = []

    # Check 1: no task claimed by two different workers
    claims_by_task = {}
    for e in all_events:
        if e["kind"] == "claimed":
            if e["task"] in claims_by_task:
                prev = claims_by_task[e["task"]]
                if prev["worker"] != e["worker"]:
                    failures.append(
                        f"DOUBLE CLAIM: task {e['task']} claimed by "
                        f"worker {prev['worker']} AND worker {e['worker']}"
                    )
            claims_by_task[e["task"]] = e

    # Check 2: every completion has a matching claim from the same worker
    for e in all_events:
        if e["kind"] == "completed":
            prev_claim = claims_by_task.get(e["task"])
            if prev_claim is None:
                failures.append(f"COMPLETION WITHOUT CLAIM: task {e['task']}")
            elif prev_claim["worker"] != e["worker"]:
                failures.append(
                    f"WORKER MISMATCH: task {e['task']} claimed by "
                    f"{prev_claim['worker']} but completed by {e['worker']}"
                )

    # Check 3: DB state — every task should be in 'done', no dangling claims
    conn = kb.connect()
    try:
        bad_status = conn.execute(
            "SELECT id, status, claim_lock, current_run_id FROM tasks "
            "WHERE status != 'done' OR claim_lock IS NOT NULL "
            "OR current_run_id IS NOT NULL"
        ).fetchall()
        if bad_status:
            for row in bad_status:
                failures.append(
                    f"BAD FINAL STATE: task {row['id']} status={row['status']} "
                    f"claim_lock={row['claim_lock']} current_run_id={row['current_run_id']}"
                )

        # Check 4: exactly one run per task, all closed as completed
        bad_runs = conn.execute(
            "SELECT task_id, COUNT(*) as n FROM task_runs "
            "GROUP BY task_id HAVING n != 1"
        ).fetchall()
        if bad_runs:
            for row in bad_runs:
                failures.append(
                    f"WRONG RUN COUNT: task {row['task_id']} has {row['n']} runs (expected 1)"
                )

        open_runs = conn.execute(
            "SELECT id, task_id FROM task_runs WHERE ended_at IS NULL"
        ).fetchall()
        for row in open_runs:
            failures.append(f"OPEN RUN: run {row['id']} on task {row['task_id']}")

        wrong_outcomes = conn.execute(
            "SELECT task_id, outcome FROM task_runs "
            "WHERE outcome IS NULL OR outcome != 'completed'"
        ).fetchall()
        for row in wrong_outcomes:
            failures.append(
                f"WRONG OUTCOME: task {row['task_id']} run outcome={row['outcome']}"
            )

        # Check 5: event counts — exactly NUM_TASKS completed events
        completed_events = conn.execute(
            "SELECT COUNT(*) as n FROM task_events WHERE kind='completed'"
        ).fetchone()["n"]
        if completed_events != NUM_TASKS:
            failures.append(
                f"EVENT COUNT MISMATCH: {completed_events} completed events "
                f"expected {NUM_TASKS}"
            )

        # Check 6: count SQLite errors that escaped retry
        sqlite_errs = sum(
            1 for e in all_events if e["kind"].startswith("sqlite_err")
        )
        if sqlite_errs > 0:
            failures.append(f"UNRETRIED SQLITE ERRORS: {sqlite_errs}")

    finally:
        conn.close()

    # ============ STATS ============
    print()
    total_claims = sum(1 for e in all_events if e["kind"] == "claimed")
    total_completes = sum(1 for e in all_events if e["kind"] == "completed")
    total_lost_races = sum(1 for e in all_events if e["kind"] == "lost_claim_race")

    per_worker = {}
    for e in all_events:
        if e["kind"] == "completed":
            per_worker.setdefault(e["worker"], 0)
            per_worker[e["worker"]] += 1

    print(f"Total claims:      {total_claims}")
    print(f"Total completes:   {total_completes}")
    print(f"Lost claim races:  {total_lost_races}  (expected contention; not a bug)")
    print(f"Elapsed:           {elapsed:.2f}s")
    print(f"Throughput:        {NUM_TASKS/elapsed:.1f} tasks/sec")
    print(f"Per-worker completions:")
    for w in sorted(per_worker.keys()):
        print(f"  worker-{w}: {per_worker[w]}")

    if failures:
        print()
        print("=" * 60)
        print(f"FAILURES ({len(failures)}):")
        print("=" * 60)
        for f in failures[:20]:
            print(f"  {f}")
        if len(failures) > 20:
            print(f"  ... and {len(failures) - 20} more")
        sys.exit(1)
    else:
        print()
        print("✔ ALL INVARIANTS HELD")


if __name__ == "__main__":
    main()
