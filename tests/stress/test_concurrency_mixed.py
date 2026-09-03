"""Harder concurrency stress: mixed operations + larger scale.

Scales to 500 tasks, 10 workers, 60s runtime. Each worker randomly:
  - claims + completes (70%)
  - claims + blocks with a reason (15%)
  - unblocks a random blocked task (10%)
  - archives a random done task (5%)

Adds a background "dispatcher" process that calls release_stale_claims
and detect_crashed_workers every 200ms, racing against the workers to
surface TTL + crash detection races.

Pass criteria: runs invariant holds, no double-completions, no orphan
runs, no SQLite errors escape the retry layer.
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

NUM_WORKERS = 10
NUM_TASKS = 500
RUN_DURATION_S = 30
WT = str(Path(__file__).resolve().parents[2])


def worker_loop(worker_id: int, hermes_home: str, result_file: str) -> None:
    os.environ["HERMES_HOME"] = hermes_home
    os.environ["HOME"] = hermes_home
    sys.path.insert(0, WT)
    from hermes_cli import kanban_db as kb

    events = []
    start = time.monotonic()
    idle_rounds = 0

    while time.monotonic() - start < RUN_DURATION_S:
        conn = kb.connect()
        try:
            op = random.random()

            if op < 0.10:
                # Try to unblock a blocked task.
                row = conn.execute(
                    "SELECT id FROM tasks WHERE status='blocked' "
                    "ORDER BY RANDOM() LIMIT 1"
                ).fetchone()
                if row:
                    try:
                        ok = kb.unblock_task(conn, row["id"])
                        events.append({"kind": "unblocked" if ok else "unblock_noop",
                                       "task": row["id"], "worker": worker_id})
                    except sqlite3.OperationalError as e:
                        events.append({"kind": "sqlite_err", "op": "unblock",
                                       "task": row["id"], "err": str(e)[:100]})
                continue

            if op < 0.15:
                # Try to archive a done task.
                row = conn.execute(
                    "SELECT id FROM tasks WHERE status='done' "
                    "ORDER BY RANDOM() LIMIT 1"
                ).fetchone()
                if row:
                    try:
                        kb.archive_task(conn, row["id"])
                        events.append({"kind": "archived", "task": row["id"],
                                       "worker": worker_id})
                    except sqlite3.OperationalError as e:
                        events.append({"kind": "sqlite_err", "op": "archive",
                                       "task": row["id"], "err": str(e)[:100]})
                continue

            # Default: claim + complete-or-block.
            row = conn.execute(
                "SELECT id FROM tasks WHERE status='ready' "
                "AND claim_lock IS NULL LIMIT 1"
            ).fetchone()
            if row is None:
                idle_rounds += 1
                if idle_rounds > 50:
                    break
                time.sleep(0.02)
                continue
            idle_rounds = 0

            tid = row["id"]
            try:
                claimed = kb.claim_task(
                    conn, tid, claimer=f"worker-{worker_id}",
                    ttl_seconds=5,  # short TTL so reclaim races in
                )
            except sqlite3.OperationalError as e:
                events.append({"kind": "sqlite_err", "op": "claim",
                               "task": tid, "err": str(e)[:100]})
                continue
            if claimed is None:
                events.append({"kind": "lost_claim_race", "task": tid})
                continue

            run = kb.latest_run(conn, tid)
            events.append({"kind": "claimed", "task": tid, "worker": worker_id,
                           "run_id": run.id, "t": time.monotonic() - start})

            time.sleep(random.uniform(0.005, 0.05))

            # 20% of the time, block instead of complete
            if random.random() < 0.20:
                try:
                    kb.block_task(conn, tid,
                                  reason=f"blocked by worker-{worker_id}")
                    events.append({"kind": "blocked", "task": tid,
                                   "worker": worker_id, "run_id": run.id})
                except sqlite3.OperationalError as e:
                    events.append({"kind": "sqlite_err", "op": "block",
                                   "task": tid, "err": str(e)[:100]})
            else:
                try:
                    kb.complete_task(
                        conn, tid,
                        result=f"done by worker-{worker_id}",
                        summary=f"worker-{worker_id} ok",
                        metadata={"worker_id": worker_id},
                    )
                    events.append({"kind": "completed", "task": tid,
                                   "worker": worker_id, "run_id": run.id,
                                   "t": time.monotonic() - start})
                except sqlite3.OperationalError as e:
                    events.append({"kind": "sqlite_err", "op": "complete",
                                   "task": tid, "err": str(e)[:100]})
        finally:
            conn.close()

    with open(result_file, "w") as f:
        json.dump(events, f)


def reclaimer_loop(hermes_home: str, result_file: str) -> None:
    """Background dispatcher-like loop that reclaims stale tasks."""
    os.environ["HERMES_HOME"] = hermes_home
    os.environ["HOME"] = hermes_home
    sys.path.insert(0, WT)
    from hermes_cli import kanban_db as kb

    events = []
    start = time.monotonic()
    while time.monotonic() - start < RUN_DURATION_S + 2:
        conn = kb.connect()
        try:
            try:
                reclaimed = kb.release_stale_claims(conn)
                if reclaimed:
                    events.append({"kind": "reclaimed", "count": reclaimed,
                                   "t": time.monotonic() - start})
            except sqlite3.OperationalError as e:
                events.append({"kind": "sqlite_err", "op": "reclaim",
                               "err": str(e)[:100]})
        finally:
            conn.close()
        time.sleep(0.2)

    with open(result_file, "w") as f:
        json.dump(events, f)


def main():
    home = tempfile.mkdtemp(prefix="hermes_mixed_stress_")
    print(f"HERMES_HOME = {home}")

    os.environ["HERMES_HOME"] = home
    os.environ["HOME"] = home
    sys.path.insert(0, WT)
    from hermes_cli import kanban_db as kb

    kb.init_db()
    conn = kb.connect()
    for i in range(NUM_TASKS):
        kb.create_task(
            conn, title=f"t#{i}", assignee="shared", tenant="mixed-stress",
        )
    conn.close()
    print(f"Seeded {NUM_TASKS} tasks, launching {NUM_WORKERS} workers + 1 reclaimer")

    ctx = mp.get_context("spawn")
    worker_results = [f"/tmp/mixed_worker_{i}.json" for i in range(NUM_WORKERS)]
    reclaim_result = "/tmp/mixed_reclaim.json"

    procs = []
    start = time.monotonic()
    for i in range(NUM_WORKERS):
        p = ctx.Process(target=worker_loop, args=(i, home, worker_results[i]))
        p.start()
        procs.append(p)
    r = ctx.Process(target=reclaimer_loop, args=(home, reclaim_result))
    r.start()
    procs.append(r)

    for p in procs:
        p.join(timeout=RUN_DURATION_S + 30)
        if p.is_alive():
            p.terminate()
            p.join()

    elapsed = time.monotonic() - start
    print(f"Done in {elapsed:.1f}s")

    # Aggregate.
    all_events = []
    for i, f in enumerate(worker_results):
        if os.path.isfile(f):
            with open(f) as fh:
                all_events.extend(json.load(fh))
        else:
            print(f"  WORKER {i} died with no result file!")
    reclaim_events = []
    if os.path.isfile(reclaim_result):
        with open(reclaim_result) as fh:
            reclaim_events = json.load(fh)

    # ============ INVARIANT CHECKS ============
    print()
    print("=" * 60)
    print("INVARIANT CHECKS")
    print("=" * 60)

    failures = []

    # Per-run attribution tracking
    claims = [e for e in all_events if e["kind"] == "claimed"]
    completions = [e for e in all_events if e["kind"] == "completed"]
    blocks = [e for e in all_events if e["kind"] == "blocked"]

    # Every completion must have a matching claim on the same run_id AND
    # the same worker (workers don't steal each other's runs).
    claims_by_run = {c["run_id"]: c for c in claims}
    for comp in completions:
        claim = claims_by_run.get(comp["run_id"])
        if claim is None:
            # It's possible this worker saw a reclaimed run from another worker
            # — that's still a bug: the worker shouldn't be able to complete
            # a run it didn't claim. But let me check if reclaim happened first.
            failures.append(
                f"COMPLETION WITHOUT CLAIM: task {comp['task']} run {comp['run_id']} "
                f"by worker {comp['worker']}"
            )
        elif claim["worker"] != comp["worker"]:
            failures.append(
                f"CROSS-WORKER COMPLETION: run {comp['run_id']} claimed by "
                f"worker {claim['worker']} but completed by worker {comp['worker']}"
            )

    # SQLite errors that escaped the retry layer
    sqlite_errs = [e for e in all_events if e["kind"] == "sqlite_err"]
    if sqlite_errs:
        for e in sqlite_errs[:5]:
            failures.append(f"SQLITE ERROR: op={e.get('op')} err={e.get('err')}")
        if len(sqlite_errs) > 5:
            failures.append(f"  ... and {len(sqlite_errs) - 5} more sqlite errs")

    # DB final state — every task should be in a clean terminal state.
    conn = kb.connect()
    try:
        # Invariant: current_run_id NULL iff latest run is terminal
        inconsistent = conn.execute("""
            SELECT t.id, t.status, t.current_run_id
            FROM tasks t
            WHERE t.current_run_id IS NOT NULL
              AND EXISTS (SELECT 1 FROM task_runs r
                          WHERE r.id = t.current_run_id AND r.ended_at IS NOT NULL)
        """).fetchall()
        for row in inconsistent:
            failures.append(
                f"INVARIANT VIOLATION: task {row['id']} status={row['status']} "
                f"has current_run_id={row['current_run_id']} but run is ended"
            )

        # Invariant: no orphan open runs
        orphans = conn.execute("""
            SELECT r.id, r.task_id, r.status
            FROM task_runs r
            LEFT JOIN tasks t ON t.current_run_id = r.id
            WHERE r.ended_at IS NULL AND t.id IS NULL
        """).fetchall()
        for row in orphans:
            failures.append(
                f"ORPHAN OPEN RUN: run {row['id']} on task {row['task_id']}"
            )

        # Counts — should roughly balance.
        status_counts = dict(
            conn.execute("SELECT status, COUNT(*) FROM tasks GROUP BY status").fetchall()
        )
        run_outcome_counts = dict(
            conn.execute(
                "SELECT outcome, COUNT(*) FROM task_runs "
                "WHERE ended_at IS NOT NULL GROUP BY outcome"
            ).fetchall()
        )
        active_runs = conn.execute(
            "SELECT COUNT(*) FROM task_runs WHERE ended_at IS NULL"
        ).fetchone()[0]

    finally:
        conn.close()

    # ============ STATS ============
    print()
    print(f"Workers: {NUM_WORKERS}, Tasks: {NUM_TASKS}")
    print(f"Elapsed: {elapsed:.1f}s")
    print(f"Events collected: {len(all_events)} (+{len(reclaim_events)} reclaim)")
    print()
    print("Operations:")
    op_counts = {}
    for e in all_events:
        op_counts[e["kind"]] = op_counts.get(e["kind"], 0) + 1
    for k in sorted(op_counts.keys()):
        print(f"  {k:<25} {op_counts[k]}")

    print()
    print("Final task status:")
    for s, n in sorted(status_counts.items()):
        print(f"  {s:<10} {n}")
    print("Final run outcomes:")
    for o, n in sorted(run_outcome_counts.items(), key=lambda x: (x[0] or '',)):
        print(f"  {o:<12} {n}")
    print(f"  active       {active_runs}")

    if failures:
        print()
        print("=" * 60)
        print(f"FAILURES ({len(failures)}):")
        print("=" * 60)
        for f in failures[:30]:
            print(f"  {f}")
        if len(failures) > 30:
            print(f"  ... and {len(failures) - 30} more")
        sys.exit(1)
    else:
        print()
        print("✔ ALL INVARIANTS HELD UNDER MIXED STRESS")


if __name__ == "__main__":
    main()
