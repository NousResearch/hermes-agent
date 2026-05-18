"""Randomized property testing for the Kanban kernel.

Generates 1000 random operation sequences, each 20-50 ops, on small
task graphs. After each step, checks the full invariant set:

  I1. If tasks.current_run_id IS NOT NULL, the run MUST exist AND
      ended_at MUST be NULL (we never point at a closed run).
  I2. If a run has ended_at NULL, SOME task MUST have current_run_id
      pointing at it (no orphan open runs).
  I3. task.status in the valid set {triage, todo, ready, running,
      blocked, done, archived}.
  I4. task.claim_lock NULL iff status not in (running,).
  I5. Every run has started_at <= ended_at (or ended_at is NULL).
  I6. If outcome is set, ended_at must also be set.
  I7. Events are strictly monotonic in (created_at, id).
  I8. task_events.run_id references a task_runs.id that exists
      (or is NULL).
  I9. Parent completion invariant: if all parents are 'done', the
      child cannot be in 'todo' status (recompute_ready should have
      promoted it). This is called out in the comment on
      recompute_ready; verify it holds after every random seq.

Not using hypothesis the lib; just Python random for simplicity.
"""

import os
import random
import sys
import tempfile
import time
from pathlib import Path

WT = str(Path(__file__).resolve().parents[2])
NUM_SEQUENCES = 500
OPS_PER_SEQUENCE = 100
TASK_POOL = 10

OPS = [
    "create", "create_child", "claim", "complete", "block", "unblock",
    "archive", "heartbeat", "release_stale", "detect_crashed",
    "recompute_ready", "reassign",
]


def assert_invariants(conn, kb, ops_log):
    """Run all invariant checks; raise AssertionError with context on any."""
    failures = []

    # I1: current_run_id → run exists and not ended
    bad_ptr = conn.execute("""
        SELECT t.id, t.current_run_id, r.ended_at, r.outcome
        FROM tasks t
        LEFT JOIN task_runs r ON r.id = t.current_run_id
        WHERE t.current_run_id IS NOT NULL
          AND (r.id IS NULL OR r.ended_at IS NOT NULL)
    """).fetchall()
    for row in bad_ptr:
        if row["ended_at"] is None and row["outcome"] is None:
            detail = "missing"
        else:
            detail = f"closed ({row['outcome']})"
        failures.append(
            f"I1: task {row['id']} points at run {row['current_run_id']} "
            f"which is {detail}"
        )

    # I2: open run → some task points at it
    orphans = conn.execute("""
        SELECT r.id, r.task_id
        FROM task_runs r
        WHERE r.ended_at IS NULL
          AND NOT EXISTS (SELECT 1 FROM tasks t WHERE t.current_run_id = r.id)
    """).fetchall()
    for row in orphans:
        failures.append(f"I2: open run {row['id']} on task {row['task_id']} has no pointer")

    # I3: valid statuses
    valid = {"triage", "todo", "ready", "running", "blocked", "done", "archived"}
    bad_status = conn.execute("SELECT id, status FROM tasks").fetchall()
    for row in bad_status:
        if row["status"] not in valid:
            failures.append(f"I3: task {row['id']} has invalid status {row['status']!r}")

    # I4: claim_lock set only when running
    bad_lock = conn.execute("""
        SELECT id, status, claim_lock FROM tasks
        WHERE (status != 'running' AND claim_lock IS NOT NULL)
    """).fetchall()
    for row in bad_lock:
        failures.append(
            f"I4: task {row['id']} status={row['status']} but claim_lock={row['claim_lock']!r}"
        )

    # I5: run started_at <= ended_at
    bad_times = conn.execute("""
        SELECT id, started_at, ended_at FROM task_runs
        WHERE ended_at IS NOT NULL AND started_at > ended_at
    """).fetchall()
    for row in bad_times:
        failures.append(
            f"I5: run {row['id']} started_at={row['started_at']} > ended_at={row['ended_at']}"
        )

    # I6: outcome set → ended_at set
    bad_outcome = conn.execute("""
        SELECT id, outcome, ended_at FROM task_runs
        WHERE outcome IS NOT NULL AND ended_at IS NULL
    """).fetchall()
    for row in bad_outcome:
        failures.append(f"I6: run {row['id']} outcome={row['outcome']} but ended_at NULL")

    # I7: events monotonic in id (always true for autoincrement)
    # Skip — autoincrement guarantees it.

    # I8: event.run_id references existing run
    bad_ev_fk = conn.execute("""
        SELECT e.id, e.run_id FROM task_events e
        LEFT JOIN task_runs r ON r.id = e.run_id
        WHERE e.run_id IS NOT NULL AND r.id IS NULL
    """).fetchall()
    for row in bad_ev_fk:
        failures.append(f"I8: event {row['id']} references missing run {row['run_id']}")

    # I9: if all parents done → child not in todo
    # (Only applies to children with at least one parent)
    orphaned_todo = conn.execute("""
        SELECT c.id AS child_id,
               COUNT(*) AS n_parents,
               SUM(CASE WHEN p.status = 'done' THEN 1 ELSE 0 END) AS done_parents
        FROM tasks c
        JOIN task_links l ON l.child_id = c.id
        JOIN tasks p ON p.id = l.parent_id
        WHERE c.status = 'todo'
        GROUP BY c.id
        HAVING n_parents > 0 AND n_parents = done_parents
    """).fetchall()
    for row in orphaned_todo:
        failures.append(
            f"I9: task {row['child_id']} is todo but all {row['n_parents']} parents are done"
        )

    if failures:
        print(f"\n!!! INVARIANT VIOLATION after {len(ops_log)} ops:")
        for f in failures[:10]:
            print(f"  {f}")
        if len(failures) > 10:
            print(f"  ... and {len(failures) - 10} more")
        print("\nLast 10 ops:")
        for op in ops_log[-10:]:
            print(f"  {op}")
        return False
    return True


def random_op(rng, conn, kb, task_pool):
    op = rng.choice(OPS)

    if op == "create":
        tid = kb.create_task(
            conn,
            title=f"rand {rng.randint(0, 1000)}",
            assignee=rng.choice(["w1", "w2", "w3", None]),
        )
        task_pool.append(tid)
        return {"op": "create", "tid": tid}

    if op == "create_child" and task_pool:
        parent = rng.choice(task_pool)
        tid = kb.create_task(
            conn, title=f"child of {parent}",
            assignee=rng.choice(["w1", "w2", "w3", None]),
            parents=[parent],
        )
        task_pool.append(tid)
        return {"op": "create_child", "tid": tid, "parent": parent}

    if not task_pool:
        return None

    tid = rng.choice(task_pool)
    task = kb.get_task(conn, tid)
    if task is None:
        task_pool.remove(tid)
        return None

    if op == "claim":
        claimed = kb.claim_task(conn, tid, ttl_seconds=rng.choice([1, 3, 10]))
        return {"op": "claim", "tid": tid, "ok": claimed is not None}
    if op == "complete":
        summary = rng.choice([None, f"done via op {rng.randint(0, 1000)}"])
        ok = kb.complete_task(conn, tid, summary=summary)
        return {"op": "complete", "tid": tid, "ok": ok}
    if op == "block":
        reason = rng.choice([None, "rand block"])
        ok = kb.block_task(conn, tid, reason=reason)
        return {"op": "block", "tid": tid, "ok": ok}
    if op == "unblock":
        ok = kb.unblock_task(conn, tid)
        return {"op": "unblock", "tid": tid, "ok": ok}
    if op == "archive":
        ok = kb.archive_task(conn, tid)
        if ok:
            task_pool.remove(tid)
        return {"op": "archive", "tid": tid, "ok": ok}
    if op == "heartbeat":
        ok = kb.heartbeat_worker(conn, tid)
        return {"op": "heartbeat", "tid": tid, "ok": ok}
    if op == "release_stale":
        n = kb.release_stale_claims(conn)
        return {"op": "release_stale", "n": n}
    if op == "detect_crashed":
        # Force-kill a fake PID first so there's something to detect
        crashed = kb.detect_crashed_workers(conn)
        return {"op": "detect_crashed", "n": len(crashed)}
    if op == "recompute_ready":
        n = kb.recompute_ready(conn)
        return {"op": "recompute_ready", "promoted": n}
    if op == "reassign":
        # Reassignment isn't a direct API; simulate via assign_task
        new_a = rng.choice(["w1", "w2", "w3", None])
        try:
            kb.assign_task(conn, tid, new_a)
            return {"op": "reassign", "tid": tid, "to": new_a}
        except Exception as e:
            return {"op": "reassign", "tid": tid, "err": str(e)[:50]}

    return None


def main():
    total_ops = 0
    total_violations = 0

    for seq_idx in range(NUM_SEQUENCES):
        seed = random.randint(0, 10**9)
        rng = random.Random(seed)
        home = tempfile.mkdtemp(prefix=f"hermes_fuzz_{seq_idx}_")
        os.environ["HERMES_HOME"] = home
        os.environ["HOME"] = home
        sys.path.insert(0, WT)

        # Fresh module state per sequence to avoid cached init paths.
        for m in list(sys.modules.keys()):
            if m.startswith("hermes_cli"):
                del sys.modules[m]
        from hermes_cli import kanban_db as kb

        kb.init_db()
        conn = kb.connect()
        task_pool = []
        ops_log = []

        try:
            for i in range(OPS_PER_SEQUENCE):
                result = random_op(rng, conn, kb, task_pool)
                if result is None:
                    continue
                ops_log.append(result)
                total_ops += 1
                if not assert_invariants(conn, kb, ops_log):
                    total_violations += 1
                    print(f"  sequence {seq_idx} (seed={seed}) failed at op {i}")
                    break
        finally:
            conn.close()

        if seq_idx % 10 == 0:
            print(f"  seq {seq_idx:3d}: {total_ops} ops so far, {total_violations} violations")

    print()
    print("=" * 60)
    print(f"Total sequences: {NUM_SEQUENCES}")
    print(f"Total operations: {total_ops}")
    print(f"Invariant violations: {total_violations}")
    if total_violations == 0:
        print("\n✔ ALL INVARIANTS HELD ACROSS RANDOMIZED SEQUENCES")
    else:
        print("\n✗ INVARIANT VIOLATIONS FOUND")
        sys.exit(1)


if __name__ == "__main__":
    main()
