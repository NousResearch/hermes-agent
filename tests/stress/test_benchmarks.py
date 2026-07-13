"""Scale benchmarks for the Kanban kernel.

Measures:
  - dispatch_once latency at 100, 1000, 10000 tasks
  - recompute_ready latency at 100, 1000, 10000 todo tasks with wide parent graphs
  - build_worker_context latency with 1, 10, 50 parent dependencies
  - board list/stats query latency
  - task_runs query latency at scale

Results printed as a table. Saved to JSON for regression-diffing in CI
or future reviews. Not a pass/fail test — records numbers so we know
when a change regresses latency by 10x and can decide whether to care.
"""

import json
import os
import random
import sys
import tempfile
import time
from pathlib import Path

WT = str(Path(__file__).resolve().parents[2])


def bench(label, fn, iterations=5):
    """Time fn over `iterations` runs, return (min, median, max) in ms."""
    times = []
    for _ in range(iterations):
        t0 = time.perf_counter()
        fn()
        times.append((time.perf_counter() - t0) * 1000)
    times.sort()
    mn = times[0]
    md = times[len(times) // 2]
    mx = times[-1]
    return {"label": label, "iter": iterations, "min_ms": mn, "median_ms": md, "max_ms": mx}


def seed_tasks(conn, kb, n, assignee="bench-worker", with_parents=False):
    """Seed n tasks. Optionally give each task 5 parents."""
    ids = []
    for i in range(n):
        if with_parents and i >= 5:
            parents = random.sample(ids[:i], 5)
        else:
            parents = ()
        tid = kb.create_task(
            conn, title=f"bench {i}", assignee=assignee,
            tenant="bench", parents=parents,
        )
        ids.append(tid)
    return ids


def main():
    home = tempfile.mkdtemp(prefix="hermes_bench_")
    os.environ["HERMES_HOME"] = home
    os.environ["HOME"] = home
    sys.path.insert(0, WT)
    from hermes_cli import kanban_db as kb

    kb.init_db()

    results = []

    # ============ dispatch_once latency ============
    for n in [100, 1000, 10000]:
        print(f"\n== dispatch_once @ {n} tasks ==")
        # Fresh DB each time so we're not measuring cumulative effects
        import shutil
        shutil.rmtree(home, ignore_errors=True)
        os.makedirs(home)
        kb._INITIALIZED_PATHS.clear()
        kb.init_db()
        conn = kb.connect()
        seed_tasks(conn, kb, n, assignee=None)  # no assignee → won't spawn
        r = bench(
            f"dispatch_once (n={n}, no spawn)",
            lambda: kb.dispatch_once(conn, spawn_fn=lambda *_: None),
            iterations=5,
        )
        print(f"  min={r['min_ms']:.1f} median={r['median_ms']:.1f} max={r['max_ms']:.1f} ms")
        r["n"] = n
        results.append(r)
        conn.close()

    # ============ recompute_ready at scale with parent graphs ============
    for n in [100, 1000, 10000]:
        print(f"\n== recompute_ready @ {n} tasks (5 parents each) ==")
        shutil.rmtree(home, ignore_errors=True)
        os.makedirs(home)
        kb._INITIALIZED_PATHS.clear()
        kb.init_db()
        conn = kb.connect()
        ids = seed_tasks(conn, kb, n, assignee=None, with_parents=True)
        # Complete the first 100 so some todo tasks might get promoted
        for tid in ids[:min(100, n // 10)]:
            kb.complete_task(conn, tid, result="bench")
        r = bench(
            f"recompute_ready (n={n}, with parents)",
            lambda: kb.recompute_ready(conn),
            iterations=5,
        )
        print(f"  min={r['min_ms']:.1f} median={r['median_ms']:.1f} max={r['max_ms']:.1f} ms")
        r["n"] = n
        results.append(r)
        conn.close()

    # ============ build_worker_context with N parents ============
    for parent_count in [1, 10, 50]:
        print(f"\n== build_worker_context with {parent_count} parents ==")
        shutil.rmtree(home, ignore_errors=True)
        os.makedirs(home)
        kb._INITIALIZED_PATHS.clear()
        kb.init_db()
        conn = kb.connect()
        # Create parents, complete them with summaries+metadata
        parent_ids = []
        for i in range(parent_count):
            pid = kb.create_task(conn, title=f"parent {i}", assignee="p")
            kb.claim_task(conn, pid)
            kb.complete_task(
                conn, pid,
                summary=f"parent {i} result that is longer than a single token "
                        f"so we actually measure the IO",
                metadata={"files": [f"file_{j}.py" for j in range(5)], "i": i},
            )
            parent_ids.append(pid)
        child_id = kb.create_task(
            conn, title="child", assignee="c", parents=parent_ids,
        )
        r = bench(
            f"build_worker_context (parents={parent_count})",
            lambda: kb.build_worker_context(conn, child_id),
            iterations=10,
        )
        print(f"  min={r['min_ms']:.1f} median={r['median_ms']:.1f} max={r['max_ms']:.1f} ms")
        r["parent_count"] = parent_count
        results.append(r)
        conn.close()

    # ============ list_tasks at scale ============
    for n in [100, 1000, 10000]:
        print(f"\n== list_tasks @ {n} ==")
        shutil.rmtree(home, ignore_errors=True)
        os.makedirs(home)
        kb._INITIALIZED_PATHS.clear()
        kb.init_db()
        conn = kb.connect()
        seed_tasks(conn, kb, n)
        r = bench(
            f"list_tasks (n={n})",
            lambda: kb.list_tasks(conn),
            iterations=5,
        )
        print(f"  min={r['min_ms']:.1f} median={r['median_ms']:.1f} max={r['max_ms']:.1f} ms")
        r["n"] = n
        results.append(r)
        conn.close()

    # ============ board_stats at scale ============
    for n in [100, 1000, 10000]:
        print(f"\n== board_stats @ {n} ==")
        shutil.rmtree(home, ignore_errors=True)
        os.makedirs(home)
        kb._INITIALIZED_PATHS.clear()
        kb.init_db()
        conn = kb.connect()
        seed_tasks(conn, kb, n)
        r = bench(
            f"board_stats (n={n})",
            lambda: kb.board_stats(conn),
            iterations=5,
        )
        print(f"  min={r['min_ms']:.1f} median={r['median_ms']:.1f} max={r['max_ms']:.1f} ms")
        r["n"] = n
        results.append(r)
        conn.close()

    # ============ list_runs at scale ============
    for n in [100, 1000]:
        print(f"\n== list_runs for task with {n} attempts ==")
        shutil.rmtree(home, ignore_errors=True)
        os.makedirs(home)
        kb._INITIALIZED_PATHS.clear()
        kb.init_db()
        conn = kb.connect()
        tid = kb.create_task(conn, title="x", assignee="w")
        # Create N attempts via claim/release
        for i in range(n):
            kb.claim_task(conn, tid, ttl_seconds=0)
            kb.release_stale_claims(conn)
        r = bench(
            f"list_runs (runs={n})",
            lambda: kb.list_runs(conn, tid),
            iterations=10,
        )
        print(f"  min={r['min_ms']:.1f} median={r['median_ms']:.1f} max={r['max_ms']:.1f} ms")
        r["run_count"] = n
        results.append(r)
        conn.close()

    # ============ SUMMARY TABLE ============
    print()
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"{'Benchmark':<50} {'min':>8} {'median':>8} {'max':>8}")
    for r in results:
        print(f"{r['label']:<50} {r['min_ms']:>7.1f}ms {r['median_ms']:>7.1f}ms {r['max_ms']:>7.1f}ms")

    # Save for future diffing.
    out_path = "/tmp/kanban_bench_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
