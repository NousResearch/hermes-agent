"""Benchmark for issue #95316: list_sessions_rich compression-chain N+1.

Builds a synthetic store (R compression roots x D-deep chains + filler
sessions) and measures list_sessions_rich() wall time (best of N reps) plus,
in a separate instrumented call, how many times the per-hop
get_compression_tip() walk is entered and how many times the batched edge
index is built. Run on the unpatched tree for the "before" column and on the
patched tree for "after".

Usage: python scripts/benchmark_list_sessions_rich.py [roots] [depth]
"""

import sys
import tempfile
import time
from pathlib import Path

# Always benchmark THIS checkout's hermes_state, not an installed copy.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from hermes_state import SessionDB  # noqa: E402


def build_store(db, roots: int, depth: int):
    """roots conversations, each compressed `depth` times (chain len depth+1),
    plus enough live filler rows to make a realistic page."""
    t0 = time.time() - 10 * 365 * 86400
    n = 0
    for r in range(roots):
        root = f"root{r}"
        db.create_session(root, "cli")
        db._conn.execute("UPDATE sessions SET started_at=? WHERE id=?", (t0 + n, root))
        db.append_message(root, "user", f"conversation {r} start")
        current = root
        for d in range(depth):
            db._conn.execute(
                "UPDATE sessions SET ended_at=?, end_reason='compression' WHERE id=?",
                (t0 + n + 60, current),
            )
            child = f"c{r}_{d}"
            db.create_session(child, "cli", parent_session_id=current)
            db._conn.execute(
                "UPDATE sessions SET started_at=? WHERE id=?", (t0 + n + 61, child)
            )
            db.append_message(child, "user", f"continuation {d} of {r}")
            if d < depth - 1:
                db._conn.execute(
                    "UPDATE sessions SET ended_at=?, end_reason='compression' WHERE id=?",
                    (t0 + n + 120, child),
                )
            current = child
        n += 2
    # Filler: plain live sessions interleaved among the chain tips so a
    # realistic LIMIT window contains both plain rows and compression roots.
    for f in range(roots):
        sid = f"filler{f}"
        db.create_session(sid, "cli")
        db._conn.execute(
            "UPDATE sessions SET started_at=? WHERE id=?",
            (t0 + (2 * f) + 30, sid),
        )
        db.append_message(sid, "user", f"filler {f}")
    db._conn.commit()


def main():
    roots = int(sys.argv[1]) if len(sys.argv) > 1 else 40
    depth = int(sys.argv[2]) if len(sys.argv) > 2 else 5

    tmp = tempfile.mkdtemp(prefix="bench95316-")
    db = SessionDB(db_path=Path(tmp) / "bench.db")
    build_store(db, roots, depth)

    kwargs_list = (
        ("order_by_last_active=True", dict(order_by_last_active=True)),
        ("order_by_last_active=False", dict(order_by_last_active=False)),
    )
    limit = roots * 2 + 20  # whole store fits on one page

    results = {}
    for label, kwargs in kwargs_list:
        # Timing: best of 9 reps (first rep warms up caches).
        best = None
        for i in range(9):
            t_start = time.perf_counter()
            rows = db.list_sessions_rich(source="cli", limit=limit, **kwargs)
            elapsed = time.perf_counter() - t_start
            best = elapsed if best is None else min(best, elapsed)

        # One instrumented call for exact per-listing counters.
        # sql_stmts counts every statement on every connection (writer +
        # WAL read-pool) via sqlite trace callbacks; tip_calls counts
        # entries into the per-hop walker.
        stmts = {"n": 0}

        def bump(_stmt, _c=stmts):
            _c["n"] += 1

        db._conn.set_trace_callback(bump)
        orig_checkout = db._checkout_read_conn

        def tracing_checkout(_orig=orig_checkout, _bump=bump):
            conn = _orig()
            if conn is not None:
                try:
                    conn.set_trace_callback(_bump)
                except Exception:
                    pass
            return conn

        db._checkout_read_conn = tracing_checkout
        tip_calls = {"n": 0}
        orig_tip = SessionDB.get_compression_tip

        def counting_tip(self, session_id, _orig=orig_tip, _c=tip_calls):
            _c["n"] += 1
            return _orig(self, session_id)

        SessionDB.get_compression_tip = counting_tip
        try:
            rows = db.list_sessions_rich(source="cli", limit=limit, **kwargs)
        finally:
            SessionDB.get_compression_tip = orig_tip
            db._conn.set_trace_callback(None)
            db._checkout_read_conn = orig_checkout
        projected = sum(1 for r in rows if r["id"].startswith("c") and "_" in r["id"])

        results[label] = {
            "ms": best * 1000.0,
            "tip_calls": tip_calls["n"],
            "sql_stmts": stmts["n"],
            "rows": len(rows),
            "projected": projected,
        }

    db.close()

    print(f"store: {roots} roots x depth {depth} chains (+{roots} fillers)")
    for label, r in results.items():
        print(
            f"  {label}: {r['ms']:.2f} ms | get_compression_tip calls: "
            f"{r['tip_calls']} | SQL statements: {r['sql_stmts']} | rows: "
            f"{r['rows']} | projected-to-tip rows: {r['projected']}"
        )


if __name__ == "__main__":
    main()
