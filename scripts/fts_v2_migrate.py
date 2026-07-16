#!/usr/bin/env python3
"""Online migration to the messages_fts_v2 CJK-bigram index.

Order of operations (safe with live writers):
  1. Load the cjk_unicode61 extension; create the v2 table + triggers
     (idempotent). From this moment every INSERT/UPDATE/DELETE on
     messages is mirrored into v2 by trigger.
  2. Snapshot max(id); backfill ids <= snapshot in batches. Each batch is
     DELETE-then-INSERT inside one transaction, so the backfill is
     idempotent and safe to interleave with trigger writes (standalone
     FTS5 table — plain rowid deletes are always safe, unlike external
     content). Progress persists in state_meta, so a killed run resumes.
  3. Verify: FTS integrity-check + rowcount parity against messages. Only
     a verified-complete index gets the fts_v2_ready marker that lets the
     read path serve v2. A needs_backfill marker (set when a tokenizer-less
     process had to drop the triggers, or when the table was recreated
     after a repair) forces a full re-backfill from scratch.

Read cutover is config-authoritative and reversible: once the ready marker
is set, reads serve from v2 by default. Set agent.fts_v2_read: false in
~/.hermes/config.yaml (and reload the gateways) to fall back to the legacy
tables; they stay in place until scripts/fts_v1_drop.py retires them.

Usage:
  venv/bin/python scripts/fts_v2_migrate.py [--db PATH] [--batch N] [--dry-run]
"""

import argparse
import sqlite3
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from hermes_state import (  # noqa: E402
    DEFAULT_DB_PATH,
    FTS_V2_NEEDS_BACKFILL,
    FTS_V2_READY_KEY,
    FTS_V2_SQL,
    fts5_cjk_so_path,
    load_fts5_cjk_extension,
)

PROGRESS_KEY = "fts_v2_backfill_next_lo"
SNAPSHOT_KEY = "fts_v2_backfill_snapshot_max"


def meta_get(conn, key):
    row = conn.execute("SELECT value FROM state_meta WHERE key = ?", (key,)).fetchone()
    return row[0] if row else None


def meta_set(conn, key, value):
    conn.execute(
        "INSERT INTO state_meta(key, value) VALUES (?, ?) "
        "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
        (key, str(value)),
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=str(DEFAULT_DB_PATH))
    ap.add_argument("--batch", type=int, default=20000)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    conn = sqlite3.connect(args.db, timeout=30.0, isolation_level=None)
    conn.row_factory = sqlite3.Row

    if not load_fts5_cjk_extension(conn):
        print(f"ERROR: cannot load tokenizer extension at {fts5_cjk_so_path()}")
        return 1
    print(f"tokenizer loaded: {fts5_cjk_so_path()}")

    if args.dry_run:
        n = conn.execute("SELECT count(*), max(id) FROM messages").fetchone()
        print(f"messages rows={n[0]} max_id={n[1]} batch={args.batch} "
              f"→ ~{(n[1] or 0) // args.batch + 1} batches")
        return 0

    # 1. table + triggers (idempotent)
    conn.executescript(FTS_V2_SQL)
    print("v2 table + triggers ensured")

    # A needs_backfill marker means the index is known to have missed writes
    # (a tokenizer-less process dropped the triggers, or the table was
    # recreated after a repair). Any persisted snapshot/progress describe the
    # PREVIOUS run — discard them so the backfill re-covers everything from
    # id 1 (the triggers ensured above cover writes from this point on).
    if meta_get(conn, FTS_V2_READY_KEY) == FTS_V2_NEEDS_BACKFILL:
        print(f"{FTS_V2_READY_KEY}={FTS_V2_NEEDS_BACKFILL} — forcing a full re-backfill")
        conn.execute(
            "DELETE FROM state_meta WHERE key IN (?, ?)",
            (SNAPSHOT_KEY, PROGRESS_KEY),
        )

    # 2. snapshot + resumable backfill
    snap = meta_get(conn, SNAPSHOT_KEY)
    if snap is None:
        snap = conn.execute("SELECT COALESCE(max(id), 0) FROM messages").fetchone()[0]
        meta_set(conn, SNAPSHOT_KEY, snap)
    snap = int(snap)
    lo = int(meta_get(conn, PROGRESS_KEY) or 1)
    print(f"backfill ids [{lo}..{snap}] in batches of {args.batch}")

    t0 = time.time()
    done_rows = 0
    while lo <= snap:
        hi = min(lo + args.batch - 1, snap)
        bt = time.time()
        conn.execute("BEGIN IMMEDIATE")
        try:
            conn.execute(
                "DELETE FROM messages_fts_v2 WHERE rowid BETWEEN ? AND ?", (lo, hi)
            )
            cur = conn.execute(
                "INSERT INTO messages_fts_v2(rowid, content, tool_name, tool_calls) "
                "SELECT id, COALESCE(content,''), COALESCE(tool_name,''), COALESCE(tool_calls,'') "
                "FROM messages WHERE id BETWEEN ? AND ?",
                (lo, hi),
            )
            done_rows += cur.rowcount
            meta_set(conn, PROGRESS_KEY, hi + 1)
            conn.execute("COMMIT")
        except BaseException:
            conn.execute("ROLLBACK")
            raise
        pct = 100.0 * hi / snap
        print(f"  [{pct:5.1f}%] ids {lo}-{hi}  +{cur.rowcount} rows  "
              f"({time.time()-bt:.1f}s, total {done_rows} in {time.time()-t0:.0f}s)",
              flush=True)
        lo = hi + 1

    # 3. verification — only a verified-complete index may be marked ready.
    # Counts and marker share one transaction so a writer racing this check
    # cannot produce a ready marker that a fresh count would contradict.
    print("running FTS integrity-check ...")
    conn.execute("INSERT INTO messages_fts_v2(messages_fts_v2) VALUES('integrity-check')")
    conn.execute("BEGIN IMMEDIATE")
    try:
        n_v2 = conn.execute("SELECT count(*) FROM messages_fts_v2").fetchone()[0]
        n_msg = conn.execute("SELECT count(*) FROM messages").fetchone()[0]
        if n_msg == n_v2:
            meta_set(conn, FTS_V2_READY_KEY, "1")
        conn.execute("COMMIT")
    except BaseException:
        conn.execute("ROLLBACK")
        raise
    print(f"rows: messages={n_msg} fts_v2={n_v2}")
    if n_msg != n_v2:
        print(f"ERROR: rowcount mismatch — {FTS_V2_READY_KEY} NOT set; "
              "re-run the migration")
        return 1
    print(f"{FTS_V2_READY_KEY}=1 (read path may now serve v2)")

    for q in ("웅기", "일본 MCP", "graphiti", "구글 캘린더"):
        t = time.time()
        rows = conn.execute(
            "SELECT count(*) FROM messages_fts_v2 WHERE messages_fts_v2 MATCH ?", (q,)
        ).fetchone()[0]
        print(f"  sample MATCH {q!r}: {rows} hits in {(time.time()-t)*1000:.0f}ms")

    print("backfill complete. Reads serve from v2 by default once gateways "
          "reload; set agent.fts_v2_read: false in ~/.hermes/config.yaml to "
          "fall back to the legacy tables.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
