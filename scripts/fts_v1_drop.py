#!/usr/bin/env python3
"""Retire the legacy v1 FTS objects after the v2 cutover has soaked.

Drops messages_fts + messages_fts_trigram and their six triggers — the
tables triple-store message text (content copies in both FTS tables plus
the trigram index bigger than the data), and every message write pays two
extra tokenizations to maintain indexes nothing reads anymore.

Preflight refuses to drop unless the v2 index is actually serving:
tokenizer loadable, table + triggers present, backfill-ready marker set,
integrity-check green, and rowcount equal to messages.

Freed pages return to SQLite's freelist (reused by future writes). Actual
file shrink needs VACUUM — hours-long exclusive rewrite of a multi-GB DB —
so it is behind --vacuum and OFF by default.

Usage:
  venv/bin/python scripts/fts_v1_drop.py [--db PATH] [--yes] [--vacuum]
"""

import argparse
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from hermes_state import (  # noqa: E402
    DEFAULT_DB_PATH,
    FTS_V2_READY_KEY,
    fts5_cjk_so_path,
    load_fts5_cjk_extension,
)

V1_OBJECTS = (
    ("trigger", "messages_fts_insert"),
    ("trigger", "messages_fts_delete"),
    ("trigger", "messages_fts_update"),
    ("trigger", "messages_fts_trigram_insert"),
    ("trigger", "messages_fts_trigram_delete"),
    ("trigger", "messages_fts_trigram_update"),
    ("table", "messages_fts"),
    ("table", "messages_fts_trigram"),
)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=str(DEFAULT_DB_PATH))
    ap.add_argument("--yes", action="store_true", help="skip confirmation")
    ap.add_argument("--vacuum", action="store_true",
                    help="VACUUM afterwards (exclusive lock, can take a long time)")
    args = ap.parse_args()

    conn = sqlite3.connect(args.db, timeout=30.0, isolation_level=None)
    conn.row_factory = sqlite3.Row

    # ── preflight: v2 must be serving ──
    if not load_fts5_cjk_extension(conn):
        print(f"ABORT: tokenizer extension not loadable ({fts5_cjk_so_path()})")
        return 1
    names = {
        r["name"] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE name LIKE 'messages_fts_v2%'"
        )
    }
    missing = {
        "messages_fts_v2", "messages_fts_v2_insert",
        "messages_fts_v2_delete", "messages_fts_v2_update",
    } - names
    if missing:
        print(f"ABORT: v2 objects missing: {sorted(missing)}")
        return 1
    marker = conn.execute(
        "SELECT value FROM state_meta WHERE key = ?", (FTS_V2_READY_KEY,)
    ).fetchone()
    if not marker or str(marker[0]) != "1":
        print(f"ABORT: {FTS_V2_READY_KEY} marker not set — run fts_v2_migrate.py first")
        return 1
    print("v2 integrity-check ...")
    conn.execute("INSERT INTO messages_fts_v2(messages_fts_v2) VALUES('integrity-check')")
    n_msg = conn.execute("SELECT count(*) FROM messages").fetchone()[0]
    n_v2 = conn.execute("SELECT count(*) FROM messages_fts_v2").fetchone()[0]
    print(f"rows: messages={n_msg} fts_v2={n_v2}")
    if n_msg != n_v2:
        print("ABORT: v2 rowcount does not match messages")
        return 1

    present = [
        (t, n) for (t, n) in V1_OBJECTS
        if conn.execute("SELECT 1 FROM sqlite_master WHERE name = ?", (n,)).fetchone()
    ]
    if not present:
        print("nothing to do: v1 objects already gone")
        return 0
    print("will drop:", ", ".join(n for _, n in present))
    if not args.yes:
        if input("proceed? [y/N] ").strip().lower() != "y":
            print("aborted")
            return 1

    conn.execute("BEGIN IMMEDIATE")
    try:
        for typ, name in present:
            conn.execute(f"DROP {typ} IF EXISTS {name}")
        conn.execute("COMMIT")
    except BaseException:
        conn.execute("ROLLBACK")
        raise
    print("dropped.")

    freelist = conn.execute("PRAGMA freelist_count").fetchone()[0]
    page_size = conn.execute("PRAGMA page_size").fetchone()[0]
    print(f"freelist: {freelist} pages (~{freelist * page_size / 1e9:.2f} GB reusable)")
    conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")

    if args.vacuum:
        print("VACUUM (this can take a long time and blocks writers) ...")
        conn.execute("VACUUM")
        print("vacuum done")

    print("v1 retirement complete. Legacy read fallbacks now degrade to "
          "LIKE-only for 1-char CJK queries; everything else serves from v2.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
