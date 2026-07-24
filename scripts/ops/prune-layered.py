#!/usr/bin/env python3
"""prune-layered.py — Tiered session retention for Hermes Agent state.db.

Hermes state.db grows unboundedly (#54189).  This script implements a
three-tier retention model directly against the sqlite3 database without
importing any Hermes internal module:

  Tier  keep_full  keep_meta  retention
       ───────────┬───────────┬──────────› age
          ①       │     ②     │     ③
  ① < keep_full_days:     keep all messages untouched
  ② keep~keep_meta:       drop tool messages, keep user+assistant
  ③ keep_meta~retention:  drop all messages, keep session metadata
  ④ > retention_days:     delete the session entirely

Usage
-----
  python3 prune-layered.py [options]

Options
-------
  --keep-full-days N   Keep all messages for N days                [7]
  --keep-meta-days N   Keep user+assistant for N days, drop tool   [30]
  --retention-days N   Delete whole sessions older than N days     [90]
  --source S           Only prune sessions from SOURCE (cron / cli …)
  --dry-run            Preview without modifying the database
  --vacuum             Run VACUUM after pruning to reclaim space
  --db-path PATH       Path to state.db  (default: auto-detect)
  --sessions-dir PATH  Path to sessions/ directory for file cleanup
  --json               Output JSON instead of human-readable text

Auto-detection of state.db (first match wins):
  1. --db-path argument
  2. $HERMES_HOME/state.db
  3. ~/.hermes/state.db

Exit code: 0 on success, 1 on error.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sqlite3
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

# ────────────────────────────────────────────────────────────────
#  Schema helpers  (knows the Hermes state.db schema, nothing more)
# ────────────────────────────────────────────────────────────────

SCHEMA_SESSIONS = """
    SELECT id, started_at, ended_at, source, message_count
    FROM sessions
    WHERE ended_at IS NOT NULL
      AND archived = 0
      AND (? < 0 OR ended_at < ?)
"""

SCHEMA_MESSAGES_BY_ROLE = """
    DELETE FROM messages
    WHERE session_id IN ({ph}) AND role = 'tool'
"""

SCHEMA_MESSAGES = """
    DELETE FROM messages
    WHERE session_id IN ({ph})
"""

SCHEMA_CHILDREN_ORPHAN = """
    UPDATE sessions
    SET parent_session_id = NULL
    WHERE parent_session_id IN ({ph})
"""

SCHEMA_SESSIONS_DELETE = """
    DELETE FROM sessions
    WHERE id IN ({ph})
"""

SCHEMA_KEEP_FULL_COUNT = """
    SELECT COUNT(*) FROM sessions
    WHERE ended_at IS NOT NULL
      AND ended_at > ?
      AND archived = 0
"""

SCHEMA_FTS_REBUILD = "INSERT INTO messages_fts(messages_fts) VALUES('rebuild')"
SCHEMA_TRIGRAM_REBUILD = "INSERT INTO messages_fts_trigram(messages_fts_trigram) VALUES('rebuild')"


# ────────────────────────────────────────────────────────────────
#  Core logic
# ────────────────────────────────────────────────────────────────

def _find_state_db(db_path: Optional[str] = None) -> Path:
    """Resolve the state.db path in order: explicit → HERMES_HOME → ~/.hermes."""
    if db_path:
        return Path(db_path).expanduser().resolve()
    env_home = os.environ.get("HERMES_HOME")
    if env_home:
        candidate = Path(env_home) / "state.db"
        if candidate.exists():
            return candidate
    default = Path.home() / ".hermes" / "state.db"
    if not default.exists():
        print(f"Error: state.db not found at {default}.  Use --db-path to specify.", file=sys.stderr)
        sys.exit(1)
    return default


def prune_layered(
    db_path: Path,
    keep_full_days: int = 7,
    keep_meta_days: int = 30,
    retention_days: int = 90,
    source: Optional[str] = None,
    sessions_dir: Optional[Path] = None,
    dry_run: bool = False,
    vacuum: bool = False,
) -> Dict[str, Any]:
    """Three-tier session retention directly on state.db.

    Returns a dict with per-tier counters:
        tool_msg_deleted, meta_session_count, meta_msg_deleted,
        session_count, keep_full_sessions (dry-run only)
    """
    now = time.time()
    result: Dict[str, Any] = {
        "tool_msg_deleted": 0,
        "meta_session_count": 0,
        "meta_msg_deleted": 0,
        "session_count": 0,
    }

    # Clamp tiers to prevent overlaps
    keep_full_days = max(keep_full_days, 0)
    keep_meta_days = max(keep_meta_days, keep_full_days)
    retention_days = max(retention_days, keep_meta_days)
    meta_window = keep_meta_days > keep_full_days
    delete_window = retention_days > keep_meta_days

    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=5000")

    try:
        # --- Collect candidate sessions ---
        source_clause = ""
        params: list = [keep_full_days]
        if source:
            source_clause = "AND source = ?"
            params.append(source)

        rows = conn.execute(
            f"SELECT id, started_at, ended_at FROM sessions "
            f"WHERE ended_at IS NOT NULL AND archived = 0 "
            f"AND ended_at < ? {source_clause} "
            f"ORDER BY ended_at ASC",
            [now - keep_full_days * 86400] + ([source] if source else []),
        ).fetchall()

        if not rows:
            if dry_run:
                result["_dry_run"] = True
                result["keep_full_sessions"] = conn.execute(
                    SCHEMA_KEEP_FULL_COUNT,
                    [now - keep_full_days * 86400],
                ).fetchone()[0]
            return result

        # --- Categorise into tiers ---
        cut_meta = now - keep_meta_days * 86400
        cut_del = now - retention_days * 86400

        tool_sids: List[str] = []
        meta_sids: List[str] = []
        del_sids: List[str] = []

        for sid, started_at, ended_at in rows:
            started = started_at or 0
            if started < cut_del:
                del_sids.append(sid)
            elif started < cut_meta and delete_window:
                meta_sids.append(sid)
            elif started < cut_meta and not delete_window and meta_window:
                tool_sids.append(sid)
            elif meta_window:
                tool_sids.append(sid)

        # --- Dry-run: report counts only ---
        if dry_run:
            result["_dry_run"] = True
            result["tool_sessions"] = len(tool_sids)
            result["meta_sessions"] = len(meta_sids)
            result["delete_sessions"] = len(del_sids)
            result["keep_full_sessions"] = conn.execute(
                SCHEMA_KEEP_FULL_COUNT,
                [now - keep_full_days * 86400],
            ).fetchone()[0]

            if tool_sids:
                ph = ",".join("?" * len(tool_sids))
                row = conn.execute(
                    f"SELECT COUNT(*) FROM messages WHERE session_id IN ({ph}) AND role='tool'",
                    tool_sids,
                ).fetchone()
                result["tool_msg_preview"] = row[0]
            if meta_sids:
                ph = ",".join("?" * len(meta_sids))
                row = conn.execute(
                    f"SELECT COUNT(*) FROM messages WHERE session_id IN ({ph})",
                    meta_sids,
                ).fetchone()
                result["meta_msg_preview"] = row[0]
            return result

        # --- Tier 1: drop tool messages ---
        if tool_sids and meta_window:
            ph = ",".join("?" * len(tool_sids))
            cur = conn.execute(
                f"DELETE FROM messages WHERE session_id IN ({ph}) AND role='tool'",
                tool_sids,
            )
            result["tool_msg_deleted"] = cur.rowcount

        # --- Tier 2: drop all messages (keep session rows) ---
        if meta_sids:
            ph = ",".join("?" * len(meta_sids))
            cur = conn.execute(f"DELETE FROM messages WHERE session_id IN ({ph})", meta_sids)
            result["meta_session_count"] = len(meta_sids)
            result["meta_msg_deleted"] = cur.rowcount

        # --- Tier 3: delete entire sessions ---
        if del_sids:
            ph = ",".join("?" * len(del_sids))
            # Orphan children (safe: column may not exist in older schemas)
            try:
                conn.execute(
                    f"UPDATE sessions SET parent_session_id = NULL "
                    f"WHERE parent_session_id IN ({ph})",
                    del_sids,
                )
            except sqlite3.OperationalError:
                pass
            conn.execute(f"DELETE FROM messages WHERE session_id IN ({ph})", del_sids)
            conn.execute(f"DELETE FROM sessions WHERE id IN ({ph})", del_sids)
            result["session_count"] = len(del_sids)

            # Clean up session files on disk (best-effort)
            if sessions_dir:
                for sid in del_sids:
                    _remove_session_files(sessions_dir, sid)

        conn.commit()

        # --- Rebuild FTS indexes (after commit) ---
        if (tool_sids and meta_window) or meta_sids or del_sids:
            try:
                conn.execute("INSERT INTO messages_fts(messages_fts) VALUES('rebuild')")
            except sqlite3.OperationalError:
                pass  # FTS5 table may not exist with external content
            try:
                conn.execute("INSERT INTO messages_fts_trigram(messages_fts_trigram) VALUES('rebuild')")
            except sqlite3.OperationalError:
                pass

        # --- Optional VACUUM (separate connection, never in a transaction) ---
        if vacuum:
            conn.close()
            vconn = sqlite3.connect(str(db_path))
            vconn.execute("VACUUM")
            vconn.close()
            conn = sqlite3.connect(str(db_path))  # reopen for the return

    finally:
        conn.close()

    return result


def _remove_session_files(sessions_dir: Path, session_id: str) -> None:
    """Remove on-disk files associated with a deleted session."""
    if sessions_dir is None:
        return
    sdir = sessions_dir / session_id
    if not sdir.exists():
        return
    for f in sdir.rglob("*"):
        try:
            if f.is_file():
                f.unlink()
        except OSError:
            pass
    try:
        shutil.rmtree(sdir, ignore_errors=True)
    except OSError:
        pass


# ────────────────────────────────────────────────────────────────
#  CLI
# ────────────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Tiered session retention for Hermes Agent state.db",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--keep-full-days", type=int, default=7, help="Keep all messages for N days (default: 7)")
    p.add_argument("--keep-meta-days", type=int, default=30, help="Keep user+assistant for N days (default: 30)")
    p.add_argument("--retention-days", type=int, default=90, help="Delete sessions older than N days (default: 90)")
    p.add_argument("--source", help="Only prune sessions from SOURCE (e.g. cron, cli)")
    p.add_argument("--dry-run", action="store_true", help="Preview without deleting")
    p.add_argument("--vacuum", action="store_true", help="VACUUM after pruning")
    p.add_argument("--db-path", help="Path to state.db (auto-detected if omitted)")
    p.add_argument("--sessions-dir", help="Path to sessions/ directory for on-disk cleanup")
    p.add_argument("--json", action="store_true", help="Output JSON")
    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = _build_parser().parse_args(argv)

    db_path = _find_state_db(args.db_path)
    sessions_dir = Path(args.sessions_dir).expanduser() if args.sessions_dir else None

    result = prune_layered(
        db_path=db_path,
        keep_full_days=args.keep_full_days,
        keep_meta_days=args.keep_meta_days,
        retention_days=args.retention_days,
        source=args.source,
        sessions_dir=sessions_dir,
        dry_run=args.dry_run,
        vacuum=args.vacuum,
    )

    if args.json:
        print(json.dumps(result, indent=2))
        return 0

    if result.get("_dry_run"):
        print("Layered prune — dry run (nothing deleted)")
        print(f"  < {args.keep_full_days}d: {result.get('keep_full_sessions', 0)} session(s) — kept fully")
        print(f"  {args.keep_full_days}d–{args.keep_meta_days}d: {result.get('tool_sessions', 0)} session(s) — {result.get('tool_msg_preview', 0)} tool message(s) to drop")
        print(f"  {args.keep_meta_days}d–{args.retention_days}d: {result.get('meta_sessions', 0)} session(s) — {result.get('meta_msg_preview', 0)} message(s) to drop, metadata kept")
        print(f"  > {args.retention_days}d: {result.get('delete_sessions', 0)} session(s) — to delete entirely")
    else:
        total = result.get("tool_msg_deleted", 0) + result.get("meta_msg_deleted", 0)
        print(f"Layered prune complete.")
        print(f"  Tool messages deleted:  {result.get('tool_msg_deleted', 0)}")
        print(f"  Sessions stripped:      {result.get('meta_session_count', 0)} ({result.get('meta_msg_deleted', 0)} messages)")
        print(f"  Sessions deleted:       {result.get('session_count', 0)}")
        print(f"  Total messages freed:   {total}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
