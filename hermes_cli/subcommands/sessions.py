"""``hermes sessions`` subcommand — session lifecycle management.

Exposes the backend pruning, archival, and VACUUM machinery that already
exists in :class:`~hermes_state.SessionDB` as user-facing CLI commands.

Commands:
    hermes sessions list [--all] [--archived] [--limit N] [--source S]
    hermes sessions prune [--days N] [--source S] [--dry-run]
    hermes sessions archive-old [--days N] [--dry-run]
    hermes sessions vacuum
    hermes sessions stats
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Callable


def build_sessions_parser(subparsers, *, cmd_sessions: Callable) -> None:
    """Attach the ``sessions`` subcommand to ``subparsers``."""
    sessions_parser = subparsers.add_parser(
        "sessions",
        help="Manage session history — list, prune, archive, vacuum",
        description="Session lifecycle management. Inspect, prune, archive, "
        "and maintain the session database (state.db).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
  hermes sessions list                   Show recent sessions (excludes archived)
  hermes sessions list --all             Include archived sessions
  hermes sessions stats                  Show database overview
  hermes sessions prune --days 30        Delete ended sessions older than 30 days
  hermes sessions prune --days 60 --dry-run  Preview without deleting
  hermes sessions archive-old --days 14  Soft-archive sessions older than 14 days
  hermes sessions vacuum                 Rebuild FTS indexes and reclaim space
""",
    )
    sessions_sub = sessions_parser.add_subparsers(dest="sessions_command")

    # ── list ──
    list_parser = sessions_sub.add_parser(
        "list", help="Browse sessions (excludes archived by default)"
    )
    list_parser.add_argument(
        "--all", action="store_true", help="Include archived sessions"
    )
    list_parser.add_argument(
        "--archived", action="store_true", help="Show only archived sessions"
    )
    list_parser.add_argument(
        "--limit", type=int, default=20, help="Max sessions to display (default: 20)"
    )
    list_parser.add_argument("--source", type=str, default=None, help="Filter by source")

    # ── prune ──
    prune_parser = sessions_sub.add_parser(
        "prune",
        help="Permanently delete ended sessions older than N days",
    )
    prune_parser.add_argument(
        "--days",
        type=int,
        default=90,
        help="Delete sessions older than N days (default: 90). Only ended sessions are pruned.",
    )
    prune_parser.add_argument(
        "--source", type=str, default=None, help="Only prune sessions from this source"
    )
    prune_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show how many would be pruned without deleting",
    )

    # ── archive-old ──
    archive_parser = sessions_sub.add_parser(
        "archive-old",
        help="Soft-archive sessions older than N days (hidden from default list)",
    )
    archive_parser.add_argument(
        "--days", type=int, default=30, help="Archive sessions older than N days (default: 30)"
    )
    archive_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show how many would be archived without changing anything",
    )

    # ── vacuum ──
    sessions_sub.add_parser(
        "vacuum",
        help="Rebuild FTS indexes and reclaim disk space (run after prune)",
    )

    # ── stats ──
    sessions_sub.add_parser(
        "stats",
        help="Show session database overview (size, counts, oldest session)",
    )

    sessions_parser.set_defaults(func=cmd_sessions)


def cmd_sessions(args: argparse.Namespace) -> int:
    """Dispatch ``hermes sessions`` subcommands."""
    command = getattr(args, "sessions_command", None)
    if command is None:
        print("Usage: hermes sessions <list|prune|archive-old|vacuum|stats>")
        return 0

    from hermes_constants import get_hermes_home
    from hermes_state import SessionDB

    hermes_home = get_hermes_home()
    db_path = hermes_home / "state.db"
    sessions_dir = hermes_home / "sessions"

    if not db_path.exists():
        print("No state.db found. Start a session first: `hermes`")
        return 1

    session_db = SessionDB(db_path)

    try:
        dispatch = {
            "list": _cmd_list,
            "prune": _cmd_prune,
            "archive-old": _cmd_archive_old,
            "vacuum": _cmd_vacuum,
            "stats": _cmd_stats,
        }
        handler = dispatch.get(command)
        if handler is None:
            print(f"Unknown sessions command: {command}")
            return 1
        if command in ("prune",):
            return handler(session_db, sessions_dir, args)
        return handler(session_db, args)
    finally:
        try:
            session_db.close()
        except Exception:
            pass


# ── Helpers ─────────────────────────────────────────────────────────────────


def _cmd_list(session_db: Any, args: argparse.Namespace) -> int:
    rows = session_db.list_sessions_rich(
        source=args.source,
        limit=args.limit,
        include_archived=args.all,
        archived_only=args.archived,
        order_by_last_active=True,
    )
    if not rows:
        print("No sessions found.")
        return 0

    header = f"{'Title':<40}  {'Source':<12}  {'Msgs':>5}  {'Session ID'}"
    print(header)
    print("-" * 80)
    for row in rows:
        title = (row.get("title") or "\u2014")[:40]
        src = (row.get("source") or "")[:12]
        msgs = str(row.get("message_count") or 0)
        sid = str(row.get("id") or "")
        print(f"{title:<40}  {src:<12}  {msgs:>5}  {sid}")

    print(f"\n{len(rows)} session(s) shown.")
    if not args.all and not args.archived:
        print("Use --all to include archived sessions, --archived for archived only.")
    return 0


def _cmd_prune(
    session_db: Any, sessions_dir: Path, args: argparse.Namespace
) -> int:
    days = args.days
    if days <= 0:
        print("Error: --days must be a positive integer.")
        return 1

    source = args.source
    label = f"older than {days} day(s)"
    if source:
        label += f" from source '{source}'"

    if args.dry_run:
        count = _count_prunable(session_db, days, source)
        print(f"Dry run: {count} ended session(s) {label} would be deleted.")
        return 0

    count = session_db.prune_sessions(
        older_than_days=days, source=source, sessions_dir=sessions_dir
    )
    if count > 0:
        print(f"Pruned {count} session(s) {label}.")
        print("Run `hermes sessions vacuum` to reclaim disk space.")
    else:
        print(f"No sessions {label} to prune.")
    return 0


def _cmd_archive_old(session_db: Any, args: argparse.Namespace) -> int:
    import time

    days = args.days
    if days <= 0:
        print("Error: --days must be a positive integer.")
        return 1

    cutoff = time.time() - (days * 86400)

    # Use the public listing API, then filter by age and archived status in
    # Python.  This avoids reaching into private _lock / _conn directly.
    all_rows = session_db.list_sessions_rich(
        limit=10000, include_archived=True, order_by_last_active=False
    )
    candidates = [
        r
        for r in all_rows
        if (r.get("started_at") or 0) < cutoff and not r.get("archived")
    ]

    if args.dry_run:
        print(
            f"Dry run: {len(candidates)} session(s) older than {days} day(s) "
            f"would be archived."
        )
        for c in candidates[:10]:
            print(
                f"  {(c.get('title') or '\u2014')[:50]:<50}  "
                f"{(c.get('source') or ''):<10}  {c.get('id', '')}"
            )
        if len(candidates) > 10:
            print(f"  ... and {len(candidates) - 10} more")
        return 0

    archived = 0
    for c in candidates:
        if session_db.set_session_archived(c["id"], archived=True):
            archived += 1

    print(f"Archived {archived} session(s) older than {days} day(s).")
    print("Archived sessions keep all data but are hidden from the default list.")
    return 0


def _cmd_vacuum(session_db: Any, args: argparse.Namespace) -> int:
    import shutil

    # Quick disk-space guard — VACUUM rewrites the entire DB.
    try:
        db_row = session_db._conn.execute("PRAGMA database_list").fetchone()
        db_file = db_row[1] if db_row else None
        if db_file:
            free_gb = shutil.disk_usage(db_file).free / (1024**3)
            if free_gb < 1.0:
                print(
                    f"Error: less than {free_gb:.1f} GB free on disk. "
                    f"VACUUM needs room for a full rewrite."
                )
                return 1
    except Exception:
        pass  # best-effort check

    print("Optimizing FTS indexes...")
    optimized = session_db.optimize_fts()
    print(f"  Optimized {optimized} FTS index(es).")
    print("Running VACUUM to reclaim disk space...")
    session_db.vacuum()
    print("  Done.")
    return 0


def _cmd_stats(session_db: Any, args: argparse.Namespace) -> int:
    import datetime

    db_path = session_db._conn.execute("PRAGMA database_list").fetchone()
    db_file = db_path[1] if db_path else None
    db_size_mb = (
        Path(db_file).stat().st_size / (1024 * 1024) if db_file else 0
    )

    with session_db._lock:
        total = session_db._conn.execute(
            "SELECT COUNT(*) FROM sessions"
        ).fetchone()[0]
        active = session_db._conn.execute(
            "SELECT COUNT(*) FROM sessions WHERE ended_at IS NULL"
        ).fetchone()[0]
        ended = session_db._conn.execute(
            "SELECT COUNT(*) FROM sessions WHERE ended_at IS NOT NULL"
        ).fetchone()[0]
        archived = session_db._conn.execute(
            "SELECT COUNT(*) FROM sessions WHERE archived = 1"
        ).fetchone()[0]
        messages = session_db._conn.execute(
            "SELECT COUNT(*) FROM messages"
        ).fetchone()[0]
        source_rows = session_db._conn.execute(
            "SELECT source, COUNT(*) FROM sessions "
            "GROUP BY source ORDER BY COUNT(*) DESC"
        ).fetchall()
        oldest_raw = session_db._conn.execute(
            "SELECT MIN(started_at) FROM sessions"
        ).fetchone()[0]

    print("Session Database Overview")
    print("=" * 50)
    print(f"  Database size:    {db_size_mb:.1f} MB")
    print(f"  Total sessions:   {total}")
    print(f"    Active:         {active}")
    print(f"    Ended:          {ended}")
    print(f"    Archived:       {archived}")
    print(f"  Total messages:   {messages}")

    if source_rows:
        print(f"\n  By source:")
        for r in source_rows:
            print(f"    {(r[0] or '(none)'):<20} {r[1]:>6}")

    if oldest_raw:
        oldest_dt = datetime.datetime.fromtimestamp(oldest_raw)
        print(f"\n  Oldest session:   {oldest_dt.strftime('%Y-%m-%d %H:%M')}")

    if db_size_mb > 100:
        print(f"\n  \u26a0  Over 100 MB. To reclaim space:")
        print(f"      hermes sessions prune --days 90")
        print(f"      hermes sessions vacuum")

    return 0


def _count_prunable(
    session_db: Any, days: int, source: str | None = None
) -> int:
    import time

    cutoff = time.time() - (days * 86400)
    with session_db._lock:
        if source:
            row = session_db._conn.execute(
                "SELECT COUNT(*) FROM sessions "
                "WHERE started_at < ? AND ended_at IS NOT NULL AND source = ?",
                (cutoff, source),
            ).fetchone()
        else:
            row = session_db._conn.execute(
                "SELECT COUNT(*) FROM sessions "
                "WHERE started_at < ? AND ended_at IS NOT NULL",
                (cutoff,),
            ).fetchone()
    return row[0] if row else 0