"""Migrate a Kanban SQLite DB for the cross-session ownership ledger (#47 Phase 2).

The ledger (``cross_session_ownership.py``) stores its two tables —
``task_ownership`` and ``task_ownership_events`` — inside the Kanban DB, the
same file the board already uses. This tool is the one-shot, reversible
migration for that schema::

    python ownership_migrate.py status    --db <kanban.db>
    python ownership_migrate.py apply     --db <kanban.db> [--dry-run]
    python ownership_migrate.py rollback  --db <kanban.db> [--backup <file>]

Safety contract
---------------
* **ABORT (exit 2)** unless ``PRAGMA journal_mode`` is ``wal``. A rollback-journal
  DB cannot give the atomic snapshot / restore this tool promises, so the tool
  refuses instead of guessing.
* ``apply`` takes a consistent snapshot with the SQLite backup API *before*
  touching the schema, creates the two tables only when absent, then verifies
  rowcounts and ``PRAGMA integrity_check``. Re-running is idempotent: when both
  tables already exist with the expected shape it reports ``already_applied``
  and writes nothing (no second backup).
* ``rollback`` restores the newest snapshot (or ``--backup``), then drops the
  two tables, then verifies the DB matches the snapshot.
* Every command prints one JSON object on stdout. ``ok`` is false only for an
  abort; the exit code is 2 for an abort, 0 otherwise.

The default DB path follows the Kanban resolution: ``--db`` >
``HERMES_KANBAN_DB`` > ``HERMES_KANBAN_HOME/kanban.db`` >
``<hermes-root>/kanban.db``.
"""
from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
from datetime import datetime, timezone, UTC
from pathlib import Path
from typing import Any, Optional

LEDGER_TABLES = ("task_ownership", "task_ownership_events")
BACKUP_SUFFIX = ".ownership-backup-"
EXIT_ABORT = 2

# Expected index -> owning table, used to validate a partially migrated DB.
_INDEX_OWNER = {
    "idx_task_ownership_session": "task_ownership",
    "idx_task_ownership_expiry": "task_ownership",
    "idx_task_ownership_events_scope": "task_ownership_events",
}


class Abort(Exception):
    """Refuse to proceed: the safety contract is not met (exit 2)."""


# --- small SQLite helpers ----------------------------------------------------

def _connect(path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(path), timeout=30, isolation_level=None)
    conn.execute("PRAGMA busy_timeout=30000")
    return conn


def _journal_mode(conn: sqlite3.Connection) -> str:
    return str(conn.execute("PRAGMA journal_mode").fetchone()[0]).lower()


def _integrity(conn: sqlite3.Connection) -> list[str]:
    return [str(r[0]) for r in conn.execute("PRAGMA integrity_check")]


def _user_tables(conn: sqlite3.Connection) -> list[str]:
    return sorted(r[0] for r in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"))


def _rowcounts(conn: sqlite3.Connection, tables: list[str]) -> dict[str, int]:
    return {t: int(conn.execute(f'SELECT COUNT(*) FROM "{t}"').fetchone()[0]) for t in tables}


def _table_columns(conn: sqlite3.Connection, table: str) -> list[tuple[str, str]]:
    return [(str(r[1]), str(r[2]).upper())
            for r in conn.execute(f"PRAGMA table_info({table})")]


def _indexes(conn: sqlite3.Connection) -> list[str]:
    return sorted(r[0] for r in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='index' AND name LIKE 'idx_task_ownership%'"))


def _snapshot(conn: sqlite3.Connection) -> dict[str, Any]:
    tables = _user_tables(conn)
    return {
        "journal_mode": _journal_mode(conn),
        "tables": tables,
        "ledger_tables_present": [t for t in LEDGER_TABLES if t in tables],
        "rowcounts": _rowcounts(conn, tables),
        "integrity": _integrity(conn),
    }


def _expected_schema() -> tuple[dict[str, list[tuple[str, str]]], list[str]]:
    """Column shape + index set the ledger DDL produces, from the real module.

    Derived by building the schema in memory with ``init_schema`` so the tool
    can never drift from the implementation it migrates.
    """
    import cross_session_ownership as cso
    mem = sqlite3.connect(":memory:")
    try:
        cso.init_schema(mem)
        columns = {t: _table_columns(mem, t) for t in LEDGER_TABLES}
        indexes = _indexes(mem)
    finally:
        mem.close()
    return columns, indexes


def _verify_schema(conn: sqlite3.Connection, present: list[str]) -> list[str]:
    """Return a list of schema problems for the ledger tables that exist."""
    expected_columns, expected_indexes = _expected_schema()
    problems: list[str] = []
    for table in present:
        actual = _table_columns(conn, table)
        if actual != expected_columns[table]:
            problems.append(
                f"{table}: column shape mismatch (expected {expected_columns[table]}, got {actual})"
            )
    actual_indexes = set(_indexes(conn))
    for index in expected_indexes:
        if _INDEX_OWNER[index] in present and index not in actual_indexes:
            problems.append(f"missing index {index}")
    return problems


def _require_wal(conn: sqlite3.Connection) -> str:
    mode = _journal_mode(conn)
    if mode != "wal":
        raise Abort(
            f"journal_mode is {mode!r}, not 'wal' — refusing to migrate a non-WAL DB"
        )
    return mode


def _backup_path(db_path: Path) -> Path:
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return db_path.with_name(f"{db_path.name}{BACKUP_SUFFIX}{stamp}-{os.getpid()}.bak")


def _newest_backup(db_path: Path) -> Optional[Path]:
    candidates = sorted(db_path.parent.glob(f"{db_path.name}{BACKUP_SUFFIX}*.bak"),
                        key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0] if candidates else None


def _backup_to(src_conn: sqlite3.Connection, dst_path: Path) -> None:
    """Consistent snapshot via the SQLite backup API (WAL-safe)."""
    dst = sqlite3.connect(str(dst_path))
    try:
        src_conn.backup(dst)
    finally:
        dst.close()


def _restore_from(backup_path: Path, db_path: Path) -> None:
    """Replace ``db_path`` with ``backup_path`` content.

    The stale ``-wal``/``-shm`` sidecars are removed first: they belong to the
    pre-restore DB and would otherwise replay old frames over the restored file.
    """
    for suffix in ("-wal", "-shm"):
        Path(str(db_path) + suffix).unlink(missing_ok=True)
    dst = sqlite3.connect(str(db_path))
    try:
        src = sqlite3.connect(str(backup_path))
        try:
            src.backup(dst)
        finally:
            src.close()
    finally:
        dst.close()


# --- commands ---------------------------------------------------------------

def cmd_status(db_path: Path) -> dict[str, Any]:
    """Read-only report. Never aborts on a non-WAL DB — reporting it is the point."""
    if not db_path.exists():
        raise Abort(f"database not found: {db_path}")
    conn = _connect(db_path)
    try:
        snap = _snapshot(conn)
        present = snap["ledger_tables_present"]
        problems = _verify_schema(conn, present) if present else []
        return {
            "db": str(db_path),
            "journal_mode": snap["journal_mode"],
            "wal": snap["journal_mode"] == "wal",
            "ledger_tables_present": present,
            "ledger_rowcounts": {t: snap["rowcounts"].get(t) for t in LEDGER_TABLES},
            "schema_ok": (not problems) if present else None,
            "schema_problems": problems,
            "integrity": snap["integrity"],
            "backup_count": len(list(db_path.parent.glob(
                f"{db_path.name}{BACKUP_SUFFIX}*.bak"))),
        }
    finally:
        conn.close()


def cmd_apply(db_path: Path, *, dry_run: bool = False) -> dict[str, Any]:
    """Backup -> create the two ledger tables if absent -> verify. Idempotent."""
    if not db_path.exists():
        raise Abort(f"database not found: {db_path}")
    conn = _connect(db_path)
    try:
        _require_wal(conn)
        before = _snapshot(conn)
        present = before["ledger_tables_present"]
        if present:
            problems = _verify_schema(conn, present)
            if problems:
                raise Abort("incompatible ledger schema: " + "; ".join(problems))

        if dry_run:
            return {
                "db": str(db_path),
                "dry_run": True,
                "already_applied": len(present) == len(LEDGER_TABLES),
                "would_create": [t for t in LEDGER_TABLES if t not in present],
                "would_backup_to": str(_backup_path(db_path)),
                "before": before,
            }

        if len(present) == len(LEDGER_TABLES):
            # Idempotent re-run: nothing to change, so no second backup either.
            return {
                "db": str(db_path),
                "dry_run": False,
                "already_applied": True,
                "created": [],
                "backup": None,
                "before": before,
                "after": before,
                "rowcounts_unchanged": True,
                "integrity": before["integrity"],
            }

        bpath = _backup_path(db_path)
        _backup_to(conn, bpath)
        bconn = _connect(bpath)
        try:
            if _integrity(bconn) != ["ok"]:
                raise Abort(f"backup integrity_check failed: {bpath}")
            backup_snapshot = _snapshot(bconn)
        finally:
            bconn.close()

        import cross_session_ownership as cso
        cso.init_schema(conn)  # CREATE ... IF NOT EXISTS; idempotent

        after = _snapshot(conn)
        if after["integrity"] != ["ok"]:
            raise Abort(f"integrity_check failed after apply: {after['integrity']}")
        rowcounts_unchanged = all(
            after["rowcounts"].get(t) == count for t, count in before["rowcounts"].items()
        )
        if not rowcounts_unchanged:
            raise Abort(
                "pre-existing rowcounts changed during apply: "
                f"{before['rowcounts']} -> {after['rowcounts']}"
            )
        return {
            "db": str(db_path),
            "dry_run": False,
            "already_applied": False,
            "created": [t for t in LEDGER_TABLES if t not in present],
            "backup": str(bpath),
            "backup_rowcounts": backup_snapshot["rowcounts"],
            "before": before,
            "after": after,
            "rowcounts_unchanged": True,
            "integrity": after["integrity"],
        }
    finally:
        conn.close()


def cmd_rollback(db_path: Path, *, backup: Optional[Path] = None) -> dict[str, Any]:
    """Restore the snapshot, then drop the two ledger tables, then verify."""
    if not db_path.exists():
        raise Abort(f"database not found: {db_path}")
    conn = _connect(db_path)
    try:
        _require_wal(conn)
        bpath = Path(backup) if backup is not None else _newest_backup(db_path)
        if bpath is None or not bpath.exists():
            raise Abort(f"no backup found for {db_path} (pass --backup <file>)")

        bconn = _connect(bpath)
        try:
            if _integrity(bconn) != ["ok"]:
                raise Abort(f"backup integrity_check failed: {bpath}")
            backup_snapshot = _snapshot(bconn)
        finally:
            bconn.close()

        before = _snapshot(conn)
        conn.close()
        _restore_from(bpath, db_path)
        conn = _connect(db_path)

        for table in LEDGER_TABLES:
            conn.execute(f"DROP TABLE IF EXISTS {table}")

        after = _snapshot(conn)
        if after["integrity"] != ["ok"]:
            raise Abort(f"integrity_check failed after rollback: {after['integrity']}")
        if any(t in after["tables"] for t in LEDGER_TABLES):
            raise Abort("ledger tables still present after rollback")
        matches_backup = (
            after["tables"] == backup_snapshot["tables"]
            and after["rowcounts"] == backup_snapshot["rowcounts"]
        )
        if not matches_backup:
            raise Abort(
                "post-rollback DB does not match the backup: "
                f"tables {after['tables']} vs {backup_snapshot['tables']}; "
                f"rowcounts {after['rowcounts']} vs {backup_snapshot['rowcounts']}"
            )
        return {
            "db": str(db_path),
            "backup": str(bpath),
            "restored_tables": backup_snapshot["tables"],
            "dropped": list(LEDGER_TABLES),
            "before": before,
            "after": after,
            "matches_backup": True,
            "integrity": after["integrity"],
        }
    finally:
        try:
            conn.close()
        except Exception:
            pass


# --- CLI --------------------------------------------------------------------

def _resolve_db(explicit: Optional[Path], board: Optional[str]) -> Path:
    if explicit is not None:
        return Path(explicit).expanduser()
    pin = os.environ.get("HERMES_KANBAN_DB", "").strip()
    if pin:
        return Path(pin).expanduser()
    home = os.environ.get("HERMES_KANBAN_HOME", "").strip()
    if home:
        root = Path(home).expanduser()
    else:
        from hermes_constants import get_default_hermes_root
        root = Path(get_default_hermes_root())
    if board and board != "default":
        return root / "kanban" / "boards" / board / "kanban.db"
    return root / "kanban.db"


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Migrate a Kanban DB for the cross-session ownership ledger (#47 Phase 2)."
    )
    parser.add_argument("--db", type=Path, default=None,
                        help="Kanban DB path (default: HERMES_KANBAN_DB / HERMES_KANBAN_HOME / <root>/kanban.db)")
    parser.add_argument("--board", default=None,
                        help="Board slug instead of --db (default board: <root>/kanban.db)")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("status", help="report journal_mode, ledger tables, rowcounts, integrity")
    p_apply = sub.add_parser("apply", help="backup -> create the 2 ledger tables -> verify")
    p_apply.add_argument("--dry-run", action="store_true",
                         help="report what would change; write nothing")
    p_roll = sub.add_parser("rollback", help="restore the backup -> drop the 2 ledger tables")
    p_roll.add_argument("--backup", type=Path, default=None,
                        help="explicit backup file (default: newest *.ownership-backup-*.bak)")
    args = parser.parse_args(argv)

    db = _resolve_db(args.db, args.board)
    try:
        if args.command == "status":
            result = cmd_status(db)
        elif args.command == "apply":
            result = cmd_apply(db, dry_run=args.dry_run)
        elif args.command == "rollback":
            result = cmd_rollback(db, backup=args.backup)
        else:  # pragma: no cover - argparse enforces the choices
            raise Abort(f"unknown command: {args.command}")
    except Abort as exc:
        print(json.dumps({"ok": False, "abort": str(exc), "db": str(db)}))
        return EXIT_ABORT
    print(json.dumps({"ok": True, **result}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
