"""Migrate a Kanban SQLite DB for the cross-session ownership ledger (#47 Phase 2).

The ledger (``cross_session_ownership.py``) stores its two tables —
``task_ownership`` and ``task_ownership_events`` — inside the Kanban DB, the
same file the board already uses. This tool is the one-shot, reversible
migration for that schema::

    python ownership_migrate.py status    --db <kanban.db>
    python ownership_migrate.py apply     --db <kanban.db> [--dry-run]
    python ownership_migrate.py backfill  --db <kanban.db> [--dry-run]
    python ownership_migrate.py rollback  --db <kanban.db> [--full] [--backup <file>]

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
* ``backfill`` claims the canonical scope of every card already ``running`` at
  activation, so enabling the gate cannot let a second session double-dispatch
  work a live worker already holds. The owner session is taken from the card's
  newest ``owner_admitted`` marker for the current run, then from
  ``tasks.session_id``; a running card with neither is SKIPPED (never claimed
  by a fabricated session). An existing ACTIVE foreign owner is reported as a
  conflict and never stolen. The sweep is ONE transaction; ``--dry-run``
  performs the identical decisions and rolls it back. Backfilled claims carry
  the same 900 s lease as a fresh admission.
* ``rollback`` is **ledger-only** by default (C5b): it drops the two ledger
  tables and their indexes and touches nothing else, so writes made after
  ``apply`` (new cards, comments, events, runs) survive. ``--full`` is the
  explicit emergency path that restores the whole DB from the snapshot and
  therefore DISCARDS every post-apply write.
* Every command prints one JSON object on stdout. ``ok`` is false only for an
  abort; the exit code is 2 for an abort, 0 otherwise.

Activation order (C5a/C5c): ``apply`` -> ``backfill`` -> verify -> export
``HERMES_KANBAN_OWNERSHIP_ENFORCE=1``. The gate itself defaults OFF.

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
import time
from datetime import datetime, timezone, UTC
from pathlib import Path
from typing import Any, Optional

LEDGER_TABLES = ("task_ownership", "task_ownership_events")
BACKUP_SUFFIX = ".ownership-backup-"
EXIT_ABORT = 2

#: C5c: a backfilled claim uses the SAME 900 s lease as a fresh admission, so a
#: card that was already ``running`` when the gate was activated is fenced with
#: exactly the semantics the gate would have given it. A longer lease would pin
#: a crashed worker's card against takeover for that whole window.
BACKFILL_TTL_SECONDS = 900

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


# --- backfill (C5c) ----------------------------------------------------------

def _backfill_marker_session(conn: sqlite3.Connection, task_id: str,
                             run_id: Optional[int]) -> Optional[str]:
    """Session recorded by the newest ``owner_admitted`` marker for THIS run."""
    if run_id is None:
        row = conn.execute(
            "SELECT payload FROM task_events WHERE task_id=? AND kind='owner_admitted'"
            " ORDER BY id DESC LIMIT 1", (task_id,)).fetchone()
    else:
        row = conn.execute(
            "SELECT payload FROM task_events WHERE task_id=? AND kind='owner_admitted'"
            " AND run_id=? ORDER BY id DESC LIMIT 1", (task_id, run_id)).fetchone()
    if row is None or not row[0]:
        return None
    try:
        payload = json.loads(row[0])
    except (ValueError, TypeError):
        return None
    session_id = str(payload.get("session_id") or "").strip()
    return session_id or None


def _backfill_owner(conn: sqlite3.Connection,
                    card: dict) -> tuple[Optional[str], Optional[str]]:
    """``(session_id, source)`` for a running card, else ``(None, reason)``.

    C5c: the owner is the session the card's newest ``owner_admitted`` marker
    recorded for the CURRENT run, else the card's own ``tasks.session_id``.
    A running card with neither is skipped — the tool never fabricates a
    session to fence a card with.
    """
    session_id = _backfill_marker_session(conn, card["id"], card["current_run_id"])
    if session_id:
        return session_id, "owner_admitted"
    session_id = str(card.get("session_id") or "").strip()
    if session_id:
        return session_id, "tasks.session_id"
    return None, "no_owner_session"


def cmd_backfill(db_path: Path, *, dry_run: bool = False,
                 ttl_seconds: int = BACKFILL_TTL_SECONDS) -> dict[str, Any]:
    """Claim the canonical scope of every card already ``running`` (C5c).

    A card that was ``running`` when the gate was activated never went through
    the fresh-admission gate, so the ledger has no row for it and a second
    session could double-dispatch its work. This sweep fences every such card:

    * the scope is derived from the card's OWN identity fields
      (``canonical_key_from_kanban``); an underivable scope is SKIPPED;
    * the owner session is the card's newest ``owner_admitted`` marker for the
      current run, else ``tasks.session_id``; a card with neither is SKIPPED
      (never claimed by a fabricated session);
    * an existing ACTIVE foreign owner is a CONFLICT, never stolen — the sweep
      fences the card for its recorded owner, it does not reassign it.

    The whole sweep is one ``BEGIN IMMEDIATE`` transaction: either every card
    is fenced or none is. ``dry_run`` performs the identical decisions and
    rolls the transaction back, so it writes nothing.

    No ``verify_session`` hook is available here (the migration tool has no
    owner-DB handle), so the owner session is format-validated only — the same
    trust level the ledger itself applies to a claim's identity.
    """
    if not db_path.exists():
        raise Abort(f"database not found: {db_path}")
    import cross_session_ownership as cso
    conn = _connect(db_path)
    try:
        _require_wal(conn)
        conn.row_factory = sqlite3.Row
        present = [t for t in LEDGER_TABLES if conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (t,)).fetchone()]
        if len(present) != len(LEDGER_TABLES):
            raise Abort("ledger tables missing — run `apply` before `backfill`")
        problems = _verify_schema(conn, present)
        if problems:
            raise Abort("incompatible ledger schema: " + "; ".join(problems))

        cards = [{k: r[k] for k in r.keys()}
                 for r in conn.execute("SELECT * FROM tasks WHERE status='running' ORDER BY id")]
        claimed: list[dict[str, Any]] = []
        skipped: list[dict[str, Any]] = []
        conflicts: list[dict[str, Any]] = []

        conn.execute("BEGIN IMMEDIATE")
        try:
            for card in cards:
                task_id = str(card.get("id") or "")
                scope_error = None
                try:
                    scope = cso.canonical_key_from_kanban(card)
                except Exception as exc:
                    scope, scope_error = None, str(exc)[:200]
                if not scope:
                    skipped.append({
                        "task_id": task_id, "scope": None, "owner_session": None,
                        "reason": "no_identity" if scope_error is None
                                  else f"scope_error:{scope_error}",
                    })
                    continue
                owner, source = _backfill_owner(conn, card)
                if owner is None:
                    skipped.append({"task_id": task_id, "scope": scope,
                                    "owner_session": None, "reason": source})
                    continue
                try:
                    cso.validate_session_id(owner)
                except cso.OwnershipError:
                    skipped.append({"task_id": task_id, "scope": scope,
                                    "owner_session": None,
                                    "reason": "invalid_owner_session"})
                    continue
                entry = {"task_id": task_id, "scope": scope,
                         "owner_session": owner, "owner_source": source}
                res = cso.claim_ownership(
                    conn, scope, owner, ttl_seconds=ttl_seconds, join_txn=True,
                    checkpoint={"backfill": True, "task_id": task_id,
                                "source": "ownership_migrate"},
                )
                entry["reason"] = res.get("reason")
                if res.get("ok"):
                    claimed.append(entry)
                else:
                    conflicts.append(entry)
        except Exception as exc:
            conn.execute("ROLLBACK")
            raise Abort(f"backfill failed: {exc}") from exc
        except BaseException:
            conn.execute("ROLLBACK")
            raise
        conn.execute("ROLLBACK" if dry_run else "COMMIT")

        return {
            "db": str(db_path),
            "dry_run": dry_run,
            "ttl_seconds": ttl_seconds,
            "scanned": len(cards),
            "claimed": claimed,
            "skipped": skipped,
            "conflicts": conflicts,
            "ledger_rowcounts": _rowcounts(conn, list(LEDGER_TABLES)),
        }
    finally:
        conn.close()


def cmd_rollback(db_path: Path, *, backup: Optional[Path] = None,
                 full: bool = False) -> dict[str, Any]:
    """Revert the ownership ledger, leaving every other Kanban table untouched.

    Default (ledger-only, C5b): drop ``task_ownership`` and
    ``task_ownership_events`` — ``DROP TABLE`` removes their indexes with them —
    and change nothing else, so writes made after ``apply`` (new cards,
    comments, events, runs) survive. No backup is needed or read.

    ``full=True`` is the explicit emergency path: it restores the whole DB from
    the snapshot (``backup`` or the newest ``*.ownership-backup-*.bak``) and
    therefore DISCARDS every post-apply write.
    """
    if not db_path.exists():
        raise Abort(f"database not found: {db_path}")
    conn = _connect(db_path)
    try:
        _require_wal(conn)
        before = _snapshot(conn)

        if not full:
            for table in LEDGER_TABLES:
                conn.execute(f"DROP TABLE IF EXISTS {table}")

            after = _snapshot(conn)
            if after["integrity"] != ["ok"]:
                raise Abort(f"integrity_check failed after rollback: {after['integrity']}")
            if any(t in after["tables"] for t in LEDGER_TABLES):
                raise Abort("ledger tables still present after rollback")
            removed = set(before["tables"]) - set(after["tables"])
            if not removed <= set(LEDGER_TABLES):
                raise Abort(
                    "non-ledger tables disappeared during rollback: "
                    f"{sorted(removed - set(LEDGER_TABLES))}"
                )
            non_ledger_before = {t: c for t, c in before["rowcounts"].items()
                                 if t not in LEDGER_TABLES}
            non_ledger_after = {t: c for t, c in after["rowcounts"].items()
                                if t not in LEDGER_TABLES}
            if non_ledger_after != non_ledger_before:
                raise Abort(
                    "non-ledger rowcounts changed during rollback: "
                    f"{non_ledger_before} -> {non_ledger_after}"
                )
            return {
                "db": str(db_path),
                "mode": "ledger",
                "backup": None,
                "dropped": [t for t in LEDGER_TABLES if t in before["tables"]],
                "before": before,
                "after": after,
                "non_ledger_rowcounts_unchanged": True,
                "integrity": after["integrity"],
            }

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
            "mode": "full",
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
    p_backfill = sub.add_parser(
        "backfill", help="claim every running card's scope (C5c activation sweep)")
    p_backfill.add_argument("--dry-run", action="store_true",
                            help="report what would be claimed; write nothing")
    p_roll = sub.add_parser("rollback", help="drop the 2 ledger tables (ledger-only); --full restores the backup")
    p_roll.add_argument("--full", action="store_true",
                        help="emergency: restore the whole DB from the backup, discarding post-apply writes")
    p_roll.add_argument("--backup", type=Path, default=None,
                        help="explicit backup file for --full (default: newest *.ownership-backup-*.bak)")
    args = parser.parse_args(argv)

    db = _resolve_db(args.db, args.board)
    try:
        if args.command == "status":
            result = cmd_status(db)
        elif args.command == "apply":
            result = cmd_apply(db, dry_run=args.dry_run)
        elif args.command == "backfill":
            result = cmd_backfill(db, dry_run=args.dry_run)
        elif args.command == "rollback":
            result = cmd_rollback(db, backup=args.backup, full=args.full)
        else:  # pragma: no cover - argparse enforces the choices
            raise Abort(f"unknown command: {args.command}")
    except Abort as exc:
        print(json.dumps({"ok": False, "abort": str(exc), "db": str(db)}))
        return EXIT_ABORT
    print(json.dumps({"ok": True, **result}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
