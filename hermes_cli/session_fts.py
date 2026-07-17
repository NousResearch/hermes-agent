"""Guarded operator workflow for external-content FTS migration."""

from __future__ import annotations

import os
import shutil
import sqlite3
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from hermes_state import (
    DEFAULT_DB_PATH,
    FTS_STORAGE_REVISION_EXTERNAL_V1,
    FTS_STORAGE_REVISION_KEY,
    SessionDB,
)

_GIB = 1024**3
_MIN_RESERVE_BYTES = 2 * _GIB


class SessionFtsMaintenanceError(RuntimeError):
    """Raised when an external-content FTS maintenance guard fails."""


def configure_migrate_fts_parser(parser) -> None:
    """Add the guarded migration flags to an argparse parser."""
    parser.add_argument(
        "--check-only",
        action="store_true",
        help=(
            "Report mode and space without logical DB changes or a backup "
            "(SQLite may maintain a WAL -shm sidecar)"
        ),
    )
    parser.add_argument(
        "--writers-stopped",
        action="store_true",
        help="Attest that gateway and all other state.db writers are stopped",
    )
    parser.add_argument(
        "--unsafe-no-backup",
        action="store_true",
        help="Skip the verified online backup (unsafe; not recommended)",
    )
    parser.add_argument(
        "--yes",
        "-y",
        action="store_true",
        help="Skip the final interactive confirmation",
    )


def _format_bytes(value: int) -> str:
    amount = float(value)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if amount < 1024 or unit == "TiB":
            return f"{amount:.2f} {unit}"
        amount /= 1024
    return f"{amount:.2f} TiB"


def _read_database_layout(db_path: Path) -> dict[str, Any]:
    if not db_path.exists():
        raise SessionFtsMaintenanceError(f"session database does not exist: {db_path}")

    conn = sqlite3.connect(f"{db_path.as_uri()}?mode=ro", uri=True)
    try:
        quick_check = str(conn.execute("PRAGMA quick_check").fetchone()[0])
        if quick_check != "ok":
            raise SessionFtsMaintenanceError(
                f"database quick_check failed before migration: {quick_check}"
            )
        page_count = int(conn.execute("PRAGMA page_count").fetchone()[0])
        page_size = int(conn.execute("PRAGMA page_size").fetchone()[0])
        rows = conn.execute(
            "SELECT name, sql FROM sqlite_master "
            "WHERE type = 'table' AND name IN (?, ?) ORDER BY name",
            ("messages_fts", "messages_fts_trigram"),
        ).fetchall()
        source_row = conn.execute(
            "SELECT type, sql FROM sqlite_master "
            "WHERE name = ? ORDER BY type LIMIT 1",
            ("messages_fts_source",),
        ).fetchone()
        try:
            revision_row = conn.execute(
                "SELECT value FROM state_meta WHERE key = ?",
                (FTS_STORAGE_REVISION_KEY,),
            ).fetchone()
        except sqlite3.OperationalError:
            revision_row = None
    finally:
        conn.close()

    table_modes = {}
    for name, sql in rows:
        table_name = str(name)
        table_modes[table_name] = SessionDB._classify_fts_table_sql(
            table_name,
            str(sql) if sql is not None else None,
        )
    source_mode = SessionDB._classify_fts_source_sql(
        str(source_row[0]) if source_row is not None else None,
        str(source_row[1])
        if source_row is not None and source_row[1] is not None
        else None,
    )

    expected_tables = {"messages_fts", "messages_fts_trigram"}
    external_present = "external" in table_modes.values()
    if source_mode == "unsupported" or (
        external_present and source_mode != "canonical"
    ):
        mode = "unsupported"
    elif not table_modes:
        mode = "missing"
    elif "unsupported" in table_modes.values():
        mode = "unsupported"
    elif set(table_modes) == expected_tables:
        values = set(table_modes.values())
        mode = values.pop() if len(values) == 1 else "mixed"
    elif set(table_modes) == {"messages_fts"}:
        mode = f"base-only-{table_modes['messages_fts']}"
    else:
        mode = "partial"

    logical_bytes = page_count * page_size
    database_bytes = max(db_path.stat().st_size, logical_bytes)
    wal_path = db_path.with_name(f"{db_path.name}-wal")
    return {
        "storage_mode": mode,
        "source_view_mode": source_mode,
        "storage_revision": str(revision_row[0]) if revision_row else None,
        "database_bytes": database_bytes,
        "wal_bytes": wal_path.stat().st_size if wal_path.exists() else 0,
        "quick_check": quick_check,
    }


def _runtime_supports_trigram() -> bool:
    conn = sqlite3.connect(":memory:")
    try:
        conn.execute(
            "CREATE VIRTUAL TABLE temp._hermes_trigram_probe "
            "USING fts5(x, tokenize='trigram')"
        )
        return True
    except sqlite3.OperationalError as exc:
        message = str(exc).lower()
        if "no such tokenizer: trigram" in message or (
            "no such module" in message and "fts5" in message
        ):
            return False
        raise
    finally:
        conn.close()


def _sqlite_temp_path() -> Path:
    configured = os.environ.get("SQLITE_TMPDIR")
    if configured:
        candidate = Path(configured).expanduser()
        if candidate.is_dir() and os.access(candidate, os.W_OK | os.X_OK):
            return candidate.resolve()
    return Path(tempfile.gettempdir()).resolve()


def _space_plan(
    db_path: Path,
    *,
    database_bytes: int,
    backup_enabled: bool,
    disk_usage_fn: Callable[[Path], Any],
) -> dict[str, Any]:
    # A whole-database staging estimate is deliberately conservative relative
    # to observed benchmarks and much faster to determine than scanning
    # multi-GiB dbstat output. It is a heuristic, not a proven upper bound;
    # repeated WAL frames and SQLite temp files can require additional space.
    staging_bytes = database_bytes
    backup_bytes = database_bytes if backup_enabled else 0
    reserve_bytes = max(_MIN_RESERVE_BYTES, (database_bytes + 3) // 4)
    required_free_bytes = backup_bytes + staging_bytes + reserve_bytes
    free_bytes = int(disk_usage_fn(db_path.parent).free)
    temp_path = _sqlite_temp_path()
    separate_temp_filesystem = (
        os.stat(db_path.parent).st_dev != os.stat(temp_path).st_dev
    )
    temp_required_bytes = staging_bytes + reserve_bytes if separate_temp_filesystem else 0
    temp_free_bytes = (
        int(disk_usage_fn(temp_path).free) if separate_temp_filesystem else free_bytes
    )
    temp_space_ok = temp_free_bytes >= temp_required_bytes
    return {
        "backup_bytes": backup_bytes,
        "staging_bytes": staging_bytes,
        "reserve_bytes": reserve_bytes,
        "required_free_bytes": required_free_bytes,
        "free_bytes": free_bytes,
        "space_ok": free_bytes >= required_free_bytes and temp_space_ok,
        "temp_path": str(temp_path),
        "separate_temp_filesystem": separate_temp_filesystem,
        "temp_required_bytes": temp_required_bytes,
        "temp_free_bytes": temp_free_bytes,
        "temp_space_ok": temp_space_ok,
    }


def _backup_path(db_path: Path) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S-%fZ")
    return db_path.with_name(f"{db_path.name}.pre-fts-external-{stamp}.bak")


def _create_verified_backup(db_path: Path, destination: Path) -> None:
    try:
        destination_fd = os.open(
            destination,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            0o600,
        )
    except FileExistsError as exc:
        raise SessionFtsMaintenanceError(
            f"refusing to overwrite existing backup: {destination}"
        ) from exc
    else:
        os.close(destination_fd)

    source = None
    backup = None
    try:
        source = sqlite3.connect(f"{db_path.as_uri()}?mode=ro", uri=True)
        backup = sqlite3.connect(str(destination))
        if os.name != "nt":
            os.chmod(destination, 0o600)
        source.backup(backup)
        result = str(backup.execute("PRAGMA quick_check").fetchone()[0])
        if result != "ok":
            raise SessionFtsMaintenanceError(
                f"backup quick_check failed: {result}"
            )
        backup.close()
        backup = None
        if os.name != "nt":
            os.chmod(destination, 0o600)
        fd = os.open(destination, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
        if os.name != "nt":
            directory_fd = os.open(destination.parent, os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
    except BaseException:
        try:
            if backup is not None:
                backup.close()
        except Exception:
            pass
        try:
            destination.unlink(missing_ok=True)
        except OSError:
            pass
        raise
    finally:
        if source is not None:
            try:
                source.close()
            except Exception:
                pass


def _print_plan(db_path: Path, layout: dict[str, Any], plan: dict[str, Any]) -> None:
    print(f"Session database: {db_path}")
    print(f"FTS storage mode: {layout['storage_mode']}")
    print(f"FTS storage revision: {layout['storage_revision'] or '(unset)'}")
    print(f"Runtime trigram support: {'yes' if layout['trigram_available'] else 'no'}")
    print(f"Database size allowance: {_format_bytes(layout['database_bytes'])}")
    print(f"Current WAL size: {_format_bytes(layout['wal_bytes'])}")
    print(f"Backup allowance: {_format_bytes(plan['backup_bytes'])}")
    print(f"Staging estimate: {_format_bytes(plan['staging_bytes'])}")
    print(f"Safety reserve: {_format_bytes(plan['reserve_bytes'])}")
    print(f"Required free space: {_format_bytes(plan['required_free_bytes'])}")
    print(f"Available free space: {_format_bytes(plan['free_bytes'])}")
    if plan["separate_temp_filesystem"]:
        print(f"SQLite temp filesystem: {plan['temp_path']}")
        print(
            "Temp-space estimate: "
            f"{_format_bytes(plan['temp_required_bytes'])} required, "
            f"{_format_bytes(plan['temp_free_bytes'])} available"
        )


def run_external_fts_migration(
    args,
    *,
    db_path: Path | None = None,
    input_fn: Callable[[str], str] = input,
    disk_usage_fn: Callable[[Path], Any] = shutil.disk_usage,
) -> dict[str, Any]:
    """Run or preflight the guarded external-content FTS migration."""
    path = Path(db_path or DEFAULT_DB_PATH).expanduser().resolve()
    backup_enabled = not bool(args.unsafe_no_backup)
    try:
        layout = _read_database_layout(path)
        layout["trigram_available"] = _runtime_supports_trigram()
        plan = _space_plan(
            path,
            database_bytes=int(layout["database_bytes"]),
            backup_enabled=backup_enabled,
            disk_usage_fn=disk_usage_fn,
        )
    except SessionFtsMaintenanceError:
        raise
    except Exception as exc:
        raise SessionFtsMaintenanceError(f"FTS preflight failed: {exc}") from exc
    _print_plan(path, layout, plan)

    base_report = {
        **layout,
        **plan,
        "check_only": bool(args.check_only),
        "backup_path": None,
    }
    if args.check_only:
        return base_report

    invalid_mode = layout["storage_mode"] in {"missing", "partial", "unsupported"}
    invalid_base_only = (
        layout["storage_mode"] in {"base-only-inline", "base-only-external"}
        and layout["trigram_available"]
    )
    if invalid_mode or invalid_base_only:
        raise SessionFtsMaintenanceError(
            "both FTS5 tables (messages_fts and messages_fts_trigram) must be "
            "present when this SQLite runtime supports trigram; "
            f"detected {layout['storage_mode']}"
        )
    if os.environ.get("_HERMES_GATEWAY") == "1":
        raise SessionFtsMaintenanceError(
            "run this maintenance command outside the gateway process"
        )
    if not args.writers_stopped:
        raise SessionFtsMaintenanceError(
            "refusing migration without --writers-stopped; stop the gateway and "
            "all other state.db writers first"
        )
    if not plan["temp_space_ok"]:
        raise SessionFtsMaintenanceError(
            "insufficient free space on SQLite temp filesystem "
            f"{plan['temp_path']}: need "
            f"{_format_bytes(plan['temp_required_bytes'])}, have "
            f"{_format_bytes(plan['temp_free_bytes'])}"
        )
    if plan["free_bytes"] < plan["required_free_bytes"]:
        raise SessionFtsMaintenanceError(
            "insufficient free space: "
            f"need {_format_bytes(plan['required_free_bytes'])}, "
            f"have {_format_bytes(plan['free_bytes'])}"
        )

    if not args.yes:
        try:
            answer = input_fn(
                "Create the verified backup and migrate available FTS indexes now? [y/N] "
            )
        except (EOFError, KeyboardInterrupt):
            answer = ""
        if answer.strip().lower() not in {"y", "yes"}:
            print("Migration cancelled; no changes made.")
            return {**base_report, "cancelled": True}

    destination = None
    backup_verified = False
    db = None
    try:
        db = SessionDB.open_for_fts_maintenance(path)
        db.begin_fts_maintenance_transaction()

        # Re-read mode and sizing after acquiring the cross-process writer lock.
        # No writer can commit between this snapshot, the backup, and migration.
        locked_layout = _read_database_layout(path)
        locked_layout["trigram_available"] = layout["trigram_available"]
        locked_invalid_mode = locked_layout["storage_mode"] in {
            "missing",
            "partial",
            "unsupported",
        }
        locked_invalid_base_only = (
            locked_layout["storage_mode"]
            in {"base-only-inline", "base-only-external"}
            and locked_layout["trigram_available"]
        )
        if locked_invalid_mode or locked_invalid_base_only:
            raise SessionFtsMaintenanceError(
                "FTS schema changed before the maintenance lock was acquired; "
                f"detected {locked_layout['storage_mode']}"
            )
        locked_plan = _space_plan(
            path,
            database_bytes=int(locked_layout["database_bytes"]),
            backup_enabled=backup_enabled,
            disk_usage_fn=disk_usage_fn,
        )
        if not locked_plan["temp_space_ok"] or (
            locked_plan["free_bytes"] < locked_plan["required_free_bytes"]
        ):
            raise SessionFtsMaintenanceError(
                "free space fell below the locked maintenance requirement"
            )
        base_report.update(locked_layout)
        base_report.update(locked_plan)

        if backup_enabled:
            destination = _backup_path(path)
            try:
                _create_verified_backup(path, destination)
            except SessionFtsMaintenanceError:
                raise
            except Exception as exc:
                raise SessionFtsMaintenanceError(
                    f"verified backup failed; migration was not attempted: {exc}"
                ) from exc
            backup_verified = True
            print(f"Verified backup: {destination}")

            try:
                remaining = _space_plan(
                    path,
                    database_bytes=int(locked_layout["database_bytes"]),
                    backup_enabled=False,
                    disk_usage_fn=disk_usage_fn,
                )
            except Exception as exc:
                raise SessionFtsMaintenanceError(
                    "post-backup free-space check failed; "
                    f"backup preserved at {destination}: {exc}"
                ) from exc
            if not remaining["space_ok"]:
                raise SessionFtsMaintenanceError(
                    "free space fell below the post-backup migration requirement; "
                    f"backup preserved at {destination}"
                )

        migration = db.migrate_fts_to_external_content()
        final_mode = db.fts_storage_mode()
        conn = db._conn
        if conn is None:
            raise SessionFtsMaintenanceError("session database connection closed unexpectedly")
        quick_check = str(conn.execute("PRAGMA quick_check").fetchone()[0])
        revision_row = conn.execute(
            "SELECT value FROM state_meta WHERE key = ?",
            (FTS_STORAGE_REVISION_KEY,),
        ).fetchone()
        final_revision = str(revision_row[0]) if revision_row else None
        if (
            final_mode != "external"
            or quick_check != "ok"
            or final_revision != FTS_STORAGE_REVISION_EXTERNAL_V1
        ):
            raise SessionFtsMaintenanceError(
                "post-migration validation failed: "
                f"mode={final_mode}, revision={final_revision}, "
                f"quick_check={quick_check}"
            )
    except Exception as exc:
        if isinstance(exc, SessionFtsMaintenanceError) and "backup preserved at" in str(exc):
            raise
        if backup_verified:
            backup_context = f"backup preserved at {destination}"
        elif backup_enabled:
            backup_context = "no verified backup was created"
        else:
            backup_context = "no backup was requested"
        raise SessionFtsMaintenanceError(
            f"FTS migration failed; {backup_context}: {exc}"
        ) from exc
    finally:
        if db is not None:
            conn = db._conn
            if conn is not None and conn.in_transaction:
                conn.rollback()
            db.close()

    print("FTS migration completed and validated.")
    print("VACUUM was not run; disk reclamation remains a separate operation.")
    return {
        **base_report,
        **migration,
        "storage_revision": final_revision,
        "backup_path": str(destination) if destination else None,
        "post_migration_quick_check": quick_check,
    }
