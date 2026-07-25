"""Offline, non-destructive recovery for a damaged Hermes session database.

The recovery path deliberately avoids in-place repair:

* the supplied source database is never opened by SQLite;
* the source file and any WAL/SHM/rollback-journal sidecars are copied into a
  disposable working directory first;
* canonical rows are copied into a newly initialized current-schema database;
* derived FTS tables and migration bookkeeping are rebuilt, not copied; and
* the recovered database is never installed over the active database.
"""

from __future__ import annotations

import json
import os
import shutil
import sqlite3
import tempfile
from pathlib import Path
from typing import Any, Callable, Optional

from hermes_state import (
    FTS_STORAGE_VERSION,
    SCHEMA_VERSION,
    SessionDB,
    _db_opens_cleanly,
)


ProgressCallback = Callable[[dict[str, Any]], None]

_CANONICAL_TABLES = (
    "sessions",
    "messages",
    "session_model_usage",
    "compression_locks",
    "gateway_routing",
    "async_delegations",
)

_TOPIC_TABLES = (
    "telegram_dm_topic_mode",
    "telegram_dm_topic_bindings",
)

# These values describe derived indexes or the schema that owns an optional
# table. A fresh destination must generate them from its own current schema.
_GENERATED_META_KEYS = frozenset({
    "fts_storage_version",
    "fts_optimize_available",
    "fts_rebuild_high_water",
    "fts_rebuild_progress",
    "fts_cjk_stale",
    "fts_cjk_rebuild_high_water",
    "fts_cjk_rebuild_progress",
    "telegram_dm_topic_schema_version",
})

_SIDECAR_SUFFIXES = ("", "-wal", "-shm", "-journal")
_MINIMUM_SPACE_HEADROOM = 256 * 1024 * 1024


class SessionRecoveryError(RuntimeError):
    """Base error for offline session recovery."""


class SessionRecoverySafetyError(SessionRecoveryError):
    """Raised before recovery when a path or overwrite guard fails."""


class SessionRecoverySourceError(SessionRecoveryError):
    """Raised when the source cannot provide the required canonical tables."""


def _sidecar_path(db_path: Path, suffix: str) -> Path:
    return db_path if not suffix else db_path.with_name(db_path.name + suffix)


def _resolved_output_path(path: Path) -> Path:
    """Resolve a not-yet-created output path without requiring it to exist."""

    parent = path.expanduser().parent.resolve(strict=True)
    return parent / path.name


def _validate_paths(
    source_path: Path,
    output_path: Optional[Path] = None,
    work_dir: Optional[Path] = None,
) -> tuple[Path, Optional[Path], Path]:
    source = source_path.expanduser().resolve(strict=True)
    if not source.is_file():
        raise SessionRecoverySafetyError(f"Source is not a file: {source}")

    output: Optional[Path] = None
    if output_path is not None:
        output = _resolved_output_path(output_path)
        protected = {
            _sidecar_path(source, suffix).resolve(strict=False)
            for suffix in _SIDECAR_SUFFIXES
        }
        if output.resolve(strict=False) in protected:
            raise SessionRecoverySafetyError(
                "The recovery output must not be the source database or one of "
                "its journal sidecars."
            )
        for suffix in _SIDECAR_SUFFIXES:
            candidate = _sidecar_path(output, suffix)
            if os.path.lexists(candidate):
                raise SessionRecoverySafetyError(
                    f"Refusing to overwrite existing recovery output: {candidate}"
                )

    work_root = (
        work_dir.expanduser().resolve(strict=True)
        if work_dir is not None
        else (output.parent if output is not None else source.parent)
    )
    if not work_root.is_dir():
        raise SessionRecoverySafetyError(
            f"Recovery work directory is not a directory: {work_root}"
        )
    return source, output, work_root


def _source_fingerprint(source: Path) -> dict[str, dict[str, int]]:
    fingerprint: dict[str, dict[str, int]] = {}
    for suffix in _SIDECAR_SUFFIXES:
        path = _sidecar_path(source, suffix)
        if not path.exists():
            continue
        stat = path.stat()
        fingerprint[suffix or "main"] = {
            "size": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
        }
    return fingerprint


def _format_bytes(value: int) -> str:
    units = ("B", "KiB", "MiB", "GiB", "TiB")
    amount = float(value)
    for unit in units:
        if amount < 1024 or unit == units[-1]:
            return f"{amount:.1f} {unit}"
        amount /= 1024
    return f"{value} B"


def _same_filesystem(left: Path, right: Path) -> bool:
    try:
        return os.stat(left).st_dev == os.stat(right).st_dev
    except OSError:
        # Existing directories were already required by _validate_paths. This
        # fallback is defensive for platforms with incomplete st_dev support.
        return left.anchor.casefold() == right.anchor.casefold()


def _disk_space_preflight(
    source: Path,
    work_root: Path,
    output_parent: Optional[Path],
) -> dict[str, Any]:
    """Require space for the disposable bundle, output, and safety headroom."""

    bundle_bytes = sum(
        _sidecar_path(source, suffix).stat().st_size
        for suffix in _SIDECAR_SUFFIXES
        if _sidecar_path(source, suffix).exists()
    )
    # The v23 external-content rebuild is normally substantially smaller than
    # a legacy database, but using the complete source bundle as the estimate
    # avoids betting the user's disk on that expectation.
    output_allowance = bundle_bytes if output_parent is not None else 0
    headroom = max(
        _MINIMUM_SPACE_HEADROOM,
        int((bundle_bytes + output_allowance) * 0.05),
    )

    work_free = int(shutil.disk_usage(work_root).free)
    report: dict[str, Any] = {
        "source_bundle_bytes": bundle_bytes,
        "estimated_output_bytes": output_allowance,
        "headroom_bytes": headroom,
        "work_dir": str(work_root),
        "work_dir_free_bytes": work_free,
    }

    if output_parent is None or _same_filesystem(work_root, output_parent):
        required = bundle_bytes + output_allowance + headroom
        report["shared_filesystem"] = True
        report["work_dir_required_bytes"] = required
        if work_free < required:
            raise SessionRecoverySafetyError(
                "Not enough free disk space for a safe recovery copy: "
                f"{_format_bytes(work_free)} available at {work_root}, "
                f"{_format_bytes(required)} required "
                f"({_format_bytes(bundle_bytes)} source bundle + "
                f"{_format_bytes(output_allowance)} output allowance + "
                f"{_format_bytes(headroom)} headroom). Use --work-dir or "
                "--output on a filesystem with more free space."
            )
        return report

    output_free = int(shutil.disk_usage(output_parent).free)
    work_required = bundle_bytes + headroom
    output_required = output_allowance + headroom
    report.update({
        "shared_filesystem": False,
        "work_dir_required_bytes": work_required,
        "output_dir": str(output_parent),
        "output_dir_free_bytes": output_free,
        "output_dir_required_bytes": output_required,
    })
    shortages: list[str] = []
    if work_free < work_required:
        shortages.append(
            f"{work_root}: {_format_bytes(work_free)} available, "
            f"{_format_bytes(work_required)} required"
        )
    if output_free < output_required:
        shortages.append(
            f"{output_parent}: {_format_bytes(output_free)} available, "
            f"{_format_bytes(output_required)} required"
        )
    if shortages:
        raise SessionRecoverySafetyError(
            "Not enough free disk space for safe recovery: "
            + "; ".join(shortages)
            + ". Choose work/output filesystems with more free space."
        )
    return report


def _copy_source_bundle(source: Path, snapshot_dir: Path) -> tuple[Path, list[str]]:
    snapshot_source = snapshot_dir / source.name
    copied: list[str] = []
    for suffix in _SIDECAR_SUFFIXES:
        source_part = _sidecar_path(source, suffix)
        if not source_part.exists():
            continue
        destination_part = _sidecar_path(snapshot_source, suffix)
        shutil.copy2(source_part, destination_part)
        copied.append(destination_part.name)
    return snapshot_source, copied


def _table_columns(conn: sqlite3.Connection, table: str) -> list[str]:
    return [str(row[1]) for row in conn.execute(f'PRAGMA table_info("{table}")')]


def _table_inventory(
    conn: sqlite3.Connection,
    table: str,
) -> dict[str, Any]:
    result: dict[str, Any] = {"available": False, "columns": [], "rows": None}
    try:
        columns = _table_columns(conn, table)
        if not columns:
            return result
        result["available"] = True
        result["columns"] = columns
        result["rows"] = int(
            conn.execute(f'SELECT COUNT(*) FROM "{table}"').fetchone()[0]
        )
    except sqlite3.DatabaseError as exc:
        result["error"] = str(exc)
    return result


def _inspect_connection(conn: sqlite3.Connection) -> dict[str, Any]:
    conn.execute("PRAGMA writable_schema=ON")
    report: dict[str, Any] = {"tables": {}, "errors": [], "warnings": []}
    try:
        row = conn.execute("PRAGMA journal_mode").fetchone()
        report["journal_mode"] = str(row[0]).lower() if row else None
    except sqlite3.DatabaseError as exc:
        report["journal_mode"] = None
        # Journal metadata is useful context but not canonical session data.
        # A damaged journal pragma must not block rows that are still readable.
        report["warnings"].append(f"journal mode: {exc}")

    for table in (*_CANONICAL_TABLES, "state_meta", *_TOPIC_TABLES):
        report["tables"][table] = _table_inventory(conn, table)

    for required in ("sessions", "messages"):
        table_report = report["tables"][required]
        if not table_report.get("available") or table_report.get("rows") is None:
            report["errors"].append(
                f"required table {required} is not completely readable"
            )
    report["recoverable"] = not report["errors"]
    return report


def _snapshot_and_inspect(
    source: Path,
    work_root: Path,
) -> tuple[tempfile.TemporaryDirectory[str], Path, dict[str, Any]]:
    before = _source_fingerprint(source)
    temp_dir = tempfile.TemporaryDirectory(
        prefix="hermes-session-recovery-",
        dir=str(work_root),
    )
    snapshot_dir = Path(temp_dir.name)
    try:
        snapshot_source, copied = _copy_source_bundle(source, snapshot_dir)
        after = _source_fingerprint(source)
        if before != after:
            raise SessionRecoverySafetyError(
                "The source database bundle changed while it was being copied. "
                "Stop every Hermes process using this profile and retry."
            )

        conn = sqlite3.connect(
            str(snapshot_source),
            isolation_level=None,
            timeout=1.0,
        )
        try:
            inspection = _inspect_connection(conn)
        finally:
            conn.close()
        inspection["source_bundle"] = copied
        inspection["source_fingerprint"] = before
        return temp_dir, snapshot_source, inspection
    except BaseException:
        temp_dir.cleanup()
        raise


def inspect_session_database(
    source_path: Path,
    *,
    work_dir: Optional[Path] = None,
) -> dict[str, Any]:
    """Inspect canonical table readability without opening the source itself."""

    source, _, work_root = _validate_paths(source_path, work_dir=work_dir)
    disk_space = _disk_space_preflight(source, work_root, None)
    temp_dir, _, inspection = _snapshot_and_inspect(source, work_root)
    try:
        return {
            "operation": "inspect",
            "source": str(source),
            "disk_space": disk_space,
            **inspection,
            "source_unchanged": _source_fingerprint(source)
            == inspection["source_fingerprint"],
        }
    finally:
        temp_dir.cleanup()


def _copy_table(
    source: sqlite3.Connection,
    destination: sqlite3.Connection,
    table: str,
    *,
    chunk_size: int,
    progress_cb: Optional[ProgressCallback],
    source_rows: Optional[int],
) -> dict[str, Any]:
    source_columns = _table_columns(source, table)
    destination_columns = _table_columns(destination, table)
    columns = [column for column in destination_columns if column in source_columns]
    result: dict[str, Any] = {
        "source_rows": source_rows,
        "copied_rows": 0,
        "columns": columns,
    }
    if not source_columns:
        result["status"] = "missing"
        return result
    if not columns:
        result["status"] = "failed"
        result["error"] = "source and destination have no compatible columns"
        return result

    quoted = ", ".join(f'"{column}"' for column in columns)
    placeholders = ", ".join("?" for _ in columns)
    select_sql = f'SELECT {quoted} FROM "{table}"'
    insert_prefix = "INSERT OR REPLACE" if table == "state_meta" else "INSERT"
    insert_sql = f'{insert_prefix} INTO "{table}" ({quoted}) VALUES ({placeholders})'

    try:
        cursor = source.execute(select_sql)
        while True:
            rows = cursor.fetchmany(chunk_size)
            if not rows:
                break
            destination.execute("BEGIN IMMEDIATE")
            try:
                destination.executemany(insert_sql, rows)
                destination.execute("COMMIT")
            except BaseException:
                destination.execute("ROLLBACK")
                raise
            result["copied_rows"] += len(rows)
            if progress_cb is not None:
                progress_cb({
                    "table": table,
                    "copied_rows": result["copied_rows"],
                    "source_rows": source_rows,
                })
    except sqlite3.DatabaseError as exc:
        result["status"] = "partial" if result["copied_rows"] else "failed"
        result["error"] = str(exc)
        return result

    result["status"] = (
        "complete"
        if source_rows is None or result["copied_rows"] == source_rows
        else "partial"
    )
    if result["status"] == "partial":
        result["error"] = (
            f"copied {result['copied_rows']} of {source_rows} readable rows"
        )
    return result


def _copy_state_meta(
    source: sqlite3.Connection,
    destination: sqlite3.Connection,
    *,
    chunk_size: int,
    progress_cb: Optional[ProgressCallback],
    source_rows: Optional[int],
) -> dict[str, Any]:
    source_columns = _table_columns(source, "state_meta")
    destination_columns = _table_columns(destination, "state_meta")
    result: dict[str, Any] = {
        "source_meta_rows": source_rows,
        "copied_rows": 0,
        "columns": ["key", "value"],
        "excluded_keys": sorted(_GENERATED_META_KEYS),
    }
    if not {"key", "value"}.issubset(source_columns):
        result["status"] = "missing"
        return result
    if not {"key", "value"}.issubset(destination_columns):
        result["status"] = "failed"
        result["error"] = "destination state_meta schema is incomplete"
        return result

    placeholders = ", ".join("?" for _ in _GENERATED_META_KEYS)
    filtered_source_rows: Optional[int] = None
    try:
        filtered_source_rows = int(
            source.execute(
                f"SELECT COUNT(*) FROM state_meta WHERE key NOT IN ({placeholders})",
                tuple(_GENERATED_META_KEYS),
            ).fetchone()[0]
        )
    except sqlite3.DatabaseError:
        # The copy loop below will return the concrete read error.
        pass

    try:
        cursor = source.execute(
            f"SELECT key, value FROM state_meta WHERE key NOT IN ({placeholders})",
            tuple(_GENERATED_META_KEYS),
        )
        while True:
            rows = cursor.fetchmany(chunk_size)
            if not rows:
                break
            destination.execute("BEGIN IMMEDIATE")
            try:
                destination.executemany(
                    "INSERT OR REPLACE INTO state_meta(key, value) VALUES (?, ?)",
                    rows,
                )
                destination.execute("COMMIT")
            except BaseException:
                destination.execute("ROLLBACK")
                raise
            result["copied_rows"] += len(rows)
            if progress_cb is not None:
                progress_cb({
                    "table": "state_meta",
                    "copied_rows": result["copied_rows"],
                    "source_rows": filtered_source_rows,
                })
    except sqlite3.DatabaseError as exc:
        result["status"] = "partial" if result["copied_rows"] else "failed"
        result["error"] = str(exc)
        return result

    result["status"] = (
        "complete"
        if filtered_source_rows is None or result["copied_rows"] == filtered_source_rows
        else "partial"
    )
    if result["status"] == "partial":
        result["error"] = (
            f"copied {result['copied_rows']} of {filtered_source_rows} readable rows"
        )
    return result


def _verify_recovered_database(
    output: Path,
    *,
    expected_counts: dict[str, Optional[int]],
    copy_report: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    verification: dict[str, Any] = {"errors": []}

    open_error = _db_opens_cleanly(output)
    verification["opens_cleanly"] = open_error is None
    if open_error is not None:
        verification["errors"].append(f"database health probe: {open_error}")

    conn = sqlite3.connect(str(output), isolation_level=None)
    try:
        integrity_rows = [
            str(row[0]) for row in conn.execute("PRAGMA integrity_check").fetchall()
        ]
        verification["integrity_check"] = integrity_rows
        if integrity_rows != ["ok"]:
            verification["errors"].append(
                "PRAGMA integrity_check did not return exactly 'ok'"
            )

        foreign_key_rows = [
            list(row) for row in conn.execute("PRAGMA foreign_key_check").fetchall()
        ]
        verification["foreign_key_check"] = foreign_key_rows
        if foreign_key_rows:
            verification["errors"].append("foreign key violations remain")

        journal_row = conn.execute("PRAGMA journal_mode").fetchone()
        verification["journal_mode"] = (
            str(journal_row[0]).lower() if journal_row else None
        )

        schema_row = conn.execute(
            "SELECT version FROM schema_version LIMIT 1"
        ).fetchone()
        verification["schema_version"] = int(schema_row[0]) if schema_row else None
        if verification["schema_version"] != SCHEMA_VERSION:
            verification["errors"].append(
                f"schema version is {verification['schema_version']}, "
                f"expected {SCHEMA_VERSION}"
            )

        meta = {
            str(row[0]): row[1]
            for row in conn.execute(
                "SELECT key, value FROM state_meta WHERE key LIKE 'fts_%'"
            ).fetchall()
        }
        verification["fts_meta"] = meta
        if meta.get("fts_storage_version") != str(FTS_STORAGE_VERSION):
            verification["errors"].append(
                "fresh FTS storage version was not established"
            )
        pending_keys = sorted(
            key
            for key in (
                "fts_optimize_available",
                "fts_rebuild_high_water",
                "fts_rebuild_progress",
                "fts_cjk_stale",
                "fts_cjk_rebuild_high_water",
                "fts_cjk_rebuild_progress",
            )
            if key in meta
        )
        verification["pending_fts_keys"] = pending_keys
        if pending_keys:
            verification["errors"].append(
                "derived FTS transition markers remain in the recovered database"
            )

        counts: dict[str, int] = {}
        for table in (*_CANONICAL_TABLES, "state_meta", *_TOPIC_TABLES):
            columns = _table_columns(conn, table)
            if columns:
                counts[table] = int(
                    conn.execute(f'SELECT COUNT(*) FROM "{table}"').fetchone()[0]
                )
        verification["table_counts"] = counts

        for table in ("sessions", "messages"):
            expected = expected_counts.get(table)
            if expected is not None and counts.get(table) != expected:
                verification["errors"].append(
                    f"{table} count is {counts.get(table)}, expected {expected}"
                )

        for table, table_report in copy_report.items():
            if table_report.get("status") in {"failed", "partial"}:
                verification["errors"].append(
                    f"{table} copy status is {table_report.get('status')}"
                )

        fts_checks: dict[str, str] = {}
        for table in ("messages_fts", "messages_fts_trigram", "messages_fts_cjk"):
            if not _table_columns(conn, table):
                continue
            try:
                conn.execute(
                    f'INSERT INTO "{table}" ("{table}") VALUES (\'integrity-check\')'
                )
                conn.execute(
                    f'SELECT 1 FROM "{table}" WHERE "{table}" MATCH \'""\' LIMIT 1'
                ).fetchone()
                fts_checks[table] = "ok"
            except sqlite3.DatabaseError as exc:
                fts_checks[table] = str(exc)
                verification["errors"].append(f"{table} integrity check failed: {exc}")
        verification["fts_checks"] = fts_checks
    except sqlite3.DatabaseError as exc:
        verification["errors"].append(f"verification query failed: {exc}")
    finally:
        conn.close()

    verification["complete"] = not verification["errors"]
    return verification


def _finalize_derived_metadata(destination: sqlite3.Connection) -> dict[str, Any]:
    """Stamp only metadata that the newly created destination actually owns."""

    fts_tables = {
        str(row[0])
        for row in destination.execute(
            "SELECT name FROM sqlite_master "
            "WHERE type='table' AND name IN ('messages_fts', 'messages_fts_trigram')"
        ).fetchall()
    }
    result: dict[str, Any] = {"fts_tables": sorted(fts_tables), "finalized": False}
    if fts_tables != {"messages_fts", "messages_fts_trigram"}:
        result["error"] = "fresh destination is missing required FTS tables"
        return result

    fts_keys = tuple(key for key in _GENERATED_META_KEYS if key.startswith("fts_"))
    placeholders = ", ".join("?" for _ in fts_keys)
    destination.execute("BEGIN IMMEDIATE")
    try:
        destination.execute(
            f"DELETE FROM state_meta WHERE key IN ({placeholders})",
            fts_keys,
        )
        destination.execute(
            "INSERT INTO state_meta(key, value) VALUES (?, ?) "
            "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            ("fts_storage_version", str(FTS_STORAGE_VERSION)),
        )
        destination.execute("COMMIT")
    except BaseException:
        destination.execute("ROLLBACK")
        raise
    result["finalized"] = True
    return result


def recover_session_database(
    source_path: Path,
    output_path: Path,
    *,
    work_dir: Optional[Path] = None,
    chunk_size: int = 1_000,
    progress_cb: Optional[ProgressCallback] = None,
) -> dict[str, Any]:
    """Recover canonical rows into a separate current-schema database.

    The source path and its sidecars are copied before SQLite opens anything.
    ``output_path`` must not exist and is never swapped into place.
    """

    if chunk_size <= 0:
        raise SessionRecoverySafetyError("chunk_size must be greater than zero")

    source, output, work_root = _validate_paths(
        source_path,
        output_path=output_path,
        work_dir=work_dir,
    )
    assert output is not None
    disk_space = _disk_space_preflight(source, work_root, output.parent)

    temp_dir, snapshot_source, inspection = _snapshot_and_inspect(source, work_root)
    try:
        if not inspection.get("recoverable"):
            reasons = "; ".join(inspection.get("errors") or ["unknown source error"])
            raise SessionRecoverySourceError(
                f"Required canonical tables are not readable: {reasons}"
            )

        source_conn = sqlite3.connect(
            str(snapshot_source),
            isolation_level=None,
            timeout=1.0,
        )
        source_conn.execute("PRAGMA writable_schema=ON")
        destination_db: Optional[SessionDB] = None
        destination_conn: Optional[sqlite3.Connection] = None
        try:
            has_topic_tables = any(
                inspection["tables"][table].get("available") for table in _TOPIC_TABLES
            )

            destination_db = SessionDB(db_path=output)
            if has_topic_tables:
                destination_db.apply_telegram_topic_migration()
            destination_db.close()
            destination_db = None

            destination_conn = sqlite3.connect(
                str(output),
                isolation_level=None,
                timeout=1.0,
            )
            destination_conn.execute("PRAGMA foreign_keys=OFF")

            copy_report: dict[str, dict[str, Any]] = {}
            for table in _CANONICAL_TABLES:
                table_inspection = inspection["tables"][table]
                copy_report[table] = _copy_table(
                    source_conn,
                    destination_conn,
                    table,
                    chunk_size=chunk_size,
                    progress_cb=progress_cb,
                    source_rows=table_inspection.get("rows"),
                )

            state_meta_inspection = inspection["tables"]["state_meta"]
            if state_meta_inspection.get("available"):
                copy_report["state_meta"] = _copy_state_meta(
                    source_conn,
                    destination_conn,
                    chunk_size=chunk_size,
                    progress_cb=progress_cb,
                    source_rows=state_meta_inspection.get("rows"),
                )
            else:
                copy_report["state_meta"] = {"status": "missing", "copied_rows": 0}

            for table in _TOPIC_TABLES:
                table_inspection = inspection["tables"][table]
                if not table_inspection.get("available"):
                    copy_report[table] = {
                        "status": "missing",
                        "copied_rows": 0,
                    }
                    continue
                copy_report[table] = _copy_table(
                    source_conn,
                    destination_conn,
                    table,
                    chunk_size=chunk_size,
                    progress_cb=progress_cb,
                    source_rows=table_inspection.get("rows"),
                )
            derived_metadata = _finalize_derived_metadata(destination_conn)
        finally:
            source_conn.close()
            if destination_conn is not None:
                destination_conn.close()
            if destination_db is not None:
                destination_db.close()

        verification = _verify_recovered_database(
            output,
            expected_counts={
                table: inspection["tables"][table].get("rows")
                for table in _CANONICAL_TABLES
            },
            copy_report=copy_report,
        )
        source_unchanged = (
            _source_fingerprint(source) == inspection["source_fingerprint"]
        )
        if not source_unchanged:
            verification["errors"].append(
                "the source database bundle changed during recovery"
            )
            verification["complete"] = False

        return {
            "operation": "recover",
            "source": str(source),
            "output": str(output),
            "source_bundle": inspection["source_bundle"],
            "source_fingerprint": inspection["source_fingerprint"],
            "source_unchanged": source_unchanged,
            "disk_space": disk_space,
            "inspection": {
                "journal_mode": inspection.get("journal_mode"),
                "tables": inspection["tables"],
                "errors": inspection["errors"],
                "warnings": inspection["warnings"],
            },
            "copy": copy_report,
            "derived_metadata": derived_metadata,
            "verification": verification,
            "complete": bool(verification.get("complete") and source_unchanged),
            "installed": False,
        }
    finally:
        temp_dir.cleanup()


def write_recovery_report(path: Path, report: dict[str, Any]) -> Path:
    """Write a JSON report without overwriting an existing file."""

    destination = _resolved_output_path(path)
    with destination.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return destination
