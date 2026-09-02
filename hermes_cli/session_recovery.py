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
    "system_prompts",
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



def _init_delivery_ledger_schema(conn: sqlite3.Connection) -> None:
    from gateway.delivery_ledger import _initialize_schema

    _initialize_schema(conn)


# Tables that live in state.db but are created lazily by a gateway module on
# first use, so base ``SessionDB`` never creates them on a fresh destination.
# Every entry maps the table to the initializer that owns its DDL; recovery
# creates the table on the destination before copying, so owed rows survive
# instead of silently vanishing from a "complete" salvage (#100313, #86236).
# Add new lazily-created state.db tables HERE, never as one-off ``if table ==``
# branches.
_AUXILIARY_TABLE_SCHEMAS: dict[str, Callable[[sqlite3.Connection], None]] = {
    "delivery_obligations": _init_delivery_ledger_schema,
}

_AUXILIARY_TABLES = tuple(_AUXILIARY_TABLE_SCHEMAS)

_INVENTORY_TABLES = (
    *_CANONICAL_TABLES,
    "state_meta",
    *_TOPIC_TABLES,
    *_AUXILIARY_TABLES,
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
_MAX_SALVAGE_RANGE_QUERIES = 10_000
_MIN_SQLITE_ROWID = -(2**63)
_MAX_SQLITE_ROWID = 2**63 - 1


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
    """Copy the source DB bundle aside so SQLite never opens the original.

    The whole copy runs inside ``offline_file_access``, which holds the
    connection-lifecycle lock for its duration. Checking for a live connection
    and *then* copying would be a check/use race: a connection could open in
    that window, and the copy's ``close()`` would cancel its POSIX advisory
    locks -- the failure class ``hermes_cli.sqlite_safe_read`` exists to
    prevent (see #71724). Holding the lock means no connection can appear
    mid-copy, across the main file and every sidecar.

    Recovery normally runs as its own short-lived CLI process against an
    offline/quarantined file, so the refusal should never fire; the guard
    keeps this path consistent with ``hermes_state._backup_db_file``.
    """
    from hermes_cli.sqlite_safe_read import LiveConnectionError, offline_file_access

    snapshot_source = snapshot_dir / source.name
    copied: list[str] = []
    try:
        with offline_file_access(source, what="snapshot"):
            for suffix in _SIDECAR_SUFFIXES:
                source_part = _sidecar_path(source, suffix)
                if not source_part.exists():
                    continue
                destination_part = _sidecar_path(snapshot_source, suffix)
                shutil.copy2(source_part, destination_part)
                copied.append(destination_part.name)
    except LiveConnectionError as exc:
        raise SessionRecoverySafetyError(str(exc)) from exc
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

    for table in _INVENTORY_TABLES:
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
                "Stop every Hermes process using this profile and retry. "
                "This includes the interactive `hermes` CLI session this "
                "command may have been launched from: a running parent CLI "
                "writes session bookkeeping (compression ticks, context "
                "tracking) to state.db in the background and counts as a "
                "Hermes process even after the gateway is stopped. Run the "
                "recovery from a fresh shell with no `hermes` session open, "
                "or point --source at an immutable snapshot copy of the "
                "database."
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


def _ensure_auxiliary_destination_schema(
    destination: sqlite3.Connection,
    table: str,
) -> None:
    """Create a lazy auxiliary table on the recovered destination.

    Recovery initializes the destination through base ``SessionDB``, which
    does not create gateway-owned tables. Copying into a missing dest table
    would report ``missing`` / ``no compatible columns`` and drop the rows.
    """

    initialize = _AUXILIARY_TABLE_SCHEMAS.get(table)
    if initialize is None:
        raise SessionRecoverySafetyError(
            f"no destination schema initializer registered for table {table!r}"
        )
    initialize(destination)


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


def _append_skipped_range(
    ranges: list[dict[str, Any]],
    low: int,
    high: int,
    error: str,
) -> None:
    """Record skipped rowid ranges without producing one entry per row."""

    if (
        ranges
        and ranges[-1]["high"] + 1 == low
        and ranges[-1]["error"] == error
    ):
        ranges[-1]["high"] = high
        return
    ranges.append({"low": low, "high": high, "error": error})


def _salvage_rowid_bounds(
    source: sqlite3.Connection,
    table: str,
) -> dict[str, Any]:
    """Find the readable rowid edges without scanning the complete table."""

    result: dict[str, Any] = {"errors": [], "fallback_edges": []}
    rows: dict[str, Optional[int]] = {"low": None, "high": None}
    directions = (("low", "ASC"), ("high", "DESC"))
    for edge, direction in directions:
        try:
            row = source.execute(
                f'SELECT rowid FROM "{table}" ORDER BY rowid {direction} LIMIT 1'
            ).fetchone()
            if row is not None:
                rows[edge] = int(row[0])
        except sqlite3.DatabaseError as exc:
            result["errors"].append(f"{edge} rowid: {exc}")

    if rows["low"] is None and rows["high"] is None and not result["errors"]:
        result["empty"] = True
        return result
    if rows["low"] is None and rows["high"] is None:
        result["unavailable"] = True
        return result

    # A damaged edge can prevent one of the ordered probes from completing.
    # Keep the other readable edge and bound the missing side by SQLite's
    # rowid domain. Range bisection can then approach the surviving data
    # without assuming that user-created databases contain only positive IDs.
    if rows["low"] is None:
        rows["low"] = _MIN_SQLITE_ROWID
        result["fallback_edges"].append("low")
    if rows["high"] is None:
        rows["high"] = _MAX_SQLITE_ROWID
        result["fallback_edges"].append("high")

    result["low"] = rows["low"]
    result["high"] = rows["high"]
    return result


def _probe_populated_edge(
    source: sqlite3.Connection,
    table: str,
    *,
    edge: str,
    anchor: int,
) -> dict[str, Any]:
    """Find a finite bound for a damaged rowid edge (issue #80205).

    When an ordered edge probe fails, :func:`_salvage_rowid_bounds` used to
    substitute the whole SQLite rowid domain. Range bisection then burned the
    entire ``_MAX_SALVAGE_RANGE_QUERIES`` budget subdividing an enormous
    synthetic tail that could not contain real rows — and once the budget was
    gone, rows that were still readable were silently recorded as skipped.

    This gallops outward from the readable ``anchor`` edge with exponentially
    growing offsets. A probe that cleanly reports "no rows beyond X" caps the
    domain at X; a probe that errors (its b-tree path crosses the damage) or
    finds a row keeps growing. At most ~64 probes per edge, so the cap costs
    a bounded, tiny slice of the salvage budget instead of all of it.
    """

    ascending = edge == "high"
    comparison = ">" if ascending else "<"
    probe_sql = (
        f'SELECT rowid FROM "{table}" WHERE rowid {comparison} ? '
        f'ORDER BY rowid {"ASC" if ascending else "DESC"} LIMIT 1'
    )
    domain_limit = _MAX_SQLITE_ROWID if ascending else _MIN_SQLITE_ROWID
    result: dict[str, Any] = {"edge": edge, "probes": 0, "capped": False}

    position = anchor
    span = 1
    while True:
        candidate = position + span if ascending else position - span
        if (ascending and candidate >= domain_limit) or (
            not ascending and candidate <= domain_limit
        ):
            # No clean empty-tail answer before the domain edge; keep the
            # domain fallback rather than inventing a bound.
            result["bound"] = domain_limit
            return result
        result["probes"] += 1
        try:
            row = source.execute(probe_sql, (candidate,)).fetchone()
        except sqlite3.DatabaseError:
            # Damage on the probe path — inconclusive, widen further.
            span *= 2
            continue
        if row is None:
            # Clean answer: nothing beyond ``candidate``. The salvageable
            # data ends at or before it, so the synthetic domain tail is
            # provably empty and need not be bisected at all.
            result["bound"] = candidate
            result["capped"] = True
            return result
        # Rows exist beyond the candidate; advance the anchor. The span keeps
        # doubling (never resets) so the whole gallop stays O(log range).
        position = int(row[0])
        span *= 2


def _copy_table_salvage(
    source: sqlite3.Connection,
    destination: sqlite3.Connection,
    table: str,
    *,
    chunk_size: int,
    progress_cb: Optional[ProgressCallback],
    source_rows: Optional[int],
    insert_prefix: str = "INSERT",
    row_filter: Optional[
        Callable[[tuple[Any, ...], tuple[str, ...]], bool]
    ] = None,
) -> dict[str, Any]:
    """Best-effort rowid-range copy that continues past damaged source pages."""

    source_columns = _table_columns(source, table)
    destination_columns = _table_columns(destination, table)
    columns = [column for column in destination_columns if column in source_columns]
    result: dict[str, Any] = {
        "mode": "rowid_range_salvage",
        "source_rows": source_rows,
        "copied_rows": 0,
        "excluded_rows": 0,
        "columns": columns,
        "range_queries": 0,
        "exact_lookup_recovered": 0,
        "skipped_rowid_ranges": [],
    }
    if not source_columns:
        result["status"] = "missing"
        return result
    if not columns:
        result["status"] = "failed"
        result["error"] = "source and destination have no compatible columns"
        return result

    bounds = _salvage_rowid_bounds(source, table)
    result["rowid_bounds"] = bounds
    if bounds.get("empty"):
        result["status"] = "complete"
        return result
    if bounds.get("low") is None or bounds.get("high") is None:
        result["status"] = "failed"
        details = "; ".join(bounds.get("errors") or [])
        result["error"] = "could not determine a rowid range for salvage"
        if details:
            result["error"] += f": {details}"
        return result

    # Issue #80205: a damaged ordered edge probe used to substitute the whole
    # SQLite rowid domain, and bisecting that synthetic tail exhausted the
    # range-query budget while readable tail rows were still waiting to be
    # copied. Gallop outward from the surviving edge for a finite bound first.
    fallback_edges = bounds.get("fallback_edges") or []
    if fallback_edges:
        bounds["edge_probes"] = []
        if "high" in fallback_edges and bounds.get("low") is not None:
            probe = _probe_populated_edge(
                source, table, edge="high", anchor=int(bounds["low"])
            )
            bounds["edge_probes"].append(probe)
            if probe["capped"]:
                bounds["high"] = int(probe["bound"])
        if "low" in fallback_edges and bounds.get("high") is not None:
            probe = _probe_populated_edge(
                source, table, edge="low", anchor=int(bounds["high"])
            )
            bounds["edge_probes"].append(probe)
            if probe["capped"]:
                bounds["low"] = int(probe["bound"])

    quoted = ", ".join(f'"{column}"' for column in columns)
    placeholders = ", ".join("?" for _ in columns)
    select_sql = (
        f'SELECT rowid, {quoted} FROM "{table}" '
        "WHERE rowid BETWEEN ? AND ? ORDER BY rowid"
    )
    insert_sql = (
        f'{insert_prefix} INTO "{table}" ({quoted}) VALUES ({placeholders})'
    )
    column_names = tuple(columns)
    stopped_at_query_limit = False
    exact_sql = f'SELECT {quoted} FROM "{table}" WHERE rowid = ?'

    def recover_exact_rowid(rowid: int) -> bool:
        """Issue #80205: salvage one row by exact-key lookup.

        A singleton range scan (``rowid BETWEEN x AND x ORDER BY rowid``)
        must advance the cursor past ``x`` to prove the range is exhausted;
        when the *next* cell or page is damaged that advance raises AFTER the
        row was produced, and the driver discards the already-fetched row. An
        equality lookup on the rowid stops at the hit, so the boundary row
        directly before a damaged page — readable in the field case and by
        SQLite's page-level ``.recover`` — is recovered instead of being
        recorded as skipped.
        """
        result["range_queries"] += 1
        try:
            row = source.execute(exact_sql, (rowid,)).fetchone()
        except sqlite3.DatabaseError:
            return False
        if row is None:
            return True  # genuinely absent: nothing to skip
        value = tuple(row)
        if row_filter is not None and not row_filter(value, column_names):
            result["excluded_rows"] += 1
            return True
        destination.execute("BEGIN IMMEDIATE")
        try:
            destination.execute(insert_sql, value)
            destination.execute("COMMIT")
        except BaseException:
            destination.execute("ROLLBACK")
            raise
        result["copied_rows"] += 1
        result["exact_lookup_recovered"] += 1
        return True

    def copy_range(low: int, high: int) -> None:
        nonlocal stopped_at_query_limit
        if low > high:
            return
        if result["range_queries"] >= _MAX_SALVAGE_RANGE_QUERIES:
            stopped_at_query_limit = True
            _append_skipped_range(
                result["skipped_rowid_ranges"],
                low,
                high,
                "salvage range query limit reached",
            )
            return

        result["range_queries"] += 1
        last_committed_rowid: Optional[int] = None
        try:
            cursor = source.execute(select_sql, (low, high))
            while True:
                fetched = cursor.fetchmany(chunk_size)
                if not fetched:
                    return

                values = [tuple(row[1:]) for row in fetched]
                if row_filter is not None:
                    included = [
                        row for row in values if row_filter(row, column_names)
                    ]
                    excluded_count = len(values) - len(included)
                else:
                    included = values
                    excluded_count = 0

                if included:
                    destination.execute("BEGIN IMMEDIATE")
                    try:
                        destination.executemany(insert_sql, included)
                        destination.execute("COMMIT")
                    except BaseException:
                        destination.execute("ROLLBACK")
                        raise

                result["copied_rows"] += len(included)
                result["excluded_rows"] += excluded_count
                last_committed_rowid = int(fetched[-1][0])
                if progress_cb is not None:
                    progress_cb({
                        "table": table,
                        "copied_rows": result["copied_rows"],
                        "source_rows": source_rows,
                        "skipped_ranges": len(result["skipped_rowid_ranges"]),
                    })
        except sqlite3.DatabaseError as exc:
            retry_low = (
                last_committed_rowid + 1
                if last_committed_rowid is not None
                else low
            )
            if retry_low > high:
                return
            if retry_low == high:
                if not recover_exact_rowid(retry_low):
                    _append_skipped_range(
                        result["skipped_rowid_ranges"],
                        retry_low,
                        high,
                        str(exc),
                    )
                return
            midpoint = retry_low + (high - retry_low) // 2
            copy_range(retry_low, midpoint)
            copy_range(midpoint + 1, high)

    copy_range(int(bounds["low"]), int(bounds["high"]))
    skipped_ranges = result["skipped_rowid_ranges"]
    result["skipped_rowid_span"] = sum(
        item["high"] - item["low"] + 1 for item in skipped_ranges
    )
    result["query_limit_reached"] = stopped_at_query_limit

    if skipped_ranges:
        result["status"] = "partial" if result["copied_rows"] else "failed"
        result["error"] = (
            f"{len(skipped_ranges)} rowid range(s) skipped"
        )
    elif (
        source_rows is not None
        and result["copied_rows"] + result["excluded_rows"] != source_rows
    ):
        result["status"] = "partial"
        result["error"] = (
            f"copied {result['copied_rows']} and excluded "
            f"{result['excluded_rows']} of {source_rows} source rows"
        )
    else:
        result["status"] = "complete"
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


def _copy_state_meta_salvage(
    source: sqlite3.Connection,
    destination: sqlite3.Connection,
    *,
    chunk_size: int,
    progress_cb: Optional[ProgressCallback],
    source_rows: Optional[int],
) -> dict[str, Any]:
    """Salvage readable user metadata while regenerating derived FTS state.

    Requires both ``key`` and ``value``, matching the non-partial
    :func:`_copy_state_meta`. A damaged ``state_meta`` can retain one column
    and lose the other; without this check a missing ``key`` raised
    ``ValueError`` from ``columns.index("key")`` and aborted the entire
    partial recovery, and a missing ``value`` would have copied key-only rows
    while reporting the table complete.

    Status matters here. An unusable-but-PRESENT table reports ``failed``, not
    ``missing``: verification only escalates ``failed``/``partial`` into a
    warning + ``loss_detected``, so reporting ``missing`` would silently drop
    real metadata and still claim ``complete=True``. ``missing`` is reserved
    for a table that genuinely is not there. Either way ``state_meta`` is
    optional, so ``--allow-partial`` records the loss and carries on
    recovering sessions and messages.
    """
    source_columns = _table_columns(source, "state_meta")
    destination_columns = _table_columns(destination, "state_meta")
    if not source_columns:
        # Genuinely absent from the source — nothing was lost.
        return {
            "mode": "rowid_range_salvage",
            "source_meta_rows": source_rows,
            "copied_rows": 0,
            "columns": ["key", "value"],
            "excluded_keys": sorted(_GENERATED_META_KEYS),
            "status": "missing",
        }
    if not {"key", "value"}.issubset(source_columns):
        # Present but unusable: this IS data loss and must be reported.
        return {
            "mode": "rowid_range_salvage",
            "source_meta_rows": source_rows,
            "copied_rows": 0,
            "columns": ["key", "value"],
            "excluded_keys": sorted(_GENERATED_META_KEYS),
            "status": "failed",
            "error": (
                "source state_meta exists but is missing the key/value "
                f"columns (found: {', '.join(source_columns) or 'none'})"
            ),
        }
    if not {"key", "value"}.issubset(destination_columns):
        return {
            "mode": "rowid_range_salvage",
            "source_meta_rows": source_rows,
            "copied_rows": 0,
            "columns": ["key", "value"],
            "excluded_keys": sorted(_GENERATED_META_KEYS),
            "status": "failed",
            "error": "destination state_meta schema is incomplete",
        }

    def keep_user_meta(
        row: tuple[Any, ...],
        columns: tuple[str, ...],
    ) -> bool:
        return str(row[columns.index("key")]) not in _GENERATED_META_KEYS

    result = _copy_table_salvage(
        source,
        destination,
        "state_meta",
        chunk_size=chunk_size,
        progress_cb=progress_cb,
        source_rows=source_rows,
        insert_prefix="INSERT OR REPLACE",
        row_filter=keep_user_meta,
    )
    result["source_meta_rows"] = result.pop("source_rows")
    result["excluded_keys"] = sorted(_GENERATED_META_KEYS)
    return result


def _reconstruct_missing_sessions(
    destination: sqlite3.Connection,
) -> dict[str, Any]:
    """Recreate placeholder session rows for salvaged orphaned messages.

    When the ``sessions`` b-tree is damaged worse than ``messages``, salvage
    can recover the conversation text while recovering few or none of the
    session rows that own it. Deleting those messages as "orphans" throws away
    the only readable copy of the user's data — the exact opposite of what
    ``--allow-partial`` is for. A real report (July 2026) copied 20,817 of
    20,824 messages and then removed every one of them, producing an output
    with 0 sessions and 0 messages.

    Instead, synthesize a minimal session row per orphaned ``session_id``
    (only ``id``/``source``/``started_at`` are NOT NULL) so the messages stay
    reachable and foreign keys hold. ``started_at`` is taken from the earliest
    surviving message so ordering stays sane. Rows are marked with
    ``source='recovered'`` and a ``title`` that says so, because a fabricated
    session must never be mistaken for an original.
    """
    result: dict[str, Any] = {"sessions_reconstructed": 0, "messages_retained": 0}
    if not _table_columns(destination, "sessions"):
        return result
    if not _table_columns(destination, "messages"):
        return result

    orphaned = destination.execute(
        "SELECT m.session_id, MIN(m.timestamp), COUNT(*) "
        "FROM messages AS m "
        "WHERE m.session_id IS NOT NULL AND NOT EXISTS ("
        "SELECT 1 FROM sessions WHERE sessions.id = m.session_id) "
        "GROUP BY m.session_id"
    ).fetchall()
    if not orphaned:
        return result

    title_sequence = 1
    for session_id, first_timestamp, message_count in orphaned:
        started_at = float(first_timestamp) if first_timestamp is not None else 0.0
        while True:
            title = (
                f"[recovered {title_sequence}] "
                "session metadata was unreadable"
            )
            title_sequence += 1
            if (
                destination.execute(
                    "SELECT 1 FROM sessions WHERE title = ? LIMIT 1",
                    (title,),
                ).fetchone()
                is None
            ):
                break

        cursor = destination.execute(
            "INSERT INTO sessions "
            "(id, source, started_at, title, message_count) "
            "VALUES (?, 'recovered', ?, ?, ?)",
            (
                session_id,
                started_at,
                title,
                int(message_count),
            ),
        )
        if cursor.rowcount != 1:
            raise sqlite3.IntegrityError(
                f"failed to reconstruct missing session {session_id!r}"
            )
        result["sessions_reconstructed"] += 1
        result["messages_retained"] += int(message_count)
    return result


def _cleanup_partial_orphans(
    destination: sqlite3.Connection,
) -> dict[str, Any]:
    """Reconcile references to sessions that could not be salvaged.

    Messages are never discarded for lack of a session row: their owning
    session is reconstructed as a placeholder first (see
    :func:`_reconstruct_missing_sessions`). Only rows that remain orphaned
    after that — and rows in tables carrying no recoverable user content —
    are removed.
    """

    result: dict[str, Any] = {
        "session_prompt_refs_cleared": 0,
        "system_prompts_removed": 0,
        "sessions_parent_cleared": 0,
        "sessions_reconstructed": 0,
        "messages_retained": 0,
        "messages_removed": 0,
        "session_model_usage_removed": 0,
        "compression_locks_removed": 0,
        "telegram_dm_topic_bindings_removed": 0,
    }
    destination.execute("BEGIN IMMEDIATE")
    try:
        # Rebuild owners BEFORE any orphan deletion so salvaged conversation
        # text is never dropped for want of a session row.
        rebuilt = _reconstruct_missing_sessions(destination)
        result["sessions_reconstructed"] = rebuilt["sessions_reconstructed"]
        result["messages_retained"] = rebuilt["messages_retained"]

        parent_count = int(
            destination.execute(
                "SELECT COUNT(*) FROM sessions AS child "
                "WHERE child.parent_session_id IS NOT NULL "
                "AND NOT EXISTS ("
                "SELECT 1 FROM sessions AS parent "
                "WHERE parent.id = child.parent_session_id)"
            ).fetchone()[0]
        )
        if parent_count:
            destination.execute(
                "UPDATE sessions SET parent_session_id = NULL "
                "WHERE parent_session_id IS NOT NULL "
                "AND NOT EXISTS ("
                "SELECT 1 FROM sessions AS parent "
                "WHERE parent.id = sessions.parent_session_id)"
            )
        result["sessions_parent_cleared"] = parent_count

        prompt_ref_count = int(
            destination.execute(
                "SELECT COUNT(*) FROM sessions "
                "WHERE system_prompt_hash IS NOT NULL "
                "AND NOT EXISTS ("
                "SELECT 1 FROM system_prompts "
                "WHERE system_prompts.hash = sessions.system_prompt_hash)"
            ).fetchone()[0]
        )
        if prompt_ref_count:
            destination.execute(
                "UPDATE sessions SET system_prompt_hash = NULL "
                "WHERE system_prompt_hash IS NOT NULL "
                "AND NOT EXISTS ("
                "SELECT 1 FROM system_prompts "
                "WHERE system_prompts.hash = sessions.system_prompt_hash)"
            )
        result["session_prompt_refs_cleared"] = prompt_ref_count

        unreferenced_prompt_count = int(
            destination.execute(
                "SELECT COUNT(*) FROM system_prompts "
                "WHERE NOT EXISTS ("
                "SELECT 1 FROM sessions "
                "WHERE sessions.system_prompt_hash = system_prompts.hash)"
            ).fetchone()[0]
        )
        if unreferenced_prompt_count:
            destination.execute(
                "DELETE FROM system_prompts "
                "WHERE NOT EXISTS ("
                "SELECT 1 FROM sessions "
                "WHERE sessions.system_prompt_hash = system_prompts.hash)"
            )
        result["system_prompts_removed"] = unreferenced_prompt_count

        dependent_tables = (
            ("messages", "messages_removed"),
            ("session_model_usage", "session_model_usage_removed"),
            ("compression_locks", "compression_locks_removed"),
            (
                "telegram_dm_topic_bindings",
                "telegram_dm_topic_bindings_removed",
            ),
        )
        for table, report_key in dependent_tables:
            if not _table_columns(destination, table):
                continue
            orphan_count = int(
                destination.execute(
                    f'SELECT COUNT(*) FROM "{table}" AS dependent '
                    "WHERE NOT EXISTS ("
                    "SELECT 1 FROM sessions "
                    "WHERE sessions.id = dependent.session_id)"
                ).fetchone()[0]
            )
            if orphan_count:
                destination.execute(
                    f'DELETE FROM "{table}" '
                    "WHERE NOT EXISTS ("
                    "SELECT 1 FROM sessions "
                    f'WHERE sessions.id = "{table}".session_id)'
                )
            result[report_key] = orphan_count
        destination.execute("COMMIT")
    except BaseException:
        destination.execute("ROLLBACK")
        raise
    # Only destructive/relinking actions belong in this total. The
    # reconstruction counters describe data RETAINED, so summing them here
    # would report saving the user's messages as if it were losing them.
    result["total_removed_or_relinked"] = (
        int(result["session_prompt_refs_cleared"])
        + int(result["sessions_parent_cleared"])
        + int(result["messages_removed"])
        + int(result["session_model_usage_removed"])
        + int(result["compression_locks_removed"])
        + int(result["telegram_dm_topic_bindings_removed"])
    )
    return result


def _verify_recovered_database(
    output: Path,
    *,
    expected_counts: dict[str, Optional[int]],
    copy_report: dict[str, dict[str, Any]],
    allow_partial: bool = False,
    orphan_cleanup: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    verification: dict[str, Any] = {
        "errors": [],
        "warnings": [],
        "loss_detected": False,
    }

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
        for table in _INVENTORY_TABLES:
            columns = _table_columns(conn, table)
            if columns:
                counts[table] = int(
                    conn.execute(f'SELECT COUNT(*) FROM "{table}"').fetchone()[0]
                )
        verification["table_counts"] = counts

        for table in ("sessions", "messages", *_AUXILIARY_TABLES):
            expected = expected_counts.get(table)
            if expected is not None and counts.get(table) != expected:
                message = (
                    f"{table} count is {counts.get(table)}, expected {expected}"
                )
                if allow_partial:
                    verification["warnings"].append(message)
                    verification["loss_detected"] = True
                else:
                    verification["errors"].append(message)

        cleanup = orphan_cleanup or {}
        rebuilt_sessions = int(cleanup.get("sessions_reconstructed") or 0)
        retained_messages = int(cleanup.get("messages_retained") or 0)
        removed_messages = int(cleanup.get("messages_removed") or 0)
        # A wholly unreadable sessions b-tree is recoverable when every output
        # parent was rebuilt from the surviving messages and none were dropped.
        # This is still data loss, but it is not structural verification failure.
        sessions_fully_reconstructed = bool(
            rebuilt_sessions > 0
            and counts.get("sessions") == rebuilt_sessions
            and counts.get("messages") == retained_messages
            and removed_messages == 0
        )

        for table, table_report in copy_report.items():
            status = table_report.get("status")
            if status not in {"failed", "partial"}:
                continue
            message = f"{table} copy status is {status}"
            if allow_partial and (
                status == "partial"
                or table not in {"sessions", "messages"}
                or (
                    table == "sessions"
                    and status == "failed"
                    and sessions_fully_reconstructed
                )
            ):
                verification["warnings"].append(message)
                verification["loss_detected"] = True
            else:
                verification["errors"].append(message)

        if orphan_cleanup:
            orphan_count = int(
                orphan_cleanup.get("total_removed_or_relinked") or 0
            )
            if orphan_count:
                verification["warnings"].append(
                    f"{orphan_count} orphaned reference(s) were removed or relinked"
                )
                verification["loss_detected"] = True
            rebuilt_sessions = int(
                orphan_cleanup.get("sessions_reconstructed") or 0
            )
            if rebuilt_sessions:
                retained = int(orphan_cleanup.get("messages_retained") or 0)
                # Not a clean recovery: the conversation text survived but its
                # session metadata did not, so these rows are placeholders.
                verification["warnings"].append(
                    f"{rebuilt_sessions} session(s) could not be salvaged and "
                    f"were reconstructed as placeholders to retain "
                    f"{retained} message(s); their metadata (title, model, "
                    "timestamps, cost) is lost"
                )
                verification["loss_detected"] = True

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

    verification["healthy"] = not verification["errors"]
    verification["complete"] = bool(
        verification["healthy"] and not verification["loss_detected"]
    )
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


# Floor for a plausible unix-epoch timestamp (2001-09-09). Salvage rows
# uniformly below it were mis-mapped, not merely unlucky (#101409).
_PLAUSIBLE_TIMESTAMP_FLOOR = 1_000_000_000.0


def _lost_and_found_plausibility_errors(
    conn: sqlite3.Connection,
) -> list[str]:
    """Flag systematic timestamp mis-mapping in a salvaged database.

    Structural checks (integrity, FK, FTS, row counts) pass on mis-mapped
    salvage because every row still inserts. Only semantics give it away:
    the physical column order of a source upgraded via ALTER TABLE differs
    from the destination template's declared order, so positional cell
    mapping lands counters/strings where ``started_at``/``timestamp``
    belong — and the NOT NULL substitutes turn gaps into 0.0. When every
    row violates the epoch floor, the mapping was wrong.
    """

    errors: list[str] = []

    (session_total,) = conn.execute(
        "SELECT COUNT(*) FROM sessions"
    ).fetchone()
    if session_total:
        (implausible,) = conn.execute(
            "SELECT COUNT(*) FROM sessions "
            "WHERE started_at IS NULL OR started_at < ?",
            (_PLAUSIBLE_TIMESTAMP_FLOOR,),
        ).fetchone()
        if implausible == session_total:
            errors.append(
                f"sessions.started_at is implausible in all "
                f"{session_total} salvaged row(s) (NULL or before 2001-09): "
                "the source's physical column order did not match the "
                "destination template, so cells were mapped onto the "
                "wrong columns"
            )

    (message_total,) = conn.execute(
        "SELECT COUNT(*) FROM messages"
    ).fetchone()
    if message_total:
        (implausible,) = conn.execute(
            "SELECT COUNT(*) FROM messages "
            "WHERE timestamp IS NULL OR timestamp < ?",
            (_PLAUSIBLE_TIMESTAMP_FLOOR,),
        ).fetchone()
        if implausible == message_total:
            errors.append(
                f"messages.timestamp is implausible in all "
                f"{message_total} salvaged row(s) (NULL or before 2001-09): "
                "the source's physical column order did not match the "
                "destination template, so cells were mapped onto the "
                "wrong columns"
            )

    return errors


def _recover_via_lost_and_found(
    *,
    source: Path,
    snapshot_source: Path,
    snapshot_dir: Path,
    output: Path,
    inspection: dict[str, Any],
    disk_space: dict[str, Any],
    missing_required: list[str],
) -> dict[str, Any]:
    """Best-effort page-level salvage when table schemas are unreadable.

    Shells out to the sqlite3 CLI's ``.recover`` (a shell-only feature, not
    part of Python's ``sqlite3`` module) to rebuild rows into a scratch
    lost_and_found database, then heuristically maps them into a fresh
    current-schema database. The result is explicitly labeled best-effort.
    """

    from hermes_cli.session_lost_and_found import (
        SQLITE3_CLI_GUIDANCE,
        LostAndFoundError,
        find_sqlite3_cli,
        map_lost_and_found_rows,
        rebuild_fts_indexes,
        run_cli_lost_and_found_recover,
        stub_missing_parent_sessions,
    )

    sqlite3_bin = find_sqlite3_cli()
    if sqlite3_bin is None:
        raise SessionRecoverySourceError(
            "Partial recovery still requires readable table schemas for: "
            + ", ".join(missing_required)
            + ". " + SQLITE3_CLI_GUIDANCE
        )

    lf_path = snapshot_dir / "lost_and_found.db"
    try:
        cli_report = run_cli_lost_and_found_recover(
            snapshot_source, lf_path, sqlite3_bin
        )
    except (LostAndFoundError, OSError) as exc:
        raise SessionRecoverySourceError(
            "Partial recovery could not read the table schemas for: "
            + ", ".join(missing_required)
            + f", and page-level .recover salvage failed: {exc}"
        ) from exc

    destination_db = SessionDB(db_path=output)
    destination_db.close()

    lf_conn = sqlite3.connect(str(lf_path), isolation_level=None)
    destination_conn = sqlite3.connect(
        str(output), isolation_level=None, timeout=1.0
    )
    try:
        destination_conn.execute("PRAGMA foreign_keys=OFF")
        mapping = map_lost_and_found_rows(lf_conn, destination_conn)
        stubbing = stub_missing_parent_sessions(destination_conn)
        fts = rebuild_fts_indexes(destination_conn)
        derived_metadata = _finalize_derived_metadata(destination_conn)
    finally:
        lf_conn.close()
        destination_conn.close()

    copy_report: dict[str, dict[str, Any]] = {
        table: {
            "mode": "lost_and_found_salvage",
            "status": "partial",
            "copied_rows": (
                int(mapping["direct_table_rows"].get(table) or 0)
                + int(mapping["mapped"].get(table) or 0)
            ),
            "error": "recovered via page-level lost_and_found salvage; "
            "row completeness cannot be verified against the source",
        }
        for table in ("sessions", "messages", "session_model_usage")
    }

    verification = _verify_recovered_database(
        output,
        expected_counts={"sessions": None, "messages": None},
        copy_report=copy_report,
        allow_partial=True,
        orphan_cleanup={
            "sessions_reconstructed": stubbing["sessions_stubbed"],
            "messages_retained": stubbing["messages_retained"],
            "messages_removed": 0,
            "total_removed_or_relinked": 0,
        },
    )
    verification["loss_detected"] = True
    verification["warnings"].append(
        "BEST-EFFORT page-level salvage: the source table schemas were "
        "unreadable, so rows were rebuilt from raw pages and mapped "
        "heuristically. Review every count before trusting this output."
    )
    verification["complete"] = False

    # Structural checks cannot see a positional mis-mapping (#101409):
    # every row still inserts, so integrity/FK/FTS stay green. A
    # systematic timestamp violation is the semantic tell — surface it
    # so a mis-mapped salvage is never reported as verified.
    plausibility_conn = sqlite3.connect(str(output), isolation_level=None)
    try:
        plausibility_errors = _lost_and_found_plausibility_errors(
            plausibility_conn
        )
    finally:
        plausibility_conn.close()
    if plausibility_errors:
        verification["errors"].extend(plausibility_errors)
        verification["healthy"] = False

    source_unchanged = (
        _source_fingerprint(source) == inspection["source_fingerprint"]
    )
    if not source_unchanged:
        verification["errors"].append(
            "the source database bundle changed during recovery"
        )
        verification["healthy"] = False

    return {
        "operation": "recover",
        "allow_partial": True,
        "mode": "lost_and_found_salvage",
        "best_effort": True,
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
        "unreadable_schemas": missing_required,
        "sqlite3_cli": cli_report,
        "lost_and_found": mapping,
        "session_stubs": stubbing,
        "fts_rebuild": fts,
        "copy": copy_report,
        "orphan_cleanup": {
            "sessions_reconstructed": stubbing["sessions_stubbed"],
            "messages_retained": stubbing["messages_retained"],
            "messages_removed": 0,
            "total_removed_or_relinked": 0,
        },
        "derived_metadata": derived_metadata,
        "verification": verification,
        "complete": False,
        "partial": True,
        "verified": bool(verification.get("healthy") and source_unchanged),
        "installed": False,
    }


def recover_session_database(
    source_path: Path,
    output_path: Path,
    *,
    work_dir: Optional[Path] = None,
    chunk_size: int = 1_000,
    progress_cb: Optional[ProgressCallback] = None,
    allow_partial: bool = False,
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
        if not inspection.get("recoverable") and not allow_partial:
            reasons = "; ".join(inspection.get("errors") or ["unknown source error"])
            raise SessionRecoverySourceError(
                f"Required canonical tables are not readable: {reasons}. "
                "Re-run with --allow-partial to salvage every readable row "
                "into a new database (the source is never modified)."
            )
        if allow_partial:
            missing_required = [
                table
                for table in ("sessions", "messages")
                if not inspection["tables"][table].get("available")
            ]
            if missing_required:
                # SQL-level salvage is impossible without readable table
                # schemas. Fall back to page-level lost_and_found salvage via
                # the sqlite3 CLI's .recover (a shell-only feature).
                return _recover_via_lost_and_found(
                    source=source,
                    snapshot_source=snapshot_source,
                    snapshot_dir=Path(temp_dir.name),
                    output=output,
                    inspection=inspection,
                    disk_space=disk_space,
                    missing_required=missing_required,
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
                copy_function = (
                    _copy_table_salvage if allow_partial else _copy_table
                )
                copy_report[table] = copy_function(
                    source_conn,
                    destination_conn,
                    table,
                    chunk_size=chunk_size,
                    progress_cb=progress_cb,
                    source_rows=table_inspection.get("rows"),
                )

            state_meta_inspection = inspection["tables"]["state_meta"]
            if state_meta_inspection.get("available"):
                state_meta_copy_function = (
                    _copy_state_meta_salvage
                    if allow_partial
                    else _copy_state_meta
                )
                copy_report["state_meta"] = state_meta_copy_function(
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
                copy_function = (
                    _copy_table_salvage if allow_partial else _copy_table
                )
                copy_report[table] = copy_function(
                    source_conn,
                    destination_conn,
                    table,
                    chunk_size=chunk_size,
                    progress_cb=progress_cb,
                    source_rows=table_inspection.get("rows"),
                )

            for table in _AUXILIARY_TABLES:
                table_inspection = inspection["tables"][table]
                if not table_inspection.get("available"):
                    copy_report[table] = {
                        "status": "missing",
                        "copied_rows": 0,
                    }
                    continue
                _ensure_auxiliary_destination_schema(destination_conn, table)
                copy_function = (
                    _copy_table_salvage if allow_partial else _copy_table
                )
                copy_report[table] = copy_function(
                    source_conn,
                    destination_conn,
                    table,
                    chunk_size=chunk_size,
                    progress_cb=progress_cb,
                    source_rows=table_inspection.get("rows"),
                )
            orphan_cleanup = (
                _cleanup_partial_orphans(destination_conn)
                if allow_partial
                else None
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
                **{
                    table: inspection["tables"][table].get("rows")
                    for table in _CANONICAL_TABLES
                },
                **{
                    table: inspection["tables"][table].get("rows")
                    for table in _AUXILIARY_TABLES
                    if inspection["tables"].get(table, {}).get("available")
                },
            },
            copy_report=copy_report,
            allow_partial=allow_partial,
            orphan_cleanup=orphan_cleanup,
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
            "allow_partial": allow_partial,
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
            "orphan_cleanup": orphan_cleanup,
            "derived_metadata": derived_metadata,
            "verification": verification,
            "complete": bool(verification.get("complete") and source_unchanged),
            "partial": bool(verification.get("loss_detected")),
            "verified": bool(verification.get("healthy") and source_unchanged),
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
