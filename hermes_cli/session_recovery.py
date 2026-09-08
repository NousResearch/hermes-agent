"""Offline, non-destructive recovery for a damaged Hermes session database.

The source is never opened by SQLite: it and its WAL/SHM/journal sidecars are copied to a disposable
work dir first. Canonical rows are copied into a fresh current-schema database; derived FTS tables and
migration bookkeeping are rebuilt, not copied; the result is never installed over the active database.
"""

from __future__ import annotations

import json
import os
import shutil
import sqlite3
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Iterator, Optional

from hermes_state import SessionDB
from hermes_state_common import FTS_STORAGE_VERSION, SCHEMA_VERSION
from hermes_state_repair import _db_opens_cleanly


ProgressCallback = Callable[[dict[str, Any]], None]
_CANONICAL_TABLES = (
    "system_prompts", "sessions", "messages", "session_model_usage", "compression_locks", "gateway_routing",
    "async_delegations",
)
_TOPIC_TABLES = ("telegram_dm_topic_mode", "telegram_dm_topic_bindings")


def _init_delivery_ledger_schema(conn: sqlite3.Connection) -> None:
    from gateway.delivery_ledger import _initialize_schema
    _initialize_schema(conn)


# state.db tables created lazily by a gateway module (base ``SessionDB`` never creates them on a fresh
# destination) -> the initializer owning their DDL. Recovery creates them before copying so owed rows
# don't silently vanish from a "complete" salvage. Register new lazy tables HERE, not as ``if table ==``.
# See #100313, #86236.
_AUXILIARY_TABLE_SCHEMAS: dict[str, Callable[[sqlite3.Connection], None]] = {
    "delivery_obligations": _init_delivery_ledger_schema,
}
_AUXILIARY_TABLES = tuple(_AUXILIARY_TABLE_SCHEMAS)
_INVENTORY_TABLES = (*_CANONICAL_TABLES, "state_meta", *_TOPIC_TABLES, *_AUXILIARY_TABLES)

# Derived-index / optional-schema markers: a fresh destination regenerates these, never copies them.
_GENERATED_META_KEYS = frozenset({
    "fts_storage_version", "fts_optimize_available", "fts_rebuild_high_water", "fts_rebuild_progress",
    "fts_cjk_stale", "fts_cjk_rebuild_high_water", "fts_cjk_rebuild_progress", "telegram_dm_topic_schema_version",
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
    return path.expanduser().parent.resolve(strict=True) / path.name


def _validate_paths(
    source_path: Path, output_path: Optional[Path] = None, work_dir: Optional[Path] = None,
) -> tuple[Path, Optional[Path], Path]:
    source = source_path.expanduser().resolve(strict=True)
    if not source.is_file():
        raise SessionRecoverySafetyError(f"Source is not a file: {source}")
    output: Optional[Path] = None
    if output_path is not None:
        output = _resolved_output_path(output_path)
        protected = {_sidecar_path(source, suffix).resolve(strict=False) for suffix in _SIDECAR_SUFFIXES}
        if output.resolve(strict=False) in protected:
            raise SessionRecoverySafetyError(
                "The recovery output must not be the source database or one of its journal sidecars."
            )
        for suffix in _SIDECAR_SUFFIXES:
            candidate = _sidecar_path(output, suffix)
            if os.path.lexists(candidate):
                raise SessionRecoverySafetyError(f"Refusing to overwrite existing recovery output: {candidate}")
    work_root = (
        work_dir.expanduser().resolve(strict=True)
        if work_dir is not None
        else (output.parent if output is not None else source.parent)
    )
    if not work_root.is_dir():
        raise SessionRecoverySafetyError(f"Recovery work directory is not a directory: {work_root}")
    return source, output, work_root


def _source_fingerprint(source: Path) -> dict[str, dict[str, int]]:
    fingerprint: dict[str, dict[str, int]] = {}
    for suffix in _SIDECAR_SUFFIXES:
        path = _sidecar_path(source, suffix)
        if path.exists():
            stat = path.stat()
            fingerprint[suffix or "main"] = {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns}
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
    except OSError:  # both dirs exist (_validate_paths); fallback for incomplete st_dev support
        return left.anchor.casefold() == right.anchor.casefold()


def _disk_space_preflight(source: Path, work_root: Path, output_parent: Optional[Path]) -> dict[str, Any]:
    """Require space for the disposable bundle, output, and safety headroom."""
    bundle_bytes = sum(info["size"] for info in _source_fingerprint(source).values())
    # The v23 external-content rebuild is usually much smaller than a legacy database, but estimating
    # with the complete bundle avoids betting the user's disk on that.
    output_allowance = bundle_bytes if output_parent is not None else 0
    headroom = max(_MINIMUM_SPACE_HEADROOM, int((bundle_bytes + output_allowance) * 0.05))
    work_free = int(shutil.disk_usage(work_root).free)
    report: dict[str, Any] = {
        "source_bundle_bytes": bundle_bytes, "estimated_output_bytes": output_allowance, "headroom_bytes": headroom,
        "work_dir": str(work_root), "work_dir_free_bytes": work_free,
    }
    if output_parent is None or _same_filesystem(work_root, output_parent):
        required = bundle_bytes + output_allowance + headroom
        report.update(shared_filesystem=True, work_dir_required_bytes=required)
        if work_free < required:
            raise SessionRecoverySafetyError(
                f"Not enough free disk space for a safe recovery copy: {_format_bytes(work_free)} available at "
                f"{work_root}, {_format_bytes(required)} required ({_format_bytes(bundle_bytes)} source bundle + "
                f"{_format_bytes(output_allowance)} output allowance + {_format_bytes(headroom)} headroom). "
                "Use --work-dir or --output on a filesystem with more free space."
            )
        return report
    output_free = int(shutil.disk_usage(output_parent).free)
    work_required = bundle_bytes + headroom
    output_required = output_allowance + headroom
    report.update({
        "shared_filesystem": False, "work_dir_required_bytes": work_required, "output_dir": str(output_parent),
        "output_dir_free_bytes": output_free, "output_dir_required_bytes": output_required,
    })
    shortages = [
        f"{where}: {_format_bytes(free)} available, {_format_bytes(required)} required"
        for where, free, required in (
            (work_root, work_free, work_required), (output_parent, output_free, output_required),
        )
        if free < required
    ]
    if shortages:
        raise SessionRecoverySafetyError(
            f"Not enough free disk space for safe recovery: {'; '.join(shortages)}. "
            "Choose work/output filesystems with more free space."
        )
    return report


def _copy_source_bundle(source: Path, snapshot_dir: Path) -> tuple[Path, list[str]]:
    """Copy the source DB bundle aside so SQLite never opens the original.

    The whole copy runs inside ``offline_file_access`` (holds the connection-lifecycle lock). Recovery
    normally runs as its own CLI process against an offline file, so the refusal should never fire; the
    guard keeps this path consistent with ``hermes_state_repair._backup_db_file``.

    Checking for a live connection and *then* copying would be a check/use race: a connection could open in
    that window, and the copy's ``close()`` would cancel its POSIX advisory locks -- the failure class
    ``hermes_cli.sqlite_safe_read`` exists to prevent (see #71724). Holding the lock means no connection can
    appear mid-copy, across the main file and every sidecar.
    """
    from hermes_cli.sqlite_safe_read import LiveConnectionError, offline_file_access
    snapshot_source = snapshot_dir / source.name
    copied: list[str] = []
    try:
        with offline_file_access(source, what="snapshot"):
            for suffix in _SIDECAR_SUFFIXES:
                source_part = _sidecar_path(source, suffix)
                if source_part.exists():
                    destination_part = _sidecar_path(snapshot_source, suffix)
                    shutil.copy2(source_part, destination_part)
                    copied.append(destination_part.name)
    except LiveConnectionError as exc:
        raise SessionRecoverySafetyError(str(exc)) from exc
    return snapshot_source, copied


def _table_columns(conn: sqlite3.Connection, table: str) -> list[str]:
    return [str(row[1]) for row in conn.execute(f'PRAGMA table_info("{table}")')]


def _count_rows(conn: sqlite3.Connection, table: str) -> int:
    return int(conn.execute(f'SELECT COUNT(*) FROM "{table}"').fetchone()[0])


def _connect(path: Path) -> sqlite3.Connection:
    """Autocommit connection with a short busy timeout (source snapshot or fresh output)."""
    return sqlite3.connect(str(path), isolation_level=None, timeout=1.0)


@contextmanager
def _immediate_transaction(conn: sqlite3.Connection) -> Iterator[None]:
    """``BEGIN IMMEDIATE`` ... ``COMMIT``, rolling back on any exception."""
    conn.execute("BEGIN IMMEDIATE")
    try:
        yield
        conn.execute("COMMIT")
    except BaseException:
        conn.execute("ROLLBACK")
        raise


def _compatible_columns(
    source: sqlite3.Connection, destination: sqlite3.Connection, table: str, result: dict[str, Any],
) -> Optional[list[str]]:
    """Columns shared by source and destination; sets a terminal status and returns None otherwise."""
    source_columns = _table_columns(source, table)
    columns = [column for column in _table_columns(destination, table) if column in source_columns]
    result["columns"] = columns
    if not source_columns:
        result["status"] = "missing"
        return None
    if not columns:
        result["status"] = "failed"
        result["error"] = "source and destination have no compatible columns"
        return None
    return columns


def _quoted_columns(columns: list[str]) -> tuple[str, str]:
    """``("a", "b")`` column list and matching ``?, ?`` placeholders."""
    return ", ".join(f'"{column}"' for column in columns), ", ".join("?" for _ in columns)


def _copy_rows(
    source: sqlite3.Connection, destination: sqlite3.Connection, select_sql: str, params: tuple[Any, ...],
    insert_sql: str, *, table: str, chunk_size: int, progress_cb: Optional[ProgressCallback],
    expected_rows: Optional[int], result: dict[str, Any],
) -> dict[str, Any]:
    """Chunked straight copy; fills ``status``/``error`` on ``result``."""
    try:
        cursor = source.execute(select_sql, params)
        while True:
            rows = cursor.fetchmany(chunk_size)
            if not rows:
                break
            with _immediate_transaction(destination):
                destination.executemany(insert_sql, rows)
            result["copied_rows"] += len(rows)
            if progress_cb is not None:
                progress_cb({"table": table, "copied_rows": result["copied_rows"], "source_rows": expected_rows})
    except sqlite3.DatabaseError as exc:
        result["status"] = "partial" if result["copied_rows"] else "failed"
        result["error"] = str(exc)
        return result
    if expected_rows is None or result["copied_rows"] == expected_rows:
        result["status"] = "complete"
    else:
        result["status"] = "partial"
        result["error"] = f"copied {result['copied_rows']} of {expected_rows} readable rows"
    return result


def _table_inventory(conn: sqlite3.Connection, table: str) -> dict[str, Any]:
    result: dict[str, Any] = {"available": False, "columns": [], "rows": None}
    try:
        columns = _table_columns(conn, table)
        if not columns:
            return result
        result.update(available=True, columns=columns)
        result["rows"] = _count_rows(conn, table)
    except sqlite3.DatabaseError as exc:
        result["error"] = str(exc)
    return result


def _journal_mode(conn: sqlite3.Connection) -> Optional[str]:
    row = conn.execute("PRAGMA journal_mode").fetchone()
    return str(row[0]).lower() if row else None


def _inspect_connection(conn: sqlite3.Connection) -> dict[str, Any]:
    conn.execute("PRAGMA writable_schema=ON")
    report: dict[str, Any] = {"tables": {}, "errors": [], "warnings": []}
    try:
        report["journal_mode"] = _journal_mode(conn)
    except sqlite3.DatabaseError as exc:
        report["journal_mode"] = None
        # Journal metadata is context, not canonical data: a damaged pragma must not block readable rows.
        report["warnings"].append(f"journal mode: {exc}")
    report["tables"] = {table: _table_inventory(conn, table) for table in _INVENTORY_TABLES}
    for required in ("sessions", "messages"):
        table_report = report["tables"][required]
        if not table_report.get("available") or table_report.get("rows") is None:
            report["errors"].append(f"required table {required} is not completely readable")
    report["recoverable"] = not report["errors"]
    return report


def _snapshot_and_inspect(
    source: Path, work_root: Path,
) -> tuple[tempfile.TemporaryDirectory[str], Path, dict[str, Any]]:
    before = _source_fingerprint(source)
    temp_dir = tempfile.TemporaryDirectory(prefix="hermes-session-recovery-", dir=str(work_root))
    try:
        snapshot_source, copied = _copy_source_bundle(source, Path(temp_dir.name))
        if _source_fingerprint(source) != before:
            raise SessionRecoverySafetyError(
                "The source database bundle changed while it was being copied. Stop every Hermes process using this "
                "profile and retry. This includes the interactive `hermes` CLI session this command may have been "
                "launched from: a running parent CLI writes session bookkeeping (compression ticks, context "
                "tracking) to state.db in the background and counts as a Hermes process even after the gateway is "
                "stopped. Run the recovery from a fresh shell with no `hermes` session open, or point --source at an "
                "immutable snapshot copy of the database."
            )
        conn = _connect(snapshot_source)
        try:
            inspection = _inspect_connection(conn)
        finally:
            conn.close()
        inspection.update(source_bundle=copied, source_fingerprint=before)
        return temp_dir, snapshot_source, inspection
    except BaseException:
        temp_dir.cleanup()
        raise


def inspect_session_database(source_path: Path, *, work_dir: Optional[Path] = None) -> dict[str, Any]:
    """Inspect canonical table readability without opening the source itself."""
    source, _, work_root = _validate_paths(source_path, work_dir=work_dir)
    disk_space = _disk_space_preflight(source, work_root, None)
    temp_dir, _, inspection = _snapshot_and_inspect(source, work_root)
    try:
        return {
            "operation": "inspect", "source": str(source), "disk_space": disk_space, **inspection,
            "source_unchanged": _source_fingerprint(source) == inspection["source_fingerprint"],
        }
    finally:
        temp_dir.cleanup()


def _fresh_destination(output: Path, *, topic_tables: bool = False) -> sqlite3.Connection:
    """Initialize a current-schema database at ``output`` and open it with foreign keys off."""
    with SessionDB(db_path=output) as destination_db:
        if topic_tables:
            destination_db.apply_telegram_topic_migration()
    conn = _connect(output)
    conn.execute("PRAGMA foreign_keys=OFF")
    return conn


def _copy_table(
    source: sqlite3.Connection, destination: sqlite3.Connection, table: str, *, salvage: bool, chunk_size: int,
    progress_cb: Optional[ProgressCallback], source_rows: Optional[int],
) -> dict[str, Any]:
    """Copy one canonical table: straight chunked copy, or rowid-range salvage when ``salvage``."""
    copy_kwargs = dict(chunk_size=chunk_size, progress_cb=progress_cb, source_rows=source_rows)
    if table == "state_meta":
        return _copy_state_meta(source, destination, salvage=salvage, **copy_kwargs)
    if salvage:
        return _copy_table_salvage(source, destination, table, **copy_kwargs)
    result: dict[str, Any] = {"source_rows": source_rows, "copied_rows": 0}
    columns = _compatible_columns(source, destination, table, result)
    if columns is None:
        return result
    quoted, placeholders = _quoted_columns(columns)
    return _copy_rows(
        source, destination, f'SELECT {quoted} FROM "{table}"', (),
        f'INSERT INTO "{table}" ({quoted}) VALUES ({placeholders})', table=table, chunk_size=chunk_size,
        progress_cb=progress_cb, expected_rows=source_rows, result=result,
    )


def _append_skipped_range(ranges: list[dict[str, Any]], low: int, high: int, error: str) -> None:
    """Record skipped rowid ranges, merging adjacent same-error ranges (not one entry per row)."""
    if ranges and ranges[-1]["high"] + 1 == low and ranges[-1]["error"] == error:
        ranges[-1]["high"] = high
        return
    ranges.append({"low": low, "high": high, "error": error})


def _salvage_rowid_bounds(source: sqlite3.Connection, table: str) -> dict[str, Any]:
    """Find the readable rowid edges without scanning the complete table."""
    result: dict[str, Any] = {"errors": [], "fallback_edges": []}
    rows: dict[str, Optional[int]] = {"low": None, "high": None}
    for edge, direction in (("low", "ASC"), ("high", "DESC")):
        try:
            row = source.execute(f'SELECT rowid FROM "{table}" ORDER BY rowid {direction} LIMIT 1').fetchone()
            if row is not None:
                rows[edge] = int(row[0])
        except sqlite3.DatabaseError as exc:
            result["errors"].append(f"{edge} rowid: {exc}")
    if rows["low"] is None and rows["high"] is None:
        result["empty" if not result["errors"] else "unavailable"] = True
        return result

    # A damaged edge can stop one ordered probe. Keep the readable edge and bound the other side by the
    # SQLite rowid domain, so bisection never assumes user databases hold only positive ids.
    if rows["low"] is None:
        rows["low"] = _MIN_SQLITE_ROWID
        result["fallback_edges"].append("low")
    if rows["high"] is None:
        rows["high"] = _MAX_SQLITE_ROWID
        result["fallback_edges"].append("high")
    result.update(rows)
    # Bisecting the whole synthetic domain tail used to exhaust the range-query budget before readable
    # tail rows were copied (#80205); gallop outward from the surviving edge for a finite bound first.
    if result["fallback_edges"]:
        result["edge_probes"] = []
        for edge, anchor_edge in (("high", "low"), ("low", "high")):
            if edge in result["fallback_edges"]:
                probe = _probe_populated_edge(source, table, edge=edge, anchor=int(result[anchor_edge]))
                result["edge_probes"].append(probe)
                if probe["capped"]:
                    result[edge] = int(probe["bound"])
    return result


def _probe_populated_edge(source: sqlite3.Connection, table: str, *, edge: str, anchor: int) -> dict[str, Any]:
    """Finite bound for a damaged rowid edge: gallop outward from the readable ``anchor`` with doubling
    offsets. A clean "no rows beyond X" caps the domain; an error or a hit keeps growing (~64 probes max).
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
        if (ascending and candidate >= domain_limit) or (not ascending and candidate <= domain_limit):
            result["bound"] = domain_limit  # no clean empty-tail answer: keep the domain fallback
            return result
        result["probes"] += 1
        try:
            row = source.execute(probe_sql, (candidate,)).fetchone()
        except sqlite3.DatabaseError:  # damage on the probe path: inconclusive, widen further
            span *= 2
            continue
        if row is None:  # nothing beyond candidate: the synthetic domain tail is provably empty
            result.update(bound=candidate, capped=True)
            return result
        position = int(row[0])  # rows exist beyond; advance. Span never resets -> O(log range)
        span *= 2


class _RowidRangeSalvage:
    """Bisecting rowid-range copy for one table; counters live on the shared ``result`` dict."""

    def __init__(
        self, source: sqlite3.Connection, destination: sqlite3.Connection, table: str, columns: list[str], *,
        chunk_size: int, progress_cb: Optional[ProgressCallback], source_rows: Optional[int], insert_prefix: str,
        row_filter: Optional[Callable[[tuple[Any, ...], tuple[str, ...]], bool]], result: dict[str, Any],
    ) -> None:
        self.source, self.destination, self.table = source, destination, table
        self.chunk_size, self.progress_cb, self.source_rows = chunk_size, progress_cb, source_rows
        self.row_filter, self.result = row_filter, result
        self.column_names = tuple(columns)
        quoted, placeholders = _quoted_columns(columns)
        self.select_sql = f'SELECT rowid, {quoted} FROM "{table}" WHERE rowid BETWEEN ? AND ? ORDER BY rowid'
        self.insert_sql = f'{insert_prefix} INTO "{table}" ({quoted}) VALUES ({placeholders})'
        self.exact_sql = f'SELECT {quoted} FROM "{table}" WHERE rowid = ?'
        self.stopped_at_query_limit = False

    def _keep(self, values: list[tuple[Any, ...]]) -> list[tuple[Any, ...]]:
        return values if self.row_filter is None else [r for r in values if self.row_filter(r, self.column_names)]

    def _skip(self, low: int, high: int, error: str) -> None:
        _append_skipped_range(self.result["skipped_rowid_ranges"], low, high, error)

    def recover_exact_rowid(self, rowid: int) -> bool:
        """Salvage one row by exact-key lookup (issue #80205).

        A singleton range scan (``rowid BETWEEN x AND x ORDER BY rowid``) must advance the cursor
        past ``x`` to prove the range is exhausted; when the *next* cell or page is damaged that
        advance raises AFTER the row was produced, and the driver discards the already-fetched row.
        """
        result = self.result
        result["range_queries"] += 1
        try:
            row = self.source.execute(self.exact_sql, (rowid,)).fetchone()
        except sqlite3.DatabaseError:
            return False
        if row is None:
            return True  # genuinely absent: nothing to skip
        value = tuple(row)
        if not self._keep([value]):
            result["excluded_rows"] += 1
            return True
        with _immediate_transaction(self.destination):
            self.destination.execute(self.insert_sql, value)
        result["copied_rows"] += 1
        result["exact_lookup_recovered"] += 1
        return True

    def copy_range(self, low: int, high: int) -> None:
        """Copy ``[low, high]``; on a read error bisect the unread remainder, exact-lookup singletons."""
        result = self.result
        if low > high:
            return
        if result["range_queries"] >= _MAX_SALVAGE_RANGE_QUERIES:
            self.stopped_at_query_limit = True
            self._skip(low, high, "salvage range query limit reached")
            return
        result["range_queries"] += 1
        last_committed_rowid: Optional[int] = None
        try:
            cursor = self.source.execute(self.select_sql, (low, high))
            while True:
                fetched = cursor.fetchmany(self.chunk_size)
                if not fetched:
                    return
                values = [tuple(row[1:]) for row in fetched]
                included = self._keep(values)
                if included:
                    with _immediate_transaction(self.destination):
                        self.destination.executemany(self.insert_sql, included)
                result["copied_rows"] += len(included)
                result["excluded_rows"] += len(values) - len(included)
                last_committed_rowid = int(fetched[-1][0])
                if self.progress_cb is not None:
                    self.progress_cb({
                        "table": self.table, "copied_rows": result["copied_rows"], "source_rows": self.source_rows,
                        "skipped_ranges": len(result["skipped_rowid_ranges"]),
                    })
        except sqlite3.DatabaseError as exc:
            retry_low = last_committed_rowid + 1 if last_committed_rowid is not None else low
            if retry_low > high:
                return
            if retry_low == high:
                if not self.recover_exact_rowid(retry_low):
                    self._skip(retry_low, high, str(exc))
                return
            midpoint = retry_low + (high - retry_low) // 2
            self.copy_range(retry_low, midpoint)
            self.copy_range(midpoint + 1, high)


def _copy_table_salvage(
    source: sqlite3.Connection, destination: sqlite3.Connection, table: str, *, chunk_size: int,
    progress_cb: Optional[ProgressCallback], source_rows: Optional[int], insert_prefix: str = "INSERT",
    row_filter: Optional[Callable[[tuple[Any, ...], tuple[str, ...]], bool]] = None,
) -> dict[str, Any]:
    """Best-effort rowid-range copy that continues past damaged source pages."""
    result: dict[str, Any] = {
        "mode": "rowid_range_salvage", "source_rows": source_rows, "copied_rows": 0, "excluded_rows": 0,
        "columns": [], "range_queries": 0, "exact_lookup_recovered": 0, "skipped_rowid_ranges": [],
    }
    columns = _compatible_columns(source, destination, table, result)
    if columns is None:
        return result
    bounds = _salvage_rowid_bounds(source, table)
    result["rowid_bounds"] = bounds
    if bounds.get("empty"):
        result["status"] = "complete"
        return result
    if bounds.get("low") is None or bounds.get("high") is None:
        details = "; ".join(bounds.get("errors") or [])
        result["status"] = "failed"
        result["error"] = "could not determine a rowid range for salvage" + (f": {details}" if details else "")
        return result
    salvage = _RowidRangeSalvage(
        source, destination, table, columns, chunk_size=chunk_size, progress_cb=progress_cb, source_rows=source_rows,
        insert_prefix=insert_prefix, row_filter=row_filter, result=result,
    )
    salvage.copy_range(int(bounds["low"]), int(bounds["high"]))
    skipped_ranges = result["skipped_rowid_ranges"]
    result["skipped_rowid_span"] = sum(item["high"] - item["low"] + 1 for item in skipped_ranges)
    result["query_limit_reached"] = salvage.stopped_at_query_limit
    if skipped_ranges:
        result["status"] = "partial" if result["copied_rows"] else "failed"
        result["error"] = f"{len(skipped_ranges)} rowid range(s) skipped"
    elif source_rows is not None and result["copied_rows"] + result["excluded_rows"] != source_rows:
        result["status"] = "partial"
        copied, excluded = result["copied_rows"], result["excluded_rows"]
        result["error"] = f"copied {copied} and excluded {excluded} of {source_rows} source rows"
    else:
        result["status"] = "complete"
    return result


def _state_meta_result(source_rows: Optional[int], **extra: Any) -> dict[str, Any]:
    return {
        "source_meta_rows": source_rows, "copied_rows": 0, "columns": ["key", "value"],
        "excluded_keys": sorted(_GENERATED_META_KEYS), **extra,
    }


def _state_meta_precheck(
    source: sqlite3.Connection, destination: sqlite3.Connection, source_rows: Optional[int], *, salvage: bool,
) -> Optional[dict[str, Any]]:
    """Terminal ``state_meta`` result when the key/value schema is unusable, else ``None``.

    In salvage mode an unusable-but-PRESENT table reports ``failed``, not ``missing``: verification
    only escalates ``failed``/``partial`` into a warning + ``loss_detected``, so ``missing`` would
    silently drop real metadata and still claim ``complete=True``.
    """
    extra = {"mode": "rowid_range_salvage"} if salvage else {}
    source_columns = _table_columns(source, "state_meta")
    if not {"key", "value"}.issubset(source_columns):
        if salvage and source_columns:  # present but unusable: real data loss
            found = ", ".join(source_columns) or "none"
            error = f"source state_meta exists but is missing the key/value columns (found: {found})"
            return _state_meta_result(source_rows, **extra, status="failed", error=error)
        return _state_meta_result(source_rows, **extra, status="missing")  # genuinely absent: nothing lost
    if not {"key", "value"}.issubset(_table_columns(destination, "state_meta")):
        error = "destination state_meta schema is incomplete"
        return _state_meta_result(source_rows, **extra, status="failed", error=error)
    return None


def _copy_state_meta(
    source: sqlite3.Connection, destination: sqlite3.Connection, *, salvage: bool, chunk_size: int,
    progress_cb: Optional[ProgressCallback], source_rows: Optional[int],
) -> dict[str, Any]:
    """Copy user metadata rows; derived FTS/topic keys (``_GENERATED_META_KEYS``) are regenerated, not copied."""
    problem = _state_meta_precheck(source, destination, source_rows, salvage=salvage)
    if problem is not None:
        return problem
    if salvage:
        def keep_user_meta(row: tuple[Any, ...], columns: tuple[str, ...]) -> bool:
            return str(row[columns.index("key")]) not in _GENERATED_META_KEYS
        result = _copy_table_salvage(
            source, destination, "state_meta", chunk_size=chunk_size, progress_cb=progress_cb,
            source_rows=source_rows, insert_prefix="INSERT OR REPLACE", row_filter=keep_user_meta,
        )
        result["source_meta_rows"] = result.pop("source_rows")
        result["excluded_keys"] = sorted(_GENERATED_META_KEYS)
        return result
    placeholders = ", ".join("?" for _ in _GENERATED_META_KEYS)
    params = tuple(_GENERATED_META_KEYS)
    filtered_source_rows: Optional[int] = None
    try:
        filtered_source_rows = int(
            source.execute(f"SELECT COUNT(*) FROM state_meta WHERE key NOT IN ({placeholders})", params).fetchone()[0]
        )
    except sqlite3.DatabaseError:
        pass  # the copy loop below will return the concrete read error
    return _copy_rows(
        source, destination, f"SELECT key, value FROM state_meta WHERE key NOT IN ({placeholders})", params,
        "INSERT OR REPLACE INTO state_meta(key, value) VALUES (?, ?)", table="state_meta", chunk_size=chunk_size,
        progress_cb=progress_cb, expected_rows=filtered_source_rows, result=_state_meta_result(source_rows),
    )


def _placeholder_titles(destination: sqlite3.Connection, prefix: str) -> Iterator[str]:
    """Yield ``[<prefix> N] session metadata was unreadable`` titles not yet present in ``sessions``."""
    sequence = 1
    while True:
        title = f"[{prefix} {sequence}] session metadata was unreadable"
        sequence += 1
        if destination.execute("SELECT 1 FROM sessions WHERE title = ? LIMIT 1", (title,)).fetchone() is None:
            yield title


def _reconstruct_missing_sessions(destination: sqlite3.Connection) -> dict[str, Any]:
    """Recreate placeholder session rows for salvaged orphaned messages.

    When ``sessions`` is damaged worse than ``messages``, deleting the recovered messages as orphans
    would discard the only readable copy of the user's data. Instead synthesize a minimal session per
    orphaned ``session_id`` (``started_at`` = earliest surviving message) so FKs hold.
    """
    result: dict[str, Any] = {"sessions_reconstructed": 0, "messages_retained": 0}
    if not (_table_columns(destination, "sessions") and _table_columns(destination, "messages")):
        return result
    orphaned = destination.execute(
        "SELECT m.session_id, MIN(m.timestamp), COUNT(*) FROM messages AS m WHERE m.session_id IS NOT NULL AND NOT "
        "EXISTS (SELECT 1 FROM sessions WHERE sessions.id = m.session_id) GROUP BY m.session_id"
    ).fetchall()
    if not orphaned:
        return result
    titles = _placeholder_titles(destination, "recovered")
    for session_id, first_timestamp, message_count in orphaned:
        started_at = float(first_timestamp) if first_timestamp is not None else 0.0
        title = next(titles)
        cursor = destination.execute(
            "INSERT INTO sessions (id, source, started_at, title, message_count) VALUES (?, 'recovered', ?, ?, ?)",
            (session_id, started_at, title, int(message_count)),
        )
        if cursor.rowcount != 1:
            raise sqlite3.IntegrityError(f"failed to reconstruct missing session {session_id!r}")
        result["sessions_reconstructed"] += 1
        result["messages_retained"] += int(message_count)
    return result


def _reconcile(destination: sqlite3.Connection, table: str, where: str, mutation: str) -> int:
    """Count rows of ``table`` matching ``where``; run ``mutation WHERE where`` only when there are any."""
    count = int(destination.execute(f'SELECT COUNT(*) FROM "{table}" WHERE {where}').fetchone()[0])
    if count:
        destination.execute(f"{mutation} WHERE {where}")
    return count


_DEPENDENT_TABLES = ("messages", "session_model_usage", "compression_locks", "telegram_dm_topic_bindings")
_RELINK_COUNTERS = ("session_prompt_refs_cleared", "sessions_parent_cleared")


def _cleanup_partial_orphans(destination: sqlite3.Connection) -> dict[str, Any]:
    """Reconcile references to unsalvageable sessions. Messages are never dropped for lack of a
    session row (placeholders are rebuilt first); only rows still orphaned after that are removed."""
    result: dict[str, Any] = {
        "session_prompt_refs_cleared": 0, "system_prompts_removed": 0, "sessions_parent_cleared": 0,
        "sessions_reconstructed": 0, "messages_retained": 0,
        **{f"{table}_removed": 0 for table in _DEPENDENT_TABLES},
    }
    with _immediate_transaction(destination):
        # Rebuild owners BEFORE any orphan deletion.
        result.update(_reconstruct_missing_sessions(destination))
        result["sessions_parent_cleared"] = _reconcile(
            destination, "sessions",
            "parent_session_id IS NOT NULL AND NOT EXISTS ("
            "SELECT 1 FROM sessions AS parent WHERE parent.id = sessions.parent_session_id)",
            "UPDATE sessions SET parent_session_id = NULL",
        )
        result["session_prompt_refs_cleared"] = _reconcile(
            destination, "sessions",
            "system_prompt_hash IS NOT NULL AND NOT EXISTS ("
            "SELECT 1 FROM system_prompts WHERE system_prompts.hash = sessions.system_prompt_hash)",
            "UPDATE sessions SET system_prompt_hash = NULL",
        )
        result["system_prompts_removed"] = _reconcile(
            destination, "system_prompts",
            "NOT EXISTS (SELECT 1 FROM sessions WHERE sessions.system_prompt_hash = system_prompts.hash)",
            "DELETE FROM system_prompts",
        )
        for table in _DEPENDENT_TABLES:
            if _table_columns(destination, table):
                result[f"{table}_removed"] = _reconcile(
                    destination, table,
                    f'NOT EXISTS (SELECT 1 FROM sessions WHERE sessions.id = "{table}".session_id)',
                    f'DELETE FROM "{table}"',
                )
    # Only destructive/relinking actions count: reconstruction counters describe RETAINED data.
    result["total_removed_or_relinked"] = sum(
        int(result[key]) for key in (*_RELINK_COUNTERS, *(f"{table}_removed" for table in _DEPENDENT_TABLES))
    )
    return result


def _verify_structure(conn: sqlite3.Connection, verification: dict[str, Any]) -> None:
    """Integrity, foreign keys, journal mode, schema version and FTS meta of the recovered file."""
    errors = verification["errors"]
    integrity_rows = [str(row[0]) for row in conn.execute("PRAGMA integrity_check").fetchall()]
    verification["integrity_check"] = integrity_rows
    if integrity_rows != ["ok"]:
        errors.append("PRAGMA integrity_check did not return exactly 'ok'")
    foreign_key_rows = [list(row) for row in conn.execute("PRAGMA foreign_key_check").fetchall()]
    verification["foreign_key_check"] = foreign_key_rows
    if foreign_key_rows:
        errors.append("foreign key violations remain")
    verification["journal_mode"] = _journal_mode(conn)
    schema_row = conn.execute("SELECT version FROM schema_version LIMIT 1").fetchone()
    verification["schema_version"] = int(schema_row[0]) if schema_row else None
    if verification["schema_version"] != SCHEMA_VERSION:
        errors.append(f"schema version is {verification['schema_version']}, expected {SCHEMA_VERSION}")
    meta = dict(conn.execute("SELECT key, value FROM state_meta WHERE key LIKE 'fts_%'").fetchall())
    meta = {str(key): value for key, value in meta.items()}
    verification["fts_meta"] = meta
    if meta.get("fts_storage_version") != str(FTS_STORAGE_VERSION):
        errors.append("fresh FTS storage version was not established")
    pending_keys = sorted(
        key for key in _GENERATED_META_KEYS if key.startswith("fts_") and key != "fts_storage_version" and key in meta
    )
    verification["pending_fts_keys"] = pending_keys
    if pending_keys:
        errors.append("derived FTS transition markers remain in the recovered database")


def _verify_fts_indexes(conn: sqlite3.Connection, verification: dict[str, Any]) -> None:
    fts_checks: dict[str, str] = {}
    for table in ("messages_fts", "messages_fts_trigram", "messages_fts_cjk"):
        if not _table_columns(conn, table):
            continue
        try:
            conn.execute(f'INSERT INTO "{table}" ("{table}") VALUES (\'integrity-check\')')
            conn.execute(f'SELECT 1 FROM "{table}" WHERE "{table}" MATCH \'""\' LIMIT 1').fetchone()
            fts_checks[table] = "ok"
        except sqlite3.DatabaseError as exc:
            fts_checks[table] = str(exc)
            verification["errors"].append(f"{table} integrity check failed: {exc}")
    verification["fts_checks"] = fts_checks


def _verify_row_counts(
    conn: sqlite3.Connection, verification: dict[str, Any], *, expected_counts: dict[str, Optional[int]],
    copy_report: dict[str, dict[str, Any]], allow_partial: bool, orphan_cleanup: Optional[dict[str, Any]],
) -> None:
    """Compare recovered counts and copy statuses against the source; classify shortfalls as loss."""

    def flag(message: str, *, soft: bool) -> None:
        """Record data loss as a warning (``soft``) or as a verification error."""
        if soft:
            verification["warnings"].append(message)
            verification["loss_detected"] = True
        else:
            verification["errors"].append(message)
    counts = {table: _count_rows(conn, table) for table in _INVENTORY_TABLES if _table_columns(conn, table)}
    verification["table_counts"] = counts
    for table in ("sessions", "messages", *_AUXILIARY_TABLES):
        expected = expected_counts.get(table)
        if expected is not None and counts.get(table) != expected:
            flag(f"{table} count is {counts.get(table)}, expected {expected}", soft=allow_partial)
    cleanup = orphan_cleanup or {}
    rebuilt_sessions = int(cleanup.get("sessions_reconstructed") or 0)
    retained_messages = int(cleanup.get("messages_retained") or 0)
    removed_messages = int(cleanup.get("messages_removed") or 0)
    # A wholly unreadable sessions b-tree is recoverable when every output parent was rebuilt from
    # the surviving messages and none were dropped: data loss, but not structural failure.
    sessions_fully_reconstructed = bool(
        rebuilt_sessions > 0 and counts.get("sessions") == rebuilt_sessions
        and counts.get("messages") == retained_messages and removed_messages == 0
    )
    for table, table_report in copy_report.items():
        status = table_report.get("status")
        if status not in {"failed", "partial"}:
            continue
        tolerable = (
            status == "partial" or table not in {"sessions", "messages"}
            or (table == "sessions" and sessions_fully_reconstructed)
        )
        flag(f"{table} copy status is {status}", soft=allow_partial and tolerable)
    if orphan_cleanup:
        orphan_count = int(orphan_cleanup.get("total_removed_or_relinked") or 0)
        if orphan_count:
            flag(f"{orphan_count} orphaned reference(s) were removed or relinked", soft=True)
        if rebuilt_sessions:
            # Not a clean recovery: the conversation text survived but its session metadata did not.
            flag(
                f"{rebuilt_sessions} session(s) could not be salvaged and were reconstructed as placeholders to "
                f"retain {retained_messages} message(s); their metadata (title, model, timestamps, cost) is lost",
                soft=True,
            )


def _verify_recovered_database(
    output: Path, *, expected_counts: dict[str, Optional[int]], copy_report: dict[str, dict[str, Any]],
    allow_partial: bool = False, orphan_cleanup: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    verification: dict[str, Any] = {"errors": [], "warnings": [], "loss_detected": False}
    open_error = _db_opens_cleanly(output)
    verification["opens_cleanly"] = open_error is None
    if open_error is not None:
        verification["errors"].append(f"database health probe: {open_error}")
    conn = sqlite3.connect(str(output), isolation_level=None)
    try:
        _verify_structure(conn, verification)
        _verify_row_counts(
            conn, verification, expected_counts=expected_counts, copy_report=copy_report, allow_partial=allow_partial,
            orphan_cleanup=orphan_cleanup,
        )
        _verify_fts_indexes(conn, verification)
    except sqlite3.DatabaseError as exc:
        verification["errors"].append(f"verification query failed: {exc}")
    finally:
        conn.close()
    verification["healthy"] = not verification["errors"]
    verification["complete"] = bool(verification["healthy"] and not verification["loss_detected"])
    return verification


def _finalize_derived_metadata(destination: sqlite3.Connection) -> dict[str, Any]:
    """Stamp only metadata that the newly created destination actually owns."""
    fts_tables = {
        str(row[0])
        for row in destination.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name IN ('messages_fts', 'messages_fts_trigram')"
        ).fetchall()
    }
    result: dict[str, Any] = {"fts_tables": sorted(fts_tables), "finalized": False}
    if fts_tables != {"messages_fts", "messages_fts_trigram"}:
        result["error"] = "fresh destination is missing required FTS tables"
        return result
    fts_keys = tuple(key for key in _GENERATED_META_KEYS if key.startswith("fts_"))
    placeholders = ", ".join("?" for _ in fts_keys)
    with _immediate_transaction(destination):
        destination.execute(f"DELETE FROM state_meta WHERE key IN ({placeholders})", fts_keys)
        destination.execute(
            "INSERT INTO state_meta(key, value) VALUES (?, ?) ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            ("fts_storage_version", str(FTS_STORAGE_VERSION)),
        )
    result["finalized"] = True
    return result


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
    mapped row violates the epoch floor, the mapping was wrong.

    Stub rows written by ``stub_missing_parent_sessions`` legitimately carry
    ``started_at = 0.0`` when no timestamped message survived, so they are
    excluded from the denominator.
    """
    from hermes_cli.session_lost_and_found import _EPOCH_LOW, STUB_TITLE_PREFIX

    errors: list[str] = []
    checks = (
        ("sessions", "started_at", f"WHERE COALESCE(title, '') NOT LIKE '{STUB_TITLE_PREFIX}%'"),
        ("messages", "timestamp", ""),
    )
    for table, column, mapped_filter in checks:
        (total,) = conn.execute(f"SELECT COUNT(*) FROM {table} {mapped_filter}").fetchone()
        if not total:
            continue
        (implausible,) = conn.execute(
            f"SELECT COUNT(*) FROM {table} {mapped_filter} "
            f"{'AND' if mapped_filter else 'WHERE'} ({column} IS NULL OR {column} < ?)",
            (_EPOCH_LOW,),
        ).fetchone()
        if implausible == total:
            errors.append(
                f"{table}.{column} is implausible in all {total} salvaged row(s) "
                "(NULL or before 2001-09): the source's physical column order "
                "did not match the destination template, so cells were mapped "
                "onto the wrong columns"
            )
    return errors


def _recover_via_lost_and_found(
    *, source: Path, snapshot_source: Path, snapshot_dir: Path, output: Path, inspection: dict[str, Any],
    disk_space: dict[str, Any], missing_required: list[str],
) -> dict[str, Any]:
    """Best-effort page-level salvage when table schemas are unreadable: the sqlite3 CLI's ``.recover``
    (shell-only, not in Python's ``sqlite3``) rebuilds rows into a scratch lost_and_found database which
    is then heuristically mapped into a fresh current-schema database."""
    from hermes_cli.session_lost_and_found import (
        SQLITE3_CLI_GUIDANCE, LostAndFoundError, find_sqlite3_cli, find_sqlite3_cli_refusal, map_lost_and_found_rows,
        rebuild_fts_indexes, run_cli_lost_and_found_recover, stub_missing_parent_sessions,
    )
    missing = ", ".join(missing_required)
    sqlite3_bin = find_sqlite3_cli()
    if sqlite3_bin is None:
        refusal = find_sqlite3_cli_refusal()
        if refusal.get("reason") == "wal_reset_vulnerable":
            raise SessionRecoverySourceError(
                "Partial recovery requires a page-level salvage shell, but "
                "the only sqlite3 CLI on PATH is not safe to use for it: it "
                + refusal["detail"]
                + ". The readable table schemas for: "
                + ", ".join(missing_required)
                + " are still required."
            )
        raise SessionRecoverySourceError(
            f"Partial recovery still requires readable table schemas for: {missing}. {SQLITE3_CLI_GUIDANCE}"
        )
    lf_path = snapshot_dir / "lost_and_found.db"
    try:
        cli_report = run_cli_lost_and_found_recover(snapshot_source, lf_path, sqlite3_bin)
    except (LostAndFoundError, OSError) as exc:
        raise SessionRecoverySourceError(
            f"Partial recovery could not read the table schemas for: {missing}, "
            f"and page-level .recover salvage failed: {exc}"
        ) from exc
    lf_conn = sqlite3.connect(str(lf_path), isolation_level=None)
    destination_conn = _fresh_destination(output)
    try:
        mapping = map_lost_and_found_rows(lf_conn, destination_conn)
        stubbing = stub_missing_parent_sessions(destination_conn)
        fts = rebuild_fts_indexes(destination_conn)
        derived_metadata = _finalize_derived_metadata(destination_conn)
    finally:
        lf_conn.close()
        destination_conn.close()
    copy_report: dict[str, dict[str, Any]] = {
        table: {
            "mode": "lost_and_found_salvage", "status": "partial",
            "copied_rows": int(mapping["direct_table_rows"].get(table) or 0) + int(mapping["mapped"].get(table) or 0),
            "error": "recovered via page-level lost_and_found salvage; "
            "row completeness cannot be verified against the source",
        }
        for table in ("sessions", "messages", "session_model_usage")
    }
    orphan_cleanup = {
        "sessions_reconstructed": stubbing["sessions_stubbed"], "messages_retained": stubbing["messages_retained"],
        "messages_removed": 0, "total_removed_or_relinked": 0,
    }
    verification = _verify_recovered_database(
        output, expected_counts={"sessions": None, "messages": None}, copy_report=copy_report, allow_partial=True,
        orphan_cleanup=orphan_cleanup,
    )
    verification["warnings"].append(
        "BEST-EFFORT page-level salvage: the source table schemas were unreadable, so rows were rebuilt from raw "
        "pages and mapped heuristically. Review every count before trusting this output."
    )
    verification.update(loss_detected=True, complete=False)
    # Structural checks cannot see a positional mis-mapping: every row still inserts, so integrity/FK/FTS
    # stay green. A systematic timestamp violation is the semantic tell — never report such a salvage as verified.
    # See #101409.
    plausibility_conn = sqlite3.connect(str(output), isolation_level=None)
    try:
        plausibility_errors = _lost_and_found_plausibility_errors(plausibility_conn)
    finally:
        plausibility_conn.close()
    if plausibility_errors:
        verification["errors"].extend(plausibility_errors)
        verification["healthy"] = False
    return _recovery_report(
        source, output, inspection, disk_space, verification, on_source_change="healthy",
        allow_partial=True, mode="lost_and_found_salvage", best_effort=True, unreadable_schemas=missing_required,
        sqlite3_cli=cli_report, lost_and_found=mapping, session_stubs=stubbing, fts_rebuild=fts, copy=copy_report,
        orphan_cleanup=orphan_cleanup, derived_metadata=derived_metadata,
    )


def _recovery_report(
    source: Path, output: Path, inspection: dict[str, Any], disk_space: dict[str, Any], verification: dict[str, Any],
    *, on_source_change: str, **fields: Any,
) -> dict[str, Any]:
    """The ``recover`` report: shared header, mode-specific ``fields``, then the verdict. A source bundle
    that changed during recovery is a verification error and also clears ``verification[on_source_change]``.
    """
    source_unchanged = _source_fingerprint(source) == inspection["source_fingerprint"]
    if not source_unchanged:
        verification["errors"].append("the source database bundle changed during recovery")
        verification[on_source_change] = False
    return {
        "operation": "recover", "source": str(source), "output": str(output),
        "source_bundle": inspection["source_bundle"], "source_fingerprint": inspection["source_fingerprint"],
        "source_unchanged": source_unchanged, "disk_space": disk_space,
        "inspection": {
            "journal_mode": inspection.get("journal_mode"), "tables": inspection["tables"],
            "errors": inspection["errors"], "warnings": inspection["warnings"],
        },
        **fields,
        "verification": verification,
        "complete": bool(verification.get("complete") and source_unchanged),
        "partial": bool(verification.get("loss_detected")),
        "verified": bool(verification.get("healthy") and source_unchanged),
        "installed": False,
    }


def recover_session_database(
    source_path: Path, output_path: Path, *, work_dir: Optional[Path] = None, chunk_size: int = 1_000,
    progress_cb: Optional[ProgressCallback] = None, allow_partial: bool = False,
) -> dict[str, Any]:
    """Recover canonical rows into a separate current-schema database. The source and its sidecars are
    copied before SQLite opens anything; ``output_path`` must not exist and is never swapped into place."""
    if chunk_size <= 0:
        raise SessionRecoverySafetyError("chunk_size must be greater than zero")
    source, output, work_root = _validate_paths(source_path, output_path=output_path, work_dir=work_dir)
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
        missing_required = [t for t in ("sessions", "messages") if not inspection["tables"][t].get("available")]
        if allow_partial and missing_required:  # no readable schema -> page-level lost_and_found salvage
            return _recover_via_lost_and_found(
                source=source, snapshot_source=snapshot_source, snapshot_dir=Path(temp_dir.name), output=output,
                inspection=inspection, disk_space=disk_space, missing_required=missing_required,
            )
        source_conn = _connect(snapshot_source)
        source_conn.execute("PRAGMA writable_schema=ON")
        destination_conn: Optional[sqlite3.Connection] = None
        try:
            destination_conn = _fresh_destination(
                output, topic_tables=any(inspection["tables"][table].get("available") for table in _TOPIC_TABLES),
            )
            copy_report: dict[str, dict[str, Any]] = {}
            for table in (*_CANONICAL_TABLES, "state_meta", *_TOPIC_TABLES, *_AUXILIARY_TABLES):
                if table not in _CANONICAL_TABLES and not inspection["tables"][table].get("available"):
                    copy_report[table] = {"status": "missing", "copied_rows": 0}
                    continue
                if table in _AUXILIARY_TABLES:  # lazy gateway table: create it or the copy reports "missing"
                    _AUXILIARY_TABLE_SCHEMAS[table](destination_conn)
                copy_report[table] = _copy_table(
                    source_conn, destination_conn, table, salvage=allow_partial, chunk_size=chunk_size,
                    progress_cb=progress_cb, source_rows=inspection["tables"][table].get("rows"),
                )
            orphan_cleanup = _cleanup_partial_orphans(destination_conn) if allow_partial else None
            derived_metadata = _finalize_derived_metadata(destination_conn)
        finally:
            source_conn.close()
            if destination_conn is not None:
                destination_conn.close()
        expected_counts = {
            table: inspection["tables"][table].get("rows")
            for table in (*_CANONICAL_TABLES, *_AUXILIARY_TABLES)
            if table in _CANONICAL_TABLES or inspection["tables"].get(table, {}).get("available")
        }
        verification = _verify_recovered_database(
            output, expected_counts=expected_counts, copy_report=copy_report, allow_partial=allow_partial,
            orphan_cleanup=orphan_cleanup,
        )
        return _recovery_report(
            source, output, inspection, disk_space, verification, on_source_change="complete",
            allow_partial=allow_partial, copy=copy_report, orphan_cleanup=orphan_cleanup,
            derived_metadata=derived_metadata,
        )
    finally:
        temp_dir.cleanup()


def write_recovery_report(path: Path, report: dict[str, Any]) -> Path:
    """Write a JSON report without overwriting an existing file."""
    destination = _resolved_output_path(path)
    with destination.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return destination
