"""Shared SQLite bootstrap helpers for hermes-lcm.

This module keeps startup DB initialization in one place so store/DAG use the
same schema-version marker, PRAGMA settings, and FTS repair behavior.
"""

from __future__ import annotations

import logging
import math
import os
import re
import shutil
import sqlite3
import time
from typing import Iterable, Sequence

logger = logging.getLogger(__name__)

SCHEMA_VERSION = 4
SQLITE_BUSY_TIMEOUT_MS = 30_000
_MIN_DISK_SPACE_BYTES = 50 * 1024 * 1024
REQUIRED_CORE_TABLES = (
    "messages",
    "metadata",
    "summary_nodes",
    "lcm_lifecycle_state",
    "lcm_migration_state",
    "messages_fts",
    "nodes_fts",
)


class ExternalContentFtsSpec:
    def __init__(
        self,
        *,
        table_name: str,
        content_table: str,
        content_rowid: str,
        indexed_column: str,
        trigger_sqls: Sequence[str],
    ) -> None:
        self.table_name = table_name
        self.content_table = content_table
        self.content_rowid = content_rowid
        self.indexed_column = indexed_column
        self.trigger_sqls = tuple(trigger_sqls)


def configure_connection(conn: sqlite3.Connection) -> None:
    """Configure SQLite connection for WAL durability and hygiene.

    In a multi-agent deployment (gateway process + CLI sessions + sub-agents),
    every process opens its own sqlite3.Connection pointing at the same
    lcm.db file.  These settings improve committed-write durability and WAL
    hygiene, but do NOT make sibling processes safe from an unexpected process
    death.  Abnormal exit still depends on normal SQLite WAL recovery;
    application-level checkpoints only run during graceful shutdown (see
    ``MessageStore.close()`` etc.).

    Key design decisions:
    - journal_mode=WAL  : writes go to a separate log; readers never block.
    - synchronous=FULL  : fsync both the WAL and the WAL index before every
                          write transaction commit.  WAL + FULL is the only
                          combination SQLite guarantees survives power loss
                          without data loss (NORMAL may lose the WAL index).
    - wal_autocheckpoint=500 : after 500 WAL pages (~2 MB) SQLite will try
                               an automatic passive checkpoint.  This is a
                               best-effort hint — it is silently skipped when
                               another connection holds a read transaction.
                               Under checkpoint starvation WAL can grow well
                               beyond this trigger.
    - journal_size_limit=67108864 (64 MiB) : limits the WAL file size after
                                             a successful checkpoint or reset.
                                             It does NOT force a checkpoint
                                             or cap growth while another
                                             connection holds an old WAL
                                             end mark.
    - mmap_size=268435456 (256 MiB)        : memory-map reads so concurrent
                                              readers cache WAL pages in RAM.
    """
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=FULL")
    conn.execute(f"PRAGMA busy_timeout={SQLITE_BUSY_TIMEOUT_MS}")
    conn.execute("PRAGMA wal_autocheckpoint=500")
    conn.execute("PRAGMA journal_size_limit=67108864")
    conn.execute("PRAGMA mmap_size=268435456")


def ensure_metadata_table(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS metadata (
            key TEXT PRIMARY KEY,
            value TEXT
        )
        """
    )


def get_schema_version(conn: sqlite3.Connection) -> int:
    ensure_metadata_table(conn)
    row = conn.execute(
        "SELECT value FROM metadata WHERE key = 'schema_version'"
    ).fetchone()
    if not row or row[0] is None:
        return 0
    try:
        return int(str(row[0]))
    except (TypeError, ValueError):
        return 0


def ensure_migration_state_table(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS lcm_migration_state (
            step_name TEXT PRIMARY KEY,
            completed_at REAL NOT NULL
        )
        """
    )


def ensure_lifecycle_state_table(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS lcm_lifecycle_state (
            conversation_id TEXT PRIMARY KEY,
            current_session_id TEXT,
            last_finalized_session_id TEXT,
            current_frontier_store_id INTEGER NOT NULL DEFAULT 0,
            last_finalized_frontier_store_id INTEGER NOT NULL DEFAULT 0,
            debt_kind TEXT,
            debt_size_estimate INTEGER NOT NULL DEFAULT 0,
            current_bound_at REAL,
            last_finalized_at REAL,
            debt_updated_at REAL,
            last_maintenance_attempt_at REAL,
            last_rollover_at REAL,
            last_reset_at REAL,
            updated_at REAL NOT NULL DEFAULT (strftime('%s','now'))
        )
        """
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_lcm_lifecycle_current_session ON lcm_lifecycle_state(current_session_id)"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_lcm_lifecycle_last_finalized_session ON lcm_lifecycle_state(last_finalized_session_id)"
    )


def ensure_lifecycle_state_columns(conn: sqlite3.Connection) -> None:
    ensure_lifecycle_state_table(conn)
    columns = {
        row[1] for row in conn.execute("PRAGMA table_info(lcm_lifecycle_state)").fetchall()
    }
    if "debt_kind" not in columns:
        conn.execute("ALTER TABLE lcm_lifecycle_state ADD COLUMN debt_kind TEXT")
    if "debt_size_estimate" not in columns:
        conn.execute(
            "ALTER TABLE lcm_lifecycle_state ADD COLUMN debt_size_estimate INTEGER NOT NULL DEFAULT 0"
        )
    if "debt_updated_at" not in columns:
        conn.execute("ALTER TABLE lcm_lifecycle_state ADD COLUMN debt_updated_at REAL")
    if "last_maintenance_attempt_at" not in columns:
        conn.execute(
            "ALTER TABLE lcm_lifecycle_state ADD COLUMN last_maintenance_attempt_at REAL"
        )
    if "last_rollover_at" not in columns:
        conn.execute("ALTER TABLE lcm_lifecycle_state ADD COLUMN last_rollover_at REAL")
    if "last_reset_at" not in columns:
        conn.execute("ALTER TABLE lcm_lifecycle_state ADD COLUMN last_reset_at REAL")


def mark_migration_step_complete(conn: sqlite3.Connection, step_name: str) -> None:
    ensure_migration_state_table(conn)
    conn.execute(
        """
        INSERT INTO lcm_migration_state(step_name, completed_at)
        VALUES(?, strftime('%s','now'))
        ON CONFLICT(step_name) DO UPDATE SET completed_at = excluded.completed_at
        """,
        (step_name,),
    )


def set_schema_version(conn: sqlite3.Connection, version: int = SCHEMA_VERSION) -> None:
    ensure_metadata_table(conn)
    conn.execute(
        """
        INSERT INTO metadata(key, value)
        VALUES('schema_version', ?)
        ON CONFLICT(key) DO UPDATE SET value = excluded.value
        """,
        (str(version),),
    )


def get_existing_table_names(conn: sqlite3.Connection, names: Iterable[str]) -> set[str]:
    existing: set[str] = set()
    for name in names:
        row = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name = ?",
            (name,),
        ).fetchone()
        if row and row[0]:
            existing.add(row[0])
    return existing


def _database_path_for_connection(conn: sqlite3.Connection | None, fallback: str = "") -> str:
    if conn is None:
        return fallback
    try:
        rows = conn.execute("PRAGMA database_list").fetchall()
    except sqlite3.DatabaseError:
        return fallback
    for row in rows:
        if len(row) >= 3 and row[1] == "main" and row[2]:
            return str(row[2])
    return fallback


def inspect_lcm_schema_health(
    conn: sqlite3.Connection | None,
    *,
    database_path: str = "",
    required_tables: Iterable[str] = REQUIRED_CORE_TABLES,
) -> dict[str, object]:
    """Return read-only health metadata for the core hermes-lcm SQLite schema."""
    required = tuple(required_tables)
    resolved_path = _database_path_for_connection(conn, database_path)
    detail: dict[str, object] = {
        "database_path": resolved_path,
        "required_tables": list(required),
        "existing_tables": [],
        "missing_tables": [],
    }
    if conn is None:
        detail["error"] = "LCM store connection is not initialized"
        return detail

    try:
        rows = conn.execute(
            """
            SELECT name
            FROM sqlite_master
            WHERE type='table'
            ORDER BY name
            """
        ).fetchall()
    except sqlite3.DatabaseError as exc:
        detail["error"] = str(exc)
        return detail

    existing = sorted(str(row[0]) for row in rows if row and row[0])
    existing_set = set(existing)
    missing = [name for name in required if name not in existing_set]
    detail["existing_tables"] = existing
    detail["missing_tables"] = missing
    return detail


def get_fts_shadow_table_names(table_name: str) -> list[str]:
    return [
        f"{table_name}_data",
        f"{table_name}_idx",
        f"{table_name}_docsize",
        f"{table_name}_config",
    ]


def quote_sql_identifier(identifier: str) -> str:
    if not identifier or not identifier.replace("_", "a").isalnum() or identifier[0].isdigit():
        raise ValueError(f"invalid SQL identifier: {identifier}")
    return f'"{identifier}"'


def _fts_needs_rebuild_structural(conn: sqlite3.Connection, spec: ExternalContentFtsSpec) -> bool:
    shadow_tables = get_fts_shadow_table_names(spec.table_name)
    existing_tables = get_existing_table_names(conn, [spec.table_name, *shadow_tables])
    if spec.table_name not in existing_tables:
        return True
    if any(name not in existing_tables for name in shadow_tables):
        return True

    try:
        info = conn.execute(
            "SELECT sql FROM sqlite_master WHERE type='table' AND name = ?",
            (spec.table_name,),
        ).fetchone()
        sql = (info[0] if info else "") or ""
        normalized = sql.lower()
        if "virtual table" not in normalized or "using fts5" not in normalized:
            return True

        columns = conn.execute(
            f"PRAGMA table_info({quote_sql_identifier(spec.table_name)})"
        ).fetchall()
        column_names = {row[1] for row in columns if len(row) > 1}
        if spec.indexed_column not in column_names:
            return True

        content_count = conn.execute(
            f"SELECT COUNT(*) FROM {quote_sql_identifier(spec.content_table)}"
        ).fetchone()[0]
        # For an external-content FTS5 table, ``COUNT(*) FROM <fts>`` reads
        # through to the content table (so it can never reveal a lagging index)
        # and is O(index size). The ``<fts>_docsize`` shadow table holds the
        # true indexed-document count and is a cheap ordinary-table count. Its
        # existence is already guaranteed by the shadow-table check above.
        docsize_table = f"{spec.table_name}_docsize"
        fts_count = conn.execute(
            f"SELECT COUNT(*) FROM {quote_sql_identifier(docsize_table)}"
        ).fetchone()[0]
        if int(content_count or 0) != int(fts_count or 0):
            return True
    except sqlite3.DatabaseError:
        return True

    return False


INTEGRITY_CHECK_INTERVAL_ENV = "LCM_FTS_INTEGRITY_CHECK_INTERVAL_HOURS"
DEFAULT_INTEGRITY_CHECK_INTERVAL_HOURS = 24.0


def _integrity_check_interval_hours() -> float:
    """Hours between startup FTS deep integrity-checks.

    ``0`` checks on every startup (previous behavior); a negative value never
    checks on startup (relies on structural checks + LIKE fallback + doctor).
    """
    raw = os.environ.get(INTEGRITY_CHECK_INTERVAL_ENV)
    if raw is None:
        return DEFAULT_INTEGRITY_CHECK_INTERVAL_HOURS
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return DEFAULT_INTEGRITY_CHECK_INTERVAL_HOURS
    if not math.isfinite(value):
        # nan/inf would suppress startup checks indefinitely once a marker
        # exists; treat non-finite values as invalid.
        return DEFAULT_INTEGRITY_CHECK_INTERVAL_HOURS
    return value


def _integrity_marker_key(spec: ExternalContentFtsSpec) -> str:
    return f"fts_integrity_checked_at:{spec.table_name}"


def _load_integrity_checked_at(
    conn: sqlite3.Connection, spec: ExternalContentFtsSpec
) -> float | None:
    ensure_metadata_table(conn)
    row = conn.execute(
        "SELECT value FROM metadata WHERE key = ?",
        (_integrity_marker_key(spec),),
    ).fetchone()
    if not row or row[0] is None:
        return None
    try:
        return float(row[0])
    except (TypeError, ValueError):
        return None


def _record_integrity_checked(
    conn: sqlite3.Connection, spec: ExternalContentFtsSpec, *, now: float | None = None
) -> None:
    ensure_metadata_table(conn)
    current = time.time() if now is None else now
    conn.execute(
        """
        INSERT INTO metadata(key, value)
        VALUES(?, ?)
        ON CONFLICT(key) DO UPDATE SET value = excluded.value
        """,
        (_integrity_marker_key(spec), str(current)),
    )


def _should_run_integrity_check(
    conn: sqlite3.Connection, spec: ExternalContentFtsSpec, *, now: float | None = None
) -> bool:
    hours = _integrity_check_interval_hours()
    if hours == 0:
        return True
    if hours < 0:
        return False
    last = _load_integrity_checked_at(conn, spec)
    if last is None:
        return True
    current = time.time() if now is None else now
    return (current - last) >= hours * 3600.0


def _fts_needs_rebuild(
    conn: sqlite3.Connection,
    spec: ExternalContentFtsSpec,
    *,
    now: float | None = None,
    throttle: bool = False,
) -> bool:
    if _fts_needs_rebuild_structural(conn, spec):
        return True
    # Structurally sound: the FTS5 integrity-check is O(index size) and was the
    # dominant startup cost on large databases (issue #235). On the startup path
    # (``throttle=True``) skip it when already checked within the interval.
    # Explicit repair (e.g. ``/lcm doctor repair apply``) uses ``throttle=False``
    # so it always runs the deep check and can fix same-row-count drift that the
    # structural checks cannot see.
    if throttle and not _should_run_integrity_check(conn, spec, now=now):
        return False
    result = check_external_content_fts_integrity(conn, spec)
    if result["status"] == "pass":
        _record_integrity_checked(conn, spec, now=now)
    return result["status"] == "fail"


def check_external_content_fts_integrity(
    conn: sqlite3.Connection,
    spec: ExternalContentFtsSpec,
) -> dict[str, str]:
    """Run SQLite's FTS5 integrity-check for an external-content table.

    FTS5 exposes this as a special INSERT command. Wrap it in a savepoint and
    roll it back so diagnostics can verify the index without leaving any state
    behind on the shared connection.
    """

    if _fts_needs_rebuild_structural(conn, spec):
        return {"status": "fail", "detail": "structural repair needed"}

    savepoint = f"lcm_fts_integrity_{spec.table_name}"
    savepoint_sql = quote_sql_identifier(savepoint)
    try:
        conn.execute(f"SAVEPOINT {savepoint_sql}")
        conn.execute(
            f"INSERT INTO {quote_sql_identifier(spec.table_name)}({quote_sql_identifier(spec.table_name)}, rank) VALUES('integrity-check', 1)"
        )
    except sqlite3.DatabaseError as exc:
        try:
            conn.execute(f"ROLLBACK TO {savepoint_sql}")
            conn.execute(f"RELEASE {savepoint_sql}")
        except sqlite3.DatabaseError:
            pass
        detail = str(exc)
        lowered = detail.lower()
        if "readonly" in lowered or "read-only" in lowered:
            return {"status": "unchecked", "detail": detail}
        return {"status": "fail", "detail": detail}

    try:
        conn.execute(f"ROLLBACK TO {savepoint_sql}")
        conn.execute(f"RELEASE {savepoint_sql}")
    except sqlite3.DatabaseError as exc:
        return {"status": "fail", "detail": str(exc)}

    return {"status": "pass", "detail": "ok"}


def _drop_fts_table(conn: sqlite3.Connection, table_name: str) -> None:
    conn.execute(f"DROP TABLE IF EXISTS {quote_sql_identifier(table_name)}")
    for shadow_name in get_fts_shadow_table_names(table_name):
        conn.execute(f"DROP TABLE IF EXISTS {quote_sql_identifier(shadow_name)}")


def _extract_trigger_name(trigger_sql: str) -> str | None:
    match = re.search(
        r"CREATE\s+TRIGGER\s+(?:IF\s+NOT\s+EXISTS\s+)?(?:\"([^\"]+)\"|([A-Za-z_][A-Za-z0-9_]*))",
        trigger_sql,
        re.IGNORECASE | re.DOTALL,
    )
    if not match:
        return None
    return match.group(1) or match.group(2)


def _drop_fts_triggers(conn: sqlite3.Connection, trigger_sqls: Sequence[str]) -> None:
    for trigger_sql in trigger_sqls:
        trigger_name = _extract_trigger_name(trigger_sql)
        if trigger_name:
            conn.execute(f"DROP TRIGGER IF EXISTS {quote_sql_identifier(trigger_name)}")


def _drop_fts_artifacts(conn: sqlite3.Connection, spec: ExternalContentFtsSpec) -> None:
    _drop_fts_triggers(conn, spec.trigger_sqls)
    _drop_fts_table(conn, spec.table_name)


def _check_disk_space(db_path: str) -> bool:
    try:
        parent = os.path.dirname(os.path.abspath(db_path)) or "."
        return shutil.disk_usage(parent).free >= _MIN_DISK_SPACE_BYTES
    except (OSError, AttributeError):
        return True


def _fts_missing_triggers(conn: sqlite3.Connection, spec: ExternalContentFtsSpec) -> bool:
    expected = {
        trigger_name
        for trigger_name in (_extract_trigger_name(sql) for sql in spec.trigger_sqls)
        if trigger_name
    }
    if not expected:
        return False
    placeholders = ",".join("?" for _ in expected)
    rows = conn.execute(
        f"SELECT name FROM sqlite_master WHERE type='trigger' AND name IN ({placeholders})",
        tuple(sorted(expected)),
    ).fetchall()
    existing = {str(row[0]) for row in rows if row and row[0]}
    return bool(expected - existing)


def _normalize_trigger_sql(sql: str) -> str:
    """Canonicalize a CREATE TRIGGER statement for content comparison.

    SQLite stores triggers with ``IF NOT EXISTS`` stripped and the trailing
    ``;`` removed, so a byte compare against the spec's source SQL always
    mismatches. Normalize both sides (drop ``IF NOT EXISTS``, collapse
    whitespace, strip the terminator, lowercase) so the only differences that
    survive are real ones — e.g. a trigger body that still references a renamed
    column.
    """
    text = sql or ""
    text = re.sub(r"(?i)\bIF\s+NOT\s+EXISTS\s+", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text.rstrip(";").strip().lower()


def _fts_stale_triggers(conn: sqlite3.Connection, spec: ExternalContentFtsSpec) -> bool:
    """True if a trigger exists by name but its body has drifted from the spec.

    ``_fts_missing_triggers`` only checks existence by name, so a trigger left
    over from an older schema (e.g. one that still inserts a since-renamed
    column) passes that check while aborting every write at runtime. Compare the
    stored trigger body against the canonical spec SQL to catch that drift.
    """
    for trigger_sql in spec.trigger_sqls:
        name = _extract_trigger_name(trigger_sql)
        if not name:
            continue
        row = conn.execute(
            "SELECT sql FROM sqlite_master WHERE type='trigger' AND name = ?",
            (name,),
        ).fetchone()
        if row is None or row[0] is None:
            # Missing entirely — handled by _fts_missing_triggers; not "stale".
            continue
        if _normalize_trigger_sql(str(row[0])) != _normalize_trigger_sql(trigger_sql):
            return True
    return False


def external_content_fts_needs_repair(conn: sqlite3.Connection, spec: ExternalContentFtsSpec) -> bool:
    return (
        _fts_needs_rebuild_structural(conn, spec)
        or _fts_missing_triggers(conn, spec)
        or _fts_stale_triggers(conn, spec)
    )


def repair_external_content_fts(
    conn: sqlite3.Connection,
    spec: ExternalContentFtsSpec,
    *,
    now: float | None = None,
    throttle: bool = False,
) -> dict[str, bool]:
    rebuilt = False
    degraded = False
    if _fts_needs_rebuild(conn, spec, now=now, throttle=throttle):
        db_path = conn.execute("PRAGMA database_list").fetchone()
        if db_path:
            db_file = db_path[2]
            if db_file and not _check_disk_space(db_file):
                logger.warning(
                    "Low disk space for FTS rebuild of '%s' (%d MB needed), degrading to LIKE search",
                    spec.table_name,
                    _MIN_DISK_SPACE_BYTES // (1024 * 1024),
                )
                _drop_fts_artifacts(conn, spec)
                conn.commit()
                return {"rebuilt": False, "degraded": True, "triggers_recreated": False}
        _drop_fts_table(conn, spec.table_name)
        conn.execute(
            f"""
            CREATE VIRTUAL TABLE {quote_sql_identifier(spec.table_name)} USING fts5(
                {quote_sql_identifier(spec.indexed_column)},
                content={quote_sql_identifier(spec.content_table)},
                content_rowid={quote_sql_identifier(spec.content_rowid)}
            )
            """
        )
        conn.execute(
            f"INSERT INTO {quote_sql_identifier(spec.table_name)}({quote_sql_identifier(spec.table_name)}) VALUES('rebuild')"
        )
        rebuilt = True

    triggers_were_missing = _fts_missing_triggers(conn, spec)
    triggers_were_stale = _fts_stale_triggers(conn, spec)
    if triggers_were_missing or triggers_were_stale:
        # A stale trigger exists by name but with a drifted body (e.g. it still
        # inserts a since-renamed column), so a bare ``CREATE TRIGGER IF NOT
        # EXISTS`` would no-op and leave the broken trigger in place. Drop first,
        # then recreate from the canonical spec so the body is guaranteed fresh.
        _drop_fts_triggers(conn, spec.trigger_sqls)
    for trigger_sql in spec.trigger_sqls:
        conn.execute(trigger_sql)
    if rebuilt:
        # A freshly rebuilt index is known-consistent; record the marker so the
        # next startup can skip the deep integrity-check within the interval.
        _record_integrity_checked(conn, spec, now=now)
    conn.commit()
    return {
        "rebuilt": rebuilt,
        "degraded": degraded,
        "triggers_recreated": triggers_were_missing or triggers_were_stale,
    }


def ensure_external_content_fts(
    conn: sqlite3.Connection, spec: ExternalContentFtsSpec, *, now: float | None = None
) -> None:
    # Startup path: throttle the deep integrity-check. Explicit repair callers
    # use ``repair_external_content_fts(..., throttle=False)`` for a forced check.
    repair_external_content_fts(conn, spec, now=now, throttle=True)


def run_versioned_migrations(conn: sqlite3.Connection) -> None:
    ensure_metadata_table(conn)
    ensure_migration_state_table(conn)

    current_version = get_schema_version(conn)
    if current_version < 2:
        mark_migration_step_complete(conn, "v2_external_content_fts_triggers")
        current_version = 2

    if current_version < 3:
        ensure_lifecycle_state_table(conn)
        mark_migration_step_complete(conn, "v3_lifecycle_state")
        current_version = 3
    else:
        ensure_lifecycle_state_table(conn)

    ensure_lifecycle_state_columns(conn)
    if current_version < 4:
        mark_migration_step_complete(conn, "v4_lifecycle_debt_columns")
        current_version = 4

    set_schema_version(conn, current_version)
