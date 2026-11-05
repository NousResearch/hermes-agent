"""The SQLite backend's state.db schema: ``_init_schema``, column reconciliation, the data-migration ladder
and the FTS setup the open and write paths run, over the DDL in ``_schema_ddl``.

The functions that touch the store take its holder first (``holder``), which provides the connection, the
write lock and the ``state_meta`` accessors they call.
"""

from __future__ import annotations

import contextlib
import datetime
import hashlib
import json
import logging
import os
import re
import sqlite3
import sys
import tempfile
import time
import uuid
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional, Sequence

from hermes_startup_watchdog import report_startup_progress
from utils import safe_json_loads
from hermes_state_common import (
    TITLE_SOURCE_DERIVED as _TITLE_SOURCE_DERIVED, TITLE_SOURCE_LLM as _TITLE_SOURCE_LLM, TITLE_SOURCE_USER as _TITLE_SOURCE_USER, escape_like as _escape_like, fts_rebuild_admission,
)
from hermes_state_errors import is_fts_scoped_corruption_error, is_sqlite_lock_error
from hermes_state_holders import _read_proc_argv

from ._schema_ddl import (
    DEFERRED_INDEX_SQL, FTS_CJK_STALE_KEY, FTS_CJK_TABLE_SQL, FTS_CJK_TRIGGER_SQL, FTS_REBUILD_DEFERRAL_KEY,
    FTS_SQL, FTS_STALE_KEY, FTS_STORAGE_VERSION, FTS_TRIGRAM_SQL, LEGACY_FTS_SQL, LEGACY_FTS_TRIGRAM_SQL,
    SCHEMA_SQL, SCHEMA_VERSION, _FTS_CJK_TRIGGERS, _FTS_TRIGGERS, _drop_orphan_fts_shadow_tables,
    _ephemeral_child_sql, _sql_json_extract, base_index_recognised, base_source_view_is_stale,
    fts5_cjk_so_path, fts_layout_current, fts_layout_drift,
)

if TYPE_CHECKING:
    from ._holder import StoreHolder

try:  # Hard dependency, but tolerate scaffold-phase imports before pip install.
    import psutil
except ImportError:  # pragma: no cover - stripped/scaffold installs only
    psutil = None  # type: ignore[assignment]

# Pre-split logger identity so log filtering/capture is unchanged.
logger = logging.getLogger("hermes_state")


def _meta_row(conn, key: str) -> Optional[sqlite3.Row]:
    """Point-read one ``state_meta`` row (``None`` when absent)."""
    return conn.execute("SELECT value FROM state_meta WHERE key = ?", (key,)).fetchone()


def _delete_meta(conn, *keys: str) -> None:
    conn.execute(f"DELETE FROM state_meta WHERE key IN ({','.join('?' for _ in keys)})", keys)


def _positive_int(name: str, value: Any) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value <= 0:
        raise ValueError(f"{name} must be greater than zero")


_NUMBERED_TITLE_RE = re.compile(r'^(.*?) #(\d+)$')


def next_title_in_lineage(conn, base_title: str) -> str:
    """Next title in a lineage ("my session" -> "my session #2") as seen by *conn*: strip any
    " #N" suffix, then increment the highest existing number."""
    match = _NUMBERED_TITLE_RE.match(base_title)
    base = match.group(1) if match else base_title
    rows = conn.execute(
        "SELECT title FROM sessions WHERE title = ? OR title LIKE ? ESCAPE '\\'",
        (base, f"{_escape_like(base)} #%")).fetchall()
    if not rows:
        return base
    # The unnumbered original counts as #1.
    numbers = [int(m.group(2)) for m in (_NUMBERED_TITLE_RE.match(row[0]) for row in rows) if m]
    return f"{base} #{max([1, *numbers]) + 1}"


_FTS_HOLDER_ESCALATE_ATTEMPTS = 3
_FTS_HOLDER_ESCALATE_SECONDS = 60.0
# The same holder PID set blocking this many deferrals over this long is a structurally resident
# peer (a supervised service on the same HERMES_HOME), not a transient one worth waiting out (#106393).
_FTS_HOLDER_FUTILE_ATTEMPTS = 10
_FTS_HOLDER_FUTILE_SECONDS = 1800.0
# retry_deferred_fts_recovery cadence: startup paid the full admission wait once; later
# retries are non-blocking probes whose spacing doubles up to the cap.
_FTS_STALE_RETRY_SECONDS = 60.0
_FTS_STALE_RETRY_MAX_SECONDS = 3600.0
# message_uid legacy backfill: rows per UPDATE (short write-lock holds) and the extra time one open may spend.
_MESSAGE_UID_BACKFILL_DONE = "message_uid_backfill"
_MESSAGE_UID_BACKFILL_CURSOR = "message_uid_backfill_id"
_MESSAGE_UID_BACKFILL_CHUNK = 2000
_MESSAGE_UID_BACKFILL_BUDGET_S = 1.0


def _holder_cmdline(pid: int) -> str:
    argv = _read_proc_argv(pid)
    return " ".join(argv)[:120] if argv else "<cmdline unavailable>"

# schema_read_probe_statements() cache (parses SCHEMA_SQL in an in-memory DB; once per process).
_READ_PROBE_STATEMENTS: Optional[tuple] = None

# Trigram triggers need the trigram tokenizer (SQLite >= 3.34); without it _ensure_fts_schema
# soft-fails that DDL and "all six present" is unsatisfiable, so a trigger's absence is
# measured only against the DDL that can create it.
_FTS_TRIGRAM_TRIGGERS = tuple(n for n in _FTS_TRIGGERS if "_trigram_" in n)
_FTS_BASE_TRIGGERS = tuple(n for n in _FTS_TRIGGERS if n not in _FTS_TRIGRAM_TRIGGERS)

# (base DDL, trigram DDL) keyed by "legacy inline layout?" — v23 external-content vs pre-v23 inline.
_FTS_DDL = {False: (FTS_SQL, FTS_TRIGRAM_SQL), True: (LEGACY_FTS_SQL, LEGACY_FTS_TRIGRAM_SQL)}
_LEGACY_INLINE_CONCAT_SQL = (
    "COALESCE(content, '') || ' ' || COALESCE(tool_name, '') || ' ' || COALESCE(tool_calls, '') "
)
_SESSION_MODEL_USAGE_INDEX_SQL = (
    "CREATE INDEX IF NOT EXISTS idx_session_model_usage_session ON session_model_usage(session_id)",
    "CREATE INDEX IF NOT EXISTS idx_session_model_usage_model ON session_model_usage(model)",
)
_SESSION_MODEL_USAGE_HEAL_DDL = """CREATE TABLE session_model_usage (
    session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    model TEXT NOT NULL,
    billing_provider TEXT NOT NULL DEFAULT '',
    billing_base_url TEXT NOT NULL DEFAULT '',
    billing_mode TEXT NOT NULL DEFAULT '',
    task TEXT NOT NULL DEFAULT '',
    api_call_count INTEGER NOT NULL DEFAULT 0,
    input_tokens INTEGER NOT NULL DEFAULT 0,
    output_tokens INTEGER NOT NULL DEFAULT 0,
    cache_read_tokens INTEGER NOT NULL DEFAULT 0,
    cache_write_tokens INTEGER NOT NULL DEFAULT 0,
    reasoning_tokens INTEGER NOT NULL DEFAULT 0,
    estimated_cost_usd REAL NOT NULL DEFAULT 0,
    actual_cost_usd REAL NOT NULL DEFAULT 0,
    cost_status TEXT,
    cost_source TEXT,
    first_seen REAL,
    last_seen REAL,
    PRIMARY KEY (session_id, model, billing_provider, billing_base_url, billing_mode, task)
)"""
# v22-migration rendering of the same table (column lines at 35 spaces, paren at 31; pinned SQL).
_SESSION_MODEL_USAGE_V22_DDL = "\n".join(
    [_SESSION_MODEL_USAGE_HEAL_DDL.splitlines()[0]]
    + [" " * 35 + ln.strip() for ln in _SESSION_MODEL_USAGE_HEAL_DDL.splitlines()[1:-1]]
    + [" " * 31 + ")"]
)
# Statement text pinned by the SQL trace harness (whitespace included).
_SESSION_MODEL_USAGE_V20_SEED_SQL = """INSERT OR IGNORE INTO session_model_usage (
                               session_id, model, billing_provider,
                               billing_base_url, billing_mode,
                               api_call_count, input_tokens,
                               output_tokens, cache_read_tokens,
                               cache_write_tokens, reasoning_tokens,
                               estimated_cost_usd, actual_cost_usd,
                               cost_status, cost_source, first_seen, last_seen
                           )
                           SELECT id, COALESCE(model, 'unknown'),
                                  COALESCE(billing_provider, ''),
                                  COALESCE(billing_base_url, ''),
                                  COALESCE(billing_mode, ''),
                                  COALESCE(api_call_count, 0),
                                  COALESCE(input_tokens, 0),
                                  COALESCE(output_tokens, 0),
                                  COALESCE(cache_read_tokens, 0),
                                  COALESCE(cache_write_tokens, 0),
                                  COALESCE(reasoning_tokens, 0),
                                  COALESCE(estimated_cost_usd, 0),
                                  COALESCE(actual_cost_usd, 0),
                                  cost_status, cost_source,
                                  started_at, COALESCE(ended_at, started_at)
                           FROM sessions
                           WHERE COALESCE(input_tokens, 0)
                                 + COALESCE(output_tokens, 0)
                                 + COALESCE(cache_read_tokens, 0)
                                 + COALESCE(cache_write_tokens, 0)
                                 + COALESCE(reasoning_tokens, 0) > 0"""
_TITLE_UNIQUE_INDEX_SQL = (
    "CREATE UNIQUE INDEX IF NOT EXISTS idx_sessions_title_unique ON sessions(title) WHERE title IS NOT NULL"
)
_STALE_KEY_UPSERT_SQL = (
    "INSERT INTO state_meta (key, value) VALUES (?, '1') ON CONFLICT(key) DO UPDATE SET value = excluded.value"
)
_STATE_META_UPSERT_SQL = (
    "INSERT INTO state_meta (key, value) VALUES (?, ?) ON CONFLICT(key) DO UPDATE SET value = excluded.value"
)
_CLEAR_REBUILD_MARKERS_SQL = "DELETE FROM state_meta WHERE key IN ('fts_rebuild_high_water', 'fts_rebuild_progress')"
# FTS_STORAGE_VERSION < 3 truncated tool rows only above a moving state_meta mark; the aligned
# projection truncates by role alone, so the retired marker is dropped with the realign.
_DROP_RETIRED_TOOL_HIGH_WATER_SQL = "DELETE FROM state_meta WHERE key = 'fts_tool_full_content_high_water'"


def _legacy_inline_reinsert_sql(table: str, indent: int, *, delete_first: bool = False) -> str:
    """Legacy inline (pre-v23) FTS re-population script fragment (whitespace pinned)."""
    pad = " " * indent
    body = f"{pad}DELETE FROM {table};\n" if delete_first else ""
    return (
        f"\n{body}{pad}INSERT INTO {table}(rowid, content)\n"
        f"{pad}SELECT id,\n"
        f"{pad}       COALESCE(content, '') || ' ' ||\n"
        f"{pad}       COALESCE(tool_name, '') || ' ' ||\n"
        f"{pad}       COALESCE(tool_calls, '')\n"
        f"{pad}FROM messages;\n{pad[:-4]}"
    )


def _q(ident: str) -> str:
    """Double-quote an SQL identifier."""
    return '"' + ident.replace('"', '""') + '"'


def schema_read_probe_statements(home: Path) -> tuple:
    """SELECT statements that fail iff a live store is behind SCHEMA_SQL. Read-only opens skip
    ``_reconcile_columns()`` (no DDL against another profile's live DB), so healing callers
    run these afterwards: a missing table/column raises at prepare time. Derived from
    SCHEMA_SQL (a hand-maintained list went stale within days). Columns are
    table-qualified: an unqualified double-quoted identifier that fails to resolve silently
    degrades to a string literal (SQLite misfeature) and would pass on the stale store."""
    global _READ_PROBE_STATEMENTS
    if _READ_PROBE_STATEMENTS is None:
        tables = _parse_schema_columns(SCHEMA_SQL, home)
        _READ_PROBE_STATEMENTS = tuple(
            "SELECT {} FROM {} LIMIT 0".format(", ".join(f"{_q(table)}.{_q(col)}" for col in cols), _q(table))
            for table, cols in sorted(tables.items())
        )
    return _READ_PROBE_STATEMENTS


# Title provenance, lowest to highest authority: auto-titling may only replace a
# strictly lower-authority title (``derived`` -> ``llm`` once; never a user-typed name).
TITLE_SOURCE_DERIVED = _TITLE_SOURCE_DERIVED
TITLE_SOURCE_LLM = _TITLE_SOURCE_LLM
TITLE_SOURCE_USER = _TITLE_SOURCE_USER
_TITLE_SOURCE_RANK = {TITLE_SOURCE_DERIVED: 0, TITLE_SOURCE_LLM: 1, TITLE_SOURCE_USER: 2}


def _dedupe_legacy_system_prompts(holder: StoreHolder, cursor: sqlite3.Cursor) -> None:
    """Move inline prompt snapshots into the shared content-addressed table. Any
    ``OperationalError`` mid-loop returns instead of raising: partial migration is safe
    (the legacy column stays a read fallback; next init resumes), whereas propagating
    left the version below 25 and re-ran this on every open (gateway crash loop)."""
    try:
        rows = cursor.execute("SELECT id, system_prompt FROM sessions WHERE system_prompt IS NOT NULL").fetchall()
    except sqlite3.OperationalError:
        return
    for session_id, prompt in rows:
        try:
            prompt_hash = holder._store_system_prompt(cursor, prompt)
            cursor.execute(
                "UPDATE sessions SET system_prompt_hash = ?, system_prompt = NULL WHERE id = ?",
                (prompt_hash, session_id),
            )
        except sqlite3.OperationalError as exc:
            logger.warning(
                "v25 prompt dedupe paused after contention (%s); "
                "unmigrated rows keep the legacy inline prompt and the next schema init resumes the migration.",
                exc,
            )
            return


def _sqlite_supports_fts5(holder: StoreHolder, cursor: sqlite3.Cursor) -> bool:
    try:
        cursor.execute("CREATE VIRTUAL TABLE temp._hermes_fts5_probe USING fts5(x)")
        cursor.execute("DROP TABLE temp._hermes_fts5_probe")
        return True
    except sqlite3.OperationalError as exc:
        if not _is_fts5_unavailable_error(exc):
            raise
        _warn_fts5_unavailable(holder, exc)
        return False


def _drop_all_fts_triggers(holder: StoreHolder, cursor: sqlite3.Cursor) -> None:
    _drop_fts_triggers(cursor)
    for trigger in _FTS_CJK_TRIGGERS:
        with contextlib.suppress(sqlite3.OperationalError):
            cursor.execute(f"DROP TRIGGER IF EXISTS {trigger}")


def _fts_triggers_missing(cursor: sqlite3.Cursor, names: Sequence[str]) -> bool:
    """True unless every trigger in *names* (one DDL half) exists."""
    if not names:
        return False  # "name IN ()" is a SQLite syntax error
    placeholders = ",".join("?" for _ in names)
    sql = f"SELECT COUNT(*) FROM sqlite_master WHERE type = 'trigger' AND name IN ({placeholders})"
    return int(cursor.execute(sql, tuple(names)).fetchone()[0]) < len(names)


def _fts_update_trigger_needs_narrowing(sql: Optional[str]) -> bool:
    """True when trigger SQL is a broad AFTER UPDATE (missing ``OF``)."""
    if not sql:
        return False
    compact = " ".join(sql.split()).upper()  # multi-line DDL still matches
    return "AFTER UPDATE OF " not in compact and "AFTER UPDATE ON " in compact


def _migrate_broad_fts_update_triggers(holder: StoreHolder, cursor: sqlite3.Cursor) -> int:
    """Replace broad AFTER UPDATE FTS triggers with AFTER UPDATE OF variants (``IF NOT EXISTS``
    never replaces an existing broad trigger). No FTS rebuild: correctness was already
    gated by WHEN clauses; OF only skips trigger evaluation. Returns the number dropped."""
    # CJK is v23-only. Decide the layout before selecting destructive candidates so the
    # legacy branch never drops a trigger it won't recreate.
    legacy_layout = _db_has_legacy_inline_fts(cursor)
    update_names = ("messages_fts_update", "messages_fts_trigram_update") + (
        () if legacy_layout else ("messages_fts_cjk_update",)
    )
    placeholders = ", ".join("?" for _ in update_names)
    sql = f"SELECT name, sql FROM sqlite_master WHERE type = 'trigger' AND name IN ({placeholders})"
    rows = cursor.execute(sql, update_names).fetchall()
    to_drop = [name for name, sql in rows if _fts_update_trigger_needs_narrowing(sql)]
    if not to_drop:
        return 0
    for name in to_drop:
        cursor.execute(f"DROP TRIGGER IF EXISTS {name}")  # names from the literal allowlist above

    # Re-apply current DDL (legacy vs v23 as _init_schema does) so CREATE TRIGGER installs OF variants.
    base_sql, trigram_sql = _FTS_DDL[legacy_layout]
    _ensure_fts_schema(holder, cursor, "messages_fts", base_sql)
    _ensure_fts_schema(holder, cursor, "messages_fts_trigram", trigram_sql)
    # Only recreate the CJK trigger this migration dropped. ``_ensure_fts_cjk_schema`` soft-fails
    # (never raises), so afterwards require a narrowed trigger or durable quarantine.
    if "messages_fts_cjk_update" in to_drop:
        try:
            _ensure_fts_cjk_schema(holder, cursor)
        except Exception:
            _quarantine_cjk_after_update_of_migration(holder, cursor)
            logger.exception("CJK FTS re-ensure after UPDATE OF migration failed")
            raise
        row = cursor.execute(
            "SELECT sql FROM sqlite_master WHERE type = 'trigger' AND name = ?", ("messages_fts_cjk_update",),
        ).fetchone()
        if not row or _fts_update_trigger_needs_narrowing(row[0]):
            _quarantine_cjk_after_update_of_migration(holder, cursor)
            logger.warning(
                "CJK FTS UPDATE trigger missing or still broad after "
                "UPDATE OF migration; marked stale and unavailable"
            )
    logger.info("Migrated %d broad FTS UPDATE trigger(s) to AFTER UPDATE OF (no rebuild required)", len(to_drop))
    return len(to_drop)


def _fts_index_is_misaligned_source(cursor: sqlite3.Cursor) -> bool:
    """True when ``messages_fts`` is still external-content over the raw
    ``messages`` table (FTS_STORAGE_VERSION < 3): its index holds a TRUNCATED
    projection for long tool rows that the checker/'delete' commands re-read
    as FULL content, a mismatch by construction. Such an index cannot be
    repaired in place — it must be 'rebuild'-filled from the aligned
    ``messages_fts_src`` view exactly once."""
    row = cursor.execute(
        "SELECT sql FROM sqlite_master WHERE type = 'table' AND name = 'messages_fts'"
    ).fetchone()
    return row is not None and "messages_fts_src" not in (row[0] or "")


def _execute_ddl_skipping_settled_triggers(cursor: sqlite3.Cursor, ddl: str) -> None:
    """Run *ddl* statement by statement, skipping each ``DROP TRIGGER IF EXISTS x`` /
    ``CREATE TRIGGER IF NOT EXISTS x …`` pair whose trigger already exists with the same body.

    ``DROP TRIGGER IF EXISTS`` on an existing trigger takes SQLite's write lock (the
    ``CREATE … IF NOT EXISTS`` forms do not), so the unconditional drop+recreate that lets a
    changed trigger body roll out would otherwise block every open of a settled database
    behind a sibling process's transaction. ``sqlite_master.sql`` stores the CREATE text
    verbatim minus ``IF NOT EXISTS`` and the ``;``, so an exact comparison decides.
    """
    stored = dict(cursor.execute("SELECT name, sql FROM sqlite_master WHERE type = 'trigger'").fetchall())
    pending_drop: Optional[str] = None
    statement = ""
    for line in ddl.splitlines():
        statement += line + "\n"
        if not sqlite3.complete_statement(statement):
            continue
        statement, current = "", statement.strip()
        upper = current.upper()
        if upper.startswith("DROP TRIGGER IF EXISTS "):
            pending_drop = current
            continue
        if upper.startswith("CREATE TRIGGER IF NOT EXISTS "):
            desired = current.replace("IF NOT EXISTS ", "", 1).rstrip(";")
            if stored.get(desired.split(None, 3)[2]) == desired:
                pending_drop = None
                continue
        if pending_drop is not None:
            cursor.execute(pending_drop)
            pending_drop = None
        cursor.execute(current)
    if statement.strip():
        raise sqlite3.OperationalError("incomplete DDL statement")


def _execute_ddl_script_transactional(cursor: sqlite3.Cursor, ddl: str) -> None:
    """Execute a DDL script without ``executescript``'s implicit commit."""
    statement = ""
    for line in ddl.splitlines():
        statement += line + "\n"
        if sqlite3.complete_statement(statement):
            cursor.execute(statement)
            statement = ""
    if statement.strip():
        raise sqlite3.OperationalError("incomplete FTS DDL statement")


def _migrate_misaligned_fts_source(holder: StoreHolder, cursor: sqlite3.Cursor, *, legacy: bool) -> None:
    """Re-point ``messages_fts`` at the stable ``messages_fts_src`` projection view and
    rebuild it ONCE (FTS_STORAGE_VERSION 2 -> 3). A v1/v2 base index carries token streams
    the raw-``messages`` external-content source cannot read back (truncated long tool
    rows, and tool rows whose full content was indexed under an old high-water mark), so
    in-place continuity is not achievable — the ONLY valid transition is a full rebuild
    from the view, under the shared cross-process rebuild admission. Legacy inline DBs
    skip this entirely (their index is self-contained; they still take the DDL on the
    optimize path), and so does a definition no release of this code created (a newer
    release's, after a downgrade)."""
    if legacy or not base_index_recognised(cursor):
        return
    stale_view = base_source_view_is_stale(cursor)
    if not stale_view and not _fts_index_is_misaligned_source(cursor):
        return
    has_messages = cursor.execute("SELECT 1 FROM messages LIMIT 1").fetchone() is not None

    def do_align() -> None:
        for name in _FTS_BASE_TRIGGERS:
            cursor.execute(f"DROP TRIGGER IF EXISTS {name}")
        cursor.execute("DROP TABLE IF EXISTS messages_fts")
        if stale_view:
            # CREATE VIEW IF NOT EXISTS never replaces a differing view; the index is rebuilt from it below.
            cursor.execute("DROP VIEW messages_fts_src")
        _ensure_fts_schema(holder, cursor, "messages_fts", FTS_SQL)
        if has_messages:
            cursor.execute("INSERT INTO messages_fts(messages_fts) VALUES('rebuild')")
            # The cleared markers below are shared with the trigram backfill: finish that index too, or rows
            # an interrupted backfill had not reached stay out of it with nothing left to resume them.
            if _meta_row(cursor, "fts_rebuild_high_water") is not None:
                trigram = _fts_table_probe(holder, cursor, "messages_fts_trigram")
                if trigram:
                    cursor.execute("INSERT INTO messages_fts_trigram(messages_fts_trigram) VALUES('rebuild')")
                elif trigram is None:
                    # No usable tokenizer here: detach the index so the first open that has one rebuilds it
                    # through the missing-trigger repair in _init_fts.
                    for name in _FTS_TRIGRAM_TRIGGERS:
                        cursor.execute(f"DROP TRIGGER IF EXISTS {name}")
        cursor.execute(_CLEAR_REBUILD_MARKERS_SQL)
        cursor.execute(_DROP_RETIRED_TOOL_HIGH_WATER_SQL)

    if not has_messages:
        # Nothing indexed and nothing to index: swap the shape in place, no rebuild authority needed.
        cursor.execute("SAVEPOINT fts_align_empty")
        try:
            do_align()
            cursor.execute("RELEASE SAVEPOINT fts_align_empty")
        except BaseException:
            cursor.execute("ROLLBACK TO SAVEPOINT fts_align_empty")
            cursor.execute("RELEASE SAVEPOINT fts_align_empty")
            raise
        return
    _run_admitted_startup_rebuild(holder, cursor, do_align)


def _sqlite_table_exists(cursor: sqlite3.Cursor, name: str) -> bool:
    return cursor.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?", (name,),
    ).fetchone() is not None


def _migrate_trigram_cron_exclusion(holder: StoreHolder, cursor: sqlite3.Cursor) -> bool:
    """Install the source-filtered trigram view and purge historical rows (v29 cron exclusion,
    v30 subagent exclusion: both only change the view/trigger predicate and rebuild from it).
    Legacy inline indexes stay opt-in (their content is private to the vtable). A v1 external
    layout whose vtable still declares ``tool_calls`` is left to ``optimize-storage``: swapping
    the view underneath would make 'rebuild' read a column the view no longer has. Otherwise
    the inverted index still holds excluded rows until FTS5 rebuilds from the new view, which
    runs under the shared cross-process admission gate. Returns False to hold schema_version back."""
    if _db_has_legacy_inline_fts(cursor) or _db_has_trigram_tool_calls_projection(cursor):
        return True
    trigram_exists = _fts_table_probe(holder, cursor, "messages_fts_trigram")
    if trigram_exists is not True:
        # Absent: the normal ensure path creates/backfills it. None: this runtime cannot
        # safely inspect an existing one, so leave the version behind for retry.
        return trigram_exists is False
    for name in _FTS_TRIGRAM_TRIGGERS:
        cursor.execute(f"DROP TRIGGER IF EXISTS {name}")
    cursor.execute("DROP VIEW IF EXISTS messages_fts_trigram_src")
    if not _ensure_fts_schema(holder, cursor, "messages_fts_trigram", FTS_TRIGRAM_SQL):
        return False
    # Always rebuild while schema_version is behind, even if the view already has the new
    # predicate: a process can die between replacing the view and rebuilding/stamping.
    _run_admitted_startup_rebuild(holder,
        cursor,
        lambda: cursor.execute("INSERT INTO messages_fts_trigram(messages_fts_trigram) VALUES('rebuild')"),
    )
    return True


def _quarantine_cjk_after_update_of_migration(holder: StoreHolder, cursor: sqlite3.Cursor) -> None:
    """Fail closed after dropping the CJK UPDATE trigger mid-migration: clear availability,
    persist ``fts_cjk_stale``, drop any residual trigger so a later open cannot
    IF-NOT-EXISTS over a gap."""
    holder._fts_cjk_available = False
    try:
        holder.set_meta(FTS_CJK_STALE_KEY, "1", cursor=cursor)
    except Exception:
        logger.debug("Could not persist CJK FTS stale breadcrumb", exc_info=True)
    try:
        cursor.execute("DROP TRIGGER IF EXISTS messages_fts_cjk_update")
    except Exception:
        logger.debug("Could not drop residual CJK UPDATE trigger after quarantine", exc_info=True)


def _rebuild_fts_indexes(cursor: sqlite3.Cursor, *, legacy: bool = False, include_trigram: bool = True) -> None:
    """v23+ external-content 'rebuild'. It indexes EVERY row, so the deferred-backfill
    markers are cleared or the worker would re-insert covered rows (duplicates).
    ``legacy`` (pre-v23 inline layout) has no external-content 'rebuild' source, so it
    DELETEs + reinserts the concatenated content the legacy triggers produced."""
    tables = ("messages_fts", "messages_fts_trigram") if include_trigram else ("messages_fts",)
    for tbl in tables:
        if legacy:
            cursor.execute(f"DELETE FROM {tbl}")
            cursor.execute(f"INSERT INTO {tbl}(rowid, content) SELECT id, {_LEGACY_INLINE_CONCAT_SQL}FROM messages")
        else:
            cursor.execute(f"INSERT INTO {tbl}({tbl}) VALUES('rebuild')")
    if not legacy:
        cursor.execute(_CLEAR_REBUILD_MARKERS_SQL)


def _fts_table_probe(holder: StoreHolder, cursor: sqlite3.Cursor, table_name: str) -> Optional[bool]:
    """True = queryable, False = absent, None = FTS module/tokenizer missing or content
    undecodable (index degraded, store accessible). Invalid UTF-8 surfaces as a bare
    UnicodeDecodeError on some builds and OperationalError("Could not decode to UTF-8")
    on others; both are caught so the probe never raises into init/recovery flows.
    Anything else (malformed schema, corrupt vtable) re-raises."""
    try:
        cursor.execute(f"SELECT * FROM {table_name} LIMIT 0")
        return True
    except UnicodeDecodeError as exc:
        decode_exc = exc
    except sqlite3.OperationalError as exc:
        if _is_fts5_unavailable_error(exc):
            # A missing trigram tokenizer only affects trigram search; only a missing
            # FTS5 module disables FTS entirely.
            if _is_trigram_unavailable_error(exc):
                _warn_trigram_unavailable(holder, exc)
            else:
                _warn_fts5_unavailable(holder, exc)
            return None
        if "no such table" in str(exc).lower():
            return False
        if "decode to utf-8" not in str(exc).lower():
            raise
        decode_exc = exc
    logger.warning(
        "%s probe encountered invalid UTF-8 in FTS content; "
        "search may return incomplete results until FTS is rebuilt: %s", table_name, decode_exc,
    )
    return None


# ── Stale-FTS recovery ─────────────────────────────────────────────────

def _defer_stale_fts_for_holders(holder: StoreHolder, cursor: sqlite3.Cursor, foreign_holders) -> bool:
    """Record a deferral diagnostic for the foreign processes holding the DB; True = defer
    (holders remain). After ``_FTS_HOLDER_ESCALATE_ATTEMPTS`` deferrals spanning
    ``_FTS_HOLDER_ESCALATE_SECONDS``, provably inactive orphan Desktop backends are
    reaped and the holders re-checked. The orphan reap is the only exit, so a supervised
    peer (never an orphan) blocks forever: once the SAME PID set has blocked
    ``_FTS_HOLDER_FUTILE_ATTEMPTS`` deferrals over ``_FTS_HOLDER_FUTILE_SECONDS`` the
    record is marked ``futile`` and the escalation names the holders and the remedy that
    works from inside a gateway session (stop only the other holder; this process's own
    retry tick admits the rebuild). A changed holder set restarts that window."""
    now = time.time()
    try:
        row = cursor.execute(
            "SELECT value FROM state_meta WHERE key = ? LIMIT 1", (FTS_REBUILD_DEFERRAL_KEY,),
        ).fetchone()
    except sqlite3.Error:
        row = None
    parsed = safe_json_loads(row[0]) if row else None
    record = parsed if isinstance(parsed, dict) else {}
    try:
        first_seen = float(record.get("first_seen", now))
        attempts = int(record.get("attempts", 0)) + 1
        holders_since = float(record.get("holders_since", now))
        holders_attempts = int(record.get("holders_attempts", 0)) + 1
    except (TypeError, ValueError):
        first_seen, attempts, holders_since, holders_attempts = now, 1, now, 1
    if first_seen > now or first_seen < 0:
        first_seen = now
    holder_pids = sorted({pid for pid, _path in foreign_holders if pid > 0})
    if holder_pids != record.get("holder_pids"):
        holders_since, holders_attempts = now, 1
    if attempts >= _FTS_HOLDER_ESCALATE_ATTEMPTS and now - first_seen >= _FTS_HOLDER_ESCALATE_SECONDS:
        reaped = _reap_inactive_orphan_desktop_holders(holder,
            foreign_holders, min_age_seconds=_FTS_HOLDER_ESCALATE_SECONDS,
        )
        if reaped:
            logger.error(
                "Reaped inactive orphan Desktop backend(s) %s after %d "
                "state.db FTS rebuild deferrals; checking holders again.", reaped, attempts,
            )
            foreign_holders = holder._foreign_state_db_holders()
            holder_pids = sorted({pid for pid, _path in foreign_holders if pid > 0})
    futile = bool(holder_pids) and (
        holders_attempts >= _FTS_HOLDER_FUTILE_ATTEMPTS and now - holders_since >= _FTS_HOLDER_FUTILE_SECONDS
    )
    diagnostic = {
        "first_seen": first_seen, "last_seen": now, "attempts": attempts, "holder_pids": holder_pids,
        "holders_since": holders_since, "holders_attempts": holders_attempts, "futile": futile,
    }
    cursor.execute(
        "INSERT INTO state_meta (key, value) VALUES (?, ?) ON CONFLICT(key) DO UPDATE SET value = excluded.value",
        (FTS_REBUILD_DEFERRAL_KEY, json.dumps(diagnostic, sort_keys=True)),
    )
    if not foreign_holders:
        holder._fts_deferred_holder_pids = None
        return False
    holder._fts_deferred_holder_pids = holder_pids
    if futile:
        logger.error(
            "state.db FTS repair has been blocked by the same holder(s) for %d deferrals over %.0f min "
            "(%s); waiting is futile. Stop ONLY the other holder(s) — this process keeps running and its "
            "own retry admits the rebuild within %.0fs of the holder leaving. `hermes doctor` shows this.",
            holders_attempts, (now - holders_since) / 60.0,
            ", ".join(f"pid {pid}: {_holder_cmdline(pid)}" for pid in holder_pids), _FTS_STALE_RETRY_SECONDS,
        )
    elif attempts >= _FTS_HOLDER_ESCALATE_ATTEMPTS and now - first_seen >= _FTS_HOLDER_ESCALATE_SECONDS:
        logger.error(
            "state.db FTS repair remains blocked after %d deferrals by holder(s) %s. Stop the listed "
            "processes (this process's own retry then rebuilds), or run `hermes sessions optimize-storage` "
            "with every holder stopped. `hermes doctor` reports this degraded state.", attempts, foreign_holders,
        )
    logger.warning(
        "Deferred stale state.db FTS rebuild while foreign processes "
        "hold the database or WAL sidecars (%s); canonical writes and LIKE search remain available (deferral %d).",
        foreign_holders, attempts,
    )
    return True


def _recover_stale_fts(holder: StoreHolder, cursor: sqlite3.Cursor, *, legacy: bool, timeout_seconds=None) -> bool:
    """Atomically rebuild stale base/trigram indexes and resume syncing. *timeout_seconds*
    bounds the admission wait (None = full startup budget, ``0`` = non-blocking retry).
    Fails closed: holders or a lost admission race leave the breadcrumb set."""
    foreign_holders = holder._foreign_state_db_holders()
    if foreign_holders and _defer_stale_fts_for_holders(holder, cursor, foreign_holders):
        return False
    with fts_rebuild_admission(holder.db_path, timeout_seconds=timeout_seconds) as admitted:
        if not admitted:
            logger.warning(
                "Deferred stale state.db FTS rebuild: another process holds the rebuild authority; "
                "canonical writes and LIKE search remain available."
            )
            return False
        return _recover_stale_fts_locked(holder, cursor, legacy=legacy)


def retry_deferred_fts_recovery(holder: StoreHolder) -> bool:
    """Retry a deferred stale-FTS rebuild (gateway housekeeping tick). ``_recover_stale_fts``
    fails closed at open, leaving search on LIKE; live write/search paths must never
    start a full rebuild, and a gateway opens state.db once for days, so "next open"
    never comes. Bounded doubling backoff, non-blocking admission, no new thread. True
    only when the index was rebuilt and sync triggers restored. Never raises.

    This is the in-process retry: bounded backoff from ``_FTS_STALE_RETRY_SECONDS`` doubling to
    ``_FTS_STALE_RETRY_MAX_SECONDS``, non-blocking admission (``timeout=0``) so a live holder is skipped
    and tried again later, no new thread — the caller is an existing periodic tick (gateway
    housekeeping). See #100108, #97940.
    """
    if not holder._fts_stale:
        return False
    if holder._quarantine_reason() is not None:
        # Quarantined: never run FTS DDL/DML against a damaged image or a stale/replaced generation
        # (mirrors _try_wal_checkpoint / close). Reset the backoff so a future un-quarantine starts
        # from the default interval.
        holder._fts_stale_retry_after = 0.0
        holder._fts_stale_retry_interval = 0.0
        return False
    if holder.read_only or holder._conn is None:
        return False
    now = time.monotonic()
    deferred_pids = getattr(holder, "_fts_deferred_holder_pids", None)
    if now < getattr(holder, "_fts_stale_retry_after", 0.0):
        # The backoff was earned by a specific holder set; once that set changes (the other
        # service stopped) a capped backoff would idle up to an hour with nothing blocking (#106393).
        if deferred_pids is None or sorted(
            {pid for pid, _path in holder._foreign_state_db_holders() if pid > 0}
        ) == deferred_pids:
            return False
        holder._fts_stale_retry_interval = 0.0
    interval = float(getattr(holder, "_fts_stale_retry_interval", 0.0))
    if interval <= 0.0:
        interval = _FTS_STALE_RETRY_SECONDS
    holder._fts_stale_retry_after = now + interval
    holder._fts_stale_retry_interval = min(
        max(interval, _FTS_STALE_RETRY_SECONDS, 1.0) * 2.0, _FTS_STALE_RETRY_MAX_SECONDS,
    )
    try:
        with holder._lock:
            if holder._conn is None or not holder._fts_stale:
                return False
            cursor = holder._conn.cursor()
            legacy = _db_has_legacy_inline_fts(cursor)
            recovered = _recover_stale_fts(holder, cursor, legacy=legacy, timeout_seconds=0.0)
            if recovered:
                # CJK was detached alongside the base indexes; its own ensure path
                # decides when it comes back online.
                _ensure_fts_cjk_schema(holder, cursor)
                holder._fts_stale_retry_interval = 0.0
            with contextlib.suppress(sqlite3.Error):
                holder._conn.commit()
            return recovered
    except Exception:
        logger.warning(
            "In-process retry of the deferred stale state.db FTS rebuild failed; will retry later.", exc_info=True,
        )
        return False


def _recover_stale_fts_locked(holder: StoreHolder, cursor: sqlite3.Cursor, *, legacy: bool) -> bool:
    """Body of :meth:`_recover_stale_fts`; caller holds rebuild authority. One write
    transaction, so no canonical writer slips between rebuild and trigger restoration."""
    try:
        trigram_present = _fts_table_probe(holder, cursor, "messages_fts_trigram") is True
    except (sqlite3.DatabaseError, UnicodeDecodeError):
        # A corrupt vtable may fail even a LIMIT 0 probe; still include it in the drop-and-recreate.
        include_trigram = True
    else:
        include_trigram = trigram_present or (not legacy and _trigram_tokenizer_available(holder, cursor))

    drop_sql = "".join(f"DROP TRIGGER IF EXISTS {trigger};" for trigger in _FTS_TRIGGERS)
    if include_trigram:
        drop_sql += "DROP TABLE IF EXISTS messages_fts_trigram;"
    drop_sql += "DROP VIEW IF EXISTS messages_fts_trigram_src;DROP TABLE IF EXISTS messages_fts;"
    if legacy:
        rebuild_sql = LEGACY_FTS_SQL + (LEGACY_FTS_TRIGRAM_SQL if include_trigram else "")
        rebuild_sql += _legacy_inline_reinsert_sql("messages_fts", 16)
        if include_trigram:
            rebuild_sql += _legacy_inline_reinsert_sql("messages_fts_trigram", 20, delete_first=True)
    else:
        rebuild_sql = FTS_SQL + (FTS_TRIGRAM_SQL if include_trigram else "")
        rebuild_sql += "INSERT INTO messages_fts(messages_fts) VALUES('rebuild');"
        if include_trigram:
            rebuild_sql += "INSERT INTO messages_fts_trigram(messages_fts_trigram) VALUES('rebuild');"
        rebuild_sql += _CLEAR_REBUILD_MARKERS_SQL + ";" + _DROP_RETIRED_TOOL_HIGH_WATER_SQL + ";"
    recovery_sql = (
        "BEGIN IMMEDIATE;" + drop_sql + rebuild_sql
        + f"DELETE FROM state_meta WHERE key IN ('{FTS_STALE_KEY}', '{FTS_REBUILD_DEFERRAL_KEY}');COMMIT;"
    )
    try:
        cursor.executescript(recovery_sql)
    except sqlite3.DatabaseError as exc:
        with contextlib.suppress(sqlite3.Error):
            holder._conn.rollback()
        # Stale indexes must stay detached even on builds whose DDL transaction behavior differs.
        _drop_all_fts_triggers(holder, cursor)
        holder._conn.commit()
        logger.error(
            "Automatic rebuild of stale FTS indexes failed (%s); "
            "canonical writes remain enabled with FTS detached.", exc,
        )
        return False
    holder._fts_stale = False
    holder._fts_enabled = True
    holder._trigram_available = include_trigram
    logger.warning("Rebuilt stale state.db FTS indexes from canonical messages and restored sync triggers.")
    return True


def _trigram_tokenizer_available(holder: StoreHolder, cursor: sqlite3.Cursor) -> bool:
    """Probe trigram support without publishing a persistent FTS object."""
    probe = "temp.hermes_fts5_trigram_probe"
    cursor.execute(f"DROP TABLE IF EXISTS {probe}")
    try:
        cursor.execute(f"CREATE VIRTUAL TABLE {probe} USING fts5(content, tokenize='trigram')")
    except sqlite3.OperationalError as exc:
        if not _is_trigram_unavailable_error(exc):
            raise
        _warn_trigram_unavailable(holder, exc)
        return False
    finally:
        cursor.execute(f"DROP TABLE IF EXISTS {probe}")
    return True


# ── Declarative column reconciliation ──────────────────────────────────

def _parse_schema_columns(schema_sql: str, home: Path) -> dict[str, dict[str, str]]:
    """Expected columns per table: execute SCHEMA_SQL in an in-memory database and read
    PRAGMA table_info (no regex). Memoized under ``home``, the store's folder, keyed by a DDL hash (~85ms per
    startup otherwise); only the reference-side parse is cached — diffing the LIVE
    database still runs every startup. A corrupt/stale cache degrades to recomputation."""
    cache_path = None
    schema_hash = hashlib.sha256(schema_sql.encode("utf-8")).hexdigest()
    with contextlib.suppress(Exception):  # missing/corrupt cache → recompute below
        cache_path = home / "cache" / "schema_columns.json"
        blob = json.loads(cache_path.read_text(encoding="utf-8"))
        tables = blob.get("tables") if isinstance(blob, dict) and blob.get("schema_hash") == schema_hash else None
        if isinstance(tables, dict) and all(
            isinstance(cols, dict) and all(isinstance(v, str) for v in cols.values()) for cols in tables.values()
        ):
            return tables

    ref = sqlite3.connect(":memory:")
    try:
        ref.executescript(schema_sql)
        table_columns: dict[str, dict[str, str]] = {}
        for (tbl,) in ref.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
        ).fetchall():
            cols: dict[str, str] = {}
            info = ref.execute(f'PRAGMA table_info("{tbl}")').fetchall()
            for _cid, col_name, col_type, notnull, default, pk in info:
                # Reconstruct the type expression for ALTER TABLE ADD COLUMN
                parts = [col_type] if col_type else []
                if notnull and not pk:
                    parts.append("NOT NULL")
                if default is not None:
                    parts.append(f"DEFAULT {default}")
                cols[col_name] = " ".join(parts)
            table_columns[tbl] = cols
    finally:
        ref.close()

    if cache_path is not None:
        with contextlib.suppress(Exception):  # cache write is best-effort
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            fd, tmp = tempfile.mkstemp(dir=str(cache_path.parent), prefix=".schema_columns.")
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                json.dump({"schema_hash": schema_hash, "tables": table_columns}, fh)
            os.replace(tmp, cache_path)
    return table_columns


def _reconcile_columns(cursor: sqlite3.Cursor, home: Path) -> None:
    """ADD every SCHEMA_SQL column missing from the live tables (SCHEMA_SQL is the single
    source of truth; column additions need no version-gated migration)."""
    expected = _parse_schema_columns(SCHEMA_SQL, home)
    for table_name, declared_cols in expected.items():
        try:
            rows = cursor.execute(f'PRAGMA table_info("{table_name}")').fetchall()
        except sqlite3.OperationalError:
            continue  # Table doesn't exist yet (shouldn't happen after executescript)
        if not rows:
            continue  # PRAGMA table_info returns [] for a missing table; the DDL creates it whole
        # PRAGMA table_info rows: (cid, name, type, notnull, dflt_value, pk)
        live_cols = {row[1] for row in rows}
        for col_name, col_type in declared_cols.items():
            if col_name in live_cols:
                continue
            try:
                cursor.execute(f'ALTER TABLE "{table_name}" ADD COLUMN {_q(col_name)} {col_type}')
            except sqlite3.OperationalError as exc:
                message = str(exc).lower()
                if "duplicate column" in message:
                    # A sibling process won the ADD race; store is correct.
                    logger.debug("reconcile %s.%s: %s", table_name, col_name, exc)
                    continue
                if is_sqlite_lock_error(exc):
                    # Swallowing lock contention left the store half-reconciled ("no such
                    # column" on every read). Re-raise so the lock-patience wrapper retries init.
                    raise
                # Anything else permanently strands the store behind SCHEMA_SQL — be loud.
                logger.warning(
                    "reconcile %s.%s failed; store remains behind SCHEMA_SQL: %s", table_name, col_name, exc,
                )


def _live_pk_columns(cursor: sqlite3.Cursor, table: str) -> Optional[list[str]]:
    """PRIMARY KEY column names of *table* in key order; None when the table is
    missing or has no columns (SCHEMA_SQL creates it correctly)."""
    try:
        rows = cursor.execute(f'PRAGMA table_info("{table}")').fetchall()
    except sqlite3.OperationalError:
        rows = None
    if not rows:
        return None
    # row: (cid, name, type, notnull, dflt_value, pk)
    return [r[1] for r in sorted((r for r in rows if r[5]), key=lambda r: r[5])]


def _rebuild_table(cursor: sqlite3.Cursor, table: str, legacy_name: str, ddl: str, copy_sql: str, indexes=()) -> None:
    """RENAME *table* to *legacy_name*, CREATE it fresh from *ddl*, copy rows back with
    *copy_sql*, DROP the legacy copy, recreate *indexes* — as ONE write transaction.

    The writer connection is autocommit (``isolation_level=None``), so without an explicit
    BEGIN each statement commits on its own and a sibling process opening the same state.db
    between RENAME and CREATE runs SCHEMA_SQL's ``CREATE TABLE IF NOT EXISTS`` first: our
    CREATE then fails with "table already exists", every row is stranded in *legacy_name*
    and the live table is empty. BEGIN IMMEDIATE holds the write lock for the whole rebuild
    so the sibling blocks (and retries under its lock patience) instead of interleaving;
    any failure rolls the RENAME back."""
    conn = cursor.connection
    if conn.in_transaction:  # caller already owns the transaction
        _rebuild_table_statements(cursor, table, legacy_name, ddl, copy_sql, indexes)
        return
    cursor.execute("BEGIN IMMEDIATE")
    try:
        _rebuild_table_statements(cursor, table, legacy_name, ddl, copy_sql, indexes)
    except BaseException:
        cursor.execute("ROLLBACK")
        raise
    cursor.execute("COMMIT")


def _rebuild_table_statements(cursor, table, legacy_name, ddl, copy_sql, indexes) -> None:
    cursor.execute(f"ALTER TABLE {table} RENAME TO {legacy_name}")
    cursor.execute(ddl)
    cursor.execute(copy_sql)
    cursor.execute(f"DROP TABLE {legacy_name}")
    for sql in indexes:
        cursor.execute(sql)


def _heal_gateway_routing_pk(holder: StoreHolder, cursor: sqlite3.Cursor) -> None:
    """Rebuild ``gateway_routing`` when its PRIMARY KEY predates scoping (``session_key TEXT
    PRIMARY KEY``): the reconciler ADDs ``scope`` but SQLite cannot ALTER a PK, so every
    routing write fails (ON CONFLICT mismatch / cross-scope UNIQUE violation). Newest
    row wins a cross-scope session_key collision (INSERT OR REPLACE in updated_at order).

    Early builds of the routing-index migration (#59203) created the table with ``session_key TEXT
    PRIMARY KEY`` and no ``scope`` column. ``_reconcile_columns()`` ADDs the missing ``scope`` column on
    those databases, but SQLite cannot ALTER a primary key, so the shipped composite ``PRIMARY KEY
    (scope, session_key)`` never lands. On such tables every write path is broken:
    """
    pk_cols = _live_pk_columns(cursor, "gateway_routing")
    if pk_cols is None or pk_cols == ["scope", "session_key"]:
        return
    logger.info(
        "gateway_routing has legacy primary key %r; rebuilding with composite (scope, session_key) key", pk_cols,
    )
    _rebuild_table(
        cursor, "gateway_routing", "gateway_routing_legacy_pk",
        """CREATE TABLE gateway_routing (
    scope TEXT NOT NULL DEFAULT '',
    session_key TEXT NOT NULL,
    entry_json TEXT NOT NULL,
    updated_at REAL NOT NULL,
    PRIMARY KEY (scope, session_key)
)""",
        "INSERT OR REPLACE INTO gateway_routing (scope, session_key, entry_json, updated_at) "
        "SELECT COALESCE(scope, ''), session_key, entry_json, updated_at "
        "FROM gateway_routing_legacy_pk ORDER BY updated_at ASC",
    )


def _heal_session_model_usage_pk(holder: StoreHolder, cursor: sqlite3.Cursor) -> None:
    """Rebuild ``session_model_usage`` when its PRIMARY KEY lacks ``task``: installs already at
    v22+ when ``task`` landed carry the 5-column PK, the reconciler ADDs ``task`` but
    SQLite cannot ALTER a PK, and the v22 rebuild is unreachable — every upsert then
    fails (ON CONFLICT mismatch), silently zeroing accounting. Idempotent. FK-off
    window: INSERT OR IGNORE does NOT suppress FK violations, so an orphaned usage row
    would abort the rebuild (PRAGMA foreign_keys is a no-op inside a transaction; none
    is open here). OR IGNORE: COALESCE(task, '') on legacy NULL rows can collide with a
    genuine ''-task row — keep the first.

    Installs whose ``state.db`` reached ``schema_version >= 22`` before the ``task`` dimension was added
    carry a 5-column PRIMARY KEY ``(session_id, model, billing_provider, billing_base_url,
    billing_mode)``. See #73823.
    """
    pk_cols = _live_pk_columns(cursor, "session_model_usage")
    if pk_cols is None or "task" in pk_cols:
        return
    logger.info(
        "session_model_usage has legacy primary key %r (missing task); rebuilding with composite 6-column key",
        sorted(pk_cols),
    )
    cursor.execute("PRAGMA foreign_keys=OFF")
    try:
        _rebuild_table(
            cursor, "session_model_usage", "session_model_usage_legacy_pk", _SESSION_MODEL_USAGE_HEAL_DDL,
            # v20: per-model usage attribution (issue #51607). Going forward update_token_counts()
            # records each API call into session_model_usage keyed by the live model, but existing
            # sessions only have their aggregate totals on the sessions row. Seed one usage row per
            # historical session from those aggregates so insights reads uniformly from the new table.
            # INSERT OR IGNORE keeps it idempotent: if newer code already wrote a (session_id, model,
            # provider) row for a session, the PK conflict skips the stale aggregate rather than
            # doubling it.
            """INSERT OR IGNORE INTO session_model_usage (
                       session_id, model, billing_provider, billing_base_url,
                       billing_mode, task, api_call_count, input_tokens,
                       output_tokens, cache_read_tokens, cache_write_tokens,
                       reasoning_tokens, estimated_cost_usd, actual_cost_usd,
                       cost_status, cost_source, first_seen, last_seen
                   )
                   SELECT session_id, model,
                          COALESCE(billing_provider, ''),
                          COALESCE(billing_base_url, ''),
                          COALESCE(billing_mode, ''),
                          COALESCE(task, ''),
                          api_call_count, input_tokens,
                          output_tokens, cache_read_tokens, cache_write_tokens,
                          reasoning_tokens, estimated_cost_usd, actual_cost_usd,
                          cost_status, cost_source, first_seen, last_seen
                   FROM session_model_usage_legacy_pk""",
            _SESSION_MODEL_USAGE_INDEX_SQL,
        )
    except sqlite3.OperationalError as exc:
        logger.debug("session_model_usage PK heal skipped: %s", exc)
    finally:
        cursor.execute("PRAGMA foreign_keys=ON")


# ── _init_schema ───────────────────────────────────────────────────────

def _init_schema(holder: StoreHolder):
    """Create tables and FTS if missing, reconcile columns, run data migrations. Column
    additions are declarative via _reconcile_columns(), so reordered migrations can
    never skip a column; schema_version remains for data migrations only."""
    # Startup-watchdog lease: on multi-GB files this is I/O-bound (near-zero CPU), which
    # the watchdog's CPU fallback would misread as a parked deadlock.
    # Declare a startup-watchdog progress lease before potentially long synchronous work: on multi-GB
    # state.db files the reconciliation + version-gated data migrations below are legitimately slow and
    # can be I/O-bound (near-zero CPU), which the watchdog's CPU fallback would misread as a parked
    # deadlock (OOF-298 / PR #89750). Single lease is deliberate: this is the one pre-loop phase that
    # can legitimately exceed the 300s default deadline (multi-GB DBs), and the lease is clamped to
    # _MAX_LEASE_S=900. Honest worst case: a genuinely wedged DB init delays supervisor respawn by up to
    # the lease duration. Per-chunk renewal would shrink that, but adds complexity to the migration
    # loops for a rare failure mode.
    report_startup_progress(600.0, phase="state_db_init_schema")
    cursor = holder._conn.cursor()
    cursor.executescript(SCHEMA_SQL)

    # Column reconciliation, then the two table-shape repairs ADD COLUMN cannot express.
    _reconcile_columns(cursor, holder.db_path.parent)
    _heal_gateway_routing_pk(holder, cursor)
    # Rebuild session_model_usage if its PRIMARY KEY lacks the ``task`` column (5-column PK on installs
    # already at v22+ when the column landed — the version-gated rebuild is unreachable there, #73823).
    # Same PK-rebuild constraint as gateway_routing above.
    _heal_session_model_usage_pk(holder, cursor)

    # Indexes referencing reconciler-added columns must be created AFTER _reconcile_columns
    # (in SCHEMA_SQL the executescript would fail on legacy DBs).
    # Heal NULL ``active`` rows unconditionally on every startup. On real-world DBs the reconciler-added
    # ``active`` column can lack its NOT NULL DEFAULT 1 (older reconciler builds reconstructed the type
    # without the default — see #51646: PRAGMA shows (17,'active','INTEGER',0,None,0) in the wild), so
    # INSERTs that omitted the column wrote NULL and the ``WHERE active = 1`` transcript loaders hid the
    # whole history. The INSERTs now set active=1 explicitly; this idempotent repair un-hides rows
    # written before the fix. It was previously gated at ``current_version < 12`` which never re-ran for
    # already-v12+ databases.
    try:
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_messages_platform_msg_id "
            "ON messages(session_id, platform_message_id) WHERE platform_message_id IS NOT NULL"
        )
    except sqlite3.OperationalError as exc:
        logger.debug("idx_messages_platform_msg_id create skipped: %s", exc)
    _execute_ddl_skipping_settled_triggers(cursor, DEFERRED_INDEX_SQL)  # same ordering constraint (``active``)

    # Heal NULL ``active`` rows on every startup: older reconciler builds added ``active``
    # without NOT NULL DEFAULT 1, so ``WHERE active = 1`` loaders hid whole histories. A
    # ``current_version < 12`` gate never re-ran for already-v12+ databases.
    # Read before writing: an UPDATE takes the write lock even when it matches no rows, so
    # the unconditional form blocked every open behind a sibling's write transaction. The
    # probe is short-circuited on the modern NOT NULL column and index-served on legacy
    # ones. Deliberately not INDEXED BY (raises "no query solution" on NOT NULL columns).
    with contextlib.suppress(sqlite3.OperationalError):
        if cursor.execute("SELECT 1 FROM messages WHERE active IS NULL LIMIT 1").fetchone() is not None:
            cursor.execute("UPDATE messages SET active = 1 WHERE active IS NULL")

    fts5_available = _sqlite_supports_fts5(holder, cursor)
    stale_row = cursor.execute("SELECT 1 FROM state_meta WHERE key = ? LIMIT 1", (FTS_STALE_KEY,)).fetchone()
    holder._fts_stale = stale_row is not None
    if holder._fts_stale:
        # A prior process detached FTS after corruption; stay detached until a full rebuild.
        _drop_all_fts_triggers(holder, cursor)
    if not fts5_available:
        # Existing FTS triggers would still fire though this runtime cannot read their
        # targets. Drop only the triggers; a future FTS5 runtime recreates them.
        _drop_fts_triggers(cursor)

    row = cursor.execute("SELECT version FROM schema_version LIMIT 1").fetchone()
    if row is None:
        cursor.execute("INSERT INTO schema_version (version) VALUES (?)", (SCHEMA_VERSION,))
        # Store provenance so fresh vs wiped stores are distinguishable.
        # See #97568.
        now_iso = datetime.datetime.now(datetime.UTC).isoformat()
        cursor.executemany(
            "INSERT OR IGNORE INTO state_meta (key, value) VALUES (?, ?)",
            [("store_instance_id", str(uuid.uuid4())), ("store_created_at_utc", now_iso),
             (_MESSAGE_UID_BACKFILL_DONE, "1")],  # a fresh store has no legacy rows to backfill
        )
    else:
        _run_data_migrations(holder, cursor, row[0], fts5_available)

    _ensure_unique_title_index(holder, cursor)
    if fts5_available:
        _init_fts(holder, cursor)
        _stamp_settled_fts_layout(holder, cursor)
    holder._conn.commit()


def _stamp_settled_fts_layout(holder: StoreHolder, cursor: sqlite3.Cursor) -> None:
    """Write ``fts_storage_version`` once the stored layout is exactly current and nothing is in flight.

    The only open-time writer of the stamp, run after _init_fts so a fresh store is stamped on creation
    and a realigned base index is stamped only when the trigram also matches. A definition this code does
    not recognise (a newer release's) is never stamped down to this release's version. An interrupted
    optimize (markers, trash, an empty external index against non-empty messages) keeps the stamp
    unwritten so the resume offer stays alive. Reads before writing: a settled store never takes the write
    lock here.
    """
    if (
        holder._fts_stale
        or not fts_layout_current(cursor)
        or _meta_row(cursor, "fts_rebuild_high_water") is not None
        or _has_fts_trash(holder, cursor)
        or _fts_external_index_empty_with_messages(holder, cursor)
    ):
        return
    if cursor.execute(
        "SELECT 1 FROM state_meta WHERE key = 'fts_storage_version' AND value = ? LIMIT 1",
        (str(FTS_STORAGE_VERSION),),
    ).fetchone() is None:
        cursor.execute(_STATE_META_UPSERT_SQL, ("fts_storage_version", str(FTS_STORAGE_VERSION)))


def _run_data_migrations(holder: StoreHolder, cursor: sqlite3.Cursor, current_version: int, fts5_available: bool) -> None:
    """Version-gated chain for DATA migrations only (row backfills); column additions never
    belong here. Advances schema_version at the end unless FTS5 is unavailable."""
    # Renew the lease: the chain can rewrite whole tables on large DBs.
    report_startup_progress(600.0, phase="state_db_data_migrations")
    # (v10 trigram backfill and v11 inline FTS re-index were superseded by v23 and removed.)
    # v11 (SUPERSEDED by v23): re-index FTS5 tables to cover tool_name + tool_calls in inline mode
    # (#16751). v23 drops and rebuilds both FTS tables in external-content form, so running the v11
    # inline backfill first would only burn startup time and WAL space before v23 throws the work away —
    # and its inline INSERT shape no longer matches the current external-content FTS_SQL anyway. Kept
    # only for source archaeology; unreachable while SCHEMA_VERSION >= 23.
    if current_version < 16:
        # v16: tag delegate subagent rows so pickers stay clean after parent deletes orphan them.
        with contextlib.suppress(sqlite3.OperationalError):
            cursor.execute(
                "UPDATE sessions SET model_config = json_set("
                "COALESCE(model_config, '{}'), '$._delegate_from', parent_session_id) "
                f"WHERE parent_session_id IS NOT NULL "
                f"AND {_sql_json_extract('model_config', '$._delegate_from')} IS NULL "
                f"AND {_ephemeral_child_sql('sessions')}"
            )
            cursor.execute(
                "UPDATE sessions SET model_config = json_set("
                "COALESCE(model_config, '{}'), '$._delegate_from', '__orphaned__') WHERE parent_session_id IS NULL "
                f"AND {_sql_json_extract('model_config', '$._delegate_from')} IS NULL "
                f"AND {_sql_json_extract('model_config', '$._branched_from')} IS NULL "
                "AND title IS NULL AND message_count <= 25 AND EXISTS (SELECT 1 FROM messages m "
                "            WHERE m.session_id = sessions.id AND m.role = 'tool') "
                "AND NOT EXISTS (SELECT 1 FROM sessions ch "
                "                WHERE ch.parent_session_id = sessions.id)"
            )
    if current_version < 18:
        # v18: best-effort gateway metadata backfill from sessions.json.
        try:
            # Backfill display_name / origin_json / expiry_finalized from sessions.json so pre-migration
            # gateway sessions are discoverable from state.db without the JSON index. See #9006.
            _backfill_gateway_metadata_from_sessions_json(holder, cursor)
        except Exception as exc:  # health: allow BLE001 -- the sessions.json backfill is best-effort
            logger.debug("v18 gateway metadata backfill skipped: %s", exc)
    if current_version < 20:
        # v20: seed session_model_usage from sessions aggregates (OR IGNORE: newer rows win).
        with contextlib.suppress(sqlite3.OperationalError):
            cursor.execute(_SESSION_MODEL_USAGE_V20_SEED_SQL)
    if current_version < 22:
        _migrate_v22_session_model_usage(holder, cursor)
    # v23: FTS storage redesign (external-content tables). OPT-IN, NOT AUTOMATIC: the
    # transition is disk-heavy (~2x transient) and long (hours on 25 GB), so an existing
    # install only gets a flag; `hermes sessions optimize-storage` performs it. The FTS
    # layout is tracked by the independent `fts_storage_version` marker, so
    # schema_version still advances for legacy-FTS users.
    if current_version < 23 and fts5_available and _db_needs_fts_storage_upgrade(cursor):
        holder.set_meta("fts_optimize_available", "1", cursor=cursor)
    if current_version < 25:
        # v25: de-duplicate system prompt snapshots (old column stays a read fallback).
        _dedupe_legacy_system_prompts(holder, cursor)
    _advance_message_uid_backfill(holder, cursor)
    fts_migrations_complete = True
    if current_version < 30 and fts5_available:
        # v29: cron sessions leave the trigram substring index (they stay in the word index);
        # v30: delegate-child transcripts too (FTS_TRIGRAM_EXCLUDED_SOURCES + _delegate_from).
        # Rebuild once so rows indexed by older view/trigger definitions do not linger.
        fts_migrations_complete = _migrate_trigram_cron_exclusion(holder, cursor)

    # The FTS layout stamp is written by _stamp_settled_fts_layout after _init_fts, not here.
    # v23: FTS storage redesign (issues #22478, #43690, #55233). The v11 inline-mode FTS tables each
    # store a full private copy of every message (content || tool_name || tool_calls), and the trigram
    # index additionally covers role='tool' rows (~90% of message bytes: base64 payloads, file dumps) at
    # ~2.6x amplification — together ~75% of state.db on heavy installs (observed: 18.9 GB of a 25 GB
    # DB). OPT-IN, NOT AUTOMATIC. The transition (demote old vtables → new external-content schema →
    # backfill → teardown → VACUUM) is disk-heavy (transient ~2x file size to fully reclaim via VACUUM)
    # and long (~1-2h background on a 25 GB DB). Doing it silently on every big user's next open — with
    # a completeness guarantee that depends on the process staying alive long enough — is the wrong
    # default. So on an EXISTING install we touch nothing here: the v22 inline FTS keeps working exactly
    # as before, and we only record a flag advertising that the optimization is available. `hermes
    # sessions optimize-storage` performs the whole transition as one deliberate, disk-checked,
    # progress-reported foreground operation. DECOUPLED VERSIONING. Crucially, this does NOT hold back
    # the main schema_version. The FTS storage LAYOUT is tracked by an independent `fts_storage_version`
    # marker (see _fts_storage_version / SETTLE below), so schema_version advances to SCHEMA_VERSION
    # here like every other migration — future v24+ migrations land automatically for legacy-FTS users
    # too. Only the FTS *layout* waits for opt-in.
    # v32: nothing to run here. The declared tables/columns #106742 added under v31 were already
    # created by _init_schema's executescript(SCHEMA_SQL) + _reconcile_columns() before this chain;
    # the version only records that the store has them (#140315).
    # Advance schema_version — deliberately NOT gated on the FTS opt-in (that would block
    # every future migration for a user who never optimizes). FTS5 unavailable is the
    # one skip: claiming current would lie.
    if current_version < SCHEMA_VERSION and fts_migrations_complete and fts5_available:
        cursor.execute("UPDATE schema_version SET version = ?", (SCHEMA_VERSION,))


def _advance_message_uid_backfill(holder: StoreHolder, cursor: sqlite3.Cursor) -> None:
    """Mint ``message_uid`` onto rows written before the column existed, one bounded slice per open.

    A single full-table UPDATE held the write lock for 261 s on an 835k-row / 4.8 GB store, so every
    sibling process timed out with "database is locked". Rowid-range chunks commit one at a time (the
    writer connection is autocommit) and an open spends at most ``_MESSAGE_UID_BACKFILL_BUDGET_S``
    after its first chunk, so a large store converges over later opens. Until then a legacy row
    simply has no uid, which every reader already treats as "no identity yet".

    The walk is keyed on ``rowid``, not ``id``: on the declared table ``id INTEGER PRIMARY KEY`` is the
    rowid alias (same plan), but a store converted from JSONL still carries ``id TEXT`` with no PRIMARY
    KEY (``_reconcile_columns`` adds columns, never rebuilds the key), where ``MAX(id)`` is a str and an
    ``id`` range compares lexicographically (#141467)."""
    if _meta_row(cursor, _MESSAGE_UID_BACKFILL_DONE) is not None:
        return
    row = _meta_row(cursor, _MESSAGE_UID_BACKFILL_CURSOR)
    done_through = int(row[0]) if row else 0
    high = cursor.execute("SELECT MAX(rowid) FROM messages").fetchone()[0] or 0
    deadline = time.monotonic() + _MESSAGE_UID_BACKFILL_BUDGET_S
    while done_through < high:
        upper = done_through + _MESSAGE_UID_BACKFILL_CHUNK
        cursor.execute(
            "UPDATE messages SET message_uid = lower(hex(randomblob(16))) "
            "WHERE rowid > ? AND rowid <= ? AND message_uid IS NULL", (done_through, upper))
        done_through = upper
        if done_through < high and time.monotonic() >= deadline:
            holder.set_meta(_MESSAGE_UID_BACKFILL_CURSOR, str(done_through), cursor=cursor)
            return
    holder.set_meta(_MESSAGE_UID_BACKFILL_DONE, "1", cursor=cursor)
    _delete_meta(cursor, _MESSAGE_UID_BACKFILL_CURSOR)


def _migrate_v22_session_model_usage(holder: StoreHolder, cursor: sqlite3.Cursor) -> None:
    """v22: ``task`` joins the session_model_usage PRIMARY KEY ('' = main loop; aux calls
    named). SQLite cannot ALTER a PK, so rebuild; existing rows → task=''."""
    try:
        # v22: task-dimension usage attribution (issue #23270). session_model_usage gains a ``task``
        # column ('' = main agent loop; 'vision'/'compression'/'title_generation'/... = auxiliary calls)
        # so aux model spend is visible in analytics. The reconciler will have already ADDed the plain
        # column on legacy DBs (harmless); the rebuild bakes it into the PK properly.
        legacy_pk = cursor.execute(
            "SELECT COUNT(*) FROM pragma_table_info('session_model_usage') WHERE name = 'task' AND pk > 0"
        ).fetchone()[0]
        if legacy_pk:
            return
        _rebuild_table(
            cursor, "session_model_usage", "session_model_usage_v21", _SESSION_MODEL_USAGE_V22_DDL,
            """INSERT INTO session_model_usage (
                                   session_id, model, billing_provider, billing_base_url,
                                   billing_mode, task, api_call_count, input_tokens,
                                   output_tokens, cache_read_tokens, cache_write_tokens,
                                   reasoning_tokens, estimated_cost_usd, actual_cost_usd,
                                   cost_status, cost_source, first_seen, last_seen
                               )
                               SELECT session_id, model, billing_provider, billing_base_url,
                                      billing_mode, '', api_call_count, input_tokens,
                                      output_tokens, cache_read_tokens, cache_write_tokens,
                                      reasoning_tokens, estimated_cost_usd, actual_cost_usd,
                                      cost_status, cost_source, first_seen, last_seen
                               FROM session_model_usage_v21""",
            _SESSION_MODEL_USAGE_INDEX_SQL,
        )
    except sqlite3.OperationalError as exc:
        logger.debug("v22 session_model_usage rebuild skipped: %s", exc)


def _ensure_unique_title_index(holder: StoreHolder, cursor: sqlite3.Cursor) -> None:
    """Unique title index. Older DBs may hold duplicate titles from before the constraint.
    Per title a user-owned (user/NULL) row beats llm > derived (``_title_rank``). User rows are
    never dropped: the oldest keeps the title and newer ones become ``Title #k`` in started_at
    order, so the "#N"-preferring lookup still opens the newest (#126764). Among auto titles
    only, the newest highest-ranked keeps it; lower-ranked auto titles are cleared. Each change
    is logged. Must never abort opening the DB, so the repair is guarded."""
    try:
        cursor.execute(_TITLE_UNIQUE_INDEX_SQL)
    except sqlite3.IntegrityError:
        # Savepoint: a failure part-way must not leave a half-repaired store for
        # _init_schema to commit without the index.
        cursor.execute("SAVEPOINT title_repair")
        try:
            rows = cursor.execute(
                "SELECT rowid, title, title_source, started_at FROM sessions WHERE title IN "
                "(SELECT title FROM sessions WHERE title IS NOT NULL GROUP BY title HAVING COUNT(*) > 1)"
            ).fetchall()
            groups: dict[str, list] = {}
            for row in sorted(rows, key=lambda r: (_title_rank(r[2]), r[3], r[0]), reverse=True):
                groups.setdefault(row[1], []).append(row)
            user_rank = _TITLE_SOURCE_RANK[TITLE_SOURCE_USER]
            for title, group in groups.items():
                # User titles: the OLDEST keeps the base and newer ones get "#k" in
                # chronological order, so resolve_session_by_title (which prefers the
                # latest "#N") still lands on the newest session.
                users = [r for r in reversed(group) if _title_rank(r[2]) == user_rank]
                keep = users[0] if users else group[0]
                for rowid, *_ in users[1:]:
                    renamed = next_title_in_lineage(cursor.connection, title)
                    cursor.execute("UPDATE sessions SET title = ? WHERE rowid = ?", (renamed, rowid))
                    logger.warning("Renamed duplicate user session title %r to %r", title, renamed)
                for rowid, _title, source, _started in group:
                    if rowid != keep[0] and _title_rank(source) != user_rank:
                        cursor.execute("UPDATE sessions SET title = NULL WHERE rowid = ?", (rowid,))
                        logger.warning("Cleared duplicate %s session title %r", source, title)
            cursor.execute(_TITLE_UNIQUE_INDEX_SQL)
            cursor.execute("RELEASE title_repair")
        except sqlite3.Error:
            cursor.execute("ROLLBACK TO title_repair")
            cursor.execute("RELEASE title_repair")
            logger.exception("Could not repair duplicate session titles; unique title index not created")
    except sqlite3.OperationalError:
        pass  # Index already exists


def _init_fts(holder: StoreHolder, cursor: sqlite3.Cursor) -> None:
    """Create/repair the FTS objects on an FTS5-capable runtime. The DDL runs even when the
    vtable exists so CREATE TRIGGER IF NOT EXISTS repairs trigger-only degradation.
    OPT-IN v23 boundary: a legacy v22 inline install keeps its inline schema + triggers
    (the v23 DDL would create the trigram source VIEW and leave a mixed state)."""
    legacy_fts = _db_has_legacy_inline_fts(cursor)
    # A `.recover`-restored image keeps the shadow tables but not the vtable rows; the DDL
    # below would fail on the first shadow. Drop only orphaned families, then rebuild the
    # recreated (empty) index like a missing-trigger repair (#103840).
    orphan_repaired = _drop_orphan_fts_shadow_tables(
        cursor, ("messages_fts", "messages_fts_trigram", "messages_fts_cjk"),
    )
    if not holder._fts_stale:
        _migrate_misaligned_fts_source(holder, cursor, legacy=legacy_fts)
    if holder._fts_stale:
        if _recover_stale_fts(holder, cursor, legacy=legacy_fts):
            # CJK was detached alongside the base indexes; its ensure path decides when it returns.
            _ensure_fts_cjk_schema(holder, cursor)
        else:
            holder._fts_enabled = holder._trigram_available = holder._fts_cjk_available = False
    else:
        base_sql, trigram_sql = _FTS_DDL[legacy_fts]
        # Measure before any DDL. Publishing missing base triggers before rebuild admission lets
        # another process write through an index whose bootstrap/repair has no owner (#105790).
        base_triggers_missing = (
            _fts_triggers_missing(cursor, _FTS_BASE_TRIGGERS) or "messages_fts" in orphan_repaired
        )
        trigram_triggers_missing = (
            _fts_triggers_missing(cursor, _FTS_TRIGRAM_TRIGGERS) or "messages_fts_trigram" in orphan_repaired
        )

        def ensure_and_rebuild() -> None:
            holder._fts_enabled = _ensure_fts_schema(holder, cursor, "messages_fts", base_sql)
            if not holder._fts_enabled:
                return
            holder._trigram_available = _ensure_fts_schema(holder, cursor, "messages_fts_trigram", trigram_sql)
            _rebuild_fts_indexes(
                cursor, legacy=legacy_fts, include_trigram=holder._trigram_available,
            )
            if not legacy_fts:
                _ensure_fts_cjk_schema(holder, cursor)

        if base_triggers_missing:
            # The authority covers the whole first-publication sequence, not merely the final rebuild.
            # ``executescript`` commits DDL statement-by-statement, so acquiring after ensure exposed a
            # partially initialized FTS family while another opener held the rebuild lock.
            _run_admitted_startup_rebuild(holder, cursor, ensure_and_rebuild)
        else:
            holder._fts_enabled = _ensure_fts_schema(holder, cursor, "messages_fts", base_sql)
            if holder._fts_enabled:
                # Trigram is optional; without it CJK search falls back to LIKE.
                trigram_enabled = _ensure_fts_schema(holder, cursor, "messages_fts_trigram", trigram_sql)
                holder._trigram_available = trigram_enabled
                if trigram_enabled and trigram_triggers_missing:
                    _run_admitted_startup_rebuild(holder,
                        cursor,
                        lambda: _rebuild_fts_indexes(
                            cursor, legacy=legacy_fts, include_trigram=trigram_enabled,
                        ),
                    )
        if holder._fts_enabled and not legacy_fts and not base_triggers_missing:
            # CJK-bigram index: strictly additive, gated on the loadable tokenizer.
            _ensure_fts_cjk_schema(holder, cursor)
    # IF NOT EXISTS cannot rewrite pre-existing broad AFTER UPDATE triggers.
    if holder._fts_enabled:
        _migrate_broad_fts_update_triggers(holder, cursor)


def _run_admitted_startup_rebuild(holder: StoreHolder, cursor, rebuild_fn) -> None:
    """Run FTS bootstrap or trigger-repair rebuild under cross-process admission.

    The fresh/base-missing path includes DDL in ``rebuild_fn`` so no process can publish
    triggers before it owns the rebuild. Other repair paths may already have recreated an
    optional trigger; deferral therefore still drops every trigger and persists the stale
    breadcrumb. A later recovery path restores the complete family.

    See #93200.
    See #105790.
    """
    with fts_rebuild_admission(holder.db_path) as admitted:
        if admitted:
            rebuild_fn()
            return
    logger.warning(
        "Deferred startup FTS rebuild: another process holds the "
        "rebuild authority for this state.db; detaching FTS sync until the stale-index recovery path rebuilds it."
    )
    cursor.execute(_STALE_KEY_UPSERT_SQL, (FTS_STALE_KEY,))
    _drop_all_fts_triggers(holder, cursor)
    holder._fts_stale = True
    holder._fts_enabled = holder._trigram_available = holder._fts_cjk_available = False


def _backfill_gateway_metadata_from_sessions_json(holder: StoreHolder, cursor: sqlite3.Cursor) -> None:
    """One-time v18 backfill of gateway metadata from sessions.json. Only fills NULL
    columns — never overwrites data written by newer code."""
    sessions_file = holder.db_path.parent / "sessions" / "sessions.json"
    if not sessions_file.exists():
        return
    with open(sessions_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        return
    for key, entry in data.items():
        if str(key).startswith("_") or not isinstance(entry, dict):
            continue
        session_id = entry.get("session_id")
        if not session_id:
            continue
        origin = entry.get("origin")
        origin_dict = origin if isinstance(origin, dict) else None
        cursor.execute(
            """UPDATE sessions
                   SET session_key = COALESCE(session_key, ?),
                       chat_id = COALESCE(chat_id, ?),
                       chat_type = COALESCE(chat_type, ?),
                       thread_id = COALESCE(thread_id, ?),
                       display_name = COALESCE(display_name, ?),
                       origin_json = COALESCE(origin_json, ?),
                       expiry_finalized = CASE
                           WHEN COALESCE(expiry_finalized, 0) = 0 AND ? = 1 THEN 1
                           ELSE expiry_finalized
                       END
                   WHERE id = ?""",
            (
                entry.get("session_key") or key, origin_dict.get("chat_id") if origin_dict is not None else None,
                entry.get("chat_type"), origin_dict.get("thread_id") if origin_dict is not None else None,
                entry.get("display_name"), json.dumps(origin) if origin_dict is not None else None,
                1 if entry.get("expiry_finalized") or entry.get("memory_flushed") else 0, str(session_id),
            ),
        )


def _is_fts5_unavailable_error(exc: sqlite3.OperationalError) -> bool:
    """No FTS5 module, or an optional tokenizer missing (same capability-error shape)."""
    err = str(exc).lower()
    return ("no such module" in err and "fts5" in err) or _is_trigram_unavailable_error(exc)


def _is_trigram_unavailable_error(exc: sqlite3.OperationalError) -> bool:
    """Only an optional tokenizer is missing (trigram needs SQLite >= 3.34;
    cjk_unicode61 is loadable): "this one index can't be served", never "disable FTS"."""
    err = str(exc).lower()
    return "no such tokenizer: trigram" in err or "no such tokenizer: cjk_unicode61" in err


def _db_has_legacy_inline_fts(cursor: sqlite3.Cursor) -> bool:
    """messages_fts exists in ANY pre-v23 shape: every legacy shape lacks
    tool_name, so "stored CREATE lacks tool_name" catches them all. False when absent.

    v23's messages_fts is external-content over THREE real columns (content, tool_name, tool_calls).
    Every pre-v23 shape lacks the tool_name/tool_calls columns — whether the old inline single-column
    form (v11..v22) or the even older external-content single-column form (v10-era, pre-#16751). We
    therefore detect "needs optimize" as "the stored CREATE lacks the tool_name column", which is the
    precise v23 marker and correctly catches BOTH legacy variants.
    """
    row = cursor.execute(
        "SELECT sql FROM sqlite_master WHERE type = 'table' AND name = 'messages_fts'"
    ).fetchone()
    return row is not None and "tool_name" not in (row[0] or "")


def _db_has_trigram_tool_calls_projection(cursor: sqlite3.Cursor) -> bool:
    """True when the trigram vtable still includes the tool_calls payload (FTS_STORAGE_VERSION 1)."""
    row = cursor.execute(
        "SELECT sql FROM sqlite_master WHERE type = 'table' AND name = 'messages_fts_trigram'"
    ).fetchone()
    return row is not None and "tool_calls" in (row[0] or "").lower()


def _db_needs_fts_storage_upgrade(cursor: sqlite3.Cursor) -> bool:
    """True when optimize-storage has layout work: a stored FTS vtable differs from what this code creates."""
    return bool(fts_layout_drift(cursor))


def _warn_trigram_unavailable(holder: StoreHolder, exc: sqlite3.OperationalError) -> None:
    """Log once that the trigram tokenizer is missing; base FTS5 stays enabled."""
    if getattr(holder, "_trigram_unavailable_warned", False):  # attr is lazily created here
        return
    holder._trigram_unavailable_warned = True
    logger.info(
        "SQLite trigram tokenizer unavailable for %s "
        "(requires SQLite >= 3.34, this build is %s); "
        "CJK/substring search will fall back to LIKE: %s",
        holder.db_path,
        sqlite3.sqlite_version,
        exc,
    )


def _warn_fts5_unavailable(holder: StoreHolder, exc: sqlite3.OperationalError) -> None:
    holder._fts_enabled = False
    if holder._fts_unavailable_warned:
        return
    holder._fts_unavailable_warned = True
    logger.warning(
        "SQLite FTS5 unavailable for %s; full-text session search "
        "disabled. Run `hermes update` to rebuild the venv with a "
        "current Python (managed uv guarantees FTS5). (underlying error: %s)",
        holder.db_path,
        exc,
    )


def _ensure_fts_cjk_schema(holder: StoreHolder, cursor) -> None:
    """Create / repair / self-heal the CJK-bigram index (see the module comment).
    Sets ``_fts_cjk_available``; never raises. Loaded + absent → create (a
    populated DB gets backfill markers and is NOT served until optimize-storage
    backfills); loaded + present → ensure triggers, honour the stale breadcrumb;
    NOT loaded + live triggers → drop them (INSERTs must not fail at trigger time)."""
    try:
        cjk_present = bool(cursor.execute(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'messages_fts_cjk'"
        ).fetchone())
        if not holder._fts_cjk_loaded:
            if cjk_present:
                live = [r[0] for r in cursor.execute(
                    "SELECT name FROM sqlite_master WHERE type = 'trigger' "
                    f"AND name IN ({','.join('?' for _ in _FTS_CJK_TRIGGERS)})",
                    _FTS_CJK_TRIGGERS,
                ).fetchall()]
                if live:
                    # Breadcrumb FIRST (a crash between the two is merely conservative).
                    logger.warning(
                        "messages_fts_cjk triggers present but the "
                        "cjk_unicode61 tokenizer is unavailable (%s) — "
                        "dropping the cjk triggers so message writes keep "
                        "working. CJK search falls back to trigram/LIKE; "
                        "run `hermes sessions optimize-storage` on a host "
                        "with the extension to rebuild.",
                        fts5_cjk_so_path(holder.db_path.parent),
                    )
                    cursor.execute(
                        "INSERT INTO state_meta (key, value) VALUES (?, '1') "
                        "ON CONFLICT(key) DO UPDATE SET value = '1'",
                        (FTS_CJK_STALE_KEY,),
                    )
                    for trig in live:
                        cursor.execute(f"DROP TRIGGER IF EXISTS {trig}")
            holder._fts_cjk_available = False
            return
    except sqlite3.OperationalError:
        logger.warning(
            "messages_fts_cjk presence check failed; CJK search stays on "
            "trigram/LIKE", exc_info=True,
        )
        holder._fts_cjk_available = False
        return
    try:
        cursor.executescript(FTS_CJK_TABLE_SQL)
        if not cjk_present:
            # An old stale breadcrumb refers to a table that no longer exists.
            cursor.execute("DELETE FROM state_meta WHERE key = ?", (FTS_CJK_STALE_KEY,))
            # Empty DB: complete by construction, no markers. Populated DB: the
            # marker pair keeps the id-gated triggers correct until backfill.
            if cursor.execute("SELECT COUNT(*) FROM messages WHERE role <> 'tool'").fetchone()[0] > 0:
                hw = cursor.execute("SELECT COALESCE(MAX(id), 0) FROM messages").fetchone()[0]
                for k, v in (
                    ("fts_cjk_rebuild_high_water", str(hw)), ("fts_cjk_rebuild_progress", "0"),
                ):
                    cursor.execute(
                        "INSERT INTO state_meta (key, value) VALUES (?, ?) "
                        "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                        (k, v),
                    )
        if cursor.execute("SELECT 1 FROM state_meta WHERE key = ?", (FTS_CJK_STALE_KEY,)).fetchone():
            # Gap of unknown extent: do NOT reinstall triggers (see module comment).
            holder._fts_cjk_available = False
            return
        cursor.executescript(FTS_CJK_TRIGGER_SQL)
        backfill_pending = cursor.execute(
            "SELECT 1 FROM state_meta WHERE key = 'fts_cjk_rebuild_high_water' LIMIT 1"
        ).fetchone()
        holder._fts_cjk_available = not backfill_pending
    except sqlite3.OperationalError:  # incl. "no such tokenizer" after a failed registration
        logger.warning(
            "messages_fts_cjk ensure failed; CJK search stays on "
            "trigram/LIKE", exc_info=True,
        )
        holder._fts_cjk_available = False


def _drop_fts_triggers(cursor: sqlite3.Cursor) -> None:
    for trigger in _FTS_TRIGGERS:
        try:
            cursor.execute(f"DROP TRIGGER IF EXISTS {trigger}")
        except sqlite3.OperationalError:
            pass


def _ensure_fts_schema(holder: StoreHolder, cursor: sqlite3.Cursor, table_name: str, ddl: str) -> bool:
    status = _fts_table_probe(holder, cursor, table_name)
    if status is None:
        return False
    try:
        # Run even when the table exists: recreates triggers a no-FTS5 runtime dropped.
        cursor.executescript(ddl)
        return True
    except sqlite3.OperationalError as exc:
        if not _is_fts5_unavailable_error(exc):
            raise
        # A missing tokenizer disables only that table; the base FTS5 table is fine.
        if _is_trigram_unavailable_error(exc):
            _warn_trigram_unavailable(holder, exc)
        else:
            _warn_fts5_unavailable(holder, exc)
        return False


def _is_fts_write_corruption_error(exc: sqlite3.DatabaseError) -> bool:
    """Corruption SQLite identifies as FTS-scoped (SQLITE_CORRUPT_VTAB, or an ``fts5:``
    report naming ``messages_fts*`` on builds without result codes); a bare malformed
    image is structural. One rule, shared with ``classify_persistence_error`` and the
    gateway transcript retry: see :func:`hermes_state_errors.is_fts_scoped_corruption_error`."""
    return is_fts_scoped_corruption_error(exc)


def _enter_fts_fail_open(
    holder: StoreHolder, exc: sqlite3.DatabaseError, *, deadline: float | None = None, patience_s: float | None = None,
) -> bool:
    """Detach corrupt FTS indexes so canonical writes can continue. Breadcrumb +
    trigger drop commit atomically: once triggers are absent the index has a
    gap of unknown extent, so nobody may reinstall them without a full rebuild.

    A busy write lock is waited out on the caller's write budget (default
    ``_WRITE_PATIENCE_S``), like ``_execute_write``: the writer connection's busy
    timeout is only 1 s, and the usual holder is a sibling writer detaching the
    same corrupt index — giving up after 1 s cost that turn's canonical write."""
    if not holder._fts_enabled or not _is_fts_write_corruption_error(exc):
        return False
    if patience_s is None:
        patience_s = holder._WRITE_PATIENCE_S
    if deadline is None:
        deadline = time.monotonic() + patience_s
    while True:
        # Re-checked every attempt: a sibling may quarantine the file while we wait for the
        # lock, and nothing may be committed on a quarantined handle.
        holder._raise_if_db_corrupt(storage=True)
        try:
            with holder._lock:
                holder._raise_if_db_replaced()
                if holder._conn is None:
                    holder._reopen_after_close_locked(context="write")
                holder._conn.execute("BEGIN IMMEDIATE")
                try:
                    holder._conn.execute(
                        "INSERT INTO state_meta (key, value) VALUES (?, '1') "
                        "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                        (FTS_STALE_KEY,),
                    )
                    cjk_triggers_present = holder._conn.execute(
                        "SELECT 1 FROM sqlite_master WHERE type = 'trigger' "
                        f"AND name IN ({','.join('?' for _ in _FTS_CJK_TRIGGERS)}) "
                        "LIMIT 1",
                        _FTS_CJK_TRIGGERS,
                    ).fetchone()
                    if cjk_triggers_present:
                        holder._conn.execute(
                            "INSERT INTO state_meta (key, value) VALUES (?, '1') "
                            "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                            (FTS_CJK_STALE_KEY,),
                        )
                    _drop_all_fts_triggers(holder, holder._conn.cursor())
                    holder._conn.commit()
                except BaseException:
                    holder._conn.rollback()
                    raise
            break
        except sqlite3.Error as detach_exc:
            if (
                isinstance(detach_exc, sqlite3.OperationalError) and is_sqlite_lock_error(detach_exc)
                and holder._sleep_before_write_retry(deadline, patience_s)
            ):
                continue
            logger.error(
                "Could not detach corrupt FTS indexes; canonical write still cannot proceed: %s",
                detach_exc,
            )
            return False
    holder._fts_stale = True
    holder._fts_enabled = False
    holder._trigram_available = False
    holder._fts_cjk_available = False
    logger.error(
        "state.db FTS indexes remain corrupt (%s); disabled FTS sync and "
        "retrying the canonical write. Search temporarily uses LIKE until "
        "a later SessionDB open rebuilds the indexes.",
        exc,
    )
    return True


# Demoted v22 FTS shadow tables awaiting teardown: DROP of a multi-GB vtable
# blocks for minutes, so the v23 migration renames the orphaned shadow tables
# to fts_v22_trash_*; the worker empties them in chunks, then drops.
_FTS_TRASH_PREFIX = "fts_v22_trash_"


def _has_fts_trash(holder: StoreHolder, conn) -> bool:
    """True when demoted v22 shadow tables are still awaiting teardown.
    Caller must hold ``self._lock`` (or pass a migration-time cursor)."""
    return bool(conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name LIKE ? ESCAPE '\\' LIMIT 1",
        (_FTS_TRASH_PREFIX.replace("_", "\\_") + "%",),
    ).fetchone())


# FTS5 tables merged on optimize; each is probed before touching (trigram may
# be disabled, cjk exists only with the loadable tokenizer).
_FTS_TABLES = ("messages_fts", "messages_fts_trigram", "messages_fts_cjk")


def _try_incremental_merge_fts(holder: StoreHolder) -> None:
    """One bounded FTS5 merge pass that never fails the already-committed write (a caller
    must never replay an ambiguous, possibly-durable write — even on the bare
    SystemError CPython's sqlite3 layer can raise under cross-thread errmsg scrambling)."""
    if not holder._fts_enabled:
        return
    try:
        _merge_fts_incrementally(holder, max_pages=holder._FTS_MERGE_MAX_PAGES_PER_INDEX)
    except Exception as exc:  # health: allow BLE001 -- the write has committed; no merge failure may escape
        # The canonical write is already committed before this cadence runs. No maintenance failure —
        # including the bare SystemError the CPython sqlite3 layer can raise under cross-thread errmsg
        # scrambling — may escape and make the caller replay an ambiguous, possibly-durable write
        # (#90734, #85079).
        logger.warning("FTS incremental merge failed after commit: %s", exc)


def _fts_external_index_empty_with_messages(holder: StoreHolder, conn) -> bool:
    """True when the base FTS table indexes nothing while ``messages`` has rows (post-demote
    empty-index shape). Caller holds ``self._lock``. docsize is the authoritative "is this
    rowid indexed" surface; EXISTS not COUNT(*) because this runs on every writable open."""
    try:
        if not conn.execute("SELECT EXISTS(SELECT 1 FROM messages)").fetchone()[0]:
            return False
        return not conn.execute("SELECT EXISTS(SELECT 1 FROM messages_fts_docsize)").fetchone()[0]
    except sqlite3.OperationalError:
        return False  # table absent / FTS disabled mid-init — not this failure class


# ── FTS maintenance commands ───────────────────────────────────────────

def _fts_table_exists(holder: StoreHolder, name: str) -> bool:
    """True if an FTS5 virtual table is queryable ("no such table" and "vtable
    constructor failed" — missing tokenizer / mid-teardown — both count as not)."""
    try:
        holder._conn.execute(f"SELECT 1 FROM {name} LIMIT 0")
        return True
    except sqlite3.DatabaseError:
        return False


def _present_fts_tables(holder: StoreHolder) -> list[str]:
    """Queryable FTS tables (caller holds ``self._lock``)."""
    return [tbl for tbl in _FTS_TABLES if _fts_table_exists(holder, tbl)]


def _merge_fts_incrementally(holder: StoreHolder, *, max_pages: int, max_commands: Optional[int] = None) -> int:
    """Run bounded FTS5 ``'merge'`` commands against each present index. A positive merge rank
    stops after ~that many output pages, so each command holds the write lock for
    milliseconds regardless of index size (``'optimize'`` takes 9-18 s per index on a
    10 GB DB). ``usermerge`` is lowered to its minimum of 2 (persisted in ``%_config``,
    once per instance) so a merge acts on ANY level with >= 2 segments; at the default
    4 a fragmented index cannot converge. Up to *max_commands* per index, stopping on
    the no-progress signal ``total_changes`` delta < 2 (the INSERT itself is 1). Each
    command is its own implicit transaction, so processes interleave mid-pass. Missing
    tables are skipped (optimize_fts_storage drops + backfills them live); other SQLite
    errors propagate. Returns commands executed."""
    _positive_int("max_pages", max_pages)
    if max_commands is None:
        max_commands = holder._FTS_MERGE_COMMANDS_PER_PASS
    _positive_int("max_commands", max_commands)
    executed = 0
    with holder._lock:
        for tbl in _present_fts_tables(holder):
            if not holder._fts_usermerge_floor_applied:
                holder._conn.execute(f"INSERT INTO {tbl}({tbl}, rank) VALUES('usermerge', 2)")
            for _ in range(max_commands):
                before = holder._conn.total_changes
                holder._conn.execute(f"INSERT INTO {tbl}({tbl}, rank) VALUES('merge', ?)", (max_pages,))
                executed += 1
                if holder._conn.total_changes - before < 2:
                    break
        holder._fts_usermerge_floor_applied = True
    return executed


def _title_rank(source: Optional[str]) -> int:
    """Rank a stored title_source. NULL (pre-provenance rows) is indistinguishable from a
    manual ``/title`` of that era, so it ranks as ``user``."""
    rank = _TITLE_SOURCE_RANK
    return rank[TITLE_SOURCE_USER] if source is None else rank.get(str(source), 0)


def _reap_inactive_orphan_desktop_holders(
    holder: StoreHolder, holders: list[tuple[int, str]], *, min_age_seconds: float) -> list[int]:
    """Terminate old PPID-1 Desktop ephemeral backends with no client.

    Fails closed: anything whose parent, age, argv, or network connections
    cannot be proved safe remains a repair-blocking holder."""
    from hermes_state_dbfile import _concrete_state_db_holder_pids, _is_inactive_orphan_desktop_holder
    if not sys.platform.startswith("linux") or psutil is None:
        return []
    try:
        from ._dashboard_procs import _is_ephemeral_port_zero_backend
    except Exception:  # health: allow BLE001 -- without the argv test the reaper signals nothing
        return []
    now = time.time()
    candidates = []
    for pid in _concrete_state_db_holder_pids(holder.db_path, holders):
        try:
            process = psutil.Process(pid)
            statuses = [conn.status for conn in process.net_connections(kind="inet")]
            if not _is_inactive_orphan_desktop_holder(
                ppid=process.ppid(), age_seconds=now - process.create_time(),
                min_age_seconds=min_age_seconds,
                ephemeral_backend=_is_ephemeral_port_zero_backend(process.cmdline()),
                connection_statuses=statuses):
                continue
        except Exception:  # health: allow BLE001 -- a process that vanished or hides its details is skipped
            continue
        candidates.append(process)
    signalled: list[int] = []
    for process in candidates:
        try:
            process.terminate()
            signalled.append(process.pid)
        except (psutil.Error, OSError):
            continue
    if not signalled:
        return []
    try:
        _gone, alive = psutil.wait_procs(candidates, timeout=1.5)
    except Exception:  # health: allow BLE001 -- a failed wait escalates to no process
        alive = []
    for process in alive:
        try:
            process.kill()
        except (psutil.Error, OSError):
            continue
    if alive:
        try:
            psutil.wait_procs(alive, timeout=1.5)
        except Exception:  # health: allow BLE001 S110 -- the last wait is best-effort; the pids return either way
            pass
    return signalled


def apply_declared_schema(cursor: sqlite3.Cursor, home: Path) -> None:
    """The one additive catch-up order for raw ``state.db`` connections: columns (SCHEMA_SQL indexes may
    name columns an old table lacks), then the canonical DDL statement by statement (no
    ``executescript`` implicit COMMIT, so callers keep their own transaction), then columns again
    (a no-op for tables the script just created whole). Locking, snapshots and commit stay with the
    callers. ``_reconcile_columns`` reads nothing from a store, so the column reconciler keeps one
    implementation."""
    _reconcile_columns(cursor, home)
    _execute_ddl_script_transactional(cursor, SCHEMA_SQL)
    _reconcile_columns(cursor, home)


def reconcile_state_schema(conn: sqlite3.Connection, home: Path) -> None:
    """Bring a raw ``state.db`` connection to the canonical SCHEMA_SQL shape and commit.

    Single durable-shape authority for callers that open ``state.db``
    outside SessionDB. The async-delegation tool used to carry its own
    CREATE TABLE and ALTER column list for ``async_delegations``; it drifted
    from SCHEMA_SQL (same-name columns with different nullability/defaults
    depending on which authority touched the database first, #94691). This
    helper instead replays the canonical DDL (every statement is
    IF NOT EXISTS/idempotent) through :func:`apply_declared_schema`.
    """
    apply_declared_schema(conn.cursor(), home)
    if conn.in_transaction:
        conn.commit()
