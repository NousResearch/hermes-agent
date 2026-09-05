"""Schema creation, column reconciliation, and FTS DDL management for SessionDB.

Plain mixin for ``hermes_state.SessionDB`` (no ``__init__``/state of its own).
Must never import hermes_state (cycle); shared constants live in hermes_state_common.
"""

import contextlib
import datetime
import hashlib
import logging
import json
import os
import sqlite3
import tempfile
import time
import uuid
from typing import Dict, List, Optional, Sequence


from hermes_constants import get_hermes_home
from hermes_startup_watchdog import report_startup_progress
from utils import safe_json_loads
from hermes_state_common import (
    DEFERRED_INDEX_SQL, FTS_CJK_STALE_KEY, FTS_REBUILD_DEFERRAL_KEY, FTS_STALE_KEY, FTS_SQL,
    FTS_STORAGE_VERSION, FTS_TOOL_FULL_CONTENT_HIGH_WATER_KEY, FTS_TRIGRAM_SQL, LEGACY_FTS_SQL,
    LEGACY_FTS_TRIGRAM_SQL, SCHEMA_SQL,
    SCHEMA_VERSION, _FTS_CJK_TRIGGERS, _FTS_TRIGGERS, _ephemeral_child_sql, fts_rebuild_admission,
)

# Pre-split logger identity so log filtering/capture is unchanged.
logger = logging.getLogger("hermes_state")

_FTS_HOLDER_ESCALATE_ATTEMPTS = 3
_FTS_HOLDER_ESCALATE_SECONDS = 60.0
# retry_deferred_fts_recovery cadence: startup paid the full admission wait once; later
# retries are non-blocking probes whose spacing doubles up to the cap.
_FTS_STALE_RETRY_SECONDS = 60.0
_FTS_STALE_RETRY_MAX_SECONDS = 3600.0

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
    "CREATE UNIQUE INDEX IF NOT EXISTS idx_sessions_title_unique ON sessions(title) "
    "WHERE title IS NOT NULL AND credential_owner IS NULL",
    "CREATE UNIQUE INDEX IF NOT EXISTS idx_sessions_owned_title_unique ON sessions(title, credential_owner) "
    "WHERE title IS NOT NULL AND credential_owner IS NOT NULL",
)
_STALE_KEY_UPSERT_SQL = (
    "INSERT INTO state_meta (key, value) VALUES (?, '1') ON CONFLICT(key) DO UPDATE SET value = excluded.value"
)
_STATE_META_UPSERT_SQL = (
    "INSERT INTO state_meta (key, value) VALUES (?, ?) ON CONFLICT(key) DO UPDATE SET value = excluded.value"
)
_CLEAR_REBUILD_MARKERS_SQL = "DELETE FROM state_meta WHERE key IN ('fts_rebuild_high_water', 'fts_rebuild_progress')"


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


def schema_read_probe_statements() -> tuple:
    """SELECT statements that fail iff a live store is behind SCHEMA_SQL. Read-only opens skip
    ``_reconcile_columns()`` (no DDL against another profile's live DB), so healing callers
    run these afterwards: a missing table/column raises at prepare time. Derived from
    SCHEMA_SQL (a hand-maintained list went stale within days). Columns are
    table-qualified: an unqualified double-quoted identifier that fails to resolve silently
    degrades to a string literal (SQLite misfeature) and would pass on the stale store."""
    global _READ_PROBE_STATEMENTS
    if _READ_PROBE_STATEMENTS is None:
        tables = SessionSchemaMixin._parse_schema_columns(SCHEMA_SQL)
        _READ_PROBE_STATEMENTS = tuple(
            "SELECT {} FROM {} LIMIT 0".format(", ".join(f"{_q(table)}.{_q(col)}" for col in cols), _q(table))
            for table, cols in sorted(tables.items())
        )
    return _READ_PROBE_STATEMENTS


class SessionSchemaMixin:
    """See module docstring — mixin for SessionDB (Schema cluster)."""

    def _dedupe_legacy_system_prompts(self, cursor: sqlite3.Cursor) -> None:
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
                prompt_hash = self._store_system_prompt(cursor, prompt)
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

    def _sqlite_supports_fts5(self, cursor: sqlite3.Cursor) -> bool:
        try:
            cursor.execute("CREATE VIRTUAL TABLE temp._hermes_fts5_probe USING fts5(x)")
            cursor.execute("DROP TABLE temp._hermes_fts5_probe")
            return True
        except sqlite3.OperationalError as exc:
            if not self._is_fts5_unavailable_error(exc):
                raise
            self._warn_fts5_unavailable(exc)
            return False

    def _drop_all_fts_triggers(self, cursor: sqlite3.Cursor) -> None:
        self._drop_fts_triggers(cursor)
        for trigger in _FTS_CJK_TRIGGERS:
            with contextlib.suppress(sqlite3.OperationalError):
                cursor.execute(f"DROP TRIGGER IF EXISTS {trigger}")

    @staticmethod
    def _fts_triggers_missing(cursor: sqlite3.Cursor, names: Sequence[str]) -> bool:
        """True unless every trigger in *names* (one DDL half) exists."""
        if not names:
            return False  # "name IN ()" is a SQLite syntax error
        placeholders = ",".join("?" for _ in names)
        sql = f"SELECT COUNT(*) FROM sqlite_master WHERE type = 'trigger' AND name IN ({placeholders})"
        return int(cursor.execute(sql, tuple(names)).fetchone()[0]) < len(names)

    @staticmethod
    def _fts_update_trigger_needs_narrowing(sql: Optional[str]) -> bool:
        """True when trigger SQL is a broad AFTER UPDATE (missing ``OF``)."""
        if not sql:
            return False
        compact = " ".join(sql.split()).upper()  # multi-line DDL still matches
        return "AFTER UPDATE OF " not in compact and "AFTER UPDATE ON " in compact

    def _migrate_broad_fts_update_triggers(self, cursor: sqlite3.Cursor) -> int:
        """Replace broad AFTER UPDATE FTS triggers with AFTER UPDATE OF variants (``IF NOT EXISTS``
        never replaces an existing broad trigger). No FTS rebuild: correctness was already
        gated by WHEN clauses; OF only skips trigger evaluation. Returns the number dropped."""
        # CJK is v23-only. Decide the layout before selecting destructive candidates so the
        # legacy branch never drops a trigger it won't recreate.
        legacy_layout = self._db_has_legacy_inline_fts(cursor)
        update_names = ("messages_fts_update", "messages_fts_trigram_update") + (
            () if legacy_layout else ("messages_fts_cjk_update",)
        )
        placeholders = ", ".join("?" for _ in update_names)
        sql = f"SELECT name, sql FROM sqlite_master WHERE type = 'trigger' AND name IN ({placeholders})"
        rows = cursor.execute(sql, update_names).fetchall()
        to_drop = [name for name, sql in rows if self._fts_update_trigger_needs_narrowing(sql)]
        if not to_drop:
            return 0
        for name in to_drop:
            cursor.execute(f"DROP TRIGGER IF EXISTS {name}")  # names from the literal allowlist above

        # Re-apply current DDL (legacy vs v23 as _init_schema does) so CREATE TRIGGER installs OF variants.
        base_sql, trigram_sql = _FTS_DDL[legacy_layout]
        self._ensure_fts_schema(cursor, "messages_fts", base_sql)
        self._ensure_fts_schema(cursor, "messages_fts_trigram", trigram_sql)
        # Only recreate the CJK trigger this migration dropped. ``_ensure_fts_cjk_schema`` soft-fails
        # (never raises), so afterwards require a narrowed trigger or durable quarantine.
        if "messages_fts_cjk_update" in to_drop:
            try:
                self._ensure_fts_cjk_schema(cursor)
            except Exception:
                self._quarantine_cjk_after_update_of_migration(cursor)
                logger.exception("CJK FTS re-ensure after UPDATE OF migration failed")
                raise
            row = cursor.execute(
                "SELECT sql FROM sqlite_master WHERE type = 'trigger' AND name = ?", ("messages_fts_cjk_update",),
            ).fetchone()
            if not row or self._fts_update_trigger_needs_narrowing(row[0]):
                self._quarantine_cjk_after_update_of_migration(cursor)
                logger.warning(
                    "CJK FTS UPDATE trigger missing or still broad after "
                    "UPDATE OF migration; marked stale and unavailable"
                )
        logger.info("Migrated %d broad FTS UPDATE trigger(s) to AFTER UPDATE OF (no rebuild required)", len(to_drop))
        return len(to_drop)

    @staticmethod
    def _stamp_fts_tool_high_water(cursor: sqlite3.Cursor) -> None:
        """Record MAX(messages.id) as the bounded-tool-content high-water mark: rows at or below it keep
        their exact stored token stream; newer tool rows index only the prefix (see ``_fts_indexed_content_sql``)."""
        high_water = cursor.execute("SELECT COALESCE(MAX(id), 0) FROM messages").fetchone()[0]
        cursor.execute(_STATE_META_UPSERT_SQL, (FTS_TOOL_FULL_CONTENT_HIGH_WATER_KEY, str(high_water)))

    @staticmethod
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

    def _migrate_bounded_tool_fts_triggers(self, cursor: sqlite3.Cursor, *, legacy: bool) -> None:
        """Replace FTS triggers without rebuilding historical indexes. Existing rows keep their
        full-content token stream; the durable high-water id makes new tool rows use the bounded
        prefix in INSERT and the matching external-content delete/update. One savepoint, so no
        concurrent writer lands in a trigger gap."""
        marker = cursor.execute(
            "SELECT 1 FROM state_meta WHERE key = ? LIMIT 1", (FTS_TOOL_FULL_CONTENT_HIGH_WATER_KEY,),
        ).fetchone()
        if marker is not None:
            return
        trigram_present = self._sqlite_table_exists(cursor, "messages_fts_trigram")
        names = _FTS_BASE_TRIGGERS + (_FTS_TRIGRAM_TRIGGERS if legacy and trigram_present else ())
        has_messages = cursor.execute("SELECT 1 FROM messages LIMIT 1").fetchone() is not None
        self._fts_tool_prefix_migration_requires_rebuild = bool(
            self._sqlite_table_exists(cursor, "messages_fts") and has_messages
            and self._fts_triggers_missing(cursor, names)
        )
        cursor.execute("SAVEPOINT bounded_tool_fts")
        try:
            self._stamp_fts_tool_high_water(cursor)
            for name in names:
                cursor.execute(f"DROP TRIGGER IF EXISTS {name}")
            if legacy:
                self._execute_ddl_script_transactional(cursor, LEGACY_FTS_SQL)
                if trigram_present:
                    self._execute_ddl_script_transactional(cursor, LEGACY_FTS_TRIGRAM_SQL)
            else:
                self._execute_ddl_script_transactional(cursor, FTS_SQL)
            cursor.execute("RELEASE SAVEPOINT bounded_tool_fts")
        except BaseException:
            cursor.execute("ROLLBACK TO SAVEPOINT bounded_tool_fts")
            cursor.execute("RELEASE SAVEPOINT bounded_tool_fts")
            raise

    @staticmethod
    def _sqlite_table_exists(cursor: sqlite3.Cursor, name: str) -> bool:
        return cursor.execute(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?", (name,),
        ).fetchone() is not None

    def _migrate_trigram_cron_exclusion(self, cursor: sqlite3.Cursor) -> bool:
        """Install the source-filtered trigram view and purge historical rows (v29 cron exclusion,
        v30 subagent exclusion: both only change the view/trigger predicate and rebuild from it).
        Legacy inline indexes stay opt-in (their content is private to the vtable). A v1 external
        layout whose vtable still declares ``tool_calls`` is left to ``optimize-storage``: swapping
        the view underneath would make 'rebuild' read a column the view no longer has. Otherwise
        the inverted index still holds excluded rows until FTS5 rebuilds from the new view, which
        runs under the shared cross-process admission gate. Returns False to hold schema_version back."""
        if self._db_has_legacy_inline_fts(cursor) or self._db_has_trigram_tool_calls_projection(cursor):
            return True
        trigram_exists = self._fts_table_probe(cursor, "messages_fts_trigram")
        if trigram_exists is not True:
            # Absent: the normal ensure path creates/backfills it. None: this runtime cannot
            # safely inspect an existing one, so leave the version behind for retry.
            return trigram_exists is False
        for name in _FTS_TRIGRAM_TRIGGERS:
            cursor.execute(f"DROP TRIGGER IF EXISTS {name}")
        cursor.execute("DROP VIEW IF EXISTS messages_fts_trigram_src")
        if not self._ensure_fts_schema(cursor, "messages_fts_trigram", FTS_TRIGRAM_SQL):
            return False
        # Always rebuild while schema_version is behind, even if the view already has the new
        # predicate: a process can die between replacing the view and rebuilding/stamping.
        self._run_admitted_startup_rebuild(
            cursor,
            lambda: cursor.execute("INSERT INTO messages_fts_trigram(messages_fts_trigram) VALUES('rebuild')"),
        )
        return True

    def _quarantine_cjk_after_update_of_migration(self, cursor: sqlite3.Cursor) -> None:
        """Fail closed after dropping the CJK UPDATE trigger mid-migration: clear availability,
        persist ``fts_cjk_stale``, drop any residual trigger so a later open cannot
        IF-NOT-EXISTS over a gap."""
        self._fts_cjk_available = False
        try:
            self.set_meta(FTS_CJK_STALE_KEY, "1", cursor=cursor)
        except Exception:
            logger.debug("Could not persist CJK FTS stale breadcrumb", exc_info=True)
        try:
            cursor.execute("DROP TRIGGER IF EXISTS messages_fts_cjk_update")
        except Exception:
            logger.debug("Could not drop residual CJK UPDATE trigger after quarantine", exc_info=True)

    @staticmethod
    def _rebuild_fts_indexes(cursor: sqlite3.Cursor, *, legacy: bool = False, include_trigram: bool = True) -> None:
        """v23+ external-content 'rebuild'. It indexes EVERY row, so the deferred-backfill
        markers are cleared or the worker would re-insert covered rows (duplicates).
        ``legacy`` (pre-v23 inline layout) has no external-content 'rebuild' source, so it
        DELETEs + reinserts the concatenated content the legacy triggers produced."""
        SessionSchemaMixin._stamp_fts_tool_high_water(cursor)
        tables = ("messages_fts", "messages_fts_trigram") if include_trigram else ("messages_fts",)
        for tbl in tables:
            if legacy:
                cursor.execute(f"DELETE FROM {tbl}")
                cursor.execute(f"INSERT INTO {tbl}(rowid, content) SELECT id, {_LEGACY_INLINE_CONCAT_SQL}FROM messages")
            else:
                cursor.execute(f"INSERT INTO {tbl}({tbl}) VALUES('rebuild')")
        if not legacy:
            cursor.execute(_CLEAR_REBUILD_MARKERS_SQL)

    def _fts_table_probe(self, cursor: sqlite3.Cursor, table_name: str) -> Optional[bool]:
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
            if self._is_fts5_unavailable_error(exc):
                # A missing trigram tokenizer only affects trigram search; only a missing
                # FTS5 module disables FTS entirely.
                if self._is_trigram_unavailable_error(exc):
                    self._warn_trigram_unavailable(exc)
                else:
                    self._warn_fts5_unavailable(exc)
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

    def _defer_stale_fts_for_holders(self, cursor: sqlite3.Cursor, foreign_holders) -> bool:
        """Record a deferral diagnostic for the foreign processes holding the DB; True = defer
        (holders remain). After ``_FTS_HOLDER_ESCALATE_ATTEMPTS`` deferrals spanning
        ``_FTS_HOLDER_ESCALATE_SECONDS``, provably inactive orphan Desktop backends are
        reaped and the holders re-checked."""
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
        except (TypeError, ValueError):
            first_seen, attempts = now, 1
        if first_seen > now or first_seen < 0:
            first_seen = now
        diagnostic = {
            "first_seen": first_seen, "last_seen": now, "attempts": attempts,
            "holder_pids": sorted({pid for pid, _path in foreign_holders if pid > 0}),
        }
        cursor.execute(
            "INSERT INTO state_meta (key, value) VALUES (?, ?) ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            (FTS_REBUILD_DEFERRAL_KEY, json.dumps(diagnostic, sort_keys=True)),
        )
        if attempts >= _FTS_HOLDER_ESCALATE_ATTEMPTS and now - first_seen >= _FTS_HOLDER_ESCALATE_SECONDS:
            reaped = self._reap_inactive_orphan_desktop_holders(
                foreign_holders, min_age_seconds=_FTS_HOLDER_ESCALATE_SECONDS,
            )
            if reaped:
                logger.error(
                    "Reaped inactive orphan Desktop backend(s) %s after %d "
                    "state.db FTS rebuild deferrals; checking holders again.", reaped, attempts,
                )
                foreign_holders = self._foreign_state_db_holders()
            if foreign_holders:
                logger.error(
                    "state.db FTS repair remains blocked after %d deferrals "
                    "by holder(s) %s. Stop the listed processes, then run "
                    "`hermes sessions optimize-storage` with the gateway stopped. "
                    "`hermes doctor` reports this degraded state.", attempts, foreign_holders,
                )
        if not foreign_holders:
            return False
        logger.warning(
            "Deferred stale state.db FTS rebuild while foreign processes "
            "hold the database or WAL sidecars (%s); canonical writes and LIKE search remain available (deferral %d).",
            foreign_holders, attempts,
        )
        return True

    def _recover_stale_fts(self, cursor: sqlite3.Cursor, *, legacy: bool, timeout_seconds=None) -> bool:
        """Atomically rebuild stale base/trigram indexes and resume syncing. *timeout_seconds*
        bounds the admission wait (None = full startup budget, ``0`` = non-blocking retry).
        Fails closed: holders or a lost admission race leave the breadcrumb set."""
        foreign_holders = self._foreign_state_db_holders()
        if foreign_holders and self._defer_stale_fts_for_holders(cursor, foreign_holders):
            return False
        with fts_rebuild_admission(self.db_path, timeout_seconds=timeout_seconds) as admitted:
            if not admitted:
                logger.warning(
                    "Deferred stale state.db FTS rebuild: another process holds the rebuild authority; "
                    "canonical writes and LIKE search remain available."
                )
                return False
            return self._recover_stale_fts_locked(cursor, legacy=legacy)

    def retry_deferred_fts_recovery(self) -> bool:
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
        if not self._fts_stale:
            return False
        if getattr(self, "_db_corrupt", False):
            # Quarantined: never run FTS DDL/DML against a damaged image (mirrors _try_wal_checkpoint /
            # close). Reset the backoff so a future un-quarantine starts from the default interval.
            self._fts_stale_retry_after = 0.0
            self._fts_stale_retry_interval = 0.0
            return False
        if self.read_only or self._conn is None:
            return False
        now = time.monotonic()
        if now < getattr(self, "_fts_stale_retry_after", 0.0):
            return False
        interval = float(getattr(self, "_fts_stale_retry_interval", 0.0))
        if interval <= 0.0:
            interval = _FTS_STALE_RETRY_SECONDS
        self._fts_stale_retry_after = now + interval
        self._fts_stale_retry_interval = min(
            max(interval, _FTS_STALE_RETRY_SECONDS, 1.0) * 2.0, _FTS_STALE_RETRY_MAX_SECONDS,
        )
        try:
            with self._lock:
                if self._conn is None or not self._fts_stale:
                    return False
                cursor = self._conn.cursor()
                legacy = self._db_has_legacy_inline_fts(cursor)
                recovered = self._recover_stale_fts(cursor, legacy=legacy, timeout_seconds=0.0)
                if recovered:
                    # CJK was detached alongside the base indexes; its own ensure path
                    # decides when it comes back online.
                    self._ensure_fts_cjk_schema(cursor)
                    self._fts_stale_retry_interval = 0.0
                with contextlib.suppress(sqlite3.Error):
                    self._conn.commit()
                return recovered
        except Exception:  # noqa: BLE001 - background retry must never raise
            logger.warning(
                "In-process retry of the deferred stale state.db FTS rebuild failed; will retry later.", exc_info=True,
            )
            return False

    def _recover_stale_fts_locked(self, cursor: sqlite3.Cursor, *, legacy: bool) -> bool:
        """Body of :meth:`_recover_stale_fts`; caller holds rebuild authority. One write
        transaction, so no canonical writer slips between rebuild and trigger restoration."""
        try:
            include_trigram = self._fts_table_probe(cursor, "messages_fts_trigram") is True
        except (sqlite3.DatabaseError, UnicodeDecodeError):
            # A corrupt vtable may fail even a LIMIT 0 probe; still include it in the drop-and-recreate.
            include_trigram = True

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
            rebuild_sql += _CLEAR_REBUILD_MARKERS_SQL + ";"
        recovery_sql = (
            "BEGIN IMMEDIATE;" + drop_sql + rebuild_sql
            + f"DELETE FROM state_meta WHERE key IN ('{FTS_STALE_KEY}', '{FTS_REBUILD_DEFERRAL_KEY}');COMMIT;"
        )
        try:
            cursor.executescript(recovery_sql)
        except sqlite3.DatabaseError as exc:
            with contextlib.suppress(sqlite3.Error):
                self._conn.rollback()
            # Stale indexes must stay detached even on builds whose DDL transaction behavior differs.
            self._drop_all_fts_triggers(cursor)
            self._conn.commit()
            logger.error(
                "Automatic rebuild of stale FTS indexes failed (%s); "
                "canonical writes remain enabled with FTS detached.", exc,
            )
            return False
        self._fts_stale = False
        self._fts_enabled = True
        self._trigram_available = include_trigram
        logger.warning("Rebuilt stale state.db FTS indexes from canonical messages and restored sync triggers.")
        return True

    # ── Declarative column reconciliation ──────────────────────────────────

    @staticmethod
    def _parse_schema_columns(schema_sql: str) -> Dict[str, Dict[str, str]]:
        """Expected columns per table: execute SCHEMA_SQL in an in-memory database and read
        PRAGMA table_info (no regex). Memoized on disk keyed by a DDL hash (~85ms per
        startup otherwise); only the reference-side parse is cached — diffing the LIVE
        database still runs every startup. A corrupt/stale cache degrades to recomputation."""
        cache_path = None
        schema_hash = hashlib.sha256(schema_sql.encode("utf-8")).hexdigest()
        with contextlib.suppress(Exception):  # missing/corrupt cache → recompute below
            # Late import: resolves a test-patched hermes_constants.get_hermes_home.
            from hermes_constants import get_hermes_home as _home
            cache_path = _home() / "cache" / "schema_columns.json"
            blob = json.loads(cache_path.read_text(encoding="utf-8"))
            tables = blob.get("tables") if isinstance(blob, dict) and blob.get("schema_hash") == schema_hash else None
            if isinstance(tables, dict) and all(
                isinstance(cols, dict) and all(isinstance(v, str) for v in cols.values()) for cols in tables.values()
            ):
                return tables

        ref = sqlite3.connect(":memory:")
        try:
            ref.executescript(schema_sql)
            table_columns: Dict[str, Dict[str, str]] = {}
            for (tbl,) in ref.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
            ).fetchall():
                cols: Dict[str, str] = {}
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

    def _reconcile_columns(self, cursor: sqlite3.Cursor) -> None:
        """ADD every SCHEMA_SQL column missing from the live tables (SCHEMA_SQL is the single
        source of truth; column additions need no version-gated migration)."""
        expected = self._parse_schema_columns(SCHEMA_SQL)
        for table_name, declared_cols in expected.items():
            try:
                rows = cursor.execute(f'PRAGMA table_info("{table_name}")').fetchall()
            except sqlite3.OperationalError:
                continue  # Table doesn't exist yet (shouldn't happen after executescript)
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
                    if "locked" in message or "busy" in message:
                        # Swallowing lock contention left the store half-reconciled ("no such
                        # column" on every read). Re-raise so the lock-patience wrapper retries init.
                        raise
                    # Anything else permanently strands the store behind SCHEMA_SQL — be loud.
                    logger.warning(
                        "reconcile %s.%s failed; store remains behind SCHEMA_SQL: %s", table_name, col_name, exc,
                    )

    @staticmethod
    def _live_pk_columns(cursor: sqlite3.Cursor, table: str) -> Optional[List[str]]:
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

    @staticmethod
    def _rebuild_table(cursor: sqlite3.Cursor, table: str, legacy_name: str, ddl: str, copy_sql: str, indexes=()) -> None:
        """RENAME *table* to *legacy_name*, CREATE it fresh from *ddl*, copy rows back with
        *copy_sql*, DROP the legacy copy, recreate *indexes*."""
        cursor.execute(f"ALTER TABLE {table} RENAME TO {legacy_name}")
        cursor.execute(ddl)
        cursor.execute(copy_sql)
        cursor.execute(f"DROP TABLE {legacy_name}")
        for sql in indexes:
            cursor.execute(sql)

    def _heal_gateway_routing_pk(self, cursor: sqlite3.Cursor) -> None:
        """Rebuild ``gateway_routing`` when its PRIMARY KEY predates scoping (``session_key TEXT
        PRIMARY KEY``): the reconciler ADDs ``scope`` but SQLite cannot ALTER a PK, so every
        routing write fails (ON CONFLICT mismatch / cross-scope UNIQUE violation). Newest
        row wins a cross-scope session_key collision (INSERT OR REPLACE in updated_at order).

        Early builds of the routing-index migration (#59203) created the table with ``session_key TEXT
        PRIMARY KEY`` and no ``scope`` column. ``_reconcile_columns()`` ADDs the missing ``scope`` column on
        those databases, but SQLite cannot ALTER a primary key, so the shipped composite ``PRIMARY KEY
        (scope, session_key)`` never lands. On such tables every write path is broken:
        """
        pk_cols = self._live_pk_columns(cursor, "gateway_routing")
        if pk_cols is None or pk_cols == ["scope", "session_key"]:
            return
        logger.info(
            "gateway_routing has legacy primary key %r; rebuilding with composite (scope, session_key) key", pk_cols,
        )
        self._rebuild_table(
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

    def _heal_session_model_usage_pk(self, cursor: sqlite3.Cursor) -> None:
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
        pk_cols = self._live_pk_columns(cursor, "session_model_usage")
        if pk_cols is None or "task" in pk_cols:
            return
        logger.info(
            "session_model_usage has legacy primary key %r (missing task); rebuilding with composite 6-column key",
            sorted(pk_cols),
        )
        cursor.execute("PRAGMA foreign_keys=OFF")
        try:
            self._rebuild_table(
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

    def _init_schema(self):
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
        cursor = self._conn.cursor()
        cursor.executescript(SCHEMA_SQL)

        # Column reconciliation, then the two table-shape repairs ADD COLUMN cannot express.
        self._reconcile_columns(cursor)
        self._heal_gateway_routing_pk(cursor)
        # Rebuild session_model_usage if its PRIMARY KEY lacks the ``task`` column (5-column PK on installs
        # already at v22+ when the column landed — the version-gated rebuild is unreachable there, #73823).
        # Same PK-rebuild constraint as gateway_routing above.
        self._heal_session_model_usage_pk(cursor)

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
        cursor.executescript(DEFERRED_INDEX_SQL)  # same ordering constraint (``active``)

        # Heal NULL ``active`` rows on every startup: older reconciler builds added ``active``
        # without NOT NULL DEFAULT 1, so ``WHERE active = 1`` loaders hid whole histories. A
        # ``current_version < 12`` gate never re-ran for already-v12+ databases.
        with contextlib.suppress(sqlite3.OperationalError):
            cursor.execute("UPDATE messages SET active = 1 WHERE active IS NULL")

        fts5_available = self._sqlite_supports_fts5(cursor)
        stale_row = cursor.execute("SELECT 1 FROM state_meta WHERE key = ? LIMIT 1", (FTS_STALE_KEY,)).fetchone()
        self._fts_stale = stale_row is not None
        if self._fts_stale:
            # A prior process detached FTS after corruption; stay detached until a full rebuild.
            self._drop_all_fts_triggers(cursor)
        if not fts5_available:
            # Existing FTS triggers would still fire though this runtime cannot read their
            # targets. Drop only the triggers; a future FTS5 runtime recreates them.
            self._drop_fts_triggers(cursor)

        row = cursor.execute("SELECT version FROM schema_version LIMIT 1").fetchone()
        if row is None:
            cursor.execute("INSERT INTO schema_version (version) VALUES (?)", (SCHEMA_VERSION,))
            # Store provenance so fresh vs wiped stores are distinguishable.
            # See #97568.
            now_iso = datetime.datetime.now(datetime.timezone.utc).isoformat()
            cursor.executemany(
                "INSERT OR IGNORE INTO state_meta (key, value) VALUES (?, ?)",
                [("store_instance_id", str(uuid.uuid4())), ("store_created_at_utc", now_iso)],
            )
        else:
            self._run_data_migrations(cursor, row[0], fts5_available)

        self._ensure_unique_title_index(cursor)
        if fts5_available:
            self._init_fts(cursor)
        self._conn.commit()

    def _run_data_migrations(self, cursor: sqlite3.Cursor, current_version: int, fts5_available: bool) -> None:
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
                    "AND json_extract(COALESCE(model_config, '{}'), '$._delegate_from') IS NULL "
                    f"AND {_ephemeral_child_sql('sessions')}"
                )
                cursor.execute(
                    "UPDATE sessions SET model_config = json_set("
                    "COALESCE(model_config, '{}'), '$._delegate_from', '__orphaned__') WHERE parent_session_id IS NULL "
                    "AND json_extract(COALESCE(model_config, '{}'), '$._delegate_from') IS NULL "
                    "AND json_extract(COALESCE(model_config, '{}'), '$._branched_from') IS NULL "
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
                self._backfill_gateway_metadata_from_sessions_json(cursor)
            except Exception as exc:
                logger.debug("v18 gateway metadata backfill skipped: %s", exc)
        if current_version < 20:
            # v20: seed session_model_usage from sessions aggregates (OR IGNORE: newer rows win).
            with contextlib.suppress(sqlite3.OperationalError):
                cursor.execute(_SESSION_MODEL_USAGE_V20_SEED_SQL)
        if current_version < 22:
            self._migrate_v22_session_model_usage(cursor)
        # v23: FTS storage redesign (external-content tables). OPT-IN, NOT AUTOMATIC: the
        # transition is disk-heavy (~2x transient) and long (hours on 25 GB), so an existing
        # install only gets a flag; `hermes sessions optimize-storage` performs it. The FTS
        # layout is tracked by the independent `fts_storage_version` marker, so
        # schema_version still advances for legacy-FTS users.
        if current_version < 23 and fts5_available and self._db_needs_fts_storage_upgrade(cursor):
            self.set_meta("fts_optimize_available", "1", cursor=cursor)
        if current_version < 25:
            # v25: de-duplicate system prompt snapshots (old column stays a read fallback).
            self._dedupe_legacy_system_prompts(cursor)
        fts_migrations_complete = True
        if current_version < 30 and fts5_available:
            # v29: cron sessions leave the trigram substring index (they stay in the word index);
            # v30: delegate-child transcripts too (FTS_TRIGRAM_EXCLUDED_SOURCES + _delegate_from).
            # Rebuild once so rows indexed by older view/trigger definitions do not linger.
            fts_migrations_complete = self._migrate_trigram_cron_exclusion(cursor)

        # Stamp the FTS layout version (fresh/optimized DBs); a legacy DB keeps its absent/0
        # marker until optimize-storage runs. An INTERRUPTED optimize (markers, trash, or an
        # empty external index against non-empty messages) is NOT stamped: the marker is the
        # source of truth for "fully optimized" and keeps the resume offer alive.
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
        if (
            fts5_available
            and not self._db_needs_fts_storage_upgrade(cursor)
            and cursor.execute(
                "SELECT 1 FROM state_meta WHERE key = 'fts_rebuild_high_water' LIMIT 1"
            ).fetchone() is None
            and not self._has_fts_trash(cursor)
            and not self._fts_external_index_empty_with_messages(cursor)
        ):
            self.set_meta("fts_storage_version", str(FTS_STORAGE_VERSION), cursor=cursor)

        # Advance schema_version — deliberately NOT gated on the FTS opt-in (that would block
        # every future migration for a user who never optimizes). FTS5 unavailable is the
        # one skip: claiming current would lie.
        if current_version < SCHEMA_VERSION and fts_migrations_complete and fts5_available:
            cursor.execute("UPDATE schema_version SET version = ?", (SCHEMA_VERSION,))

    def _migrate_v22_session_model_usage(self, cursor: sqlite3.Cursor) -> None:
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
            self._rebuild_table(
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

    def _ensure_unique_title_index(self, cursor: sqlite3.Cursor) -> None:
        """Keep titles unique within explicit credential-owner scope and legacy global scope."""
        try:
            legacy = cursor.execute(
                "SELECT sql FROM sqlite_master WHERE type = 'index' AND name = 'idx_sessions_title_unique'"
            ).fetchone()
            legacy_sql = str(legacy[0] or "").lower() if legacy else ""
            if legacy and "credential_owner" not in legacy_sql:
                cursor.execute("DROP INDEX idx_sessions_title_unique")
            for statement in _TITLE_UNIQUE_INDEX_SQL:
                cursor.execute(statement)
        except sqlite3.IntegrityError:
            try:
                cursor.execute("""UPDATE sessions AS older
                       SET title = NULL
                       WHERE title IS NOT NULL
                         AND EXISTS (
                             SELECT 1 FROM sessions AS newer
                             WHERE newer.title = older.title
                               AND newer.credential_owner IS older.credential_owner
                               AND newer.rowid > older.rowid
                         )""")
                logger.warning(
                    "Cleared %d duplicate session title(s) while restoring the unique index", cursor.rowcount,
                )
                for statement in _TITLE_UNIQUE_INDEX_SQL:
                    cursor.execute(statement)
            except sqlite3.Error:
                logger.exception("Could not repair duplicate session titles; unique title index not created")
        except sqlite3.OperationalError:
            pass  # Index already exists

    def _init_fts(self, cursor: sqlite3.Cursor) -> None:
        """Create/repair the FTS objects on an FTS5-capable runtime. The DDL runs even when the
        vtable exists so CREATE TRIGGER IF NOT EXISTS repairs trigger-only degradation.
        OPT-IN v23 boundary: a legacy v22 inline install keeps its inline schema + triggers
        (the v23 DDL would create the trigram source VIEW and leave a mixed state)."""
        legacy_fts = self._db_has_legacy_inline_fts(cursor)
        if not self._fts_stale:
            self._migrate_bounded_tool_fts_triggers(cursor, legacy=legacy_fts)
        if self._fts_stale:
            if self._recover_stale_fts(cursor, legacy=legacy_fts):
                # CJK was detached alongside the base indexes; its ensure path decides when it returns.
                self._ensure_fts_cjk_schema(cursor)
            else:
                self._fts_enabled = self._trigram_available = self._fts_cjk_available = False
        else:
            base_sql, trigram_sql = _FTS_DDL[legacy_fts]
            # Measure BEFORE the DDL below runs (pre-repair state). Whether the trigram half is
            # creatable is only known AFTER _ensure_fts_schema, hence the halves combine at the `if`.
            base_triggers_missing = self._fts_triggers_missing(cursor, _FTS_BASE_TRIGGERS) or getattr(
                self, "_fts_tool_prefix_migration_requires_rebuild", False)
            trigram_triggers_missing = self._fts_triggers_missing(cursor, _FTS_TRIGRAM_TRIGGERS)
            self._fts_enabled = self._ensure_fts_schema(cursor, "messages_fts", base_sql)
            if self._fts_enabled:
                # Trigram is optional; without it CJK search falls back to LIKE.
                trigram_enabled = self._ensure_fts_schema(cursor, "messages_fts_trigram", trigram_sql)
                self._trigram_available = trigram_enabled
                if base_triggers_missing or (trigram_enabled and trigram_triggers_missing):
                    self._run_admitted_startup_rebuild(
                        cursor,
                        lambda: self._rebuild_fts_indexes(cursor, legacy=legacy_fts, include_trigram=trigram_enabled),
                    )
                if not legacy_fts:
                    # CJK-bigram index: strictly additive, gated on the loadable tokenizer.
                    self._ensure_fts_cjk_schema(cursor)
        # IF NOT EXISTS cannot rewrite pre-existing broad AFTER UPDATE triggers.
        if self._fts_enabled:
            self._migrate_broad_fts_update_triggers(cursor)

    def _run_admitted_startup_rebuild(self, cursor, rebuild_fn) -> None:
        """Run a full trigger-repair FTS rebuild under cross-process admission (the sync triggers
        were missing and the DDL just recreated them: the index has a gap of unknown
        extent). Two processes opening the same DB after an update commonly hit this
        simultaneously (the interleaving that corrupted state.db in production), so this
        FAILS CLOSED: on deferral the just-repaired triggers are dropped again and the
        stale breadcrumb persisted — triggers must never be live over an unrebuilt gap
        (``_enter_fts_fail_open``'s ordering contract); a later recovery path restores both.

        See #93200.
        """
        with fts_rebuild_admission(self.db_path) as admitted:
            if admitted:
                rebuild_fn()
                return
        logger.warning(
            "Deferred startup FTS rebuild: another process holds the "
            "rebuild authority for this state.db; detaching FTS sync until the stale-index recovery path rebuilds it."
        )
        cursor.execute(_STALE_KEY_UPSERT_SQL, (FTS_STALE_KEY,))
        self._drop_all_fts_triggers(cursor)
        self._fts_stale = True
        self._fts_enabled = self._trigram_available = self._fts_cjk_available = False

    def _backfill_gateway_metadata_from_sessions_json(self, cursor: sqlite3.Cursor) -> None:
        """One-time v18 backfill of gateway metadata from sessions.json. Only fills NULL
        columns — never overwrites data written by newer code."""
        sessions_file = get_hermes_home() / "sessions" / "sessions.json"
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
