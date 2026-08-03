"""Schema creation, column reconciliation, and FTS DDL management for SessionDB.

Mixin contract: this is a plain mixin class consumed by
``hermes_state.SessionDB``. It defines no ``__init__`` and no state of its
own; methods access the host's attributes (``self._conn``, ``self.db_path``,
``self._execute_write`` and other SessionDB methods) established by
``SessionDB.__init__``. It must never import hermes_state (cycle) — shared
module-level constants live in hermes_state_common.
"""

import logging
import json
import sqlite3
from typing import Dict, Optional

from hermes_constants import get_hermes_home
from hermes_state_common import (
    DEFERRED_INDEX_SQL,
    FTS_CJK_STALE_KEY,
    FTS_SQL,
    FTS_STORAGE_VERSION,
    FTS_TRIGRAM_SQL,
    LEGACY_FTS_SQL,
    LEGACY_FTS_TRIGRAM_SQL,
    SCHEMA_SQL,
    SCHEMA_VERSION,
    _FTS_TRIGGERS,
    _ephemeral_child_sql,
)

# Moved methods logged under the "hermes_state" logger before the split;
# keep that logger identity so log filtering/capture behavior is unchanged.
logger = logging.getLogger("hermes_state")


class SessionSchemaMixin:
    """See module docstring — mixin for SessionDB (Schema cluster)."""

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

    @staticmethod
    def _fts_trigger_count(cursor: sqlite3.Cursor) -> int:
        placeholders = ",".join("?" for _ in _FTS_TRIGGERS)
        row = cursor.execute(
            f"SELECT COUNT(*) FROM sqlite_master "
            f"WHERE type = 'trigger' AND name IN ({placeholders})",
            _FTS_TRIGGERS,
        ).fetchone()
        return int(row[0] if not isinstance(row, sqlite3.Row) else row[0])


    @staticmethod
    def _fts_update_trigger_needs_narrowing(sql: Optional[str]) -> bool:
        """True when trigger SQL is missing AFTER UPDATE OF (still broad)."""
        if not sql:
            return False
        # Collapse whitespace so multi-line DDL still matches.
        compact = " ".join(sql.split()).upper()
        # Already narrowed.
        if "AFTER UPDATE OF " in compact:
            return False
        # Broad UPDATE trigger that we still need to replace.
        return "AFTER UPDATE ON " in compact

    def _migrate_broad_fts_update_triggers(self, cursor: sqlite3.Cursor) -> int:
        """Replace broad AFTER UPDATE FTS triggers with AFTER UPDATE OF variants.

        ``CREATE TRIGGER IF NOT EXISTS`` will not replace an existing broad
        trigger, so installs that already created ``AFTER UPDATE ON messages``
        would keep firing on every messages row touch (status/compaction
        writes included). Inspect ``sqlite_master``, drop any still-broad
        UPDATE triggers, and re-apply the current DDL constants.

        No FTS rebuild: content correctness was already gated by WHEN clauses
        on modern installs; OF only skips unnecessary trigger evaluation.

        Returns the number of triggers dropped (0 when already converged).
        """
        # CJK is a v23-only surface.  Decide the layout before selecting
        # destructive candidates so the legacy branch never drops a trigger
        # it does not recreate.
        legacy_layout = self._db_has_legacy_inline_fts(cursor)
        update_names = (
            "messages_fts_update",
            "messages_fts_trigram_update",
        )
        if not legacy_layout and hasattr(self, "_ensure_fts_cjk_schema"):
            update_names += ("messages_fts_cjk_update",)
        placeholders = ", ".join("?" for _ in update_names)
        rows = cursor.execute(
            "SELECT name, sql FROM sqlite_master "
            f"WHERE type = 'trigger' AND name IN ({placeholders})",
            update_names,
        ).fetchall()
        to_drop = []
        for row in rows:
            name = row[0] if not isinstance(row, sqlite3.Row) else row["name"]
            sql = row[1] if not isinstance(row, sqlite3.Row) else row["sql"]
            if self._fts_update_trigger_needs_narrowing(sql):
                to_drop.append(name)
        if not to_drop:
            return 0

        for name in to_drop:
            # Names are drawn from the update_names literal allowlist above —
            # never user input — so the identifier is interpolation-safe.
            cursor.execute(f"DROP TRIGGER IF EXISTS {name}")

        # Re-apply current DDL so CREATE TRIGGER installs the OF variants.
        # Choose legacy vs v23 the same way _init_schema does.
        if legacy_layout:
            self._ensure_fts_schema(cursor, "messages_fts", LEGACY_FTS_SQL)
            self._ensure_fts_schema(
                cursor, "messages_fts_trigram", LEGACY_FTS_TRIGRAM_SQL
            )
        else:
            self._ensure_fts_schema(cursor, "messages_fts", FTS_SQL)
            self._ensure_fts_schema(
                cursor, "messages_fts_trigram", FTS_TRIGRAM_SQL
            )
            # CJK triggers live on the host SessionDB; only recreate one that
            # this migration actually dropped. ``_ensure_fts_cjk_schema`` is
            # documented never-raises and soft-fails OperationalError by
            # clearing availability — raise-path handling alone is not
            # enough. After ensure, require a narrowed CJK UPDATE trigger or
            # durable quarantine (stale breadcrumb + unavailable).
            if "messages_fts_cjk_update" in to_drop:
                try:
                    self._ensure_fts_cjk_schema(cursor)
                except Exception:
                    self._quarantine_cjk_after_update_of_migration(cursor)
                    logger.exception(
                        "CJK FTS re-ensure after UPDATE OF migration failed"
                    )
                    raise
                if not self._cjk_update_trigger_is_narrowed(cursor):
                    self._quarantine_cjk_after_update_of_migration(cursor)
                    logger.warning(
                        "CJK FTS UPDATE trigger missing or still broad after "
                        "UPDATE OF migration; marked stale and unavailable"
                    )

        logger.info(
            "Migrated %d broad FTS UPDATE trigger(s) to AFTER UPDATE OF "
            "(no rebuild required)",
            len(to_drop),
        )
        return len(to_drop)

    def _cjk_update_trigger_is_narrowed(self, cursor: sqlite3.Cursor) -> bool:
        """True when messages_fts_cjk_update exists with AFTER UPDATE OF."""
        row = cursor.execute(
            "SELECT sql FROM sqlite_master "
            "WHERE type = 'trigger' AND name = ?",
            ("messages_fts_cjk_update",),
        ).fetchone()
        if not row:
            return False
        sql = row[0] if not isinstance(row, sqlite3.Row) else row["sql"]
        return not self._fts_update_trigger_needs_narrowing(sql)

    def _quarantine_cjk_after_update_of_migration(
        self, cursor: sqlite3.Cursor
    ) -> None:
        """Fail-closed after dropping CJK UPDATE during OF migration.

        Clears availability, persists ``fts_cjk_stale``, and drops any
        residual broad/partial CJK UPDATE trigger so a later open cannot
        ``CREATE TRIGGER IF NOT EXISTS`` a gap without rebuild.
        """
        self._fts_cjk_available = False
        try:
            self.set_meta(FTS_CJK_STALE_KEY, "1", cursor=cursor)
        except Exception:
            logger.debug(
                "Could not persist CJK FTS stale breadcrumb",
                exc_info=True,
            )
        try:
            cursor.execute("DROP TRIGGER IF EXISTS messages_fts_cjk_update")
        except Exception:
            logger.debug(
                "Could not drop residual CJK UPDATE trigger after quarantine",
                exc_info=True,
            )


    @staticmethod
    def _rebuild_fts_indexes(
        cursor: sqlite3.Cursor,
        *,
        include_trigram: bool = True,
    ) -> None:
        # Both FTS tables are external-content (v23+): the special 'rebuild'
        # command wipes the inverted index and repopulates it from the
        # content source (messages for the standard index, the tool-row-
        # excluding messages_fts_trigram_src view for the trigram index).
        cursor.execute("INSERT INTO messages_fts(messages_fts) VALUES('rebuild')")
        if include_trigram:
            cursor.execute(
                "INSERT INTO messages_fts_trigram(messages_fts_trigram) VALUES('rebuild')"
            )
        # 'rebuild' indexes EVERY row, so any deferred-backfill markers are
        # now satisfied — clear them, otherwise the background worker would
        # re-insert rows the rebuild already covered (duplicate entries).
        cursor.execute(
            "DELETE FROM state_meta WHERE key IN "
            "('fts_rebuild_high_water', 'fts_rebuild_progress')"
        )

    @staticmethod
    def _rebuild_legacy_fts_indexes(
        cursor: sqlite3.Cursor,
        *,
        include_trigram: bool = True,
    ) -> None:
        """Rebuild the LEGACY inline FTS indexes (pre-v23) from messages.

        Used only to repair a legacy DB whose triggers degraded under an
        earlier no-FTS5 runtime. Inline tables have no external-content
        'rebuild' source, so we DELETE + reinsert the concatenated content
        the legacy triggers produced. Never touches the v23 shape.
        """
        cursor.execute("DELETE FROM messages_fts")
        cursor.execute(
            "INSERT INTO messages_fts(rowid, content) "
            "SELECT id, "
            "COALESCE(content, '') || ' ' || "
            "COALESCE(tool_name, '') || ' ' || "
            "COALESCE(tool_calls, '') "
            "FROM messages"
        )
        if not include_trigram:
            return
        cursor.execute("DELETE FROM messages_fts_trigram")
        cursor.execute(
            "INSERT INTO messages_fts_trigram(rowid, content) "
            "SELECT id, "
            "COALESCE(content, '') || ' ' || "
            "COALESCE(tool_name, '') || ' ' || "
            "COALESCE(tool_calls, '') "
            "FROM messages"
        )

    def _fts_table_probe(self, cursor: sqlite3.Cursor, table_name: str) -> Optional[bool]:
        try:
            cursor.execute(f"SELECT * FROM {table_name} LIMIT 0")
            return True
        except sqlite3.OperationalError as exc:
            if self._is_fts5_unavailable_error(exc):
                # Only disable FTS entirely when the whole module is missing.
                # A missing trigram tokenizer only affects trigram searches.
                if self._is_trigram_unavailable_error(exc):
                    self._warn_trigram_unavailable(exc)
                else:
                    self._warn_fts5_unavailable(exc)
                return None
            if "no such table" in str(exc).lower():
                return False
            raise

    @staticmethod
    def _parse_schema_columns(schema_sql: str) -> Dict[str, Dict[str, str]]:
        """Extract expected columns per table from SCHEMA_SQL.

        Uses an in-memory SQLite database to parse the SQL — SQLite itself
        handles all syntax (DEFAULT expressions with commas, inline
        REFERENCES, CHECK constraints, etc.) so there are zero regex
        edge cases.  The in-memory DB is opened, the schema DDL is
        executed, and PRAGMA table_info extracts the column metadata.

        Adding a column to SCHEMA_SQL is all that's needed; the
        reconciliation loop picks it up automatically.
        """
        ref = sqlite3.connect(":memory:")
        try:
            ref.executescript(schema_sql)
            table_columns: Dict[str, Dict[str, str]] = {}
            for (tbl,) in ref.execute(
                "SELECT name FROM sqlite_master "
                "WHERE type='table' AND name NOT LIKE 'sqlite_%'"
            ).fetchall():
                cols: Dict[str, str] = {}
                for row in ref.execute(
                    f'PRAGMA table_info("{tbl}")'
                ).fetchall():
                    # row: (cid, name, type, notnull, dflt_value, pk)
                    col_name = row[1]
                    col_type = row[2] or ""
                    notnull = row[3]
                    default = row[4]
                    pk = row[5]
                    # Reconstruct the type expression for ALTER TABLE ADD COLUMN
                    parts = [col_type] if col_type else []
                    if notnull and not pk:
                        parts.append("NOT NULL")
                    if default is not None:
                        parts.append(f"DEFAULT {default}")
                    cols[col_name] = " ".join(parts)
                table_columns[tbl] = cols
            return table_columns
        finally:
            ref.close()

    def _reconcile_columns(self, cursor: sqlite3.Cursor) -> None:
        """Ensure live tables have every column declared in SCHEMA_SQL.

        Follows the Beets/sqlite-utils pattern: the CREATE TABLE definition
        in SCHEMA_SQL is the single source of truth for the desired schema.
        On every startup this method diffs the live columns (via PRAGMA
        table_info) against the declared columns, and ADDs any that are
        missing.

        This makes column additions a declarative operation — just add
        the column to SCHEMA_SQL and it appears on the next startup.
        Version-gated migration blocks are no longer needed for ADD COLUMN.
        """
        expected = self._parse_schema_columns(SCHEMA_SQL)
        for table_name, declared_cols in expected.items():
            # Get current columns from the live table
            try:
                rows = cursor.execute(
                    f'PRAGMA table_info("{table_name}")'
                ).fetchall()
            except sqlite3.OperationalError:
                continue  # Table doesn't exist yet (shouldn't happen after executescript)
            live_cols = set()
            for row in rows:
                # PRAGMA table_info returns (cid, name, type, notnull, dflt_value, pk)
                name = row[1] if isinstance(row, (tuple, list)) else row["name"]
                live_cols.add(name)

            for col_name, col_type in declared_cols.items():
                if col_name not in live_cols:
                    safe_name = col_name.replace('"', '""')
                    try:
                        cursor.execute(
                            f'ALTER TABLE "{table_name}" ADD COLUMN "{safe_name}" {col_type}'
                        )
                    except sqlite3.OperationalError as exc:
                        # Expected: "duplicate column name" from a race or
                        # re-run.  Unexpected: "Cannot add a NOT NULL column
                        # with default value NULL" from a schema mistake.
                        # Log at DEBUG so it's visible in agent.log.
                        logger.debug(
                            "reconcile %s.%s: %s", table_name, col_name, exc,
                        )

    def _heal_gateway_routing_pk(self, cursor: sqlite3.Cursor) -> None:
        """Rebuild ``gateway_routing`` when its PRIMARY KEY predates scoping.

        Early builds of the routing-index migration (#59203) created the
        table with ``session_key TEXT PRIMARY KEY`` and no ``scope`` column.
        ``_reconcile_columns()`` ADDs the missing ``scope`` column on those
        databases, but SQLite cannot ALTER a primary key, so the shipped
        composite ``PRIMARY KEY (scope, session_key)`` never lands.  On such
        tables every write path is broken:

        * ``save_gateway_routing_entry`` fails with "ON CONFLICT clause does
          not match any PRIMARY KEY or UNIQUE constraint" (its upsert targets
          the composite key), and
        * ``replace_gateway_routing_entries`` fails with "UNIQUE constraint
          failed: gateway_routing.session_key" whenever the same session_key
          exists under a different scope — the exact isolation the composite
          key exists to provide.

        Each failed save logs a warning and falls back to sessions.json,
        so a legacy-shaped table produces endless per-save warning spam.
        Rebuild it once, preserving rows.  On a session_key collision across
        scopes (possible while the PK was wrong) the newest row wins.
        """
        try:
            rows = cursor.execute(
                'PRAGMA table_info("gateway_routing")'
            ).fetchall()
        except sqlite3.OperationalError:
            return
        if not rows:
            return

        def _col(row, idx, name):
            return row[idx] if isinstance(row, (tuple, list)) else row[name]

        pk_cols = [
            _col(r, 1, "name")
            for r in sorted(
                (r for r in rows if _col(r, 5, "pk")),
                key=lambda r: _col(r, 5, "pk"),
            )
        ]
        if pk_cols == ["scope", "session_key"]:
            return

        logger.info(
            "gateway_routing has legacy primary key %r; rebuilding with "
            "composite (scope, session_key) key",
            pk_cols,
        )
        cursor.execute(
            "ALTER TABLE gateway_routing RENAME TO gateway_routing_legacy_pk"
        )
        cursor.execute(
            """CREATE TABLE gateway_routing (
    scope TEXT NOT NULL DEFAULT '',
    session_key TEXT NOT NULL,
    entry_json TEXT NOT NULL,
    updated_at REAL NOT NULL,
    PRIMARY KEY (scope, session_key)
)"""
        )
        # INSERT OR REPLACE + updated_at ordering: if the broken PK ever let
        # two scopes race over one session_key, keep the newest row per
        # (scope, session_key) pair.
        cursor.execute(
            "INSERT OR REPLACE INTO gateway_routing "
            "(scope, session_key, entry_json, updated_at) "
            "SELECT COALESCE(scope, ''), session_key, entry_json, updated_at "
            "FROM gateway_routing_legacy_pk ORDER BY updated_at ASC"
        )
        cursor.execute("DROP TABLE gateway_routing_legacy_pk")

    def _heal_session_model_usage_pk(self, cursor: sqlite3.Cursor) -> None:
        """Rebuild ``session_model_usage`` when its PRIMARY KEY lacks ``task``.

        Installs whose ``state.db`` reached ``schema_version >= 22`` before
        the ``task`` dimension was added carry a 5-column PRIMARY KEY
        ``(session_id, model, billing_provider, billing_base_url,
        billing_mode)``.  ``_reconcile_columns()`` ADDs the ``task`` column
        as a bare nullable, but SQLite cannot ALTER a primary key, so the
        shipped composite 6-column key never lands.  The version-gated v22
        rebuild is unreachable on those installs (``current_version < 22``
        is already false), so every upsert in ``_record_model_usage()``
        fails with "ON CONFLICT clause does not match any PRIMARY KEY or
        UNIQUE constraint" — aborting the enclosing write transaction and
        silently zeroing all token *and* cost accounting (#73823).

        Idempotent; runs unconditionally on every open, same pattern as
        :meth:`_heal_gateway_routing_pk` above.  On healthy databases the
        PRAGMA check short-circuits and this is a no-op.
        """
        try:
            rows = cursor.execute(
                'PRAGMA table_info("session_model_usage")'
            ).fetchall()
        except sqlite3.OperationalError:
            return
        if not rows:
            # Table doesn't exist yet — SCHEMA_SQL creates it correctly.
            return

        def _col(row, idx, name):
            return row[idx] if isinstance(row, (tuple, list)) else row[name]

        pk_cols = {
            _col(r, 1, "name") for r in rows if _col(r, 5, "pk")
        }
        if "task" in pk_cols:
            # task is already in the PK — healthy.
            return

        logger.info(
            "session_model_usage has legacy primary key %r (missing task); "
            "rebuilding with composite 6-column key",
            sorted(pk_cols),
        )
        # FK-off window: the connection enables PRAGMA foreign_keys=ON
        # before _init_schema runs, and session_model_usage.session_id
        # REFERENCES sessions(id).  INSERT OR IGNORE does NOT suppress
        # foreign-key violations (OR IGNORE only covers uniqueness/NOT
        # NULL conflicts), so an orphaned usage row — possible after a
        # partial prune while accounting was broken — would abort the
        # whole rebuild.  Disable FK enforcement for the copy and restore
        # it afterwards.  PRAGMA foreign_keys is a no-op inside a
        # transaction, which is fine here: _init_schema runs on an
        # isolation_level=None connection with no transaction open.
        cursor.execute("PRAGMA foreign_keys=OFF")
        try:
            cursor.execute(
                "ALTER TABLE session_model_usage "
                "RENAME TO session_model_usage_legacy_pk"
            )
            cursor.execute(
                """CREATE TABLE session_model_usage (
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
            )
            # OR IGNORE: while the PK was wrong the reconciler may have left
            # ``task`` NULL on old rows; COALESCE to '' can theoretically
            # collide with a genuine ''-task row — keep the first, drop the
            # duplicate rather than fail the heal.
            cursor.execute(
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
                   FROM session_model_usage_legacy_pk"""
            )
            cursor.execute("DROP TABLE session_model_usage_legacy_pk")
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_session_model_usage_session "
                "ON session_model_usage(session_id)"
            )
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_session_model_usage_model "
                "ON session_model_usage(model)"
            )
        except sqlite3.OperationalError as exc:
            logger.debug("session_model_usage PK heal skipped: %s", exc)
        finally:
            cursor.execute("PRAGMA foreign_keys=ON")

    def _init_schema(self):
        """Create tables and FTS if they don't exist, reconcile columns.

        Schema management follows the declarative reconciliation pattern
        (Beets, sqlite-utils): SCHEMA_SQL is the single source of truth.
        On existing databases, _reconcile_columns() diffs live columns
        against SCHEMA_SQL and ADDs any missing ones.  This eliminates
        the version-gated migration chain for column additions, making
        it impossible for reordered or inserted migrations to skip columns.

        The schema_version table is retained for future data migrations
        (transforming existing rows) which cannot be handled declaratively.
        """
        cursor = self._conn.cursor()

        cursor.executescript(SCHEMA_SQL)

        # ── Declarative column reconciliation ──────────────────────────
        # Diff live tables against SCHEMA_SQL and ADD any missing columns.
        # This is idempotent and self-healing: even if a version-gated
        # migration was skipped (e.g. due to version renumbering), the
        # column gets created here.
        self._reconcile_columns(cursor)

        # Rebuild gateway_routing if it still carries the pre-scope PRIMARY
        # KEY (session_key alone). ADD COLUMN cannot fix a PK, so this is
        # the one table-shape repair reconciliation can't express.
        self._heal_gateway_routing_pk(cursor)

        # Rebuild session_model_usage if its PRIMARY KEY lacks the ``task``
        # column (5-column PK on installs already at v22+ when the column
        # landed — the version-gated rebuild is unreachable there, #73823).
        # Same PK-rebuild constraint as gateway_routing above.
        self._heal_session_model_usage_pk(cursor)

        # Indexes that reference reconciler-added columns must be created
        # AFTER _reconcile_columns runs — declaring them in SCHEMA_SQL
        # makes the initial executescript fail on legacy DBs (the index's
        # WHERE clause references a column that doesn't exist yet).
        try:
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_messages_platform_msg_id "
                "ON messages(session_id, platform_message_id) "
                "WHERE platform_message_id IS NOT NULL"
            )
        except sqlite3.OperationalError as exc:
            logger.debug("idx_messages_platform_msg_id create skipped: %s", exc)

        # Deferred indexes that reference the reconciler-added ``active``
        # column (idx_messages_session_active) — same ordering constraint.
        cursor.executescript(DEFERRED_INDEX_SQL)

        # Heal NULL ``active`` rows unconditionally on every startup.
        # On real-world DBs the reconciler-added ``active`` column can lack
        # its NOT NULL DEFAULT 1 (older reconciler builds reconstructed the
        # type without the default — see #51646: PRAGMA shows
        # (17,'active','INTEGER',0,None,0) in the wild), so INSERTs that
        # omitted the column wrote NULL and the ``WHERE active = 1``
        # transcript loaders hid the whole history.  The INSERTs now set
        # active=1 explicitly; this idempotent repair un-hides rows written
        # before the fix.  It was previously gated at ``current_version <
        # 12`` which never re-ran for already-v12+ databases.
        try:
            cursor.execute(
                "UPDATE messages SET active = 1 WHERE active IS NULL"
            )
        except sqlite3.OperationalError:
            pass

        fts5_available = self._sqlite_supports_fts5(cursor)
        fts_migrations_complete = True
        if not fts5_available:
            # Existing FTS triggers can still fire on messages INSERT/UPDATE
            # even though the current sqlite runtime cannot read the virtual
            # tables they target. Drop only the triggers so core persistence
            # continues; if a future runtime has FTS5, _ensure_fts_schema()
            # recreates them.
            self._drop_fts_triggers(cursor)

        # ── Schema version bookkeeping ─────────────────────────────────
        # Bump to current so future data migrations (if any) can gate on
        # version.  No version-gated column additions remain.
        cursor.execute("SELECT version FROM schema_version LIMIT 1")
        row = cursor.fetchone()
        if row is None:
            cursor.execute(
                "INSERT INTO schema_version (version) VALUES (?)",
                (SCHEMA_VERSION,),
            )
        else:
            current_version = row["version"] if isinstance(row, sqlite3.Row) else row[0]
            # Data migrations that can't be expressed declaratively (row
            # backfills, index changes tied to a specific version step) stay
            # in a version-gated chain. Column additions are handled by
            # _reconcile_columns() above and no longer need entries here.
            if current_version < 10 and SCHEMA_VERSION == 10:
                # v10: trigram FTS5 table for CJK/substring search. The
                # virtual table + triggers are created unconditionally via
                # FTS_TRIGRAM_SQL below, but existing rows need a one-time
                # backfill into the FTS index.
                #
                # Only run this when v10 itself is the target schema. Current
                # v11+ code drops and rebuilds both FTS tables below, so doing
                # the v10-only trigram backfill first only burns startup time
                # and WAL space before v11 throws the work away.
                if fts5_available:
                    _fts_trigram_exists = self._fts_table_probe(
                        cursor, "messages_fts_trigram"
                    )
                    if _fts_trigram_exists is False:
                        if self._ensure_fts_schema(
                            cursor, "messages_fts_trigram", FTS_TRIGRAM_SQL
                        ):
                            cursor.execute(
                                "INSERT INTO messages_fts_trigram(rowid, content) "
                                "SELECT id, content FROM messages WHERE content IS NOT NULL"
                            )
                        else:
                            fts_migrations_complete = False
                    elif _fts_trigram_exists is None:
                        fts_migrations_complete = False
                else:
                    fts_migrations_complete = False
            if current_version < 11 and SCHEMA_VERSION < 23:
                # v11 (SUPERSEDED by v23): re-index FTS5 tables to cover
                # tool_name + tool_calls in inline mode (#16751). v23 drops
                # and rebuilds both FTS tables in external-content form, so
                # running the v11 inline backfill first would only burn
                # startup time and WAL space before v23 throws the work
                # away — and its inline INSERT shape no longer matches the
                # current external-content FTS_SQL anyway. Kept only for
                # source archaeology; unreachable while SCHEMA_VERSION >= 23.
                pass
            if current_version < 16:
                # v16: tag delegate subagent rows so pickers stay clean after
                # parent deletes that used to orphan them (parent_session_id → NULL).
                try:
                    cursor.execute(
                        "UPDATE sessions SET model_config = json_set("
                        "COALESCE(model_config, '{}'), '$._delegate_from', parent_session_id) "
                        f"WHERE parent_session_id IS NOT NULL "
                        "AND json_extract(COALESCE(model_config, '{}'), '$._delegate_from') IS NULL "
                        f"AND {_ephemeral_child_sql('sessions')}"
                    )
                    cursor.execute(
                        "UPDATE sessions SET model_config = json_set("
                        "COALESCE(model_config, '{}'), '$._delegate_from', '__orphaned__') "
                        "WHERE parent_session_id IS NULL "
                        "AND json_extract(COALESCE(model_config, '{}'), '$._delegate_from') IS NULL "
                        "AND json_extract(COALESCE(model_config, '{}'), '$._branched_from') IS NULL "
                        "AND title IS NULL "
                        "AND message_count <= 25 "
                        "AND EXISTS (SELECT 1 FROM messages m "
                        "            WHERE m.session_id = sessions.id AND m.role = 'tool') "
                        "AND NOT EXISTS (SELECT 1 FROM sessions ch "
                        "                WHERE ch.parent_session_id = sessions.id)"
                    )
                except sqlite3.OperationalError:
                    pass
            if current_version < 18:
                # v18: gateway metadata consolidation (#9006). Backfill
                # display_name / origin_json / expiry_finalized from
                # sessions.json so pre-migration gateway sessions are
                # discoverable from state.db without the JSON index.
                try:
                    self._backfill_gateway_metadata_from_sessions_json(cursor)
                except Exception as exc:
                    # Backfill is best-effort: sessions.json may be absent,
                    # corrupted, or partially stale. Missing metadata simply
                    # means consumers fall back to sessions.json for those
                    # rows until the gateway rewrites them.
                    logger.debug("v18 gateway metadata backfill skipped: %s", exc)
            if current_version < 20:
                # v20: per-model usage attribution (issue #51607). Going
                # forward update_token_counts() records each API call into
                # session_model_usage keyed by the live model, but existing
                # sessions only have their aggregate totals on the sessions
                # row. Seed one usage row per historical session from those
                # aggregates so insights reads uniformly from the new table.
                # INSERT OR IGNORE keeps it idempotent: if newer code already
                # wrote a (session_id, model, provider) row for a session, the
                # PK conflict skips the stale aggregate rather than doubling it.
                try:
                    cursor.execute(
                        """INSERT OR IGNORE INTO session_model_usage (
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
                    )
                except sqlite3.OperationalError:
                    pass
            if current_version < 22:
                # v22: task-dimension usage attribution (issue #23270).
                # session_model_usage gains a ``task`` column ('' = main agent
                # loop; 'vision'/'compression'/'title_generation'/... =
                # auxiliary calls) so aux model spend is visible in analytics.
                # The column participates in the PRIMARY KEY and SQLite cannot
                # ALTER a PK, so rebuild the table. The reconciler will have
                # already ADDed the plain column on legacy DBs (harmless);
                # the rebuild bakes it into the PK properly. Existing rows are
                # main-loop accounting by definition → task=''.
                try:
                    legacy_pk = cursor.execute(
                        "SELECT COUNT(*) FROM pragma_table_info('session_model_usage') "
                        "WHERE name = 'task' AND pk > 0"
                    ).fetchone()[0]
                    if not legacy_pk:
                        cursor.execute("ALTER TABLE session_model_usage RENAME TO session_model_usage_v21")
                        cursor.execute(
                            """CREATE TABLE session_model_usage (
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
                        )
                        cursor.execute(
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
                               FROM session_model_usage_v21"""
                        )
                        cursor.execute("DROP TABLE session_model_usage_v21")
                        cursor.execute(
                            "CREATE INDEX IF NOT EXISTS idx_session_model_usage_session "
                            "ON session_model_usage(session_id)"
                        )
                        cursor.execute(
                            "CREATE INDEX IF NOT EXISTS idx_session_model_usage_model "
                            "ON session_model_usage(model)"
                        )
                except sqlite3.OperationalError as exc:
                    logger.debug("v22 session_model_usage rebuild skipped: %s", exc)
            if current_version < 23:
                # v23: FTS storage redesign (issues #22478, #43690, #55233).
                # The v11 inline-mode FTS tables each store a full private
                # copy of every message (content || tool_name || tool_calls),
                # and the trigram index additionally covers role='tool' rows
                # (~90% of message bytes: base64 payloads, file dumps) at
                # ~2.6x amplification — together ~75% of state.db on heavy
                # installs (observed: 18.9 GB of a 25 GB DB).
                #
                # OPT-IN, NOT AUTOMATIC. The transition (demote old vtables →
                # new external-content schema → backfill → teardown → VACUUM)
                # is disk-heavy (transient ~2x file size to fully reclaim via
                # VACUUM) and long (~1-2h background on a 25 GB DB). Doing it
                # silently on every big user's next open — with a completeness
                # guarantee that depends on the process staying alive long
                # enough — is the wrong default. So on an EXISTING install we
                # touch nothing here: the v22 inline FTS keeps working exactly
                # as before, and we only record a flag advertising that the
                # optimization is available. `hermes sessions optimize-storage`
                # performs the whole transition as one deliberate, disk-checked,
                # progress-reported foreground operation.
                #
                # DECOUPLED VERSIONING. Crucially, this does NOT hold back the
                # main schema_version. The FTS storage LAYOUT is tracked by an
                # independent `fts_storage_version` marker (see
                # _fts_storage_version / SETTLE below), so schema_version
                # advances to SCHEMA_VERSION here like every other migration —
                # future v24+ migrations land automatically for legacy-FTS
                # users too. Only the FTS *layout* waits for opt-in.
                if fts5_available and self._db_has_legacy_inline_fts(cursor):
                    self.set_meta("fts_optimize_available", "1", cursor=cursor)

            # The FTS storage layout is versioned independently of the main
            # schema (see the v23 note above). Stamp the current layout so the
            # main version can always advance: a fresh/optimized DB is at
            # FTS_STORAGE_VERSION; a legacy DB is left at whatever it had
            # (absent/0) until `optimize-storage` runs. An INTERRUPTED
            # optimize (legacy vtables already demoted, but rebuild markers
            # or demoted trash tables still present, or an empty external
            # index against non-empty messages) is NOT stamped either —
            # the marker is the source of truth for "fully optimized", and
            # `fts_optimize_available()` keeps offering the resume until the
            # transition actually completes.
            if (
                fts5_available
                and not self._db_has_legacy_inline_fts(cursor)
                and cursor.execute(
                    "SELECT 1 FROM state_meta "
                    "WHERE key = 'fts_rebuild_high_water' LIMIT 1"
                ).fetchone() is None
                and not self._has_fts_trash(cursor)
                and not self._fts_external_index_empty_with_messages(cursor)
            ):
                self.set_meta(
                    "fts_storage_version", str(FTS_STORAGE_VERSION), cursor=cursor
                )

            # Advance schema_version to current for ALL non-FTS-layout
            # migrations. This is deliberately NOT gated on the FTS opt-in —
            # holding the whole version back would block every future schema
            # migration for a user who never optimizes. FTS5 being unavailable
            # is the one case we skip (we can't have created the current FTS
            # objects, so claiming the current schema would be a lie).
            if (
                current_version < SCHEMA_VERSION
                and fts_migrations_complete
                and fts5_available
            ):
                cursor.execute(
                    "UPDATE schema_version SET version = ?",
                    (SCHEMA_VERSION,),
                )

        # Unique title index — always ensure it exists. Older databases may
        # contain duplicate aliases from before the constraint was enforced;
        # preserve every session while letting the newest one retain the alias.
        title_index_sql = (
            "CREATE UNIQUE INDEX IF NOT EXISTS idx_sessions_title_unique "
            "ON sessions(title) WHERE title IS NOT NULL"
        )
        try:
            cursor.execute(title_index_sql)
        except sqlite3.IntegrityError:
            # The index is an optimization — its creation must never abort
            # opening the database, so the repair itself is also guarded.
            try:
                cursor.execute(
                    """UPDATE sessions AS older
                       SET title = NULL
                       WHERE title IS NOT NULL
                         AND EXISTS (
                             SELECT 1 FROM sessions AS newer
                             WHERE newer.title = older.title
                               AND newer.rowid > older.rowid
                         )"""
                )
                logger.warning(
                    "Cleared %d duplicate session title(s) while restoring the unique index",
                    cursor.rowcount,
                )
                cursor.execute(title_index_sql)
            except sqlite3.Error:
                logger.exception(
                    "Could not repair duplicate session titles; "
                    "unique title index not created"
                )
        except sqlite3.OperationalError:
            pass  # Index already exists

        if fts5_available:
            # FTS5 setup. Run the DDL even when the virtual table exists so
            # CREATE TRIGGER IF NOT EXISTS repairs trigger-only degradation from
            # an earlier no-FTS5 runtime.
            #
            # OPT-IN v23 boundary: a legacy v22 install (inline-content FTS,
            # not yet opted into `hermes db optimize`) must keep its EXISTING
            # inline schema + triggers. Running the v23 external-content DDL
            # here would create the trigram source VIEW and leave the DB in a
            # mixed inline/external state. So for a legacy DB we only ensure
            # its inline triggers exist (via the legacy DDL), and skip the
            # v23 view/external tables entirely. Fresh installs and opted-in
            # DBs have no legacy inline FTS, so they get the v23 DDL.
            if self._db_has_legacy_inline_fts(cursor):
                triggers_need_repair = (
                    self._fts_trigger_count(cursor) < len(_FTS_TRIGGERS)
                )
                self._fts_enabled = self._ensure_fts_schema(
                    cursor, "messages_fts", LEGACY_FTS_SQL
                )
                if self._fts_enabled:
                    trigram_enabled = self._ensure_fts_schema(
                        cursor, "messages_fts_trigram", LEGACY_FTS_TRIGRAM_SQL
                    )
                    self._trigram_available = trigram_enabled
                    if triggers_need_repair:
                        self._rebuild_legacy_fts_indexes(
                            cursor, include_trigram=trigram_enabled
                        )
            else:
                triggers_need_repair = (
                    self._fts_trigger_count(cursor) < len(_FTS_TRIGGERS)
                )
                self._fts_enabled = self._ensure_fts_schema(
                    cursor, "messages_fts", FTS_SQL
                )

                # Trigram FTS5 for CJK/substring search. This is optional
                # relative to the main FTS table; if it cannot be created,
                # CJK search falls back to LIKE.
                if self._fts_enabled:
                    trigram_enabled = self._ensure_fts_schema(
                        cursor, "messages_fts_trigram", FTS_TRIGRAM_SQL
                    )
                    self._trigram_available = trigram_enabled
                    if triggers_need_repair:
                        self._rebuild_fts_indexes(
                            cursor,
                            include_trigram=trigram_enabled,
                        )
                    # CJK-bigram index (cjk_unicode61). Strictly additive to
                    # the surfaces above and gated on the loadable tokenizer:
                    self._ensure_fts_cjk_schema(cursor)

            # Replace any pre-existing broad AFTER UPDATE triggers with
            # AFTER UPDATE OF variants. IF NOT EXISTS cannot rewrite them.
            if getattr(self, "_fts_enabled", False):
                self._migrate_broad_fts_update_triggers(cursor)

        self._conn.commit()

    def _backfill_gateway_metadata_from_sessions_json(
        self, cursor: sqlite3.Cursor
    ) -> None:
        """One-time v18 backfill of gateway metadata from sessions.json.

        Existing gateway sessions predate the display_name / origin_json /
        expiry_finalized columns; copy what sessions.json knows so consumers
        can switch to state.db without losing pre-migration sessions.
        Only fills NULL columns — never overwrites data written by newer code.
        """
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
                    entry.get("session_key") or key,
                    (origin or {}).get("chat_id") if isinstance(origin, dict) else None,
                    entry.get("chat_type"),
                    (origin or {}).get("thread_id") if isinstance(origin, dict) else None,
                    entry.get("display_name"),
                    json.dumps(origin) if isinstance(origin, dict) else None,
                    1 if entry.get("expiry_finalized") or entry.get("memory_flushed") else 0,
                    str(session_id),
                ),
            )
