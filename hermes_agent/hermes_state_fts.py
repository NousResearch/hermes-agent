"""FTS5 index setup for SessionDB: per-table schema ensure with tokenizer-capability fallback,
FTS-scoped corruption detection and the atomic fail-open trigger detach."""

import logging
import re
import sqlite3
import time
from typing import Sequence

from hermes_agent.hermes_state_common import FTS_SQL, FTS_STALE_KEY, FTS_TRIGRAM_SQL, _FTS_TRIGGERS
from hermes_agent.hermes_state_errors import is_fts_scoped_corruption_error, is_sqlite_lock_error

# caplog tests pin the "hermes_agent.hermes_state" logger name.
logger = logging.getLogger("hermes_agent.hermes_state")

_FTS_DDL_TABLES = ("messages_fts", "messages_fts_trigram")


def _stored_form(sql: str) -> str:
    """``sqlite_master.sql`` as compared here: whitespace collapsed, none kept around parentheses and commas (the
    same definition written with different spacing is the same layout), and the object name unquoted (an ALTER
    TABLE RENAME round trip stores it as ``"name"`` without changing the layout)."""
    collapsed = re.sub(r"\s*([(),])\s*", r"\1", " ".join((sql or "").split()))
    return re.sub(r'^CREATE (VIRTUAL TABLE|VIEW) "([^"]+)"', r"CREATE \1 \2", collapsed)


def _expected_stored_sql(ddl: str, kind: str, name: str) -> str:
    """The stored form of the ``CREATE <kind> IF NOT EXISTS <name>`` statement in *ddl* (``IF NOT EXISTS``
    and ``;`` dropped), i.e. what ``sqlite_master.sql`` holds once the DDL has run."""
    start = ddl.index(f"CREATE {kind} IF NOT EXISTS {name} ")
    end = ddl.index(";", start)
    return _stored_form(ddl[start:end].replace(" IF NOT EXISTS", "", 1))


def _expected_vtable_sql(ddl: str, table: str) -> str:
    return _expected_stored_sql(ddl, "VIRTUAL TABLE", table)


# Every stored definition a released version has created, oldest first; the LAST entry is the one this code
# creates. optimize-storage rebuilds only the older entries: an unrecognised definition (a newer release's, after
# a downgrade) is left alone. Literal on purpose: a change to FTS_SQL / FTS_TRIGRAM_SQL fails
# test_the_layout_registry_ends_with_the_shipping_ddl until its new form is appended here, which keeps the
# previous form recognisable as old.
FTS_VTABLE_LAYOUTS: dict[str, tuple[str, ...]] = {
    "messages_fts": (
        "CREATE VIRTUAL TABLE messages_fts USING fts5(content,content=messages,content_rowid=id)",
        "CREATE VIRTUAL TABLE messages_fts USING fts5(content)",
        "CREATE VIRTUAL TABLE messages_fts USING fts5(content,tool_name,tool_calls,content='messages',"
        "content_rowid='id')",
        "CREATE VIRTUAL TABLE messages_fts USING fts5(content,tool_name,tool_calls,"
        "content='messages_fts_src',content_rowid='id')",
    ),
    "messages_fts_trigram": (
        "CREATE VIRTUAL TABLE messages_fts_trigram USING fts5(content,content=messages,content_rowid=id,"
        "tokenize='trigram')",
        "CREATE VIRTUAL TABLE messages_fts_trigram USING fts5(content,tokenize='trigram')",
        "CREATE VIRTUAL TABLE messages_fts_trigram USING fts5(content,tool_name,tool_calls,"
        "content='messages_fts_trigram_src',content_rowid='id',tokenize='trigram')",
        "CREATE VIRTUAL TABLE messages_fts_trigram USING fts5(content,tool_name,"
        "content='messages_fts_trigram_src',content_rowid='id',tokenize='trigram')",
    ),
}


def _stored_vtables(cursor: sqlite3.Cursor) -> dict[str, str]:
    rows = cursor.execute(
        "SELECT name, sql FROM sqlite_master WHERE type = 'table' AND name IN (?, ?)", _FTS_DDL_TABLES).fetchall()
    return {name: _stored_form(sql) for name, sql in rows}


def _is_older_layout(table: str, stored: str) -> bool:
    """A definition an earlier release created. The two shape checks keep every store the pre-registry code
    treated as old (any base index without ``tool_name``, any trigram with ``tool_calls``) on the opt-in path."""
    if stored in FTS_VTABLE_LAYOUTS[table][:-1]:
        return True
    if table == "messages_fts":
        return "tool_name" not in stored
    return "tool_calls" in stored.lower()


def base_index_recognised(cursor: sqlite3.Cursor) -> bool:
    """``messages_fts`` holds a definition this code or an earlier release created; the open-time realign
    replaces nothing else."""
    return _stored_vtables(cursor).get("messages_fts") in FTS_VTABLE_LAYOUTS["messages_fts"]


def base_index_realigned_on_open(stored: str) -> bool:
    """External-content base index not reading ``messages_fts_src``: the next read-write open rebuilds it from
    the current DDL (``_migrate_misaligned_fts_source``), so it is never opt-in work."""
    return "tool_name" in stored and "messages_fts_src" not in stored


def fts_layout_drift(cursor: sqlite3.Cursor) -> list[str]:
    """FTS tables on an older release's definition that only ``optimize-storage`` replaces; empty otherwise.

    Shared by the update notice, doctor and optimize-storage's demote decision. An absent table is not drift
    (the trigram index is optional; the base index is created on open), nor is a base index the next open
    realigns, nor a definition this code does not recognise.
    """
    return [table for table, stored in _stored_vtables(cursor).items()
            if _is_older_layout(table, stored)
            and not (table == "messages_fts" and base_index_realigned_on_open(stored))]


def fts_layout_current(cursor: sqlite3.Cursor) -> bool:
    """True when every present FTS table, and the base index's source view, is exactly what this code creates:
    the only state the ``fts_storage_version`` stamp may be written over."""
    stored = _stored_vtables(cursor)
    if any(sql != FTS_VTABLE_LAYOUTS[table][-1] for table, sql in stored.items()):
        return False
    if "messages_fts" not in stored:
        return True
    row = cursor.execute("SELECT sql FROM sqlite_master WHERE type = 'view' AND name = 'messages_fts_src'").fetchone()
    return row is not None and _stored_form(row[0]) == _expected_stored_sql(FTS_SQL, "VIEW", "messages_fts_src")


def base_source_view_is_stale(cursor: sqlite3.Cursor) -> bool:
    """``messages_fts_src`` exists but differs from what this code creates (``CREATE VIEW IF NOT EXISTS`` never
    replaces it)."""
    row = cursor.execute("SELECT sql FROM sqlite_master WHERE type = 'view' AND name = 'messages_fts_src'").fetchone()
    return row is not None and _stored_form(row[0]) != _expected_stored_sql(FTS_SQL, "VIEW", "messages_fts_src")


# FTS5 shadow tables the virtual-table engine owns. `sqlite3 .recover` re-emits them
# as ordinary tables but cannot re-emit the CREATE VIRTUAL TABLE row, so every later
# CREATE VIRTUAL TABLE fails with "fts5: error creating shadow table <name>: table
# already exists" until the orphans are dropped (#103840).
_FTS5_SHADOW_SUFFIXES = ("content", "data", "docsize", "idx", "config")


def _drop_orphan_fts_shadow_tables(cursor: sqlite3.Cursor, families: Sequence[str]) -> list[str]:
    """Drop, per family, shadow tables whose virtual table row is absent from sqlite_master.

    Matches exact shadow names only (never a prefix LIKE, so the base family cannot reach
    ``messages_fts_trigram_*``) and leaves a family alone whenever its vtable is live. The
    shadows are derived index state; the caller recreates and rebuilds from ``messages``.
    Returns the families that were repaired.
    """
    repaired: list[str] = []
    for family in families:
        vtable_live = cursor.execute(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ? "
            "AND sql LIKE 'CREATE VIRTUAL TABLE%'",
            (family,),
        ).fetchone()
        if vtable_live:
            continue
        shadows = [f"{family}_{suffix}" for suffix in _FTS5_SHADOW_SUFFIXES]
        orphans = [row[0] for row in cursor.execute(
            f"SELECT name FROM sqlite_master WHERE type = 'table' AND name IN ({','.join('?' for _ in shadows)})",
            shadows,
        ).fetchall()]
        if not orphans:
            continue
        for name in orphans:
            cursor.execute(f'DROP TABLE "{name}"')
        logger.warning(
            "Dropped orphan FTS5 shadow tables of %s (%s); the index is recreated from messages",
            family, ", ".join(orphans),
        )
        repaired.append(family)
    return repaired


_RETIRED_CJK_TRIGGERS = ("messages_fts_cjk_insert", "messages_fts_cjk_delete", "messages_fts_cjk_update")
_RETIRED_CJK_OBJECTS = (*_RETIRED_CJK_TRIGGERS, "messages_fts_cjk_src", "messages_fts_cjk",
                        *(f"messages_fts_cjk_{suffix}" for suffix in _FTS5_SHADOW_SUFFIXES))


def retired_cjk_index_present(cursor: sqlite3.Cursor) -> bool:
    """True when an object of the ``messages_fts_cjk`` index an earlier release built is still in the schema."""
    return cursor.execute(
        f"SELECT 1 FROM sqlite_master WHERE name IN ({','.join('?' for _ in _RETIRED_CJK_OBJECTS)}) LIMIT 1",
        _RETIRED_CJK_OBJECTS,
    ).fetchone() is not None


def drop_retired_cjk_index(cursor: sqlite3.Cursor) -> bool:
    """Remove the ``messages_fts_cjk`` index an earlier release built with the loadable cjk_unicode61
    tokenizer, and its markers. The shadow tables are renamed into the FTS trash, which
    ``optimize_fts_storage`` empties in chunks. Idempotent; True when anything was removed.
    Reuse a caller's transaction without committing it; otherwise own the atomic retirement."""
    if not retired_cjk_index_present(cursor):
        return False
    owns_transaction = not cursor.connection.in_transaction
    if owns_transaction:
        cursor.execute("BEGIN IMMEDIATE")
    try:
        removed = _stage_retired_cjk_index(cursor)
        if owns_transaction:
            cursor.connection.commit()
        return removed
    except BaseException:
        if owns_transaction:
            cursor.connection.rollback()
        raise


def _stage_retired_cjk_index(cursor: sqlite3.Cursor) -> bool:
    """Recheck after acquiring the write lock: another opener may have finished retirement."""
    if not retired_cjk_index_present(cursor):
        return False
    # Triggers first: they call the tokenizer, so while they exist every outside write to messages fails.
    for trigger in _RETIRED_CJK_TRIGGERS:
        cursor.execute(f"DROP TRIGGER IF EXISTS {trigger}")
    cursor.execute("DROP VIEW IF EXISTS messages_fts_cjk_src")
    cursor.execute(
        "DELETE FROM state_meta WHERE key IN ('fts_cjk_stale', 'fts_cjk_rebuild_high_water', 'fts_cjk_rebuild_progress')"
    )
    # DROP TABLE needs the tokenizer before SQLite 3.47 and deletes every page under the write lock.
    cursor.execute("PRAGMA writable_schema=ON")
    cursor.execute("DELETE FROM sqlite_master WHERE type = 'table' AND name = 'messages_fts_cjk'")
    cursor.execute("PRAGMA writable_schema=RESET")
    # Nothing references the shadow tables once the triggers and view are gone, so the rename need not re-parse
    # the schema; the modern rename does, and fails on another index whose tokenizer this build lacks (trigram).
    cursor.execute("PRAGMA legacy_alter_table=ON")
    try:
        for suffix in _FTS5_SHADOW_SUFFIXES:
            shadow = f"messages_fts_cjk_{suffix}"
            if cursor.execute("SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?", (shadow,)).fetchone():
                cursor.execute(f"ALTER TABLE {shadow} RENAME TO {SessionFtsSetupMixin._FTS_TRASH_PREFIX}{shadow}")
    finally:
        cursor.execute("PRAGMA legacy_alter_table=OFF")
    return True


class SessionFtsSetupMixin:
    """FTS table/trigger lifecycle shared by schema init, optimize and the write path."""

    @staticmethod
    def _is_fts5_unavailable_error(exc: sqlite3.OperationalError) -> bool:
        """No FTS5 module, or an optional tokenizer missing (same capability-error shape)."""
        err = str(exc).lower()
        return ("no such module" in err and "fts5" in err) or SessionFtsSetupMixin._is_trigram_unavailable_error(exc)

    @staticmethod
    def _is_trigram_unavailable_error(exc: sqlite3.OperationalError) -> bool:
        """Only the trigram tokenizer is missing (it needs SQLite >= 3.34): "this one index can't be served",
        never "disable FTS"."""
        return "no such tokenizer: trigram" in str(exc).lower()

    @staticmethod
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

    @staticmethod
    def _db_has_trigram_tool_calls_projection(cursor: sqlite3.Cursor) -> bool:
        """True when the trigram vtable still includes the tool_calls payload (FTS_STORAGE_VERSION 1)."""
        row = cursor.execute(
            "SELECT sql FROM sqlite_master WHERE type = 'table' AND name = 'messages_fts_trigram'"
        ).fetchone()
        return row is not None and "tool_calls" in (row[0] or "").lower()

    @classmethod
    def _db_needs_fts_storage_upgrade(cls, cursor: sqlite3.Cursor) -> bool:
        """True when optimize-storage has layout work: a stored FTS vtable differs from what this code creates."""
        return bool(fts_layout_drift(cursor))

    def _warn_trigram_unavailable(self, exc: sqlite3.OperationalError) -> None:
        """Log once that the trigram tokenizer is missing; base FTS5 stays enabled."""
        if getattr(self, "_trigram_unavailable_warned", False):  # attr is lazily created here
            return
        self._trigram_unavailable_warned = True
        logger.info(
            "SQLite trigram tokenizer unavailable for %s "
            "(requires SQLite >= 3.34, this build is %s); "
            "CJK/substring search will fall back to LIKE: %s",
            self.db_path,
            sqlite3.sqlite_version,
            exc,
        )

    def _warn_fts5_unavailable(self, exc: sqlite3.OperationalError) -> None:
        self._fts_enabled = False
        if self._fts_unavailable_warned:
            return
        self._fts_unavailable_warned = True
        logger.warning(
            "SQLite FTS5 unavailable for %s; full-text session search "
            "disabled. Run `hermes update` to rebuild the venv with a "
            "current Python (managed uv guarantees FTS5). (underlying error: %s)",
            self.db_path,
            exc,
        )

    @staticmethod
    def _drop_fts_triggers(cursor: sqlite3.Cursor) -> None:
        for trigger in _FTS_TRIGGERS:
            try:
                cursor.execute(f"DROP TRIGGER IF EXISTS {trigger}")
            except sqlite3.OperationalError:
                pass

    def _ensure_fts_schema(self, cursor: sqlite3.Cursor, table_name: str, ddl: str) -> bool:
        status = self._fts_table_probe(cursor, table_name)
        if status is None:
            return False
        try:
            # Run even when the table exists: recreates triggers a no-FTS5 runtime dropped.
            cursor.executescript(ddl)
            return True
        except sqlite3.OperationalError as exc:
            if not self._is_fts5_unavailable_error(exc):
                raise
            # A missing tokenizer disables only that table; the base FTS5 table is fine.
            if self._is_trigram_unavailable_error(exc):
                self._warn_trigram_unavailable(exc)
            else:
                self._warn_fts5_unavailable(exc)
            return False

    @staticmethod
    def _is_fts_write_corruption_error(exc: sqlite3.DatabaseError) -> bool:
        """Corruption SQLite identifies as FTS-scoped (SQLITE_CORRUPT_VTAB, or an ``fts5:``
        report naming ``messages_fts*`` on builds without result codes); a bare malformed
        image is structural. One rule, shared with ``classify_persistence_error`` and the
        gateway transcript retry: see :func:`hermes_agent.hermes_state_errors.is_fts_scoped_corruption_error`."""
        return is_fts_scoped_corruption_error(exc)

    def _enter_fts_fail_open(
        self, exc: sqlite3.DatabaseError, *, deadline: float | None = None, patience_s: float | None = None,
    ) -> bool:
        """Detach corrupt FTS indexes so canonical writes can continue. Breadcrumb +
        trigger drop commit atomically: once triggers are absent the index has a
        gap of unknown extent, so nobody may reinstall them without a full rebuild.

        A busy write lock is waited out on the caller's write budget (default
        ``_WRITE_PATIENCE_S``), like ``_execute_write``: the writer connection's busy
        timeout is only 1 s, and the usual holder is a sibling writer detaching the
        same corrupt index — giving up after 1 s cost that turn's canonical write."""
        if not self._fts_enabled or not self._is_fts_write_corruption_error(exc):
            return False
        if patience_s is None:
            patience_s = self._WRITE_PATIENCE_S
        if deadline is None:
            deadline = time.monotonic() + patience_s
        while True:
            # Re-checked every attempt: a sibling may quarantine the file while we wait for the
            # lock, and nothing may be committed on a quarantined handle.
            self._raise_if_db_corrupt(storage=True)
            try:
                with self._lock:
                    self._raise_if_db_replaced()
                    if self._conn is None:
                        self._reopen_after_close_locked(context="write")
                    self._conn.execute("BEGIN IMMEDIATE")
                    try:
                        self._conn.execute(
                            "INSERT INTO state_meta (key, value) VALUES (?, '1') "
                            "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                            (FTS_STALE_KEY,),
                        )
                        self._drop_all_fts_triggers(self._conn.cursor())
                        self._conn.commit()
                    except BaseException:
                        self._conn.rollback()
                        raise
                break
            except sqlite3.Error as detach_exc:
                if (
                    isinstance(detach_exc, sqlite3.OperationalError) and is_sqlite_lock_error(detach_exc)
                    and self._sleep_before_write_retry(deadline, patience_s)
                ):
                    continue
                logger.error(
                    "Could not detach corrupt FTS indexes; canonical write still cannot proceed: %s",
                    detach_exc,
                )
                return False
        self._fts_stale = True
        self._fts_enabled = False
        self._trigram_available = False
        logger.error(
            "state.db FTS indexes remain corrupt (%s); disabled FTS sync and "
            "retrying the canonical write. Search temporarily uses LIKE until "
            "a later SessionDB open rebuilds the indexes.",
            exc,
        )
        return True

    # ── Chunked FTS rebuild engine (v23 opt-in optimize) ──
    # One blocking rebuild held the write lock ~16 min on a 25 GB DB, so the
    # backfill and teardown run in chunks: each its own bounded transaction (WAL
    # growth stays small), resumable from its marker (claimed by CAS), and a
    # progress tick. They run back to back with no pause while no other process
    # holds state.db: the foreground `hermes sessions optimize-storage` refuses to
    # start under a holder (held_store_refusal, #110054), and a fixed 0.2 s pause
    # after each few-ms chunk was >95% of the wall time. Chunk size (in ROWS, never
    # an id span: a pruned store has id gaps) adapts toward TARGET seconds, since
    # rows cost ~1000x more in trigram_data than in docsize. A side thread re-runs
    # the holder scan every _HOLDER_SCAN_INTERVAL_S (`--force` past a holder, or a
    # writer that opens the store mid-run): while one is there the duty cycle comes
    # back — chunks near PACED_TARGET, a pause of max(MIN_PAUSE, cost x
    # DUTY_FACTOR) after each — so that holder's writes are not starved.
    _FTS_REBUILD_CHUNK_ROWS = 500       # first chunk of every phase and trash table; later ones adapt
    _FTS_REBUILD_CHUNK_MIN_ROWS = 50
    _FTS_REBUILD_CHUNK_MAX_ROWS = 100_000
    _FTS_REBUILD_CHUNK_TARGET_S = 0.25
    _FTS_REBUILD_PACED_TARGET_S = 0.05
    _FTS_REBUILD_DUTY_FACTOR = 4.0      # paced: sleep >= 4x chunk cost (≤20% duty)
    _FTS_REBUILD_MIN_PAUSE = 0.2        # paced: seconds — floor between chunks
    _fts_chunk_rows = _FTS_REBUILD_CHUNK_ROWS  # rows per step; optimize's drive loop adapts it per instance

    def _next_chunk_rows(self, rows: int, cost_s: float, target_s: float) -> int:
        """Rows for the next chunk so it costs about *target_s*; at most doubles or halves per
        step so one noisy chunk (a cold page, a checkpoint) can't swing it."""
        scale = target_s / max(cost_s, 1e-4)
        rows = int(rows * min(2.0, max(0.5, scale)))
        return min(self._FTS_REBUILD_CHUNK_MAX_ROWS, max(self._FTS_REBUILD_CHUNK_MIN_ROWS, rows))

    # Demoted v22 FTS shadow tables awaiting teardown: DROP of a multi-GB vtable
    # blocks for minutes, so the v23 migration renames the orphaned shadow tables
    # to fts_v22_trash_*; the worker empties them in chunks, then drops.
    _FTS_TRASH_PREFIX = "fts_v22_trash_"

    def _has_fts_trash(self, conn) -> bool:
        """True when demoted v22 shadow tables are still awaiting teardown.
        Caller must hold ``self._lock`` (or pass a migration-time cursor)."""
        return bool(conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name LIKE ? ESCAPE '\\' LIMIT 1",
            (self._FTS_TRASH_PREFIX.replace("_", "\\_") + "%",),
        ).fetchone())

    # FTS5 tables merged on optimize; each is probed before touching (trigram may be disabled).
    _FTS_TABLES = ("messages_fts", "messages_fts_trigram")
