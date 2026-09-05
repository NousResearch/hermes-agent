"""FTS5 index setup for SessionDB: the CJK-bigram (cjk_unicode61) index DDL and
tokenizer loader, per-table schema ensure with tokenizer-capability fallback,
FTS-scoped corruption detection and the atomic fail-open trigger detach."""

import logging
import os
import sqlite3
from pathlib import Path

from hermes_constants import get_hermes_home
from hermes_state_common import FTS_CJK_STALE_KEY, FTS_STALE_KEY, _FTS_CJK_TRIGGERS, _FTS_TRIGGERS

# caplog tests pin the "hermes_state" logger name.
logger = logging.getLogger("hermes_state")

# ── CJK-bigram FTS index (replaces the trigram index when available) ────
# Trigram needs >=3 chars per term, so 1-2 char CJK terms fell through to a LIKE
# table scan; ``cjk_unicode61`` (native/fts5_cjk/, loadable) re-emits CJK runs as
# overlapping bigrams. Same v23 discipline as the trigram table: external-content
# over a tool-row-excluding view, triggers gated on a DEDICATED marker pair
# (fts_cjk_rebuild_high_water / _progress). The table exists ONLY when the
# tokenizer loads; a process that cannot load it drops the cjk triggers (writes
# keep working; the index goes stale until the next optimize-storage).
#
# Split DDL: the table/view is safe to ensure any time; triggers are created ONLY
# while the index is complete-or-marker-gated. A stale index must keep its
# triggers DROPPED — an external-content 'delete' for a rowid the index never
# held is the canonical FTS5 corruption hazard.
# The trigram tokenizer needs >=3 chars per query term, so 1-2 char CJK terms (ubiquitous in Korean/Chinese:
# 일본, 구글, 项目, ...) fall through to a LIKE full-table scan — measured 3-6s CPU per query on multi-GB installs
# and the dominant base cost of session_search on CJK workloads. ``cjk_unicode61`` (native/fts5_cjk/, a
# ~250-line loadable FTS5 tokenizer with no dependencies) wraps unicode61: maximal CJK runs are re-emitted
# as overlapping character bigrams (Lucene CJKAnalyzer semantics), everything else passes through unchanged.
# FTS5 phrase semantics turn a query term's consecutive bigrams into exact substring matching down to 2
# chars at index speed. Contributed by Soju06 (PR #65544).
FTS_CJK_TABLE_SQL = """
CREATE VIEW IF NOT EXISTS messages_fts_cjk_src AS
    SELECT id, role, content, tool_name, tool_calls
    FROM messages
    WHERE role <> 'tool';

CREATE VIRTUAL TABLE IF NOT EXISTS messages_fts_cjk USING fts5(
    content,
    tool_name,
    tool_calls,
    content='messages_fts_cjk_src',
    content_rowid='id',
    tokenize='cjk_unicode61'
);
"""

FTS_CJK_TRIGGER_SQL = """
CREATE TRIGGER IF NOT EXISTS messages_fts_cjk_insert AFTER INSERT ON messages
WHEN new.role <> 'tool'
   AND (new.id > COALESCE((SELECT CAST(value AS INTEGER) FROM state_meta
                           WHERE key = 'fts_cjk_rebuild_high_water'), -1)
     OR new.id <= COALESCE((SELECT CAST(value AS INTEGER) FROM state_meta
                            WHERE key = 'fts_cjk_rebuild_progress'), -1))
BEGIN
    INSERT INTO messages_fts_cjk(rowid, content, tool_name, tool_calls)
    VALUES (new.id, new.content, new.tool_name, new.tool_calls);
END;

CREATE TRIGGER IF NOT EXISTS messages_fts_cjk_delete AFTER DELETE ON messages
WHEN old.role <> 'tool'
   AND (old.id > COALESCE((SELECT CAST(value AS INTEGER) FROM state_meta
                           WHERE key = 'fts_cjk_rebuild_high_water'), -1)
     OR old.id <= COALESCE((SELECT CAST(value AS INTEGER) FROM state_meta
                            WHERE key = 'fts_cjk_rebuild_progress'), -1))
BEGIN
    INSERT INTO messages_fts_cjk(messages_fts_cjk, rowid, content, tool_name, tool_calls)
    VALUES ('delete', old.id, old.content, old.tool_name, old.tool_calls);
END;

CREATE TRIGGER IF NOT EXISTS messages_fts_cjk_update
AFTER UPDATE OF content, tool_name, tool_calls, role ON messages
WHEN (old.content IS NOT new.content
    OR old.tool_name IS NOT new.tool_name
    OR old.tool_calls IS NOT new.tool_calls
    OR old.role IS NOT new.role)
   AND (old.id > COALESCE((SELECT CAST(value AS INTEGER) FROM state_meta
                           WHERE key = 'fts_cjk_rebuild_high_water'), -1)
     OR old.id <= COALESCE((SELECT CAST(value AS INTEGER) FROM state_meta
                            WHERE key = 'fts_cjk_rebuild_progress'), -1))
BEGIN
    INSERT INTO messages_fts_cjk(messages_fts_cjk, rowid, content, tool_name, tool_calls)
    SELECT 'delete', old.id, old.content, old.tool_name, old.tool_calls
    WHERE old.role <> 'tool';
    INSERT INTO messages_fts_cjk(rowid, content, tool_name, tool_calls)
    SELECT new.id, new.content, new.tool_name, new.tool_calls
    WHERE new.role <> 'tool';
END;
"""


def fts5_cjk_so_path() -> Path:
    """Location of the cjk_unicode61 loadable extension."""
    env = os.getenv("HERMES_FTS5_CJK_SO")
    return Path(env).expanduser() if env else get_hermes_home() / "lib" / "libfts5_cjk.so"


def _cjk_fts_config_enabled() -> bool:
    """config.yaml ``sessions.cjk_fts`` (default on), via its env bridge."""
    return os.getenv("HERMES_CJK_FTS", "1").strip().lower() not in ("0", "false", "off", "no")


def load_fts5_cjk_extension(conn: sqlite3.Connection) -> bool:
    """Best-effort load of the cjk_unicode61 tokenizer; False (never raises) when
    the .so is absent, ``sessions.cjk_fts`` is off, or loading is compiled out."""
    path = fts5_cjk_so_path()
    if not _cjk_fts_config_enabled() or not path.exists():
        return False
    try:
        conn.enable_load_extension(True)
        try:
            conn.load_extension(str(path))
        finally:
            conn.enable_load_extension(False)
        return True
    except Exception:
        logger.warning("fts5_cjk extension load failed (%s)", path, exc_info=True)
        return False


class SessionFtsSetupMixin:
    """FTS table/trigger lifecycle shared by schema init, optimize and the write path."""

    @staticmethod
    def _is_fts5_unavailable_error(exc: sqlite3.OperationalError) -> bool:
        """No FTS5 module, or an optional tokenizer missing (same capability-error shape)."""
        err = str(exc).lower()
        return ("no such module" in err and "fts5" in err) or SessionFtsSetupMixin._is_trigram_unavailable_error(exc)

    @staticmethod
    def _is_trigram_unavailable_error(exc: sqlite3.OperationalError) -> bool:
        """Only an optional tokenizer is missing (trigram needs SQLite >= 3.34;
        cjk_unicode61 is loadable): "this one index can't be served", never "disable FTS"."""
        err = str(exc).lower()
        return "no such tokenizer: trigram" in err or "no such tokenizer: cjk_unicode61" in err

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
        """True when the current FTS storage layout should be treated as stale (optimize-storage has work)."""
        return cls._db_has_legacy_inline_fts(cursor) or cls._db_has_trigram_tool_calls_projection(cursor)

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

    def _ensure_fts_cjk_schema(self, cursor) -> None:
        """Create / repair / self-heal the CJK-bigram index (see the module comment).
        Sets ``_fts_cjk_available``; never raises. Loaded + absent → create (a
        populated DB gets backfill markers and is NOT served until optimize-storage
        backfills); loaded + present → ensure triggers, honour the stale breadcrumb;
        NOT loaded + live triggers → drop them (INSERTs must not fail at trigger time)."""
        try:
            cjk_present = bool(cursor.execute(
                "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'messages_fts_cjk'"
            ).fetchone())
            if not self._fts_cjk_loaded:
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
                            fts5_cjk_so_path(),
                        )
                        cursor.execute(
                            "INSERT INTO state_meta (key, value) VALUES (?, '1') "
                            "ON CONFLICT(key) DO UPDATE SET value = '1'",
                            (FTS_CJK_STALE_KEY,),
                        )
                        for trig in live:
                            cursor.execute(f"DROP TRIGGER IF EXISTS {trig}")
                self._fts_cjk_available = False
                return
        except sqlite3.OperationalError:
            logger.warning(
                "messages_fts_cjk presence check failed; CJK search stays on "
                "trigram/LIKE", exc_info=True,
            )
            self._fts_cjk_available = False
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
                self._fts_cjk_available = False
                return
            cursor.executescript(FTS_CJK_TRIGGER_SQL)
            backfill_pending = cursor.execute(
                "SELECT 1 FROM state_meta WHERE key = 'fts_cjk_rebuild_high_water' LIMIT 1"
            ).fetchone()
            self._fts_cjk_available = not backfill_pending
        except sqlite3.OperationalError:  # incl. "no such tokenizer" after a failed registration
            logger.warning(
                "messages_fts_cjk ensure failed; CJK search stays on "
                "trigram/LIKE", exc_info=True,
            )
            self._fts_cjk_available = False

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
        """Corruption SQLite identifies as FTS-scoped (SQLITE_CORRUPT_VTAB, or an
        ``fts5:`` message on older builds); a bare malformed image is structural."""
        error_code = getattr(exc, "sqlite_errorcode", None)
        if error_code is not None:
            return error_code == getattr(sqlite3, "SQLITE_CORRUPT_VTAB", 267)
        msg = str(exc).lower()
        return msg.startswith("fts5:") and "corrupt structure" in msg

    def _enter_fts_fail_open(self, exc: sqlite3.DatabaseError) -> bool:
        """Detach corrupt FTS indexes so canonical writes can continue. Breadcrumb +
        trigger drop commit atomically: once triggers are absent the index has a
        gap of unknown extent, so nobody may reinstall them without a full rebuild."""
        if not self._fts_enabled or not self._is_fts_write_corruption_error(exc):
            return False
        self._raise_if_db_corrupt()
        self._halt_if_db_generation_changed()
        try:
            with self._lock:
                self._conn.execute("BEGIN IMMEDIATE")
                try:
                    self._conn.execute(
                        "INSERT INTO state_meta (key, value) VALUES (?, '1') "
                        "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                        (FTS_STALE_KEY,),
                    )
                    cjk_triggers_present = self._conn.execute(
                        "SELECT 1 FROM sqlite_master WHERE type = 'trigger' "
                        f"AND name IN ({','.join('?' for _ in _FTS_CJK_TRIGGERS)}) "
                        "LIMIT 1",
                        _FTS_CJK_TRIGGERS,
                    ).fetchone()
                    if cjk_triggers_present:
                        self._conn.execute(
                            "INSERT INTO state_meta (key, value) VALUES (?, '1') "
                            "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                            (FTS_CJK_STALE_KEY,),
                        )
                    self._drop_all_fts_triggers(self._conn.cursor())
                    self._conn.commit()
                except BaseException:
                    self._conn.rollback()
                    raise
        except sqlite3.Error as detach_exc:
            logger.error(
                "Could not detach corrupt FTS indexes; canonical write still cannot proceed: %s",
                detach_exc,
            )
            return False
        self._fts_stale = True
        self._fts_enabled = False
        self._trigram_available = False
        self._fts_cjk_available = False
        logger.error(
            "state.db FTS indexes remain corrupt (%s); disabled FTS sync and "
            "retrying the canonical write. Search temporarily uses LIKE until "
            "a later SessionDB open rebuilds the indexes.",
            exc,
        )
        return True

    # ── Chunked FTS rebuild engine (v23 opt-in optimize) ──
    # One blocking rebuild held the write lock ~16 min on a 25 GB DB, so the
    # backfill runs in small chunks (each its own short transaction, resumable from
    # fts_rebuild_progress, claimed by CAS). A greedy loop starved other writers:
    # a pause of max(MIN_PAUSE, chunk cost x DUTY_FACTOR) caps the duty cycle
    # cross-process, unlike any same-process activity stamp.
    _FTS_REBUILD_CHUNK_ROWS = 500
    _FTS_REBUILD_DUTY_FACTOR = 4.0      # sleep >= 4x chunk cost (≤20% duty)
    _FTS_REBUILD_MIN_PAUSE = 0.2        # seconds — floor between chunks

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

    # FTS5 tables merged on optimize; each is probed before touching (trigram may
    # be disabled, cjk exists only with the loadable tokenizer).
    _FTS_TABLES = ("messages_fts", "messages_fts_trigram", "messages_fts_cjk")
