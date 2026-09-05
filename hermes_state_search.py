"""Full-text / trigram / CJK message search and FTS maintenance for SessionDB.

Plain mixin for ``hermes_state.SessionDB`` (no ``__init__``/state of its own).
Must never import hermes_state (cycle); shared constants live in hermes_state_common.
"""

import contextlib
import logging
import re
import sqlite3
import time
from typing import Any, Callable, Collection, Dict, List, Optional, Tuple

from agent.skill_commands import describe_skill_invocation
from utils import env_float
from hermes_state_common import (
    FTS_CJK_STALE_KEY, FTS_SQL, FTS_STALE_KEY, FTS_STORAGE_VERSION, FTS_TOOL_CONTENT_PREFIX_CHARS,
    FTS_TOOL_FULL_CONTENT_HIGH_WATER_KEY, FTS_TRIGRAM_EXCLUDED_SOURCES, FTS_TRIGRAM_SQL,
    MAX_FTS5_QUERY_CHARS, SCHEMA_VERSION, _FTS_CJK_TRIGGERS,
    escape_like as _escape_like, fts_rebuild_admission, fts_trigram_session_sql,
)

# Pre-split logger identity so log filtering/capture is unchanged.
logger = logging.getLogger("hermes_state")

# Characters FTS5's query grammar rejects outside a quoted phrase (anything missing
# reaches MATCH raw and raises -> zero results). ``%`` is deliberately excluded: the
# CJK LIKE fallback needs it literal (that path escapes wildcards itself).
_FTS5_SPECIAL_CHARS = '+{}():"^@/#&|~[]<>,;!?$=\\\''
_FTS5_SPECIAL_RE = re.compile(f"[{re.escape(_FTS5_SPECIAL_CHARS)}]")

_FTS_OPERATORS = frozenset({"AND", "OR", "NOT"})
_LIKE_SKIP_TOKENS = _FTS_OPERATORS | {"NEAR"}
_LIKE_TOKEN_RE = re.compile(r'"[^"]+"|\S+')
_QUOTED_PHRASE_RE = re.compile(r'"[^"]*"')

# Column list shared by every search route (snippet + metadata, never content).
_SEARCH_SELECT_TAIL = "m.timestamp, m.tool_name, s.source, s.model, s.started_at AS session_started"
_LIKE_SNIPPET_SQL = "substr(m.content, max(1, instr(m.content, ?) - 40), 120) AS snippet"
_LIKE_ANY_COLUMN_SQL = (
    "(m.content LIKE ? ESCAPE '\\' OR m.tool_name LIKE ? ESCAPE '\\' OR m.tool_calls LIKE ? ESCAPE '\\')"
)
_LIKE_COALESCED_COLUMN_SQL = (
    "(COALESCE(m.content, '') LIKE ? ESCAPE '\\' OR "
    "COALESCE(m.tool_name, '') LIKE ? ESCAPE '\\' OR "
    "COALESCE(m.tool_calls, '') LIKE ? ESCAPE '\\')"
)
# ``sort`` -> ORDER BY for the FTS routes; unknown values are rank-only (user input passes through).
_FTS_ORDER_BY = {"newest": "ORDER BY m.timestamp DESC, rank", "oldest": "ORDER BY m.timestamp ASC, rank"}
# One row before + the hit + one row after, in (timestamp, id) order.
_CONTEXT_WINDOW_SQL = """WITH target AS (
                               SELECT session_id, timestamp, id
                               FROM messages
                               WHERE id = ?
                           )
                           SELECT role, content
                           FROM (
                               SELECT m.id, m.timestamp, m.role, m.content
                               FROM messages m
                               JOIN target t ON t.session_id = m.session_id
                               WHERE (m.timestamp < t.timestamp)
                                  OR (m.timestamp = t.timestamp AND m.id < t.id)
                               ORDER BY m.timestamp DESC, m.id DESC
                               LIMIT 1
                           )
                           UNION ALL
                           SELECT role, content
                           FROM messages
                           WHERE id = ?
                           UNION ALL
                           SELECT role, content
                           FROM (
                               SELECT m.id, m.timestamp, m.role, m.content
                               FROM messages m
                               JOIN target t ON t.session_id = m.session_id
                               WHERE (m.timestamp > t.timestamp)
                                  OR (m.timestamp = t.timestamp AND m.id > t.id)
                               ORDER BY m.timestamp ASC, m.id ASC
                               LIMIT 1
                           )"""
# Unified Ideographs, Extension A, Extension B, CJK Symbols, Hiragana, Katakana, Hangul Syllables.
_CJK_RANGES = (
    (0x4E00, 0x9FFF), (0x3400, 0x4DBF), (0x20000, 0x2A6DF), (0x3000, 0x303F), (0x3040, 0x309F),
    (0x30A0, 0x30FF), (0xAC00, 0xD7AF),
)


def _meta_row(conn, key: str) -> Optional[sqlite3.Row]:
    """Point-read one ``state_meta`` row (``None`` when absent)."""
    return conn.execute("SELECT value FROM state_meta WHERE key = ?", (key,)).fetchone()


def _delete_meta(conn, *keys: str) -> None:
    conn.execute(f"DELETE FROM state_meta WHERE key IN ({','.join('?' for _ in keys)})", keys)


def _is_cjk(cp: int) -> bool:
    return any(lo <= cp <= hi for lo, hi in _CJK_RANGES)


def _non_operator_tokens(raw_query: str) -> List[str]:
    return [t for t in raw_query.split() if t.upper() not in _FTS_OPERATORS]


def _quote_fts_tokens(raw_query: str) -> str:
    """Quote each non-operator token (neutralising FTS5 special characters), keeping AND/OR/NOT."""
    return " ".join(
        tok if tok.upper() in _FTS_OPERATORS else '"' + tok.replace('"', '""') + '"' for tok in raw_query.split()
    )


def _like_params(term: str) -> List[str]:
    """One ``%term%`` bind per column of ``_LIKE_ANY_COLUMN_SQL``."""
    return [f"%{_escape_like(term)}%"] * 3


def _flatten_text(decoded: Any) -> str:
    """Multimodal part list -> joined text (or the placeholder); str passes through; else ''."""
    if isinstance(decoded, list):
        parts = [p.get("text", "") for p in decoded if isinstance(p, dict) and p.get("type") == "text"]
        return " ".join(t for t in parts if t).strip() or "[multimodal content]"
    return decoded if isinstance(decoded, str) else ""


def _positive_int(name: str, value: Any) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value <= 0:
        raise ValueError(f"{name} must be greater than zero")


def _search_select_sql(snippet_sql: str, from_sql: str, where: List[str], order_by: str, limit_sql: str) -> str:
    """Result-row SELECT shared by the FTS and LIKE routes (SQL text is pinned)."""
    return f"""
            SELECT m.id, m.session_id, m.role,
                   {snippet_sql},
                   {_SEARCH_SELECT_TAIL}
            FROM {from_sql}
            JOIN sessions s ON s.id = m.session_id
            WHERE {' AND '.join(where)}
            {order_by}
            {limit_sql}
        """


def _search_filter_clauses(
    where: List[str], params: list, *, include_inactive: bool, source_filter: Optional[List[str]],
    exclude_sources: Optional[List[str]], role_filter: Optional[List[str]]) -> None:
    """Append the visibility/source/role predicates every search route shares. Live rows
    (active=1) AND compaction-archived rows (compacted=1) are discoverable; only
    rewind/undo rows (active=0, compacted=0) are hidden."""
    if not include_inactive:
        where.append("(m.active = 1 OR m.compacted = 1)")
    if source_filter is not None:
        where.append(f"s.source IN ({','.join('?' for _ in source_filter)})")
        params.extend(source_filter)
    if exclude_sources is not None:
        where.append(f"s.source NOT IN ({','.join('?' for _ in exclude_sources)})")
        params.extend(exclude_sources)
    if role_filter:
        where.append(f"m.role IN ({','.join('?' for _ in role_filter)})")
        params.extend(role_filter)


class SessionSearchMixin:
    """See module docstring — mixin for SessionDB (Search cluster)."""

    _SEARCH_MESSAGE_RESULT_FIELDS = (
        "id", "session_id", "role", "snippet", "timestamp", "tool_name", "source", "model", "session_started", "context"
    )

    @classmethod
    def _search_message_fields(cls, fields: Optional[Collection[str]]) -> Optional[Tuple[str, ...]]:
        """Validate and canonically order an optional result projection."""
        if fields is None:
            return None
        if isinstance(fields, str):
            raise TypeError("search fields must be a collection of field names, not a string")
        requested = set(fields)
        unknown = requested.difference(cls._SEARCH_MESSAGE_RESULT_FIELDS)
        if unknown:
            raise ValueError(f"unknown search result field(s): {', '.join(sorted(unknown))}")
        return tuple(field for field in cls._SEARCH_MESSAGE_RESULT_FIELDS if field in requested)

    def _try_incremental_merge_fts(self) -> None:
        """One bounded FTS5 merge pass that never fails the already-committed write (a caller
        must never replay an ambiguous, possibly-durable write — even on the bare
        SystemError CPython's sqlite3 layer can raise under cross-thread errmsg scrambling)."""
        if not self._fts_enabled:
            return
        try:
            self._merge_fts_incrementally(max_pages=self._FTS_MERGE_MAX_PAGES_PER_INDEX)
        except Exception as exc:  # noqa: BLE001 - post-commit maintenance
            # The canonical write is already committed before this cadence runs. No maintenance failure —
            # including the bare SystemError the CPython sqlite3 layer can raise under cross-thread errmsg
            # scrambling — may escape and make the caller replay an ambiguous, possibly-durable write
            # (#90734, #85079).
            logger.warning("FTS incremental merge failed after commit: %s", exc)

    # ── Deferred rebuild engine (base + CJK backfills) ─────────────────────

    def fts_rebuild_status(self) -> Optional[Dict[str, Any]]:
        """Deferred-rebuild progress ``{"pending", "total", "indexed", "percent"}`` or None. Reads
        via the pooled reader (not get_meta/self._lock) so search never blocks on the writer."""
        return self._rebuild_status("fts_rebuild")

    def fts_cjk_rebuild_status(self) -> Optional[Dict[str, Any]]:
        """CJK-index backfill progress, or None when none is pending."""
        return self._rebuild_status("fts_cjk_rebuild")

    def _rebuild_status(self, prefix: str) -> Optional[Dict[str, Any]]:
        rows = self._read_all("SELECT key, value FROM state_meta WHERE key IN (?, ?)",
                              (f"{prefix}_high_water", f"{prefix}_progress"))
        meta = {r["key"]: r["value"] for r in rows}
        high_water = meta.get(f"{prefix}_high_water")
        if high_water is None or int(high_water) <= 0:
            return None
        total, progress = int(high_water), int(meta.get(f"{prefix}_progress") or 0)
        return {"pending": True, "total": total, "indexed": progress, "percent": min(100, int(100 * progress / total))}

    # Re-index rows in an id window the index is missing. docsize has one row
    # per indexed doc, so the anti-join is exact. Params: (lo, hi) — the base sweep
    # takes (hw, prefix_chars, lo, hi): tool rows past the high water index only a prefix.
    _BOUNDARY_SWEEP_SQL = (
        "INSERT INTO {table}(rowid, content, tool_name, tool_calls) "
        "SELECT m.id, m.content, m.tool_name, m.tool_calls FROM messages m WHERE m.id > ? AND m.id <= ? {extra}"
        "AND NOT EXISTS (SELECT 1 FROM {table}_docsize d WHERE d.id = m.id)"
    )
    _BASE_BOUNDARY_SWEEP_SQL = (
        "INSERT INTO messages_fts(rowid, content, tool_name, tool_calls) "
        "SELECT m.id, CASE WHEN m.role = 'tool' AND m.id > ? THEN substr(COALESCE(m.content, ''), 1, ?) "
        "ELSE m.content END, m.tool_name, m.tool_calls FROM messages m WHERE m.id > ? AND m.id <= ? "
        "AND NOT EXISTS (SELECT 1 FROM messages_fts_docsize d WHERE d.id = m.id)"
    )
    # Trigram excludes tool rows and FTS_TRIGRAM_EXCLUDED_SOURCES sessions; no tool_calls column.
    _TRIGRAM_BOUNDARY_SWEEP_SQL = (
        "INSERT INTO messages_fts_trigram(rowid, content, tool_name) "
        "SELECT m.id, m.content, m.tool_name FROM messages m JOIN sessions s ON s.id = m.session_id "
        f"WHERE m.id > ? AND m.id <= ? AND m.role <> 'tool' AND {fts_trigram_session_sql('s')} "
        "AND NOT EXISTS (SELECT 1 FROM messages_fts_trigram_docsize d WHERE d.id = m.id)"
    )
    _CHUNK_INSERT_SQL = (
        "INSERT INTO {table}(rowid, content, tool_name, tool_calls) "
        "SELECT id, content, tool_name, tool_calls FROM messages WHERE id > ? AND id <= ?{extra}"
    )
    _TRIGRAM_CHUNK_INSERT_SQL = (
        "INSERT INTO messages_fts_trigram(rowid, content, tool_name) "
        "SELECT m.id, m.content, m.tool_name FROM messages m JOIN sessions s ON s.id = m.session_id "
        f"WHERE m.id > ? AND m.id <= ? AND m.role <> 'tool' AND {fts_trigram_session_sql('s')}"
    )

    def _fts_rebuild_finish(self) -> None:
        """Finalize the deferred rebuild: boundary sweep + clear markers. The sweep is cheap
        insurance against a write that slipped between high_water capture and trigger
        activation. The trigram half is gated on ``_trigram_available``: without the
        tokenizer/table an unconditional INSERT raises and aborts the whole rebuild."""
        sweeps = [(self._BASE_BOUNDARY_SWEEP_SQL, True)]
        if self._trigram_available:
            sweeps.append((self._TRIGRAM_BOUNDARY_SWEEP_SQL, False))
        self._rebuild_finish("fts_rebuild", sweeps)
        logger.info("Deferred FTS rebuild complete — all messages indexed.")

    def _fts_cjk_rebuild_finish(self) -> None:
        """Boundary sweep + clear the cjk markers; index becomes servable."""
        sweep = self._BOUNDARY_SWEEP_SQL.format(table="messages_fts_cjk", extra="AND m.role <> 'tool' ")
        self._rebuild_finish("fts_cjk_rebuild", [(sweep, False)])
        self._fts_cjk_available = True
        logger.info("CJK FTS index backfill complete — serving CJK search.")

    def _rebuild_finish(self, prefix: str, sweep_sqls: List[Tuple[str, bool]]) -> None:
        """Sweep a generous window around the high-water boundary, then clear the markers.
        ``(sql, bounded)``: a bounded sweep takes the (hw, prefix_chars) tool-content params first."""
        def _do(conn):
            hw_row = _meta_row(conn, f"{prefix}_high_water")
            if hw_row is not None:
                hw = int(hw_row[0])
                for sql, bounded in sweep_sqls:
                    params = (hw, FTS_TOOL_CONTENT_PREFIX_CHARS) if bounded else ()
                    conn.execute(sql, (*params, hw - 1000, hw + 1000))
            _delete_meta(conn, f"{prefix}_high_water", f"{prefix}_progress")
        self._execute_write(_do)

    def fts_rebuild_step(self) -> bool:
        """Backfill one chunk of the deferred FTS rebuild; True while work remains. Chunks are
        claimed atomically inside the write transaction, so concurrent processes interleave
        instead of duplicating rows."""
        if not self._fts_enabled:
            return False
        inserts = [self._CHUNK_INSERT_SQL.format(table="messages_fts", extra="")]
        if self._trigram_available:
            inserts.append(self._TRIGRAM_CHUNK_INSERT_SQL)
        return self._rebuild_step("fts_rebuild", inserts, fail_msg="FTS rebuild chunk failed (will retry): %s",
                                  finish=self._fts_rebuild_finish, finish_when_empty=True)

    def fts_cjk_rebuild_step(self) -> bool:
        """Backfill one chunk of the CJK index. True while work remains."""
        if not self._fts_enabled or not self._fts_cjk_loaded:
            return False
        insert = self._CHUNK_INSERT_SQL.format(table="messages_fts_cjk", extra=" AND role <> 'tool'")
        return self._rebuild_step("fts_cjk_rebuild", [insert], finish=self._fts_cjk_rebuild_finish,
                                  fail_msg="CJK FTS rebuild chunk failed (will retry): %s")

    def _rebuild_step(self, prefix: str, insert_sqls: List[str], *, fail_msg: str, finish,
                      finish_when_empty: bool = False) -> bool:
        """Shared chunk engine for the base and CJK deferred backfills. ``finish_when_empty``
        finalizes a high_water <= 0 marker (empty messages table) instead of leaving it pending."""
        high_water_raw = self.get_meta(f"{prefix}_high_water")
        if high_water_raw is None:
            return False
        high_water = int(high_water_raw)
        chunk = self._FTS_REBUILD_CHUNK_ROWS

        def _do(conn):
            # Re-reading progress inside the BEGIN IMMEDIATE held by _execute_write IS
            # the claim: two workers cannot read the same value.
            row = _meta_row(conn, f"{prefix}_progress")
            if row is None:
                return False  # finished (or cleared) by another process
            progress = int(row[0])
            if progress >= high_water:
                return False
            # Upper bound is an id, not a row count, so deleted-row gaps don't shrink chunks.
            upper = min(progress + chunk, high_water)
            for sql in insert_sqls:
                conn.execute(sql, (progress, upper))
            # Progress lands in the same transaction as its rows (crash-atomic).
            conn.execute("UPDATE state_meta SET value = ? WHERE key = ?", (str(upper), f"{prefix}_progress"))
            return upper < high_water

        try:
            more = self._execute_write(_do)
        except sqlite3.OperationalError as exc:
            logger.debug(fail_msg, exc)
            return True  # transient (lock contention) — caller retries
        if more is False:
            status = self._rebuild_status(prefix)
            if (finish_when_empty and high_water <= 0) or (
                status is not None and status["indexed"] >= status["total"]
            ):
                finish()
            return False
        return bool(more)

    def _fts_teardown_trash_step(self) -> bool:
        """Tear down one chunk of a demoted v22 FTS shadow table (a PLAIN table now); True while
        work remains. INTEGER single-column-key tables drain with a high-water marker so
        each chunk's scan is bounded (restarting the scan was O(n²)); compound-key tables
        keep the chunked ``LIMIT`` delete — they are small by construction.

        Single-column-key trash tables (the common shape — FTS shadow tables carry a rowid/integer PK) are
        drained with a high-water marker mirroring :meth:`fts_rebuild_step`: each chunk deletes only rows
        after the previously-drained key, so the per-chunk scan is bounded instead of re-scanning from the
        start of the table every chunk (O(n²) total on large trash tables, #79324).
        """
        with self._read_ctx() as conn:
            trash = [r[0] for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table' AND name LIKE ? ESCAPE '\\'",
                (self._FTS_TRASH_PREFIX.replace("_", "\\_") + "%",),
            ).fetchall()]
        if not trash:
            return False
        tbl = trash[0]

        def _do(conn):
            pk_info = [(r[1], (r[2] or "").upper()) for r in conn.execute(f"PRAGMA table_info({tbl})") if r[5] > 0]
            key = ", ".join(name for name, _typ in pk_info) if pk_info else "rowid"
            if len(pk_info) == 1 and pk_info[0][1] == "INTEGER":
                # High-water drain; marker read/written in the same BEGIN IMMEDIATE as the
                # DELETE so concurrent callers claim disjoint ranges. Only INTEGER pks
                # anchor the comparison (the TEXT-pk config shadow table falls through).
                marker_key = f"fts_teardown_{tbl}_progress"
                row = _meta_row(conn, marker_key)
                high_water = int(row[0]) if row is not None else 0
                # Claim the LAST row of the LIMIT window so a full chunk goes per step.
                upper_rows = conn.execute(
                    f"SELECT {key} FROM {tbl} WHERE {key} > ? ORDER BY {key} LIMIT {self._FTS_REBUILD_CHUNK_ROWS}",
                    (high_water,),
                ).fetchall()
                if not upper_rows:
                    return _drop(conn, marker_key)
                upper = upper_rows[-1][0]
                cur = conn.execute(f"DELETE FROM {tbl} WHERE {key} > ? AND {key} <= ?", (high_water, upper))
                if cur.rowcount > 0:
                    self.set_meta(marker_key, str(upper), cursor=conn)
                return True
            # Compound-key or rowid trash table: legacy chunked delete. These shadow tables are small, so
            # the quadratic re-scan is not a concern (#79324 keeps the high-water path for the big
            # single-key tables).
            cur = conn.execute(
                f"DELETE FROM {tbl} WHERE ({key}) IN (SELECT {key} FROM {tbl} LIMIT {self._FTS_REBUILD_CHUNK_ROWS})"
            )
            return _drop(conn) if cur.rowcount == 0 else True  # True: more trash tables / chunks may remain

        def _drop(conn, marker_key: Optional[str] = None) -> bool:
            """Drained — the DROP is cheap now. True: re-check for more trash."""
            conn.execute(f"DROP TABLE IF EXISTS {tbl}")
            if marker_key is not None:
                _delete_meta(conn, marker_key)
            logger.info("Old FTS shadow table %s torn down.", tbl)
            return True

        try:
            return bool(self._execute_write(_do))
        except sqlite3.OperationalError as exc:
            logger.debug("FTS trash teardown chunk failed (will retry): %s", exc)
            return True

    def _fts_cjk_reset_if_stale(self) -> None:
        """From-scratch rebuild of a stale cjk index (triggers were dropped, gap extent unknown):
        drop table + triggers, clear the breadcrumb, recreate (fresh backfill markers)."""
        if not self._fts_cjk_loaded:
            return

        def _do(conn):
            if _meta_row(conn, FTS_CJK_STALE_KEY) is None:
                return False
            for trig in _FTS_CJK_TRIGGERS:
                conn.execute(f"DROP TRIGGER IF EXISTS {trig}")
            conn.execute("DROP TABLE IF EXISTS messages_fts_cjk")
            conn.execute("DROP VIEW IF EXISTS messages_fts_cjk_src")
            _delete_meta(conn, FTS_CJK_STALE_KEY, "fts_cjk_rebuild_high_water", "fts_cjk_rebuild_progress")
            return True
        if self._execute_write(_do):
            # Recreate OUTSIDE the write transaction: executescript() implicitly commits.
            self._ensure_cjk_schema_committed()

    def _ensure_cjk_schema_committed(self) -> None:
        with self._lock:
            self._ensure_fts_cjk_schema(self._conn)
            self._conn.commit()

    def _fts_external_index_empty_with_messages(self, conn) -> bool:
        """True when the base FTS table indexes nothing while ``messages`` has rows (post-demote
        empty-index shape). Caller holds ``self._lock``. docsize is the authoritative "is this
        rowid indexed" surface; EXISTS not COUNT(*) because this runs on every writable open."""
        try:
            if not conn.execute("SELECT EXISTS(SELECT 1 FROM messages)").fetchone()[0]:
                return False
            return not conn.execute("SELECT EXISTS(SELECT 1 FROM messages_fts_docsize)").fetchone()[0]
        except sqlite3.OperationalError:
            return False  # table absent / FTS disabled mid-init — not this failure class

    def _reseed_missing_progress(self, conn) -> None:
        """high_water without progress: fts_rebuild_step reads missing progress as "done by
        another process" and optimize would no-op then stamp. Reset to known-empty, re-seed.
        Truncation goes through FTS5 ``'delete-all'`` (a plain DELETE is O(rows) and corrupts
        the index when indexed rows diverged from ``messages``); the backfill worker replays
        without an anti-join, so it needs a known-empty index. A missing docsize table counts
        as empty."""
        if _meta_row(conn, "fts_rebuild_progress") is None:
            if not self._fts_index_known_empty(conn):
                self._reset_fts_index_to_empty(conn)
            self.set_meta("fts_rebuild_progress", "0", cursor=conn)

    @staticmethod
    def _fts_index_known_empty(conn) -> bool:
        """True when the base external-content index holds no rows (a missing table counts as empty)."""
        try:
            return int(conn.execute("SELECT COUNT(*) FROM messages_fts_docsize").fetchone()[0]) == 0
        except sqlite3.OperationalError:
            return True

    @staticmethod
    def _reset_fts_index_to_empty(conn) -> None:
        """Truncate the v23 external-content tables via FTS5 ``'delete-all'`` (O(1); a plain DELETE is
        O(rows) and corrupts the index when indexed rows diverged from ``messages``)."""
        for tbl in ("messages_fts", "messages_fts_trigram"):
            with contextlib.suppress(sqlite3.OperationalError):  # table absent — already an empty surface
                conn.execute(f"INSERT INTO {tbl}({tbl}) VALUES('delete-all')")

    def _seed_fts_rebuild_markers(self, conn, *, force: bool = False) -> int:
        """Write ``fts_rebuild_high_water`` / ``fts_rebuild_progress`` for a full backfill; returns
        the high-water id. Without ``force`` an existing high_water only gets a missing
        progress key repaired. Caller holds the write transaction."""
        existing_hw = _meta_row(conn, "fts_rebuild_high_water")
        if existing_hw is not None and not force:
            self._reseed_missing_progress(conn)
            self.set_meta(FTS_TOOL_FULL_CONTENT_HIGH_WATER_KEY, str(int(existing_hw[0])), cursor=conn)
            return int(existing_hw[0])
        hw = conn.execute("SELECT COALESCE(MAX(id), 0) FROM messages").fetchone()[0]
        self.set_meta("fts_rebuild_high_water", str(hw), cursor=conn)
        self.set_meta("fts_rebuild_progress", "0", cursor=conn)
        self.set_meta(FTS_TOOL_FULL_CONTENT_HIGH_WATER_KEY, str(hw), cursor=conn)
        return int(hw)

    def _repair_optimize_bookkeeping(self) -> None:
        """Heal interrupted demote/backfill bookkeeping before optimize runs: orphan high_water
        gets progress re-seeded; an empty external index with messages and no markers gets
        a full backfill claim. Never invents markers on a still-legacy inline DB — optimize
        would then skip demote and INSERT against the inline table forever.

        Covers two post-#65798 failure classes:
        """
        def _do(conn):
            if _meta_row(conn, "fts_rebuild_high_water") is not None:
                self._reseed_missing_progress(conn)
                return
            if self._db_has_legacy_inline_fts(conn):
                return  # demote owns marker creation
            if self._fts_external_index_empty_with_messages(conn):
                _delete_meta(conn, "fts_storage_version")
                self._seed_fts_rebuild_markers(conn, force=True)
        self._execute_write(_do)

    def fts_optimize_available(self) -> bool:
        """True when `optimize_fts_storage()` has work: legacy inline FTS or a v23 trigram still
        carrying ``tool_calls`` (``_db_needs_fts_storage_upgrade``), an interrupted optimize
        (markers/trash), a CJK backfill on this tokenizer-capable host, or an empty external
        index without markers. False when FTS5 is unavailable."""
        if not self._fts_enabled or self.read_only:
            return False
        with self._read_ctx() as conn:
            return (
                self._db_needs_fts_storage_upgrade(conn)
                or _meta_row(conn, "fts_rebuild_high_water") is not None  # interrupted optimize
                # CJK work is only offerable when THIS process can tokenize.
                or (self._fts_cjk_loaded and (
                    _meta_row(conn, "fts_cjk_rebuild_high_water") is not None
                    or _meta_row(conn, FTS_CJK_STALE_KEY) is not None
                ))
                or self._has_fts_trash(conn)
                or self._fts_external_index_empty_with_messages(conn)
            )

    def _demote_legacy_fts_to_trash(self) -> int:
        """Demote upgrade-eligible FTS vtables and stage their shadow tables for chunked
        teardown; returns MAX(messages.id) as the rebuild high water. O(1) schema surgery
        — the heavy delete is deferred. Markers land in the same BEGIN IMMEDIATE, BEFORE
        the empty v23 schema is created (``executescript`` implicitly COMMITs), closing
        the crash window where trash + empty v23 tables exist with no backfill claim."""
        def _stage(conn):
            self._drop_fts_triggers(conn)
            conn.execute("DROP VIEW IF EXISTS messages_fts_trigram_src")
            if conn.execute(
                "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name IN ('messages_fts', 'messages_fts_trigram') "
                "AND sql LIKE 'CREATE VIRTUAL TABLE%' LIMIT 1"
            ).fetchone():
                conn.execute("PRAGMA writable_schema=ON")
                conn.execute(
                    "DELETE FROM sqlite_master WHERE type = 'table' "
                    "AND name IN ('messages_fts', 'messages_fts_trigram') AND sql LIKE 'CREATE VIRTUAL TABLE%'"
                )
                conn.execute("PRAGMA writable_schema=RESET")
                for row in conn.execute(
                    "SELECT name FROM sqlite_master WHERE type = 'table' "
                    "AND (name LIKE 'messages_fts_%' ESCAPE '\\' "
                    "OR name LIKE 'messages_fts_trigram_%' ESCAPE '\\')"
                ).fetchall():
                    conn.execute(f"ALTER TABLE {row[0]} RENAME TO fts_v22_trash_{row[0]}")
            # Claim the backfill BEFORE the empty v23 tables exist so a crash before
            # schema ensure resumes instead of stamping an empty index.
            hw = self._seed_fts_rebuild_markers(conn, force=True)
            _delete_meta(conn, "fts_optimize_available")
            return hw

        hw = int(self._execute_write(_stage))
        # Outside the write transaction (executescript commits); markers are durable.
        self._ensure_v23_fts_tables("failed to create v23 messages_fts during optimize-storage demote")
        return hw

    def _ensure_v23_fts_tables(self, failure_message: str) -> None:
        """Ensure the v23 base + trigram tables under the lock (IF NOT EXISTS, cheap); raise
        *failure_message* without the base table (the backfill loop would retry forever)."""
        with self._lock:
            base_ok = self._ensure_fts_schema(self._conn, "messages_fts", FTS_SQL)
            trigram_ok = self._ensure_fts_schema(self._conn, "messages_fts_trigram", FTS_TRIGRAM_SQL)
            self._trigram_available = bool(trigram_ok)
            if not base_ok:
                raise sqlite3.OperationalError(failure_message)
            self._conn.commit()

    def _optimize_vacuum(self) -> bool:
        """Phase 3: reclaim freed pages to the OS. False when VACUUM failed (usually no free disk
        for its temp copy; a later VACUUM reclaims)."""
        try:
            with self._lock:
                self._conn.execute("VACUUM")
            vacuum_ok = True
        except sqlite3.OperationalError as exc:
            logger.warning("VACUUM after FTS optimize failed: %s", exc)
            vacuum_ok = False
        # Best-effort WAL fold-back, REFUSED (SQLITE_BUSY) while another connection holds a
        # read-mark (callers size via logical_size_bytes, not stat()). PASSIVE, never TRUNCATE:
        # a TRUNCATE reset from a transient CLI would race a live writer.
        try:
            with self._lock:
                # Best-effort: fold the WAL back into the main file so the on-disk size settles now rather
                # than at close(). Callers must therefore NOT size the result by stat()ing the file; use
                # :meth:`logical_size_bytes`, which is truthful immediately regardless of readers. See
                # #45383.
                self._conn.execute("PRAGMA wal_checkpoint(PASSIVE)")
        except Exception as exc:
            logger.debug("WAL checkpoint (PASSIVE) after optimize VACUUM failed: %s", exc)
        return vacuum_ok

    def _optimize_settle(self, conn) -> Optional[str]:
        """Phase 4 (inside the write transaction, so a concurrent writer cannot race a stamp past
        incomplete work): stamp the FTS layout (source of truth for "optimized"), clear the
        "available" flag, advance a lagging schema_version. Returns a refusal reason or None.
        Refuses while optimize work remains; an empty base index against non-empty messages
        also refuses (settling there meant permanent search-index loss)."""
        if _meta_row(conn, "fts_rebuild_high_water") is not None:
            return "backfill_incomplete"
        if self._has_fts_trash(conn):
            return "teardown_incomplete"
        if self._fts_external_index_empty_with_messages(conn):
            return "backfill_incomplete"
        self.set_meta("fts_storage_version", str(FTS_STORAGE_VERSION), cursor=conn)
        _delete_meta(conn, "fts_optimize_available")
        conn.execute("UPDATE schema_version SET version = ? WHERE version < ?", (SCHEMA_VERSION, SCHEMA_VERSION))
        return None

    def optimize_fts_storage(
        self, *, progress_cb: Optional[Callable[[Dict[str, Any]], None]] = None, vacuum: bool = True
    ) -> Dict[str, Any]:
        """Repair an older FTS layout into the current v23 shape, foreground and to completion:
        legacy-v22 inline -> external-content, or a v23 ``messages_fts_trigram`` that still stores
        ``tool_calls``. Re-running resumes. ``progress_cb`` receives {"phase", "percent",
        "indexed", "total"}. A missing trigram tokenizer is not fatal (CJK falls back to LIKE)."""
        if not self._fts_enabled:
            return {"ok": False, "reason": "fts5_unavailable"}
        if self.read_only:
            return {"ok": False, "reason": "read_only"}

        # Heal bookkeeping BEFORE deciding whether to demote again.
        self._repair_optimize_bookkeeping()
        with self._lock:
            needs_storage_upgrade = self._db_needs_fts_storage_upgrade(self._conn)
        pending = self.get_meta("fts_rebuild_high_water") is not None
        if needs_storage_upgrade and not pending:
            self._demote_legacy_fts_to_trash()
        elif pending and not needs_storage_upgrade:
            # Resume mid-demote: the process may have died between the staged demote
            # commit and schema ensure.
            self._ensure_v23_fts_tables("failed to re-create v23 messages_fts on optimize-storage resume")

        # A stale CJK index can only be recovered from scratch; then ensure table +
        # markers exist (a v23 DB gaining the cjk index for the first time).
        self._fts_cjk_reset_if_stale()
        if self._fts_cjk_loaded:
            self._ensure_cjk_schema_committed()

        def _emit(phase: str) -> None:
            if progress_cb is None:
                return
            st = self.fts_rebuild_status() or self.fts_cjk_rebuild_status()
            progress_cb({"phase": phase, "percent": st["percent"] if st else 100,
                         "indexed": st["indexed"] if st else 0, "total": st["total"] if st else 0})

        def _drive(phase: str, step) -> None:
            """Run *step* to completion; the inter-chunk sleep is the single place the duty
            cycle is enforced — back-to-back BEGIN IMMEDIATE chunks starve a live
            gateway/CLI out of its lock retries."""
            while True:
                _t0 = time.monotonic()
                if not step():
                    break
                _emit(phase)
                time.sleep(max(self._FTS_REBUILD_MIN_PAUSE, (time.monotonic() - _t0) * self._FTS_REBUILD_DUTY_FACTOR))

        # Phase 1: base backfill; 1b: CJK-bigram backfill (own marker pair).
        _emit("backfill")
        _drive("backfill", self.fts_rebuild_step)
        _emit("backfill")
        _drive("backfill", self.fts_cjk_rebuild_step)
        # Phase 2: tear down the demoted legacy shadow tables in chunks.
        _emit("teardown")
        _drive("teardown", self._fts_teardown_trash_step)
        with self._read_ctx() as conn:
            still_pending = _meta_row(conn, "fts_rebuild_high_water") is not None
            still_trash = self._has_fts_trash(conn)
            empty_index = self._fts_external_index_empty_with_messages(conn)
        if still_pending or still_trash or empty_index:
            reason = "backfill_incomplete" if still_pending or empty_index else "teardown_incomplete"
            logger.warning("FTS storage optimization did not settle (%s): pending=%s trash=%s empty_index=%s",
                           reason, still_pending, still_trash, empty_index)
            return {"ok": False, "reason": reason, "vacuumed": None}

        vacuum_ok = None
        if vacuum:
            _emit("vacuum")
            vacuum_ok = self._optimize_vacuum()
        refusal = self._execute_write(self._optimize_settle)
        if refusal is not None:
            # A concurrent process changed state since the pre-vacuum check; a re-run can still settle.
            logger.warning("FTS storage optimization settle refused (%s)", refusal)
            return {"ok": False, "reason": refusal, "vacuumed": vacuum_ok}
        _emit("done")
        logger.info("FTS storage optimization complete (layout v%d).", FTS_STORAGE_VERSION)
        return {"ok": True, "vacuumed": vacuum_ok}

    # ── Read views ─────────────────────────────────────────────────────────

    def get_anchored_view(
        self, session_id: str, around_message_id: int, window: int = 5, bookend: int = 3,
        keep_roles: Optional[Tuple[str, ...]] = ("user", "assistant")) -> Dict[str, Any]:
        """Anchored window (``get_messages_around``) plus session bookends, so one call yields the
        goal and the resolution of a long session. ``window`` is filtered to ``keep_roles``
        (None disables) EXCEPT the anchor; ``bookend_start`` / ``bookend_end`` are the
        first/last ``bookend`` non-empty-content messages with ids strictly outside the
        window (empty when it overlaps the head/tail). Empty result when the anchor isn't
        in the session."""
        bookend = max(bookend, 0)
        primitive = self.get_messages_around(session_id, around_message_id, window=window)
        window_rows = primitive["window"]
        if not window_rows:
            return {"window": [], "messages_before": 0, "messages_after": 0, "bookend_start": [], "bookend_end": []}

        filtered_window = window_rows
        if keep_roles is not None:
            keep_set = set(keep_roles)
            filtered_window = [m for m in window_rows if m.get("id") == around_message_id or m.get("role") in keep_set]
        bookend_start_rows: List[Any] = []
        bookend_end_rows: List[Any] = []
        if bookend > 0:
            role_clause = "" if keep_roles is None else f" AND role IN ({','.join('?' for _ in keep_roles)})"
            role_params = [] if keep_roles is None else list(keep_roles)
            with self._read_ctx() as conn:
                def _bookend(op: str, boundary_id: int, order: str):
                    return conn.execute(
                        f"SELECT * FROM messages "
                        f"WHERE session_id = ? AND id {op} ?{role_clause} "
                        f"AND length(content) > 0 "
                        f"ORDER BY id {order} LIMIT ?",
                        (session_id, boundary_id, *role_params, bookend),
                    ).fetchall()
                bookend_start_rows = _bookend("<", window_rows[0]["id"], "ASC")
                # End rows come back DESC for the LIMIT cap; flip to ASC.
                bookend_end_rows = list(reversed(_bookend(">", window_rows[-1]["id"], "DESC")))

        def _hydrate(row) -> Dict[str, Any]:
            return self._row_to_message_dict(row, warn_context="get_anchored_view", summary_flag=False)
        return {
            "window": filtered_window, "messages_before": primitive["messages_before"],
            "messages_after": primitive["messages_after"],
            "bookend_start": [_hydrate(r) for r in bookend_start_rows],
            "bookend_end": [_hydrate(r) for r in bookend_end_rows],
        }

    def list_recent_user_messages(
        self, session_id: str, limit: int = 20, include_inactive: bool = False) -> List[Dict[str, Any]]:
        """The *limit* most-recent real user turns, newest first, as ``{id, timestamp, preview}``
        (80 chars, whitespace collapsed); used by /rewind and ``/undo [N]``. Bookkeeping rows
        (``display_kind`` set) are excluded. Legacy compaction handoffs are role='user' rows
        with NO display_kind — invisible to SQL — so fetch with headroom and drop them in the
        decode loop; otherwise ``/undo N`` pairs an in-memory count that excludes handoffs
        with a DB pick that includes them."""
        active_clause = "" if include_inactive else " AND active = 1"
        display_clause = " AND (display_kind IS NULL OR display_kind = '')"
        with self._read_ctx() as conn:
            rows = conn.execute(
                "SELECT id, timestamp, content FROM messages WHERE session_id = ? AND role = 'user'"
                f"{active_clause}{display_clause} "
                "ORDER BY id DESC LIMIT ?",
                (session_id, int(limit) * 2 + 5),
            ).fetchall()
        from agent.context_compressor import ContextCompressor
        result: List[Dict[str, Any]] = []
        for row in rows:
            if len(result) >= int(limit):
                break
            decoded = self._decode_content(row["content"])
            if ContextCompressor._is_context_summary_content(decoded):
                continue  # compaction handoff — never a user-originated turn
            if isinstance(decoded, str):  # a /skill turn embeds the whole skill body; show what was typed
                preview = describe_skill_invocation(decoded) or decoded
            else:
                preview = _flatten_text(decoded)
            preview = " ".join(preview.split())
            if len(preview) > 80:
                preview = preview[:77] + "..."
            result.append({"id": row["id"], "timestamp": row["timestamp"], "preview": preview})
        return result

    # ── Query analysis ─────────────────────────────────────────────────────

    @staticmethod
    def _sanitize_fts5_query(query: str) -> str:
        """Sanitize user input for FTS5 MATCH (raw special characters raise): preserve paired
        quoted phrases, strip unmatched special characters, and quote hyphenated/dotted
        terms so FTS5 matches them as phrases (``chat-send``, ``P2.2``, ``my-app.config.ts``)."""
        # Cap before any regex processing so adversarial input stays bounded.
        query = query[:MAX_FTS5_QUERY_CHARS]

        # 1. Protect balanced quoted phrases via numbered placeholders (``"[^"]*"`` pairs
        # left-to-right without backtracking); a leftover unmatched quote becomes whitespace.
        _quoted_parts: list = []

        def _hold(m: "re.Match[str]") -> str:
            _quoted_parts.append(m.group(0))
            return f"\x00Q{len(_quoted_parts) - 1}\x00"

        sanitized = _QUOTED_PHRASE_RE.sub(_hold, query).replace('"', " ")

        # 2. Strip FTS5-special characters (an unquoted ``TODO: fix`` parses as
        # ``column:term``). ``%`` is only spared for the CJK LIKE fallback.
        sanitized = _FTS5_SPECIAL_RE.sub(" ", sanitized)
        if "%" in sanitized and not SessionSearchMixin._contains_cjk(sanitized):
            sanitized = sanitized.replace("%", " ")
        # 3. Collapse repeated * and drop leading * (prefix needs a char).
        sanitized = re.sub(r"\*+", "*", sanitized)
        sanitized = re.sub(r"(^|\s)\*", r"\1", sanitized)
        # 4. Drop dangling boolean operators at start/end (syntax errors).
        sanitized = re.sub(r"(?i)^(AND|OR|NOT)\b\s*", "", sanitized.strip())
        sanitized = re.sub(r"(?i)\s+(AND|OR|NOT)\s*$", "", sanitized.strip())
        # 5. Quote dotted/hyphenated/underscored terms in ONE pass (sequential passes
        # double-quote ``my-app.config``).
        sanitized = re.sub(r"\b(\w+(?:[._-]\w+)+)\b", r'"\1"', sanitized)
        # 6. Restore preserved quoted phrases.
        for i, quoted in enumerate(_quoted_parts):
            sanitized = sanitized.replace(f"\x00Q{i}\x00", quoted)
        return sanitized.strip()

    @staticmethod
    def _contains_cjk(text: str) -> bool:
        return any(_is_cjk(ord(ch)) for ch in text)

    @staticmethod
    def _count_cjk(text: str) -> int:
        return sum(1 for ch in text if _is_cjk(ord(ch)))

    @staticmethod
    def _has_lone_cjk_run(query: str) -> bool:
        """True when any maximal CJK run is a single char: the cjk-bigram index stores unigrams
        only for isolated chars, so such a term can't match inside longer runs — keep LIKE."""
        run = 0
        for ch in query:
            if _is_cjk(ord(ch)):
                run += 1
            else:
                if run == 1:
                    return True
                run = 0
        return run == 1

    @staticmethod
    def _trigram_eligible_tokens(query: str) -> bool:
        """True when every non-operator token is >=3 chars: a shorter token produces no
        trigrams, and with FTS5's implicit AND one such token empties the whole MATCH."""
        tokens = _non_operator_tokens(query.strip('"').strip())
        return bool(tokens) and all(len(t) >= 3 for t in tokens)

    @classmethod
    def _has_short_cjk_token(cls, raw_query: str) -> bool:
        """True when any non-operator CJK token has fewer than 3 CJK chars — the trigram
        tokenizer needs >=3 per token, so such a query must take the LIKE route."""
        return any(cls._count_cjk(t) < 3 for t in _non_operator_tokens(raw_query) if cls._contains_cjk(t))

    def _trigram_route_ok(self, raw_query: str) -> bool:
        """Per-token CJK length gate for the trigram index: ``广西 OR 桂林 OR 漓江`` has 6
        CJK chars total but 2 per token, so trigram returns 0."""
        return (self._count_cjk(raw_query) >= 3 and not self._has_short_cjk_token(raw_query)
                and self._trigram_available)

    def _describe_search_path(self, query: str) -> str:
        """Best-effort name of the routing path a query takes (log-only)."""
        try:
            if self._fts_stale:
                return "like_scan_fts_stale"
            sanitized = self._sanitize_fts5_query(query or "")
            if not sanitized:
                return "empty"
            if not self._contains_cjk(sanitized):
                return "fts5"
            raw = sanitized.strip('"').strip()
            if self._fts_cjk_available and not self._has_lone_cjk_run(raw):
                return "fts_cjk"
            return "trigram" if self._trigram_route_ok(raw) else "like_scan"
        except Exception:
            return "unknown"

    # ── Query builders / runners ───────────────────────────────────────────

    @staticmethod
    def _fts_match_sql(table: str, match_query: str, order_by_sql: str, *, limit: int, offset: int,
                       **filters) -> Tuple[str, list]:
        """MATCH query + params against one FTS5 index joined to messages/sessions."""
        where = [f"{table} MATCH ?"]
        params: list = [match_query]
        _search_filter_clauses(where, params, **filters)
        params.extend([limit, offset])
        sql = _search_select_sql(
            f"snippet({table}, -1, '>>>', '<<<', '...', 40) AS snippet",
            f"{table}\n            JOIN messages m ON m.id = {table}.rowid", where, order_by_sql, "LIMIT ? OFFSET ?",
        )
        return sql, params

    def _match_rows(self, table: str, match_query: str, order_by_sql: str, *, fail_open: Optional[str] = None,
                    operational_debug: Optional[str] = None, **kwargs) -> Optional[List[Dict[str, Any]]]:
        """Run one MATCH against *table*; ``None`` when the query cannot execute (tokenizer /
        syntax) so the caller falls back. *fail_open* names the index for the
        substring-capable routes: a corruption-class ``DatabaseError`` there detaches the
        derived indexes (``_enter_fts_fail_open``) and answers from canonical rows — a live
        search never runs the unbounded rebuild. Other ``DatabaseError``s propagate."""
        sql, params = self._fts_match_sql(table, match_query, order_by_sql, **kwargs)
        try:
            return [dict(row) for row in self._read_all(sql, params)]
        except sqlite3.OperationalError as exc:
            # A missing tokenizer is permanent: log it once instead of silently retrying on every CJK query.
            if self._is_trigram_unavailable_error(exc):
                self._warn_trigram_unavailable(exc)
            if operational_debug:
                logger.debug(operational_debug, exc_info=True)
            return None
        except sqlite3.DatabaseError as exc:
            if fail_open is None or not self._enter_fts_fail_open(exc):
                raise
            logger.warning(
                "%s FTS search hit a corruption error (%s); detached FTS and falling back to canonical LIKE.",
                fail_open, exc)
            return None

    def _like_rows(self, where: List[str], params: list, *, order_by: str, limit_sql: str) -> List[Dict[str, Any]]:
        """Canonical-table LIKE scan; ``params[0]`` is the snippet anchor term."""
        sql = _search_select_sql(_LIKE_SNIPPET_SQL, "messages m", where, order_by, limit_sql)
        return [dict(row) for row in self._read_all(sql, params)]

    @staticmethod
    def _compile_like_boolean_query(query: str) -> Tuple[str, List[Any], Optional[str]]:
        """Compile the supported FTS boolean subset into LIKE predicates: terms within an OR
        group are ANDed (FTS5's implicit conjunction) and ``NOT`` negates the next term."""
        groups: List[List[Tuple[str, bool]]] = [[]]
        negate_next = False
        for raw_token in _LIKE_TOKEN_RE.findall(query):
            operator = raw_token.upper()
            if operator == "OR":
                if groups[-1]:
                    groups.append([])
                negate_next = False
                continue
            if operator in {"AND", "NEAR"}:
                continue
            if operator == "NOT":
                negate_next = True
                continue
            term = raw_token.strip('"').strip("*").strip()
            if term:
                groups[-1].append((term, negate_next))
                negate_next = False

        compiled_groups: List[str] = []
        params: List[Any] = []
        snippet_term: Optional[str] = None
        for group in groups:
            if not group or not any(not negated for _, negated in group):
                continue
            clauses: List[str] = []
            for term, negated in group:
                clauses.append(f"NOT {_LIKE_COALESCED_COLUMN_SQL}" if negated else _LIKE_COALESCED_COLUMN_SQL)
                params.extend(_like_params(term))
                if snippet_term is None and not negated:
                    snippet_term = term
            compiled_groups.append(f"({' AND '.join(clauses)})")
        return " OR ".join(compiled_groups), params, snippet_term

    def _search_messages_like_fallback(
        self, query: str, *, limit: int, offset: int, sort: Optional[str], **filters) -> List[Dict[str, Any]]:
        """Search canonical messages while derived FTS state is stale."""
        predicate, params, snippet_term = self._compile_like_boolean_query(query)
        if not predicate or snippet_term is None:
            return []
        where = [f"({predicate})"]
        _search_filter_clauses(where, params, **filters)
        order = "ASC" if isinstance(sort, str) and sort.strip().lower() == "oldest" else "DESC"
        return self._like_rows(where, [snippet_term, *params, limit, offset],
                               order_by=f"ORDER BY m.timestamp {order}, m.id {order}", limit_sql="LIMIT ? OFFSET ?")

    def _refresh_fts_stale_state(self) -> None:
        """Observe fail-open initiated by another process sharing state.db."""
        if self._fts_stale or not self._fts_enabled:
            return
        try:
            stale = self._read_one("SELECT 1 FROM state_meta WHERE key = ? LIMIT 1", (FTS_STALE_KEY,))
        except sqlite3.Error:
            return
        if stale is not None:
            self._fts_stale = True
            self._fts_enabled = self._trigram_available = self._fts_cjk_available = False

    def _finalize_search_matches(
        self, matches: List[Dict[str, Any]], result_fields: Optional[Collection[str]] = None) -> List[Dict[str, Any]]:
        """Attach neighboring messages (1 before + after, only when the projection consumes
        ``context``) and trim full content. Each context query takes its own read txn."""
        if result_fields is None or "context" in result_fields:
            for match in matches:
                try:
                    with self._read_ctx() as conn:
                        rows = conn.execute(_CONTEXT_WINDOW_SQL, (match["id"], match["id"])).fetchall()
                        match["context"] = [
                            {"role": r["role"], "content": _flatten_text(self._decode_content(r["content"]))[:200]}
                            for r in rows]
                except Exception:
                    match["context"] = []
        # No route selects full content; the pop guards any future one that does.
        for match in matches:
            match.pop("content", None)
        if result_fields is not None:
            matches = [{field: match[field] for field in result_fields if field in match} for match in matches]
        return matches

    # ── search_messages ────────────────────────────────────────────────────

    def search_messages(
        self, query: str, source_filter: List[str] = None, exclude_sources: List[str] = None,
        role_filter: List[str] = None, limit: int = 20, offset: int = 0, sort: str = None,
        include_inactive: bool = False, fields: Optional[Collection[str]] = None,
    ) -> List[Dict[str, Any]]:
        """:meth:`_search_messages_impl` plus one log line per slow search with the routing
        path taken. Threshold HERMES_SEARCH_SLOW_MS (default 1000; 0 logs every call)."""
        started = time.time()
        rows = None
        try:
            rows = self._search_messages_impl(
                query, source_filter=source_filter, exclude_sources=exclude_sources, role_filter=role_filter,
                limit=limit, offset=offset, sort=sort, include_inactive=include_inactive, fields=fields)
            return rows
        finally:
            elapsed_ms = (time.time() - started) * 1000.0
            if elapsed_ms >= env_float("HERMES_SEARCH_SLOW_MS", 1000.0):
                logger.info("slow session search: path=%s elapsed=%.0fms rows=%s query=%r",
                            self._describe_search_path(query), elapsed_ms, len(rows) if rows is not None else "err",
                            query[: 200])

    def _search_messages_impl(
        self, query: str, source_filter: List[str] = None, exclude_sources: List[str] = None,
        role_filter: List[str] = None, limit: int = 20, offset: int = 0, sort: str = None,
        include_inactive: bool = False, fields: Optional[Collection[str]] = None,
    ) -> List[Dict[str, Any]]:
        """FTS5 search across session messages (keywords, ``"phrases"``, AND/OR/NOT, ``prefix*``).
        Returns snippet + session metadata + 1-message context per hit; ``fields`` selects a
        projection. ``sort``: None = BM25 rank; "newest"/"oldest" = timestamp then rank (the
        CJK LIKE fallback ignores it). Rewound rows (``active=0, compacted=0``) are excluded
        by default; compaction-archived rows ARE included; ``include_inactive`` = every row."""
        result_fields = self._search_message_fields(fields)
        if not query or not query.strip():
            return []
        query = self._sanitize_fts5_query(query)
        if not query:
            return []
        filters = dict(include_inactive=include_inactive, source_filter=source_filter,
                       exclude_sources=exclude_sources, role_filter=role_filter)
        # New oversized tool results index only a bounded prefix; an explicit tool-role search is the
        # opt-in full-body path and scans canonical rows via LIKE.
        if role_filter and "tool" in role_filter:
            matches = self._search_messages_like_fallback(query, limit=limit, offset=offset, sort=sort, **filters)
            return self._finalize_search_matches(matches, result_fields=result_fields)
        self._refresh_fts_stale_state()
        if self._fts_stale:
            matches = self._search_messages_like_fallback(query, limit=limit, offset=offset, sort=sort, **filters)
            return self._finalize_search_matches(matches, result_fields=result_fields)
        if not self._fts_enabled:
            return []

        order_by_sql = _FTS_ORDER_BY.get(sort.strip().lower() if isinstance(sort, str) else None, "ORDER BY rank")
        route = dict(order_by_sql=order_by_sql, limit=limit, offset=offset, **filters)
        # Tool rows and FTS_TRIGRAM_EXCLUDED_SOURCES sessions are excluded from the trigram/cjk
        # indexes (see FTS_TRIGRAM_SQL); an explicit filter for them must scan the base table.
        wants_unindexed_rows = (bool(role_filter) and "tool" in role_filter) or (
            bool(source_filter) and any(src in FTS_TRIGRAM_EXCLUDED_SOURCES for src in source_filter))
        is_cjk = self._contains_cjk(query)
        if is_cjk:
            matches = self._search_cjk(query, wants_unindexed_rows, route)
        else:
            sql, params = self._fts_match_sql("messages_fts", query, **route)
            try:
                matches = [dict(row) for row in self._read_all(sql, params)]
            except sqlite3.OperationalError:
                return []  # FTS5 syntax error despite sanitization
            except sqlite3.DatabaseError as exc:
                # Corruption parent class: detach the derived indexes and answer from
                # canonical rows; repair paths own the rebuild.
                # A corrupt FTS index raises the malformed / "fts5: corrupt structure record" class on the
                # MATCH read, the same class the write path handles (#66296). OperationalError (query
                # syntax) is a subclass caught above; this arm is the corruption parent. The existing
                # stale-open/repair paths retain rebuild ownership.
                if not self._enter_fts_fail_open(exc):
                    raise
                matches = self._search_messages_like_fallback(query, limit=limit, offset=offset, sort=sort, **filters)

        # Deferred-rebuild supplement: while the backfill is pending the FTS indexes miss
        # the (progress, high_water] gap; top up with a bounded LIKE scan so old messages
        # never vanish mid-rebuild. Cost decays to zero as the backfill advances.
        if self.fts_rebuild_status() is not None and len(matches) < limit:
            try:
                gap_matches = self._search_unindexed_gap(query, limit - len(matches), **filters)
                seen_ids = {m["id"] for m in matches}
                matches.extend(m for m in gap_matches if m["id"] not in seen_ids)
            except sqlite3.OperationalError as exc:
                logger.debug("Unindexed-gap supplement skipped: %s", exc)

        # unicode61 puts no boundary between Latin and adjacent CJK ("修改youer服务端" is
        # one token, so MATCH "youer" misses). On a zero-result Latin miss retry the
        # substring-capable indexes: cjk first (exact ranked match), then trigram (>=3-char
        # tokens). Gated on a miss so hits keep their ranking ("cat" may then match
        # "concatenate"). Skipped for role='tool' (both indexes exclude tool rows).
        if not matches and not is_cjk and not (bool(role_filter) and "tool" in role_filter):
            fb_query = _quote_fts_tokens(query.strip('"').strip())
            # ── CJK-bigram route (messages_fts_cjk, cjk_unicode61) ────── When the bigram index is
            # available it serves EVERY CJK query shape the legacy code split between trigram (>=3
            # chars/token) and LIKE full scans (1-2 char tokens) — the whole point of the index (PR #65544).
            # Exceptions stay on the legacy routes: - role_filter=['tool'] queries (tool rows aren't in the
            # cjk index, same exclusion as trigram), - queries containing a LONE 1-char CJK run: the index
            # stores bigrams for runs >=2, so a single-char term can only match isolated chars — LIKE
            # substring semantics are broader.
            if self._fts_cjk_available:
                matches = self._match_rows("messages_fts_cjk", fb_query, **route) or matches
            if not matches and self._trigram_available and self._trigram_eligible_tokens(query):
                matches = self._match_rows("messages_fts_trigram", fb_query, **route) or matches
        return self._finalize_search_matches(matches, result_fields=result_fields)

    def _search_cjk(self, query: str, wants_unindexed_rows: bool, route: Dict[str, Any]) -> List[Dict[str, Any]]:
        """CJK routing: the unicode61 table splits CJK into single characters (false positives,
        missed phrases). cjk-bigram serves every shape except queries wanting rows the
        substring indexes exclude (role='tool', cron/subagent sources) and LONE
        1-char CJK runs (bigrams only exist for runs >=2 — LIKE is broader); then trigram
        (>=3 CJK chars per token); then a LIKE substring scan with one clause per
        non-operator token so "广西 OR 桂林 OR 漓江" matches each term."""
        raw_query = query.strip('"').strip()
        match_query = _quote_fts_tokens(raw_query)
        if self._fts_cjk_available and not wants_unindexed_rows and not self._has_lone_cjk_run(raw_query):
            matches = self._match_rows(
                "messages_fts_cjk", match_query, fail_open="CJK-bigram",
                operational_debug="messages_fts_cjk query failed; falling back to trigram/LIKE", **route)
            if matches is not None:
                return matches
        if self._trigram_route_ok(raw_query) and not wants_unindexed_rows:
            matches = self._match_rows("messages_fts_trigram", match_query, fail_open="Trigram", **route)
            if matches is not None:
                return matches
        non_op_tokens = _non_operator_tokens(raw_query) or [raw_query]
        like_params: list = [p for tok in non_op_tokens for p in _like_params(tok)]
        like_where = [f"({' OR '.join([_LIKE_ANY_COLUMN_SQL] * len(non_op_tokens))})"]
        filters = {k: route[k] for k in ("include_inactive", "source_filter", "exclude_sources", "role_filter")}
        _search_filter_clauses(like_where, like_params, **filters)
        # instr() for the snippet uses the first search token.
        return self._like_rows(like_where, [non_op_tokens[0], *like_params, route["limit"], route["offset"]],
                               order_by="ORDER BY m.timestamp DESC", limit_sql="LIMIT ? OFFSET ?")

    def _search_unindexed_gap(self, fts_query: str, limit: int, **filters) -> List[Dict[str, Any]]:
        """LIKE-scan ids in (fts_rebuild_progress, fts_rebuild_high_water] — rows the deferred
        rebuild hasn't indexed yet. The FTS query degrades to AND-joined substring terms
        (quoted phrases kept whole): recall-over-precision mid-rebuild."""
        status = self.fts_rebuild_status()
        if status is None or limit <= 0:
            return []
        terms = [tok for tok in (t.strip('"').strip("*").strip() for t in _LIKE_TOKEN_RE.findall(fts_query))
                 if tok and tok.upper() not in _LIKE_SKIP_TOKENS]
        if not terms:
            return []
        where = ["m.id > ? AND m.id <= ?", *([_LIKE_ANY_COLUMN_SQL] * len(terms))]
        params: list = [status["indexed"], status["total"], *(p for term in terms for p in _like_params(term))]
        _search_filter_clauses(where, params, **filters)
        return self._like_rows(where, [terms[0], *params, limit], order_by="ORDER BY m.timestamp DESC",
                               limit_sql="LIMIT ?")

    def search_sessions_by_id(
        self, query: str, limit: int = 20, include_archived: bool = True, source: str = None,
        sources: List[str] = None, exclude_sources: List[str] = None) -> List[Dict[str, Any]]:
        """Search surfaced sessions by exact/prefix/substring session id. Also matches
        ``_lineage_root_id`` so an old compression root id resolves to the live continuation."""
        needle = (query or "").strip().lower()
        if not needle or limit <= 0:
            return []
        # list_sessions_rich pushes the id LIKE filter (own id + forward compression
        # chain) into SQL; over-fetch so the in-Python ranking has candidates.
        candidates = self.list_sessions_rich(
            source=source, sources=sources, exclude_sources=exclude_sources, limit=max(limit * 4, limit),
            offset=0, include_archived=include_archived, order_by_last_active=True, id_query=needle)

        def score(row: Dict[str, Any]) -> int:
            normalized = [v.lower() for v in (str(row.get("id") or ""), str(row.get("_lineage_root_id") or "")) if v]
            if any(value == needle for value in normalized):
                return 0
            return 1 if any(value.startswith(needle) for value in normalized) else 2
        ranked = sorted(enumerate(candidates), key=lambda item: (score(item[1]), item[0]))
        return [row for _, row in ranked[:limit]]

    # ── FTS maintenance commands ───────────────────────────────────────────

    def _fts_table_exists(self, name: str) -> bool:
        """True if an FTS5 virtual table is queryable ("no such table" and "vtable
        constructor failed" — missing tokenizer / mid-teardown — both count as not)."""
        try:
            self._conn.execute(f"SELECT 1 FROM {name} LIMIT 0")
            return True
        except sqlite3.DatabaseError:
            return False

    def _present_fts_tables(self) -> List[str]:
        """Queryable FTS tables (caller holds ``self._lock``)."""
        return [tbl for tbl in self._FTS_TABLES if self._fts_table_exists(tbl)]

    def optimize_fts(self) -> int:
        """Merge fragmented FTS5 segments into one per index (``'optimize'``). Pure
        maintenance: changes neither results nor ``snippet()`` output, only layout and
        speed; VACUUM then returns the freed pages. Returns the number optimized."""
        optimized = 0
        with self._lock:
            for tbl in self._present_fts_tables():
                try:
                    self._conn.execute(f"INSERT INTO {tbl}({tbl}) VALUES('optimize')")
                    optimized += 1
                except sqlite3.OperationalError as exc:
                    logger.warning("FTS optimize failed for %s: %s", tbl, exc)
        return optimized

    def rebuild_fts(self) -> int:
        """Rebuild FTS5 indexes from ``messages`` (``'rebuild'``) — the recovery for a corrupt index
        that rejects writes while reads succeed. Two processes rebuilding one state.db
        concurrently corrupted production DBs, so this admits through
        ``fts_rebuild_admission`` and FAILS CLOSED, returning 0 on deferral (callers treat 0
        as "no progress" and use the stale-FTS breadcrumb path). Returns indexes rebuilt.

        Uses the FTS5 ``'rebuild'`` command, which rewrites the internal b-tree segments from the content
        rows. Unlike ``optimize_fts`` (which merges existing segments), ``rebuild`` discards and recreates
        the index data entirely. See #50502.
        A full structural rebuild must never run concurrently in two processes sharing one state.db — that
        interleaving has structurally corrupted the database in production (PR #93200) — so this admits
        through the cross-process ``fts_rebuild_admission`` authority and FAILS CLOSED: if another process
        holds the rebuild lock beyond the bounded wait, this call defers (returns 0) rather than racing it.
        Callers already treat 0 as "rebuild made no progress" and fall back to the stale-FTS breadcrumb
        path, which retries in-process from the gateway housekeeping tick (``retry_deferred_fts_recovery``)
        and at next startup.
        """
        rebuilt = 0
        with fts_rebuild_admission(self.db_path) as admitted:
            if not admitted:
                logger.warning(
                    "Deferred in-place FTS rebuild: another process holds the rebuild authority for this state.db.")
                return 0
            with self._lock:
                high_water = self._conn.execute("SELECT COALESCE(MAX(id), 0) FROM messages").fetchone()[0]
                self._conn.execute(
                    "INSERT INTO state_meta (key, value) VALUES (?, ?) ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                    (FTS_TOOL_FULL_CONTENT_HIGH_WATER_KEY, str(high_water)),
                )
                for tbl in self._present_fts_tables():
                    try:
                        self._conn.execute(f"INSERT INTO {tbl}({tbl}) VALUES('rebuild')")
                        self._conn.commit()
                        rebuilt += 1
                    except sqlite3.OperationalError as exc:
                        self._conn.rollback()
                        logger.warning("FTS rebuild failed for %s: %s", tbl, exc)
        return rebuilt

    def _merge_fts_incrementally(self, *, max_pages: int, max_commands: Optional[int] = None) -> int:
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
            max_commands = self._FTS_MERGE_COMMANDS_PER_PASS
        _positive_int("max_commands", max_commands)
        executed = 0
        with self._lock:
            for tbl in self._present_fts_tables():
                if not self._fts_usermerge_floor_applied:
                    self._conn.execute(f"INSERT INTO {tbl}({tbl}, rank) VALUES('usermerge', 2)")
                for _ in range(max_commands):
                    before = self._conn.total_changes
                    self._conn.execute(f"INSERT INTO {tbl}({tbl}, rank) VALUES('merge', ?)", (max_pages,))
                    executed += 1
                    if self._conn.total_changes - before < 2:
                        break
            self._fts_usermerge_floor_applied = True
        return executed


# ---- BEGIN PLUGIN-COMPAT (revert-scheduled; see COMPAT_MANIFEST.md) ----
# Names external plugins imported from this module before the Sep 2026 decomposition.
# Internal code MUST NOT use these (scripts/check_compat_pointers.py fails CI if it does).
# The whole block is removed by reverting the commit that added it.
import json  # noqa: F401,E402
import os  # noqa: F401,E402
# ---- END PLUGIN-COMPAT ----
