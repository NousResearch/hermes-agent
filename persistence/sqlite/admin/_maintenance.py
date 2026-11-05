"""MaintenanceAdmin operations of the SQLite store admin: VACUUM with its FTS merge, the page statistics, the
auto-maintenance lock and holder scan, the opt-in FTS storage rewrite, the FTS layout probe, and declarative schema
drift detection and repair."""

from __future__ import annotations

import contextlib
import contextvars
import logging
import sqlite3
import threading
import time
from dataclasses import dataclass, field
from functools import partial
from typing import Any, Callable, Optional

from hermes_state_holders import foreign_state_db_holders, in_process_state_db_holders
from hermes_state_repair import _release_auto_maintenance_lock, _try_acquire_auto_maintenance_lock

from .._backend import translated
from .._holder import StoreHolder
from .._schema import (
    _FTS_TRASH_PREFIX, _db_has_legacy_inline_fts, _db_needs_fts_storage_upgrade, _delete_meta, _drop_fts_triggers,
    _ensure_fts_schema, _fts_external_index_empty_with_messages, _has_fts_trash, _meta_row, _present_fts_tables,
)
from .._schema_ddl import (
    FTS_SQL, FTS_STORAGE_VERSION, FTS_TOOL_CONTENT_PREFIX_CHARS, FTS_TRIGRAM_SQL, SCHEMA_VERSION, fts_layout_current,
    fts_layout_drift,
)
from ..conversation._store.search import (
    _BASE_BOUNDARY_SWEEP_SQL, _CHUNK_INSERT_SQL, _HOLDER_SCAN_INTERVAL_S, _OPTIMIZE_CHUNK_ATTEMPTS,
    _TRIGRAM_BOUNDARY_SWEEP_SQL, _TRIGRAM_CHUNK_INSERT_SQL, _VACUUM_TICK_S, _backfill_progress, _BackfillProgress,
    fts_rebuild_status,
)
from . import _reconcile
from ._base import AdminFile

# caplog tests pin the "hermes_state" logger name.
logger = logging.getLogger("hermes_state")


def _optimize_fts(holder: StoreHolder) -> int:
    """Merge fragmented FTS5 segments into one per index (``'optimize'``). Pure
    maintenance: changes neither results nor ``snippet()`` output, only layout and
    speed; VACUUM then returns the freed pages. Returns the number optimized. A quarantined
    handle never issues ``'optimize'``: it rewrites index segments in place and would compound
    structural damage (or a split WAL generation) instead of leaving it diagnosable."""
    holder._raise_if_db_corrupt()
    optimized = 0
    with holder._lock:
        holder._raise_if_db_replaced()
        if holder._conn is None:
            holder._reopen_after_close_locked(context="write")
        for tbl in _present_fts_tables(holder):
            try:
                holder._conn.execute(f"INSERT INTO {tbl}({tbl}) VALUES('optimize')")
                optimized += 1
            except sqlite3.DatabaseError as exc:  # SQLITE_CORRUPT is not an OperationalError
                logger.warning("FTS optimize failed for %s: %s", tbl, exc)
    return optimized


def _try_checkpoint(holder: StoreHolder, mode: str, fail_msg: str) -> None:
    try:
        holder._conn.execute(f"PRAGMA wal_checkpoint({mode})")
    except Exception as exc:  # health: allow BLE001 -- a failed checkpoint is logged at debug
        logger.debug(fail_msg, exc)


@dataclass
class _TeardownProgress:
    """Which demoted shadow table optimize-storage is draining and how far (``progress_cb``
    payload). Owned and kept current by :meth:`_FtsStorageOptimizer._fts_teardown_trash_step`.
    ``counted=False`` (no progress callback) skips the per-table ``COUNT(*)``."""

    table: str = ""
    table_index: int = 0
    table_count: int = 0
    total: int = 0
    drained: int = 0
    counted: bool = True

    def info(self) -> dict[str, Any]:
        done, total = self.drained, self.total
        return {"phase": "teardown", "percent": min(100, 100 * done // total) if total else 100,
                "indexed": done, "total": total, "table": self.table,
                "table_index": self.table_index, "table_count": self.table_count}


@dataclass
class _VacuumProgress:
    """VACUUM is one statement: liveness is elapsed time against the size before and the estimated
    size after (live pages only), never a percent."""

    size_bytes: int
    estimated_bytes: int
    started: float = field(default_factory=time.monotonic)

    def info(self) -> dict[str, Any]:
        return {"phase": "vacuum", "percent": 0, "indexed": 0, "total": 0,
                "elapsed": time.monotonic() - self.started, "size_bytes": self.size_bytes,
                "estimated_bytes": self.estimated_bytes}


class _HolderWatch:
    """Re-runs the foreign-holder scan (the one the held-store refusal and ``--force`` use) every
    *interval* seconds on a side thread while optimize-storage rebuilds and tears down. The chunk
    loop only reads :attr:`held`, so a quiet store pays nothing per chunk, while a writer that opens
    state.db mid-run (Desktop launched, gateway restarted) moves the loop onto the paced cycle within
    one interval instead of waiting behind back-to-back chunks."""

    def __init__(self, scan: Callable[[], list], interval: float):
        self._scan, self._interval = scan, interval
        self._held, self._stop = threading.Event(), threading.Event()
        self._thread: Optional[threading.Thread] = None

    @property
    def held(self) -> bool:
        return self._held.is_set()

    def _refresh(self) -> None:
        (self._held.set if self._scan() else self._held.clear)()

    def _run(self) -> None:
        while not self._stop.wait(self._interval):
            self._refresh()

    def __enter__(self) -> _HolderWatch:
        self._refresh()  # a holder already there paces from the first chunk
        # copy_context().run: the watcher keeps the caller's profile scope.
        self._thread = threading.Thread(target=contextvars.copy_context().run, args=(self._run,),
                                        name="optimize-holder-watch", daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *_exc) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join()


class _FtsStorageOptimizer:
    """The opt-in FTS storage rewrite on one holder.

    One blocking rebuild held the write lock ~16 min on a 25 GB DB, so the backfill and teardown
    run in chunks: each its own bounded transaction (WAL growth stays small), resumable from its
    marker (claimed by CAS), and a progress tick. They run back to back with no pause while no other
    process holds state.db: the foreground `hermes sessions optimize-storage` refuses to start under
    a holder (held_store_refusal), and a fixed 0.2 s pause after each few-ms chunk was >95% of the
    wall time. Chunk size (in ROWS, never an id span: a pruned store has id gaps) adapts toward
    TARGET seconds, since rows cost ~1000x more in trigram_data than in docsize. A side thread
    re-runs the holder scan every _HOLDER_SCAN_INTERVAL_S (`--force` past a holder, or a writer
    that opens the store mid-run): while one is there the duty cycle comes back, chunks near
    PACED_TARGET and a pause of max(MIN_PAUSE, cost x DUTY_FACTOR) after each, so that holder's
    writes are not starved."""

    _FTS_REBUILD_CHUNK_ROWS = 500       # first chunk of every phase and trash table; later ones adapt
    _FTS_REBUILD_CHUNK_MIN_ROWS = 50
    _FTS_REBUILD_CHUNK_MAX_ROWS = 100_000
    _FTS_REBUILD_CHUNK_TARGET_S = 0.25
    _FTS_REBUILD_PACED_TARGET_S = 0.05
    _FTS_REBUILD_DUTY_FACTOR = 4.0      # paced: sleep >= 4x chunk cost (≤20% duty)
    _FTS_REBUILD_MIN_PAUSE = 0.2        # paced: seconds, floor between chunks

    def __init__(self, holder: StoreHolder) -> None:
        self._holder = holder
        self._fts_chunk_rows = self._FTS_REBUILD_CHUNK_ROWS  # rows per step; the drive loop adapts it

    def _next_chunk_rows(self, rows: int, cost_s: float, target_s: float) -> int:
        """Rows for the next chunk so it costs about *target_s*; at most doubles or halves per
        step so one noisy chunk (a cold page, a checkpoint) can't swing it."""
        scale = target_s / max(cost_s, 1e-4)
        rows = int(rows * min(2.0, max(0.5, scale)))
        return min(self._FTS_REBUILD_CHUNK_MAX_ROWS, max(self._FTS_REBUILD_CHUNK_MIN_ROWS, rows))

    # ── Deferred rebuild engine ─────────────────────

    def _fts_rebuild_finish(self) -> None:
        """Finalize the deferred rebuild: boundary sweep + clear markers. The sweep is cheap
        insurance against a write that slipped between high_water capture and trigger
        activation. The trigram half is gated on ``_trigram_available``: without the
        tokenizer/table an unconditional INSERT raises and aborts the whole rebuild."""
        def _do(conn):
            hw_row = _meta_row(conn, "fts_rebuild_high_water")
            if hw_row is not None:
                hw = int(hw_row[0])
                conn.execute(_BASE_BOUNDARY_SWEEP_SQL, (FTS_TOOL_CONTENT_PREFIX_CHARS, hw - 1000, hw + 1000))
                if self._holder._trigram_available:
                    conn.execute(_TRIGRAM_BOUNDARY_SWEEP_SQL, (hw - 1000, hw + 1000))
            _delete_meta(conn, "fts_rebuild_high_water", "fts_rebuild_progress")
        self._holder._execute_write(_do)
        logger.info("Deferred FTS rebuild complete — all messages indexed.")

    def fts_rebuild_step(self, progress: Optional[_BackfillProgress] = None) -> bool:
        """Backfill one chunk (``_fts_chunk_rows`` messages) of the deferred FTS rebuild; True while
        work remains. Chunks are claimed atomically inside the write transaction, so concurrent
        processes interleave instead of duplicating rows. A failed chunk raises (nothing committed;
        the caller decides whether to retry). *progress* gains the messages the chunk covered."""
        if not self._holder._fts_enabled:
            return False
        inserts = [_CHUNK_INSERT_SQL]
        if self._holder._trigram_available:
            inserts.append(_TRIGRAM_CHUNK_INSERT_SQL)
        high_water_raw = self._holder.get_meta("fts_rebuild_high_water")
        if high_water_raw is None:
            return False
        high_water = int(high_water_raw)
        chunk = self._fts_chunk_rows

        def _do(conn):
            # Re-reading progress inside the BEGIN IMMEDIATE held by _execute_write IS
            # the claim: two workers cannot read the same value.
            row = _meta_row(conn, "fts_rebuild_progress")
            if row is None:
                return False, 0  # finished (or cleared) by another process
            done = int(row[0])
            if done >= high_water:
                return False, 0
            # The window is the next `chunk` messages, not `chunk` ids: an id gap (pruned
            # sessions) neither shrinks a chunk nor lets the next one span a million dense ids.
            nth = conn.execute("SELECT id FROM messages WHERE id > ? AND id <= ? ORDER BY id LIMIT 1 OFFSET ?",
                               (done, high_water, chunk - 1)).fetchone()
            upper = int(nth[0]) if nth is not None else high_water
            covered = chunk if nth is not None else 0
            if nth is None and progress is not None:  # the last, short window (< chunk rows)
                covered = conn.execute("SELECT COUNT(*) FROM messages WHERE id > ? AND id <= ?",
                                       (done, upper)).fetchone()[0]
            for sql in inserts:
                conn.execute(sql, (done, upper))
            # Progress lands in the same transaction as its rows (crash-atomic).
            conn.execute("UPDATE state_meta SET value = ? WHERE key = ?", (str(upper), "fts_rebuild_progress"))
            return upper < high_water, covered

        more, covered = self._holder._execute_write(_do)
        if progress is not None:
            # Counted after the commit: _execute_write may replay _do on contention.
            progress.indexed += covered
        if more is False:
            status = fts_rebuild_status(self._holder)
            if high_water <= 0 or (status is not None and status["indexed"] >= status["total"]):
                self._fts_rebuild_finish()
            return False
        return bool(more)

    def _fts_teardown_trash_step(self, progress: Optional[_TeardownProgress] = None) -> bool:
        """Tear down one chunk (``_fts_chunk_rows`` rows) of a demoted v22 FTS shadow table (a PLAIN
        table now); True while work remains. A failed chunk raises (nothing committed).

        With *progress* (optimize-storage), the first step on each trash table deletes nothing: it
        points *progress* at the table and restarts the chunk size (rows cost ~1000x more in one
        shadow table than in another); later steps add the rows they deleted.

        Single-column INTEGER-key trash tables (the common shape: FTS shadow tables carry a rowid/integer
        PK) are drained with a high-water marker mirroring :meth:`fts_rebuild_step`: each chunk deletes only
        rows after the previously-drained key, so the per-chunk scan is bounded instead of re-scanning from
        the start of the table every chunk (O(n²) total on large trash tables). Compound-key tables
        keep the chunked ``LIMIT`` delete; they are small by construction.
        """
        limit = self._fts_chunk_rows
        with self._holder._read_ctx() as conn:
            trash = [r[0] for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table' AND name LIKE ? ESCAPE '\\'",
                (_FTS_TRASH_PREFIX.replace("_", "\\_") + "%",),
            ).fetchall()]
            if trash and progress is not None and progress.table != trash[0]:
                self._fts_teardown_start_table(conn, trash, progress)
                return True
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
                upper_row = conn.execute(
                    f"SELECT {key} FROM {tbl} WHERE {key} > ? ORDER BY {key} LIMIT 1 OFFSET ?",
                    (high_water, limit - 1),
                ).fetchone() or conn.execute(f"SELECT MAX({key}) FROM {tbl} WHERE {key} > ?", (high_water,)).fetchone()
                if upper_row is None or upper_row[0] is None:
                    return _drop(conn, marker_key), 0
                upper = upper_row[0]
                cur = conn.execute(f"DELETE FROM {tbl} WHERE {key} > ? AND {key} <= ?", (high_water, upper))
                if cur.rowcount > 0:
                    self._holder.set_meta(marker_key, str(upper), cursor=conn)
                return True, cur.rowcount
            # Compound-key or rowid trash table: legacy chunked delete. These shadow tables are small, so
            # the quadratic re-scan is not a concern (the high-water path covers the big single-key tables).
            cur = conn.execute(f"DELETE FROM {tbl} WHERE ({key}) IN (SELECT {key} FROM {tbl} LIMIT {int(limit)})")
            return (_drop(conn) if cur.rowcount == 0 else True), cur.rowcount

        def _drop(conn, marker_key: Optional[str] = None) -> bool:
            """Drained, so the DROP is cheap now. True: re-check for more trash."""
            conn.execute(f"DROP TABLE IF EXISTS {tbl}")
            if marker_key is not None:
                _delete_meta(conn, marker_key)
            logger.info("Old FTS shadow table %s torn down.", tbl)
            return True

        more, deleted = self._holder._execute_write(_do)
        if progress is not None:
            # Counted after the commit: _execute_write may replay _do on contention.
            progress.drained += deleted
        return bool(more)

    def _fts_teardown_start_table(self, conn, trash: list[str], progress: _TeardownProgress) -> None:
        """Point *progress* at ``trash[0]`` (its place in the run, the rows left past its resume
        marker) and restart the chunk size. One COUNT per table, on the pooled reader *conn*."""
        tbl = trash[0]
        # Only INTEGER-pk tables (rowid aliases) carry a resume marker; count what is left past it.
        marker = _meta_row(conn, f"fts_teardown_{tbl}_progress")
        where, params = ("WHERE rowid > ?", (int(marker[0]),)) if marker is not None else ("", ())
        table_count = progress.table_count or len(trash)
        # COUNT before publishing the table: a failed COUNT must leave progress on the previous
        # table so the retry runs this start step again instead of inheriting its totals.
        total = int(conn.execute(f"SELECT COUNT(*) FROM {tbl} {where}", params).fetchone()[0]) if progress.counted else 0
        progress.table_count, progress.total, progress.drained = table_count, total, 0
        progress.table_index = max(1, table_count - len(trash) + 1)
        self._fts_chunk_rows = self._FTS_REBUILD_CHUNK_ROWS
        progress.table = tbl

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
            self._holder.set_meta("fts_rebuild_progress", "0", cursor=conn)

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
            with contextlib.suppress(sqlite3.OperationalError):  # table absent: already an empty surface
                conn.execute(f"INSERT INTO {tbl}({tbl}) VALUES('delete-all')")

    def _seed_fts_rebuild_markers(self, conn, *, force: bool = False) -> int:
        """Write ``fts_rebuild_high_water`` / ``fts_rebuild_progress`` for a full backfill; returns
        the high-water id. Without ``force`` an existing high_water only gets a missing
        progress key repaired. Caller holds the write transaction."""
        existing_hw = _meta_row(conn, "fts_rebuild_high_water")
        if existing_hw is not None and not force:
            self._reseed_missing_progress(conn)
            return int(existing_hw[0])
        hw = conn.execute("SELECT COALESCE(MAX(id), 0) FROM messages").fetchone()[0]
        self._holder.set_meta("fts_rebuild_high_water", str(hw), cursor=conn)
        self._holder.set_meta("fts_rebuild_progress", "0", cursor=conn)
        return int(hw)

    def _repair_optimize_bookkeeping(self) -> None:
        """Heal interrupted demote/backfill bookkeeping before optimize runs: orphan high_water
        gets progress re-seeded; an empty external index with messages and no markers gets
        a full backfill claim. Never invents markers on a still-legacy inline DB: optimize
        would then skip demote and INSERT against the inline table forever."""
        holder = self._holder

        def _do(conn):
            if _meta_row(conn, "fts_rebuild_high_water") is not None:
                self._reseed_missing_progress(conn)
                return
            if _db_has_legacy_inline_fts(conn):
                return  # demote owns marker creation
            if _fts_external_index_empty_with_messages(holder, conn):
                _delete_meta(conn, "fts_storage_version")
                self._seed_fts_rebuild_markers(conn, force=True)
        holder._execute_write(_do)

    def fts_optimize_available(self) -> bool:
        """True when `optimize_fts_storage()` has work: legacy inline FTS or a v23 trigram still
        carrying ``tool_calls`` (``_db_needs_fts_storage_upgrade``), an interrupted optimize
        (markers/trash), or an empty external index without markers. False when FTS5 is unavailable."""
        holder = self._holder
        if not holder._fts_enabled or holder.read_only:
            return False
        with holder._read_ctx() as conn:
            return (
                _db_needs_fts_storage_upgrade(conn)
                or _meta_row(conn, "fts_rebuild_high_water") is not None  # interrupted optimize
                or _has_fts_trash(holder, conn)
                or _fts_external_index_empty_with_messages(holder, conn)
            )

    def _demote_legacy_fts_to_trash(self) -> int:
        """Demote upgrade-eligible FTS vtables and stage their shadow tables for chunked
        teardown; returns MAX(messages.id) as the rebuild high water. O(1) schema surgery:
        the heavy delete is deferred. Markers land in the same BEGIN IMMEDIATE, BEFORE
        the empty v23 schema is created (``executescript`` implicitly COMMITs), closing
        the crash window where trash + empty v23 tables exist with no backfill claim."""
        holder = self._holder

        def _stage(conn):
            _drop_fts_triggers(conn)
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

        hw = int(holder._execute_write(_stage))
        # Outside the write transaction (executescript commits); markers are durable.
        self._ensure_v23_fts_tables("failed to create v23 messages_fts during optimize-storage demote")
        return hw

    def _ensure_v23_fts_tables(self, failure_message: str) -> None:
        """Ensure the v23 base + trigram tables under the lock (IF NOT EXISTS, cheap); raise
        *failure_message* without the base table (the backfill loop would retry forever)."""
        holder = self._holder
        with holder._lock:
            base_ok = _ensure_fts_schema(holder, holder._conn, "messages_fts", FTS_SQL)
            trigram_ok = _ensure_fts_schema(holder, holder._conn, "messages_fts_trigram", FTS_TRIGRAM_SQL)
            holder._trigram_available = bool(trigram_ok)
            if not base_ok:
                raise sqlite3.OperationalError(failure_message)
            holder._conn.commit()

    def _optimize_drive(self, step: Callable[[], bool], info: Optional[Callable[[], dict[str, Any]]],
                        emit: Callable[[dict[str, Any]], None], watch: _HolderWatch) -> None:
        """Run *step* to completion, emitting ``info()`` (plus ``paced``) after every chunk. Back to
        back while no other process holds state.db (see the class docstring); while *watch* sees a
        holder, the duty cycle (short chunks + a pause) keeps its writes moving. Chunk size adapts
        to its measured cost; a step may restart it (teardown on each new table). A chunk that fails
        _OPTIMIZE_CHUNK_ATTEMPTS times in a row ends the run (each lock failure already waited out
        _execute_write's patience); a full disk or I/O error never heals by retrying."""
        self._fts_chunk_rows = self._FTS_REBUILD_CHUNK_ROWS
        failures = 0
        while True:
            t0 = time.monotonic()
            try:
                more = step()
            except sqlite3.OperationalError as exc:
                failures += 1
                if failures >= _OPTIMIZE_CHUNK_ATTEMPTS:
                    raise
                logger.warning("FTS optimize chunk failed (attempt %d, retrying): %s", failures, exc)
                time.sleep(0.25 * failures)
                continue
            failures = 0
            if not more:
                return
            cost, paced = time.monotonic() - t0, watch.held
            if info is not None:
                emit({**info(), "paced": paced})
            target = self._FTS_REBUILD_PACED_TARGET_S if paced else self._FTS_REBUILD_CHUNK_TARGET_S
            self._fts_chunk_rows = self._next_chunk_rows(self._fts_chunk_rows, cost, target)
            if paced:
                time.sleep(max(self._FTS_REBUILD_MIN_PAUSE, cost * self._FTS_REBUILD_DUTY_FACTOR))

    def _optimize_vacuum(self, tick: Optional[Callable[[dict[str, Any]], None]] = None) -> bool:
        """Phase 3: reclaim freed pages to the OS. False when VACUUM failed (usually no free disk
        for its temp copy; a later VACUUM reclaims). VACUUM is one statement that runs for minutes
        on a big store, so *tick* gets a :class:`_VacuumProgress` payload at once and every
        ``_VACUUM_TICK_S`` from a side thread, which is stopped and joined before this returns."""
        holder = self._holder
        ticker, stop = None, threading.Event()
        if tick is not None:
            with holder._read_ctx() as conn:
                page_size, pages, free = (conn.execute(f"PRAGMA {p}").fetchone()[0]
                                          for p in ("page_size", "page_count", "freelist_count"))
            state = _VacuumProgress(size_bytes=pages * page_size, estimated_bytes=(pages - free) * page_size)

            def _tick() -> None:
                tick(state.info())
                while not stop.wait(_VACUUM_TICK_S):
                    tick(state.info())

            # copy_context().run: the ticker keeps the caller's profile scope.
            ticker = threading.Thread(target=contextvars.copy_context().run, args=(_tick,),
                                      name="optimize-vacuum-ticker", daemon=True)
            ticker.start()
        try:
            with holder._lock:
                holder._conn.execute("VACUUM")
            vacuum_ok = True
        except sqlite3.OperationalError as exc:
            logger.warning("VACUUM after FTS optimize failed: %s", exc)
            vacuum_ok = False
        finally:
            stop.set()
            if ticker is not None:
                ticker.join()
        # Best-effort WAL fold-back, REFUSED (SQLITE_BUSY) while another connection holds a
        # read-mark (callers size via logical_size_bytes, not stat()). PASSIVE, never TRUNCATE:
        # a TRUNCATE reset from a transient CLI would race a live writer.
        try:
            with holder._lock:
                # Best-effort: fold the WAL back into the main file so the on-disk size settles now rather
                # than at close(). Callers must therefore NOT size the result by stat()ing the file; use
                # logical_size_bytes, which is truthful immediately regardless of readers.
                holder._conn.execute("PRAGMA wal_checkpoint(PASSIVE)")
        except Exception as exc:  # health: allow BLE001 -- the checkpoint after VACUUM is best-effort
            logger.debug("WAL checkpoint (PASSIVE) after optimize VACUUM failed: %s", exc)
        return vacuum_ok

    def _optimize_settle(self, conn) -> Optional[str]:
        """Phase 4 (inside the write transaction, so a concurrent writer cannot race a stamp past
        incomplete work): stamp the FTS layout (source of truth for "optimized"), clear the
        "available" flag, advance a lagging schema_version. Returns a refusal reason or None.
        Refuses while optimize work remains; an empty base index against non-empty messages
        also refuses (settling there meant permanent search-index loss)."""
        holder = self._holder
        if _meta_row(conn, "fts_rebuild_high_water") is not None:
            return "backfill_incomplete"
        if _has_fts_trash(holder, conn):
            return "teardown_incomplete"
        if _fts_external_index_empty_with_messages(holder, conn):
            return "backfill_incomplete"
        current = fts_layout_current(conn)
        if not current and fts_layout_drift(conn):
            return "layout_drift"
        # Not current with no older layout left means a definition this code does not recognise (a newer
        # release's): the work is done, but the stamp is not this release's to write.
        if current:
            holder.set_meta("fts_storage_version", str(FTS_STORAGE_VERSION), cursor=conn)
        _delete_meta(conn, "fts_optimize_available")
        conn.execute("UPDATE schema_version SET version = ? WHERE version < ?", (SCHEMA_VERSION, SCHEMA_VERSION))
        return None

    def optimize_fts_storage(
        self, *, progress_cb: Optional[Callable[[dict[str, Any]], None]] = None, vacuum: bool = True,
    ) -> dict[str, Any]:
        """Repair an older FTS layout into the current v23 shape, foreground and to completion:
        legacy-v22 inline -> external-content, or a v23 ``messages_fts_trigram`` that still stores
        ``tool_calls``. Re-running resumes; a chunk that keeps failing raises (nothing is lost).
        ``progress_cb`` receives {"phase", "percent", "indexed", "total", "paced"} after every chunk
        (``backfill`` counts messages; plus :meth:`_TeardownProgress.info` fields)
        and every second of the VACUUM (:meth:`_VacuumProgress.info`). Whenever another process holds
        state.db (re-scanned every ``_HOLDER_SCAN_INTERVAL_S``) chunks are paced so its writes get
        through. A missing trigram tokenizer is not fatal (CJK falls back to LIKE)."""
        holder = self._holder
        if not holder._fts_enabled:
            return {"ok": False, "reason": "fts5_unavailable"}
        if holder.read_only:
            return {"ok": False, "reason": "read_only"}

        # Heal bookkeeping BEFORE deciding whether to demote again.
        self._repair_optimize_bookkeeping()
        with holder._lock:
            needs_storage_upgrade = _db_needs_fts_storage_upgrade(holder._conn)
        pending = holder.get_meta("fts_rebuild_high_water") is not None
        if needs_storage_upgrade and not pending:
            self._demote_legacy_fts_to_trash()
        elif pending and not needs_storage_upgrade:
            # Resume mid-demote: the process may have died between the staged demote
            # commit and schema ensure.
            self._ensure_v23_fts_tables("failed to re-create v23 messages_fts on optimize-storage resume")

        emit = progress_cb or (lambda _info: None)
        teardown = _TeardownProgress(counted=progress_cb is not None)
        with _HolderWatch(holder._foreign_state_db_holders, _HOLDER_SCAN_INTERVAL_S) as watch:
            # Phase 1: base backfill.
            bf = _backfill_progress(holder, "backfill") if progress_cb else None
            if bf is not None:
                emit({**bf.info(), "paced": watch.held})
            step = partial(self.fts_rebuild_step, bf) if bf else self.fts_rebuild_step
            self._optimize_drive(step, bf.info if bf else None, emit, watch)
            # Phase 2: tear down the demoted legacy shadow tables in chunks.
            self._optimize_drive(partial(self._fts_teardown_trash_step, teardown), teardown.info, emit, watch)
        with holder._read_ctx() as conn:
            still_pending = _meta_row(conn, "fts_rebuild_high_water") is not None
            still_trash = _has_fts_trash(holder, conn)
            empty_index = _fts_external_index_empty_with_messages(holder, conn)
        if still_pending or still_trash or empty_index:
            reason = "backfill_incomplete" if still_pending or empty_index else "teardown_incomplete"
            logger.warning("FTS storage optimization did not settle (%s): pending=%s trash=%s empty_index=%s",
                           reason, still_pending, still_trash, empty_index)
            return {"ok": False, "reason": reason, "vacuumed": None}

        vacuum_ok, vacuum_skipped = None, False
        if vacuum and holder._foreign_state_db_holders():
            # VACUUM holds the write lock for minutes; a live writer would time out mid-append.
            logger.info("Skipping VACUUM after FTS optimize: another process is using %s", holder.db_path)
            vacuum_ok, vacuum_skipped = False, True
        elif vacuum:
            vacuum_ok = self._optimize_vacuum(progress_cb)
        refusal = holder._execute_write(self._optimize_settle)
        if refusal is not None:
            # A concurrent process changed state since the pre-vacuum check; a re-run can still settle.
            logger.warning("FTS storage optimization settle refused (%s)", refusal)
            return {"ok": False, "reason": refusal, "vacuumed": vacuum_ok}
        emit({"phase": "done", "percent": 100, "indexed": 0, "total": 0})
        logger.info("FTS storage optimization complete (layout v%d).", FTS_STORAGE_VERSION)
        return {"ok": True, "vacuumed": vacuum_ok, **({"vacuum_skipped": "in_use"} if vacuum_skipped else {})}


class MaintenanceAdmin(AdminFile):
    """VACUUM, page statistics, the auto-maintenance lock, the FTS storage rewrite and layout probe, and declarative
    schema drift on the handle's file."""

    def vacuum(self) -> int:
        """VACUUM to reclaim space after large deletes (SQLite never shrinks on its own).
        Takes an exclusive lock: callers must ensure no other writers are active. FTS5
        segments are merged first (``'optimize'``) so their pages are reclaimed too;
        returns the number of FTS indexes optimized (0 on merge failure / no FTS). A quarantined
        handle (corrupt image, replaced file, lost WAL generation) raises before any rewrite: a
        VACUUM reads every page and commits the result back, turning contained damage into an
        amplified one. Same guard ``_execute_write`` applies to every write."""
        with self._writer("vacuum") as holder:
            holder._raise_if_db_corrupt()
            optimized = 0
            try:
                optimized = _optimize_fts(holder)  # manages its own lock
            except Exception as exc:  # health: allow BLE001 -- a failed FTS optimize is logged and VACUUM still runs
                logger.warning("FTS optimize before VACUUM failed: %s", exc)
            with holder._lock:
                holder._raise_if_db_replaced()
                if holder._conn is None:
                    holder._reopen_after_close_locked(context="write")
                # PASSIVE, not TRUNCATE: a manual `hermes sessions vacuum` runs in a transient CLI
                # process; a TRUNCATE reset here would race a live gateway writer.
                _try_checkpoint(holder, "PASSIVE", "WAL checkpoint (PASSIVE) before VACUUM failed: %s")
                holder._conn.execute("VACUUM")
                # VACUUM rewrites every page THROUGH the WAL; without this TRUNCATE a 3 GB DB leaves a 3 GB -wal.
                _try_checkpoint(holder, "TRUNCATE", "WAL checkpoint (TRUNCATE) after VACUUM failed: %s")
                # TRUNCATE may replace the WAL inode; adopt the new sidecars so the
                # write-path generation guard does not halt this connection.
                holder._record_db_file_identity()
            return optimized

    def _page_pragmas(self, names: tuple[str, ...], fail_msg: str) -> Optional[list]:
        """Integer PRAGMAs over the store's connection (never a byte probe); None + debug log on failure."""
        try:
            with self._reader() as holder, holder._read_ctx() as conn:
                if holder._conn is None:
                    return None
                return [int(conn.execute(f"PRAGMA {name}").fetchone()[0]) for name in names]
        except Exception as exc:  # health: allow BLE001 -- an unreadable store reports no page counts
            logger.debug(fail_msg, exc)
            return None

    def logical_size_bytes(self) -> Optional[int]:
        """``page_count * page_size``: main-file size once the WAL is checkpointed in. Prefer
        over ``os.path.getsize`` when reporting a VACUUM: in WAL mode the rewrite lands in
        ``-wal`` and the checkpoint is refused while another connection holds a read-mark, so
        a stat() delta understates the win and can go negative."""
        values = self._page_pragmas(("page_count", "page_size"), "Could not read logical DB size: %s")
        return None if values is None else values[0] * values[1]

    def freelist_ratio(self) -> Optional[float]:
        """Reclaimable fraction (``freelist_count / page_count``) gating automatic VACUUM; None = fall back to the
        time throttle. Read over the store's connection, never a byte-level probe of the live file. This is what
        VACUUM would actually give back, so it decides whether a full rewrite pays off."""
        values = self._page_pragmas(("page_count", "freelist_count"), "Could not read freelist ratio: %s")
        return None if values is None else (values[1] / values[0] if values[0] > 0 else 0.0)

    def vacuum_blockers(self) -> tuple[tuple[int, str], ...]:
        """Every holder that makes VACUUM unsafe: VACUUM plus the TRUNCATE checkpoint retire the WAL generation a
        sibling writer still holds. The foreign scan skips this pid, so it is paired with the in-process arm: under
        one multiplexed process another live store generation for this same path is as much a holder as another
        process. VACUUM runs on the registry's writer for the file, so that generation is not counted."""
        with self._writer("vacuum") as writer:
            in_process = in_process_state_db_holders(self.db_path, exclude=writer)
        return tuple(foreign_state_db_holders(self.db_path) + in_process)

    def try_acquire_auto_maintenance_lock(self) -> Optional[Any]:
        """Non-blocking cross-process lock for one auto-maintenance pass; None means skip the pass."""
        return _try_acquire_auto_maintenance_lock(self.db_path)

    def release_auto_maintenance_lock(self, handle: Any) -> None:
        _release_auto_maintenance_lock(handle)

    def fts_optimize_available(self) -> bool:
        with self._reader() as holder:
            return _FtsStorageOptimizer(holder).fts_optimize_available()

    def optimize_fts_storage(
        self, *, progress_cb: Optional[Callable[[dict[str, Any]], None]] = None, vacuum: bool = True,
    ) -> dict[str, Any]:
        with self._reader() as holder:
            return _FtsStorageOptimizer(holder).optimize_fts_storage(progress_cb=progress_cb, vacuum=vacuum)

    def fts_layout_drift(self) -> list[str]:
        """FTS tables on an older release's definition that only ``optimize_fts_storage`` replaces, read on a
        ``mode=ro`` connection that never creates, migrates or repairs the file; empty when the file cannot be
        read."""
        try:
            with translated():
                holder = StoreHolder(self.db_path, read_only=True)
            try:
                return fts_layout_drift(holder._conn)
            finally:
                holder.close()
        except Exception:  # health: allow BLE001 -- an unreadable file reports no drift
            return []

    def fts_storage_notice(self) -> dict[str, Any]:
        """What the search-index storage notice reads, on one ``mode=ro`` connection that never creates, migrates or
        repairs the file: ``drift`` (as :meth:`fts_layout_drift`), ``legacy_inline`` (``messages_fts`` in a pre-v23
        shape) and ``interrupted`` (an ``optimize_fts_storage`` run left its backfill markers or trash tables)."""
        with translated():
            holder = StoreHolder(self.db_path, read_only=True)
            try:
                conn = holder._conn
                drift = fts_layout_drift(conn)
                legacy_inline = _db_has_legacy_inline_fts(conn)
                interrupted = bool(
                    conn.execute(
                        "SELECT 1 FROM state_meta "
                        "WHERE key = 'fts_rebuild_high_water' LIMIT 1"
                    ).fetchone()
                    or conn.execute(
                        "SELECT 1 FROM sqlite_master WHERE type = 'table' "
                        "AND name LIKE 'fts\\_v22\\_trash\\_%' ESCAPE '\\' LIMIT 1"
                    ).fetchone()
                )
            finally:
                holder.close()
        return {"drift": drift, "legacy_inline": legacy_inline, "interrupted": interrupted}

    def schema_drift_reason(self) -> Optional[str]:
        with translated():
            return _reconcile.schema_drift_reason(self.db_path)

    def repair_declarative_schema(self, *, backup: bool = True) -> dict[str, Any]:
        self._require_writable("repair_declarative_schema")
        with translated():
            return _reconcile.repair_declarative_schema(self.db_path, backup=backup)

    def reconcile_drift_for_browse(self) -> bool:
        self._require_writable("reconcile_drift_for_browse")
        with translated():
            return _reconcile.reconcile_drift_for_browse(self.db_path)
