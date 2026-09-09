"""Compression lineage, cooldown/streak counters, locks and turn leases for SessionDB.

Mixin bound onto ``SessionDB`` via the MRO, built on its ``_read_ctx`` /
``_execute_write`` / ``_write_sql`` / ``_read_one`` primitives."""

from __future__ import annotations

import json
import logging
import sqlite3
import time
from typing import Any, Dict, List, Optional, Tuple

from hermes_state_common import (
    _COMPRESSION_LOCK_ROW_SQL as _LOCK_ROW_SQL, _ENDED_ROW_SQL, _ended_by_compression, _sql_session_last_active,
    is_automatic_end_reason)

# Log-record parity with the origin module (caplog tests pin "hermes_state").
logger = logging.getLogger("hermes_state")

_COOLDOWN_ROW_SQL = (
    "SELECT compression_failure_cooldown_until, compression_failure_error FROM sessions WHERE id = ?"
)

# One forward step of get_compression_chain: the preferred continuation child of ``?``.
_CHAIN_STEP_SQL = f"""
                    SELECT child.id
                    FROM sessions parent
                    JOIN sessions child ON child.parent_session_id = parent.id
                    WHERE parent.id = ?
                      AND parent.end_reason = 'compression'
                      AND json_extract(COALESCE(child.model_config, '{{}}'), '$._branched_from') IS NULL
                      AND json_extract(COALESCE(child.model_config, '{{}}'), '$._delegate_from') IS NULL
                      AND COALESCE(child.source, '') != 'tool'
                    ORDER BY
                      CASE
                        WHEN child.end_reason = 'compression' THEN 0
                        WHEN child.ended_at IS NULL THEN 1
                        ELSE 2
                      END,
                      {_sql_session_last_active("child")} DESC,
                      child.started_at DESC,
                      child.id DESC
                    LIMIT 1
                    """


def _cooldown_row(exists: bool, cooldown_until, error) -> Dict[str, Any]:
    return {"session_exists": exists,
            "cooldown_until": float(cooldown_until) if cooldown_until is not None else None, "error": error}


def _claim_lease_row(conn, table: str, key_col: str, key: str, holder: str, now: float, expires_at: float,
                     stale) -> Tuple[bool, Optional[str]]:
    """Single-transaction lease claim: DELETE a stale holder's row (``stale(holder,
    expires_at)``), INSERT OR IGNORE ours, then SELECT to confirm ownership (INSERT OR
    IGNORE gives no rowcount signal). Returns ``(acquired, reclaimed_holder)``."""
    reclaimed_holder = None
    row = conn.execute(f"SELECT holder, expires_at FROM {table} WHERE {key_col} = ?", (key,)).fetchone()
    if row is not None and stale(row["holder"], row["expires_at"]):
        conn.execute(f"DELETE FROM {table} WHERE {key_col} = ? AND holder = ?", (key, row["holder"]))
        reclaimed_holder = row["holder"]
    conn.execute(
        f"INSERT OR IGNORE INTO {table} ({key_col}, holder, acquired_at, expires_at) VALUES (?, ?, ?, ?)",
        (key, holder, now, expires_at))
    owner = conn.execute(f"SELECT holder FROM {table} WHERE {key_col} = ?", (key,)).fetchone()
    return owner is not None and owner["holder"] == holder, reclaimed_holder


class SessionCompressionMixin:
    """Compression lineage, cooldown/streak counters, locks and turn leases."""

    def find_live_compression_child(self, parent_session_id: str) -> Optional[Dict[str, Any]]:
        """The unique live direct child of a compression-ended session, else None. A stale
        agent whose parent was rotated elsewhere may recover only when the lineage names
        exactly one live continuation; more than one fails closed."""
        if not parent_session_id:
            return None
        with self._read_ctx() as conn:
            if not _ended_by_compression(conn.execute(_ENDED_ROW_SQL, (parent_session_id,)).fetchone()):
                return None
            rows = conn.execute(
                """
                SELECT s.*,
                       COALESCE(sp.prompt, s.system_prompt)
                           AS _system_prompt_resolved
                FROM sessions s
                LEFT JOIN system_prompts sp ON sp.hash = s.system_prompt_hash
                WHERE s.parent_session_id = ?
                  AND s.ended_at IS NULL
                """
                + self._NON_CONTINUATION_CHILD_FILTER_SQL.format(alias="s.")
                + """
                ORDER BY s.started_at ASC
                LIMIT 2
                """,
                (parent_session_id, parent_session_id, parent_session_id),
            ).fetchall()
        return self._session_row_dict(rows[0]) if len(rows) == 1 else None

    def reopen_orphaned_compression_session(self, session_id: str) -> bool:
        """Reopen a compression parent only when no continuation was published (older
        builds could leave a closed parent after an interrupted handoff). Conservative:
        an active lease or any canonical child means another path owns the lineage."""
        if not session_id:
            return False
        def _do(conn):
            if not _ended_by_compression(conn.execute(_ENDED_ROW_SQL, (session_id,)).fetchone()):
                return False
            # Any non-branch/non-delegate/non-tool child is a continuation, ended or not.
            child = conn.execute(
                """
                SELECT 1
                FROM sessions
                WHERE parent_session_id = ?
                """
                + self._NON_CONTINUATION_CHILD_FILTER_SQL.format(alias="")
                + """
                LIMIT 1
                """,
                (session_id, session_id, session_id),
            ).fetchone()
            if child is not None:
                return False
            # refresh_compression_lock() lets an owner revive its own expired row, so reclaim
            # it inside this write txn: refresh-first makes the lease active and aborts
            # recovery; recovery-first deletes the holder so a refresh can't resurrect it.
            now = time.time()
            lock_row = conn.execute(_LOCK_ROW_SQL, (session_id,)).fetchone()
            if lock_row is not None:
                expires_at = lock_row["expires_at"]
                if expires_at is None or float(expires_at) >= now:
                    return False
                deleted = conn.execute(
                    "DELETE FROM compression_locks WHERE session_id = ? AND holder = ? AND expires_at = ?",
                    (session_id, lock_row["holder"], expires_at))
                if deleted.rowcount != 1:
                    return False
            updated = conn.execute(
                # A parent stamped ended by AUTOMATIC cleanup (tui_shutdown, ws_disconnect, orphan reap,
                # idle/LRU evict) while a live agent is publishing its rotation is stale by construction —
                # this writer holds the compression lease and is actively continuing the conversation the
                # stamp claims is over. Left in place it wedges rotation forever: every attempt aborts here,
                # nothing clears the stamp, and each attempt's pre-publish flush re-grows the parent until
                # the provider rejects the request (#88197: 303 unique messages → 2,611 rows → HTTP 400).
                # Clear it in this same transaction and proceed; the closure UPDATE below re-stamps the
                # parent with its true boundary (end_reason='compression'). Deliberate boundaries
                # (compression, session_reset, explicit close) still fail closed — those mean another path
                # owns lineage.
                "UPDATE sessions SET ended_at = NULL, end_reason = NULL "
                "WHERE id = ? AND ended_at IS NOT NULL AND end_reason = 'compression'",
                (session_id,))
            # rowcount==1 is guaranteed by the parent SELECT in this same txn. A False return added past
            # this point must raise instead: the lease DELETE above commits unless _do raises.
            return updated.rowcount == 1
        return bool(self._execute_write(_do))

    def _publish_child_session_row(self, conn, parent, *, parent_session_id, child_session_id, source,
                                   model, model_config, system_prompt, cwd, profile_name) -> None:
        """INSERT the compression child's ``sessions`` row copied from *parent*. Same contract as
        _insert_session_row's compression-fork backfill: the child stays on the parent's profile and keeps
        gateway routing/origin columns; no owner on either side -> this store's profile."""
        system_prompt_hash = self._store_system_prompt(conn, system_prompt)
        conn.execute(
            """INSERT INTO sessions (
                   id, source, model, model_config, system_prompt,
                   system_prompt_hash,
                   parent_session_id, cwd, git_branch, git_repo_root,
                   profile_name, user_id, session_key, chat_id, chat_type,
                   thread_id, display_name, origin_json, started_at
                ) VALUES (?, ?, ?, ?, NULL, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                child_session_id, source, model, json.dumps(model_config) if model_config else None,
                system_prompt_hash, parent_session_id, cwd or parent["cwd"], parent["git_branch"],
                parent["git_repo_root"],
                profile_name or parent["profile_name"] or self._own_profile_name(),
                parent["user_id"], parent["session_key"], parent["chat_id"], parent["chat_type"],
                parent["thread_id"], parent["display_name"], parent["origin_json"], time.time()),
        )

    def publish_compression_child(
        self, *, parent_session_id: str, child_session_id: str, source: str,
        messages: List[Dict[str, Any]], model: str = None, model_config: Dict[str, Any] = None,
        system_prompt: str = None, cwd: str = None, profile_name: str = None,
        compression_lock_holder: str = None, require_compression_lease: bool = True,
        require_lease_refresh: bool = False, lease_ttl_seconds: float = 300.0,
        watermark: Optional[int] = None, watermark_ceiling: Optional[int] = None) -> None:
        """Atomically close a parent and publish its durable compression child: closure, child row, and
        handoff commit in one transaction, so readers see the live parent or a complete child, never an
        ended parent with a missing/empty child. *watermark* (parent's ``get_active_message_watermark`` at compression start): parent rows with ``id
        > watermark`` — appends landed during the slow summary — are column-cloned into the child AFTER the
        handoff. *watermark_ceiling* bounds the clone: the rotation path flushes its OWN transcript to the
        parent just before publishing and those rows are already in the handoff, so only ``(watermark,
        watermark_ceiling]`` is foreign tail (``None`` = unbounded). *require_lease_refresh* +
        *compression_lock_holder* refreshes the lease on the same ``conn`` before the expiry check (no
        TOCTOU window), so a refresher that died on transient DB errors gets one last chance.

        See #75316.
        ``None`` = unbounded (no internal flush happened). See #47202.
        """
        from hermes_state_errors import CompressionSessionBusyError
        def _do(conn):
            if require_lease_refresh and compression_lock_holder:
                conn.execute(
                    "UPDATE compression_locks SET expires_at = ? WHERE session_id = ? AND holder = ?",
                    (time.time() + lease_ttl_seconds, parent_session_id, compression_lock_holder))
            lock_row = conn.execute(_LOCK_ROW_SQL, (parent_session_id,)).fetchone()
            if require_compression_lease and (
                lock_row is None or not compression_lock_holder
                or lock_row["holder"] != compression_lock_holder
                or float(lock_row["expires_at"]) <= time.time()
            ):
                raise CompressionSessionBusyError(
                    f"Compression lease lost before publication: {parent_session_id}")
            parent = conn.execute(
                """SELECT ended_at, end_reason, cwd, git_branch, git_repo_root,
                          user_id, session_key, chat_id, chat_type,
                          thread_id, display_name, origin_json, profile_name
                   FROM sessions WHERE id = ?""",
                (parent_session_id,),
            ).fetchone()
            if parent is None:
                raise RuntimeError(f"Compression parent not found: {parent_session_id}")
            if parent["ended_at"] is not None:
                # An AUTOMATIC end stamp (tui_shutdown, ws_disconnect, orphan reap, idle/LRU
                # evict) is stale by construction — this lease holder is still continuing the
                # conversation, and left alone it wedges rotation forever. Clear it; the closure
                # UPDATE below re-stamps end_reason='compression'. Deliberate boundaries fail closed.
                if not is_automatic_end_reason(parent["end_reason"]):
                    raise RuntimeError(f"Compression parent already ended: {parent_session_id}")
                conn.execute(
                    "UPDATE sessions SET ended_at = NULL, end_reason = NULL WHERE id = ?",
                    (parent_session_id,))
            if not messages:
                raise RuntimeError("Compression child handoff must not be empty")
            self._publish_child_session_row(
                conn, parent, parent_session_id=parent_session_id, child_session_id=child_session_id,
                source=source, model=model, model_config=model_config, system_prompt=system_prompt,
                cwd=cwd, profile_name=profile_name)
            total_messages, total_tool_calls = self._insert_message_rows(conn, child_session_id, messages)
            if watermark is not None:
                # Clone the parent's concurrent tail into the child after the handoff;
                # originals stay in the closed parent for lineage recovery.
                bounded = watermark_ceiling is not None
                tail_ids, tail_tool_calls = self._tail_rows_after_watermark(
                    conn, "SELECT id, tool_calls FROM messages "
                    "WHERE session_id = ? AND active = 1 AND id > ?"
                    f"{' AND id <= ?' if bounded else ''} ORDER BY id",
                    [parent_session_id, int(watermark), *([int(watermark_ceiling)] if bounded else [])])
                if tail_ids:
                    self._clone_message_rows_with_lineage(conn, tail_ids, child_session_id)
                    total_messages += len(tail_ids)
                    total_tool_calls += tail_tool_calls
            conn.execute(
                "UPDATE sessions SET message_count = ?, tool_call_count = ? WHERE id = ?",
                (total_messages, total_tool_calls, child_session_id))
            updated = conn.execute(
                "UPDATE sessions SET ended_at = ?, end_reason = 'compression' "
                "WHERE id = ? AND ended_at IS NULL", (time.time(), parent_session_id))
            if updated.rowcount != 1:
                raise RuntimeError(f"Compression parent changed during publication: {parent_session_id}")
            self._bump_display_revision(conn, parent_session_id)
        self._execute_write(_do)

    def _write_sql_logged(self, op: str, session_id: str, sql: str, params) -> None:
        """``_write_sql`` that logs (never raises) on ``sqlite3.Error``."""
        try:
            self._write_sql(sql, params)
        except sqlite3.Error as exc:
            logger.warning("%s(%s) failed: %s", op, session_id, exc)

    def record_compression_failure_cooldown(
        self, session_id: str, cooldown_until: float, error: Optional[str] = None) -> None:
        """Persist the active compression-failure cooldown. Merge-max with any longer live deadline so a
        later shorter write can't reopen the thrash window; error always takes the latest diagnostic."""
        if not session_id:
            return
        self._write_sql_logged(
            "record_compression_failure_cooldown", session_id,
            # Merge-max with any longer live deadline so a later shorter write cannot reopen the thrash
            # window (#96775). The error column always takes the latest diagnostic.
            "UPDATE sessions SET compression_failure_cooldown_until = CASE "
            "WHEN compression_failure_cooldown_until IS NOT NULL  AND compression_failure_cooldown_until > ? "
            "THEN compression_failure_cooldown_until ELSE ? END, compression_failure_error = ? WHERE id = ?",
            (cooldown_until, cooldown_until, error, session_id))

    def get_compression_failure_cooldown(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Return the active (unexpired) compression-failure cooldown, or None."""
        now = time.time()
        row = self._read_one(_COOLDOWN_ROW_SQL, (session_id,)) if session_id else None
        if row is None or row[0] is None or float(row[0]) <= now:
            return None
        return {"cooldown_until": float(row[0]), "remaining_seconds": float(row[0]) - now, "error": row[1]}

    def get_compression_failure_cooldown_row(self, session_id: str) -> Dict[str, Any]:
        """Exact stored cooldown columns, no expiry filtering, so compression cancellation
        can roll back an expired, partially-null, or absent row exactly."""
        row = self._read_one(_COOLDOWN_ROW_SQL, (session_id,)) if session_id else None
        return _cooldown_row(False, None, None) if row is None else _cooldown_row(True, row[0], row[1])

    def restore_compression_failure_cooldown_row(self, session_id: str, snapshot: Dict[str, Any]) -> None:
        """Restore and verify an exact cooldown-row snapshot. Unlike record/clear this
        rollback API propagates write and verification failures: cancellation must not be
        reported mutation-free when compensation failed."""
        if not snapshot.get("session_exists", False):
            if self.get_compression_failure_cooldown_row(session_id).get("session_exists", False):
                raise RuntimeError("cannot restore absent compression cooldown row: session now exists")
            return
        deadline = snapshot.get("cooldown_until")
        error = snapshot.get("error")
        def _do(conn):
            cursor = conn.execute(
                "UPDATE sessions SET compression_failure_cooldown_until = ?, "
                "compression_failure_error = ? WHERE id = ?", (deadline, error, session_id))
            if cursor.rowcount != 1:
                raise RuntimeError(f"compression cooldown rollback session missing: {session_id}")
        self._execute_write(_do)
        actual = self.get_compression_failure_cooldown_row(session_id)
        expected = _cooldown_row(True, deadline, error)
        if actual != expected:
            raise RuntimeError(
                f"compression cooldown rollback verification failed: expected={expected!r}, actual={actual!r}")

    def clear_compression_failure_cooldown(self, session_id: str) -> None:
        """Clear any persisted compression-failure cooldown for a session."""
        if not session_id:
            return
        self._write_sql_logged(
            "clear_compression_failure_cooldown", session_id,
            "UPDATE sessions SET compression_failure_cooldown_until = NULL, "
            "compression_failure_error = NULL WHERE id = ?", (session_id,))

    def _read_session_number(self, column: str, session_id: str, cast: type, zero: Any) -> Any:
        """Read one numeric ``sessions`` column clamped at ``zero``; a missing session,
        NULL, or unparsable value also reads as ``zero``."""
        row = self._read_one(f"SELECT {column} FROM sessions WHERE id = ?", (session_id,)) if session_id else None
        try:
            return zero if row is None else max(zero, cast(row[0] or zero))
        except (TypeError, ValueError):
            return zero

    def _write_session_column(self, column: str, session_id: str, value: Any) -> None:
        self._write_sql(f"UPDATE sessions SET {column} = ? WHERE id = ?", (value, session_id))

    def get_compression_fallback_streak(self, session_id: str) -> int:
        """Return the persisted deterministic-fallback streak."""
        return self._read_session_number("compression_fallback_streak", session_id, int, 0)

    def set_compression_fallback_streak(self, session_id: str, streak: int) -> None:
        """Persist the deterministic-fallback streak for one session."""
        if session_id:
            self._write_session_column("compression_fallback_streak", session_id, max(0, int(streak)))

    def get_compression_ineffective_count(self, session_id: str) -> int:
        """Persisted ineffective-compaction strike count — the durable half of the built-in
        compressor's anti-thrash guard, so a fresh compressor bound to a resumed session
        inherits an armed/tripped guard across restarts."""
        return self._read_session_number("compression_ineffective_count", session_id, int, 0)

    def set_compression_ineffective_count(self, session_id: str, count: int) -> None:
        """Persist the ineffective-compaction strike count for one session."""
        if session_id:
            self._write_session_column("compression_ineffective_count", session_id, max(0, int(count)))

    def get_compression_recovery_deadline(self, session_id: str) -> float:
        """Persisted anti-thrash recovery deadline (epoch; ``0.0`` = not armed). Durable
        because the gateway rebuilds the compressor every turn / cache eviction.

        The deadline is the durable half of the 14694 recovery clock: the gateway rebuilds the compressor on
        every turn / cache eviction, so a process-local deadline restarted the wait on each rebuild and a
        tripped session never earned its probe (#100185).
        """
        return self._read_session_number("compression_recovery_deadline", session_id, float, 0.0)

    def set_compression_recovery_deadline(self, session_id: str, deadline: float) -> None:
        """Persist the anti-thrash recovery deadline; ``0`` / ``None`` disarms it."""
        if not session_id:
            return
        try:
            normalized = max(0.0, float(deadline or 0.0))
        except (TypeError, ValueError):
            normalized = 0.0
        self._write_session_column("compression_recovery_deadline", session_id, normalized or None)

    def refresh_compression_lock(self, session_id: str, holder: str, ttl_seconds: float = 300.0) -> bool:
        """Extend the compression lock lease if ``holder`` still owns it. Ownership is decided by ``holder``
        alone, deliberately NOT ``expires_at``: a live owner whose refresher stalled past its TTL must be
        able to revive its still-unclaimed row, otherwise it keeps compressing with no lease — the window in
        which a competing path can fork the lineage. It cannot resurrect a lock someone else took: SQLite
        serialises writes, so the reclaim (DELETE-expired + INSERT OR IGNORE) never interleaves with this
        UPDATE."""
        if not session_id or not holder:
            return False
        expires_at = time.time() + ttl_seconds
        try:
            return self._write_rowcount(
                "UPDATE compression_locks SET expires_at = ? WHERE session_id = ? AND holder = ?",
                (expires_at, session_id, holder)) > 0
        except sqlite3.Error as exc:
            logger.warning("refresh_compression_lock(%s) failed: %s", session_id, exc)
            return False

    def try_acquire_compression_lock(self, session_id: str, holder: str, ttl_seconds: float = 300.0) -> bool:
        """Try to atomically acquire the compression lock for ``session_id``. ``False``: another holder owns
        a live lock and the caller MUST NOT compress (its rotation would split the lineage). Expired
        locks and structured holders whose local ``pid=`` is dead are reclaimed transparently."""
        from hermes_state import _compression_lock_holder_process_is_dead
        if not session_id:
            return False
        now = time.time()
        expires_at = now + ttl_seconds
        def _do(conn):
            return _claim_lease_row(
                conn, "compression_locks", "session_id", session_id, holder, now, expires_at,
                lambda h, e: e < now or _compression_lock_holder_process_is_dead(h))

        try:
            acquired, reclaimed_holder = self._execute_write(_do)
            if reclaimed_holder:
                logger.warning("Reclaimed stale compression lock for session=%s (holder=%s)",
                               session_id, reclaimed_holder)
            return bool(acquired)
        except sqlite3.Error as exc:
            # False makes the caller skip compression — safe when the lock subsystem is broken.
            logger.warning("try_acquire_compression_lock(%s) failed: %s", session_id, exc)
            return False

    def release_compression_lock(self, session_id: str, holder: str) -> None:
        """Release the compression lock iff we own it; idempotent when gone/reclaimed."""
        if not session_id:
            return
        self._write_sql_logged(
            "release_compression_lock", session_id,
            "DELETE FROM compression_locks WHERE session_id = ? AND holder = ?",
            (session_id, holder))

    def _session_turn_lease_key_on_conn(self, conn, session_id: str) -> str:
        """Walk compression parents on ``conn`` to the conversation lease key. Must share
        the connection of the lease INSERT/UPDATE/DELETE: a failed lookup must not yield a
        child id the write then persists. Markers bind to ``parent_session_id``. Lock
        errors propagate so ``_execute_write`` can retry."""
        if not session_id:
            return session_id
        def _row(sid: str):
            row = conn.execute(
                "SELECT id, parent_session_id, source, model_config, end_reason FROM sessions WHERE id = ?",
                (sid,)).fetchone()
            return dict(row) if row else None

        current = _row(session_id)
        seen = {session_id}
        while current:
            parent_id = current.get("parent_session_id")
            if not parent_id or parent_id in seen or self._is_explicit_fork_child_row(current):
                break
            parent = _row(parent_id)
            if not parent or parent.get("end_reason") != "compression":
                break
            seen.add(parent_id)
            current = parent
        return str(current.get("id") or session_id) if current else session_id

    def _session_turn_lease_key(self, session_id: str) -> str:
        """Stable serialization key for every compression segment (tests/diagnostics; the
        write paths resolve it inside their own txn). Does not swallow lock errors."""
        if not session_id:
            return session_id
        with self._read_ctx() as conn:
            return self._session_turn_lease_key_on_conn(conn, session_id)

    def try_acquire_session_turn_lease(
        self, session_id: str, holder: str, *, ttl_seconds: float = 300.0, patience_s: Optional[float] = None,
    ) -> bool:
        """Atomically acquire the cross-process turn lease for a conversation (keyed by the
        lineage root). The walk, the INSERT, and reclaim of expired or dead-local-PID leases
        share one write transaction."""
        from hermes_state import _compression_lock_holder_process_is_dead
        if not session_id or not holder:
            return False
        now = time.time()
        expires_at = now + max(0.1, float(ttl_seconds))
        def _do(conn):
            conversation_id = self._session_turn_lease_key_on_conn(conn, session_id)
            return _claim_lease_row(
                conn, "session_turn_leases", "conversation_id", conversation_id, holder, now, expires_at,
                lambda h, e: float(e) <= now or _compression_lock_holder_process_is_dead(h),
            )[0]
        return bool(self._execute_write(_do, patience_s=patience_s))

    def acquire_session_turn_lease(
        self, session_id: str, holder: str, *, ttl_seconds: float = 300.0,
        wait_seconds: float = 1800.0, poll_interval_seconds: float = 1.0, on_wait=None,
        wait_notice_interval_seconds: float = 15.0, should_abort=None, acquire_patience_s: float = 0.5,
    ) -> bool:
        """Wait for a cross-process turn lease without holding a SQLite lock. ``on_wait(elapsed)`` is
        best-effort: called when the first attempt fails and about every ``wait_notice_interval_seconds``
        after. ``should_abort()`` True (e.g. ``/stop``) returns False at once."""
        from hermes_state import classify_persistence_error
        deadline = time.monotonic() + max(0.0, float(wait_seconds))
        wait_started = None
        last_notice_at = None
        notice_every = max(0.0, float(wait_notice_interval_seconds))
        while True:
            if should_abort is not None:
                try:
                    if should_abort():
                        return False
                except Exception:
                    logger.debug("session turn lease should_abort callback failed", exc_info=True)
            try:
                if self.try_acquire_session_turn_lease(
                    session_id, holder, ttl_seconds=ttl_seconds, patience_s=acquire_patience_s):
                    return True
            except sqlite3.Error as exc:
                # Long holder transactions can exhaust one write-patience budget; keep
                # polling until wait_seconds or should_abort.
                if classify_persistence_error(exc) != "locked":
                    raise
            now = time.monotonic()
            remaining = deadline - now
            if remaining <= 0:
                return False
            if wait_started is None:
                wait_started = now
            if on_wait is not None and (
                last_notice_at is None or notice_every == 0.0 or (now - last_notice_at) >= notice_every
            ):
                try:
                    on_wait(max(0.0, now - wait_started))
                except Exception:
                    logger.debug("session turn lease on_wait callback failed", exc_info=True)
                last_notice_at = now
            time.sleep(min(max(0.01, float(poll_interval_seconds)), remaining))

    def refresh_session_turn_lease(self, session_id: str, holder: str, *, ttl_seconds: float = 300.0) -> bool:
        """Extend a turn lease only while ``holder`` still owns it."""
        if not session_id or not holder:
            return False
        expires_at = time.time() + max(0.1, float(ttl_seconds))
        def _do(conn):
            conversation_id = self._session_turn_lease_key_on_conn(conn, session_id)
            return conn.execute(
                "UPDATE session_turn_leases SET expires_at = ? "
                "WHERE conversation_id = ? AND holder = ?", (expires_at, conversation_id, holder),
            ).rowcount > 0
        return bool(self._execute_write(_do))

    def release_session_turn_lease(self, session_id: str, holder: str) -> None:
        """Release a turn lease iff ``holder`` still owns it; idempotent."""
        if not session_id or not holder:
            return
        def _do(conn):
            conversation_id = self._session_turn_lease_key_on_conn(conn, session_id)
            conn.execute(
                "DELETE FROM session_turn_leases WHERE conversation_id = ? AND holder = ?",
                (conversation_id, holder))
        self._execute_write(_do)

    def get_compression_lock_holder(self, session_id: str) -> Optional[str]:
        """Current (non-expired) holder for ``session_id``, or None. Diagnostic only."""
        if not session_id:
            return None
        row = self._read_one(
            "SELECT holder FROM compression_locks WHERE session_id = ? AND expires_at >= ?", (session_id, time.time()))
        return None if row is None else row[0]

    def finalize_orphaned_compression_sessions(self) -> int:
        """Mark orphaned compression continuations (parent ended by compression; child has
        messages, no end_reason/ended_at, api_call_count=0, older than 7 days) as
        ``orphaned_compression``. Non-destructive.

        Fix for #20001.
        """
        cutoff = time.time() - 604800  # 7 days
        return self._write_rowcount(
                """
                UPDATE sessions
                SET ended_at = ?,
                    end_reason = 'orphaned_compression'
                WHERE api_call_count = 0
                  AND end_reason IS NULL
                  AND ended_at IS NULL
                  AND started_at < ?
                  AND parent_session_id IS NOT NULL
                  AND EXISTS (
                      SELECT 1 FROM sessions p
                      WHERE p.id = sessions.parent_session_id
                        AND p.end_reason = 'compression'
                        AND p.ended_at IS NOT NULL
                  )
                  AND EXISTS (
                      SELECT 1 FROM messages m
                      WHERE m.session_id = sessions.id
                  )
                """,
                (time.time(), cutoff),
        ) or 0

    def get_compression_chain(self, session_id: str) -> List[str]:
        """Walk the compression-continuation chain forward: root-first through the tip (``[session_id]``
        when no continuation); ``get_compression_tip`` is the last element. A continuation is a child of
        a session with ``end_reason='compression'``. The old ``child.started_at >= parent.ended_at`` test
        was too brittle (gateway + compression races insert the real continuation before ``ended_at`` is
        written, while a stale websocket later creates a sibling that passes it). Instead exclude
        branch/delegate/tool children and prefer children that continue the chain or are still live over
        stale closed siblings such as ``ws_orphan_reap``."""
        with self._read_ctx() as conn:
            return self._get_compression_chain_from_conn(conn, session_id)

    def get_compression_tip(self, session_id: str) -> Optional[str]:
        """Live tip of a compression chain (``get_compression_chain`` semantics); the input
        id when no continuation exists."""
        with self._read_ctx() as conn:
            return self._get_compression_tip_from_conn(conn, session_id)

    def _is_compression_child_row(self, child: Dict[str, Any]) -> bool:
        parent_id = child.get("parent_session_id")
        if not parent_id or self._is_explicit_fork_child_row(child):
            return False
        parent = self.get_session(parent_id)
        return bool(parent and parent.get("end_reason") == "compression")

    def get_compression_lineage(self, session_id: str) -> List[str]:
        """Return compression ancestors through tip in chronological order."""
        session = self.get_session(session_id)
        if not session or self._is_explicit_fork_child_row(session):
            return [session_id] if session else []
        root = session
        ancestors = {root["id"]}
        while self._is_compression_child_row(root):
            parent = self.get_session(root["parent_session_id"])
            if not parent or parent["id"] in ancestors:
                break
            root = parent
            ancestors.add(root["id"])
        lineage = [root["id"]]
        seen = {root["id"]}
        current = root
        while current.get("end_reason") == "compression":
            rows = self._read_all(
                """
                SELECT * FROM sessions
                WHERE parent_session_id = ?
                ORDER BY started_at ASC
                """, (current["id"],))
            next_child = next((dict(row) for row in rows if self._is_compression_child_row(dict(row))), None)
            if not next_child or next_child["id"] in seen:
                break
            lineage.append(next_child["id"])
            seen.add(next_child["id"])
            current = next_child
        # Later tips are included only when the requested session itself was compacted.
        return lineage if session_id in lineage else [session_id]
