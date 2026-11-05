"""Compression lineage: the chain walks, child publication, orphan repair, cooldowns and the durable counters."""


from __future__ import annotations
import json
import logging
import sqlite3
import time
from typing import Any, Optional
from hermes_state_common import _BOUNDARY_END_REASONS, _ended_by_compression, is_automatic_end_reason
from persistence.sqlite._schema_ddl import _RESET_CHILD_SQL, _sql_json_extract
from persistence.sqlite.conversation._store.sql_common import (
    _COMPRESSION_LOCK_ROW_SQL as _LOCK_ROW_SQL, _ENDED_ROW_SQL, _non_continuation_child_sql, _sql_session_last_active,
)
from persistence.sqlite._holder import StoreHolder


logger = logging.getLogger("hermes_state")


_COOLDOWN_ROW_SQL = (
    "SELECT compression_failure_cooldown_until, compression_failure_error FROM sessions WHERE id = ?"
)


_CHAIN_STEP_SQL = f"""
                    SELECT child.id
                    FROM sessions parent
                    JOIN sessions child ON child.parent_session_id = parent.id
                    WHERE parent.id = ?
                      AND parent.end_reason = 'compression'
                      AND {_sql_json_extract('child.model_config', '$._branched_from')} IS NULL
                      AND {_sql_json_extract('child.model_config', '$._delegate_from')} IS NULL
                      AND NOT ({_RESET_CHILD_SQL.format(a='child')})
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


def _cooldown_row(exists: bool, cooldown_until, error) -> dict[str, Any]:
    return {"session_exists": exists,
            "cooldown_until": float(cooldown_until) if cooldown_until is not None else None, "error": error}


_CHAIN_CAP = 1000


_CONVERSATION_MODEL_CONFIG_KEYS = ("onboarding_run", "prompt_refresh_epoch")


def _walk_compression_chain(session_id: str, step) -> list[str]:
    """Root-first ids from *session_id* through its tip; ``step(id)`` returns the ``_CHAIN_STEP_SQL`` row."""
    current = session_id
    chain = [current] if current else []
    seen = set(chain)
    # Defensive bound; ``seen`` guards cycles. 100 truncated real
    # compression lineages (~180 deep), stranding root→tip walks on a
    # stale mid id (#125041).
    for _ in range(_CHAIN_CAP):
        row = step(current)
        child_id = row["id"] if row is not None else None
        if not child_id or child_id in seen:
            return chain
        seen.add(child_id)
        current = child_id
        chain.append(child_id)
    return chain


_UNSET_CREATED_SOURCES = frozenset({None, "", "unknown", "local"})


_UNSET_CREATED_SQL = "COALESCE({col}, '') IN ('', 'unknown', 'local')"


_CREATED_ROW_SQL = "SELECT parent_session_id, created_source FROM sessions WHERE id = ?"


_UNSET_ANCESTOR_SQL = """
    WITH RECURSIVE up(pid, n) AS (
        SELECT parent_session_id, 1 FROM sessions WHERE id = ?
        UNION ALL
        SELECT s.parent_session_id, up.n + 1 FROM up JOIN sessions s ON s.id = up.pid
        WHERE s.end_reason = 'compression' AND s.created_source = ? AND up.n < ?
    )
    SELECT 1 FROM up CROSS JOIN sessions s ON s.id = up.pid
    WHERE s.end_reason = 'compression' AND """ + _UNSET_CREATED_SQL.format(col="s.created_source") + """
    LIMIT 1
"""


def stamp_compression_ancestors(conn, session_id: str) -> None:
    """Copy *session_id*'s ``created_source`` onto compression ancestors that have none.

    A list page shows a compressed conversation as its root, painted as the tip, and filters on
    the root's ``created_source``. Automatic compression publishes the child before any writer
    knows the surface, so the turn's usage write stamps only the tip. Stamping the chain here,
    in that same transaction, keeps the root matchable without a list-time chain walk. Each hop
    must be the parent's preferred continuation (``_CHAIN_STEP_SQL``); ``unknown`` and ``local``
    count as unset, as in the creator upsert. An ancestor already on the same surface is walked through
    (a NULL root behind a stamped mid segment still gets stamped); another surface stops it.
    A chain with no unset ancestor returns after one query (``_UNSET_ANCESTOR_SQL``).
    """
    row = conn.execute(_CREATED_ROW_SQL, (session_id,)).fetchone()
    surface = row["created_source"] if row else None
    if surface in _UNSET_CREATED_SOURCES:
        return
    if conn.execute(_UNSET_ANCESTOR_SQL, (session_id, surface, _CHAIN_CAP)).fetchone() is None:
        return
    current, parent_id = session_id, row["parent_session_id"]
    for _ in range(_CHAIN_CAP):
        parent = parent_id and conn.execute(_CREATED_ROW_SQL, (parent_id,)).fetchone()
        if not parent or parent["created_source"] not in (*_UNSET_CREATED_SOURCES, surface):
            return
        step = conn.execute(_CHAIN_STEP_SQL, (parent_id,)).fetchone()
        if step is None or step["id"] != current:
            return
        if parent["created_source"] != surface:
            conn.execute("UPDATE sessions SET created_source = ? WHERE id = ?", (surface, parent_id))
        current, parent_id = parent_id, parent["parent_session_id"]


_UNSTAMPED_EDGE_CHILD_SQL = f"""
    SELECT child.id FROM sessions parent JOIN sessions child ON child.parent_session_id = parent.id
    WHERE parent.end_reason = 'compression' AND {_UNSET_CREATED_SQL.format(col="parent.created_source")}
      AND NOT {_UNSET_CREATED_SQL.format(col="child.created_source")}
{_non_continuation_child_sql("child.", "parent.id")}"""


def reopen_if_explicitly_closed(
    holder: StoreHolder, session_id: str, *, provenance: str, patience_s: Optional[float] = None,
) -> Optional[str]:
    """Clear an explicit-close stamp (``tui_close``, ``cli_close``, ``webhook_complete``, ...) from a
    session a HOST has just proven is still routed to it, returning the reason cleared or None (#106459).
    Narrow twin of ``reopen_session()``: automatic stamps are left to publish (#88197); ``compression``,
    boundary (reset reasons, CLI ``new_session``) and stamps with a published continuation own lineage
    elsewhere and are never touched. The UPDATE is conditional on the exact stamp read, so a close
    landing between read and write survives.

    Only the routing host can make this call. Publication cannot: ``end_session()`` is first-stamp-wins,
    so a close made during a turn that began on a stale stamp is a no-op write. Turn-lease admission
    cannot: the TUI starts its worker before it reaches ``run_conversation()``, so ``session.close`` can
    stamp ``tui_close`` in between and a late lease would clear a deliberate close. Call it under the
    lock that makes the host's registry claim atomic with its teardown (#54878 on the routing table)."""
    if not session_id:
        return None

    def _do(conn):
        row = conn.execute(_ENDED_ROW_SQL, (session_id,)).fetchone()
        if row is None or row["ended_at"] is None:
            return None
        reason = row["end_reason"]
        if is_automatic_end_reason(reason) or reason == "compression" or reason in _BOUNDARY_END_REASONS:
            return None
        superseded = conn.execute(
            "SELECT 1 FROM sessions WHERE parent_session_id = ?"
            + _NON_CONTINUATION_CHILD_FILTER_SQL.format(alias="") + " LIMIT 1",
            (session_id,) * 4).fetchone()
        if superseded is not None:
            return None
        conn.execute(
            "UPDATE sessions SET ended_at = NULL, end_reason = NULL "
            "WHERE id = ? AND ended_at = ? AND end_reason = ?",
            (session_id, row["ended_at"], reason))
        return str(reason)
    reason = holder._execute_write(_do, patience_s=patience_s)
    if reason is not None:
        logger.warning(
            "Session %s carried a stale %r end stamp while %s; cleared so the conversation can "
            "compress and a later close is recorded (#106459)", session_id, reason, provenance)
    return reason


def stamp_legacy_compression_chains(holder: StoreHolder) -> int:
    """Stamp the unset ancestors of chains compressed before write-time stamping from their
    stamped segments, through ``stamp_compression_ancestors`` (same edge checks; a known value
    is never replaced). One write transaction and idempotent, so the owner runs it on every
    start: a store with nothing to repair pays one scan of sessions under the write lock, plus
    a one-hop walk per stamped child of an unset compression parent that the walk does not
    follow (a non-preferred sibling). Returns the rows stamped."""
    def _do(conn):
        before = conn.total_changes
        for row in conn.execute(_UNSTAMPED_EDGE_CHILD_SQL).fetchall():
            stamp_compression_ancestors(conn, row["id"])
        return conn.total_changes - before
    return holder._execute_write(_do)


def find_live_compression_child(holder: StoreHolder, parent_session_id: str) -> Optional[dict[str, Any]]:
    """The unique live direct child of a compression-ended session, else None. A stale
    agent whose parent was rotated elsewhere may recover only when the lineage names
    exactly one live continuation; more than one fails closed."""
    if not parent_session_id:
        return None
    with holder._read_ctx() as conn:
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
            + _NON_CONTINUATION_CHILD_FILTER_SQL.format(alias="s.")
            + """
                ORDER BY s.started_at ASC
                LIMIT 2
                """,
            (parent_session_id,) * 4,
        ).fetchall()
    return _session_row_dict(rows[0]) if len(rows) == 1 else None


def reopen_orphaned_compression_session(holder: StoreHolder, session_id: str) -> bool:
    """Reopen a compression parent only when no continuation was published (older
    builds could leave a closed parent after an interrupted handoff). Conservative:
    an active lease or any canonical child means another path owns the lineage."""
    if not session_id:
        return False
    def _do(conn):
        if not _ended_by_compression(conn.execute(_ENDED_ROW_SQL, (session_id,)).fetchone()):
            return False
        # Any non-branch/non-delegate/non-reset/non-tool child is a continuation, ended or not.
        child = conn.execute(
            """
                SELECT 1
                FROM sessions
                WHERE parent_session_id = ?
                """
            + _NON_CONTINUATION_CHILD_FILTER_SQL.format(alias="")
            + """
                LIMIT 1
                """,
            (session_id,) * 4,
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
    return bool(holder._execute_write(_do))


def _publish_child_session_row(holder: StoreHolder, conn, parent, *, parent_session_id, child_session_id, source,
                               model, model_config, system_prompt, cwd, profile_name) -> None:
    """INSERT the compression child's ``sessions`` row copied from *parent*. Same contract as
    _insert_session_row's compression-fork backfill: the child stays on the parent's profile and keeps
    gateway routing/origin columns; no owner on either side -> this store's profile."""
    from persistence.sqlite.conversation._store.sessions import _parse_model_config
    # Pin/archive and the conversation keys are compression-lineage state; canonical branch/reset
    # verbs also publish through here, and their child is its own conversation.
    continuation = not any(k in (model_config or {}) for k in ("_branched_from", "_reset_from"))
    carried = {key: value for key, value in _parse_model_config(parent["model_config"]).items()
               if continuation and key in _CONVERSATION_MODEL_CONFIG_KEYS}
    model_config = {**carried, **(model_config or {})}
    system_prompt_hash = holder._store_system_prompt(conn, system_prompt)
    # Copy a real creating surface. Automatic compression passes the parent's live
    # label (``local`` after the peer refresh); reset, branch and the compress
    # mutation pass the policy label. ``local`` and ``unknown`` are not surfaces,
    # so those stay NULL. Compression sets ``_session_db_created`` and does not
    # create the child again. The turn's ``update_token_counts(source=...)`` fills
    # that NULL; auxiliary usage must not write ``unknown`` into it first.
    from session_rules.surfaces import created_surface
    created_source = created_surface(parent["created_source"]) or created_surface(source)
    # The child continues the parent's tools[] pin (the compaction refresh re-pinned it just
    # before publish), or its first hop to another surface re-derives the array.
    conn.execute(
        """INSERT INTO sessions (
                   id, source, created_source, model, model_config, system_prompt,
                   system_prompt_hash, tool_names,
                   parent_session_id, cwd, git_branch, git_repo_root,
                   profile_name, user_id, session_key, chat_id, chat_type,
                   thread_id, display_name, origin_json, pinned, started_at,
                   archived, auto_archived
                ) VALUES (?, ?, ?, ?, ?, NULL, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            child_session_id, source, created_source, model, json.dumps(model_config) if model_config else None,
            system_prompt_hash, parent["tool_names"], parent_session_id, cwd or parent["cwd"], parent["git_branch"],
            parent["git_repo_root"],
            profile_name or parent["profile_name"] or _own_profile_name(holder),
            parent["user_id"], parent["session_key"], parent["chat_id"], parent["chat_type"],
            parent["thread_id"], parent["display_name"], parent["origin_json"],
            # The pin is lineage-wide (set_session_pinned); a segment published after it joins it.
            int(parent["pinned"] or 0) if continuation else 0, time.time(),
            # Inherit the lineage's archive state so a manually archived chat stays uniformly
            # archived (a mixed lineage let the sweep re-stamp its fresh tip as auto-archived).
            (parent["archived"] or 0) if continuation else 0,
            (parent["auto_archived"] or 0) if continuation else 0),
    )


def publish_compression_child(
    holder: StoreHolder, *, parent_session_id: str, child_session_id: str, source: str,
    messages: list[dict[str, Any]], model: str | None = None, model_config: dict[str, Any] | None = None,
    system_prompt: str | None = None, cwd: str | None = None, profile_name: str | None = None,
    compression_lock_holder: str | None = None, require_compression_lease: bool = True,
    require_lease_refresh: bool = False, lease_ttl_seconds: float = 300.0,
    watermark: Optional[int] = None, watermark_ceiling: Optional[int] = None) -> None:
    """Atomically close a parent and publish its durable compression child: closure, child row, and
    handoff commit in one transaction, so readers see the live parent or a complete child, never an
    ended parent with a missing/empty child. *watermark* (the parent's highest row already represented
    in the handoff: its ``get_active_message_watermark`` at compression start, or the newest row of an
    adopted durable snapshot): parent rows with ``id > watermark`` — appends landed during the slow
    summary — are column-cloned into the child AFTER the handoff. *watermark_ceiling* bounds the clone:
    the rotation path flushes its OWN transcript to the parent just before publishing and those rows are
    already in the handoff, so only ``(watermark, watermark_ceiling]`` is foreign tail (``None`` =
    unbounded). *require_lease_refresh* +
    *compression_lock_holder* refreshes the lease on the same ``conn`` before the expiry check (no
    TOCTOU window), so a refresher that died on transient DB errors gets one last chance.

    See #75316.
    ``None`` = unbounded (no internal flush happened). See #47202.
    """
    _execute_transcript_write(holder, lambda conn: _publish_compression_child_on_conn(holder,
        conn, parent_session_id=parent_session_id, child_session_id=child_session_id, source=source,
        messages=messages, model=model, model_config=model_config, system_prompt=system_prompt,
        cwd=cwd, profile_name=profile_name, compression_lock_holder=compression_lock_holder,
        require_compression_lease=require_compression_lease, require_lease_refresh=require_lease_refresh,
        lease_ttl_seconds=lease_ttl_seconds, watermark=watermark, watermark_ceiling=watermark_ceiling), messages)


def _publish_compression_child_on_conn(
        holder: StoreHolder, conn, *, parent_session_id: str, child_session_id: str, source: str,
        messages: list[dict[str, Any]], model: str | None = None, model_config: dict[str, Any] | None = None,
        system_prompt: str | None = None, cwd: str | None = None, profile_name: str | None = None,
        compression_lock_holder: str | None = None, require_compression_lease: bool = True,
        require_lease_refresh: bool = False, lease_ttl_seconds: float = 300.0,
        watermark: Optional[int] = None, watermark_ceiling: Optional[int] = None) -> None:
    """:meth:`publish_compression_child` on the caller's write transaction (never commits). The
    owner and the worker/canonical-mutation verbs share this one body so they cannot drift."""
    from hermes_state_errors import CompressionSessionBusyError
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
                      thread_id, display_name, origin_json, profile_name, tool_names,
                      archived, auto_archived, pinned, model_config, created_source
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
    _publish_child_session_row(holder,
        conn, parent, parent_session_id=parent_session_id, child_session_id=child_session_id,
        source=source, model=model, model_config=model_config, system_prompt=system_prompt,
        cwd=cwd, profile_name=profile_name)
    # Carried handoff tail rows arrive without a timestamp and would otherwise be stamped
    # `now`, breaking their _display_dedupe_key identity with the parent's durable originals
    # and duplicating them in the lineage display read (#59661).
    _carry_parent_timestamps(holder, conn, parent_session_id, messages)
    total_messages, total_tool_calls = _insert_message_rows(holder, conn, child_session_id, messages)
    if watermark is not None:
        # Clone the parent's concurrent tail into the child after the handoff;
        # originals stay in the closed parent for lineage recovery.
        bounded = watermark_ceiling is not None
        tail_ids, tail_tool_calls = _tail_rows_after_watermark(holder,
            conn, "SELECT id, tool_calls FROM messages "
            "WHERE session_id = ? AND active = 1 AND id > ?"
            f"{' AND id <= ?' if bounded else ''} ORDER BY id",
            [parent_session_id, int(watermark), *([int(watermark_ceiling)] if bounded else [])])
        if tail_ids:
            _clone_message_rows(holder, conn, tail_ids, session_id=child_session_id)
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
    # A child on a real surface stamps the chain's unset ancestors, so compressing a conversation
    # whose root predates write-time stamping lists it again under its surface.
    stamp_compression_ancestors(conn, child_session_id)
    from persistence.sqlite.conversation._store.local_lineage import advance_local_target
    advance_local_target(conn, parent_session_id, child_session_id)
    if parent["archived"]:
        # A live continuation under an idle-sweep archive re-activates the chat; after the
        # closure above the child is linked into the lineage walk (#117713).
        _unarchive_auto_archived_lineage(conn, child_session_id)


def _write_sql_logged(holder: StoreHolder, op: str, session_id: str, sql: str, params) -> None:
    """``_write_sql`` that logs (never raises) on ``sqlite3.Error``."""
    try:
        holder._write_sql(sql, params)
    except sqlite3.Error as exc:
        logger.warning("%s(%s) failed: %s", op, session_id, exc)


def record_compression_failure_cooldown(
    holder: StoreHolder, session_id: str, cooldown_until: float, error: Optional[str] = None) -> None:
    """Persist the active compression-failure cooldown. Merge-max with any longer live deadline so a
    later shorter write can't reopen the thrash window; error always takes the latest diagnostic."""
    if not session_id:
        return
    _write_sql_logged(holder,
        "record_compression_failure_cooldown", session_id,
        # Merge-max with any longer live deadline so a later shorter write cannot reopen the thrash
        # window (#96775). The error column always takes the latest diagnostic.
        "UPDATE sessions SET compression_failure_cooldown_until = CASE "
        "WHEN compression_failure_cooldown_until IS NOT NULL  AND compression_failure_cooldown_until > ? "
        "THEN compression_failure_cooldown_until ELSE ? END, compression_failure_error = ? WHERE id = ?",
        (cooldown_until, cooldown_until, error, session_id))


def get_compression_failure_cooldown(holder: StoreHolder, session_id: str) -> Optional[dict[str, Any]]:
    """Return the active (unexpired) compression-failure cooldown, or None."""
    now = time.time()
    row = holder._read_one(_COOLDOWN_ROW_SQL, (session_id,)) if session_id else None
    if row is None or row[0] is None or float(row[0]) <= now:
        return None
    return {"cooldown_until": float(row[0]), "remaining_seconds": float(row[0]) - now, "error": row[1]}


def get_compression_failure_cooldown_row(holder: StoreHolder, session_id: str) -> dict[str, Any]:
    """Exact stored cooldown columns, no expiry filtering, so compression cancellation
    can roll back an expired, partially-null, or absent row exactly."""
    row = holder._read_one(_COOLDOWN_ROW_SQL, (session_id,)) if session_id else None
    return _cooldown_row(False, None, None) if row is None else _cooldown_row(True, row[0], row[1])


def restore_compression_failure_cooldown_row(holder: StoreHolder, session_id: str, snapshot: dict[str, Any]) -> None:
    """Restore and verify an exact cooldown-row snapshot. Unlike record/clear this
    rollback API propagates write and verification failures: cancellation must not be
    reported mutation-free when compensation failed. The tolerated exception is a
    session row that vanished mid-attempt — before or after the compensating write —
    since its cooldown died with it and there is nothing left to restore (#106271)."""
    if not snapshot.get("session_exists", False):
        if get_compression_failure_cooldown_row(holder, session_id).get("session_exists", False):
            raise RuntimeError("cannot restore absent compression cooldown row: session now exists")
        return
    deadline = snapshot.get("cooldown_until")
    error = snapshot.get("error")
    def _do(conn):
        cursor = conn.execute(
            "UPDATE sessions SET compression_failure_cooldown_until = ?, "
            "compression_failure_error = ? WHERE id = ?", (deadline, error, session_id))
        return cursor.rowcount == 1
    if not holder._execute_write(_do):
        logger.warning("compression cooldown rollback session missing: %s", session_id)
        return
    actual = get_compression_failure_cooldown_row(holder, session_id)
    expected = _cooldown_row(True, deadline, error)
    if actual != expected:
        if not actual.get("session_exists", False):
            logger.warning("compression cooldown rollback session missing after restore: %s", session_id)
            return
        raise RuntimeError(
            f"compression cooldown rollback verification failed: expected={expected!r}, actual={actual!r}")


def clear_compression_failure_cooldown(holder: StoreHolder, session_id: str) -> None:
    """Clear any persisted compression-failure cooldown for a session."""
    if not session_id:
        return
    _write_sql_logged(holder,
        "clear_compression_failure_cooldown", session_id,
        "UPDATE sessions SET compression_failure_cooldown_until = NULL, "
        "compression_failure_error = NULL WHERE id = ?", (session_id,))


def _read_session_number(holder: StoreHolder, column: str, session_id: str, cast: type, zero: Any) -> Any:
    """Read one numeric ``sessions`` column clamped at ``zero``; a missing session,
    NULL, or unparsable value also reads as ``zero``."""
    row = holder._read_one(f"SELECT {column} FROM sessions WHERE id = ?", (session_id,)) if session_id else None
    try:
        return zero if row is None else max(zero, cast(row[0] or zero))
    except (TypeError, ValueError):
        return zero


def _write_session_column(holder: StoreHolder, column: str, session_id: str, value: Any) -> None:
    holder._write_sql(f"UPDATE sessions SET {column} = ? WHERE id = ?", (value, session_id))


def get_compression_fallback_streak(holder: StoreHolder, session_id: str) -> int:
    """Return the persisted deterministic-fallback streak."""
    return _read_session_number(holder, "compression_fallback_streak", session_id, int, 0)


def set_compression_fallback_streak(holder: StoreHolder, session_id: str, streak: int) -> None:
    """Persist the deterministic-fallback streak for one session."""
    if session_id:
        _write_session_column(holder, "compression_fallback_streak", session_id, max(0, int(streak)))


def get_compression_ineffective_count(holder: StoreHolder, session_id: str) -> int:
    """Persisted ineffective-compaction strike count — the durable half of the built-in
    compressor's anti-thrash guard, so a fresh compressor bound to a resumed session
    inherits an armed/tripped guard across restarts."""
    return _read_session_number(holder, "compression_ineffective_count", session_id, int, 0)


def set_compression_ineffective_count(holder: StoreHolder, session_id: str, count: int) -> None:
    """Persist the ineffective-compaction strike count for one session."""
    if session_id:
        _write_session_column(holder, "compression_ineffective_count", session_id, max(0, int(count)))


def get_compression_recovery_deadline(holder: StoreHolder, session_id: str) -> float:
    """Persisted anti-thrash recovery deadline (epoch; ``0.0`` = not armed). Durable
    because the gateway rebuilds the compressor every turn / cache eviction.

    The deadline is the durable half of the 14694 recovery clock: the gateway rebuilds the compressor on
    every turn / cache eviction, so a process-local deadline restarted the wait on each rebuild and a
    tripped session never earned its probe (#100185).
    """
    return _read_session_number(holder, "compression_recovery_deadline", session_id, float, 0.0)


def set_compression_recovery_deadline(holder: StoreHolder, session_id: str, deadline: float) -> None:
    """Persist the anti-thrash recovery deadline; ``0`` / ``None`` disarms it."""
    if not session_id:
        return
    try:
        normalized = max(0.0, float(deadline or 0.0))
    except (TypeError, ValueError):
        normalized = 0.0
    _write_session_column(holder, "compression_recovery_deadline", session_id, normalized or None)


def get_compression_overload_streak(holder: StoreHolder, session_id: str) -> int:
    """Return the persisted sustained-overload abort streak (#123167)."""
    return _read_session_number(holder, "compression_overload_streak", session_id, int, 0)


def set_compression_overload_streak(holder: StoreHolder, session_id: str, streak: int) -> None:
    """Persist the sustained-overload abort streak for one session."""
    if session_id:
        _write_session_column(holder, "compression_overload_streak", session_id, max(0, int(streak)))


def increment_compression_overload_streak(holder: StoreHolder, session_id: str) -> Optional[int]:
    """Atomically bump the overload streak and return the new value (None when no row).
    One UPDATE ... RETURNING, so concurrent agents on one session cannot lose a strike."""
    if not session_id:
        return None
    def _do(conn):
        row = conn.execute(
            "UPDATE sessions SET compression_overload_streak = compression_overload_streak + 1"
            " WHERE id = ? RETURNING compression_overload_streak", (session_id,)).fetchone()
        return None if row is None else int(row[0])
    return holder._execute_write(_do)


def _session_turn_lease_key_on_conn(holder: StoreHolder, conn, session_id: str) -> str:
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
        if not parent_id or parent_id in seen or _is_explicit_fork_child_row(holder, current, include_reset=True):
            break
        parent = _row(parent_id)
        if not parent or parent.get("end_reason") != "compression":
            break
        seen.add(parent_id)
        current = parent
    return str(current.get("id") or session_id) if current else session_id


def finalize_orphaned_compression_sessions(holder: StoreHolder) -> int:
    """Mark orphaned compression continuations (parent ended by compression; child has
    messages, no end_reason/ended_at, api_call_count=0, older than 7 days) as
    ``orphaned_compression``. Non-destructive.

    Fix for #20001.
    """
    cutoff = time.time() - 604800  # 7 days
    return holder._write_rowcount(
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


def get_compression_chain(holder: StoreHolder, session_id: str) -> list[str]:
    """Walk the compression-continuation chain forward: root-first through the tip (``[session_id]``
    when no continuation); ``get_compression_tip`` is the last element. A continuation is a child of
    a session with ``end_reason='compression'``. The old ``child.started_at >= parent.ended_at`` test
    was too brittle (gateway + compression races insert the real continuation before ``ended_at`` is
    written, while a stale websocket later creates a sibling that passes it). Instead exclude
    branch/delegate/tool children and prefer children that continue the chain or are still live over
    stale closed siblings such as ``ws_orphan_reap``."""
    def step(current: str):
        with holder._read_ctx() as conn:
            return conn.execute(_CHAIN_STEP_SQL, (current,)).fetchone()
    return _walk_compression_chain(session_id, step)


def get_compression_tip(holder: StoreHolder, session_id: str) -> Optional[str]:
    """Live tip of a compression chain (``get_compression_chain`` semantics); the input
    id when no continuation exists."""
    chain = get_compression_chain(holder, session_id)
    return chain[-1] if chain else session_id


def _compression_tip_on(holder: StoreHolder, conn, session_id: str) -> str:
    """``get_compression_tip`` read on a writer's own connection: inside its transaction no rotation
    can publish between finding the tip and writing it."""
    return _walk_compression_chain(session_id, lambda current: conn.execute(_CHAIN_STEP_SQL, (current,)).fetchone())[-1]


def _is_compression_child_row(holder: StoreHolder, child: dict[str, Any]) -> bool:
    parent_id = child.get("parent_session_id")
    # A reset fork of a compression-ended parent is its own conversation, not the continuation (#114271).
    if not parent_id or _is_explicit_fork_child_row(holder, child, include_reset=True):
        return False
    parent = get_session(holder, parent_id)
    return bool(parent and parent.get("end_reason") == "compression")


def get_compression_lineage(holder: StoreHolder, session_id: str) -> list[str]:
    """Return compression ancestors through tip in chronological order."""
    session = get_session(holder, session_id)
    if not session or _is_explicit_fork_child_row(holder, session):
        return [session_id] if session else []
    root = session
    ancestors = {root["id"]}
    while _is_compression_child_row(holder, root):
        parent = get_session(holder, root["parent_session_id"])
        if not parent or parent["id"] in ancestors:
            break
        root = parent
        ancestors.add(root["id"])
    lineage = [root["id"]]
    seen = {root["id"]}
    current = root
    while current.get("end_reason") == "compression":
        rows = holder._read_all(
            """
                SELECT * FROM sessions
                WHERE parent_session_id = ?
                ORDER BY started_at ASC
                """, (current["id"],))
        next_child = next((dict(row) for row in rows if _is_compression_child_row(holder, dict(row))), None)
        if not next_child or next_child["id"] in seen:
            break
        lineage.append(next_child["id"])
        seen.add(next_child["id"])
        current = next_child
    # Later tips are included only when the requested session itself was compacted.
    return lineage if session_id in lineage else [session_id]


def get_tool_rotation_root(holder: StoreHolder, session_id: str) -> str:
    """Root of a ``--source tool`` run's compression chain. ``get_compression_lineage`` treats every ``tool`` row as a
    fork, but rotation copies the parent's source onto its continuation, so climb only hops where both rows are
    ``tool`` and the child is an unmarked continuation of a compression-ended parent. A ``tool`` fork of another
    source's session keeps its own id."""
    row, seen = get_session(holder, session_id), {session_id}
    while row and row.get("source") == "tool" and (parent_id := row.get("parent_session_id")) not in seen:
        parent = get_session(holder, parent_id) if parent_id else None
        if not parent or parent.get("source") != "tool" or not _is_compression_child_row(holder, {**row, "source": None}):
            break
        seen.add(parent_id)
        row = parent
    return row["id"] if row else session_id


# Sibling store modules import this module's copies in turn, so their copies are imported after these.
from persistence.sqlite.conversation._store.core import _session_row_dict
from persistence.sqlite.conversation._store.messages import _carry_parent_timestamps, _clone_message_rows, _execute_transcript_write, _insert_message_rows, _is_explicit_fork_child_row, _tail_rows_after_watermark
from persistence.sqlite.conversation._store.sessions import _NON_CONTINUATION_CHILD_FILTER_SQL, _own_profile_name, _unarchive_auto_archived_lineage, get_session
