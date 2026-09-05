"""Gateway-facing SessionDB persistence: routing index, peers, orphans, heartbeats, handoffs.

Mixin bound onto ``SessionDB`` via the MRO; built on its ``_read_ctx`` /
``_execute_write`` / ``_write_sql`` / ``_read_all`` primitives."""

from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from hermes_state_common import _RECOVERABLE_END_REASONS_SQL, _RESET_END_REASONS_SQL, _sql_session_last_active

# Log-record parity with the origin module (caplog tests pin "hermes_state").
logger = logging.getLogger("hermes_state")

# Recursive CTE naming a session plus its compression ancestors (rows a
# resume must keep on one routing peer); branch/delegate/tool rows stop it.
_COMPRESSION_LINEAGE_CTE = """
                    WITH RECURSIVE compression_lineage(id) AS (
                        SELECT ?
                        UNION
                        SELECT parent.id
                        FROM compression_lineage lineage
                        JOIN sessions child ON child.id = lineage.id
                        JOIN sessions parent ON parent.id = child.parent_session_id
                        WHERE parent.end_reason = 'compression'
                          AND json_extract(
                              COALESCE(child.model_config, '{}'),
                              '$._branched_from'
                          ) IS NULL
                          AND json_extract(
                              COALESCE(child.model_config, '{}'),
                              '$._delegate_from'
                          ) IS NULL
                          AND COALESCE(child.source, '') != 'tool'
                    )
                """

# Projection shared by both peer-recovery queries (exact key, then peer tuple).
_PEER_SELECT_HEAD = """
                SELECT s.*,
                       COALESCE(sp.prompt, s.system_prompt)
                           AS _system_prompt_resolved,
                       (COALESCE(s.message_count, 0) > 0 OR EXISTS (
                           SELECT 1 FROM messages WHERE messages.session_id = s.id LIMIT 1
                       )) AS _has_messages
                FROM sessions s
                LEFT JOIN system_prompts sp ON sp.hash = s.system_prompt_hash
"""
_PEER_BY_KEY_SQL = f"""{_PEER_SELECT_HEAD}                WHERE s.session_key = ?
                  AND s.source = ?
                  AND s.credential_owner IS ?
                  AND (s.ended_at IS NULL OR s.end_reason IN ({_RECOVERABLE_END_REASONS_SQL}))
                  AND NOT EXISTS (
                      SELECT 1 FROM sessions b
                      WHERE b.session_key = s.session_key
                        AND b.source = s.source
                        AND b.credential_owner IS ?
                        AND b.ended_at IS NOT NULL
                        AND b.end_reason IN ({_RESET_END_REASONS_SQL})
                        AND b.ended_at
                            > COALESCE(s.last_activity_at, s.started_at)
                  )
                ORDER BY _has_messages DESC,
                         COALESCE(s.last_activity_at, s.started_at) DESC
                LIMIT 1
                """
_PEER_BY_TUPLE_SQL = f"""{_PEER_SELECT_HEAD}                WHERE s.source = ?
                  AND COALESCE(s.user_id, '') = COALESCE(?, '')
                  AND COALESCE(s.chat_id, '') = COALESCE(?, '')
                  AND COALESCE(s.chat_type, '') = COALESCE(?, '')
                  AND COALESCE(s.thread_id, '') = COALESCE(?, '')
                  AND (? IS NULL OR COALESCE(s.profile_name, ?) = ?)
                  AND (s.ended_at IS NULL OR s.end_reason IN ({_RECOVERABLE_END_REASONS_SQL}))
                  AND (COALESCE(s.message_count, 0) > 0 OR EXISTS (
                      SELECT 1 FROM messages WHERE messages.session_id = s.id LIMIT 1
                  ))
                  AND NOT EXISTS (
                      SELECT 1 FROM sessions b
                      WHERE b.source = s.source
                        AND COALESCE(b.user_id, '') = COALESCE(s.user_id, '')
                        AND COALESCE(b.chat_id, '') = COALESCE(s.chat_id, '')
                        AND COALESCE(b.chat_type, '') = COALESCE(s.chat_type, '')
                        AND COALESCE(b.thread_id, '') = COALESCE(s.thread_id, '')
                        AND b.ended_at IS NOT NULL
                        AND b.end_reason IN ({_RESET_END_REASONS_SQL})
                        AND b.ended_at
                            > COALESCE(s.last_activity_at, s.started_at)
                  )
                ORDER BY COALESCE(s.last_activity_at, s.started_at) DESC
                LIMIT 1
                """

_ORPHAN_DONOR_COLUMNS = (
    "d.id, d.session_key, d.chat_id, d.chat_type, d.thread_id, "
    "d.user_id, d.origin_json, d.display_name, d.end_reason"
)
_ORPHANS_SQL = f"""
                SELECT o.id, o.source, o.user_id, o.started_at,
                       o.parent_session_id,
                       {_sql_session_last_active("o")} AS last_active,
                       (SELECT COUNT(*) FROM messages m
                         WHERE m.session_id = o.id) AS message_count
                FROM sessions o
                WHERE o.session_key IS NULL
                  AND EXISTS (SELECT 1 FROM messages m
                               WHERE m.session_id = o.id)
                  AND COALESCE(o.source, '') != 'tool'
                  AND json_extract(COALESCE(o.model_config, '{{}}'),
                                   '$._branched_from') IS NULL
                  AND json_extract(COALESCE(o.model_config, '{{}}'),
                                   '$._delegate_from') IS NULL
                ORDER BY o.started_at ASC
                """
_ORPHAN_LINEAGE_DONOR_SQL = f"""
                        SELECT {_ORPHAN_DONOR_COLUMNS}
                        FROM sessions d
                        WHERE d.id = ?
                          AND d.session_key IS NOT NULL
                          AND COALESCE(d.source, '') = COALESCE(?, '')
                        """
_ORPHAN_CONTIGUITY_DONORS_SQL = f"""
                        SELECT {_ORPHAN_DONOR_COLUMNS}, {_sql_session_last_active("d")} AS last_active
                        FROM sessions d
                        WHERE d.session_key IS NOT NULL
                          AND d.id != ?
                          AND COALESCE(d.source, '') = COALESCE(?, '')
                          AND (COALESCE(d.user_id, '') = ''
                               OR COALESCE(?, '') = ''
                               OR d.user_id = ?)
                          AND {_sql_session_last_active("d")} BETWEEN ? AND ?
                          AND {_sql_session_last_active("d")} < ?
                        ORDER BY last_active DESC
                        LIMIT 2
                        """
_HANDOFF_FAIL_SQL = "UPDATE sessions SET handoff_state = 'failed', handoff_error = ? WHERE "


class SessionGatewayMixin:
    """Routing index, session peers/orphans, hygiene streaks, heartbeats, handoffs."""

    def _reap_inactive_orphan_desktop_holders(
        self, holders: List[Tuple[int, str]], *, min_age_seconds: float) -> List[int]:
        """Terminate old PPID-1 Desktop ephemeral backends with no client.

        Fails closed: anything whose parent, age, argv, or network connections
        cannot be proved safe remains a repair-blocking holder."""
        from hermes_state import psutil
        from hermes_state_dbfile import _concrete_state_db_holder_pids, _is_inactive_orphan_desktop_holder
        if not sys.platform.startswith("linux") or psutil is None:
            return []
        try:
            from hermes_cli.dashboard_procs import _is_ephemeral_port_zero_backend
        except Exception:
            return []
        now = time.time()
        candidates = []
        for pid in _concrete_state_db_holder_pids(self.db_path, holders):
            try:
                process = psutil.Process(pid)
                statuses = [conn.status for conn in process.net_connections(kind="inet")]
                if not _is_inactive_orphan_desktop_holder(
                    ppid=process.ppid(), age_seconds=now - process.create_time(),
                    min_age_seconds=min_age_seconds,
                    ephemeral_backend=_is_ephemeral_port_zero_backend(process.cmdline()),
                    connection_statuses=statuses):
                    continue
            except Exception:
                continue
            candidates.append(process)
        signalled: List[int] = []
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
        except Exception:
            alive = []
        for process in alive:
            try:
                process.kill()
            except (psutil.Error, OSError):
                continue
        if alive:
            try:
                psutil.wait_procs(alive, timeout=1.5)
            except Exception:
                pass
        return signalled

    def record_gateway_session_peer(
        self, session_id: str, *, source: str, user_id: str = None, session_key: str = None,
        credential_owner: str = None,
        chat_id: str = None, chat_type: str = None, thread_id: str = None, display_name: str = None,
        origin_json: str = None, include_compression_ancestors: bool = False) -> None:
        """Persist the gateway routing peer for an existing session row. ``display_name`` / ``origin_json``:
        ``None`` leaves the stored value untouched (consumers read routing data from state.db, not
        sessions.json). ``include_compression_ancestors`` keeps a compression lineage on one routing peer
        when an explicit resume moves its tip to another lane; per-turn refreshes update only the supplied
        row. Self-healing: a missing target row (deferred ``create_session`` write, or crash between routing
        publication and row creation) is INSERTed with full identity rather than no-opped, so a gateway row
        is never first-created by the identity-less lazy writer (``update_token_counts``) and left
        unroutable forever.

        See #9006.
        """
        if not session_id or not session_key:
            return
        identity = (
            session_key,
            source,
            user_id,
            credential_owner,
            chat_id,
            chat_type,
            thread_id,
            display_name,
            origin_json,
        )
        ancestors = include_compression_ancestors
        query_params = [session_id, *identity] if ancestors else [*identity, session_id]
        def _do(conn):
            conn.execute(
                f"""{_COMPRESSION_LINEAGE_CTE if ancestors else ""}
                    UPDATE sessions
                    SET session_key = ?, source = ?, user_id = ?,
                        credential_owner = COALESCE(?, credential_owner), chat_id = ?,
                        chat_type = ?, thread_id = ?,
                        display_name = COALESCE(?, display_name),
                        origin_json = COALESCE(?, origin_json)
                   {"WHERE id IN (SELECT id FROM compression_lineage)" if ancestors else "WHERE id = ?"}""",
                query_params,
            )
            if ancestors:
                return
            # The UPDATE silently no-ops on a missing row — insert it with full identity.
            if conn.execute("SELECT 1 FROM sessions WHERE id = ? LIMIT 1", (session_id,)).fetchone() is None:
                conn.execute(
                    """INSERT INTO sessions (
                               id, source, user_id, credential_owner, session_key, chat_id,
                               chat_type, thread_id, display_name, origin_json,
                               profile_name, started_at
                            )
                           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                           ON CONFLICT(id) DO UPDATE SET
                                session_key = COALESCE(sessions.session_key, excluded.session_key),
                                chat_id = COALESCE(sessions.chat_id, excluded.chat_id),
                               chat_type = COALESCE(sessions.chat_type, excluded.chat_type),
                               thread_id = COALESCE(sessions.thread_id, excluded.thread_id),
                               display_name = COALESCE(sessions.display_name, excluded.display_name),
                               origin_json = COALESCE(sessions.origin_json, excluded.origin_json)""",
                    # Same ownership stamp as _insert_session_row: an unowned (NULL) row
                    # vanishes from profile-keyed consumers.
                    (
                        session_id,
                        source,
                        user_id,
                        credential_owner,
                        session_key,
                        chat_id,
                        chat_type,
                        thread_id,
                        display_name,
                        origin_json,
                        self._own_profile_name(),
                        time.time(),
                    ),
                )
        self._execute_write(_do)

    def save_gateway_routing_entry(self, session_key: str, entry_json: str, *, scope: str = "") -> None:
        """Upsert one gateway routing entry (session_key -> SessionEntry JSON); ``scope``
        namespaces the index per sessions_dir so two stores never share routing state."""
        if not session_key or not entry_json:
            return
        self._write_sql(
            """INSERT INTO gateway_routing (scope, session_key, entry_json, updated_at)
               VALUES (?, ?, ?, ?)
               ON CONFLICT(scope, session_key) DO UPDATE SET
                   entry_json = excluded.entry_json,
                   updated_at = excluded.updated_at""",
            (scope, session_key, entry_json, time.time()),
        )

    def replace_gateway_routing_entries(self, entries: Dict[str, str], *, scope: str = "") -> None:
        """Atomically replace the routing index for *scope* (keys absent from *entries*
        are removed); other scopes untouched."""
        now = time.time()
        def _do(conn):
            conn.execute("DELETE FROM gateway_routing WHERE scope = ?", (scope,))
            if entries:
                conn.executemany(
                    "INSERT INTO gateway_routing (scope, session_key, entry_json, updated_at) "
                    "VALUES (?, ?, ?, ?)",
                    [(scope, k, v, now) for k, v in entries.items() if k and v])
        self._execute_write(_do)

    def load_gateway_routing_entries(self, *, scope: str = "") -> Dict[str, str]:
        """Load routing entries for *scope* as {session_key: entry_json}."""
        rows = self._read_all("SELECT session_key, entry_json FROM gateway_routing WHERE scope = ?", (scope,))
        return {r["session_key"]: r["entry_json"] for r in rows}

    def list_never_active_keyed_sessions(self, *, older_than_days: float) -> List[Dict[str, Any]]:
        """Keyed, still-open rows with no evidence of a single turn (no messages, tokens, tool/API calls,
        activity, or title): leaked fixtures or chats routed but never answered. Safe to drop — the gateway
        mints a fresh session on the next message. Needs its own selector because ``bulk prune``/``archive``
        are pinned to ``ended_at IS NOT NULL``. ``pinned``/``archived`` = explicit keep intent.

        That is exactly the shape of a leaked test fixture (#82770) — and also of a chat that was routed but
        never answered.
        """
        cutoff = time.time() - (float(older_than_days) * 86400.0)
        rows = self._read_all(
            """
            SELECT s.id, s.session_key, s.source, s.chat_id,
                   s.chat_type, s.user_id, s.started_at
              FROM sessions s
             WHERE s.session_key IS NOT NULL
               AND s.ended_at IS NULL
               AND s.title IS NULL
               AND s.last_activity_at IS NULL
               AND COALESCE(s.message_count, 0) = 0
               AND COALESCE(s.tool_call_count, 0) = 0
               AND COALESCE(s.api_call_count, 0) = 0
               AND COALESCE(s.input_tokens, 0) = 0
               AND COALESCE(s.output_tokens, 0) = 0
               AND COALESCE(s.pinned, 0) = 0
               AND COALESCE(s.archived, 0) = 0
               AND s.started_at IS NOT NULL
               AND s.started_at < ?
               AND NOT EXISTS (
                       SELECT 1 FROM messages m WHERE m.session_id = s.id
                   )
             ORDER BY s.started_at
            """,
            (cutoff,),
        )
        return [dict(r) for r in rows]

    def _delete_routing_entries_for_sessions(self, session_ids: Set[str]) -> int:
        """Drop ``gateway_routing`` rows pointing at any of *session_ids*; the target id
        lives only inside ``entry_json``, so matching is done in Python over all scopes."""
        if not session_ids:
            return 0
        doomed: List[Tuple[str, str]] = []
        for row in self._read_all("SELECT scope, session_key, entry_json FROM gateway_routing"):
            try:
                entry = json.loads(row["entry_json"] or "{}")
            except Exception:
                continue
            if isinstance(entry, dict) and entry.get("session_id") in session_ids:
                doomed.append((row["scope"], row["session_key"]))
        if not doomed:
            return 0
        self._write_sql("DELETE FROM gateway_routing WHERE scope = ? AND session_key = ?", doomed, many=True)
        return len(doomed)

    def prune_never_active_keyed_sessions(
        self, *, older_than_days: float, sessions_dir: Optional[Path] = None) -> Tuple[int, int]:
        """Delete never-active keyed rows and the routing entries naming them; returns
        ``(sessions_deleted, routing_entries_deleted)``. Routing entries go first: a stale
        entry outliving its target would have the gateway resume a nonexistent id.
        Deletion goes through :meth:`delete_session` (delegate cascade, FTS, transcripts)."""
        candidates = self.list_never_active_keyed_sessions(older_than_days=older_than_days)
        if not candidates:
            return (0, 0)
        ids = {str(row["id"]) for row in candidates}
        routing_deleted = self._delete_routing_entries_for_sessions(ids)
        deleted = sum(1 for sid in ids if self.delete_session(sid, sessions_dir=sessions_dir))
        return (deleted, routing_deleted)

    def list_gateway_sessions(
        self, *, platform: Optional[str] = None, active_only: bool = True) -> List[Dict[str, Any]]:
        """List gateway sessions (rows with a session_key): newest row per key, one live
        mapping per routing key. ``platform`` filters on ``source``."""
        # Full rows carry token/cost totals — drain queued async accounting deltas first.
        self.flush_token_counts()
        query = f"""
            SELECT sessions.*,
                   COALESCE(sp.prompt, sessions.system_prompt)
                       AS _system_prompt_resolved,
                   {_sql_session_last_active("sessions")} AS last_active
            FROM sessions
            LEFT JOIN system_prompts sp
              ON sp.hash = sessions.system_prompt_hash
            WHERE session_key IS NOT NULL
              AND started_at = (
                  SELECT MAX(s2.started_at) FROM sessions s2
                  WHERE s2.session_key = sessions.session_key
              )
        """
        params: list = [platform] if platform else []
        query += (" AND LOWER(source) = LOWER(?)" if platform else "") + (
            " AND ended_at IS NULL" if active_only else "") + " ORDER BY last_active DESC"
        return [self._session_row_dict(r) for r in self._read_all(query, params)]

    def find_latest_gateway_session_for_peer(
        self, *, source: str, user_id: Optional[str] = None, session_key: Optional[str] = None,
        chat_id: Optional[str] = None, chat_type: Optional[str] = None, thread_id: Optional[str] = None,
        credential_owner: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """Find the latest recoverable gateway session for a routing peer. The durable ``session_key`` on the row rebuilds a missing/pruned ``sessions.json`` mapping. Rows
        ended only by the old ``agent_close`` bug or a mistaken TUI ``ws_orphan_reap`` are recoverable;
        explicit boundaries (/new, /resume switches, compression splits) are not. Ranked by
        ``last_activity_at`` (fallback ``started_at`` — alone it resurrected days-old zombies); rows with
        messages win, but an empty keyed row still beats ``None`` (which mints a new id while the transcript
        may live under a compression child). Reset fence: a candidate is rejected when a peer boundary row
        ended *after* its last activity, or the has-messages ranking could reach behind a /new. The
        exact-key fallback requires the complete peer tuple (never cross chats/threads/users) plus a profile
        fence: a Telegram DM's tuple is identical for every bot, so a sibling profile's legacy row would
        otherwise be adopted. Ours = profile_name is the owner or NULL; stores outside the profile tree
        derive no owner and stay unfenced.

        See #60609.
        Reset boundaries fence recovery (#68539): an intentional boundary such as ``session_reset`` (or any
        explicit non-recoverable end_reason) must block fallback to an *older* row for the same peer. Each
        candidate is therefore rejected when a boundary row for the peer ended *after* the candidate's last
        activity — if the conversation's most recent event is an intentional reset, recovery returns nothing
        rather than reaching behind it.
        """
        if not session_key:
            return None
        with self._read_ctx() as conn:
            row = conn.execute(
                _PEER_BY_KEY_SQL,
                (
                    session_key,
                    source,
                    credential_owner,
                    credential_owner,
                ),
            ).fetchone()
            if row is not None:
                return self._session_row_dict(row)
            if chat_id is None or chat_type is None:
                return None
            # Profile fence (#74285): a Telegram DM's peer tuple is identical for every bot (chat_id ==
            # user_id, no thread), so a sibling profile's row written into this store before the per-profile
            # partition (legacy data) would otherwise be adopted here. Every profile-tree store has one
            # owner; a row is ours when its profile_name is the owner or NULL (legacy rows this store
            # minted). Stores outside the tree derive no owner and keep the historical unfenced behavior.
            owner = self._own_profile_name()
            row = conn.execute(
                _PEER_BY_TUPLE_SQL, (source, user_id, chat_id, chat_type, thread_id, owner, owner, owner)
            ).fetchone()
        return self._session_row_dict(row) if row else None

    def find_orphaned_gateway_sessions(self, *, max_gap_s: Optional[float] = None) -> List[Dict[str, Any]]:
        """Report message-bearing rows that lost their routing identity (messages, no ``session_key``).
        Adoptable only when exactly one keyed predecessor can be named: ``lineage`` (``parent_session_id``
        is a keyed row of the same source; no time window) or ``contiguity`` (exactly one keyed same-source
        row with compatible ``user_id`` fell quiet within *max_gap_s* of the orphan's start and is older
        than its last activity). Ambiguity is reported ``adoptable=False`` with a reason, never guessed —
        mis-adopting splices one person's conversation into another's chat. Branch/delegate/tool rows are
        excluded: unkeyed by design, not damage."""
        gap = self._ORPHAN_ADOPTION_MAX_GAP_S if max_gap_s is None else float(max_gap_s)
        records: List[Dict[str, Any]] = []
        with self._read_ctx() as conn:
            for orphan in conn.execute(_ORPHANS_SQL).fetchall():
                donor = None
                reason = ""
                if orphan["parent_session_id"]:
                    evidence = "lineage"
                    donor = conn.execute(
                        _ORPHAN_LINEAGE_DONOR_SQL, (orphan["parent_session_id"], orphan["source"])).fetchone()
                    if donor is None:
                        reason = "parent session carries no gateway identity of this source"
                else:
                    evidence = "contiguity"
                    started = orphan["started_at"] or 0
                    candidates = conn.execute(
                        _ORPHAN_CONTIGUITY_DONORS_SQL,
                        (orphan["id"], orphan["source"], orphan["user_id"], orphan["user_id"],
                         started - gap, started + gap, orphan["last_active"]),
                    ).fetchall()
                    if not candidates:
                        reason = f"no keyed predecessor fell quiet within {gap:.0f}s of this session's start"
                    elif len(candidates) > 1:
                        reason = "ambiguous: more than one keyed predecessor matches this window"
                    else:
                        donor = candidates[0]
                records.append({
                    "orphan_id": orphan["id"], "source": orphan["source"],
                    "message_count": orphan["message_count"], "started_at": orphan["started_at"],
                    "last_active": orphan["last_active"], "donor_id": donor["id"] if donor else None,
                    "session_key": donor["session_key"] if donor else None,
                    "evidence": evidence if donor else "", "adoptable": donor is not None, "reason": reason})
        # Two unkeyed successors claiming one predecessor: at most one continues that chat.
        contested = {
            r["donor_id"] for r in records
            if r["adoptable"] and sum(1 for x in records if x["donor_id"] == r["donor_id"]) > 1
        }
        for record in records:
            if record["donor_id"] in contested:
                record["adoptable"] = False
                record["reason"] = "ambiguous: more than one unkeyed session claims this predecessor"
        return records

    def adopt_orphaned_gateway_session(self, orphan_id: str, donor_id: str) -> bool:
        """Stamp *orphan_id* with *donor_id*'s routing identity, retire *donor_id*.
        Re-verifies the pair inside the write txn so a concurrent gateway that healed
        either row makes this a no-op. Non-NULL orphan columns are preserved."""
        if not orphan_id or not donor_id or orphan_id == donor_id:
            return False
        def _do(conn):
            donor = conn.execute(
                "SELECT session_key, chat_id, chat_type, thread_id, user_id, "
                "origin_json, display_name, source FROM sessions WHERE id = ?",
                (donor_id,),
            ).fetchone()
            orphan = conn.execute(
                "SELECT session_key, source FROM sessions WHERE id = ?", (orphan_id,)).fetchone()
            if (donor is None or orphan is None or not donor["session_key"] or orphan["session_key"]
                    or (donor["source"] or "") != (orphan["source"] or "")):
                return False
            # Belt-and-suspenders for gateway routing metadata (#59527): the gateway re-records the peer on
            # the child after rotation (d5b4879d4), but a hard crash between child creation and that write
            # leaves the child row without origin columns, so ``find_latest_gateway_session_for_peer`` can't
            # recover the mapping on restart. Inherit them from the parent at creation time — but ONLY for
            # compression forks (parent already ended with end_reason='compression'). Delegate/subagent
            # children are spawned while the parent is still live and must NOT inherit routing keys, or peer
            # recovery could repoint gateway traffic into a subagent's session.
            conn.execute(
                """UPDATE sessions
                      SET session_key = ?,
                          chat_id = COALESCE(chat_id, ?),
                          chat_type = COALESCE(chat_type, ?),
                          thread_id = COALESCE(thread_id, ?),
                          user_id = COALESCE(user_id, ?),
                          origin_json = COALESCE(origin_json, ?),
                          display_name = COALESCE(display_name, ?),
                          parent_session_id = COALESCE(parent_session_id, ?)
                    WHERE id = ? AND session_key IS NULL""",
                (donor["session_key"], donor["chat_id"], donor["chat_type"], donor["thread_id"],
                 donor["user_id"], donor["origin_json"], donor["display_name"], donor_id, orphan_id),
            )
            # Retire under a reason recovery does NOT treat as resumable — 'agent_close' /
            # 'ws_orphan_reap' would keep it in the running and the orphan could lose the chat again.
            conn.execute(
                "UPDATE sessions SET ended_at = COALESCE(ended_at, ?), "
                "end_reason = 'superseded_by_repair' WHERE id = ?",
                (time.time(), donor_id))
            return True
        return self._execute_write(_do)

    def increment_hygiene_failure_streak(self, session_key: str) -> int:
        """Atomically increment the session-hygiene failure streak for one chat."""
        if not session_key:
            return 1
        def _do(conn):
            conn.execute(
                """INSERT INTO gateway_hygiene_state (session_key, failure_streak)
                   VALUES (?, 1)
                   ON CONFLICT(session_key) DO UPDATE SET
                       failure_streak = gateway_hygiene_state.failure_streak + 1""",
                (session_key,),
            )
            row = conn.execute(
                "SELECT failure_streak FROM gateway_hygiene_state WHERE session_key = ?", (session_key,),
            ).fetchone()
            return int(row[0])
        return self._execute_write(_do)

    def reset_hygiene_failure_streak(self, session_key: str) -> None:
        """Clear the persisted session-hygiene failure streak for one chat."""
        if not session_key:
            return
        self._write_sql("DELETE FROM gateway_hygiene_state WHERE session_key = ?", (session_key,))

    @staticmethod
    def session_gateway_runtime(session_meta: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Read the persisted runtime route off a session row dict (``model_config`` as
        JSON string or parsed dict). Precedence: nested ``gateway_runtime`` (gateway sync /
        CLI ``/model``), then top-level ``provider``/``base_url``/``api_mode`` (TUI), then
        ``billing_provider`` so sessions that never ran ``/model`` still restore the
        provider that served them. Empty dict on parse failure — resume uses ambient config."""
        from hermes_state import _BARE_BILLING_PROVIDERS
        raw = (session_meta or {}).get("model_config")
        if isinstance(raw, str):
            try:
                raw = json.loads(raw)
            except Exception:
                raw = {}
        if not isinstance(raw, dict):
            raw = {}
        runtime = raw.get("gateway_runtime")
        # Filter None: the persist path writes or-None to trigger deletion in the top-level
        # merge, but gateway_runtime is replaced whole (not deep-merged), so None survives here.
        if isinstance(runtime, dict) and runtime.get("provider"):
            return {k: v for k, v in runtime.items() if v is not None}
        top_level = {key: raw.get(key) for key in ("provider", "base_url", "api_mode") if raw.get(key)}
        if top_level:
            return top_level
        # billing_provider is COALESCE-written on the first accounted API call — the only durable
        # record for sessions that never ran /model. Bare buckets ("auto"/"custom") are not
        # routable identities; filter them so resume falls back to the ambient default.
        billing_provider = str((session_meta or {}).get("billing_provider") or "").strip()
        if billing_provider and billing_provider.lower() not in _BARE_BILLING_PROVIDERS:
            return {"provider": billing_provider}
        if not isinstance(runtime, dict):
            return {}
        return {k: v for k, v in runtime.items() if v is not None}

    def register_backend_heartbeat(
        self, *, backend_id: str, pid: int, started_at: float, last_heartbeat: Optional[float] = None,
        profile: str = "", host: str = "") -> None:
        """Upsert this backend's liveness row. ``backend_id`` MUST be stable for the process
        lifetime (e.g. ``f"{profile}@{host}:{pid}"``) so a respawn cannot inherit a dead
        predecessor's heartbeat; ``started_at`` is when THIS process started, so a backend
        whose previous run died is not mistaken for a freshly-spawned sibling."""
        if not backend_id:
            return
        ts = time.time() if last_heartbeat is None else float(last_heartbeat)
        self._write_sql(
            "INSERT INTO gateway_heartbeats (backend_id, pid, started_at, last_heartbeat, profile, host)"
            " VALUES (?, ?, ?, ?, ?, ?) ON CONFLICT(backend_id) DO UPDATE SET pid = excluded.pid,"
            " started_at = excluded.started_at, last_heartbeat = excluded.last_heartbeat,"
            " profile = excluded.profile, host = excluded.host",
            (str(backend_id), int(pid), float(started_at), ts, str(profile), str(host)))

    def clear_backend_heartbeat(self, backend_id: str) -> bool:
        """Remove this backend's heartbeat row (from ``atexit``); True if removed. A crashed
        backend's row is reclaimed later by ``prune_stale_heartbeats``."""
        if not backend_id:
            return False
        return self._write_rowcount(
            "DELETE FROM gateway_heartbeats WHERE backend_id = ?", (str(backend_id),)) > 0

    def prune_stale_heartbeats(self, *, max_age_seconds: float) -> List[str]:
        """Drop heartbeat rows older than the staleness window; return removed backend ids.
        Safe from any process — only stale rows are touched."""
        if max_age_seconds <= 0:
            return []
        cutoff = time.time() - max_age_seconds
        def _do(conn):
            cur = conn.execute(
                "DELETE FROM gateway_heartbeats WHERE last_heartbeat < ? RETURNING backend_id",
                (cutoff,))
            return [str(r[0]) for r in cur.fetchall()]
        return list(self._execute_write(_do) or [])

    def list_backend_heartbeats(self) -> List[Dict[str, Any]]:
        """Snapshot of every backend heartbeat (diagnostics/tests); fields mirror the table."""
        rows = self._read_all(
            "SELECT backend_id, pid, started_at, last_heartbeat, profile, host FROM gateway_heartbeats"
            " ORDER BY last_heartbeat DESC")
        return [dict(r) for r in rows]

    def request_handoff(self, session_id: str, platform: str) -> bool:
        """Mark a session pending handoff to *platform*; False if a handoff is already in flight."""
        return self._write_rowcount(
            "UPDATE sessions SET handoff_state = 'pending',     handoff_platform = ?, "
            "    handoff_error = NULL WHERE id = ? AND (handoff_state IS NULL "
            "                  OR handoff_state IN ('completed', 'failed'))",
            (platform, session_id),
        ) > 0

    def get_handoff_state(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Return ``{"state", "platform", "error"}`` or None if the session has no handoff record."""
        try:
            row = self._read_one(
                "SELECT handoff_state, handoff_platform, handoff_error FROM sessions WHERE id = ?",
                (session_id,))
            if not row:
                return None
            return {"state": row["handoff_state"], "platform": row["handoff_platform"],
                    "error": row["handoff_error"]}
        except Exception:
            return None

    def list_pending_handoffs(self) -> List[Dict[str, Any]]:
        """All sessions in handoff_state='pending', oldest first (gateway handoff watcher)."""
        try:
            rows = self._read_all(
                "SELECT s.*, COALESCE(sp.prompt, s.system_prompt) AS _system_prompt_resolved FROM sessions s "
                "LEFT JOIN system_prompts sp ON sp.hash = s.system_prompt_hash "
                "WHERE s.handoff_state = 'pending' "
                "ORDER BY s.started_at ASC")
            return [self._session_row_dict(r) for r in rows]
        except Exception:
            return []

    def claim_handoff(self, session_id: str) -> bool:
        """Atomically transition pending → running. Returns True if claimed."""
        return self._write_rowcount(
            "UPDATE sessions SET handoff_state = 'running' WHERE id = ? AND handoff_state = 'pending'",
            (session_id,),
        ) > 0

    def complete_handoff(self, session_id: str) -> None:
        """Mark a handoff as completed."""
        self._write_sql(
            "UPDATE sessions SET handoff_state = 'completed', handoff_error = NULL WHERE id = ?",
            (session_id,))

    def fail_handoff(
        self, session_id: str, error: str, *, only_states: Optional[Tuple[str, ...]] = None) -> bool:
        """Mark a handoff failed and record the reason; True when a row transitioned. ``only_states`` makes
        the write a compare-and-swap on ``handoff_state``. Waiters that give up (CLI 60s poll, Desktop
        bounded poll) MUST pass ``only_states=("pending",)``: once the watcher has claimed the row
        (``running``) it owns the terminal state, and an unconditional waiter-side fail races the dispatch —
        the gateway later overwrites ``failed`` → ``completed`` after the user was told the gateway is down
        (split-brain: the handoff delivered and ``switch_session`` re-pointed the session). The watcher
        fails its OWN claimed row unconditionally."""
        states = tuple(only_states) if only_states else ()
        sql = _HANDOFF_FAIL_SQL + "id = ?" + (
            f" AND handoff_state IN ({', '.join('?' for _ in states)})" if states else "")
        return self._write_rowcount(sql, (error[:500], session_id, *states)) > 0

    def reclaim_stale_running_handoffs(self, error: str) -> List[str]:
        """Fail every handoff stuck in ``running``; returns the ids reclaimed. Only the gateway watcher sets
        ``running``, for one in-process dispatch — so a ``running`` row at watcher startup belongs to a
        PREVIOUS gateway that died mid-dispatch. It is poisonous: ``request_handoff`` only accepts
        NULL/``completed``/``failed``, so the session could never hand off again, with no error surfaced.
        Failing rather than re-queueing is deliberate: the dead gateway may already have switched the
        session key and dispatched the synthetic turn, so a blind retry risks double delivery; a clean
        terminal state the user can retry from is right."""
        def _do(conn):
            cur = conn.execute("SELECT id FROM sessions WHERE handoff_state = 'running'")
            ids = [r[0] for r in cur.fetchall()]
            if ids:
                conn.execute(_HANDOFF_FAIL_SQL + "handoff_state = 'running'", (error[:500],))
            return ids
        try:
            return self._execute_write(_do) or []
        except Exception:
            # Swallow but never silently: a persistently failing reclaim leaves poisonous
            # 'running' rows in place, so the operator needs a trace.
            logger.warning(
                "reclaim_stale_running_handoffs failed; stranded 'running' "
                "handoff rows (if any) were left in place", exc_info=True)
            return []
