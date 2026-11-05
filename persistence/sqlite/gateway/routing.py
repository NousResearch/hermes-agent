"""``RoutingRepository`` on SQLite: the gateway routing index, the routing columns of the bound store's ``sessions``
rows, orphan repair, never-active pruning and the Telegram DM topic tables, each verb one transaction (or a fixed
sequence of them) through the store holder."""

from __future__ import annotations

import contextlib
import json
import logging
import sqlite3
import time
from collections.abc import Iterable, Mapping
from typing import Any, Optional

from persistence.sqlite._state.errors import StateDbReplacedError
from persistence import JsonValue, SessionId, SessionRowMissing
from persistence.core.artifacts import SessionArtifacts
from persistence.gateway.routing import (
    ORPHAN_ADOPTION_MAX_GAP_S, NeverActivePrune, NeverActiveSession, OrphanReport, RecoveredPeerSession,
    TopicAlreadyLinked, TopicBinding, normalize_topic_profile_name,
)

from .._backend import SQLiteStore, register_repository, translated
from ..conversation._store.core import _session_row_dict
from ..conversation._store.local import owned_lineage_ids
from ..conversation._store.sessions import _own_profile_name
from ..conversation.sessions import session_record
from ._local_lineage import hand_off_conversations, moved_receipts
from ._routes import (
    _COMPRESSION_LINEAGE_CTE, _ORPHAN_CONTIGUITY_DONORS_SQL, _ORPHAN_LINEAGE_DONOR_SQL, _ORPHANS_SQL,
    _PEER_BY_KEY_SQL, _PEER_BY_TUPLE_SQL, ROUTE_RETIRED_SQL, apply_telegram_topic_migration, fence_shared_routes,
    route_retired_params,
)

__all__ = ["SQLiteRoutingRepository"]

# caplog tests pin the "hermes_state" logger name.
logger = logging.getLogger("hermes_state")


class SQLiteRoutingRepository:
    """Routing index, routing peers and Telegram DM topic state of the bound profile's ``state.db``."""

    def __init__(self, store: SQLiteStore) -> None:
        self._store = store

    def _own_profile_name(self) -> Optional[str]:
        return _own_profile_name(self._store.holder())

    def _topic_read(self, read, empty):
        """Run ``read()``; an absent table reads as ``empty``. A pre-v3 table left by an upgrade
        (#103363) raises ``no such column: profile_name``: heal it and retry, otherwise the
        write-only migration is never reached and topic mode silently reads as off forever."""
        try:
            return read()
        except sqlite3.OperationalError as exc:
            if "no such column: profile_name" not in str(exc):
                return empty
        try:
            apply_telegram_topic_migration(self._store.holder())
        except (sqlite3.Error, StateDbReplacedError):
            logger.warning("telegram topic tables are pre-v3 and the heal failed; reading as empty", exc_info=True)
            return empty
        return read()

    def _topic_read_one(self, sql: str, params):
        return self._topic_read(lambda: self._store.holder()._read_one(sql, params), None)

    def _topic_read_all(self, sql: str, params):
        return self._topic_read(lambda: self._store.holder()._read_all(sql, params), [])

    def _routing_entries_for_sessions(self, session_ids: set[str]) -> list[tuple[str, str, str]]:
        """``(scope, session_key, session_id)`` of ``gateway_routing`` rows pointing at any of
        *session_ids*; the target id lives only inside ``entry_json``, so matching is in Python."""
        found: list[tuple[str, str, str]] = []
        if not session_ids:
            return found
        for row in self._store.holder()._read_all("SELECT scope, session_key, entry_json FROM gateway_routing"):
            try:
                entry = json.loads(row["entry_json"] or "{}")
            except (ValueError, TypeError, RecursionError):  # malformed / non-text / over-nested entry
                continue
            if isinstance(entry, dict) and entry.get("session_id") in session_ids:
                found.append((row["scope"], row["session_key"], entry["session_id"]))
        return found

    def _delete_routing_entries_for_sessions(self, session_ids: set[str]) -> int:
        """Drop ``gateway_routing`` rows pointing at any of *session_ids*."""
        doomed = [(scope, key) for scope, key, _ in self._routing_entries_for_sessions(session_ids)]
        if not doomed:
            return 0
        self._store.holder()._write_sql(
            "DELETE FROM gateway_routing WHERE scope = ? AND session_key = ?", doomed, many=True)
        return len(doomed)

    # ── writes ──────────────────────────────────────────────────────────────

    def upsert_routing_entry(self, session_key: str, entry_json: str, *, routing_scope: str = "") -> None:
        if not session_key or not entry_json:
            return
        with translated():
            holder = self._store.holder()
            holder._write_sql(
                f"""INSERT INTO gateway_routing (scope, session_key, entry_json, updated_at)
                   SELECT :scope, :key, :entry, :now WHERE NOT {ROUTE_RETIRED_SQL}
                   ON CONFLICT(scope, session_key) DO UPDATE SET
                       entry_json = excluded.entry_json,
                       updated_at = excluded.updated_at""",
                {'scope': routing_scope, 'key': session_key,
                 **route_retired_params(session_key, entry_json, time.time())},
            )

    def replace_routing_entries(self, entries: Mapping[str, str], *, routing_scope: str = "") -> set[str]:
        now = time.time()
        rows = [{'scope': routing_scope, 'key': k, **route_retired_params(k, v, now)}
                for k, v in entries.items() if k and v]

        def _do(conn):
            conn.execute("DELETE FROM gateway_routing WHERE scope = ?", (routing_scope,))
            if rows:
                conn.executemany(
                    "INSERT INTO gateway_routing (scope, session_key, entry_json, updated_at) "
                    "SELECT :scope, :key, :entry, :now WHERE NOT " + ROUTE_RETIRED_SQL,
                    rows)
            kept = {r[0] for r in conn.execute(
                "SELECT session_key FROM gateway_routing WHERE scope = ?", (routing_scope,))}
            return {row['key'] for row in rows} - kept
        with translated():
            holder = self._store.holder()
            return holder._execute_write(_do)

    def fence_retired_routes(self, session_ids: Iterable[str], *, namespace: str, expiring: Iterable[str] = ()) -> None:
        ids = sorted({str(sid) for sid in session_ids if sid})
        if not ids:
            return

        def _do(conn):
            fence_shared_routes(conn, namespace, ids, expiring=expiring)
        with translated():
            holder = self._store.holder()
            holder._execute_write(_do)

    def record_session_peer(
        self, session_id: str, *, source: str, user_id: str | None = None, session_key: str | None = None,
        chat_id: str | None = None, chat_type: str | None = None, thread_id: str | None = None,
        display_name: str | None = None, origin_json: str | None = None, include_compression_ancestors: bool = False,
        messaging_lane: bool = False, transport_profile: str | None = None,
    ) -> None:
        if not session_id or not session_key:
            return
        identity = (
            session_key, source, user_id, chat_id, chat_type, thread_id, display_name, origin_json,
            transport_profile)
        ancestors = include_compression_ancestors
        query_params = [session_id, *identity] if ancestors else [*identity, session_id]

        def _do(conn):
            conn.execute(
                f"""{_COMPRESSION_LINEAGE_CTE if ancestors else ""}
                   UPDATE sessions
                   SET session_key = ?, source = ?, user_id = ?, chat_id = ?,
                       chat_type = ?, thread_id = ?,
                       display_name = COALESCE(?, display_name),
                       origin_json = COALESCE(?, origin_json),
                       transport_profile = COALESCE(?, transport_profile)
                   {"WHERE id IN (SELECT id FROM compression_lineage)" if ancestors else "WHERE id = ?"}""",
                query_params,
            )
            if ancestors:
                # An explicit repoint to a messaging lane moved these rows: the local conversation whose receipt
                # names one of them goes with it in the same write, or it would match neither lane.
                if messaging_lane:
                    moved = {row[0] for row in conn.execute(
                        f"{_COMPRESSION_LINEAGE_CTE} SELECT id FROM compression_lineage", (session_id,))}
                    hand_off_conversations(conn, moved_receipts(conn, moved))
                return
            # The UPDATE silently no-ops on a missing row: insert it with full identity.
            if conn.execute("SELECT 1 FROM sessions WHERE id = ? LIMIT 1", (session_id,)).fetchone() is None:
                conn.execute(
                    """INSERT INTO sessions (
                               id, source, created_source, user_id, session_key, chat_id,
                               chat_type, thread_id, display_name, origin_json,
                               profile_name, transport_profile, started_at
                           )
                           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                           ON CONFLICT(id) DO UPDATE SET
                               session_key = COALESCE(sessions.session_key, excluded.session_key),
                               chat_id = COALESCE(sessions.chat_id, excluded.chat_id),
                               chat_type = COALESCE(sessions.chat_type, excluded.chat_type),
                               thread_id = COALESCE(sessions.thread_id, excluded.thread_id),
                               display_name = COALESCE(sessions.display_name, excluded.display_name),
                               origin_json = COALESCE(sessions.origin_json, excluded.origin_json),
                               transport_profile = COALESCE(sessions.transport_profile, excluded.transport_profile),
                               created_source = COALESCE(sessions.created_source, excluded.created_source)""",
                    # Same ownership stamp as _insert_session_row: an unowned (NULL) row
                    # vanishes from profile-keyed consumers.
                    (session_id, source, source, user_id, session_key, chat_id, chat_type, thread_id, display_name,
                     origin_json, self._own_profile_name(), transport_profile, time.time()),
                )
        with translated():
            holder = self._store.holder()
            holder._execute_write(_do)

    def adopt_orphan(self, orphan_id: str, donor_id: str) -> bool:
        if not orphan_id or not donor_id or orphan_id == donor_id:
            return False
        owner = self._own_profile_name()

        def _do(conn):
            donor = conn.execute(
                "SELECT session_key, chat_id, chat_type, thread_id, user_id, "
                "origin_json, display_name, source, profile_name FROM sessions WHERE id = ?",
                (donor_id,),
            ).fetchone()
            orphan = conn.execute(
                "SELECT session_key, source, profile_name FROM sessions WHERE id = ?",
                (orphan_id,)).fetchone()
            if (donor is None or orphan is None or not donor["session_key"] or orphan["session_key"]
                    or (donor["source"] or "") != (orphan["source"] or "")
                    or (owner is not None
                        and (donor["profile_name"] or owner) != (orphan["profile_name"] or owner))):
                return False
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
            # Retire under a reason recovery does NOT treat as resumable: 'agent_close' /
            # 'ws_orphan_reap' would keep it in the running and the orphan could lose the chat again.
            conn.execute(
                "UPDATE sessions SET ended_at = COALESCE(ended_at, ?), "
                "end_reason = 'superseded_by_repair' WHERE id = ?",
                (time.time(), donor_id))
            return True
        with translated():
            holder = self._store.holder()
            return holder._execute_write(_do)

    def prune_never_active(
        self, *, older_than_days: float, artifacts: SessionArtifacts | None = None, media_root: str,
    ) -> NeverActivePrune:
        candidates = self.never_active_keyed_sessions(older_than_days=older_than_days)
        # A fresh local reset tip is its conversation's current target, not a never-active chat:
        # the conversation holds the earlier segments' history and the owner's queued work.
        with translated():
            with self._store.holder()._read_ctx() as conn:
                ids = {str(row.session_id) for row in candidates
                       if owned_lineage_ids(conn, str(row.session_id)) == [str(row.session_id)]}
            if not ids:
                return NeverActivePrune(0, 0, 0, (), ())
            # The delete retires the deleted rows' routing entries inside its own transaction, so count
            # what pointed at the candidates before it commits; the caller's sweep then only catches
            # entries a concurrent writer re-added.
            routed_before = self._routing_entries_for_sessions(ids)
        outcome = self._store.sessions.delete_many(
            [SessionId(sid) for sid in sorted(ids)], skip_guarded=True, media_root=media_root)
        if artifacts is not None:
            artifacts.remove_for(outcome.removed_ids)
        kept = {str(sid) for sid in outcome.skipped_guarded}
        routing_deleted = sum(1 for _, _, sid in routed_before if sid not in kept)
        return NeverActivePrune(outcome.deleted, routing_deleted, len(kept), tuple(sorted(ids - kept)), tuple(sorted(outcome.removed_ids)))

    def sweep_routing_entries_for_sessions(self, session_ids: Iterable[str]) -> int:
        with translated():
            return self._delete_routing_entries_for_sessions({str(sid) for sid in session_ids})

    def prepare_topic_storage(self) -> None:
        with translated():
            apply_telegram_topic_migration(self._store.holder())

    def enable_topic_mode(
        self, *, chat_id: str, user_id: str, profile_name: str = "default", has_topics_enabled: bool | None = None,
        allows_users_to_create_topics: bool | None = None,
    ) -> None:
        with translated():
            holder = self._store.holder()
            apply_telegram_topic_migration(holder)
            now = time.time()
            profile_name = normalize_topic_profile_name(profile_name)

            def _to_int(value: Optional[bool]) -> Optional[int]:
                return None if value is None else (1 if value else 0)

            holder._write_sql("""
                INSERT INTO telegram_dm_topic_mode (
                    profile_name, chat_id, user_id, enabled, activated_at, updated_at,
                    has_topics_enabled, allows_users_to_create_topics,
                    capability_checked_at
                ) VALUES (?, ?, ?, 1, ?, ?, ?, ?, ?)
                ON CONFLICT(profile_name, chat_id) DO UPDATE SET
                    user_id = excluded.user_id,
                    enabled = 1,
                    updated_at = excluded.updated_at,
                    has_topics_enabled = excluded.has_topics_enabled,
                    allows_users_to_create_topics = excluded.allows_users_to_create_topics,
                    capability_checked_at = excluded.capability_checked_at
                """, (profile_name, str(chat_id), str(user_id), now, now,
                      _to_int(has_topics_enabled), _to_int(allows_users_to_create_topics), now))

    def disable_topic_mode(self, *, chat_id: str, profile_name: str = "default", clear_bindings: bool = True) -> None:
        profile_name = normalize_topic_profile_name(profile_name)

        def _do(conn):
            with contextlib.suppress(sqlite3.OperationalError):
                conn.execute(
                    "UPDATE telegram_dm_topic_mode SET enabled = 0, updated_at = ? "
                    "WHERE profile_name = ? AND chat_id = ?",
                    (time.time(), profile_name, str(chat_id)),
                )
                if clear_bindings:
                    conn.execute(
                        "DELETE FROM telegram_dm_topic_bindings WHERE profile_name = ? AND chat_id = ?",
                        (profile_name, str(chat_id)),
                    )
        with translated():
            holder = self._store.holder()
            holder._execute_write(_do)

    def delete_topic_binding(self, *, chat_id: str, thread_id: str, profile_name: str = "default") -> int:
        chat_id, thread_id = str(chat_id), str(thread_id)
        profile_name = normalize_topic_profile_name(profile_name)

        def _do(conn) -> int:
            try:
                deleted = conn.execute("""
                    DELETE FROM telegram_dm_topic_bindings
                    WHERE profile_name = ? AND chat_id = ? AND thread_id = ?
                    """, (profile_name, chat_id, thread_id)).rowcount or 0
            except sqlite3.OperationalError:
                return 0
            if not deleted:
                return 0
            # Last binding gone: disable topic mode in the same transaction (no
            # read-after-prune race). telegram_dm_topic_mode absent: the binding prune still stands.
            with contextlib.suppress(sqlite3.OperationalError):
                remaining = conn.execute("""
                    SELECT 1 FROM telegram_dm_topic_bindings
                    WHERE profile_name = ? AND chat_id = ? LIMIT 1
                    """, (profile_name, chat_id)).fetchone()
                if remaining is None:
                    conn.execute(
                        "UPDATE telegram_dm_topic_mode SET enabled = 0, updated_at = ? "
                        "WHERE profile_name = ? AND chat_id = ?",
                        (time.time(), profile_name, chat_id),
                    )
            return deleted
        with translated():
            holder = self._store.holder()
            return holder._execute_write(_do)

    def bind_topic(
        self, *, chat_id: str, thread_id: str, user_id: str, session_key: str, session_id: str,
        managed_mode: str = "auto", profile_name: str = "default", expected_session_id: str | None = None,
        move_session: bool = False,
    ) -> bool:
        with translated():
            holder = self._store.holder()
            apply_telegram_topic_migration(holder)
            now = time.time()
            chat_id, thread_id, user_id = str(chat_id), str(thread_id), str(user_id)
            session_key, session_id = str(session_key), str(session_id)
            profile_name = normalize_topic_profile_name(profile_name)

            def _do(conn):
                if expected_session_id is not None:
                    current = conn.execute(
                        "SELECT session_id FROM telegram_dm_topic_bindings "
                        "WHERE profile_name = ? AND chat_id = ? AND thread_id = ?",
                        (profile_name, chat_id, thread_id),
                    ).fetchone()
                    if current is None or str(current[0]) != str(expected_session_id):
                        return False
                existing_session = conn.execute("""
                    SELECT profile_name, chat_id, thread_id
                    FROM telegram_dm_topic_bindings
                    WHERE session_id = ?
                    """, (session_id,)).fetchone()
                if existing_session is not None:
                    linked_profile, linked_chat, linked_thread = existing_session
                    linked = (str(linked_profile), str(linked_chat), str(linked_thread))
                    if linked != (profile_name, chat_id, thread_id):
                        if not move_session or linked[0] != profile_name:  # never across profiles
                            raise TopicAlreadyLinked(session_id)
                        conn.execute("DELETE FROM telegram_dm_topic_bindings WHERE session_id = ?", (session_id,))
                conn.execute("""
                    INSERT INTO telegram_dm_topic_bindings (
                        profile_name, chat_id, thread_id, user_id, session_key, session_id,
                        managed_mode, linked_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(profile_name, chat_id, thread_id) DO UPDATE SET
                        user_id = excluded.user_id,
                        session_key = excluded.session_key,
                        session_id = excluded.session_id,
                        managed_mode = excluded.managed_mode,
                        updated_at = excluded.updated_at
                    """, (profile_name, chat_id, thread_id, user_id, session_key, session_id, managed_mode, now, now))
                return True
            try:
                return holder._execute_write(_do)
            except sqlite3.IntegrityError as exc:
                # The binding's session_id references sessions(id): a missing row fails the foreign key.
                if "FOREIGN KEY" not in str(exc):
                    raise
                raise SessionRowMissing(session_id) from exc

    # ── reads ───────────────────────────────────────────────────────────────

    def route_retired(self, session_key: str, entry_json: str) -> bool:
        with translated():
            holder = self._store.holder()
            with holder._read_ctx() as conn:
                return bool(conn.execute('SELECT ' + ROUTE_RETIRED_SQL,
                                         route_retired_params(session_key, entry_json, time.time())).fetchone()[0])

    def load_routing_entries(self, *, routing_scope: str = "") -> dict[str, str]:
        with translated():
            holder = self._store.holder()
            rows = holder._read_all(
                "SELECT session_key, entry_json FROM gateway_routing WHERE scope = ?", (routing_scope,))
        return {r["session_key"]: r["entry_json"] for r in rows}

    def routing_entry_for_session(self, session_id: str) -> Mapping[str, JsonValue] | None:
        with translated():
            holder = self._store.holder()
            rows = holder._read_all("SELECT entry_json FROM gateway_routing")
        for row in rows:
            try:
                entry = json.loads(row["entry_json"] or "{}")
            except (ValueError, TypeError, RecursionError):  # malformed / non-text / over-nested entry
                continue
            if isinstance(entry, dict) and entry.get("session_id") == session_id:
                return entry
        return None

    @staticmethod
    def _recovered(row: sqlite3.Row) -> RecoveredPeerSession:
        data = _session_row_dict(row)
        return RecoveredPeerSession(session=session_record(data), has_messages=bool(data["_has_messages"]))

    def latest_session_for_peer(
        self, *, source: str, user_id: str | None = None, session_key: str | None = None,
        chat_id: str | None = None, chat_type: str | None = None, thread_id: str | None = None,
    ) -> RecoveredPeerSession | None:
        if not session_key:
            return None
        with translated():
            holder = self._store.holder()
            with holder._read_ctx() as conn:
                row = conn.execute(_PEER_BY_KEY_SQL, (session_key, source)).fetchone()
                if row is not None:
                    return self._recovered(row)
                if chat_id is None or chat_type is None:
                    return None
                # Profile fence: a Telegram DM's peer tuple is identical for every bot (chat_id ==
                # user_id, no thread), so a sibling profile's legacy row in this store would otherwise be
                # adopted. A row is ours when its profile_name is the owner or NULL; stores outside the
                # profile tree derive no owner and keep the unfenced behavior.
                owner = self._own_profile_name()
                row = conn.execute(
                    _PEER_BY_TUPLE_SQL,
                    (source, user_id, chat_id, chat_type, thread_id, owner, owner, owner, owner, owner, owner),
                ).fetchone()
        return self._recovered(row) if row else None

    def session_for_origin(
        self, *, platform: str, chat_id: str, thread_id: str | None = None, user_id: str | None = None,
    ) -> str | None:
        if not platform or chat_id in (None, ""):
            return None
        query = """
            SELECT id, user_id, started_at FROM sessions
            WHERE LOWER(source) = LOWER(?)
              AND session_key IS NOT NULL
              AND chat_id = ?
              AND ended_at IS NULL
        """
        params: list = [platform, str(chat_id)]
        if thread_id is not None:
            query += " AND COALESCE(thread_id, '') = ?"
            params.append(str(thread_id))
        with translated():
            holder = self._store.holder()
            rows = [dict(r) for r in holder._read_all(query + " ORDER BY started_at DESC", params)]
        if not rows:
            return None
        if user_id:
            exact = [r for r in rows if str(r.get("user_id") or "") == str(user_id)]
            if exact:
                return str(exact[0]["id"])
            if len(rows) > 1:
                return None
        elif len({u for u in (str(r.get("user_id") or "").strip() for r in rows) if u}) > 1:
            return None
        return str(rows[0]["id"])

    def orphaned_sessions(self, *, max_gap_s: float | None = None) -> tuple[OrphanReport, ...]:
        gap = ORPHAN_ADOPTION_MAX_GAP_S if max_gap_s is None else float(max_gap_s)
        owner = self._own_profile_name()
        records: list[dict[str, Any]] = []
        with translated():
            holder = self._store.holder()
            with holder._read_ctx() as conn:
                for orphan in conn.execute(_ORPHANS_SQL).fetchall():
                    donor = None
                    reason = ""
                    orphan_owner = orphan["profile_name"] or owner
                    if orphan["parent_session_id"]:
                        evidence = "lineage"
                        donor = conn.execute(
                            _ORPHAN_LINEAGE_DONOR_SQL,
                            (orphan["parent_session_id"], orphan["source"], owner, owner, orphan_owner),
                        ).fetchone()
                        if donor is None:
                            reason = "parent session carries no gateway identity of this source"
                    else:
                        evidence = "contiguity"
                        started = orphan["started_at"] or 0
                        candidates = conn.execute(
                            _ORPHAN_CONTIGUITY_DONORS_SQL,
                            (orphan["id"], orphan["source"], orphan["user_id"], orphan["user_id"],
                             started - gap, started + gap, orphan["last_active"],
                             owner, owner, orphan_owner),
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
        return tuple(OrphanReport(**record) for record in records)

    def never_active_keyed_sessions(self, *, older_than_days: float) -> tuple[NeverActiveSession, ...]:
        if older_than_days < 0:
            raise ValueError(
                f"older_than_days must be >= 0, got {older_than_days!r}: a negative "
                "retention builds a future cutoff that matches every never-active keyed row.")
        cutoff = time.time() - (float(older_than_days) * 86400.0)
        with translated():
            holder = self._store.holder()
            rows = holder._read_all(
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
        return tuple(
            NeverActiveSession(
                session_id=r["id"], session_key=r["session_key"], source=r["source"], chat_id=r["chat_id"],
                chat_type=r["chat_type"], user_id=r["user_id"], started_at=r["started_at"])
            for r in rows)

    def is_topic_mode_enabled(self, *, chat_id: str, user_id: str, profile_name: str = "default") -> bool:
        profile_name = normalize_topic_profile_name(profile_name)
        with translated():
            row = self._topic_read_one("""
                    SELECT enabled FROM telegram_dm_topic_mode
                    WHERE profile_name = ? AND chat_id = ? AND user_id = ?
                    """, (profile_name, str(chat_id), str(user_id)))
        return bool(row[0]) if row is not None else False

    def get_topic_binding(self, *, chat_id: str, thread_id: str, profile_name: str = "default") -> TopicBinding | None:
        profile_name = normalize_topic_profile_name(profile_name)
        with translated():
            row = self._topic_read_one("""
                    SELECT * FROM telegram_dm_topic_bindings
                    WHERE profile_name = ? AND chat_id = ? AND thread_id = ?
                    """, (profile_name, str(chat_id), str(thread_id)))
        return TopicBinding(**dict(row)) if row else None

    def list_topic_bindings_for_chat(self, *, chat_id: str, profile_name: str = "default") -> tuple[TopicBinding, ...]:
        profile_name = normalize_topic_profile_name(profile_name)
        with translated():
            rows = self._topic_read_all(
                "SELECT * FROM telegram_dm_topic_bindings WHERE profile_name = ? AND chat_id = ? "
                "ORDER BY updated_at DESC",
                (profile_name, str(chat_id)),
            )
        return tuple(TopicBinding(**dict(row)) for row in rows)

    def get_topic_binding_for_session(self, *, session_id: str) -> TopicBinding | None:
        with translated():
            row = self._topic_read_one("""
                    SELECT * FROM telegram_dm_topic_bindings
                    WHERE session_id = ?
                    """, (str(session_id),))
        return TopicBinding(**dict(row)) if row else None

    def is_session_linked_to_topic(self, *, session_id: str) -> bool:
        with translated():
            row = self._topic_read_one("""
                    SELECT 1 FROM telegram_dm_topic_bindings
                    WHERE session_id = ?
                    LIMIT 1
                    """, (str(session_id),))
        return row is not None


register_repository("routing", SQLiteRoutingRepository)
