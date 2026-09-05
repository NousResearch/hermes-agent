"""Telegram DM topic-mode mixin for :class:`hermes_state.SessionDB`."""

from __future__ import annotations

import contextlib
import logging
import sqlite3
import time
from typing import Any, Dict, List, Optional

from hermes_state_common import _PREVIEW_ELIGIBLE_SQL, _PREVIEW_RAW_SELECT, _sql_session_last_active

# caplog tests pin the "hermes_state" logger name.
logger = logging.getLogger("hermes_state")


def _normalize_telegram_topic_profile_name(profile_name: Optional[str] = None) -> str:
    """Empty/missing → ``"default"`` (single namespace for non-multiplexed gateways).
    Multiplexed callers must pass the *routed* profile (``source.profile``), never the
    process-global active profile."""
    name = str(profile_name or "").strip()
    return name if name else "default"


# (table, column list, DDL body). profile_name leads the PK: a private chat_id is the
# user id, identical across bots sharing one state.db.
_TOPIC_TABLES = (
    (
        "telegram_dm_topic_mode",
        "profile_name, chat_id, user_id, enabled, activated_at, updated_at, "
        "has_topics_enabled, allows_users_to_create_topics, capability_checked_at, intro_message_id, pinned_message_id",
        """
                    profile_name TEXT NOT NULL DEFAULT 'default',
                    chat_id TEXT NOT NULL,
                    user_id TEXT NOT NULL,
                    enabled INTEGER NOT NULL DEFAULT 1,
                    activated_at REAL NOT NULL,
                    updated_at REAL NOT NULL,
                    has_topics_enabled INTEGER,
                    allows_users_to_create_topics INTEGER,
                    capability_checked_at REAL,
                    intro_message_id TEXT,
                    pinned_message_id TEXT,
                    PRIMARY KEY (profile_name, chat_id)
                """,
    ),
    (
        "telegram_dm_topic_bindings",
        "profile_name, chat_id, thread_id, user_id, session_key, session_id, managed_mode, linked_at, updated_at",
        """
                    profile_name TEXT NOT NULL DEFAULT 'default',
                    chat_id TEXT NOT NULL,
                    thread_id TEXT NOT NULL,
                    user_id TEXT NOT NULL,
                    session_key TEXT NOT NULL,
                    session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
                    managed_mode TEXT NOT NULL DEFAULT 'auto',
                    linked_at REAL NOT NULL,
                    updated_at REAL NOT NULL,
                    PRIMARY KEY (profile_name, chat_id, thread_id)
                """,
    ),
)

# Shared SELECT for the unlinked-session listing; the profile/bindings clauses are
# spliced in only when the bindings table exists.
_UNLINKED_SELECT_HEAD = f"""
                    SELECT s.*,
                        COALESCE(sp.prompt, s.system_prompt)
                            AS _system_prompt_resolved,
                        COALESCE(
                            (SELECT {_PREVIEW_RAW_SELECT}
                             FROM messages m
                             WHERE m.session_id = s.id AND m.role = 'user' AND m.content IS NOT NULL
                               AND {_PREVIEW_ELIGIBLE_SQL}
                             ORDER BY m.timestamp, m.id LIMIT 1),
                            ''
                        ) AS _preview_raw,
                        {_sql_session_last_active("s")} AS last_active
                    FROM sessions s
                    LEFT JOIN system_prompts sp
                      ON sp.hash = s.system_prompt_hash
                    WHERE s.source = 'telegram'
                      AND s.user_id = ?
"""
_UNLINKED_SELECT_TAIL = """                    ORDER BY last_active DESC, s.started_at DESC
                    LIMIT ?
                    """
# sessions.profile_name is NULL/empty for legacy rows → treat as default.
_UNLINKED_SCOPE_CLAUSES = """                      AND COALESCE(NULLIF(TRIM(s.profile_name), ''), 'default') = ?
                      AND NOT EXISTS (
                          SELECT 1 FROM telegram_dm_topic_bindings b
                          WHERE b.session_id = s.id
                      )
"""


class SessionTelegramTopicsMixin:
    """Telegram DM topic-mode tables, bindings and lookups. Read paths tolerate absent
    tables (nobody ran ``/topic``) by returning their empty value and never create them;
    pre-v3 tables left by an upgrade are self-healed on read."""

    def _topic_read_one(self, sql: str, params):
        """``fetchone`` that treats an unmigrated table as None. A pre-v3 table left by an
        upgrade (issue #103363) is healed to v3 once and the read retried, so topic mode
        keeps working on existing installs instead of silently reading as empty."""
        try:
            return self._read_one(sql, params)
        except sqlite3.OperationalError as exc:
            if "no such column: profile_name" not in str(exc):
                return None
            self._heal_pre_v3_topic_tables()
            try:
                return self._read_one(sql, params)
            except sqlite3.OperationalError:
                return None

    def _heal_pre_v3_topic_tables(self) -> None:
        """Heal pre-v3 topic tables left by an upgrade. A failed heal (lock timeout,
        cross-process race) warns once instead of surfacing through the readers as the
        silent-return shape they exist to avoid; the next read retries the heal."""
        try:
            self.apply_telegram_topic_migration()
        except sqlite3.Error:
            if not getattr(self, "_topic_heal_warned", False):
                self._topic_heal_warned = True
                logger.warning(
                    "telegram topic self-heal to v3 failed; reads keep hitting the "
                    "pre-v3 schema and read as empty until the heal succeeds",
                    exc_info=True,
                )

    def apply_telegram_topic_migration(self) -> None:
        """Create Telegram DM topic-mode tables on explicit /topic opt-in and self-heal
        pre-v3 tables on read (#103363); absent tables stay read-only, and startup
        reconciliation never runs this. Schema versions: v1 initial; v2 session_id FK
        ON DELETE CASCADE (pruning clears bindings); v3 ``profile_name`` on both tables so
        multiplexed gateways sharing one state.db isolate topic state per profile.

        See #76423.
        """
        def _do(conn):
            for table, columns, ddl in _TOPIC_TABLES:
                conn.execute(f"CREATE TABLE IF NOT EXISTS {table} ({ddl})")
                have = {row[1] for row in conn.execute(f"PRAGMA table_info('{table}')")}
                if "profile_name" in have:
                    continue
                # v1/v2 → v3. SQLite can't ALTER a PK or FK, so rebuild (also supplies v2's
                # ON DELETE CASCADE). Legacy rows land in "default" only. Plain execute()
                # keeps every statement inside _execute_write's BEGIN IMMEDIATE:
                # executescript would implicitly commit the open transaction and run each
                # statement in autocommit, so a crash after the DROP strands legacy rows
                # in {table}_new (#42004 diagnosed the same shape for v1→v2) and breaks
                # the whole-callback retry contract.
                legacy_columns = columns.replace("profile_name, ", "", 1)
                conn.execute(f"CREATE TABLE {table}_new ({ddl})")
                conn.execute(
                    f"INSERT INTO {table}_new ({columns}) "
                    f"SELECT 'default', {legacy_columns} FROM {table}"
                )
                conn.execute(f"DROP TABLE {table}")
                conn.execute(f"ALTER TABLE {table}_new RENAME TO {table}")
            # Indexes after any rebuild: the user index needs profile_name.
            conn.execute(
                "CREATE UNIQUE INDEX IF NOT EXISTS idx_telegram_dm_topic_bindings_session "
                "ON telegram_dm_topic_bindings(session_id)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_telegram_dm_topic_bindings_user "
                "ON telegram_dm_topic_bindings(profile_name, user_id, chat_id)"
            )
            conn.execute(
                "INSERT INTO state_meta (key, value) VALUES (?, ?) "
                "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                ("telegram_dm_topic_schema_version", "3"),
            )
        self._execute_write(_do)

    def enable_telegram_topic_mode(
        self, *, chat_id: str, user_id: str, profile_name: str="default", has_topics_enabled: Optional[bool]=None,
        allows_users_to_create_topics: Optional[bool]=None,
    ) -> None:
        """Enable Telegram DM topic mode for one private chat/user. Owns the explicit topic
        migration; SessionDB startup must not create these tables.

        ``profile_name`` namespaces rows under a shared multiplex ``state.db`` (issue #76423). Callers
        handling a multiplexed event must pass the routed profile from ``source.profile``, not the
        process-global active profile.
        """
        self.apply_telegram_topic_migration()
        now = time.time()
        profile_name = _normalize_telegram_topic_profile_name(profile_name)

        def _to_int(value: Optional[bool]) -> Optional[int]:
            return None if value is None else (1 if value else 0)

        self._write_sql("""
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

    def disable_telegram_topic_mode(
        self, *, chat_id: str, profile_name: str = "default", clear_bindings: bool = True
    ) -> None:
        """Disable Telegram DM topic mode for one private chat. ``clear_bindings`` also drops
        the chat's bindings so a later re-enable starts clean. Never creates the tables;
        absent tables are a no-op."""
        profile_name = _normalize_telegram_topic_profile_name(profile_name)

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
        self._execute_write(_do)

    def is_telegram_topic_mode_enabled(self, *, chat_id: str, user_id: str, profile_name: str = "default") -> bool:
        """Return whether Telegram DM topic mode is enabled for this chat/user."""
        profile_name = _normalize_telegram_topic_profile_name(profile_name)
        row = self._topic_read_one("""
                    SELECT enabled FROM telegram_dm_topic_mode
                    WHERE profile_name = ? AND chat_id = ? AND user_id = ?
                    """, (profile_name, str(chat_id), str(user_id)))
        return bool(row[0]) if row is not None else False

    def get_telegram_topic_binding(
        self, *, chat_id: str, thread_id: str, profile_name: str = "default"
    ) -> Optional[Dict[str, Any]]:
        """Return the session binding for a Telegram DM topic, if present."""
        profile_name = _normalize_telegram_topic_profile_name(profile_name)
        row = self._topic_read_one("""
                    SELECT * FROM telegram_dm_topic_bindings
                    WHERE profile_name = ? AND chat_id = ? AND thread_id = ?
                    """, (profile_name, str(chat_id), str(thread_id)))
        return dict(row) if row else None

    def list_telegram_topic_bindings_for_chat(
        self, *, chat_id: str, profile_name: str = "default"
    ) -> List[Dict[str, Any]]:
        """All bindings for one chat, newest first ([] when the table is absent)."""
        profile_name = _normalize_telegram_topic_profile_name(profile_name)
        try:
            rows = self._read_all(
                "SELECT * FROM telegram_dm_topic_bindings WHERE profile_name = ? AND chat_id = ? ORDER BY updated_at DESC",
                (profile_name, str(chat_id)),
            )
        except sqlite3.OperationalError as exc:
            if "no such column: profile_name" not in str(exc):
                return []
            # Same pre-v3 self-heal as _topic_read_one (issue #103363).
            self._heal_pre_v3_topic_tables()
            try:
                rows = self._read_all(
                    "SELECT * FROM telegram_dm_topic_bindings WHERE profile_name = ? AND chat_id = ? ORDER BY updated_at DESC",
                    (profile_name, str(chat_id)),
                )
            except sqlite3.OperationalError:
                return []
        return [dict(row) for row in rows]

    def get_telegram_topic_binding_by_session(self, *, session_id: str) -> Optional[Dict[str, Any]]:
        """Reverse lookup via the UNIQUE INDEX on session_id; None when unbound."""
        row = self._topic_read_one("""
                    SELECT * FROM telegram_dm_topic_bindings
                    WHERE session_id = ?
                    """, (str(session_id),))
        return dict(row) if row else None

    def delete_telegram_topic_binding(self, *, chat_id: str, thread_id: str, profile_name: str = "default") -> int:
        """Remove the binding row for one (chat, thread) pair. Called when the Bot API confirms
        a topic was deleted externally (``Thread not found`` after the same-thread retry
        failed); otherwise ``gateway.run._recover_telegram_topic_thread_id`` keeps
        redirecting inbound messages to the dead topic. If this removes the chat's *last*
        binding, ``telegram_dm_topic_mode`` is flipped to ``enabled = 0`` in the same
        transaction, or a user who disabled topics in the Telegram client (not via
        ``/topic off``) stays stuck. Returns the number of rows deleted; absent binding or
        unmigrated tables are silent no-ops (never raise from a cleanup hot path).

        Without this prune, the stale row keeps living in ``telegram_dm_topic_bindings`` and the recovery
        logic in ``gateway.run._recover_telegram_topic_thread_id`` cheerfully redirects future inbound
        messages to the deleted topic, causing tool progress, approvals, and replies to land in the wrong
        place. Issue #31501.
        """
        chat_id, thread_id = str(chat_id), str(thread_id)
        profile_name = _normalize_telegram_topic_profile_name(profile_name)

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
            # Last binding gone → disable topic mode in the same transaction (no
            # read-after-prune race). telegram_dm_topic_mode absent — binding prune still stands.
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

        return self._execute_write(_do)

    def bind_telegram_topic(
        self, *, chat_id: str, thread_id: str, user_id: str, session_key: str,
        session_id: str, managed_mode: str = "auto", profile_name: str = "default",
    ) -> None:
        """Bind one Telegram DM topic thread to one Hermes session. A session may be linked to
        only one topic: rebinding the same pair is idempotent; linking the session to a
        different topic raises ValueError."""
        self.apply_telegram_topic_migration()
        now = time.time()
        chat_id, thread_id, user_id = str(chat_id), str(thread_id), str(user_id)
        session_key, session_id = str(session_key), str(session_id)
        profile_name = _normalize_telegram_topic_profile_name(profile_name)

        def _do(conn):
            existing_session = conn.execute("""
                SELECT profile_name, chat_id, thread_id
                FROM telegram_dm_topic_bindings
                WHERE session_id = ?
                """, (session_id,)).fetchone()
            if existing_session is not None:
                linked_profile, linked_chat, linked_thread = existing_session
                if (str(linked_profile), str(linked_chat), str(linked_thread)) != (profile_name, chat_id, thread_id):
                    raise ValueError("session is already linked to another Telegram topic")
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
        self._execute_write(_do)

    def is_telegram_session_linked_to_topic(self, *, session_id: str) -> bool:
        """True if the session is bound to any Telegram DM topic (absent tables → False)."""
        row = self._topic_read_one("""
                    SELECT 1 FROM telegram_dm_topic_bindings
                    WHERE session_id = ?
                    LIMIT 1
                    """, (str(session_id),))
        return row is not None

    def list_unlinked_telegram_sessions_for_user(
        self, *, chat_id: str, user_id: str, profile_name: str = "default", limit: int = 10
    ) -> List[Dict[str, Any]]:
        """This user's Telegram sessions not bound to a topic. Read-only: if the bindings table
        is absent, every session is unlinked and the profile-unscoped query is used.
        Scoped by ``profile_name`` so multiplexed profiles do not surface each other.

        See #76423.
        """
        profile_name = _normalize_telegram_topic_profile_name(profile_name)
        with self._read_ctx() as conn:
            try:
                rows = conn.execute(
                    _UNLINKED_SELECT_HEAD + _UNLINKED_SCOPE_CLAUSES + _UNLINKED_SELECT_TAIL,
                    (str(user_id), profile_name, int(limit)),
                ).fetchall()
            except sqlite3.OperationalError:
                rows = conn.execute(
                    _UNLINKED_SELECT_HEAD + _UNLINKED_SELECT_TAIL, (str(user_id), int(limit)),
                ).fetchall()
        return [self._rich_row(row) for row in rows]
