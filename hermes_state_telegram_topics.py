"""Telegram forum-topic mode and topic/session bindings for SessionDB.

Mixin contract: this is a plain mixin class consumed by
``hermes_state.SessionDB``. It defines no ``__init__`` and no state of its
own; methods access the host's attributes (``self._conn``, ``self.db_path``,
``self._execute_write`` and other SessionDB methods) established by
``SessionDB.__init__``. It must never import hermes_state (cycle) - shared
module-level constants live in hermes_state_common.

This is the Telegram-specific cluster: the opt-in schema migration for
topic mode, the enable/disable/query of that mode, and the CRUD over the
chat-topic-to-session bindings it creates. It is the only platform-specific
code in SessionDB, which is otherwise a platform-neutral session store, and
none of it is reachable unless a user runs ``/topic``.

Behavior-neutral: every method is lifted verbatim from SessionDB.
"""

import sqlite3
import time
from typing import Any, Dict, List, Optional

from hermes_state_common import (
    _PREVIEW_RAW_SELECT,
    _shape_preview,
)


class SessionTelegramTopicsMixin:
    """See module docstring - mixin for SessionDB (Telegram topics cluster)."""

    def apply_telegram_topic_migration(self) -> None:
        """Create Telegram DM topic-mode tables on explicit /topic opt-in.

        This migration is deliberately not part of automatic SessionDB startup
        reconciliation. Operators must be able to upgrade Hermes, keep the old
        Telegram bot behavior running, and only mutate topic-mode state when the
        user executes /topic to opt into the feature.

        Schema versions:
          v1 — initial shape (no ON DELETE CASCADE on session_id FK)
          v2 — session_id FK gets ON DELETE CASCADE so session pruning
               automatically clears bindings.
        """
        def _do(conn):
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS telegram_dm_topic_mode (
                    chat_id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    enabled INTEGER NOT NULL DEFAULT 1,
                    activated_at REAL NOT NULL,
                    updated_at REAL NOT NULL,
                    has_topics_enabled INTEGER,
                    allows_users_to_create_topics INTEGER,
                    capability_checked_at REAL,
                    intro_message_id TEXT,
                    pinned_message_id TEXT
                );

                CREATE TABLE IF NOT EXISTS telegram_dm_topic_bindings (
                    chat_id TEXT NOT NULL,
                    thread_id TEXT NOT NULL,
                    user_id TEXT NOT NULL,
                    session_key TEXT NOT NULL,
                    session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
                    managed_mode TEXT NOT NULL DEFAULT 'auto',
                    linked_at REAL NOT NULL,
                    updated_at REAL NOT NULL,
                    PRIMARY KEY (chat_id, thread_id)
                );

                CREATE UNIQUE INDEX IF NOT EXISTS idx_telegram_dm_topic_bindings_session
                ON telegram_dm_topic_bindings(session_id);

                CREATE INDEX IF NOT EXISTS idx_telegram_dm_topic_bindings_user
                ON telegram_dm_topic_bindings(user_id, chat_id);
                """
            )

            # v1 → v2: rebuild telegram_dm_topic_bindings if its session_id FK
            # lacks ON DELETE CASCADE. SQLite can't ALTER a foreign key, so we
            # rebuild the table. Only runs once per DB (version gate).
            current = conn.execute(
                "SELECT value FROM state_meta WHERE key = ?",
                ("telegram_dm_topic_schema_version",),
            ).fetchone()
            current_version = int(current[0]) if current and str(current[0]).isdigit() else 0
            if current_version < 2:
                fk_rows = conn.execute(
                    "PRAGMA foreign_key_list('telegram_dm_topic_bindings')"
                ).fetchall()
                needs_rebuild = any(
                    row[2] == "sessions" and (row[6] or "") != "CASCADE"
                    for row in fk_rows
                )
                if needs_rebuild:
                    conn.executescript(
                        """
                        CREATE TABLE telegram_dm_topic_bindings_new (
                            chat_id TEXT NOT NULL,
                            thread_id TEXT NOT NULL,
                            user_id TEXT NOT NULL,
                            session_key TEXT NOT NULL,
                            session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
                            managed_mode TEXT NOT NULL DEFAULT 'auto',
                            linked_at REAL NOT NULL,
                            updated_at REAL NOT NULL,
                            PRIMARY KEY (chat_id, thread_id)
                        );
                        INSERT INTO telegram_dm_topic_bindings_new
                            SELECT chat_id, thread_id, user_id, session_key,
                                   session_id, managed_mode, linked_at, updated_at
                            FROM telegram_dm_topic_bindings;
                        DROP TABLE telegram_dm_topic_bindings;
                        ALTER TABLE telegram_dm_topic_bindings_new
                            RENAME TO telegram_dm_topic_bindings;
                        CREATE UNIQUE INDEX idx_telegram_dm_topic_bindings_session
                            ON telegram_dm_topic_bindings(session_id);
                        CREATE INDEX idx_telegram_dm_topic_bindings_user
                            ON telegram_dm_topic_bindings(user_id, chat_id);
                        """
                    )

            conn.execute(
                "INSERT INTO state_meta (key, value) VALUES (?, ?) "
                "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                ("telegram_dm_topic_schema_version", "2"),
            )
        self._execute_write(_do)

    def enable_telegram_topic_mode(
        self,
        *,
        chat_id: str,
        user_id: str,
        has_topics_enabled: Optional[bool] = None,
        allows_users_to_create_topics: Optional[bool] = None,
    ) -> None:
        """Enable Telegram DM topic mode for one private chat/user.

        This method intentionally owns the explicit topic migration. Ordinary
        SessionDB startup must not create these side tables.
        """
        self.apply_telegram_topic_migration()
        now = time.time()

        def _to_int(value: Optional[bool]) -> Optional[int]:
            if value is None:
                return None
            return 1 if value else 0

        def _do(conn):
            conn.execute(
                """
                INSERT INTO telegram_dm_topic_mode (
                    chat_id, user_id, enabled, activated_at, updated_at,
                    has_topics_enabled, allows_users_to_create_topics,
                    capability_checked_at
                ) VALUES (?, ?, 1, ?, ?, ?, ?, ?)
                ON CONFLICT(chat_id) DO UPDATE SET
                    user_id = excluded.user_id,
                    enabled = 1,
                    updated_at = excluded.updated_at,
                    has_topics_enabled = excluded.has_topics_enabled,
                    allows_users_to_create_topics = excluded.allows_users_to_create_topics,
                    capability_checked_at = excluded.capability_checked_at
                """,
                (
                    str(chat_id),
                    str(user_id),
                    now,
                    now,
                    _to_int(has_topics_enabled),
                    _to_int(allows_users_to_create_topics),
                    now,
                ),
            )
        self._execute_write(_do)

    def disable_telegram_topic_mode(
        self,
        *,
        chat_id: str,
        clear_bindings: bool = True,
    ) -> None:
        """Disable Telegram DM topic mode for one private chat.

        When ``clear_bindings`` is True (default) the (chat_id, thread_id)
        bindings for this chat are also cleared so re-enabling later
        starts from a clean slate. Set to False if the operator wants to
        preserve bindings for a later re-enable.

        Never creates the topic-mode tables from scratch; if they don't
        exist there is nothing to disable and the call is a no-op.
        """
        def _do(conn):
            try:
                conn.execute(
                    "UPDATE telegram_dm_topic_mode SET enabled = 0, updated_at = ? "
                    "WHERE chat_id = ?",
                    (time.time(), str(chat_id)),
                )
                if clear_bindings:
                    conn.execute(
                        "DELETE FROM telegram_dm_topic_bindings WHERE chat_id = ?",
                        (str(chat_id),),
                    )
            except sqlite3.OperationalError:
                # Tables don't exist yet — nothing to disable.
                return
        self._execute_write(_do)

    def is_telegram_topic_mode_enabled(self, *, chat_id: str, user_id: str) -> bool:
        """Return whether Telegram DM topic mode is enabled for this chat/user."""
        with self._lock:
            try:
                row = self._conn.execute(
                    """
                    SELECT enabled FROM telegram_dm_topic_mode
                    WHERE chat_id = ? AND user_id = ?
                    """,
                    (str(chat_id), str(user_id)),
                ).fetchone()
            except sqlite3.OperationalError:
                return False
        if row is None:
            return False
        enabled = row["enabled"] if isinstance(row, sqlite3.Row) else row[0]
        return bool(enabled)

    def get_telegram_topic_binding(
        self,
        *,
        chat_id: str,
        thread_id: str,
    ) -> Optional[Dict[str, Any]]:
        """Return the session binding for a Telegram DM topic, if present."""
        with self._lock:
            try:
                row = self._conn.execute(
                    """
                    SELECT * FROM telegram_dm_topic_bindings
                    WHERE chat_id = ? AND thread_id = ?
                    """,
                    (str(chat_id), str(thread_id)),
                ).fetchone()
            except sqlite3.OperationalError:
                return None
        return dict(row) if row else None

    def list_telegram_topic_bindings_for_chat(
        self,
        *,
        chat_id: str,
    ) -> List[Dict[str, Any]]:
        """All Telegram DM topic bindings for one chat, newest first.

        Read-only; returns [] if the bindings table doesn't exist yet
        (does not trigger the topic-mode migration).
        """
        with self._lock:
            try:
                rows = self._conn.execute(
                    "SELECT * FROM telegram_dm_topic_bindings "
                    "WHERE chat_id = ? ORDER BY updated_at DESC",
                    (str(chat_id),),
                ).fetchall()
            except sqlite3.OperationalError:
                return []
        return [dict(row) for row in rows]

    def get_telegram_topic_binding_by_session(
        self,
        *,
        session_id: str,
    ) -> Optional[Dict[str, Any]]:
        """Return the Telegram DM topic binding for a given session_id, if present.

        Uses the UNIQUE INDEX on telegram_dm_topic_bindings(session_id) for an
        efficient reverse lookup. Returns None when the session has no binding or
        the table does not exist yet.
        """
        with self._lock:
            try:
                row = self._conn.execute(
                    """
                    SELECT * FROM telegram_dm_topic_bindings
                    WHERE session_id = ?
                    """,
                    (str(session_id),),
                ).fetchone()
            except sqlite3.OperationalError:
                return None
        return dict(row) if row else None

    def delete_telegram_topic_binding(
        self,
        *,
        chat_id: str,
        thread_id: str,
    ) -> int:
        """Remove the binding row for a single (chat, thread) pair.

        Called when the Telegram Bot API confirms a topic was deleted
        externally (``Thread not found`` after the same-thread retry
        already failed).  Without this prune, the stale row keeps
        living in ``telegram_dm_topic_bindings`` and the
        recovery logic in ``gateway.run._recover_telegram_topic_thread_id``
        cheerfully redirects future inbound messages to the deleted
        topic, causing tool progress, approvals, and replies to land
        in the wrong place.  Issue #31501.

        When this prune removes the chat's *last* remaining binding,
        the chat's row in ``telegram_dm_topic_mode`` is also flipped to
        ``enabled = 0`` in the same transaction.  Otherwise the chat
        would be left in topic mode with zero lanes — and
        ``gateway.run._recover_telegram_topic_thread_id`` keeps treating
        the chat as topic-enabled, lobby messages keep hunting for a
        binding that no longer exists, and a user who disabled topics in
        the Telegram client (rather than via ``/topic off``) stays stuck
        until the next send happens to fail. Clearing the flag makes
        recovery fully stand down once the dead topics are gone.

        Returns the number of binding rows deleted (0 when the binding
        was already absent or the topic-mode tables haven't been
        migrated yet — both are silent no-ops; we never raise from
        a cleanup hot path).
        """
        chat_id = str(chat_id)
        thread_id = str(thread_id)
        deleted = {"count": 0}

        def _do(conn):
            try:
                cursor = conn.execute(
                    """
                    DELETE FROM telegram_dm_topic_bindings
                    WHERE chat_id = ? AND thread_id = ?
                    """,
                    (chat_id, thread_id),
                )
                deleted["count"] = cursor.rowcount or 0
            except sqlite3.OperationalError:
                # Tables don't exist yet — nothing to prune.
                deleted["count"] = 0
                return
            if not deleted["count"]:
                return
            # If that was the chat's last binding, disable topic mode for
            # the chat so recovery stops steering lobby messages at a now
            # empty lane set. Same transaction → no read-after-prune race.
            try:
                remaining = conn.execute(
                    """
                    SELECT 1 FROM telegram_dm_topic_bindings
                    WHERE chat_id = ? LIMIT 1
                    """,
                    (chat_id,),
                ).fetchone()
                if remaining is None:
                    conn.execute(
                        "UPDATE telegram_dm_topic_mode "
                        "SET enabled = 0, updated_at = ? WHERE chat_id = ?",
                        (time.time(), chat_id),
                    )
            except sqlite3.OperationalError:
                # telegram_dm_topic_mode absent — binding prune still stands.
                pass

        self._execute_write(_do)
        return deleted["count"]

    def bind_telegram_topic(
        self,
        *,
        chat_id: str,
        thread_id: str,
        user_id: str,
        session_key: str,
        session_id: str,
        managed_mode: str = "auto",
    ) -> None:
        """Bind one Telegram DM topic thread to one Hermes session.

        A Hermes session may only be linked to one Telegram topic in MVP.
        Rebinding the same topic to the same session is idempotent; trying to
        link the same session to a different topic raises ValueError.
        """
        self.apply_telegram_topic_migration()
        now = time.time()
        chat_id = str(chat_id)
        thread_id = str(thread_id)
        user_id = str(user_id)
        session_key = str(session_key)
        session_id = str(session_id)

        def _do(conn):
            existing_session = conn.execute(
                """
                SELECT chat_id, thread_id FROM telegram_dm_topic_bindings
                WHERE session_id = ?
                """,
                (session_id,),
            ).fetchone()
            if existing_session is not None:
                linked_chat = existing_session["chat_id"] if isinstance(existing_session, sqlite3.Row) else existing_session[0]
                linked_thread = existing_session["thread_id"] if isinstance(existing_session, sqlite3.Row) else existing_session[1]
                if str(linked_chat) != chat_id or str(linked_thread) != thread_id:
                    raise ValueError("session is already linked to another Telegram topic")

            conn.execute(
                """
                INSERT INTO telegram_dm_topic_bindings (
                    chat_id, thread_id, user_id, session_key, session_id,
                    managed_mode, linked_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(chat_id, thread_id) DO UPDATE SET
                    user_id = excluded.user_id,
                    session_key = excluded.session_key,
                    session_id = excluded.session_id,
                    managed_mode = excluded.managed_mode,
                    updated_at = excluded.updated_at
                """,
                (
                    chat_id,
                    thread_id,
                    user_id,
                    session_key,
                    session_id,
                    managed_mode,
                    now,
                    now,
                ),
            )
        self._execute_write(_do)

    def is_telegram_session_linked_to_topic(self, *, session_id: str) -> bool:
        """Return True if a Hermes session is already bound to any Telegram DM topic.

        Read-only: does NOT trigger the telegram-topic migration. If the
        topic-mode tables have not been created yet (i.e. nobody has run
        ``/topic`` in this profile), the session is by definition unbound
        and we return False.
        """
        with self._lock:
            try:
                row = self._conn.execute(
                    """
                    SELECT 1 FROM telegram_dm_topic_bindings
                    WHERE session_id = ?
                    LIMIT 1
                    """,
                    (str(session_id),),
                ).fetchone()
            except sqlite3.OperationalError:
                return False
        return row is not None

    def list_unlinked_telegram_sessions_for_user(
        self,
        *,
        chat_id: str,
        user_id: str,
        limit: int = 10,
    ) -> List[Dict[str, Any]]:
        """List previous Telegram sessions for this user that are not bound to a topic.

        Read-only: does NOT trigger the telegram-topic migration. If the
        topic-mode tables are absent, fall back to a simpler query that
        just returns this user's Telegram sessions — there can't be any
        bindings yet.
        """
        with self._lock:
            try:
                rows = self._conn.execute(
                    f"""
                    SELECT s.*,
                        COALESCE(
                            (SELECT {_PREVIEW_RAW_SELECT}
                             FROM messages m
                             WHERE m.session_id = s.id AND m.role = 'user' AND m.content IS NOT NULL
                             ORDER BY m.timestamp, m.id LIMIT 1),
                            ''
                        ) AS _preview_raw,
                        COALESCE(
                            (SELECT MAX(m2.timestamp) FROM messages m2 WHERE m2.session_id = s.id),
                            s.started_at
                        ) AS last_active
                    FROM sessions s
                    WHERE s.source = 'telegram'
                      AND s.user_id = ?
                      AND NOT EXISTS (
                          SELECT 1 FROM telegram_dm_topic_bindings b
                          WHERE b.session_id = s.id
                      )
                    ORDER BY last_active DESC, s.started_at DESC
                    LIMIT ?
                    """,
                    (str(user_id), int(limit)),
                ).fetchall()
            except sqlite3.OperationalError:
                # telegram_dm_topic_bindings doesn't exist yet — no bindings
                # means every telegram session for this user is "unlinked".
                rows = self._conn.execute(
                    f"""
                    SELECT s.*,
                        COALESCE(
                            (SELECT {_PREVIEW_RAW_SELECT}
                             FROM messages m
                             WHERE m.session_id = s.id AND m.role = 'user' AND m.content IS NOT NULL
                             ORDER BY m.timestamp, m.id LIMIT 1),
                            ''
                        ) AS _preview_raw,
                        COALESCE(
                            (SELECT MAX(m2.timestamp) FROM messages m2 WHERE m2.session_id = s.id),
                            s.started_at
                        ) AS last_active
                    FROM sessions s
                    WHERE s.source = 'telegram'
                      AND s.user_id = ?
                    ORDER BY last_active DESC, s.started_at DESC
                    LIMIT ?
                    """,
                    (str(user_id), int(limit)),
                ).fetchall()

        sessions: List[Dict[str, Any]] = []
        for row in rows:
            session = dict(row)
            session["preview"] = _shape_preview(session.pop("_preview_raw", ""))
            sessions.append(session)
        return sessions
