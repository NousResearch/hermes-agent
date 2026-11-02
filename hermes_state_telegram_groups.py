"""Persistent Telegram group-forum labels, independent of opt-in DM topic mode."""

from __future__ import annotations

import time


class SessionTelegramGroupTopicsMixin:
    """Labels are scoped to the receiving transport, even in a shared state.db."""

    def record_telegram_group_topic_observation(
        self, *, profile_name: str, bot_id: str, chat_id: str, thread_id: str,
    ) -> None:
        """Own an observed thread without inventing a name (empty-string marker).

        Known observations are read-only; timestamps and learned renames do not churn
        on every message. INSERT remains atomic against concurrent name learning.
        """
        if not all(isinstance(value, str) and value.strip() for value in (profile_name, bot_id)):
            raise ValueError("Telegram group topic cache requires an explicit transport namespace")
        key = (profile_name, bot_id, str(chat_id), str(thread_id))
        if self._read_one("""SELECT 1 FROM telegram_group_topic_cache
            WHERE profile_name = ? AND bot_id = ? AND chat_id = ? AND thread_id = ?""", key):
            return
        self._execute_write(lambda conn: conn.execute(
            """INSERT INTO telegram_group_topic_cache
               (profile_name, bot_id, chat_id, thread_id, name, updated_at, source)
               VALUES (?, ?, ?, ?, '', ?, 'message_observation')
               ON CONFLICT(profile_name, bot_id, chat_id, thread_id) DO NOTHING""", (*key, time.time()),
        ))

    def set_telegram_group_topic_name(
        self, *, profile_name: str, bot_id: str, chat_id: str, thread_id: str,
        name: str, source: str = "forum_topic_status_update", only_if_unknown: bool = False,
    ) -> None:
        """Record a create/rename, or atomically seed an unknown label from an old reply."""
        if not isinstance(name, str) or not name.strip():
            return
        if not all(isinstance(value, str) and value.strip() for value in (profile_name, bot_id)):
            raise ValueError("Telegram group topic cache requires an explicit transport namespace")
        conflict = """DO UPDATE SET name = excluded.name,
            updated_at = excluded.updated_at, source = excluded.source"""
        if only_if_unknown:
            conflict += " WHERE telegram_group_topic_cache.name = ''"
        self._execute_write(lambda conn: conn.execute(
            f"""INSERT INTO telegram_group_topic_cache
                (profile_name, bot_id, chat_id, thread_id, name, updated_at, source)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(profile_name, bot_id, chat_id, thread_id) {conflict}""",
            (profile_name, bot_id, str(chat_id), str(thread_id), name.strip(), time.time(), source),
        ))

    def get_telegram_group_topic_name(
        self, *, profile_name: str, bot_id: str, chat_id: str, thread_id: str,
    ) -> str | None:
        """Read persisted labels without requiring a preceding learning write."""
        row = self._read_one(
            """SELECT name FROM telegram_group_topic_cache
               WHERE profile_name = ? AND bot_id = ? AND chat_id = ? AND thread_id = ?""",
            (profile_name, bot_id, str(chat_id), str(thread_id)),
        )
        return (row["name"] or None) if row else None

    def list_telegram_group_topic_names(self, *, profile_name: str, bot_id: str) -> list[dict]:
        """Directory observations for this bot/profile, including unknown empty-name markers."""
        return [dict(row) for row in self._read_all(
            """SELECT chat_id, thread_id, name FROM telegram_group_topic_cache
               WHERE profile_name = ? AND bot_id = ?""", (profile_name, bot_id),
        )]
