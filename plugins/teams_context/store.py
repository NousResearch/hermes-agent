"""SQLite storage for selected Teams chat context."""

from __future__ import annotations

import json
import os
import sqlite3
from pathlib import Path
from typing import Any

from hermes_constants import get_hermes_home
from plugins.teams_context.models import TeamsChatMessage, serialize_datetime


DEFAULT_STORE_FILENAME = "teams_context.sqlite"


def resolve_store_path(path: str | Path | None = None) -> Path:
    if path:
        return Path(str(path)).expanduser()
    env_path = os.getenv("TEAMS_CONTEXT_STORE_PATH", "").strip()
    if env_path:
        return Path(env_path).expanduser()
    return get_hermes_home() / DEFAULT_STORE_FILENAME


class TeamsContextStore:
    def __init__(self, path: str | Path | None = None):
        self.path = resolve_store_path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.path))
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self.connect() as conn:
            conn.executescript(
                """
                PRAGMA journal_mode=WAL;
                CREATE TABLE IF NOT EXISTS messages (
                    tenant_id TEXT,
                    chat_id TEXT NOT NULL,
                    message_id TEXT NOT NULL,
                    source_type TEXT NOT NULL DEFAULT 'channel',
                    source_label TEXT,
                    sender_id TEXT,
                    sender_name TEXT,
                    created_at TEXT,
                    updated_at TEXT,
                    deleted_at TEXT,
                    text TEXT NOT NULL DEFAULT '',
                    html TEXT,
                    web_url TEXT,
                    meeting_id TEXT,
                    raw_json TEXT NOT NULL DEFAULT '{}',
                    ingested_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (chat_id, message_id)
                );
                CREATE INDEX IF NOT EXISTS idx_messages_chat_time
                    ON messages(chat_id, created_at);
                CREATE INDEX IF NOT EXISTS idx_messages_meeting
                    ON messages(meeting_id);
                CREATE TABLE IF NOT EXISTS subscriptions (
                    subscription_id TEXT PRIMARY KEY,
                    chat_id TEXT NOT NULL,
                    resource TEXT NOT NULL,
                    change_type TEXT NOT NULL,
                    notification_url TEXT NOT NULL,
                    expiration_datetime TEXT,
                    client_state TEXT,
                    status TEXT NOT NULL DEFAULT 'active',
                    latest_renewal_at TEXT,
                    raw_json TEXT NOT NULL DEFAULT '{}',
                    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                );
                CREATE VIRTUAL TABLE IF NOT EXISTS messages_fts
                    USING fts5(chat_id UNINDEXED, message_id UNINDEXED, sender_name, text);
                CREATE TABLE IF NOT EXISTS kb_chunks (
                    source_id TEXT NOT NULL,
                    item_id TEXT NOT NULL,
                    source_type TEXT NOT NULL,
                    source_label TEXT NOT NULL,
                    chunk_index INTEGER NOT NULL DEFAULT 0,
                    sender_name TEXT,
                    created_at TEXT,
                    updated_at TEXT,
                    text TEXT NOT NULL DEFAULT '',
                    html TEXT,
                    web_url TEXT,
                    meeting_id TEXT,
                    metadata_json TEXT NOT NULL DEFAULT '{}',
                    ingested_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (source_id, item_id)
                );
                CREATE INDEX IF NOT EXISTS idx_kb_chunks_source
                    ON kb_chunks(source_type, source_id, chunk_index);
                CREATE INDEX IF NOT EXISTS idx_kb_chunks_meeting
                    ON kb_chunks(meeting_id);
                CREATE VIRTUAL TABLE IF NOT EXISTS kb_chunks_fts
                    USING fts5(source_id UNINDEXED, item_id UNINDEXED, source_label, text);
                """
            )
            self._migrate_schema(conn)

    def _migrate_schema(self, conn: sqlite3.Connection) -> None:
        message_columns = {
            row["name"]
            for row in conn.execute("PRAGMA table_info(messages)").fetchall()
        }
        if "source_type" not in message_columns:
            conn.execute(
                "ALTER TABLE messages ADD COLUMN source_type TEXT NOT NULL DEFAULT 'channel'"
            )
        if "source_label" not in message_columns:
            conn.execute("ALTER TABLE messages ADD COLUMN source_label TEXT")

    def upsert_message(self, message: TeamsChatMessage) -> None:
        with self.connect() as conn:
            conn.execute(
                """
                INSERT INTO messages (
                    tenant_id, chat_id, message_id, source_type, source_label, sender_id, sender_name,
                    created_at, updated_at, deleted_at, text, html, web_url,
                    meeting_id, raw_json, ingested_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                ON CONFLICT(chat_id, message_id) DO UPDATE SET
                    tenant_id=excluded.tenant_id,
                    source_type=excluded.source_type,
                    source_label=excluded.source_label,
                    sender_id=excluded.sender_id,
                    sender_name=excluded.sender_name,
                    created_at=excluded.created_at,
                    updated_at=excluded.updated_at,
                    deleted_at=excluded.deleted_at,
                    text=excluded.text,
                    html=excluded.html,
                    web_url=excluded.web_url,
                    meeting_id=excluded.meeting_id,
                    raw_json=excluded.raw_json,
                    ingested_at=CURRENT_TIMESTAMP
                """,
                (
                    message.tenant_id,
                    message.chat_id,
                    message.message_id,
                    _message_source_type(message),
                    _message_source_label(message),
                    message.sender_id,
                    message.sender_name,
                    serialize_datetime(message.created_at),
                    serialize_datetime(message.updated_at),
                    serialize_datetime(message.deleted_at),
                    message.text or "",
                    message.html,
                    message.web_url,
                    message.meeting_id,
                    json.dumps(message.raw, sort_keys=True),
                ),
            )
            self._replace_fts(conn, message.chat_id, message.message_id, message.sender_name, message.text)

    def mark_deleted(self, chat_id: str, message_id: str) -> None:
        with self.connect() as conn:
            conn.execute(
                """
                INSERT INTO messages (chat_id, message_id, deleted_at, text, raw_json)
                VALUES (?, ?, CURRENT_TIMESTAMP, '', '{}')
                ON CONFLICT(chat_id, message_id) DO UPDATE SET
                    deleted_at=CURRENT_TIMESTAMP,
                    text='',
                    html=NULL,
                    ingested_at=CURRENT_TIMESTAMP
                """,
                (chat_id, message_id),
            )
            conn.execute(
                "DELETE FROM messages_fts WHERE chat_id = ? AND message_id = ?",
                (chat_id, message_id),
            )

    def _replace_fts(
        self,
        conn: sqlite3.Connection,
        chat_id: str,
        message_id: str,
        sender_name: str | None,
        text: str,
    ) -> None:
        conn.execute(
            "DELETE FROM messages_fts WHERE chat_id = ? AND message_id = ?",
            (chat_id, message_id),
        )
        if text:
            conn.execute(
                "INSERT INTO messages_fts(chat_id, message_id, sender_name, text) VALUES (?, ?, ?, ?)",
                (chat_id, message_id, sender_name or "", text),
            )

    def search(
        self,
        query: str,
        *,
        chat_id: str | None = None,
        start_time: str | None = None,
        end_time: str | None = None,
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        limit = max(1, min(int(limit or 10), 50))
        clauses = ["messages.deleted_at IS NULL", "messages_fts MATCH ?"]
        params: list[Any] = [query]
        if chat_id:
            clauses.append("messages.chat_id = ?")
            params.append(chat_id)
        if start_time:
            clauses.append("messages.created_at >= ?")
            params.append(start_time)
        if end_time:
            clauses.append("messages.created_at <= ?")
            params.append(end_time)
        params.append(limit)
        sql = f"""
            SELECT messages.*, bm25(messages_fts) AS rank
            FROM messages_fts
            JOIN messages
              ON messages.chat_id = messages_fts.chat_id
             AND messages.message_id = messages_fts.message_id
            WHERE {' AND '.join(clauses)}
            ORDER BY rank, messages.created_at DESC
            LIMIT ?
        """
        with self.connect() as conn:
            return [dict(row) for row in conn.execute(sql, params).fetchall()]

    def unified_search(
        self,
        query: str,
        *,
        source_id: str | None = None,
        source_type: str | None = None,
        start_time: str | None = None,
        end_time: str | None = None,
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        limit = max(1, min(int(limit or 10), 100))
        query = str(query or "").strip()
        if not query:
            return self.dashboard_items(
                source_id=source_id,
                source_type=source_type,
                limit=limit,
            )["items"]

        message_clauses = ["messages.deleted_at IS NULL", "messages_fts MATCH ?"]
        message_params: list[Any] = [query]
        if source_id:
            message_clauses.append("messages.chat_id = ?")
            message_params.append(source_id)
        if source_type:
            message_clauses.append("messages.source_type = ?")
            message_params.append(source_type)
        if start_time:
            message_clauses.append("messages.created_at >= ?")
            message_params.append(start_time)
        if end_time:
            message_clauses.append("messages.created_at <= ?")
            message_params.append(end_time)

        chunk_clauses = ["kb_chunks_fts MATCH ?"]
        chunk_params: list[Any] = [query]
        if source_id:
            chunk_clauses.append("kb_chunks.source_id = ?")
            chunk_params.append(source_id)
        if source_type:
            chunk_clauses.append("kb_chunks.source_type = ?")
            chunk_params.append(source_type)
        if start_time:
            chunk_clauses.append("kb_chunks.created_at >= ?")
            chunk_params.append(start_time)
        if end_time:
            chunk_clauses.append("kb_chunks.created_at <= ?")
            chunk_params.append(end_time)

        sql = f"""
            SELECT * FROM (
                SELECT
                    messages.chat_id AS source_id,
                    messages.source_type AS source_type,
                    COALESCE(messages.source_label, messages.chat_id) AS source_label,
                    messages.message_id AS item_id,
                    messages.sender_name AS sender_name,
                    messages.created_at AS created_at,
                    messages.updated_at AS updated_at,
                    messages.text AS text,
                    messages.html AS html,
                    messages.web_url AS web_url,
                    messages.meeting_id AS meeting_id,
                    NULL AS chunk_index,
                    messages.raw_json AS metadata_json,
                    messages.ingested_at AS ingested_at,
                    bm25(messages_fts) AS rank
                FROM messages_fts
                JOIN messages
                  ON messages.chat_id = messages_fts.chat_id
                 AND messages.message_id = messages_fts.message_id
                WHERE {' AND '.join(message_clauses)}
                UNION ALL
                SELECT
                    kb_chunks.source_id AS source_id,
                    kb_chunks.source_type AS source_type,
                    kb_chunks.source_label AS source_label,
                    kb_chunks.item_id AS item_id,
                    kb_chunks.sender_name AS sender_name,
                    kb_chunks.created_at AS created_at,
                    kb_chunks.updated_at AS updated_at,
                    kb_chunks.text AS text,
                    kb_chunks.html AS html,
                    kb_chunks.web_url AS web_url,
                    kb_chunks.meeting_id AS meeting_id,
                    kb_chunks.chunk_index AS chunk_index,
                    kb_chunks.metadata_json AS metadata_json,
                    kb_chunks.ingested_at AS ingested_at,
                    bm25(kb_chunks_fts) AS rank
                FROM kb_chunks_fts
                JOIN kb_chunks
                  ON kb_chunks.source_id = kb_chunks_fts.source_id
                 AND kb_chunks.item_id = kb_chunks_fts.item_id
                WHERE {' AND '.join(chunk_clauses)}
            )
            ORDER BY rank, COALESCE(created_at, ingested_at) DESC
            LIMIT ?
        """
        with self.connect() as conn:
            rows = conn.execute(
                sql,
                [*message_params, *chunk_params, limit],
            ).fetchall()
            return [_normalize_item(dict(row)) for row in rows]

    def upsert_kb_chunk(
        self,
        *,
        source_id: str,
        item_id: str,
        source_type: str,
        source_label: str,
        text: str,
        chunk_index: int = 0,
        sender_name: str | None = None,
        created_at: str | None = None,
        updated_at: str | None = None,
        html: str | None = None,
        web_url: str | None = None,
        meeting_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        if source_type not in {"channel", "meeting", "recording", "transcript"}:
            raise ValueError(f"Unsupported TeamContext source type: {source_type}")
        text = str(text or "").strip()
        if not text:
            return
        with self.connect() as conn:
            conn.execute(
                """
                INSERT INTO kb_chunks (
                    source_id, item_id, source_type, source_label, chunk_index,
                    sender_name, created_at, updated_at, text, html, web_url,
                    meeting_id, metadata_json, ingested_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                ON CONFLICT(source_id, item_id) DO UPDATE SET
                    source_type=excluded.source_type,
                    source_label=excluded.source_label,
                    chunk_index=excluded.chunk_index,
                    sender_name=excluded.sender_name,
                    created_at=excluded.created_at,
                    updated_at=excluded.updated_at,
                    text=excluded.text,
                    html=excluded.html,
                    web_url=excluded.web_url,
                    meeting_id=excluded.meeting_id,
                    metadata_json=excluded.metadata_json,
                    ingested_at=CURRENT_TIMESTAMP
                """,
                (
                    source_id,
                    item_id,
                    source_type,
                    source_label,
                    int(chunk_index),
                    sender_name,
                    created_at,
                    updated_at,
                    text,
                    html,
                    web_url,
                    meeting_id,
                    json.dumps(metadata or {}, sort_keys=True),
                ),
            )
            conn.execute(
                "DELETE FROM kb_chunks_fts WHERE source_id = ? AND item_id = ?",
                (source_id, item_id),
            )
            conn.execute(
                """
                INSERT INTO kb_chunks_fts(source_id, item_id, source_label, text)
                VALUES (?, ?, ?, ?)
                """,
                (source_id, item_id, source_label, text),
            )

    def dashboard_items(
        self,
        *,
        query: str = "",
        source_id: str | None = None,
        source_type: str | None = None,
        limit: int = 100,
    ) -> dict[str, Any]:
        limit = max(1, min(int(limit or 100), 500))
        if query:
            items = self.unified_search(
                query,
                source_id=source_id,
                source_type=source_type,
                limit=limit,
            )
            sources = self.list_sources()
            return {"sources": sources, "items": items, "total": len(items)}

        clauses = ["1=1"]
        params: list[Any] = []
        if source_id:
            clauses.append("source_id = ?")
            params.append(source_id)
        if source_type:
            clauses.append("source_type = ?")
            params.append(source_type)
        sql = f"""
            SELECT * FROM (
                SELECT
                    chat_id AS source_id,
                    source_type AS source_type,
                    COALESCE(source_label, chat_id) AS source_label,
                    message_id AS item_id,
                    sender_name,
                    created_at,
                    updated_at,
                    text,
                    html,
                    web_url,
                    meeting_id,
                    NULL AS chunk_index,
                    raw_json AS metadata_json,
                    ingested_at
                FROM messages
                WHERE deleted_at IS NULL
                UNION ALL
                SELECT
                    source_id,
                    source_type,
                    source_label,
                    item_id,
                    sender_name,
                    created_at,
                    updated_at,
                    text,
                    html,
                    web_url,
                    meeting_id,
                    chunk_index,
                    metadata_json,
                    ingested_at
                FROM kb_chunks
            )
            WHERE {' AND '.join(clauses)}
            ORDER BY COALESCE(created_at, ingested_at) DESC, chunk_index ASC
            LIMIT ?
        """
        with self.connect() as conn:
            rows = conn.execute(sql, [*params, limit]).fetchall()
            total = conn.execute(
                f"""
                SELECT COUNT(*) AS count FROM (
                    SELECT chat_id AS source_id, source_type FROM messages WHERE deleted_at IS NULL
                    UNION ALL
                    SELECT source_id, source_type FROM kb_chunks
                ) WHERE {' AND '.join(clauses)}
                """,
                params,
            ).fetchone()["count"]
            return {
                "sources": self.list_sources(conn),
                "items": [_normalize_item(dict(row)) for row in rows],
                "total": int(total or 0),
            }

    def list_sources(self, conn: sqlite3.Connection | None = None) -> list[dict[str, Any]]:
        own_conn = conn is None
        if conn is None:
            conn = self.connect()
        try:
            rows = conn.execute(
                """
                SELECT
                    source_id,
                    source_type,
                    source_label AS label,
                    COUNT(*) AS item_count,
                    MAX(COALESCE(created_at, ingested_at)) AS latest_at,
                    MAX(ingested_at) AS last_ingested
                FROM (
                    SELECT
                        chat_id AS source_id,
                        source_type,
                        COALESCE(source_label, chat_id) AS source_label,
                        created_at,
                        ingested_at
                    FROM messages
                    WHERE deleted_at IS NULL
                    UNION ALL
                    SELECT
                        source_id,
                        source_type,
                        source_label,
                        created_at,
                        ingested_at
                    FROM kb_chunks
                )
                GROUP BY source_id, source_type, source_label
                ORDER BY latest_at DESC, label ASC
                """
            ).fetchall()
            return [dict(row) for row in rows]
        finally:
            if own_conn:
                conn.close()

    def thread(self, chat_id: str, message_id: str, *, before: int = 10, after: int = 10) -> list[dict[str, Any]]:
        with self.connect() as conn:
            row = conn.execute(
                "SELECT created_at FROM messages WHERE chat_id = ? AND message_id = ?",
                (chat_id, message_id),
            ).fetchone()
            if not row:
                return []
            created_at = row["created_at"] or ""
            before_rows = conn.execute(
                """
                SELECT * FROM messages
                WHERE chat_id = ? AND deleted_at IS NULL AND created_at < ?
                ORDER BY created_at DESC
                LIMIT ?
                """,
                (chat_id, created_at, max(0, min(int(before), 50))),
            ).fetchall()
            anchor_and_after = conn.execute(
                """
                SELECT * FROM messages
                WHERE chat_id = ? AND deleted_at IS NULL AND created_at >= ?
                ORDER BY created_at ASC
                LIMIT ?
                """,
                (chat_id, created_at, 1 + max(0, min(int(after), 50))),
            ).fetchall()
            rows = list(reversed(before_rows)) + list(anchor_and_after)
            return [dict(item) for item in rows]

    def meeting_context(self, meeting_id: str, *, limit: int = 50) -> list[dict[str, Any]]:
        with self.connect() as conn:
            rows = conn.execute(
                """
                SELECT * FROM messages
                WHERE meeting_id = ? AND deleted_at IS NULL
                ORDER BY created_at ASC
                LIMIT ?
                """,
                (meeting_id, max(1, min(int(limit or 50), 200))),
            ).fetchall()
            return [dict(row) for row in rows]

    def upsert_subscription(self, payload: dict[str, Any], *, chat_id: str) -> None:
        subscription_id = str(payload.get("id") or payload.get("subscription_id") or "").strip()
        if not subscription_id:
            raise ValueError("subscription id is required")
        with self.connect() as conn:
            conn.execute(
                """
                INSERT INTO subscriptions (
                    subscription_id, chat_id, resource, change_type,
                    notification_url, expiration_datetime, client_state,
                    status, raw_json, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                ON CONFLICT(subscription_id) DO UPDATE SET
                    chat_id=excluded.chat_id,
                    resource=excluded.resource,
                    change_type=excluded.change_type,
                    notification_url=excluded.notification_url,
                    expiration_datetime=excluded.expiration_datetime,
                    client_state=excluded.client_state,
                    status=excluded.status,
                    raw_json=excluded.raw_json,
                    updated_at=CURRENT_TIMESTAMP
                """,
                (
                    subscription_id,
                    chat_id,
                    payload.get("resource", ""),
                    payload.get("changeType") or payload.get("change_type") or "",
                    payload.get("notificationUrl") or payload.get("notification_url") or "",
                    payload.get("expirationDateTime") or payload.get("expiration_datetime"),
                    payload.get("clientState") or payload.get("client_state"),
                    payload.get("status") or "active",
                    json.dumps(payload, sort_keys=True),
                ),
            )

    def list_subscriptions(self) -> list[dict[str, Any]]:
        with self.connect() as conn:
            return [
                dict(row)
                for row in conn.execute(
                    "SELECT * FROM subscriptions ORDER BY expiration_datetime"
                ).fetchall()
            ]

    def subscription_chat_id(self, subscription_id: str) -> str | None:
        with self.connect() as conn:
            row = conn.execute(
                "SELECT chat_id FROM subscriptions WHERE subscription_id = ?",
                (subscription_id,),
            ).fetchone()
            return str(row["chat_id"]) if row else None


def _message_source_type(message: TeamsChatMessage) -> str:
    raw_value = message.raw.get("source_type") if isinstance(message.raw, dict) else None
    value = str(raw_value or "channel").strip().lower()
    return value if value in {"channel", "meeting", "recording", "transcript"} else "channel"


def _message_source_label(message: TeamsChatMessage) -> str | None:
    if not isinstance(message.raw, dict):
        return None
    for key in ("source_label", "channel_title", "chat_name", "team_title"):
        value = message.raw.get(key)
        if value:
            return str(value)
    return None


def _normalize_item(row: dict[str, Any]) -> dict[str, Any]:
    try:
        metadata = json.loads(row.get("metadata_json") or "{}")
    except Exception:
        metadata = {}
    row["metadata"] = metadata if isinstance(metadata, dict) else {}
    row.pop("metadata_json", None)
    return row
