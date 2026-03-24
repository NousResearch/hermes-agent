"""Durable async write-behind queue for the native RetainDB integration."""

from __future__ import annotations

import json
import logging
import queue
import sqlite3
import threading
import time
from datetime import datetime, timezone
from typing import Any

from retaindb_integration.client import RetainDBClient, RetainDBClientConfig
from retaindb_integration.identity import ResolvedRetainDBIdentity

logger = logging.getLogger(__name__)

_ASYNC_SHUTDOWN = object()


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class DurableRetainDBWriteQueue:
    """Crash-safe local spool for background RetainDB session ingestion."""

    def __init__(self, client: RetainDBClient, config: RetainDBClientConfig):
        self._client = client
        self._config = config
        self._db_path = config.queue_db_path
        self._identity_by_session: dict[str, ResolvedRetainDBIdentity] = {}
        self._queue: queue.Queue | None = None
        self._thread: threading.Thread | None = None

        self._init_db()

        if config.write_frequency == "async":
            self._queue = queue.Queue()
            self._thread = threading.Thread(
                target=self._writer_loop,
                name="retaindb-async-writer",
                daemon=True,
            )
            self._thread.start()
            for session_id in self.pending_session_ids():
                self._queue.put(session_id)

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._db_path, timeout=30)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS pending_ingest (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    user_id TEXT,
                    agent_id TEXT,
                    project TEXT NOT NULL,
                    turn_id TEXT NOT NULL,
                    message_index INTEGER NOT NULL,
                    payload_checksum TEXT NOT NULL,
                    messages_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    last_error TEXT
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS flush_state (
                    session_id TEXT PRIMARY KEY,
                    last_flushed_message_index INTEGER,
                    last_flushed_turn_id TEXT,
                    payload_checksum TEXT,
                    updated_at TEXT NOT NULL
                )
                """
            )
            conn.commit()

    def _mark_error(self, row_ids: list[int], message: str) -> None:
        if not row_ids:
            return
        with self._connect() as conn:
            conn.executemany(
                "UPDATE pending_ingest SET last_error = ? WHERE id = ?",
                [(message, row_id) for row_id in row_ids],
            )
            conn.commit()

    def enqueue(
        self,
        identity: ResolvedRetainDBIdentity,
        *,
        turn_id: str,
        message_index: int,
        payload_checksum: str,
        messages: list[dict[str, Any]],
    ) -> None:
        self._identity_by_session[identity.session_id] = identity
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO pending_ingest (
                    session_id, user_id, agent_id, project, turn_id,
                    message_index, payload_checksum, messages_json, created_at, last_error
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, NULL)
                """,
                (
                    identity.session_id,
                    identity.user_id,
                    identity.agent_id,
                    identity.project,
                    turn_id,
                    message_index,
                    payload_checksum,
                    json.dumps(messages, ensure_ascii=False),
                    _now_iso(),
                ),
            )
            conn.commit()

        if self._config.write_frequency == "async" and self._queue is not None:
            self._queue.put(identity.session_id)

    def pending_session_ids(self) -> list[str]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT DISTINCT session_id FROM pending_ingest ORDER BY id ASC"
            ).fetchall()
        return [str(row["session_id"]) for row in rows]

    def _load_rows(self, session_id: str) -> list[sqlite3.Row]:
        with self._connect() as conn:
            return conn.execute(
                """
                SELECT * FROM pending_ingest
                WHERE session_id = ?
                ORDER BY id ASC
                LIMIT ?
                """,
                (session_id, int(self._config.flush_batch_size)),
            ).fetchall()

    def flush_session(self, identity: ResolvedRetainDBIdentity) -> bool:
        rows = self._load_rows(identity.session_id)
        if not rows:
            return True

        row_ids = [int(row["id"]) for row in rows]
        message_index = max(int(row["message_index"]) for row in rows)
        last_turn_id = str(rows[-1]["turn_id"])
        payload_checksum = str(rows[-1]["payload_checksum"])
        messages: list[dict[str, Any]] = []
        for row in rows:
            messages.extend(json.loads(str(row["messages_json"])))

        try:
            self._client.ingest_session(
                project=identity.project,
                session_id=identity.session_id,
                user_id=identity.user_id,
                messages=messages,
                write_mode="sync",
                timeout_ms=max(4000, self._config.prefetch_timeout_ms * 4),
            )
        except Exception as exc:
            self._mark_error(row_ids, str(exc))
            logger.warning("RetainDB ingest failed for %s: %s", identity.session_id, exc)
            return False

        with self._connect() as conn:
            conn.executemany(
                "DELETE FROM pending_ingest WHERE id = ?",
                [(row_id,) for row_id in row_ids],
            )
            conn.execute(
                """
                INSERT INTO flush_state (
                    session_id, last_flushed_message_index, last_flushed_turn_id,
                    payload_checksum, updated_at
                ) VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(session_id) DO UPDATE SET
                    last_flushed_message_index = excluded.last_flushed_message_index,
                    last_flushed_turn_id = excluded.last_flushed_turn_id,
                    payload_checksum = excluded.payload_checksum,
                    updated_at = excluded.updated_at
                """,
                (
                    identity.session_id,
                    message_index,
                    last_turn_id,
                    payload_checksum,
                    _now_iso(),
                ),
            )
            conn.commit()

        return True

    def flush_all(self) -> None:
        for session_id in self.pending_session_ids():
            identity = self._identity_by_session.get(session_id)
            if identity is None:
                continue
            self.flush_session(identity)

    def get_flush_state(self, session_id: str) -> dict[str, Any]:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM flush_state WHERE session_id = ?",
                (session_id,),
            ).fetchone()
        return dict(row) if row is not None else {}

    def _writer_loop(self) -> None:
        while True:
            try:
                item = self._queue.get(timeout=5)
                if item is _ASYNC_SHUTDOWN:
                    break
                session_id = str(item)
                identity = self._identity_by_session.get(session_id)
                if identity is None:
                    continue
                success = self.flush_session(identity)
                if not success:
                    time.sleep(2)
            except queue.Empty:
                continue
            except Exception as exc:
                logger.error("RetainDB async writer error: %s", exc)

    def shutdown(self) -> None:
        self.flush_all()
        if self._queue is not None and self._thread is not None:
            self._queue.put(_ASYNC_SHUTDOWN)
            self._thread.join(timeout=10)
