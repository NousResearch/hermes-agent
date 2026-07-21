"""PostgreSQL entity sidecar — structural deduplication for explicit (infer=False) mem0 writes.

Prevents exact-text duplicates from accumulating in Qdrant. Only applied to
explicit mem0_add tool calls (infer=False). LLM-extracted memories (infer=True)
are handled by the monthly semantic dedup scan.

Table: sprint_automation.mem0_entity_names
  (user_id, entity_name_normalized) PRIMARY KEY → qdrant_point_id
"""

from __future__ import annotations

import logging
import os
import threading
from typing import Optional

logger = logging.getLogger(__name__)

_DSN = "dbname=sprint_automation host=localhost"
_INIT_LOCK = threading.Lock()
_INITIALIZED = False


def _connect():
    import psycopg2
    dsn = os.environ.get("HERMES_SIDECAR_DSN", _DSN)
    return psycopg2.connect(dsn)


def init_db() -> None:
    global _INITIALIZED
    with _INIT_LOCK:
        if _INITIALIZED:
            return
        try:
            conn = _connect()
            with conn:
                with conn.cursor() as cur:
                    cur.execute("""
                        CREATE TABLE IF NOT EXISTS mem0_entity_names (
                            user_id                TEXT NOT NULL,
                            entity_name_normalized TEXT NOT NULL,
                            qdrant_point_id        TEXT NOT NULL,
                            created_at             TIMESTAMPTZ NOT NULL DEFAULT now(),
                            updated_at             TIMESTAMPTZ NOT NULL DEFAULT now(),
                            PRIMARY KEY (user_id, entity_name_normalized)
                        )
                    """)
                    cur.execute("""
                        CREATE INDEX IF NOT EXISTS idx_mem0_entity_point
                            ON mem0_entity_names (qdrant_point_id)
                    """)
            conn.close()
            _INITIALIZED = True
        except Exception as exc:
            logger.warning("sidecar init_db error: %s", exc)


def normalize(text: str) -> str:
    return text.lower().strip()


def lookup(user_id: str, key: str) -> Optional[str]:
    """Return existing qdrant_point_id or None (key must already be normalized)."""
    try:
        conn = _connect()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT qdrant_point_id FROM mem0_entity_names WHERE user_id=%s AND entity_name_normalized=%s",
                    (user_id, key),
                )
                row = cur.fetchone()
                return row[0] if row else None
        finally:
            conn.close()
    except Exception as exc:
        logger.warning("sidecar lookup error: %s", exc)
        return None


def upsert(user_id: str, key: str, point_id: str) -> None:
    """Insert or update entity→point mapping."""
    try:
        conn = _connect()
        try:
            with conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                    INSERT INTO mem0_entity_names (user_id, entity_name_normalized, qdrant_point_id)
                    VALUES (%s, %s, %s)
                    ON CONFLICT (user_id, entity_name_normalized) DO UPDATE SET
                        qdrant_point_id = EXCLUDED.qdrant_point_id,
                        updated_at = now()
                    """,
                        (user_id, key, point_id),
                    )
        finally:
            conn.close()
    except Exception as exc:
        logger.warning("sidecar upsert error: %s", exc)


def delete_by_point_id(point_id: str) -> None:
    """Remove sidecar entry when a Qdrant point is deleted via mem0_delete."""
    _write("DELETE FROM mem0_entity_names WHERE qdrant_point_id=%s", (point_id,))


def delete_stale(user_id: str, key: str) -> None:
    """Remove a stale entry whose Qdrant point no longer exists."""
    _write(
        "DELETE FROM mem0_entity_names WHERE user_id=%s AND entity_name_normalized=%s",
        (user_id, key),
    )


def _write(sql: str, params: tuple) -> None:
    try:
        conn = _connect()
        try:
            with conn:
                with conn.cursor() as cur:
                    cur.execute(sql, params)
        finally:
            conn.close()
    except Exception as exc:
        logger.warning("sidecar write error: %s", exc)
