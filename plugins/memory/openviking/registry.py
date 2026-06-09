"""
Standalone OpenViking Session Registry — zero agent dependencies.

A process-safe, thread-safe SQLite registry that any code path (CLI, gateway,
cron, batch) can write to without importing the OpenViking memory provider.

Schema:
    sessions(session_id TEXT PRIMARY KEY,
             state TEXT,           -- CREATED|IN_SYNC|FINALIZING|COMMITTED|FAILED
             turn_count INT,
             message_count INT,
             source TEXT,          -- cli, cron, telegram, batch, api_server
             created_at TEXT,
             updated_at TEXT,
             commit_attempted_at TEXT,
             error TEXT,
             cached_messages TEXT)  -- JSON array of {role, content} dicts

Usage:
    from plugins.memory.openviking.registry import (
        register_session, update_state, get_uncommitted_sessions
    )
    register_session("session_abc123", source="cron")
    update_state("session_abc123", "COMMITTED")
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
import threading
import time
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_HERMES_HOME = os.path.expanduser("~/.hermes")
_SESSION_DB_PATH = os.path.join(_HERMES_HOME, "openviking-sessions.db")

# ---------------------------------------------------------------------------
# Lock — thread-safe across all registry operations
# ---------------------------------------------------------------------------
_lock = threading.Lock()

# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------
_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS sessions (
    session_id TEXT PRIMARY KEY,
    state TEXT NOT NULL DEFAULT 'CREATED',
    turn_count INTEGER NOT NULL DEFAULT 0,
    message_count INTEGER NOT NULL DEFAULT 0,
    source TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at TEXT NOT NULL DEFAULT (datetime('now')),
    commit_attempted_at TEXT,
    error TEXT,
    cached_messages TEXT NOT NULL DEFAULT '[]',
    retry_count INTEGER NOT NULL DEFAULT 0
);
"""


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def ensure_schema() -> None:
    """Create the sessions table if it does not already exist.

    Safe to call multiple times — uses IF NOT EXISTS.
    Also migrates older tables by adding missing columns.
    """
    try:
        os.makedirs(_HERMES_HOME, exist_ok=True)
        with _lock:
            with sqlite3.connect(_SESSION_DB_PATH, timeout=10.0) as conn:
                conn.execute(_SCHEMA_SQL)
                # Migration: add columns that may be missing from older schemas
                _migrate_add_column(conn, "sessions", "cached_messages", "TEXT NOT NULL DEFAULT '[]'")
                _migrate_add_column(conn, "sessions", "message_count", "INTEGER NOT NULL DEFAULT 0")
                _migrate_add_column(conn, "sessions", "source", "TEXT NOT NULL DEFAULT ''")
                _migrate_add_column(conn, "sessions", "retry_count", "INTEGER NOT NULL DEFAULT 0")
                conn.commit()
        # Phase 1 patch (2026-06-05, OV 3xstanbrain follow-on): install the
        # schema invariant that prevents unverified COMMITTED writes. The
        # trigger + verified_ov_dir column are part of registry_invariant.py
        # and are idempotent — safe to call on every ensure_schema().
        try:
            from plugins.memory.openviking.registry_invariant import install_invariant
            install_invariant()
        except Exception as exc:
            logger.debug("OpenViking registry invariant install failed: %s", exc)
    except Exception as exc:
        logger.debug("OpenViking registry ensure_schema failed: %s", exc)


def _migrate_add_column(conn: sqlite3.Connection, table: str, column: str, col_def: str) -> None:
    """Add a column to *table* if it doesn't already exist.

    Safe to call multiple times. Only adds columns that are missing.
    """
    try:
        pragma = conn.execute(f"PRAGMA table_info({table})").fetchall()
        existing = {row[1] for row in pragma}
        if column not in existing:
            conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {col_def}")
            logger.info("Migrated %s: added column %s", table, column)
    except Exception as exc:
        logger.debug("Migration failed for %s.%s: %s", table, column, exc)


def register_session(
    session_id: str,
    *,
    source: str = "",
    turn_count: int = 0,
    state: str = "CREATED",
) -> None:
    """Register a session in the registry.

    This is a no-op if the session_id already exists (upsert only updates
    source if source was empty). Call once per session lifecycle.
    """
    if not session_id:
        return
    try:
        ensure_schema()
        now = _now()
        with _lock:
            with sqlite3.connect(_SESSION_DB_PATH, timeout=10.0) as conn:
                conn.execute(
                    """
                    INSERT INTO sessions
                        (session_id, state, turn_count, source, created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?)
                    ON CONFLICT(session_id) DO UPDATE SET
                        source = CASE WHEN sessions.source = '' THEN excluded.source
                                      ELSE sessions.source END,
                        updated_at = excluded.updated_at
                    """,
                    (session_id, state, turn_count, source, now, now),
                )
                conn.commit()
    except Exception as exc:
        logger.debug("OpenViking registry register_session failed: %s", exc)


def update_state(
    session_id: str,
    state: str,
    *,
    turn_count: Optional[int] = None,
    messages: Optional[List[Dict[str, Any]]] = None,
    error: Optional[str] = None,
) -> None:
    """Update a session's state and optional metadata.

    State transitions are not enforced at the DB level — any string is
    accepted. The canonical lifecycle is:
        CREATED → IN_SYNC → FINALIZING → COMMITTED
                                      → FAILED
    """
    if not session_id:
        return
    try:
        ensure_schema()
        now = _now()
        msg_json = json.dumps(messages or [])
        tc = turn_count if turn_count is not None else -1  # sentinel for no update
        with _lock:
            with sqlite3.connect(_SESSION_DB_PATH, timeout=10.0) as conn:
                conn.execute(
                    """
                    INSERT INTO sessions
                        (session_id, state, turn_count, updated_at,
                         commit_attempted_at, error, cached_messages)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(session_id) DO UPDATE SET
                        state = excluded.state,
                        turn_count = CASE
                            WHEN excluded.turn_count >= 0 THEN excluded.turn_count
                            ELSE turn_count END,
                        updated_at = excluded.updated_at,
                        commit_attempted_at = CASE
                            WHEN excluded.state IN ('COMMITTED', 'FAILED', 'FINALIZING')
                            THEN excluded.updated_at
                            ELSE commit_attempted_at END,
                        error = CASE
                            WHEN excluded.state = 'FAILED' THEN ?
                            ELSE error END,
                        cached_messages = CASE
                            WHEN excluded.cached_messages != '[]'
                            THEN excluded.cached_messages
                            ELSE cached_messages END,
                        retry_count = CASE
                            WHEN excluded.state = 'FAILED'
                            THEN retry_count + 1
                            ELSE retry_count END
                    """,
                    (
                        session_id, state, tc, now,
                        now if state in ("COMMITTED", "FAILED", "FINALIZING") else None,
                        error or "", msg_json, error or "",
                    ),
                )
                conn.commit()
    except Exception as exc:
        logger.debug("OpenViking registry update_state failed: %s", exc)


def get_uncommitted_sessions(
    limit: int = 5000,
    offset: int = 0,
    max_retry: int = 3,
) -> List[Dict[str, Any]]:
    """Return sessions not yet in a terminal state (COMMITTED, FAILED, or DEAD).

    Sessions with retry_count >= max_retry are excluded (circuit breaker).

    Results ordered by created_at ASC.
    """
    if not _db_exists():
        return []
    try:
        with _lock:
            conn = sqlite3.connect(
                f"file:{_SESSION_DB_PATH}?mode=ro", uri=True, timeout=5.0,
            )
            conn.row_factory = sqlite3.Row
            try:
                rows = conn.execute(
                    """
                    SELECT * FROM sessions
                    WHERE state NOT IN ('COMMITTED', 'FAILED', 'DEAD')
                      AND (retry_count < ? OR retry_count IS NULL)
                    ORDER BY created_at ASC
                    LIMIT ? OFFSET ?
                    """,
                    (max_retry, limit, offset),
                ).fetchall()
                result = []
                for r in rows:
                    d = dict(r)
                    try:
                        d["cached_messages"] = json.loads(d.get("cached_messages", "[]"))
                    except Exception:
                        d["cached_messages"] = []
                    result.append(d)
                return result
            finally:
                conn.close()
    except Exception as exc:
        logger.debug("OpenViking registry get_uncommitted_sessions failed: %s", exc)
        return []


def get_session(session_id: str) -> Optional[Dict[str, Any]]:
    """Return a single session record, or None if not found."""
    if not _db_exists():
        return None
    try:
        with _lock:
            conn = sqlite3.connect(
                f"file:{_SESSION_DB_PATH}?mode=ro", uri=True, timeout=5.0,
            )
            conn.row_factory = sqlite3.Row
            try:
                row = conn.execute(
                    "SELECT * FROM sessions WHERE session_id = ?",
                    (session_id,),
                ).fetchone()
                if row:
                    d = dict(row)
                    try:
                        d["cached_messages"] = json.loads(d.get("cached_messages", "[]"))
                    except Exception:
                        d["cached_messages"] = []
                    return d
                return None
            finally:
                conn.close()
    except Exception as exc:
        logger.debug("OpenViking registry get_session failed: %s", exc)
        return None


def get_session_count() -> Dict[str, int]:
    """Return total number of sessions in the registry, by state."""
    if not _db_exists():
        return {}
    try:
        with _lock:
            conn = sqlite3.connect(
                f"file:{_SESSION_DB_PATH}?mode=ro", uri=True, timeout=5.0,
            )
            try:
                rows = conn.execute(
                    "SELECT state, COUNT(*) as cnt FROM sessions GROUP BY state"
                ).fetchall()
                return {r[0]: r[1] for r in rows}
            finally:
                conn.close()
    except Exception as exc:
        logger.debug("OpenViking registry get_session_count failed: %s", exc)
        return {}


def get_all_session_ids() -> List[str]:
    """Return all session IDs in the registry.

    Used for reconciliation against both Hermes state DB and OV API.
    """
    if not _db_exists():
        return []
    try:
        with _lock:
            conn = sqlite3.connect(
                f"file:{_SESSION_DB_PATH}?mode=ro", uri=True, timeout=5.0,
            )
            try:
                rows = conn.execute(
                    "SELECT session_id FROM sessions ORDER BY created_at ASC"
                ).fetchall()
                return [r[0] for r in rows]
            finally:
                conn.close()
    except Exception as exc:
        logger.debug("OpenViking registry get_all_session_ids failed: %s", exc)
        return []


def clear_stale_finalizing(max_age_minutes: int = 30) -> int:
    """Reset sessions stuck in FINALIZING for longer than max_age_minutes to CREATED.

    Returns the number of sessions unstuck.
    """
    if not _db_exists():
        return 0
    try:
        with _lock:
            conn = sqlite3.connect(_SESSION_DB_PATH, timeout=10.0)
            now = _now()
            try:
                # Use the exclude filter so only truly stuck sessions are caught
                c = conn.execute(
                    """
                    UPDATE sessions SET
                        state = 'CREATED',
                        error = 'unstuck from FINALIZING (stale > %d min)',
                        updated_at = ?
                    WHERE state = 'FINALIZING'
                      AND updated_at < datetime('now', ?)
                    """ % (max_age_minutes,),
                    (now, f'-{max_age_minutes} minutes'),
                )
                conn.commit()
                return c.rowcount
            finally:
                conn.close()
    except Exception as exc:
        logger.debug("OpenViking registry clear_stale_finalizing failed: %s", exc)
        return 0


def mark_dead(session_id: str, error: str = "max retries exceeded") -> None:
    """Mark a session as DEAD (terminal, will not be retried)."""
    update_state(session_id, "DEAD", error=error)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _db_exists() -> bool:
    """Check if the database file exists (fast path)."""
    return os.path.isfile(_SESSION_DB_PATH)


def _now() -> str:
    """Return ISO-8601 timestamp string."""
    return time.strftime("%Y-%m-%dT%H:%M:%S")
