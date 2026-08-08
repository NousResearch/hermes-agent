"""Durable profile-local store for external background tasks.

Owns the SQLite plumbing (schema, connections, transactions) for the plugin
external background-task lifecycle API in ``agent/background_tasks``. The
delivery rail itself stays in ``tools.async_delegation``; this module only
stores the plugin-owned task lifecycle rows and the persisted handle HMAC key.

The table lives in the SAME profile-local ``state.db`` as the async-delegation
registry so a terminal transition and its delivery-row insert are atomic in
one transaction. Everything here is private to the lifecycle API.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import random
import secrets
import sqlite3
import threading
import time
from contextlib import contextmanager
from typing import Any, Dict, Iterator

from hermes_constants import get_hermes_home

_HMAC_META_KEY = "ext_background_tasks.hmac_key"

_DB_LOCK = threading.Lock()

_SCHEMA = """
CREATE TABLE IF NOT EXISTS state_meta (
    key TEXT PRIMARY KEY,
    value TEXT
);
CREATE TABLE IF NOT EXISTS external_background_tasks (
    task_id TEXT PRIMARY KEY,
    plugin_id TEXT NOT NULL,
    parent_session_id TEXT NOT NULL,
    session_key TEXT NOT NULL DEFAULT '',
    origin_ui_session_id TEXT NOT NULL DEFAULT '',
    origin_session_id TEXT NOT NULL DEFAULT '',
    external_id TEXT NOT NULL,
    idempotency_key TEXT NOT NULL,
    label TEXT NOT NULL DEFAULT '',
    payload_hash TEXT NOT NULL,
    payload_json TEXT NOT NULL DEFAULT '{}',
    state TEXT NOT NULL,
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL,
    cancel_requested_at REAL,
    completed_at REAL,
    terminal_event_id TEXT,
    terminal_payload_hash TEXT,
    summary TEXT,
    error TEXT,
    result_json TEXT,
    delivery_delegation_id TEXT,
    delivery_state TEXT NOT NULL DEFAULT 'pending'
);
CREATE UNIQUE INDEX IF NOT EXISTS idx_ext_tasks_dedup
    ON external_background_tasks(plugin_id, parent_session_id, idempotency_key);
CREATE INDEX IF NOT EXISTS idx_ext_tasks_plugin_state
    ON external_background_tasks(plugin_id, state);
"""

_ROW_COLUMNS = (
    "task_id",
    "plugin_id",
    "parent_session_id",
    "session_key",
    "origin_ui_session_id",
    "origin_session_id",
    "external_id",
    "idempotency_key",
    "label",
    "payload_hash",
    "payload_json",
    "state",
    "created_at",
    "updated_at",
    "cancel_requested_at",
    "completed_at",
    "terminal_event_id",
    "terminal_payload_hash",
    "summary",
    "error",
    "result_json",
    "delivery_delegation_id",
    "delivery_state",
)


def _db_path():
    return get_hermes_home() / "state.db"


def connect() -> sqlite3.Connection:
    path = _db_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path, timeout=10)
    try:
        from hermes_state import apply_wal_with_fallback

        conn.execute("PRAGMA busy_timeout=10000")
        deadline = time.monotonic() + 10
        while True:
            try:
                apply_wal_with_fallback(conn, db_label="state.db (background_tasks)")
                break
            except sqlite3.OperationalError as exc:
                lock_contention = "locked" in str(exc).lower() or "busy" in str(exc).lower()
                if not lock_contention or time.monotonic() >= deadline:
                    raise
                time.sleep(random.uniform(0.01, 0.05))
        conn.executescript(_SCHEMA)
    except Exception:
        conn.close()
        raise
    return conn


@contextmanager
def transaction() -> Iterator[sqlite3.Connection]:
    """Open a connection, commit/rollback on exit, and ALWAYS close it."""
    conn = connect()
    try:
        with conn:
            yield conn
    finally:
        conn.close()


def load_or_create_hmac_key(conn: sqlite3.Connection) -> bytes:
    """Load the persisted profile-local handle key, creating it on first use."""
    owns_transaction = not conn.in_transaction
    if owns_transaction:
        conn.execute("BEGIN IMMEDIATE")
    try:
        candidate = secrets.token_bytes(32)
        conn.execute(
            "INSERT OR IGNORE INTO state_meta (key, value) VALUES (?, ?)",
            (_HMAC_META_KEY, candidate.hex()),
        )
        row = conn.execute(
            "SELECT value FROM state_meta WHERE key=?", (_HMAC_META_KEY,)
        ).fetchone()
        if row is None:
            raise RuntimeError("background-task HMAC key initialization failed")
        if owns_transaction:
            conn.commit()
        return bytes.fromhex(row[0])
    except Exception:
        if owns_transaction:
            conn.rollback()
        raise


def sign_handle(
    key: bytes, task_id: str, plugin_id: str, parent_session_id: str, created_at: float
) -> str:
    value = f"{task_id}|{plugin_id}|{parent_session_id}|{created_at:.6f}".encode(
        "utf-8"
    )
    return hmac.new(key, value, hashlib.sha256).hexdigest()


def row_to_dict(row: sqlite3.Row) -> Dict[str, Any]:
    return dict(zip(_ROW_COLUMNS, row))


def canonical_hash(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True).encode("utf-8")
    ).hexdigest()


def terminal_hash(
    status: str, event_id: str, summary: Any, error: Any, result_payload: Any
) -> str:
    payload = {
        "status": status,
        "event_id": event_id,
        "summary": summary,
        "error": error,
        "result_payload": result_payload,
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()
