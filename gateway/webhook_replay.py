"""Durable replay claims for privileged webhook profile handoffs.

Ordinary webhook delivery IDs are transport hints and remain covered by the
adapter's short-lived in-memory cache. Trusted handoffs can expose terminal or
file capabilities, so their signed delivery identity is claimed atomically in
``state.db`` before dispatch and is never expired automatically. Keeping only a
SHA-256 key makes the durable security boundary small without persisting task
content or authentication material.
"""

from __future__ import annotations

import hashlib
import sqlite3
import threading
import time
from contextlib import contextmanager
from typing import Iterator

from hermes_constants import get_hermes_home

_DB_LOCK = threading.Lock()


def _replay_key(route_name: str, source_profile: str, delivery_id: str) -> str:
    material = "\0".join((route_name, source_profile, delivery_id))
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


def _connect() -> sqlite3.Connection:
    path = get_hermes_home() / "state.db"
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path, timeout=10)
    try:
        conn.execute("PRAGMA busy_timeout=10000")
        from hermes_state import apply_wal_with_fallback

        apply_wal_with_fallback(conn, db_label="state.db (webhook_handoff_replay)")
        conn.execute("PRAGMA synchronous=FULL")
        conn.execute(
            """CREATE TABLE IF NOT EXISTS webhook_handoff_receipts (
                replay_key TEXT PRIMARY KEY,
                claimed_at REAL NOT NULL
            )"""
        )
    except Exception:
        conn.close()
        raise
    return conn


@contextmanager
def _transaction() -> Iterator[sqlite3.Connection]:
    conn = _connect()
    try:
        with conn:
            yield conn
    finally:
        conn.close()


def claim_handoff_delivery(
    *, route_name: str, source_profile: str, delivery_id: str
) -> bool:
    """Atomically claim one authenticated handoff delivery.

    Returns ``True`` exactly once for a route/profile/delivery tuple, including
    across gateway reconstruction and process restarts. Database failures are
    intentionally propagated so the request handler can fail closed before
    dispatching privileged work.
    """

    replay_key = _replay_key(route_name, source_profile, delivery_id)
    with _DB_LOCK, _transaction() as conn:
        cursor = conn.execute(
            """INSERT OR IGNORE INTO webhook_handoff_receipts
               (replay_key, claimed_at) VALUES (?, ?)""",
            (replay_key, time.time()),
        )
        return cursor.rowcount == 1


def is_handoff_delivery_claimed(
    *, route_name: str, source_profile: str, delivery_id: str
) -> bool:
    """Return whether an authenticated handoff delivery was already claimed."""

    replay_key = _replay_key(route_name, source_profile, delivery_id)
    with _DB_LOCK, _transaction() as conn:
        row = conn.execute(
            "SELECT 1 FROM webhook_handoff_receipts WHERE replay_key = ?",
            (replay_key,),
        ).fetchone()
        return row is not None
