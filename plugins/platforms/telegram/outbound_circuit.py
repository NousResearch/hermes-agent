"""Small process-shared Telegram PEER_FLOOD circuit.

Only a normalized chat key and wall-clock expiry are stored in the profile's
existing ``state.db``. Message bodies and credentials never enter this table.
"""

from __future__ import annotations

import asyncio
import logging
import math
import sqlite3
import threading
import time
from contextlib import contextmanager
from typing import Iterator

from hermes_constants import get_hermes_home
from plugins.platforms.telegram.telegram_ids import telegram_chat_id_key

logger = logging.getLogger(__name__)

DEFAULT_COOLDOWN_SECONDS = 300.0
MAX_COOLDOWN_SECONDS = 300.0
DB_BUSY_TIMEOUT_SECONDS = 0.05
DB_ERROR_COOLDOWN_SECONDS = 5.0
GLOBAL_CIRCUIT_KEY = "telegram:global"
_DB_LOCK = threading.Lock()
_ERROR_STATE_LOCK = threading.Lock()
_READ_ERROR_UNTIL: dict[tuple[str, str], float] = {}
_WRITE_ERROR_UNTIL: dict[tuple[str, str], float] = {}


def chat_key(chat_id: object) -> str:
    """Return the canonical key shared by adapter and standalone sender."""
    return telegram_chat_id_key(str(chat_id).strip())


def _db_path():
    return get_hermes_home() / "state.db"


@contextmanager
def _connection() -> Iterator[sqlite3.Connection]:
    path = _db_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path, timeout=DB_BUSY_TIMEOUT_SECONDS)
    try:
        from hermes_state import apply_wal_with_fallback

        apply_wal_with_fallback(conn, db_label="state.db (telegram outbound circuit)")
        conn.execute(
            """CREATE TABLE IF NOT EXISTS telegram_peer_flood_circuits (
                chat_key TEXT PRIMARY KEY,
                expires_at REAL NOT NULL
            )"""
        )
        with conn:
            yield conn
    finally:
        conn.close()


def _bounded_delay(delay: object) -> float:
    try:
        value = float(delay)
        if not math.isfinite(value) or value <= 0:
            raise ValueError
    except (TypeError, ValueError):
        value = DEFAULT_COOLDOWN_SECONDS
    return min(value, MAX_COOLDOWN_SECONDS)


def open_circuit(chat_id: object, delay: object = None) -> float:
    """Persist a cooldown, preserving a later expiry already opened elsewhere."""
    bounded = _bounded_delay(delay)
    expires_at = time.time() + bounded
    try:
        if not _DB_LOCK.acquire(timeout=DB_BUSY_TIMEOUT_SECONDS):
            raise sqlite3.OperationalError("Telegram circuit database lock timed out")
        try:
            with _connection() as conn:
                conn.execute(
                    """INSERT INTO telegram_peer_flood_circuits(chat_key, expires_at)
                       VALUES (?, ?)
                       ON CONFLICT(chat_key) DO UPDATE SET expires_at =
                           MAX(telegram_peer_flood_circuits.expires_at, excluded.expires_at)""",
                    (chat_key(chat_id), expires_at),
                )
                stored = conn.execute(
                    "SELECT expires_at FROM telegram_peer_flood_circuits WHERE chat_key=?",
                    (chat_key(chat_id),),
                ).fetchone()
        finally:
            _DB_LOCK.release()
        return max(0.0, float(stored[0]) - time.time()) if stored else bounded
    except (OSError, sqlite3.Error):
        logger.warning("Could not persist Telegram PEER_FLOOD circuit")
        failure_key = (str(_db_path()), chat_key(chat_id))
        with _ERROR_STATE_LOCK:
            _WRITE_ERROR_UNTIL[failure_key] = max(
                _WRITE_ERROR_UNTIL.get(failure_key, 0.0),
                time.monotonic() + DB_ERROR_COOLDOWN_SECONDS,
            )
        return bounded


async def open_circuit_async(chat_id: object, delay: object = None) -> float:
    """Persist a cooldown in a worker thread for asynchronous callers."""
    return await asyncio.to_thread(open_circuit, chat_id, delay)


def remaining(chat_id: object) -> float | None:
    """Return cooldown; DB errors fail closed once for a finite short window.

    After that window one call is allowed through before a persistent error can
    open another window. This prioritizes PEER_FLOOD suppression without making
    an unavailable state database an unconditional permanent outage.
    """
    now = time.time()
    path_key = str(_db_path())
    monotonic_now = time.monotonic()
    failure_key = (path_key, chat_key(chat_id))
    with _ERROR_STATE_LOCK:
        write_error_until = _WRITE_ERROR_UNTIL.get(failure_key)
        if write_error_until is not None:
            left = write_error_until - monotonic_now
            if left > 0:
                return left
            _WRITE_ERROR_UNTIL.pop(failure_key, None)
        error_until = _READ_ERROR_UNTIL.get(failure_key)
        if error_until is not None:
            left = error_until - monotonic_now
            if left > 0:
                return left
            _READ_ERROR_UNTIL.pop(failure_key, None)
            return None
    try:
        if not _DB_LOCK.acquire(timeout=DB_BUSY_TIMEOUT_SECONDS):
            raise sqlite3.OperationalError("Telegram circuit database lock timed out")
        try:
            with _connection() as conn:
                row = conn.execute(
                    "SELECT expires_at FROM telegram_peer_flood_circuits WHERE chat_key=?",
                    (chat_key(chat_id),),
                ).fetchone()
                if row is None:
                    return None
                left = float(row[0]) - now
                if left <= 0:
                    conn.execute(
                        "DELETE FROM telegram_peer_flood_circuits WHERE chat_key=?",
                        (chat_key(chat_id),),
                    )
                    return None
                return left
        finally:
            _DB_LOCK.release()
    except (OSError, sqlite3.Error):
        logger.warning("Could not read Telegram PEER_FLOOD circuit")
        with _ERROR_STATE_LOCK:
            _READ_ERROR_UNTIL[failure_key] = (
                time.monotonic() + DB_ERROR_COOLDOWN_SECONDS
            )
        return DB_ERROR_COOLDOWN_SECONDS


async def remaining_async(chat_id: object) -> float | None:
    """Read a cooldown in a worker thread for asynchronous callers."""
    return await asyncio.to_thread(remaining, chat_id)
