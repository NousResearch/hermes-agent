"""Operator-requested system prompt refresh epochs, kept in state_meta."""


from __future__ import annotations
from typing import Optional
from persistence.sqlite._holder import StoreHolder


PROFILE_KEY = "prompt_refresh.epoch"


SESSION_KEY_PREFIX = "prompt_refresh.epoch:"


def request_prompt_refresh(holder: StoreHolder, session_id: Optional[str] = None) -> int:
    """Record a refresh request for every session (or just ``session_id``); returns the epoch."""
    key = PROFILE_KEY if session_id is None else SESSION_KEY_PREFIX + session_id

    def _do(conn) -> int:
        row = conn.execute(
            "SELECT MAX(CAST(value AS INTEGER)) FROM state_meta WHERE key = ? OR substr(key, 1, ?) = ?",
            (PROFILE_KEY, len(SESSION_KEY_PREFIX), SESSION_KEY_PREFIX),
        ).fetchone()
        epoch = int(row[0] or 0) + 1
        conn.execute(
            "INSERT INTO state_meta (key, value) VALUES (?, ?) "
            "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            (key, str(epoch)),
        )
        return epoch

    return holder._execute_write(_do)


def prompt_refresh_epoch(holder: StoreHolder, session_id: str) -> int:
    """The newest refresh epoch that applies to ``session_id`` (0 = never requested)."""
    row = holder._read_one(
        "SELECT MAX(CAST(value AS INTEGER)) FROM state_meta WHERE key IN (?, ?)",
        (PROFILE_KEY, SESSION_KEY_PREFIX + session_id),
    )
    return int(row[0] or 0) if row else 0


def count_sessions_with_stored_prompt(holder: StoreHolder) -> int:
    """Sessions whose next turn would replay a stored prompt (the ones a refresh rebuilds)."""
    row = holder._read_one(
        "SELECT COUNT(*) FROM sessions WHERE system_prompt_hash IS NOT NULL OR system_prompt IS NOT NULL"
    )
    return int(row[0]) if row else 0
