"""Operator-requested system prompt refresh epochs (``hermes sessions refresh-prompts``).

A continuing session replays its stored system prompt every turn so the provider prefix cache
stays warm. After an operator updates the profile's prompt sources (SOUL.md, skills, project
context files) the verb records a refresh epoch here; a session whose stored prompt predates it
rebuilds once on its next turn (``agent/prompt_refresh.py``). Epochs live in ``state_meta`` of
the profile's own ``state.db``: durable across restarts, writable while the gateway runs, and
never seen by another profile. No row means no request, and the restore path behaves as before.
"""

from __future__ import annotations

from typing import Optional

PROFILE_KEY = "prompt_refresh.epoch"
SESSION_KEY_PREFIX = "prompt_refresh.epoch:"


class SessionPromptRefreshMixin:
    """Record and read refresh epochs. One counter orders profile-wide and per-session requests."""

    def request_prompt_refresh(self, session_id: Optional[str] = None) -> int:
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

        return self._execute_write(_do)

    def prompt_refresh_epoch(self, session_id: str) -> int:
        """The newest refresh epoch that applies to ``session_id`` (0 = never requested)."""
        row = self._read_one(
            "SELECT MAX(CAST(value AS INTEGER)) FROM state_meta WHERE key IN (?, ?)",
            (PROFILE_KEY, SESSION_KEY_PREFIX + session_id),
        )
        return int(row[0] or 0) if row else 0

    def count_sessions_with_stored_prompt(self) -> int:
        """Sessions whose next turn would replay a stored prompt (the ones a refresh rebuilds)."""
        row = self._read_one(
            "SELECT COUNT(*) FROM sessions WHERE system_prompt_hash IS NOT NULL OR system_prompt IS NOT NULL"
        )
        return int(row[0]) if row else 0
