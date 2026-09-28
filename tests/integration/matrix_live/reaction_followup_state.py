"""Inspect persisted reaction watches and transcripts in the live gateway."""

from __future__ import annotations

import sqlite3
from contextlib import closing
from pathlib import Path


def persisted_state(home: Path, room_id: str, thread_id: str) -> dict:
    watch_paths = list(home.rglob("reaction-followups.sqlite"))
    watches = []
    if watch_paths:
        assert len(watch_paths) == 1, watch_paths
        with closing(sqlite3.connect(f"file:{watch_paths[0]}?mode=ro", uri=True)) as db:
            watches = db.execute(
                "SELECT event_id, session_key, session_id, text_content FROM watches "
                "WHERE room_id = ? AND thread_id = ? ORDER BY event_id",
                (room_id, thread_id),
            ).fetchall()
    with closing(sqlite3.connect(f"file:{home / 'state.db'}?mode=ro", uri=True)) as db:
        sessions = db.execute(
            "SELECT id, session_key FROM sessions WHERE chat_id = ? AND thread_id = ?",
            (room_id, thread_id),
        ).fetchall()
        messages = db.execute(
            "SELECT m.session_id, m.role, m.content FROM messages m "
            "JOIN sessions s ON s.id = m.session_id "
            "WHERE s.chat_id = ? AND s.thread_id = ? ORDER BY m.id",
            (room_id, thread_id),
        ).fetchall()
    return {
        "watches": [list(row) for row in watches],
        "sessions": [list(row) for row in sessions],
        "messages": [list(row) for row in messages],
    }
