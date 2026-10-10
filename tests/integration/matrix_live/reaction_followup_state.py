"""Inspect persisted reaction watches and transcripts in the live gateway."""

from __future__ import annotations

import json
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


def failure_state(home: Path, room_id: str) -> dict:
    with closing(sqlite3.connect(f"file:{home / 'state.db'}?mode=ro", uri=True)) as db:
        db.row_factory = sqlite3.Row
        sessions = db.execute(
            "SELECT id, session_key, thread_id FROM sessions WHERE chat_id = ? LIMIT 16",
            (room_id,),
        ).fetchall()
        messages = db.execute(
            "SELECT m.id, m.session_id, m.role, m.content FROM messages m "
            "JOIN sessions s ON s.id = m.session_id WHERE s.chat_id = ? "
            "ORDER BY m.id DESC LIMIT 64",
            (room_id,),
        ).fetchall()
    watches = []
    for path in home.rglob("reaction-followups.sqlite"):
        with closing(sqlite3.connect(f"file:{path}?mode=ro", uri=True)) as db:
            db.row_factory = sqlite3.Row
            watches.extend(
                dict(row)
                for row in db.execute(
                    "SELECT * FROM watches WHERE room_id = ? LIMIT 16", (room_id,)
                )
            )
    cursors = {
        str(path.relative_to(home)): json.loads(path.read_text())["next_batch"]
        for path in home.rglob("sync-*.json")
    }
    return {
        "sessions": [dict(row) for row in sessions],
        "messages": [dict(row) for row in reversed(messages)],
        "watches": watches,
        "cursors": cursors,
    }
