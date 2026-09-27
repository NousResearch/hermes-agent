"""Durable, single-use watches for reactions to delivered Matrix replies."""

from __future__ import annotations

import json
import sqlite3
import time
from collections.abc import Callable, Iterable
from contextlib import closing
from pathlib import Path
from typing import Any


WATCH_SECONDS = 600


class ReactionWatchStore:
    def __init__(self, path: Path, *, clock: Callable[[], float] = time.time) -> None:
        self.path = path
        self.clock = clock
        path.parent.mkdir(parents=True, exist_ok=True)
        with closing(self._connect()) as db, db:
            db.execute("""
                CREATE TABLE IF NOT EXISTS watches (
                    event_id TEXT PRIMARY KEY,
                    turn_id TEXT NOT NULL,
                    profile TEXT NOT NULL,
                    room_id TEXT NOT NULL,
                    thread_id TEXT NOT NULL,
                    session_key TEXT NOT NULL,
                    session_id TEXT NOT NULL,
                    requester TEXT NOT NULL,
                    source_json TEXT NOT NULL,
                    emoji_json TEXT NOT NULL,
                    text_content TEXT NOT NULL DEFAULT '',
                    expires_at REAL NOT NULL
                )
            """)
            columns = {row[1] for row in db.execute("PRAGMA table_info(watches)")}
            if "session_id" not in columns:
                db.execute("ALTER TABLE watches ADD COLUMN session_id TEXT NOT NULL DEFAULT ''")
            if "text_content" not in columns:
                db.execute("ALTER TABLE watches ADD COLUMN text_content TEXT NOT NULL DEFAULT ''")

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.path, timeout=5)

    def arm(
        self,
        turn_id: str,
        event_ids: Iterable[str],
        *,
        profile: str,
        room_id: str,
        thread_id: str,
        session_key: str,
        session_id: str,
        requester: str,
        source: dict[str, Any],
        emoji_filter: tuple[str, ...],
        text_content: str = "",
    ) -> None:
        rows = [
            (
                event_id,
                turn_id,
                profile,
                room_id,
                thread_id,
                session_key,
                session_id,
                requester,
                json.dumps(source),
                json.dumps(emoji_filter),
                text_content,
                self.clock() + WATCH_SECONDS,
            )
            for event_id in event_ids
            if event_id
        ]
        if not rows:
            return
        with closing(self._connect()) as db, db:
            db.execute("DELETE FROM watches WHERE expires_at <= ?", (self.clock(),))
            db.executemany(
                """INSERT OR REPLACE INTO watches
                   (event_id, turn_id, profile, room_id, thread_id, session_key,
                    session_id, requester, source_json, emoji_json, text_content, expires_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                rows,
            )

    def claim(
        self,
        profile: str,
        room_id: str,
        target_event_id: str,
        sender: str,
        emoji: str,
        *,
        reaction_time: float | None = None,
    ) -> dict[str, Any] | None:
        with closing(self._connect()) as db, db:
            db.execute("BEGIN IMMEDIATE")
            db.execute("DELETE FROM watches WHERE expires_at <= ?", (self.clock(),))
            row = db.execute(
                """
                SELECT turn_id, thread_id, session_key, session_id, requester,
                       source_json, emoji_json, expires_at, text_content
                FROM watches WHERE event_id = ? AND profile = ? AND room_id = ?
            """,
                (target_event_id, profile, room_id),
            ).fetchone()
            if row is None or row[4] != sender or not row[3]:
                return None
            if reaction_time is not None and reaction_time < row[7] - WATCH_SECONDS:
                return None
            allowed = tuple(json.loads(row[6]))
            if allowed and emoji not in allowed:
                return None
            db.execute("DELETE FROM watches WHERE turn_id = ?", (row[0],))
        return {
            "profile": profile,
            "room_id": room_id,
            "thread_id": row[1],
            "session_key": row[2],
            "session_id": row[3],
            "requester": sender,
            "source": json.loads(row[5]),
            "emoji": emoji,
            "target_event_id": target_event_id,
            "text_content": row[8],
        }

    def candidate(self, room_id: str, target_event_id: str) -> dict[str, Any] | None:
        with closing(self._connect()) as db, db:
            row = db.execute(
                """
                SELECT profile, thread_id, session_key, session_id, requester,
                       source_json, expires_at
                FROM watches WHERE event_id = ? AND room_id = ?
            """,
                (target_event_id, room_id),
            ).fetchone()
        if row is None or row[6] <= self.clock():
            return None
        return {
            "profile": row[0],
            "thread_id": row[1],
            "session_key": row[2],
            "session_id": row[3],
            "requester": row[4],
            "source": json.loads(row[5]),
        }
