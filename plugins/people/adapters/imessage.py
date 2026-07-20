"""Read-only iMessage adapter — macOS chat.db (Phase 1, #12323).

Never writes to chat.db. Callers pass an explicit path (tests use fixtures;
production uses ``~/Library/Messages/chat.db`` when the user has granted
Full Disk Access).
"""

from __future__ import annotations

import os
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, List, Optional

from plugins.people.identity import IdentityResolver, infer_kind
from plugins.people.store import IngestMessage, PeopleMessageStore

CHANNEL = "imessage"


@dataclass(frozen=True)
class RawIMessage:
    ext_msg_id: str
    ts: float
    body: str
    is_from_me: bool
    handle: str
    thread_id: str


def _apple_ns_to_unix(ns_since_2001: Optional[int]) -> float:
    """Apple Core Data timestamp → unix seconds."""
    if ns_since_2001 is None:
        return time.time()
    v = float(ns_since_2001)
    if abs(v) > 1e12:
        v = v / 1e9
    return v + 978307200.0


class IMessageAdapter:
    """Incremental reader for Apple Messages ``chat.db``."""

    def __init__(self, chat_db_path: os.PathLike | str) -> None:
        self.chat_db_path = Path(chat_db_path)

    def iter_messages(
        self,
        *,
        since_unix: Optional[float] = None,
        limit: int = 5000,
    ) -> Iterator[RawIMessage]:
        if not self.chat_db_path.is_file():
            return
            yield  # pragma: no cover — makes this a generator
        uri = f"file:{self.chat_db_path}?mode=ro"
        conn = sqlite3.connect(uri, uri=True)
        conn.row_factory = sqlite3.Row
        try:
            sql = """
            SELECT
                m.ROWID AS rowid,
                m.guid AS guid,
                m.text AS text,
                m.is_from_me AS is_from_me,
                m.date AS date,
                m.cache_roomnames AS room,
                h.id AS handle
            FROM message m
            LEFT JOIN handle h ON m.handle_id = h.ROWID
            WHERE m.text IS NOT NULL AND TRIM(m.text) != ''
            """
            params: list = []
            if since_unix is not None:
                cocoa = float(since_unix) - 978307200.0
                sql += (
                    " AND (CASE WHEN ABS(m.date) > 1e12 THEN m.date/1e9 "
                    "ELSE m.date END) > ?"
                )
                params.append(cocoa)
            sql += " ORDER BY m.date ASC LIMIT ?"
            params.append(int(limit))
            cur = conn.execute(sql, params)
            for row in cur:
                guid = row["guid"] or f"row-{row['rowid']}"
                handle = (row["handle"] or "").strip() or "unknown"
                yield RawIMessage(
                    ext_msg_id=str(guid),
                    ts=_apple_ns_to_unix(row["date"]),
                    body=str(row["text"] or ""),
                    is_from_me=bool(row["is_from_me"]),
                    handle=handle,
                    thread_id=str(row["room"] or handle),
                )
        finally:
            conn.close()

    def sync_to_store(
        self,
        store: PeopleMessageStore,
        *,
        resolver: Optional[IdentityResolver] = None,
        since_unix: Optional[float] = None,
        limit: int = 5000,
    ) -> tuple[int, int]:
        """Ingest messages with store-level dedupe. Returns (inserted, skipped)."""
        resolver = resolver or IdentityResolver(store)
        batch: List[IngestMessage] = []
        for raw in self.iter_messages(since_unix=since_unix, limit=limit):
            person_id = None
            if raw.handle and raw.handle != "unknown":
                person_id = resolver.resolve(
                    raw.handle,
                    kind=infer_kind(raw.handle),
                    source="imessage",
                )
            batch.append(
                IngestMessage(
                    channel=CHANNEL,
                    ext_msg_id=raw.ext_msg_id,
                    direction="out" if raw.is_from_me else "in",
                    ts=raw.ts,
                    body=raw.body,
                    thread_id=raw.thread_id,
                    raw_handle=raw.handle,
                    person_id=person_id,
                )
            )
        return store.ingest_many(batch)
