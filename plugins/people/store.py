"""Normalized SQLite store for cross-channel personal messages.

Default path: ``~/.hermes/people/messages.db`` (on-device only).
Incremental sync via UNIQUE(channel, ext_msg_id).
"""

from __future__ import annotations

import os
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, List, Optional, Sequence, Tuple


SCHEMA_VERSION = 1

_SCHEMA_SQL = """
PRAGMA journal_mode=WAL;

CREATE TABLE IF NOT EXISTS schema_meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS people (
    person_id TEXT PRIMARY KEY,
    display_name TEXT,
    slug TEXT UNIQUE,
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS identities (
    identity_id INTEGER PRIMARY KEY AUTOINCREMENT,
    person_id TEXT NOT NULL REFERENCES people(person_id),
    kind TEXT NOT NULL,          -- phone | email | handle | imessage | manual
    value TEXT NOT NULL,
    normalized TEXT NOT NULL,
    source TEXT,
    created_at REAL NOT NULL,
    UNIQUE(kind, normalized)
);

CREATE TABLE IF NOT EXISTS identity_overrides (
    override_id INTEGER PRIMARY KEY AUTOINCREMENT,
    kind TEXT NOT NULL,
    normalized TEXT NOT NULL,
    person_id TEXT NOT NULL REFERENCES people(person_id),
    note TEXT,
    created_at REAL NOT NULL,
    UNIQUE(kind, normalized)
);

CREATE TABLE IF NOT EXISTS messages (
    msg_id INTEGER PRIMARY KEY AUTOINCREMENT,
    person_id TEXT REFERENCES people(person_id),
    channel TEXT NOT NULL,
    ext_msg_id TEXT NOT NULL,
    direction TEXT NOT NULL CHECK(direction IN ('in', 'out', 'unknown')),
    ts REAL NOT NULL,
    body TEXT,
    thread_id TEXT,
    attachments_json TEXT,
    raw_handle TEXT,
    ingested_at REAL NOT NULL,
    UNIQUE(channel, ext_msg_id)
);

CREATE INDEX IF NOT EXISTS idx_messages_person_ts ON messages(person_id, ts);
CREATE INDEX IF NOT EXISTS idx_messages_channel_ts ON messages(channel, ts);
CREATE INDEX IF NOT EXISTS idx_identities_person ON identities(person_id);
"""


def default_db_path() -> Path:
    hermes_home = os.environ.get("HERMES_HOME") or str(Path.home() / ".hermes")
    return Path(hermes_home) / "people" / "messages.db"


@dataclass(frozen=True)
class IngestMessage:
    channel: str
    ext_msg_id: str
    direction: str  # in | out | unknown
    ts: float
    body: Optional[str] = None
    thread_id: Optional[str] = None
    attachments_json: Optional[str] = None
    raw_handle: Optional[str] = None
    person_id: Optional[str] = None


class PeopleMessageStore:
    def __init__(self, db_path: Optional[os.PathLike | str] = None) -> None:
        self.db_path = Path(db_path) if db_path else default_db_path()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.db_path))
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._init_schema()

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> "PeopleMessageStore":
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()

    def _init_schema(self) -> None:
        self._conn.executescript(_SCHEMA_SQL)
        cur = self._conn.execute(
            "SELECT value FROM schema_meta WHERE key = 'version'"
        )
        row = cur.fetchone()
        if row is None:
            self._conn.execute(
                "INSERT INTO schema_meta(key, value) VALUES ('version', ?)",
                (str(SCHEMA_VERSION),),
            )
            self._conn.commit()

    def upsert_person(
        self,
        person_id: str,
        *,
        display_name: Optional[str] = None,
        slug: Optional[str] = None,
    ) -> str:
        now = time.time()
        cur = self._conn.execute(
            "SELECT person_id FROM people WHERE person_id = ?", (person_id,)
        )
        if cur.fetchone() is None:
            self._conn.execute(
                "INSERT INTO people(person_id, display_name, slug, created_at, updated_at) "
                "VALUES (?, ?, ?, ?, ?)",
                (person_id, display_name, slug, now, now),
            )
        else:
            self._conn.execute(
                "UPDATE people SET display_name = COALESCE(?, display_name), "
                "slug = COALESCE(?, slug), updated_at = ? WHERE person_id = ?",
                (display_name, slug, now, person_id),
            )
        self._conn.commit()
        return person_id

    def link_identity(
        self,
        person_id: str,
        kind: str,
        value: str,
        *,
        normalized: Optional[str] = None,
        source: Optional[str] = None,
    ) -> None:
        from plugins.people.identity import normalize_identity

        norm = normalized or normalize_identity(kind, value)
        now = time.time()
        self.upsert_person(person_id)
        try:
            self._conn.execute(
                "INSERT INTO identities(person_id, kind, value, normalized, source, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (person_id, kind, value, norm, source, now),
            )
            self._conn.commit()
        except sqlite3.IntegrityError:
            # Already linked — leave as-is (deterministic identity resolution).
            self._conn.rollback()

    def set_manual_override(
        self,
        kind: str,
        value: str,
        person_id: str,
        *,
        note: Optional[str] = None,
    ) -> None:
        from plugins.people.identity import normalize_identity

        norm = normalize_identity(kind, value)
        now = time.time()
        self.upsert_person(person_id)
        self._conn.execute(
            "INSERT INTO identity_overrides(kind, normalized, person_id, note, created_at) "
            "VALUES (?, ?, ?, ?, ?) "
            "ON CONFLICT(kind, normalized) DO UPDATE SET "
            "person_id = excluded.person_id, note = excluded.note",
            (kind, norm, person_id, note, now),
        )
        self._conn.commit()

    def get_override_person(self, kind: str, value: str) -> Optional[str]:
        from plugins.people.identity import normalize_identity

        norm = normalize_identity(kind, value)
        cur = self._conn.execute(
            "SELECT person_id FROM identity_overrides WHERE kind = ? AND normalized = ?",
            (kind, norm),
        )
        row = cur.fetchone()
        return str(row["person_id"]) if row else None

    def find_person_by_identity(self, kind: str, value: str) -> Optional[str]:
        from plugins.people.identity import normalize_identity

        norm = normalize_identity(kind, value)
        cur = self._conn.execute(
            "SELECT person_id FROM identities WHERE kind = ? AND normalized = ?",
            (kind, norm),
        )
        row = cur.fetchone()
        return str(row["person_id"]) if row else None

    def ingest_many(self, messages: Sequence[IngestMessage]) -> Tuple[int, int]:
        """Insert messages with dedupe. Returns (inserted, skipped)."""
        inserted = 0
        skipped = 0
        now = time.time()
        for m in messages:
            direction = m.direction if m.direction in ("in", "out", "unknown") else "unknown"
            try:
                self._conn.execute(
                    "INSERT INTO messages("
                    "person_id, channel, ext_msg_id, direction, ts, body, "
                    "thread_id, attachments_json, raw_handle, ingested_at"
                    ") VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        m.person_id,
                        m.channel,
                        m.ext_msg_id,
                        direction,
                        float(m.ts),
                        m.body,
                        m.thread_id,
                        m.attachments_json,
                        m.raw_handle,
                        now,
                    ),
                )
                inserted += 1
            except sqlite3.IntegrityError:
                skipped += 1
        self._conn.commit()
        return inserted, skipped

    def list_people(self) -> List[sqlite3.Row]:
        return list(
            self._conn.execute(
                "SELECT * FROM people ORDER BY updated_at DESC"
            ).fetchall()
        )

    def messages_for_person(
        self, person_id: str, *, limit: int = 200
    ) -> List[sqlite3.Row]:
        return list(
            self._conn.execute(
                "SELECT * FROM messages WHERE person_id = ? ORDER BY ts DESC LIMIT ?",
                (person_id, int(limit)),
            ).fetchall()
        )

    def message_count(self) -> int:
        row = self._conn.execute("SELECT COUNT(*) AS c FROM messages").fetchone()
        return int(row["c"])
