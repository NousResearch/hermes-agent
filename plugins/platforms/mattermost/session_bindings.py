"""Persistent Hermes-session to Mattermost-thread bindings."""

from __future__ import annotations

import re
import sqlite3
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from hermes_cli.sqlite_util import open_db, transaction
from plugins.plugin_storage import plugin_db

PLUGIN_STORAGE_NAME = "mattermost-platform"
DATABASE_FILENAME = "session-bindings.db"
SCHEMA_VERSION = 1

MAX_SESSION_ID_LENGTH = 256
MAX_MATTERMOST_ID_LENGTH = 64

_SESSION_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:@-]*$")
_MATTERMOST_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]*$")


class BindingValidationError(ValueError):
    """A binding identifier is missing, malformed, or too long."""


@dataclass(frozen=True)
class SessionBinding:
    session_id: str
    channel_id: str
    root_post_id: str
    created_at: float
    updated_at: float

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def normalize_session_id(value: Any) -> str:
    return _normalize_identifier(
        value,
        field="session_id",
        maximum=MAX_SESSION_ID_LENGTH,
        pattern=_SESSION_ID_RE,
    )


def normalize_mattermost_id(value: Any, *, field: str) -> str:
    if field not in {"channel_id", "root_post_id"}:
        raise ValueError(f"unsupported Mattermost identifier field: {field}")
    return _normalize_identifier(
        value,
        field=field,
        maximum=MAX_MATTERMOST_ID_LENGTH,
        pattern=_MATTERMOST_ID_RE,
    )


def _normalize_identifier(value: Any, *, field: str, maximum: int, pattern: re.Pattern[str]) -> str:
    if not isinstance(value, str):
        raise BindingValidationError(f"{field} must be a string")
    normalized = value.strip()
    if not normalized:
        raise BindingValidationError(f"{field} is required")
    if len(normalized) > maximum:
        raise BindingValidationError(f"{field} exceeds {maximum} characters")
    if not pattern.fullmatch(normalized):
        raise BindingValidationError(f"{field} contains invalid characters")
    return normalized


class MattermostSessionBindingStore:
    """Small profile-scoped SQLite store with deterministic replacement."""

    def __init__(self, path: Path | None = None):
        self._path = Path(path) if path is not None else None
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        if self._path is None:
            conn = plugin_db(PLUGIN_STORAGE_NAME, DATABASE_FILENAME)
            conn.row_factory = sqlite3.Row
            return conn
        return open_db(
            self._path,
            db_label=f"plugin-data/{PLUGIN_STORAGE_NAME}/{DATABASE_FILENAME}",
            foreign_keys=True,
            row_factory=sqlite3.Row,
            check_same_thread=False,
        )

    def _initialize(self) -> None:
        conn = self._connect()
        try:
            current = int(conn.execute("PRAGMA user_version").fetchone()[0])
            if current > SCHEMA_VERSION:
                raise RuntimeError(
                    f"session binding schema {current} is newer than supported version {SCHEMA_VERSION}"
                )
            with conn:
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS session_bindings (
                        session_id TEXT PRIMARY KEY,
                        channel_id TEXT NOT NULL,
                        root_post_id TEXT NOT NULL,
                        created_at REAL NOT NULL,
                        updated_at REAL NOT NULL,
                        UNIQUE (channel_id, root_post_id)
                    )
                    """
                )
                conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_session_bindings_thread "
                    "ON session_bindings(channel_id, root_post_id)"
                )
                if current < SCHEMA_VERSION:
                    conn.execute(f"PRAGMA user_version={SCHEMA_VERSION}")
        finally:
            conn.close()

    @staticmethod
    def _from_row(row: sqlite3.Row | None) -> SessionBinding | None:
        if row is None:
            return None
        return SessionBinding(
            session_id=row["session_id"],
            channel_id=row["channel_id"],
            root_post_id=row["root_post_id"],
            created_at=float(row["created_at"]),
            updated_at=float(row["updated_at"]),
        )

    def replace(self, session_id: Any, channel_id: Any, root_post_id: Any) -> SessionBinding:
        session = normalize_session_id(session_id)
        channel = normalize_mattermost_id(channel_id, field="channel_id")
        root = normalize_mattermost_id(root_post_id, field="root_post_id")
        now = time.time()
        with transaction(self._connect(), immediate=True) as conn:
            conn.execute(
                "DELETE FROM session_bindings "
                "WHERE session_id = ? OR (channel_id = ? AND root_post_id = ?)",
                (session, channel, root),
            )
            conn.execute(
                "INSERT INTO session_bindings "
                "(session_id, channel_id, root_post_id, created_at, updated_at) "
                "VALUES (?, ?, ?, ?, ?)",
                (session, channel, root, now, now),
            )
        # Return the row this transaction committed instead of reading it back. A
        # concurrent replacement may legitimately take ownership immediately after
        # our commit; a post-commit SELECT would then turn a successful write into a
        # spurious assertion failure.
        return SessionBinding(
            session_id=session,
            channel_id=channel,
            root_post_id=root,
            created_at=now,
            updated_at=now,
        )

    def get_by_session(self, session_id: Any) -> SessionBinding | None:
        session = normalize_session_id(session_id)
        conn = self._connect()
        try:
            row = conn.execute(
                "SELECT session_id, channel_id, root_post_id, created_at, updated_at "
                "FROM session_bindings WHERE session_id = ?",
                (session,),
            ).fetchone()
            return self._from_row(row)
        finally:
            conn.close()

    def resolve(self, channel_id: Any, root_post_id: Any) -> SessionBinding | None:
        channel = normalize_mattermost_id(channel_id, field="channel_id")
        root = normalize_mattermost_id(root_post_id, field="root_post_id")
        conn = self._connect()
        try:
            row = conn.execute(
                "SELECT session_id, channel_id, root_post_id, created_at, updated_at "
                "FROM session_bindings WHERE channel_id = ? AND root_post_id = ?",
                (channel, root),
            ).fetchone()
            return self._from_row(row)
        finally:
            conn.close()

    def list_bindings(self, *, limit: int = 100, offset: int = 0) -> list[SessionBinding]:
        """Return bindings ordered by most recently replaced first."""
        safe_limit = max(0, min(int(limit), 200))
        safe_offset = max(0, int(offset))
        conn = self._connect()
        try:
            rows = conn.execute(
                "SELECT session_id, channel_id, root_post_id, created_at, updated_at "
                "FROM session_bindings ORDER BY updated_at DESC, session_id ASC LIMIT ? OFFSET ?",
                (safe_limit, safe_offset),
            ).fetchall()
            return [binding for row in rows if (binding := self._from_row(row)) is not None]
        finally:
            conn.close()

    def delete(self, session_id: Any) -> bool:
        session = normalize_session_id(session_id)
        with transaction(self._connect(), immediate=True) as conn:
            cursor = conn.execute("DELETE FROM session_bindings WHERE session_id = ?", (session,))
            return cursor.rowcount > 0
