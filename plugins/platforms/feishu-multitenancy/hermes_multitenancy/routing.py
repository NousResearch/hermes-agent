"""SQLite-backed routing table for Feishu sender-to-profile routes.

Schema:
  - user_id PRIMARY KEY (business key for ops/feishu-sync)
  - profile_name (NOT UNIQUE — guest profile can serve multiple users)
  - open_id (queryable hot path, UNIQUE among active rows only)
  - union_id (cross-app stable id, optional)
  - active flag + deleted_at + synced_at + version (soft-delete + sync-staleness)

Read/write contract:
  - feishu-sync writes user_id / profile_name / open_id / union_id (and version++)
  - router only updates last_active_at via touch_active(open_id)
"""
from __future__ import annotations

import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

# Default db path — independent from ~/.hermes/state.db so router writes don't
# contend with gateway sessions/pairing/cron writes for the WAL.
DEFAULT_DB_PATH = Path.home() / ".hermes" / "multitenancy.db"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS multitenancy_routing (
    user_id        TEXT PRIMARY KEY NOT NULL,
    profile_name   TEXT NOT NULL,
    open_id        TEXT NOT NULL,
    union_id       TEXT,
    active         INTEGER NOT NULL DEFAULT 1,
    deleted_at     INTEGER,
    synced_at      INTEGER NOT NULL,
    version        INTEGER NOT NULL DEFAULT 1,
    last_active_at INTEGER,
    created_at     INTEGER NOT NULL,
    updated_at     INTEGER NOT NULL
);
CREATE UNIQUE INDEX IF NOT EXISTS idx_routing_open_id_active
    ON multitenancy_routing(open_id) WHERE active = 1;
CREATE INDEX IF NOT EXISTS idx_routing_active_user
    ON multitenancy_routing(active, user_id);
"""


@dataclass(frozen=True)
class RoutingRow:
    user_id: str
    profile_name: str
    open_id: str
    union_id: Optional[str]
    active: bool
    last_active_at: Optional[int]
    synced_at: int
    version: int


class RoutingTable:
    """SQLite-backed routing table.

    Use ``RoutingTable(":memory:")`` in tests for isolation.
    Use ``RoutingTable()`` (default) in production — points to
    ``~/.hermes/multitenancy.db``.
    """

    def __init__(self, db_path: Path | str | None = None) -> None:
        self.db_path = str(db_path) if db_path is not None else str(DEFAULT_DB_PATH)
        if self.db_path != ":memory:":
            Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        # check_same_thread=False so the same connection survives across the
        # asyncio task switches that the plugin does. SQLite operations are
        # short and serial within a single dispatch.
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript("PRAGMA journal_mode=WAL;")
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    # -- read path (router) -----------------------------------------------

    def lookup_by_open_id(self, open_id: str) -> Optional[RoutingRow]:
        """Return the active row for open_id, or None if missing/soft-deleted."""
        cur = self._conn.execute(
            "SELECT * FROM multitenancy_routing WHERE open_id = ? AND active = 1",
            (open_id,),
        )
        row = cur.fetchone()
        return _row_to_dataclass(row) if row else None

    def touch_active(self, open_id: str) -> None:
        """Update last_active_at — router-only, does NOT bump version."""
        self._conn.execute(
            "UPDATE multitenancy_routing SET last_active_at = ? WHERE open_id = ? AND active = 1",
            (_now(), open_id),
        )
        self._conn.commit()

    # -- write path (feishu-sync) ----------------------------------------

    def upsert(
        self,
        *,
        user_id: str,
        profile_name: str,
        open_id: str,
        union_id: Optional[str] = None,
    ) -> None:
        """Insert or refresh a route. Bumps version, sets synced_at to now."""
        now = _now()
        self._conn.execute(
            """
            INSERT INTO multitenancy_routing
                (user_id, profile_name, open_id, union_id, active,
                 synced_at, version, created_at, updated_at)
            VALUES (?, ?, ?, ?, 1, ?, 1, ?, ?)
            ON CONFLICT(user_id) DO UPDATE SET
                profile_name = excluded.profile_name,
                open_id      = excluded.open_id,
                union_id     = excluded.union_id,
                active       = 1,
                deleted_at   = NULL,
                synced_at    = excluded.synced_at,
                version      = version + 1,
                updated_at   = excluded.updated_at
            """,
            (user_id, profile_name, open_id, union_id, now, now, now),
        )
        self._conn.commit()

    def soft_delete(self, user_id: str) -> bool:
        """Mark a route as inactive. Returns True if a row was updated."""
        now = _now()
        cur = self._conn.execute(
            """
            UPDATE multitenancy_routing
            SET active = 0, deleted_at = ?, updated_at = ?, version = version + 1
            WHERE user_id = ? AND active = 1
            """,
            (now, now, user_id),
        )
        self._conn.commit()
        return cur.rowcount > 0

    # -- diagnostics -------------------------------------------------------

    def count_active(self) -> int:
        cur = self._conn.execute(
            "SELECT COUNT(*) FROM multitenancy_routing WHERE active = 1"
        )
        return int(cur.fetchone()[0])

    def close(self) -> None:
        self._conn.close()


def _row_to_dataclass(row: sqlite3.Row) -> RoutingRow:
    return RoutingRow(
        user_id=row["user_id"],
        profile_name=row["profile_name"],
        open_id=row["open_id"],
        union_id=row["union_id"],
        active=bool(row["active"]),
        last_active_at=row["last_active_at"],
        synced_at=row["synced_at"],
        version=row["version"],
    )


def _now() -> int:
    return int(time.time())
