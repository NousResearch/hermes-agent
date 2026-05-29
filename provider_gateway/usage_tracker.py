"""SQLite usage tracking helpers for the opt-in provider gateway."""

from __future__ import annotations

import logging
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from hermes_constants import get_hermes_home

logger = logging.getLogger(__name__)

# Bump this when the schema changes.  _init_schema() stores the version and
# future migrations can inspect it before applying ALTER TABLE statements.
SCHEMA_VERSION = 1


@dataclass(frozen=True)
class ProviderUsageRecord:
    """One provider request outcome."""

    provider: str
    model: str
    api_mode: str
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    reasoning_tokens: int = 0
    estimated_cost_usd: float = 0.0
    latency_ms: float = 0.0
    status: str = "success"
    session_id: str | None = None
    error_type: str | None = None
    created_at: float | None = None


class ProviderUsageTracker:
    """Persist and summarize provider gateway usage records."""

    def __init__(self, db_path: str | Path | None = None) -> None:
        self.db_path = Path(db_path) if db_path is not None else get_hermes_home() / "provider_usage.db"
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    def record_usage(self, record: ProviderUsageRecord) -> int:
        """Insert one provider usage record and return its row id."""
        with self._connect() as conn:
            cursor = conn.execute(
                """
                INSERT INTO provider_usage (
                    created_at,
                    session_id,
                    provider,
                    model,
                    api_mode,
                    input_tokens,
                    output_tokens,
                    total_tokens,
                    cache_read_tokens,
                    cache_write_tokens,
                    reasoning_tokens,
                    estimated_cost_usd,
                    latency_ms,
                    status,
                    error_type
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record.created_at if record.created_at is not None else time.time(),
                    record.session_id,
                    record.provider,
                    record.model,
                    record.api_mode,
                    record.input_tokens,
                    record.output_tokens,
                    record.total_tokens,
                    record.cache_read_tokens,
                    record.cache_write_tokens,
                    record.reasoning_tokens,
                    record.estimated_cost_usd,
                    record.latency_ms,
                    record.status,
                    record.error_type,
                ),
            )
            conn.commit()
            return int(cursor.lastrowid)

    def summarize_by_provider(
        self,
        *,
        since: float | None = None,
        until: float | None = None,
    ) -> list[dict[str, Any]]:
        """Return aggregate request, token, cost, and latency totals by provider.

        Parameters
        ----------
        since:
            If given, only include records with ``created_at >= since``
            (Unix timestamp).
        until:
            If given, only include records with ``created_at <= until``
            (Unix timestamp).
        """
        conditions: list[str] = []
        params: list[Any] = []
        if since is not None:
            conditions.append("created_at >= ?")
            params.append(since)
        if until is not None:
            conditions.append("created_at <= ?")
            params.append(until)
        where_clause = f" WHERE {' AND '.join(conditions)}" if conditions else ""

        with self._connect() as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                f"""
                SELECT
                    provider,
                    COUNT(*) AS request_count,
                    SUM(CASE WHEN status = 'success' THEN 1 ELSE 0 END) AS success_count,
                    SUM(CASE WHEN status = 'success' THEN 0 ELSE 1 END) AS error_count,
                    COALESCE(SUM(total_tokens), 0) AS total_tokens,
                    COALESCE(SUM(estimated_cost_usd), 0.0) AS estimated_cost_usd,
                    COALESCE(AVG(latency_ms), 0.0) AS avg_latency_ms
                FROM provider_usage{where_clause}
                GROUP BY provider
                ORDER BY provider ASC
                """,
                params,
            ).fetchall()

        return [
            {
                "provider": row["provider"],
                "request_count": int(row["request_count"]),
                "success_count": int(row["success_count"]),
                "error_count": int(row["error_count"]),
                "total_tokens": int(row["total_tokens"]),
                "estimated_cost_usd": round(float(row["estimated_cost_usd"]), 6),
                "avg_latency_ms": round(float(row["avg_latency_ms"]), 2),
            }
            for row in rows
        ]

    def get_schema_version(self) -> int:
        """Return the current schema version stored in the database."""
        try:
            with self._connect() as conn:
                row = conn.execute(
                    "SELECT version FROM provider_usage_schema_version ORDER BY version DESC LIMIT 1"
                ).fetchone()
                return int(row[0]) if row else 0
        except Exception:
            return 0

    def _connect(self) -> sqlite3.Connection:
        """Create a new connection with recommended pragmas."""
        conn = sqlite3.connect(self.db_path)
        conn.execute("PRAGMA journal_mode = WAL")
        conn.execute("PRAGMA synchronous = NORMAL")
        conn.execute("PRAGMA busy_timeout = 5000")
        return conn

    def _init_schema(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS provider_usage_schema_version (
                    version INTEGER PRIMARY KEY,
                    applied_at REAL NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS provider_usage (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    created_at REAL NOT NULL,
                    session_id TEXT,
                    provider TEXT NOT NULL,
                    model TEXT NOT NULL,
                    api_mode TEXT NOT NULL,
                    input_tokens INTEGER NOT NULL DEFAULT 0,
                    output_tokens INTEGER NOT NULL DEFAULT 0,
                    total_tokens INTEGER NOT NULL DEFAULT 0,
                    cache_read_tokens INTEGER NOT NULL DEFAULT 0,
                    cache_write_tokens INTEGER NOT NULL DEFAULT 0,
                    reasoning_tokens INTEGER NOT NULL DEFAULT 0,
                    estimated_cost_usd REAL NOT NULL DEFAULT 0.0,
                    latency_ms REAL NOT NULL DEFAULT 0.0,
                    status TEXT NOT NULL,
                    error_type TEXT
                )
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_provider_usage_provider_created
                ON provider_usage(provider, created_at)
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_provider_usage_session
                ON provider_usage(session_id)
                """
            )
            # Record schema version if not yet stored.
            existing = conn.execute(
                "SELECT version FROM provider_usage_schema_version WHERE version = ?",
                (SCHEMA_VERSION,),
            ).fetchone()
            if existing is None:
                conn.execute(
                    "INSERT INTO provider_usage_schema_version (version, applied_at) VALUES (?, ?)",
                    (SCHEMA_VERSION, time.time()),
                )
            conn.commit()
