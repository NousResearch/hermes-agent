"""Persistent storage for normalized failure records.

Uses the Hermes state DB (SQLite) with a dedicated normalized_failures table.
Follows the same patterns as agent.evals.storage.EvalStore.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import time
from pathlib import Path
from typing import Any, Optional

from hermes_constants import get_hermes_home
from .types import NormalizedFailure

logger = logging.getLogger(__name__)

FAILURE_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS normalized_failures (
    id TEXT PRIMARY KEY,
    created_at REAL NOT NULL,
    source_surface TEXT DEFAULT '',
    eval_run_id TEXT,
    case_id TEXT,
    session_id TEXT,
    task_id TEXT,
    failure_type TEXT NOT NULL,
    failure_subtype TEXT NOT NULL,
    severity TEXT DEFAULT 'medium',
    tool_name TEXT,
    model TEXT,
    provider TEXT,
    summary TEXT DEFAULT '',
    evidence_json TEXT,
    fingerprint TEXT DEFAULT ''
);

CREATE INDEX IF NOT EXISTS idx_nf_created ON normalized_failures(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_nf_fingerprint ON normalized_failures(fingerprint);
CREATE INDEX IF NOT EXISTS idx_nf_type ON normalized_failures(failure_type, failure_subtype);
CREATE INDEX IF NOT EXISTS idx_nf_eval_run ON normalized_failures(eval_run_id);
"""


class FailureStore:
    """SQLite storage for normalized failure records."""

    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = db_path or (get_hermes_home() / "state.db")
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._init_tables()

    def _init_tables(self) -> None:
        self._conn.executescript(FAILURE_SCHEMA_SQL)
        self._conn.commit()

    def close(self) -> None:
        if self._conn:
            self._conn.close()

    # ── Write ──

    def insert(self, failure: NormalizedFailure) -> None:
        """Insert a single normalized failure record."""
        self._conn.execute(
            "INSERT OR IGNORE INTO normalized_failures "
            "(id, created_at, source_surface, eval_run_id, case_id, "
            "session_id, task_id, failure_type, failure_subtype, severity, "
            "tool_name, model, provider, summary, evidence_json, fingerprint) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                failure.id, failure.created_at, failure.source_surface,
                failure.eval_run_id, failure.case_id,
                failure.session_id, failure.task_id,
                failure.failure_type, failure.failure_subtype, failure.severity,
                failure.tool_name, failure.model, failure.provider,
                failure.summary, failure.evidence_json, failure.fingerprint,
            ),
        )
        self._conn.commit()

    def insert_many(self, failures: list[NormalizedFailure]) -> int:
        """Insert multiple failures. Returns count inserted."""
        count = 0
        for f in failures:
            try:
                self.insert(f)
                count += 1
            except Exception:
                logger.debug("Skipped duplicate failure %s", f.id, exc_info=True)
        return count

    # ── Read ──

    def list_recent(self, limit: int = 20) -> list[dict[str, Any]]:
        """Return recent failures, newest first."""
        cur = self._conn.execute(
            "SELECT * FROM normalized_failures ORDER BY created_at DESC LIMIT ?",
            (limit,),
        )
        return [dict(row) for row in cur.fetchall()]

    def get_by_fingerprint(self, fingerprint: str) -> list[dict[str, Any]]:
        """Return all failures matching a fingerprint."""
        cur = self._conn.execute(
            "SELECT * FROM normalized_failures WHERE fingerprint = ? "
            "ORDER BY created_at DESC",
            (fingerprint,),
        )
        return [dict(row) for row in cur.fetchall()]

    def get_by_eval_run(self, eval_run_id: str) -> list[dict[str, Any]]:
        """Return all failures for a specific eval run."""
        cur = self._conn.execute(
            "SELECT * FROM normalized_failures WHERE eval_run_id = ? "
            "ORDER BY created_at DESC",
            (eval_run_id,),
        )
        return [dict(row) for row in cur.fetchall()]

    def top_fingerprints(
        self,
        window_seconds: float = 7 * 86400,
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        """Return top recurring failure fingerprints within a time window.

        Returns dicts with: fingerprint, failure_type, failure_subtype,
        count, latest_at, summary (from most recent occurrence).
        """
        cutoff = time.time() - window_seconds
        cur = self._conn.execute(
            """
            SELECT
                fingerprint,
                failure_type,
                failure_subtype,
                COUNT(*) as count,
                MAX(created_at) as latest_at,
                severity,
                tool_name,
                model
            FROM normalized_failures
            WHERE created_at >= ?
            GROUP BY fingerprint
            ORDER BY count DESC, latest_at DESC
            LIMIT ?
            """,
            (cutoff, limit),
        )
        rows = [dict(row) for row in cur.fetchall()]
        # Attach summary from the most recent occurrence
        for row in rows:
            latest = self._conn.execute(
                "SELECT summary FROM normalized_failures "
                "WHERE fingerprint = ? ORDER BY created_at DESC LIMIT 1",
                (row["fingerprint"],),
            ).fetchone()
            row["summary"] = latest["summary"] if latest else ""
        return rows

    def count_total(self, window_seconds: Optional[float] = None) -> int:
        """Count total failures, optionally within a time window."""
        if window_seconds is not None:
            cutoff = time.time() - window_seconds
            cur = self._conn.execute(
                "SELECT COUNT(*) FROM normalized_failures WHERE created_at >= ?",
                (cutoff,),
            )
        else:
            cur = self._conn.execute("SELECT COUNT(*) FROM normalized_failures")
        return cur.fetchone()[0]
