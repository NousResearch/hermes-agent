"""Persistent storage for eval runs and case results.

Uses the Hermes state DB (SQLite) with eval-specific tables added via
schema migration. Follows the same patterns as SessionDB.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import time
from pathlib import Path
from typing import Any, Optional

from hermes_constants import get_hermes_home
from .types import CaseResult, CaseStatus, RunSummary

logger = logging.getLogger(__name__)

EVAL_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS eval_runs (
    id TEXT PRIMARY KEY,
    created_at REAL NOT NULL,
    label TEXT,
    suite_name TEXT NOT NULL,
    case_count INTEGER DEFAULT 0,
    passed_count INTEGER DEFAULT 0,
    failed_count INTEGER DEFAULT 0,
    avg_score REAL DEFAULT 0.0,
    notes TEXT
);

CREATE TABLE IF NOT EXISTS eval_case_results (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL REFERENCES eval_runs(id),
    case_id TEXT NOT NULL,
    category TEXT,
    status TEXT NOT NULL,
    deterministic_score REAL DEFAULT 0.0,
    total_score REAL DEFAULT 0.0,
    duration_ms INTEGER DEFAULT 0,
    failure_summary TEXT,
    raw_result_json TEXT
);

CREATE INDEX IF NOT EXISTS idx_eval_runs_created ON eval_runs(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_eval_case_results_run ON eval_case_results(run_id);
"""


class EvalStore:
    """SQLite storage for eval runs and case results."""

    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = db_path or (get_hermes_home() / "state.db")
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._init_tables()

    def _init_tables(self) -> None:
        self._conn.executescript(EVAL_SCHEMA_SQL)
        self._conn.commit()

    def close(self) -> None:
        if self._conn:
            self._conn.close()

    # ── Write operations ──

    def save_run(self, summary: RunSummary) -> None:
        """Persist a run and all its case results."""
        cur = self._conn.cursor()
        cur.execute(
            "INSERT INTO eval_runs (id, created_at, label, suite_name, "
            "case_count, passed_count, failed_count, avg_score) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                summary.run_id,
                summary.created_at,
                summary.label,
                summary.suite_name,
                summary.case_count,
                summary.passed_count,
                summary.failed_count,
                summary.avg_score,
            ),
        )
        for cr in summary.case_results:
            raw = json.dumps(cr.raw_result) if cr.raw_result else None
            cur.execute(
                "INSERT INTO eval_case_results "
                "(run_id, case_id, category, status, deterministic_score, "
                "total_score, duration_ms, failure_summary, raw_result_json) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    cr.run_id,
                    cr.case_id,
                    cr.category,
                    cr.status.value if isinstance(cr.status, CaseStatus) else cr.status,
                    cr.deterministic_score,
                    cr.total_score,
                    cr.duration_ms,
                    cr.failure_summary,
                    raw,
                ),
            )
        self._conn.commit()

        # Auto-ingest failed cases into the failure analysis subsystem.
        if summary.failed_count > 0:
            self._emit_failure_records(summary)

    def _emit_failure_records(self, summary: RunSummary) -> None:
        """Emit normalized failure records for failed/error eval cases."""
        try:
            from agent.failure_analysis.aggregator import ingest_eval_failures
            from agent.failure_analysis.storage import FailureStore

            case_dicts = []
            for cr in summary.case_results:
                status = cr.status.value if isinstance(cr.status, CaseStatus) else cr.status
                if status in ("passed", "skipped"):
                    continue
                case_dicts.append({
                    "case_id": cr.case_id,
                    "status": status,
                    "failure_summary": cr.failure_summary,
                    "deterministic_score": cr.deterministic_score,
                    "category": cr.category,
                })

            if case_dicts:
                fs = FailureStore(db_path=self.db_path)
                try:
                    ingest_eval_failures(summary.run_id, case_dicts, store=fs)
                finally:
                    fs.close()
        except Exception:
            logger.debug("Failed to emit failure records", exc_info=True)

    # ── Read operations ──

    def list_runs(self, limit: int = 10) -> list[dict[str, Any]]:
        """Return recent runs, newest first."""
        cur = self._conn.execute(
            "SELECT * FROM eval_runs ORDER BY created_at DESC LIMIT ?",
            (limit,),
        )
        return [dict(row) for row in cur.fetchall()]

    def get_run(self, run_id: str) -> Optional[dict[str, Any]]:
        """Return a single run record."""
        cur = self._conn.execute(
            "SELECT * FROM eval_runs WHERE id = ?", (run_id,)
        )
        row = cur.fetchone()
        return dict(row) if row else None

    def get_case_results(self, run_id: str) -> list[dict[str, Any]]:
        """Return all case results for a run."""
        cur = self._conn.execute(
            "SELECT * FROM eval_case_results WHERE run_id = ? ORDER BY id",
            (run_id,),
        )
        return [dict(row) for row in cur.fetchall()]

    def get_run_with_results(self, run_id: str) -> Optional[dict[str, Any]]:
        """Return a run record with its case results nested."""
        run = self.get_run(run_id)
        if not run:
            return None
        run["case_results"] = self.get_case_results(run_id)
        return run
