"""Persistent storage for evolution traces, results, and state.

Uses SQLite (same pattern as hermes_state.py) for evolution history.
Each evolution run creates a record with full trace data, evaluation
results, improvement proposals, and outcome tracking.

Schema::

    CREATE TABLE evolution_runs (
        run_id TEXT PRIMARY KEY,
        task_name TEXT NOT NULL,
        task_domain TEXT,
        task_complexity INTEGER,
        status TEXT NOT NULL DEFAULT 'pending',
        session_id TEXT,
        iterations INTEGER DEFAULT 0,
        max_iterations INTEGER DEFAULT 5,
        final_score REAL,
        created_at TEXT NOT NULL,
        completed_at TEXT,
        trace_path TEXT,
        summary TEXT
    );

    CREATE TABLE evolution_iterations (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        run_id TEXT NOT NULL REFERENCES evolution_runs(run_id),
        iteration_num INTEGER NOT NULL,
        status TEXT NOT NULL,  -- 'attempting', 'evaluating', 'analyzing', 'improving', 'done'
        score REAL,
        improvement_action TEXT,
        improvement_target TEXT,
        trace_json TEXT,       -- compressed trajectory for this iteration
        eval_json TEXT,        -- evaluation results
        analysis_json TEXT,    -- failure analysis (if failed)
        proposal_json TEXT,    -- improvement proposal
        error_message TEXT,
        started_at TEXT NOT NULL,
        completed_at TEXT,
        token_usage_json TEXT
    );

    CREATE TABLE regression_baseline (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        task_name TEXT NOT NULL UNIQUE,
        task_domain TEXT,
        baseline_score REAL NOT NULL,
        trace_path TEXT,
        last_verified_at TEXT NOT NULL,
        harness_variant TEXT DEFAULT 'default'
    );

    CREATE INDEX idx_evolution_runs_task ON evolution_runs(task_name);
    CREATE INDEX idx_evolution_runs_status ON evolution_runs(status);
    CREATE INDEX idx_evolution_iterations_run ON evolution_iterations(run_id);
    CREATE INDEX idx_regression_baseline_task ON regression_baseline(task_name);
"""

from __future__ import annotations

import json
import logging
import sqlite3
import threading
import time
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

from hermes_constants import get_hermes_home

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS evolution_runs (
    run_id TEXT PRIMARY KEY,
    task_name TEXT NOT NULL,
    task_domain TEXT,
    task_complexity INTEGER,
    status TEXT NOT NULL DEFAULT 'pending',
    session_id TEXT,
    iterations INTEGER DEFAULT 0,
    max_iterations INTEGER DEFAULT 5,
    final_score REAL,
    created_at TEXT NOT NULL,
    completed_at TEXT,
    trace_path TEXT,
    summary TEXT
);

CREATE TABLE IF NOT EXISTS evolution_iterations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL REFERENCES evolution_runs(run_id),
    iteration_num INTEGER NOT NULL,
    status TEXT NOT NULL,
    score REAL,
    improvement_action TEXT,
    improvement_target TEXT,
    trace_json TEXT,
    eval_json TEXT,
    analysis_json TEXT,
    proposal_json TEXT,
    error_message TEXT,
    started_at TEXT NOT NULL,
    completed_at TEXT,
    token_usage_json TEXT
);

CREATE TABLE IF NOT EXISTS regression_baseline (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    task_name TEXT NOT NULL UNIQUE,
    task_domain TEXT,
    baseline_score REAL NOT NULL,
    trace_path TEXT,
    last_verified_at TEXT NOT NULL,
    harness_variant TEXT DEFAULT 'default'
);

CREATE INDEX IF NOT EXISTS idx_evolution_runs_task ON evolution_runs(task_name);
CREATE INDEX IF NOT EXISTS idx_evolution_runs_status ON evolution_runs(status);
CREATE INDEX IF NOT EXISTS idx_evolution_iterations_run ON evolution_iterations(run_id);
CREATE INDEX IF NOT EXISTS idx_regression_baseline_task ON regression_baseline(task_name);
"""


# ---------------------------------------------------------------------------
# Store
# ---------------------------------------------------------------------------


class EvolutionStore:
    """Thread-safe persistent storage for evolution data."""

    def __init__(self, db_path: Optional[Path] = None):
        if db_path is None:
            db_path = get_hermes_home() / "evolution" / "store.db"
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn: Optional[sqlite3.Connection] = None

    # -- Connection management -------------------------------------------------

    @property
    def conn(self) -> sqlite3.Connection:
        if self._conn is None:
            self._conn = self._open()
        return self._conn

    def _open(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        conn.executescript(_SCHEMA_SQL)
        conn.commit()
        return conn

    def close(self) -> None:
        with self._lock:
            if self._conn is not None:
                try:
                    self._conn.close()
                except Exception:
                    pass
                self._conn = None

    # -- Run management --------------------------------------------------------

    def create_run(
        self,
        task_name: str,
        task_domain: str = "general",
        task_complexity: int = 1,
        max_iterations: int = 5,
        session_id: Optional[str] = None,
        harness_variant: str = "default",
    ) -> str:
        """Create a new evolution run. Returns run_id."""
        run_id = f"evo_{uuid.uuid4().hex[:16]}"
        now = datetime.now(timezone.utc).isoformat()
        with self._lock:
            self.conn.execute(
                """INSERT INTO evolution_runs
                   (run_id, task_name, task_domain, task_complexity, status,
                    session_id, max_iterations, created_at)
                   VALUES (?, ?, ?, ?, 'pending', ?, ?, ?)""",
                (run_id, task_name, task_domain, task_complexity, session_id, max_iterations, now),
            )
            self.conn.commit()
        logger.info("Created evolution run %s for task '%s'", run_id, task_name)
        return run_id

    def update_run_status(self, run_id: str, status: str, **kwargs: Any) -> None:
        """Update the status and optional fields of a run."""
        sets = ["status = ?"]
        params: List[Any] = [status]
        for key, value in kwargs.items():
            if key in ("final_score", "iterations", "completed_at", "trace_path", "summary"):
                sets.append(f"{key} = ?")
                params.append(value)
        params.append(run_id)
        with self._lock:
            self.conn.execute(
                f"UPDATE evolution_runs SET {', '.join(sets)} WHERE run_id = ?",
                params,
            )
            self.conn.commit()

    def get_run(self, run_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve a run by ID."""
        row = self.conn.execute(
            "SELECT * FROM evolution_runs WHERE run_id = ?", (run_id,)
        ).fetchone()
        if row is None:
            return None
        return dict(row)

    def list_runs(
        self,
        task_name: Optional[str] = None,
        status: Optional[str] = None,
        limit: int = 50,
    ) -> List[Dict[str, Any]]:
        """List runs, optionally filtered."""
        where = []
        params: List[Any] = []
        if task_name:
            where.append("task_name = ?")
            params.append(task_name)
        if status:
            where.append("status = ?")
            params.append(status)
        sql = "SELECT * FROM evolution_runs"
        if where:
            sql += " WHERE " + " AND ".join(where)
        sql += " ORDER BY created_at DESC LIMIT ?"
        params.append(limit)
        return [dict(row) for row in self.conn.execute(sql, params).fetchall()]

    # -- Iteration management --------------------------------------------------

    def add_iteration(
        self,
        run_id: str,
        iteration_num: int,
        status: str = "attempting",
        score: Optional[float] = None,
        improvement_action: Optional[str] = None,
        improvement_target: Optional[str] = None,
        trace_json: Optional[str] = None,
        eval_json: Optional[str] = None,
        analysis_json: Optional[str] = None,
        proposal_json: Optional[str] = None,
        error_message: Optional[str] = None,
        token_usage_json: Optional[str] = None,
    ) -> int:
        """Record an evolution iteration. Returns the row ID."""
        now = datetime.now(timezone.utc).isoformat()
        with self._lock:
            cursor = self.conn.execute(
                """INSERT INTO evolution_iterations
                   (run_id, iteration_num, status, score, improvement_action,
                    improvement_target, trace_json, eval_json, analysis_json,
                    proposal_json, error_message, started_at, completed_at,
                    token_usage_json)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    run_id, iteration_num, status, score, improvement_action,
                    improvement_target, trace_json, eval_json, analysis_json,
                    proposal_json, error_message, now, now, token_usage_json,
                ),
            )
            # Update parent run's iteration count
            self.conn.execute(
                "UPDATE evolution_runs SET iterations = MAX(iterations, ?) WHERE run_id = ?",
                (iteration_num, run_id),
            )
            self.conn.commit()
            return cursor.lastrowid

    def get_iterations(self, run_id: str) -> List[Dict[str, Any]]:
        """Get all iterations for a run, ordered by iteration number."""
        rows = self.conn.execute(
            "SELECT * FROM evolution_iterations WHERE run_id = ? ORDER BY iteration_num",
            (run_id,),
        ).fetchall()
        return [dict(row) for row in rows]

    def get_last_iteration(self, run_id: str) -> Optional[Dict[str, Any]]:
        """Get the most recent iteration for a run."""
        row = self.conn.execute(
            "SELECT * FROM evolution_iterations WHERE run_id = ? ORDER BY iteration_num DESC LIMIT 1",
            (run_id,),
        ).fetchone()
        return dict(row) if row else None

    # -- Regression baseline ---------------------------------------------------

    def set_baseline(
        self,
        task_name: str,
        score: float,
        task_domain: str = "general",
        trace_path: Optional[str] = None,
        harness_variant: str = "default",
    ) -> None:
        """Record a successful task execution as a regression baseline."""
        now = datetime.now(timezone.utc).isoformat()
        with self._lock:
            self.conn.execute(
                """INSERT OR REPLACE INTO regression_baseline
                   (task_name, task_domain, baseline_score, trace_path,
                    last_verified_at, harness_variant)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (task_name, task_domain, score, trace_path, now, harness_variant),
            )
            self.conn.commit()

    def get_baseline(self, task_name: str, harness_variant: str = "default") -> Optional[Dict[str, Any]]:
        """Get the regression baseline for a task."""
        row = self.conn.execute(
            "SELECT * FROM regression_baseline WHERE task_name = ? AND harness_variant = ?",
            (task_name, harness_variant),
        ).fetchone()
        return dict(row) if row else None

    def get_all_baselines(self, harness_variant: str = "default") -> List[Dict[str, Any]]:
        """Get all regression baselines for a harness variant."""
        return [
            dict(row)
            for row in self.conn.execute(
                "SELECT * FROM regression_baseline WHERE harness_variant = ? ORDER BY task_name",
                (harness_variant,),
            ).fetchall()
        ]

    def delete_baseline(self, task_name: str, harness_variant: str = "default") -> bool:
        """Remove a regression baseline."""
        with self._lock:
            cursor = self.conn.execute(
                "DELETE FROM regression_baseline WHERE task_name = ? AND harness_variant = ?",
                (task_name, harness_variant),
            )
            self.conn.commit()
            return cursor.rowcount > 0

    # -- Maintenance -----------------------------------------------------------

    def vacuum(self) -> None:
        """Reclaim storage space."""
        with self._lock:
            self.conn.execute("PRAGMA optimize")
            self.conn.commit()

    def __del__(self) -> None:
        self.close()


# ---------------------------------------------------------------------------
# Singleton access (lazy init, same pattern as HermesState)
# ---------------------------------------------------------------------------

_store: Optional[EvolutionStore] = None
_store_lock = threading.Lock()


def get_evolution_store() -> EvolutionStore:
    """Return the process-wide EvolutionStore singleton."""
    global _store
    if _store is None:
        with _store_lock:
            if _store is None:
                _store = EvolutionStore()
    return _store
