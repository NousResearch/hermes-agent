"""
Self Evolution Plugin — Independent SQLite Database
=====================================================
Independent from state.db to avoid upstream schema conflicts.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

from self_evolution.paths import DATA_DIR as DB_DIR, DB_PATH

SCHEMA_VERSION = 1

VALID_TABLES = frozenset({
    "tool_invocations", "session_scores", "outcome_signals",
    "reflection_reports", "evolution_proposals", "improvement_units",
    "strategy_versions", "_meta",
})


def _validate_table(table: str) -> None:
    """Reject table names not in the known schema."""
    if table not in VALID_TABLES:
        raise ValueError(f"Invalid table name: {table!r}")


SCHEMA = """
-- Tool invocation telemetry
CREATE TABLE IF NOT EXISTS tool_invocations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    tool_name TEXT NOT NULL,
    duration_ms INTEGER,
    success BOOLEAN NOT NULL,
    error_type TEXT,
    turn_number INTEGER,
    created_at REAL NOT NULL DEFAULT (strftime('%s','now'))
);

-- Session quality scores
CREATE TABLE IF NOT EXISTS session_scores (
    session_id TEXT PRIMARY KEY,
    composite_score REAL,
    completion_rate REAL,
    efficiency_score REAL,
    cost_efficiency REAL,
    satisfaction_proxy REAL,
    task_category TEXT,
    model TEXT,
    created_at REAL NOT NULL DEFAULT (strftime('%s','now'))
);

-- Outcome signals
CREATE TABLE IF NOT EXISTS outcome_signals (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    signal_type TEXT NOT NULL,
    signal_value REAL,
    metadata TEXT,
    created_at REAL NOT NULL DEFAULT (strftime('%s','now'))
);

-- Reflection reports
CREATE TABLE IF NOT EXISTS reflection_reports (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    period_start REAL,
    period_end REAL,
    sessions_analyzed INTEGER,
    avg_score REAL,
    error_summary TEXT DEFAULT '',
    waste_summary TEXT DEFAULT '',
    code_change_summary TEXT DEFAULT '',
    worst_patterns TEXT DEFAULT '[]',
    best_patterns TEXT DEFAULT '[]',
    tool_insights TEXT DEFAULT '{}',
    recommendations TEXT DEFAULT '[]',
    model_used TEXT DEFAULT '',
    created_at REAL NOT NULL DEFAULT (strftime('%s','now'))
);

-- Evolution proposals
CREATE TABLE IF NOT EXISTS evolution_proposals (
    id TEXT PRIMARY KEY,
    report_id INTEGER REFERENCES reflection_reports(id),
    proposal_type TEXT NOT NULL,
    title TEXT NOT NULL,
    description TEXT NOT NULL,
    expected_impact TEXT DEFAULT '',
    risk_assessment TEXT DEFAULT 'low',
    rollback_plan TEXT DEFAULT '',
    status TEXT NOT NULL DEFAULT 'pending_approval',
    user_feedback TEXT DEFAULT '',
    created_at REAL NOT NULL DEFAULT (strftime('%s','now')),
    resolved_at REAL
);

-- Improvement unit tracking (A/B testing)
CREATE TABLE IF NOT EXISTS improvement_units (
    id TEXT PRIMARY KEY,
    proposal_id TEXT REFERENCES evolution_proposals(id),
    change_type TEXT NOT NULL,
    version INTEGER DEFAULT 0,
    baseline_score REAL DEFAULT 0.0,
    current_score REAL DEFAULT 0.0,
    sessions_sampled INTEGER DEFAULT 0,
    min_sessions INTEGER DEFAULT 10,
    min_improvement REAL DEFAULT 0.05,
    max_regression REAL DEFAULT 0.10,
    status TEXT NOT NULL DEFAULT 'active',
    created_at REAL NOT NULL DEFAULT (strftime('%s','now')),
    resolved_at REAL
);

-- Strategy version history
CREATE TABLE IF NOT EXISTS strategy_versions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    version INTEGER NOT NULL,
    strategies_json TEXT NOT NULL,
    avg_score REAL,
    active_from REAL NOT NULL,
    active_until REAL
);

-- Schema version tracking
CREATE TABLE IF NOT EXISTS _meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

-- Indexes
CREATE INDEX IF NOT EXISTS idx_tool_invocations_session ON tool_invocations(session_id);
CREATE INDEX IF NOT EXISTS idx_tool_invocations_created ON tool_invocations(created_at);
CREATE INDEX IF NOT EXISTS idx_session_scores_created ON session_scores(created_at);
CREATE INDEX IF NOT EXISTS idx_outcome_signals_session ON outcome_signals(session_id);
CREATE INDEX IF NOT EXISTS idx_evolution_proposals_status ON evolution_proposals(status);
CREATE INDEX IF NOT EXISTS idx_improvement_units_status ON improvement_units(status);
"""


def _ensure_dir():
    DB_DIR.mkdir(parents=True, exist_ok=True)


_local = threading.local()


def get_connection() -> sqlite3.Connection:
    """Return a thread-local cached connection (reused across calls)."""
    conn = getattr(_local, "conn", None)
    if conn is not None:
        try:
            conn.execute("SELECT 1")
            return conn
        except sqlite3.Error:
            try:
                conn.close()
            except Exception:
                pass
    _ensure_dir()
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    _local.conn = conn
    return conn


def close_connection():
    """Close the thread-local connection (for test cleanup / teardown)."""
    conn = getattr(_local, "conn", None)
    if conn is not None:
        try:
            conn.close()
        except Exception:
            pass
        _local.conn = None


def init_db():
    """Initialize database with schema."""
    conn = get_connection()
    conn.executescript(SCHEMA)
    conn.execute(
        "INSERT OR REPLACE INTO _meta (key, value) VALUES (?, ?)",
        ("schema_version", str(SCHEMA_VERSION)),
    )
    conn.commit()
    logger.info("self_evolution database initialized at %s", DB_PATH)

    # Schema migration: add code_change_summary column if missing
    try:
        conn.execute("ALTER TABLE reflection_reports ADD COLUMN code_change_summary TEXT DEFAULT ''")
        logger.info("Added code_change_summary column to reflection_reports")
    except sqlite3.OperationalError:
        pass  # Column already exists

    # Close after init so subsequent calls get a fresh connection with the new schema
    close_connection()


# ── Generic CRUD ─────────────────────────────────────────────────────────

def insert(table: str, data: dict) -> int:
    """Insert a row into a table. Returns the rowid."""
    _validate_table(table)
    conn = get_connection()
    cols = ", ".join(data.keys())
    placeholders = ", ".join("?" for _ in data)
    sql = f"INSERT INTO {table} ({cols}) VALUES ({placeholders})"
    cur = conn.execute(sql, list(data.values()))
    conn.commit()
    return cur.lastrowid


def insert_many(table: str, rows: List[dict]):
    """Insert multiple rows."""
    _validate_table(table)
    if not rows:
        return
    conn = get_connection()
    cols = list(rows[0].keys())
    placeholders = ", ".join("?" for _ in cols)
    sql = f"INSERT INTO {table} ({', '.join(cols)}) VALUES ({placeholders})"
    conn.executemany(sql, [[row.get(c) for c in cols] for row in rows])
    conn.commit()


def update(table: str, data: dict, where: str, where_params: tuple = ()):
    """Update rows matching where clause."""
    _validate_table(table)
    conn = get_connection()
    set_clause = ", ".join(f"{k} = ?" for k in data.keys())
    sql = f"UPDATE {table} SET {set_clause} WHERE {where}"
    conn.execute(sql, list(data.values()) + list(where_params))
    conn.commit()


def fetch_one(table: str, where: str = "", params: tuple = ()) -> Optional[Dict[str, Any]]:
    """Fetch a single row as dict."""
    _validate_table(table)
    conn = get_connection()
    sql = f"SELECT * FROM {table}"
    if where:
        sql += f" WHERE {where}"
    sql += " LIMIT 1"
    row = conn.execute(sql, params).fetchone()
    return dict(row) if row else None


def fetch_all(table: str, where: str = "", params: tuple = (),
              order_by: str = "", limit: int = 0) -> List[Dict[str, Any]]:
    """Fetch all matching rows as list of dicts."""
    _validate_table(table)
    conn = get_connection()
    sql = f"SELECT * FROM {table}"
    if where:
        sql += f" WHERE {where}"
    if order_by:
        sql += f" ORDER BY {order_by}"
    if limit:
        sql += f" LIMIT {int(limit)}"
    rows = conn.execute(sql, params).fetchall()
    return [dict(r) for r in rows]


def query(sql: str, params: tuple = ()) -> List[Dict[str, Any]]:
    """Run a raw query."""
    conn = get_connection()
    rows = conn.execute(sql, params).fetchall()
    return [dict(r) for r in rows]


def execute(sql: str, params: tuple = ()):
    """Run a raw execute."""
    conn = get_connection()
    conn.execute(sql, params)
    conn.commit()


def cleanup(days: int = 30):
    """Remove data older than N days."""
    cutoff = time.time() - (days * 86400)
    conn = get_connection()
    for table in ["tool_invocations", "outcome_signals"]:
        conn.execute(f"DELETE FROM {table} WHERE created_at < ?", (cutoff,))
    conn.commit()
    logger.info("Cleaned up data older than %d days", days)
