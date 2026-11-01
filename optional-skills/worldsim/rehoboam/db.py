"""
REHOBOAM Database Layer
SQLite setup, migrations, and query helpers.
"""

import sqlite3
import os
from pathlib import Path
from datetime import datetime

DB_DIR = Path.home() / ".hermes" / "rehoboam" / "db"
MAIN_DB = DB_DIR / "rehoboam.db"

SCHEMA_VERSION = 1

SCHEMA_SQL = """
-- Core tables
CREATE TABLE IF NOT EXISTS profiles (
    handle TEXT PRIMARY KEY,
    platform TEXT NOT NULL,
    display_name TEXT,
    last_updated TEXT NOT NULL,
    staleness TEXT NOT NULL,
    profile_path TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS simulations (
    sim_id TEXT PRIMARY KEY,
    created_at TEXT NOT NULL,
    scenario TEXT NOT NULL,
    participant_count INTEGER,
    duration_sec REAL,
    model_used TEXT,
    config_path TEXT,
    output_path TEXT
);

CREATE TABLE IF NOT EXISTS sim_participants (
    sim_id TEXT REFERENCES simulations(sim_id),
    handle TEXT REFERENCES profiles(handle),
    role TEXT,
    PRIMARY KEY (sim_id, handle)
);

CREATE TABLE IF NOT EXISTS sim_dynamics (
    sim_id TEXT REFERENCES simulations(sim_id),
    handle TEXT,
    post_count INTEGER,
    word_count INTEGER,
    avg_sentiment REAL,
    dominance_score REAL,
    agreement_score REAL,
    controversy_score REAL,
    ratio_score REAL,
    influence_in_sim REAL,
    PRIMARY KEY (sim_id, handle)
);

CREATE TABLE IF NOT EXISTS sim_interactions (
    sim_id TEXT REFERENCES simulations(sim_id),
    from_handle TEXT,
    to_handle TEXT,
    interaction_type TEXT,
    count INTEGER,
    avg_sentiment REAL,
    PRIMARY KEY (sim_id, from_handle, to_handle, interaction_type)
);

CREATE TABLE IF NOT EXISTS predictions (
    pred_id TEXT PRIMARY KEY,
    created_at TEXT NOT NULL,
    sim_id TEXT,
    handle TEXT,
    prediction_type TEXT,
    prediction_text TEXT NOT NULL,
    confidence REAL NOT NULL,
    calibrated_confidence REAL,
    timeframe_days INTEGER,
    resolved_at TEXT,
    outcome TEXT,
    outcome_evidence TEXT,
    accuracy_score REAL
);

CREATE TABLE IF NOT EXISTS social_edges (
    from_handle TEXT,
    to_handle TEXT,
    relationship_type TEXT,
    weight REAL,
    first_observed TEXT,
    last_observed TEXT,
    observation_count INTEGER,
    source TEXT,
    PRIMARY KEY (from_handle, to_handle, relationship_type)
);

CREATE TABLE IF NOT EXISTS social_clusters (
    cluster_id TEXT PRIMARY KEY,
    name TEXT,
    description TEXT,
    member_handles TEXT,
    computed_at TEXT,
    cohesion_score REAL
);

CREATE TABLE IF NOT EXISTS monitoring_events (
    event_id TEXT PRIMARY KEY,
    handle TEXT,
    detected_at TEXT NOT NULL,
    event_type TEXT,
    description TEXT,
    related_prediction_id TEXT,
    severity TEXT,
    acknowledged INTEGER DEFAULT 0
);

CREATE TABLE IF NOT EXISTS audit_log (
    log_id TEXT PRIMARY KEY,
    timestamp TEXT NOT NULL,
    sim_id TEXT,
    action TEXT NOT NULL,
    handle TEXT,
    details TEXT,
    duration_sec REAL,
    model_used TEXT,
    token_count INTEGER,
    error TEXT
);

-- Indexes
CREATE INDEX IF NOT EXISTS idx_predictions_handle ON predictions(handle);
CREATE INDEX IF NOT EXISTS idx_predictions_type ON predictions(prediction_type);
CREATE INDEX IF NOT EXISTS idx_predictions_unresolved ON predictions(outcome) WHERE outcome IS NULL;
CREATE INDEX IF NOT EXISTS idx_audit_action ON audit_log(action);
CREATE INDEX IF NOT EXISTS idx_audit_sim ON audit_log(sim_id);
CREATE INDEX IF NOT EXISTS idx_social_edges_from ON social_edges(from_handle);
CREATE INDEX IF NOT EXISTS idx_social_edges_to ON social_edges(to_handle);
CREATE INDEX IF NOT EXISTS idx_monitoring_handle ON monitoring_events(handle);
CREATE INDEX IF NOT EXISTS idx_monitoring_unack ON monitoring_events(acknowledged) WHERE acknowledged = 0;

-- Schema version tracking
CREATE TABLE IF NOT EXISTS schema_meta (
    key TEXT PRIMARY KEY,
    value TEXT
);
"""


def init_db() -> sqlite3.Connection:
    """Initialize the database, creating tables if needed."""
    DB_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(MAIN_DB))
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.executescript(SCHEMA_SQL)
    conn.execute(
        "INSERT OR REPLACE INTO schema_meta (key, value) VALUES (?, ?)",
        ("schema_version", str(SCHEMA_VERSION))
    )
    conn.commit()
    return conn


def get_db() -> sqlite3.Connection:
    """Get a database connection, initializing if needed."""
    if not MAIN_DB.exists():
        return init_db()
    conn = sqlite3.connect(str(MAIN_DB))
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.row_factory = sqlite3.Row
    return conn


def log_audit(conn: sqlite3.Connection, action: str, handle: str = None,
              sim_id: str = None, details: str = None, duration_sec: float = None,
              model_used: str = None, token_count: int = None, error: str = None):
    """Write an entry to the audit log."""
    from schemas import gen_id
    conn.execute(
        """INSERT INTO audit_log
           (log_id, timestamp, sim_id, action, handle, details, duration_sec, model_used, token_count, error)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (gen_id("log_"), datetime.utcnow().isoformat() + "Z", sim_id, action,
         handle, details, duration_sec, model_used, token_count, error)
    )
    conn.commit()


# -- Query Helpers --

def get_prediction_accuracy(conn: sqlite3.Connection, prediction_type: str = None) -> dict:
    """Get prediction accuracy statistics."""
    query = """
        SELECT prediction_type,
               COUNT(*) as total,
               SUM(CASE WHEN outcome='correct' THEN 1 ELSE 0 END) as correct,
               SUM(CASE WHEN outcome='partially_correct' THEN 1 ELSE 0 END) as partial,
               SUM(CASE WHEN outcome='incorrect' THEN 1 ELSE 0 END) as incorrect,
               AVG(confidence) as avg_confidence,
               AVG(CASE WHEN outcome='correct' THEN 1.0
                        WHEN outcome='partially_correct' THEN 0.5
                        ELSE 0.0 END) as accuracy
        FROM predictions WHERE outcome IS NOT NULL
    """
    params = []
    if prediction_type:
        query += " AND prediction_type = ?"
        params.append(prediction_type)
    query += " GROUP BY prediction_type"
    return [dict(row) for row in conn.execute(query, params).fetchall()]


def get_open_predictions(conn: sqlite3.Connection, handle: str = None) -> list:
    """Get unresolved predictions."""
    query = "SELECT * FROM predictions WHERE outcome IS NULL"
    params = []
    if handle:
        query += " AND handle = ?"
        params.append(handle)
    query += " ORDER BY created_at DESC"
    return [dict(row) for row in conn.execute(query, params).fetchall()]


def get_social_neighborhood(conn: sqlite3.Connection, handle: str, depth: int = 1) -> list:
    """Get a person's social graph neighborhood."""
    query = """
        SELECT from_handle, to_handle, relationship_type, weight
        FROM social_edges
        WHERE from_handle = ? OR to_handle = ?
        ORDER BY weight DESC
    """
    return [dict(row) for row in conn.execute(query, (handle, handle)).fetchall()]


def get_unread_alerts(conn: sqlite3.Connection) -> list:
    """Get unacknowledged monitoring alerts."""
    query = """
        SELECT * FROM monitoring_events
        WHERE acknowledged = 0
        ORDER BY detected_at DESC
    """
    return [dict(row) for row in conn.execute(query).fetchall()]


if __name__ == "__main__":
    conn = init_db()
    print(f"Database initialized at {MAIN_DB}")
    conn.close()
