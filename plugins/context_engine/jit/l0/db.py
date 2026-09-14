"""L0 SQLite WAL Database layer for Hermes JIT Context Engine."""

from __future__ import annotations

import os
import sqlite3
from pathlib import Path
from typing import Optional

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS sessions (
    session_id TEXT PRIMARY KEY,
    active_scope TEXT NOT NULL,
    scope_epoch INTEGER NOT NULL DEFAULT 1,
    scope_confidence REAL NOT NULL DEFAULT 1.0,
    last_seq INTEGER NOT NULL DEFAULT 0,
    last_cwd TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS events (
    event_id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL,
    seq INTEGER NOT NULL,
    turn_id TEXT,
    origin TEXT NOT NULL,
    role TEXT NOT NULL,
    content TEXT NOT NULL,
    content_hash TEXT NOT NULL,
    root_event_id TEXT,
    authority REAL NOT NULL DEFAULT 0.0,
    effective_at TEXT NOT NULL,
    created_at TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'active',
    FOREIGN KEY(session_id) REFERENCES sessions(session_id)
);

CREATE INDEX IF NOT EXISTS idx_events_session_seq ON events(session_id, seq);
CREATE INDEX IF NOT EXISTS idx_events_content_hash ON events(session_id, content_hash);

CREATE TABLE IF NOT EXISTS overlay (
    entry_id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL,
    kind TEXT NOT NULL,
    key TEXT NOT NULL,
    value TEXT NOT NULL,
    source_event_id TEXT NOT NULL,
    seq INTEGER NOT NULL,
    authority REAL NOT NULL DEFAULT 1.0,
    status TEXT NOT NULL DEFAULT 'active',
    supersedes TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    FOREIGN KEY(session_id) REFERENCES sessions(session_id)
);

CREATE INDEX IF NOT EXISTS idx_overlay_session_status ON overlay(session_id, status);
"""


def get_default_db_path() -> Path:
    env_path = os.environ.get("HERMES_JIT_DB_PATH")
    if env_path:
        return Path(env_path)
    state_dir = Path.home() / ".hermes" / "state"
    state_dir.mkdir(parents=True, exist_ok=True)
    return state_dir / "jit_context.db"


def get_db(db_path: Optional[str | Path] = None) -> sqlite3.Connection:
    """Connect to SQLite database with WAL mode and initialize tables."""
    target_path = Path(db_path) if db_path else get_default_db_path()
    if str(target_path) != ":memory:":
        target_path.parent.mkdir(parents=True, exist_ok=True)
    
    conn = sqlite3.connect(str(target_path), timeout=5.0)
    conn.row_factory = sqlite3.Row
    if str(target_path) != ":memory:":
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
    conn.execute("PRAGMA busy_timeout=5000;")
    conn.executescript(SCHEMA_SQL)
    conn.commit()
    return conn
