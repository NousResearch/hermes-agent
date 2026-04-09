#!/usr/bin/env python3
"""Dummy database setup for Argus testing."""

import os
import sqlite3
from pathlib import Path

DUMMY_DB_PATH = Path(__file__).parent / "dummy_argus.db"
# tests/argus/simulation/ -> tests/argus/ -> tests/ -> hermes-dev/ -> argus/
SCHEMA_PATH = Path(__file__).parent.parent.parent.parent / "argus" / "watcher_schema.sql"


def init_dummy_database(db_path: Path = DUMMY_DB_PATH) -> sqlite3.Connection:
    """Initialize a fresh dummy database with Argus schema."""
    if db_path.exists():
        db_path.unlink()
    
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    
    with open(SCHEMA_PATH, "r") as f:
        schema = f.read()
    conn.executescript(schema)
    conn.commit()
    
    print(f"[DB] Initialized dummy database at {db_path}")
    return conn


def get_dummy_connection(db_path: Path = DUMMY_DB_PATH) -> sqlite3.Connection:
    """Get connection to existing dummy database."""
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    return conn


def reset_dummy_database(db_path: Path = DUMMY_DB_PATH) -> sqlite3.Connection:
    """Reset dummy database to clean state."""
    return init_dummy_database(db_path)


if __name__ == "__main__":
    conn = init_dummy_database()
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = [row[0] for row in cursor.fetchall()]
    print(f"[DB] Created tables: {', '.join(tables)}")
    conn.close()
