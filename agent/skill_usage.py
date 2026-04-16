"""Lightweight skill usage tracker.

Records every skill load/invocation to a SQLite database.
Called from agent/skill_commands.py at the two skill loading entry points.
"""
import sqlite3
import time
from pathlib import Path

USAGE_DB = Path.home() / ".hermes" / "skill_usage.db"

def _get_conn():
    conn = sqlite3.connect(str(USAGE_DB))
    conn.execute("""
        CREATE TABLE IF NOT EXISTS skill_invocations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            skill_name TEXT NOT NULL,
            source TEXT NOT NULL DEFAULT 'slash_command',
            timestamp REAL NOT NULL
        )
    """)
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_invocations_name
        ON skill_invocations(skill_name)
    """)
    conn.commit()
    return conn

def record_skill_usage(skill_name: str, source: str = "slash_command") -> None:
    """Record a skill invocation. Source is 'slash_command' or 'preload'."""
    try:
        conn = _get_conn()
        conn.execute(
            "INSERT INTO skill_invocations (skill_name, source, timestamp) VALUES (?, ?, ?)",
            (skill_name, source, time.time()),
        )
        conn.commit()
        conn.close()
    except Exception:
        pass  # Tracking is non-critical — never break skill loading

def get_skill_usage_counts() -> dict[str, int]:
    """Return {skill_name: total_invocation_count} for all skills."""
    try:
        conn = _get_conn()
        cursor = conn.execute(
            "SELECT skill_name, COUNT(*) as cnt FROM skill_invocations GROUP BY skill_name ORDER BY cnt DESC"
        )
        counts = {row[0]: row[1] for row in cursor.fetchall()}
        conn.close()
        return counts
    except Exception:
        return {}
