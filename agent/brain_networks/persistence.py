"""SQLite persistence for Brain Networks (ECN focus + dream history).

Profile-aware via ``get_hermes_home()``. Thread-safe with a process lock.
Never stores secrets — only task labels, focus scores, and dream narratives.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from hermes_constants import get_hermes_home

_lock = threading.RLock()


def _db_path() -> Path:
    root = get_hermes_home() / "brain_networks"
    root.mkdir(parents=True, exist_ok=True)
    return root / "orchestrator.db"


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(str(_db_path()), timeout=10, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=5000")
    return conn


def _ensure_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS ecn_focus (
            session_id TEXT PRIMARY KEY,
            current_task TEXT,
            task_stack_json TEXT NOT NULL DEFAULT '[]',
            focus_level REAL NOT NULL DEFAULT 0.5,
            distraction_count INTEGER NOT NULL DEFAULT 0,
            state TEXT NOT NULL DEFAULT 'idle',
            pinned INTEGER NOT NULL DEFAULT 0,
            updated_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS dream_episodes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            narrative TEXT NOT NULL,
            insights_json TEXT NOT NULL DEFAULT '[]',
            emotional_tone TEXT NOT NULL DEFAULT 'neutral',
            source_count INTEGER NOT NULL DEFAULT 0,
            source_episodes_json TEXT NOT NULL DEFAULT '[]',
            created_at TEXT NOT NULL
        );

        CREATE INDEX IF NOT EXISTS idx_dreams_ts ON dream_episodes(timestamp DESC);
        """
    )


def save_ecn_state(
    session_id: str,
    *,
    current_task: Optional[str],
    task_stack: List[str],
    focus_level: float,
    distraction_count: int,
    state: str,
    pinned: bool = False,
) -> None:
    """Persist ECN focus for a session (upsert)."""
    sid = (session_id or "").strip() or "_default"
    now = datetime.now(timezone.utc).isoformat()
    with _lock:
        conn = _connect()
        try:
            _ensure_schema(conn)
            conn.execute(
                """
                INSERT INTO ecn_focus (
                    session_id, current_task, task_stack_json, focus_level,
                    distraction_count, state, pinned, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(session_id) DO UPDATE SET
                    current_task=excluded.current_task,
                    task_stack_json=excluded.task_stack_json,
                    focus_level=excluded.focus_level,
                    distraction_count=excluded.distraction_count,
                    state=excluded.state,
                    pinned=excluded.pinned,
                    updated_at=excluded.updated_at
                """,
                (
                    sid,
                    current_task,
                    json.dumps(list(task_stack or []), ensure_ascii=False),
                    float(focus_level),
                    int(distraction_count),
                    str(state or "idle"),
                    1 if pinned else 0,
                    now,
                ),
            )
            conn.commit()
        finally:
            conn.close()


def load_ecn_state(session_id: str) -> Optional[Dict[str, Any]]:
    """Load persisted ECN focus for a session, or None if absent."""
    sid = (session_id or "").strip() or "_default"
    with _lock:
        conn = _connect()
        try:
            _ensure_schema(conn)
            row = conn.execute(
                "SELECT * FROM ecn_focus WHERE session_id = ?", (sid,)
            ).fetchone()
            if not row:
                return None
            stack: List[str] = []
            try:
                raw = json.loads(row["task_stack_json"] or "[]")
                if isinstance(raw, list):
                    stack = [str(x) for x in raw]
            except (TypeError, json.JSONDecodeError):
                stack = []
            return {
                "session_id": row["session_id"],
                "current_task": row["current_task"],
                "task_stack": stack,
                "focus_level": float(row["focus_level"]),
                "distraction_count": int(row["distraction_count"]),
                "state": row["state"],
                "pinned": bool(row["pinned"]),
                "updated_at": row["updated_at"],
            }
        finally:
            conn.close()


def clear_ecn_state(session_id: str) -> bool:
    """Delete persisted ECN focus for a session. Returns True if a row was removed."""
    sid = (session_id or "").strip() or "_default"
    with _lock:
        conn = _connect()
        try:
            _ensure_schema(conn)
            cur = conn.execute("DELETE FROM ecn_focus WHERE session_id = ?", (sid,))
            conn.commit()
            return cur.rowcount > 0
        finally:
            conn.close()


def save_dream_episode(dream: Dict[str, Any]) -> int:
    """Persist a dream episode. Returns new row id."""
    now = datetime.now(timezone.utc).isoformat()
    timestamp = str(dream.get("timestamp") or now)
    narrative = str(dream.get("narrative") or "")
    insights = dream.get("insights") or []
    if not isinstance(insights, list):
        insights = [str(insights)]
    tone = str(dream.get("emotional_tone") or "neutral")
    source_count = int(dream.get("source_count") or 0)
    source_eps = dream.get("source_episodes") or []
    if not isinstance(source_eps, list):
        source_eps = []
    with _lock:
        conn = _connect()
        try:
            _ensure_schema(conn)
            cur = conn.execute(
                """
                INSERT INTO dream_episodes (
                    timestamp, narrative, insights_json, emotional_tone,
                    source_count, source_episodes_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    timestamp,
                    narrative,
                    json.dumps(insights, ensure_ascii=False),
                    tone,
                    source_count,
                    json.dumps(source_eps, ensure_ascii=False),
                    now,
                ),
            )
            conn.commit()
            return int(cur.lastrowid or 0)
        finally:
            conn.close()


def load_recent_dreams(limit: int = 5) -> List[Dict[str, Any]]:
    """Return recent dream episodes (newest first)."""
    lim = max(1, min(int(limit or 5), 50))
    with _lock:
        conn = _connect()
        try:
            _ensure_schema(conn)
            rows = conn.execute(
                """
                SELECT timestamp, narrative, insights_json, emotional_tone,
                       source_count
                FROM dream_episodes
                ORDER BY id DESC
                LIMIT ?
                """,
                (lim,),
            ).fetchall()
            out: List[Dict[str, Any]] = []
            for row in rows:
                insights: List[str] = []
                try:
                    raw = json.loads(row["insights_json"] or "[]")
                    if isinstance(raw, list):
                        insights = [str(x) for x in raw]
                except (TypeError, json.JSONDecodeError):
                    insights = []
                out.append(
                    {
                        "timestamp": row["timestamp"],
                        "narrative": (row["narrative"] or "")[:500],
                        "insights": insights,
                        "emotional_tone": row["emotional_tone"],
                        "source_count": int(row["source_count"] or 0),
                    }
                )
            return out
        finally:
            conn.close()


def dream_count() -> int:
    with _lock:
        conn = _connect()
        try:
            _ensure_schema(conn)
            row = conn.execute("SELECT COUNT(*) AS c FROM dream_episodes").fetchone()
            return int(row["c"] if row else 0)
        finally:
            conn.close()
