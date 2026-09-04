"""Routing telemetry — SQLite history of every pipeline decision.

Records mode, stage, turn type, selected/suggested model, candidate scores,
and routing latency for observability (``hermes router history/stats``).
Privacy: prompts are never stored — only a djb2 hash and a bounded length.
Bounded to the newest ~10k rows. Failure-safe: telemetry never breaks routing.
"""
from __future__ import annotations

import json
import os
import sqlite3
import threading
import time
from pathlib import Path

from .types import RoutingDecision, RoutingRequest

MAX_ROWS = 10_000

_SCHEMA = """
CREATE TABLE IF NOT EXISTS routing_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts REAL NOT NULL,
    session_id TEXT NOT NULL DEFAULT '',
    mode TEXT NOT NULL,
    stage TEXT NOT NULL,
    turn_type TEXT NOT NULL DEFAULT 'unknown',
    selected_model TEXT NOT NULL,
    suggestion TEXT NOT NULL DEFAULT '',
    reason_code TEXT NOT NULL DEFAULT '',
    pinned INTEGER NOT NULL DEFAULT 0,
    candidates_json TEXT NOT NULL DEFAULT '[]',
    rejected_json TEXT NOT NULL DEFAULT '[]',
    latency_ms REAL NOT NULL DEFAULT 0,
    prompt_hash TEXT NOT NULL DEFAULT '',
    prompt_chars INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_routing_history_ts ON routing_history(ts DESC);
CREATE INDEX IF NOT EXISTS idx_routing_history_session ON routing_history(session_id, ts DESC);
"""


def prompt_hash(text: str) -> str:
    """Deterministic djb2 hash — identifies repeat prompts without storing them."""
    h = 5381
    for ch in (text or "")[:256]:
        h = ((h << 5) + h + ord(ch)) & 0xFFFFFFFF
    return f"{h:x}"


class RouterTelemetry:
    """SQLite-backed routing decision log. Thread-safe, failure-safe."""

    def __init__(self, db_path):
        self._db_path = str(db_path)
        self._lock = threading.Lock()
        self._available = True
        try:
            Path(self._db_path).parent.mkdir(parents=True, exist_ok=True)
            with self._connect() as conn:
                conn.executescript(_SCHEMA)
            os.chmod(self._db_path, 0o600)
        except Exception:
            self._available = False

    @property
    def available(self) -> bool:
        return self._available

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._db_path, timeout=5)
        conn.execute("PRAGMA journal_mode=WAL")
        return conn

    def record(self, request: RoutingRequest, decision: RoutingDecision, *, mode: str, session_id: str = "") -> None:
        if not self._available:
            return
        try:
            candidates = [
                {"model": c.model_id, "score": round(c.score, 4), "rejected": c.rejected_reason}
                for c in decision.candidates
            ]
            with self._lock, self._connect() as conn:
                conn.execute(
                    "INSERT INTO routing_history (ts, session_id, mode, stage, turn_type,"
                    " selected_model, suggestion, reason_code, pinned, candidates_json,"
                    " rejected_json, latency_ms, prompt_hash, prompt_chars)"
                    " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        time.time(),
                        session_id or request.session_id or "",
                        mode,
                        decision.stage,
                        decision.turn_type,
                        decision.selected_model,
                        decision.suggestion,
                        decision.reason_code,
                        1 if decision.pinned else 0,
                        json.dumps(candidates),
                        json.dumps(list(decision.rejected)),
                        decision.routing_latency_ms,
                        prompt_hash(request.prompt_text),
                        len(request.prompt_text or ""),
                    ),
                )
                conn.execute(
                    "DELETE FROM routing_history WHERE id NOT IN"
                    " (SELECT id FROM routing_history ORDER BY id DESC LIMIT ?)",
                    (MAX_ROWS,),
                )
        except Exception:
            pass

    def history(self, *, limit: int = 20, session_id: str = "") -> list:
        if not self._available:
            return []
        try:
            with self._lock, self._connect() as conn:
                if session_id:
                    rows = conn.execute(
                        "SELECT ts, session_id, mode, stage, turn_type, selected_model,"
                        " suggestion, reason_code, pinned, latency_ms FROM routing_history"
                        " WHERE session_id = ? ORDER BY id DESC LIMIT ?",
                        (session_id, limit),
                    ).fetchall()
                else:
                    rows = conn.execute(
                        "SELECT ts, session_id, mode, stage, turn_type, selected_model,"
                        " suggestion, reason_code, pinned, latency_ms FROM routing_history"
                        " ORDER BY id DESC LIMIT ?",
                        (limit,),
                    ).fetchall()
        except Exception:
            return []
        return [
            {
                "ts": r[0], "session_id": r[1], "mode": r[2], "stage": r[3],
                "turn_type": r[4], "selected_model": r[5], "suggestion": r[6],
                "reason_code": r[7], "pinned": bool(r[8]), "latency_ms": r[9],
            }
            for r in rows
        ]

    def stats(self) -> dict:
        if not self._available:
            return {}
        try:
            with self._lock, self._connect() as conn:
                total = conn.execute("SELECT COUNT(*) FROM routing_history").fetchone()[0]
                by_stage = dict(
                    conn.execute(
                        "SELECT stage, COUNT(*) FROM routing_history GROUP BY stage ORDER BY 2 DESC"
                    ).fetchall()
                )
                by_model = dict(
                    conn.execute(
                        "SELECT selected_model, COUNT(*) FROM routing_history"
                        " GROUP BY selected_model ORDER BY 2 DESC"
                    ).fetchall()
                )
                avg_latency = conn.execute(
                    "SELECT AVG(latency_ms) FROM routing_history"
                ).fetchone()[0]
        except Exception:
            return {}
        return {
            "total": total,
            "by_stage": by_stage,
            "by_model": by_model,
            "avg_latency_ms": round(avg_latency or 0.0, 3),
        }
