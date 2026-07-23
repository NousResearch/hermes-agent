"""SQLite-backed queue for monitor_event / dequeue_events."""

from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path
from typing import Any, Dict, List, Optional


def default_queue_path() -> Path:
    import os

    override = os.environ.get("WEB3_MCP_QUEUE_DB", "").strip()
    if override:
        return Path(override)
    home = Path.home()
    base = home / ".hermes" / "web3-mcp"
    base.mkdir(parents=True, exist_ok=True)
    return base / "event_queue.sqlite3"


class EventQueue:
    def __init__(self, db_path: Optional[Path] = None) -> None:
        self._path = Path(db_path) if db_path else default_queue_path()
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self._path), check_same_thread=False)
        self._conn.execute(
            """CREATE TABLE IF NOT EXISTS events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                chain TEXT NOT NULL,
                payload TEXT NOT NULL,
                created REAL NOT NULL,
                status TEXT NOT NULL DEFAULT 'pending'
            )"""
        )
        self._conn.commit()

    def enqueue(self, chain: str, payload: Dict[str, Any]) -> int:
        cur = self._conn.execute(
            "INSERT INTO events (chain, payload, created, status) VALUES (?,?,?,?)",
            (chain, json.dumps(payload), time.time(), "pending"),
        )
        self._conn.commit()
        return int(cur.lastrowid)

    def dequeue(self, limit: int = 20) -> List[Dict[str, Any]]:
        cur = self._conn.execute(
            "SELECT id, chain, payload, created FROM events WHERE status='pending' "
            "ORDER BY id ASC LIMIT ?",
            (limit,),
        )
        rows = cur.fetchall()
        out: List[Dict[str, Any]] = []
        for eid, chain, payload, created in rows:
            self._conn.execute("UPDATE events SET status='delivered' WHERE id=?", (eid,))
            out.append(
                {"id": eid, "chain": chain, "payload": json.loads(payload), "created": created}
            )
        self._conn.commit()
        return out

    def close(self) -> None:
        self._conn.close()
