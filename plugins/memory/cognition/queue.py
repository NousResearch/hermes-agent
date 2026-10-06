"""Durable session-end push queue for the cognition memory provider.

Session-end extraction must never lose facts to a crashed or offline push, so
every push is first persisted to a local SQLite FIFO queue. A worker drains due
entries; failures retry on a 1m / 5m / 30m schedule (plus jitter) for up to 24h
before the entry is dead-lettered with a tombstone. The queue is bounded
(default 1000 pending entries): overflow drops the oldest pending entry and
records a tombstone for it, so loss is always visible.

Backlog surfacing: :meth:`DurablePushQueue.backlog` reports pending count,
oldest entry age, and next scheduled attempt for the status tool and prompt.
"""

from __future__ import annotations

import json
import os
import random
import sqlite3
import threading
import time
from typing import Any, Dict, List, Optional

# Retry delays (seconds) by consecutive-failure count: 1m, 5m, then 30m.
# Attempts past the schedule keep the 30m cadence until the 24h cap.
_RETRY_DELAYS = (60.0, 300.0, 1800.0)
_RETRY_JITTER_FRACTION = 0.1
# Maximum entry age before it is dead-lettered instead of retried.
_MAX_ENTRY_AGE_SECS = 24 * 3600
# Default bound on pending entries; overflow drops oldest with a tombstone.
DEFAULT_MAX_ENTRIES = 1000

_SCHEMA = """
CREATE TABLE IF NOT EXISTS push_queue (
    seq INTEGER PRIMARY KEY AUTOINCREMENT,
    ref TEXT UNIQUE NOT NULL,
    session_id TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'pending',
    attempts INTEGER NOT NULL DEFAULT 0,
    created_at REAL NOT NULL,
    next_due_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS tombstones (
    ref TEXT PRIMARY KEY,
    reason TEXT NOT NULL,
    dropped_at REAL NOT NULL,
    payload_json TEXT
);
"""


def entry_ref(session_id: str, seq: int) -> str:
    """Deterministic queue ref: ``cog-{sessionId}-{seq}``."""
    return "cog-%s-%d" % (session_id, seq)


def retry_delay(attempts: int, rng: Optional[random.Random] = None) -> float:
    """Delay before the next attempt: 1m / 5m / 30m schedule plus jitter."""
    base = _RETRY_DELAYS[min(attempts, len(_RETRY_DELAYS) - 1)]
    jitter = (rng or random).uniform(
        -_RETRY_JITTER_FRACTION * base, _RETRY_JITTER_FRACTION * base
    )
    return max(1.0, base + jitter)


class DurablePushQueue:
    """SQLite-backed FIFO push queue with retry, bound, and tombstones."""

    def __init__(self, path: str, max_entries: int = DEFAULT_MAX_ENTRIES):
        directory = os.path.dirname(path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        self._path = path
        self._max_entries = max(1, int(max_entries))
        self._lock = threading.Lock()
        self._db = sqlite3.connect(path, check_same_thread=False)
        with self._db:
            self._db.executescript(_SCHEMA)

    def close(self) -> None:
        """Close the backing database."""
        with self._lock:
            self._db.close()

    # -- writes ---------------------------------------------------------

    def enqueue(
        self, payload: Dict[str, Any], *, session_id: str, now: Optional[float] = None
    ) -> str:
        """Persist a push payload; returns its deterministic ref.

        When the pending bound is reached, the oldest pending entry is dropped
        and a tombstone is recorded for it before the new entry lands.
        """
        moment = now if now is not None else time.time()
        payload_text = json.dumps(payload, ensure_ascii=False, sort_keys=True)
        with self._lock:
            with self._db:
                pending = self._db.execute(
                    "SELECT COUNT(*) FROM push_queue WHERE status = 'pending'"
                ).fetchone()[0]
                if pending >= self._max_entries:
                    oldest = self._db.execute(
                        "SELECT ref, payload_json FROM push_queue WHERE status = 'pending'"
                        " ORDER BY seq ASC LIMIT 1"
                    ).fetchone()
                    if oldest is not None:
                        self._db.execute(
                            "DELETE FROM push_queue WHERE ref = ?", (oldest[0],)
                        )
                        self._db.execute(
                            "INSERT OR REPLACE INTO tombstones (ref, reason, dropped_at, payload_json)"
                            " VALUES (?, ?, ?, ?)",
                            (oldest[0], "evicted-drop-oldest", moment, oldest[1]),
                        )
                cursor = self._db.execute(
                    "INSERT INTO push_queue (ref, session_id, payload_json, status, attempts,"
                    " created_at, next_due_at) VALUES (?, ?, ?, 'pending', 0, ?, ?)",
                    ("tmp", session_id, payload_text, moment, moment),
                )
                seq = cursor.lastrowid
                assert seq is not None  # AUTOINCREMENT always yields a rowid
                ref = entry_ref(session_id, seq)
                self._db.execute(
                    "UPDATE push_queue SET ref = ? WHERE seq = ?", (ref, seq)
                )
        return ref

    def ack(self, ref: str) -> None:
        """Remove a successfully pushed entry."""
        with self._lock:
            with self._db:
                self._db.execute("DELETE FROM push_queue WHERE ref = ?", (ref,))

    def fail(
        self,
        ref: str,
        *,
        now: Optional[float] = None,
        rng: Optional[random.Random] = None,
    ) -> str:
        """Record a failed attempt: reschedule, or dead-letter past 24h.

        Returns ``"retry"`` or ``"dead-letter"``.
        """
        moment = now if now is not None else time.time()
        with self._lock:
            with self._db:
                row = self._db.execute(
                    "SELECT attempts, created_at, payload_json FROM push_queue WHERE ref = ?",
                    (ref,),
                ).fetchone()
                if row is None:
                    return "retry"
                attempts, created_at, payload_text = row
                attempts += 1
                if moment - created_at >= _MAX_ENTRY_AGE_SECS:
                    self._db.execute("DELETE FROM push_queue WHERE ref = ?", (ref,))
                    self._db.execute(
                        "INSERT OR REPLACE INTO tombstones (ref, reason, dropped_at, payload_json)"
                        " VALUES (?, ?, ?, ?)",
                        (ref, "expired-max-24h", moment, payload_text),
                    )
                    return "dead-letter"
                self._db.execute(
                    "UPDATE push_queue SET attempts = ?, next_due_at = ? WHERE ref = ?",
                    (attempts, moment + retry_delay(attempts - 1, rng), ref),
                )
                return "retry"

    # -- reads ----------------------------------------------------------

    def due(self, now: Optional[float] = None) -> List[Dict[str, Any]]:
        """Due pending entries, oldest first (FIFO)."""
        moment = now if now is not None else time.time()
        with self._lock:
            rows = self._db.execute(
                "SELECT ref, session_id, payload_json, attempts, created_at, next_due_at"
                " FROM push_queue WHERE status = 'pending' AND next_due_at <= ?"
                " ORDER BY seq ASC",
                (moment,),
            ).fetchall()
        return [
            {
                "ref": ref,
                "session_id": session_id,
                "payload": json.loads(payload_text),
                "attempts": attempts,
                "created_at": created_at,
                "next_due_at": next_due_at,
            }
            for ref, session_id, payload_text, attempts, created_at, next_due_at in rows
        ]

    def pending_count(self) -> int:
        """Number of unpushed (pending) entries."""
        with self._lock:
            return self._db.execute(
                "SELECT COUNT(*) FROM push_queue WHERE status = 'pending'"
            ).fetchone()[0]

    def tombstones(self) -> List[Dict[str, Any]]:
        """Every recorded tombstone (evictions and expiries), oldest first."""
        with self._lock:
            rows = self._db.execute(
                "SELECT ref, reason, dropped_at FROM tombstones ORDER BY dropped_at ASC"
            ).fetchall()
        return [
            {"ref": ref, "reason": reason, "dropped_at": dropped_at}
            for ref, reason, dropped_at in rows
        ]

    def backlog(self, now: Optional[float] = None) -> Dict[str, Any]:
        """Backlog summary for surfacing: pending, oldest age, next attempt."""
        moment = now if now is not None else time.time()
        with self._lock:
            row = self._db.execute(
                "SELECT COUNT(*), MIN(created_at), MIN(next_due_at) FROM push_queue"
                " WHERE status = 'pending'"
            ).fetchone()
        pending, oldest, next_due = row[0], row[1], row[2]
        return {
            "pending": pending,
            "oldest_age_secs": (moment - oldest) if oldest is not None else 0.0,
            "next_due_in_secs": max(0.0, next_due - moment) if next_due is not None else 0.0,
            "tombstones": len(self.tombstones()),
        }
