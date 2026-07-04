"""Durable capture queue for mem0 salient auto-capture (Track A-lite).

A small, dependency-free SQLite-backed work queue that lives PLUGIN-SIDE on the gateway host
(the mem0 plugin reaches the store only over HTTP — it has no direct Postgres access — so the
queue cannot live in the mem0 Postgres for A-lite; local SQLite gives real transactions + a
lease lock via BEGIN IMMEDIATE).

Design (spec D-5/D-8/D-10/D-13):
- One row per accepted turn: {idem_key, payload(json), status, attempts, next_attempt_at,
  leased_until, model_verdict, created_at}.
- status: pending -> inflight (leased) -> done | dead. A reaper returns an expired-lease
  inflight row to pending; it increments attempts ONLY if a model_verdict=fault was recorded
  (an environmental/no-verdict crash does NOT burn the retry budget — D-10).
- Idempotency: enqueue of a duplicate idem_key is a no-op (INV-5); the WRITE path (the drain
  worker) is responsible for the exactly-once reconcile against the store by idem_key.
- Durability: WAL mode + synchronous=NORMAL; a committed enqueue survives a process crash
  (INV-1). Postgres-down at the mem0 side is irrelevant here — this queue is the local spool
  itself, so D-13's "spool fallback" is the queue's own durability (no separate spool needed
  when the queue IS local-durable).

This module is pure/unit-testable: it takes a db path, opens its own connection, and never
imports the mem0 client. The drain worker (capture_drain.py) composes it with the HTTP client.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import sqlite3
import time
from typing import Any, Dict, List, Optional

_SCHEMA = """
CREATE TABLE IF NOT EXISTS capture_queue (
    idem_key        TEXT PRIMARY KEY,
    payload         TEXT NOT NULL,
    status          TEXT NOT NULL DEFAULT 'pending',
    attempts        INTEGER NOT NULL DEFAULT 0,
    next_attempt_at REAL NOT NULL DEFAULT 0,
    leased_until    REAL,
    model_verdict   TEXT,
    add_committed   INTEGER NOT NULL DEFAULT 0,
    last_error      TEXT,
    created_at      REAL NOT NULL,
    updated_at      REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_cq_status_due ON capture_queue(status, next_attempt_at);
CREATE INDEX IF NOT EXISTS idx_cq_lease ON capture_queue(status, leased_until);
"""

_VALID_STATUS = ("pending", "inflight", "done", "dead")


def normalize_turn(user: str, assistant: str) -> str:
    """Normalization used for the idempotency hash: lowercase, collapse whitespace, strip.
    Kept deliberately simple + stable so the same turn always maps to the same key."""
    j = (user or "") + "\x00" + (assistant or "")
    return re.sub(r"\s+", " ", j).strip().lower()


def idem_key(session_id: str, turn_ordinal: int, user: str, assistant: str) -> str:
    """sha256(session_id + turn_ordinal + normalized(user+assistant)) — spec D-8."""
    h = hashlib.sha256()
    h.update(str(session_id or "").encode("utf-8"))
    h.update(b"\x1f")
    h.update(str(int(turn_ordinal)).encode("utf-8"))
    h.update(b"\x1f")
    h.update(normalize_turn(user, assistant).encode("utf-8"))
    return h.hexdigest()


class CaptureQueue:
    def __init__(self, db_path: str):
        self.db_path = db_path
        d = os.path.dirname(db_path)
        if d:
            os.makedirs(d, exist_ok=True)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        # A fresh connection per op keeps it thread-safe (the drain worker + the enqueue path
        # run on different threads). WAL lets readers not block the single writer.
        conn = sqlite3.connect(self.db_path, timeout=30.0, isolation_level=None)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute("PRAGMA busy_timeout=30000")
        return conn

    def _init_db(self) -> None:
        conn = self._connect()
        try:
            conn.executescript(_SCHEMA)
            # Idempotent migration for DBs created before add_committed existed.
            cols = {r["name"] for r in conn.execute("PRAGMA table_info(capture_queue)")}
            if "add_committed" not in cols:
                conn.execute("ALTER TABLE capture_queue ADD COLUMN add_committed INTEGER NOT NULL DEFAULT 0")
        finally:
            conn.close()

    # ---- enqueue (the durability boundary, INV-1) --------------------------
    def enqueue(self, key: str, payload: Dict[str, Any], *, now: Optional[float] = None) -> bool:
        """Insert a pending row. Duplicate idem_key -> no-op (returns False). INV-5.
        Returns True if a new row was inserted."""
        now = time.time() if now is None else now
        conn = self._connect()
        try:
            cur = conn.execute(
                "INSERT OR IGNORE INTO capture_queue "
                "(idem_key,payload,status,attempts,next_attempt_at,created_at,updated_at) "
                "VALUES (?,?,'pending',0,?,?,?)",
                (key, json.dumps(payload), now, now, now),
            )
            return cur.rowcount > 0
        finally:
            conn.close()

    # ---- lease (drain worker claims due rows, D-10) ------------------------
    def lease_one(self, *, lease_s: float = 120.0, now: Optional[float] = None) -> Optional[Dict[str, Any]]:
        """Atomically claim ONE due pending row -> inflight with a lease. Returns the row dict
        or None. Uses BEGIN IMMEDIATE so two concurrent drainers never claim the same row
        (the SQLite equivalent of FOR UPDATE SKIP LOCKED for a single-writer-at-a-time claim)."""
        now = time.time() if now is None else now
        conn = self._connect()
        try:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute(
                "SELECT * FROM capture_queue WHERE status='pending' AND next_attempt_at<=? "
                "ORDER BY next_attempt_at ASC LIMIT 1",
                (now,),
            ).fetchone()
            if row is None:
                conn.execute("COMMIT")
                return None
            conn.execute(
                "UPDATE capture_queue SET status='inflight', leased_until=?, model_verdict=NULL, updated_at=? "
                "WHERE idem_key=?",
                (now + lease_s, now, row["idem_key"]),
            )
            conn.execute("COMMIT")
            d = dict(row)
            d["status"] = "inflight"
            d["leased_until"] = now + lease_s
            d["payload"] = json.loads(d["payload"])
            return d
        except Exception:
            try:
                conn.execute("ROLLBACK")
            except Exception:
                pass
            raise
        finally:
            conn.close()

    def record_verdict(self, key: str, verdict: str, *, now: Optional[float] = None) -> None:
        """Stamp the durable model_verdict ('ok'|'fault') the moment a model call returns,
        BEFORE anything else (D-10). A crash after this but before mark_done is caught by the
        reaper as a model-fault (attempts++); a crash BEFORE this is environmental (no ++)."""
        now = time.time() if now is None else now
        conn = self._connect()
        try:
            conn.execute(
                "UPDATE capture_queue SET model_verdict=?, updated_at=? WHERE idem_key=?",
                (verdict, now, key),
            )
        finally:
            conn.close()

    def mark_add_committed(self, key: str, *, now: Optional[float] = None) -> None:
        """Set a STICKY flag (survives re-leasing, unlike model_verdict which lease_one clears) the
        instant a server add() has committed. The drainer uses it to know a remote row may exist even
        after a crash+reap reset attempts to 0, so an idem-check failure on a later lease never
        dead-letters and abandons a possibly-secret-bearing written row."""
        now = time.time() if now is None else now
        conn = self._connect()
        try:
            conn.execute(
                "UPDATE capture_queue SET add_committed=1, updated_at=? WHERE idem_key=?",
                (now, key),
            )
        finally:
            conn.close()

    def mark_done(self, key: str, *, now: Optional[float] = None) -> None:
        now = time.time() if now is None else now
        conn = self._connect()
        try:
            conn.execute(
                "UPDATE capture_queue SET status='done', leased_until=NULL, updated_at=? WHERE idem_key=?",
                (now, key),
            )
        finally:
            conn.close()

    def mark_retry(self, key: str, *, backoff_s: float, error: str = "", max_attempts: int = 5,
                   now: Optional[float] = None) -> str:
        """A real model-fault outcome: ++attempts, schedule a retry, or dead-letter at the cap.
        Returns the new status ('pending'|'dead')."""
        now = time.time() if now is None else now
        conn = self._connect()
        try:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute("SELECT attempts FROM capture_queue WHERE idem_key=?", (key,)).fetchone()
            attempts = (row["attempts"] if row else 0) + 1
            if attempts >= max_attempts:
                conn.execute(
                    "UPDATE capture_queue SET status='dead', attempts=?, leased_until=NULL, "
                    "last_error=?, updated_at=? WHERE idem_key=?",
                    (attempts, error[:500], now, key),
                )
                conn.execute("COMMIT")
                return "dead"
            conn.execute(
                "UPDATE capture_queue SET status='pending', attempts=?, next_attempt_at=?, "
                "leased_until=NULL, last_error=?, updated_at=? WHERE idem_key=?",
                (attempts, now + backoff_s, error[:500], now, key),
            )
            conn.execute("COMMIT")
            return "pending"
        except Exception:
            try:
                conn.execute("ROLLBACK")
            except Exception:
                pass
            raise
        finally:
            conn.close()

    def mark_scrub_retry(self, key: str, *, backoff_s: float, error: str = "",
                         now: Optional[float] = None) -> str:
        """A POST-WRITE SCRUB failure: the add already succeeded but the row's scrub couldn't be
        proven clean. Unlike mark_retry, this NEVER dead-letters — abandoning the row would leave a
        secret-bearing memory live/recallable with no automatic scrub path (a scrub is cheap +
        idempotent, so retrying forever is safe). ++attempts and reschedule as pending. Always
        returns 'pending'."""
        now = time.time() if now is None else now
        conn = self._connect()
        try:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute("SELECT attempts FROM capture_queue WHERE idem_key=?", (key,)).fetchone()
            attempts = (row["attempts"] if row else 0) + 1
            conn.execute(
                "UPDATE capture_queue SET status='pending', attempts=?, next_attempt_at=?, "
                "leased_until=NULL, last_error=?, updated_at=? WHERE idem_key=?",
                (attempts, now + backoff_s, error[:500], now, key),
            )
            conn.execute("COMMIT")
            return "pending"
        except Exception:
            try:
                conn.execute("ROLLBACK")
            except Exception:
                pass
            raise
        finally:
            conn.close()

    # ---- reaper (D-10): expired-lease inflight rows -> pending -------------
    def reap(self, *, now: Optional[float] = None, backoff_s: float = 30.0,
             max_attempts: int = 5) -> Dict[str, int]:
        """Sweep inflight rows whose lease expired. If a model_verdict='fault' is recorded, it
        was a real model fault -> treat as a retry (++attempts, may dead-letter). If NO verdict
        (or 'ok' with no mark_done -> shouldn't happen but treat safe), it's an environmental
        orphan (crash before/at the model call) -> back to pending WITHOUT ++attempts (no
        poison-loop). Returns counts."""
        now = time.time() if now is None else now
        counts = {"requeued_env": 0, "requeued_fault": 0, "dead": 0}
        conn = self._connect()
        try:
            conn.execute("BEGIN IMMEDIATE")
            rows = conn.execute(
                "SELECT idem_key, attempts, model_verdict FROM capture_queue "
                "WHERE status='inflight' AND leased_until IS NOT NULL AND leased_until<?",
                (now,),
            ).fetchall()
            for r in rows:
                if r["model_verdict"] == "fault":
                    attempts = r["attempts"] + 1
                    if attempts >= max_attempts:
                        conn.execute(
                            "UPDATE capture_queue SET status='dead', attempts=?, leased_until=NULL, "
                            "last_error='reaped: model fault, budget exhausted', updated_at=? WHERE idem_key=?",
                            (attempts, now, r["idem_key"]),
                        )
                        counts["dead"] += 1
                    else:
                        conn.execute(
                            "UPDATE capture_queue SET status='pending', attempts=?, next_attempt_at=?, "
                            "leased_until=NULL, updated_at=? WHERE idem_key=?",
                            (attempts, now + backoff_s, now, r["idem_key"]),
                        )
                        counts["requeued_fault"] += 1
                else:
                    # environmental orphan: no model verdict -> attempts NOT incremented
                    conn.execute(
                        "UPDATE capture_queue SET status='pending', next_attempt_at=?, "
                        "leased_until=NULL, updated_at=? WHERE idem_key=?",
                        (now, now, r["idem_key"]),
                    )
                    counts["requeued_env"] += 1
            conn.execute("COMMIT")
            return counts
        except Exception:
            try:
                conn.execute("ROLLBACK")
            except Exception:
                pass
            raise
        finally:
            conn.close()

    # ---- observability -----------------------------------------------------
    def counts(self) -> Dict[str, int]:
        conn = self._connect()
        try:
            out = {s: 0 for s in _VALID_STATUS}
            for r in conn.execute("SELECT status, COUNT(*) c FROM capture_queue GROUP BY status"):
                out[r["status"]] = r["c"]
            return out
        finally:
            conn.close()

    def purge_done(self, *, older_than_s: float = 86400.0, now: Optional[float] = None) -> int:
        """TTL cleanup of completed rows (D-15: the raw payload should not linger)."""
        now = time.time() if now is None else now
        conn = self._connect()
        try:
            cur = conn.execute(
                "DELETE FROM capture_queue WHERE status='done' AND updated_at<?",
                (now - older_than_s,),
            )
            return cur.rowcount
        finally:
            conn.close()
