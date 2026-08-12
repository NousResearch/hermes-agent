"""Delivery tracking with attempt counting and a dead-letter queue.

Extracted from the t_0553ea32 scratch pipeline (publishing_bridge/tracker.py)
and adapted to the LIVE content engine. The live path had an idempotent
enqueue claim but no retry accounting: a failed Postiz insert just released
the claim forever, with no attempt cap, no dead-letter queue, and no
observability of repeated failures.

This module is additive. It layers a delivery table + DLQ over the live
``drafts`` table (same DB). The existing ``enqueue_state`` column contract
(NULL -> 'claiming' -> 'enqueued' on success / released on failure) is
preserved; this tracker adds ``publish_delivery`` accounting around it.

Schema:

    publish_delivery:
        draft_id      TEXT PRIMARY KEY  -- mirrors drafts.id
        platform      TEXT NOT NULL
        attempt_count INTEGER NOT NULL DEFAULT 0
        max_attempts  INTEGER NOT NULL DEFAULT 3
        last_error    TEXT
        next_retry_at TEXT
        status        TEXT NOT NULL DEFAULT 'pending'
        created_at    TEXT NOT NULL
        updated_at    TEXT NOT NULL

    dead_letter_queue:
        id           TEXT PRIMARY KEY
        draft_id     TEXT NOT NULL
        platform     TEXT NOT NULL
        attempts     INTEGER NOT NULL
        last_error   TEXT
        moved_at     TEXT NOT NULL
        acknowledged INTEGER NOT NULL DEFAULT 0

Flow:

    pending ─claim()─► claiming ─success─► enqueued ─► published (in drafts)
                  └─fail─► failed (attempt+1) ─► retry after backoff
                           └─attempts exhausted─► dead_letter
"""

from __future__ import annotations

import os
import sqlite3
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional


def _default_db_path() -> str:
    env = os.environ.get("CONTENT_ENGINE_DB_PATH")
    if env:
        return str(Path(env).resolve())
    try:
        from config import DB_PATH

        return str(DB_PATH)
    except Exception:
        return str(Path(__file__).resolve().parent / "db" / "content_engine.db")


DELIVERY_SCHEMA = """
CREATE TABLE IF NOT EXISTS publish_delivery (
    draft_id      TEXT PRIMARY KEY,
    platform      TEXT NOT NULL,
    attempt_count INTEGER NOT NULL DEFAULT 0,
    max_attempts  INTEGER NOT NULL DEFAULT 3,
    last_error    TEXT,
    next_retry_at TEXT,
    status        TEXT NOT NULL DEFAULT 'pending',
    created_at    TEXT NOT NULL,
    updated_at    TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_pd_status ON publish_delivery(status);
CREATE INDEX IF NOT EXISTS idx_pd_retry ON publish_delivery(next_retry_at);

CREATE TABLE IF NOT EXISTS dead_letter_queue (
    id           TEXT PRIMARY KEY,
    draft_id     TEXT NOT NULL,
    platform     TEXT NOT NULL,
    attempts     INTEGER NOT NULL,
    last_error   TEXT,
    moved_at     TEXT NOT NULL,
    acknowledged INTEGER NOT NULL DEFAULT 0
);

CREATE INDEX IF NOT EXISTS idx_dlq_ack ON dead_letter_queue(acknowledged);
CREATE INDEX IF NOT EXISTS idx_dlq_draft ON dead_letter_queue(draft_id);
"""


class DeliveryError(Exception):
    """Raised on invalid delivery-state operations."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class PublishTracker:
    """Attempt-counted delivery tracking over the live content engine DB."""

    def __init__(self, db_path: Optional[str] = None):
        self.db_path = db_path or _default_db_path()

    def _conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    # ── registration ───────────────────────────────────────────────────────

    def register(self, draft_id: str, platform: str, max_attempts: int = 3) -> None:
        """Idempotently create the delivery row for a draft+platform."""
        now = _utc_now()
        conn = self._conn()
        try:
            conn.executescript(DELIVERY_SCHEMA)
            conn.execute(
                """
                INSERT OR IGNORE INTO publish_delivery
                  (draft_id, platform, max_attempts, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (draft_id, platform, max_attempts, now, now),
            )
            conn.commit()
        finally:
            conn.close()

    # ── claim (attempt + 1) ────────────────────────────────────────────────

    def claim(self, draft_id: str, platform: str) -> bool:
        """Atomically claim the draft's enqueue slot and increment attempts.

        Preserves the live ``drafts.enqueue_state`` contract (NULL/pending ->
        'claiming') and increments the delivery attempt counter in the same
        transaction. Returns True if this caller won the claim.
        """
        now = _utc_now()
        conn = self._conn()
        try:
            conn.executescript(DELIVERY_SCHEMA)
            cur = conn.execute(
                """
                UPDATE drafts SET enqueue_state = 'claiming'
                WHERE id = ? AND (enqueue_state IS NULL OR enqueue_state = 'pending')
                """,
                (draft_id,),
            )
            won = cur.rowcount > 0
            if won:
                conn.execute(
                    """
                    UPDATE publish_delivery
                       SET attempt_count = attempt_count + 1,
                           status = 'claiming',
                           updated_at = ?
                     WHERE draft_id = ?
                    """,
                    (now, draft_id),
                )
            conn.commit()
            return won
        finally:
            conn.close()

    # ── outcomes ───────────────────────────────────────────────────────────

    def mark_enqueued(self, draft_id: str, platform: str) -> None:
        conn = self._conn()
        try:
            conn.execute(
                """UPDATE publish_delivery SET status = 'enqueued', updated_at = ?
                   WHERE draft_id = ?""",
                (_utc_now(), draft_id),
            )
            conn.commit()
        finally:
            conn.close()

    def mark_failed(
        self,
        draft_id: str,
        platform: str,
        error: str,
        backoff_minutes: int = 30,
    ) -> None:
        """Record a failed attempt and schedule a retry (or dead-letter)."""
        now = _utc_now()
        conn = self._conn()
        try:
            conn.executescript(DELIVERY_SCHEMA)
            row = conn.execute(
                "SELECT attempt_count, max_attempts FROM publish_delivery WHERE draft_id = ?",
                (draft_id,),
            ).fetchone()
            if row is None:
                raise DeliveryError(f"no delivery row for draft {draft_id}")
            attempts = row["attempt_count"]
            max_attempts = row["max_attempts"]
            next_retry = (
                datetime.now(timezone.utc) + timedelta(minutes=backoff_minutes)
            ).isoformat()
            conn.execute(
                """UPDATE publish_delivery
                   SET status = 'failed', last_error = ?, next_retry_at = ?, updated_at = ?
                   WHERE draft_id = ?""",
                (error, next_retry, now, draft_id),
            )
            # Release the drafts enqueue claim so a retry can pick it up.
            conn.execute(
                """UPDATE drafts SET enqueue_state = 'pending'
                   WHERE id = ? AND enqueue_state = 'claiming'""",
                (draft_id,),
            )
            if attempts >= max_attempts:
                self._dead_letter(conn, draft_id, platform, error, attempts)
            conn.commit()
        finally:
            conn.close()

    def _dead_letter(
        self,
        conn: sqlite3.Connection,
        draft_id: str,
        platform: str,
        error: str,
        attempts: int,
    ) -> None:
        now = _utc_now()
        conn.execute(
            """
            INSERT OR IGNORE INTO dead_letter_queue
              (id, draft_id, platform, attempts, last_error, moved_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (uuid.uuid4().hex, draft_id, platform, attempts, error, now),
        )
        conn.execute(
            """UPDATE publish_delivery SET status = 'dead_letter', next_retry_at = NULL,
               updated_at = ? WHERE draft_id = ?""",
            (now, draft_id),
        )

    # ── recovery + queries ─────────────────────────────────────────────────

    def reset_stuck_claims(self, max_age_minutes: int = 30) -> int:
        """Release drafts stuck in 'claiming' for too long (crashed worker).

        Returns the number of records reset.
        """
        cutoff = (
            datetime.now(timezone.utc) - timedelta(minutes=max_age_minutes)
        ).isoformat()
        conn = self._conn()
        try:
            conn.executescript(DELIVERY_SCHEMA)
            rows = conn.execute(
                """SELECT draft_id FROM publish_delivery
                   WHERE status = 'claiming' AND updated_at <= ?""",
                (cutoff,),
            ).fetchall()
            for r in rows:
                conn.execute(
                    """UPDATE drafts SET enqueue_state = 'pending'
                       WHERE id = ? AND enqueue_state = 'claiming'""",
                    (r["draft_id"],),
                )
            cur = conn.execute(
                """UPDATE publish_delivery SET status = 'failed', updated_at = ?,
                   last_error = 'claim timeout: process crashed or stalled'
                   WHERE status = 'claiming' AND updated_at <= ?""",
                (_utc_now(), cutoff),
            )
            conn.commit()
            return cur.rowcount
        finally:
            conn.close()

    def list_dead_letters(self, acknowledged: Optional[int] = None) -> list[dict]:
        conn = self._conn()
        try:
            conn.executescript(DELIVERY_SCHEMA)
            if acknowledged is None:
                rows = conn.execute(
                    "SELECT * FROM dead_letter_queue ORDER BY moved_at ASC"
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM dead_letter_queue WHERE acknowledged = ? ORDER BY moved_at ASC",
                    (acknowledged,),
                ).fetchall()
            return [dict(r) for r in rows]
        finally:
            conn.close()

    def status(self, draft_id: str) -> Optional[dict]:
        conn = self._conn()
        try:
            row = conn.execute(
                "SELECT * FROM publish_delivery WHERE draft_id = ?", (draft_id,)
            ).fetchone()
            return dict(row) if row else None
        finally:
            conn.close()
