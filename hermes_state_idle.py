"""Durable post-reply idle work, scoped to one profile's SessionDB."""

import math
import time
from typing import Optional


class IdleCompactionSuperseded(ValueError):
    """An optional background attempt lost its durable claim; not a model failure."""


class SessionIdleMixin:
    def invalidate_post_reply_idle(self, session_id: str) -> int:
        """Fence a queued or running worker at accepted inbound; return the new generation."""
        def _do(conn):
            if conn.execute("SELECT 1 FROM sessions WHERE id = ? AND ended_at IS NULL", (session_id,)).fetchone() is None:
                raise ValueError("no live session for idle invalidation")
            conn.execute("""INSERT INTO post_reply_idle(session_id, generation) VALUES (?, 1)
                ON CONFLICT(session_id) DO UPDATE SET generation = generation + 1,
                    due_at = NULL, claimed_by = NULL, claimed_until = NULL""", (session_id,))
            return conn.execute("SELECT generation FROM post_reply_idle WHERE session_id = ?", (session_id,)).fetchone()[0]
        return self._execute_write(_do)

    def arm_post_reply_idle(self, session_id: str, session_key: str, due_at: float,
                            *, expected_generation: int) -> bool:
        """Arm after persisted reply only if no later inbound invalidated this turn."""
        if not session_key or not math.isfinite(due_at):
            raise ValueError("session key and finite wall-clock due_at required")
        def _do(conn):
            return bool(conn.execute("""UPDATE post_reply_idle SET session_key = ?, due_at = ?,
                transcript_watermark = (SELECT COALESCE(MAX(id), 0) FROM messages WHERE session_id = ? AND active = 1),
                claimed_by = NULL, claimed_until = NULL
                WHERE session_id = ? AND generation = ? AND EXISTS (
                    SELECT 1 FROM sessions WHERE id = ? AND ended_at IS NULL AND session_key = ?)""",
                (session_key, due_at, session_id, session_id, expected_generation, session_id, session_key)).rowcount)
        return self._execute_write(_do)

    def claim_due_post_reply_idle(self, holder: str, *, now: Optional[float] = None,
                                   lease_ttl: float = 300.0, session_id: Optional[str] = None,
                                   expected_generation: Optional[int] = None):
        """Claim one due job; returns (session_id, generation, transcript_watermark).

        Pass both session_id and expected_generation for a scheduled callback; omit both
        to scan the next due job on startup. Each claim holder must be unique per attempt.
        """
        if (session_id is None) != (expected_generation is None):
            raise ValueError("session_id and expected_generation must be supplied together")
        if not holder or lease_ttl <= 0 or not math.isfinite(lease_ttl):
            raise ValueError("holder and positive finite lease_ttl required")
        now = time.time() if now is None else now
        def _do(conn):
            row = conn.execute("""SELECT i.session_id, i.generation, i.transcript_watermark
                FROM post_reply_idle i JOIN sessions s ON s.id = i.session_id
                WHERE s.ended_at IS NULL AND s.session_key = i.session_key
                  AND i.due_at <= ? AND (i.claimed_until IS NULL OR i.claimed_until <= ?)
                  AND i.transcript_watermark > i.last_compacted_watermark
                  AND i.transcript_watermark = (SELECT COALESCE(MAX(id), 0) FROM messages
                                                 WHERE session_id = i.session_id AND active = 1)
                  AND (? IS NULL OR i.session_id = ? AND i.generation = ?)
                ORDER BY i.due_at, i.session_id LIMIT 1""",
                (now, now, session_id, session_id, expected_generation)).fetchone()
            if row is None:
                return None
            conn.execute("UPDATE post_reply_idle SET claimed_by = ?, claimed_until = ? WHERE session_id = ?",
                         (holder, now + lease_ttl, row[0]))
            return tuple(row)
        return self._execute_write(_do)

    def renew_post_reply_idle(self, session_id: str, generation: int, holder: str, *,
                              now: Optional[float] = None, lease_ttl: float = 300.0) -> bool:
        """Keep one live worker's claim while it prepares or summarizes."""
        if not holder or lease_ttl <= 0 or not math.isfinite(lease_ttl):
            raise ValueError("holder and positive finite lease_ttl required")
        now = time.time() if now is None else now
        def _do(conn):
            return bool(conn.execute("""UPDATE post_reply_idle SET claimed_until = ?
                WHERE session_id = ? AND generation = ? AND claimed_by = ?
                  AND claimed_until > ? AND due_at IS NOT NULL
                  AND EXISTS (SELECT 1 FROM sessions WHERE id = ? AND ended_at IS NULL)""",
                (now + lease_ttl, session_id, generation, holder, now, session_id)).rowcount)
        return self._execute_write(_do)

    def release_post_reply_idle(self, session_id: str, generation: int, holder: str,
                                *, retry_at: Optional[float] = None) -> bool:
        """Release an attempt or delay its retry without reviving a stale generation."""
        def _do(conn):
            return bool(conn.execute("""UPDATE post_reply_idle SET claimed_by = NULL,
                claimed_until = NULL, due_at = COALESCE(?, due_at)
                WHERE session_id = ? AND generation = ? AND claimed_by = ?""",
                (retry_at, session_id, generation, holder)).rowcount)
        return self._execute_write(_do)

    def _check_idle_compaction_claim(self, conn, session_id, generation, watermark, holder,
                                     pending_check=None):
        if pending_check is not None and pending_check():
            raise IdleCompactionSuperseded("idle compaction fence lost; inbound is buffered")
        row = conn.execute("""SELECT i.generation, i.transcript_watermark,
            i.claimed_by, i.claimed_until, i.due_at, i.last_compacted_watermark
            FROM post_reply_idle i JOIN sessions s ON s.id = i.session_id
            WHERE i.session_id = ? AND s.ended_at IS NULL AND s.session_key = i.session_key""",
            (session_id,)).fetchone()
        current = conn.execute("SELECT COALESCE(MAX(id), 0) FROM messages WHERE session_id = ? AND active = 1",
                               (session_id,)).fetchone()[0]
        turn_lease = conn.execute(
            "SELECT 1 FROM session_turn_leases WHERE conversation_id = ? AND expires_at > ?",
            (self._session_turn_lease_key_on_conn(conn, session_id), time.time()),
        ).fetchone()
        if (turn_lease is not None or row is None or row[0] != generation or row[1] != watermark or row[2] != holder
                or row[3] is None or row[3] <= time.time() or row[4] is None
                or watermark <= row[5] or current != watermark):
            raise IdleCompactionSuperseded("idle compaction fence lost; refusing stale summary")

    def _finish_idle_compaction(self, conn, session_id):
        watermark = conn.execute("SELECT COALESCE(MAX(id), 0) FROM messages WHERE session_id = ? AND active = 1",
                                 (session_id,)).fetchone()[0]
        conn.execute("""UPDATE post_reply_idle SET due_at = NULL, claimed_by = NULL,
            claimed_until = NULL, last_compacted_watermark = ?, transcript_watermark = NULL
            WHERE session_id = ?""", (watermark, session_id))
