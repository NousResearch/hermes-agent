"""Canonical pre-ACK journal, on the RouterStore transaction/connection.

Receipt is NOT permission to execute. Native admission and wire confirmation
are independent durable facts. A send intent is committed as ``uncertain``
before calling Slack: a crash or transport exception cannot prove non-delivery.
Only confirmed + admitted receipts can enter the existing Router service.
Unknown receipts await an actual Slack redelivery; never invent ACK evidence.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import re
import time
import weakref
from dataclasses import replace

from .models import CanonicalEvent

# Retain canonical routing text, never the SocketMode payload, files, blocks,
# response_url or credentials. Redaction quarantines, rather than executing an
# altered directive. Oversized input fails before ACK rather than truncating it.
_SECRET = re.compile(
    r"xox[baprs]-[A-Za-z0-9-]+|xapp-[A-Za-z0-9-]+|"
    r"https://hooks\.slack\.com/[^\s<>]+|"
    r"(?i:bearer\s+)[A-Za-z0-9._~+/-]+=*|"
    r"(?i:(?:password|api[_-]?key|access[_-]?token|secret)\s*[:=]\s*)[^\s,;]+"
)
_METADATA = frozenset({"slack_team_id", "slack_event_subtype", "edited", "deleted", "file_only", "forwarded"})


class AckJournal:
    MAX_PAYLOAD_BYTES = 65536
    MAX_PENDING = 10000

    def __init__(self, store):
        self.store = store
        self._locks = weakref.WeakValueDictionary()
        with store._lock:
            # Explicit FULL also covers a pre-existing database configured for WAL.
            store._conn.execute("PRAGMA synchronous=FULL")
        with store._transaction():
            store._conn.execute("""CREATE TABLE IF NOT EXISTS router_ack_receipts (
                event_id TEXT PRIMARY KEY, payload_json TEXT, payload_hash TEXT NOT NULL,
                wire_state TEXT NOT NULL DEFAULT 'prepared',
                native_state TEXT NOT NULL DEFAULT 'pending',
                handoff_state TEXT NOT NULL DEFAULT 'pending',
                reason TEXT NOT NULL DEFAULT '', created_at REAL NOT NULL,
                updated_at REAL NOT NULL)""")
            store._conn.execute("""CREATE TABLE IF NOT EXISTS router_ack_attempts (
                envelope_id TEXT PRIMARY KEY, event_id TEXT NOT NULL,
                state TEXT NOT NULL, attempts INTEGER NOT NULL DEFAULT 0,
                updated_at REAL NOT NULL,
                FOREIGN KEY(event_id) REFERENCES router_ack_receipts(event_id))""")

    def lock(self, kind, identity):
        key = (kind, identity)
        lock = self._locks.get(key)
        if lock is None:
            lock = asyncio.Lock()
            self._locks[key] = lock
        return lock

    @staticmethod
    def canonical_payload(event):
        if not event.event_id or not event.channel_id or not event.thread_ts:
            raise ValueError("ACK receipt missing canonical identity")
        metadata = {k: v for k, v in event.metadata.items() if k in _METADATA}
        value = replace(event, metadata=metadata).to_dict()
        raw = json.dumps(value, sort_keys=True, ensure_ascii=False)
        if len(raw.encode()) > AckJournal.MAX_PAYLOAD_BYTES:
            raise ValueError("ACK receipt payload exceeds retention bound")
        def scrub(value):
            if isinstance(value, str):
                return _SECRET.sub("[REDACTED]", value)
            if isinstance(value, dict):
                return {key: scrub(item) for key, item in value.items()}
            if isinstance(value, list):
                return [scrub(item) for item in value]
            return value
        safe = json.dumps(scrub(value), sort_keys=True, ensure_ascii=False)
        return safe, safe != raw

    def prepare(self, event, envelope_id):
        payload, redacted = self.canonical_payload(event)
        digest = hashlib.sha256(payload.encode()).hexdigest()
        now = time.time()
        with self.store._transaction():
            conn = self.store._conn
            row = conn.execute("SELECT payload_hash FROM router_ack_receipts WHERE event_id=?", (event.event_id,)).fetchone()
            if row and row[0] != digest:
                raise ValueError("ACK canonical identity collision")
            if row is None:
                count = conn.execute("SELECT count(*) FROM router_ack_receipts WHERE payload_json IS NOT NULL").fetchone()[0]
                if count >= self.MAX_PENDING:
                    raise RuntimeError("ACK receipt retention capacity reached")
                conn.execute("""INSERT INTO router_ack_receipts
                    (event_id, payload_json, payload_hash, native_state, reason, created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?)""", (
                        event.event_id, payload, digest,
                        'rejected' if redacted else 'pending',
                        'privacy_redacted' if redacted else '', now, now))
            old = conn.execute("SELECT event_id FROM router_ack_attempts WHERE envelope_id=?", (envelope_id,)).fetchone()
            if old and old[0] != event.event_id:
                raise ValueError("ACK envelope identity collision")
            conn.execute("""INSERT OR IGNORE INTO router_ack_attempts
                (envelope_id, event_id, state, updated_at) VALUES (?, ?, 'prepared', ?)""",
                (envelope_id, event.event_id, now))
        return self.get(event.event_id)

    def get(self, event_id):
        with self.store._lock:
            row = self.store._conn.execute("SELECT * FROM router_ack_receipts WHERE event_id=?", (event_id,)).fetchone()
        return dict(row) if row else None

    def begin_wire(self, event_id, envelope_id):
        """False means this exact envelope already has confirmed wire evidence.

        A redelivered uncertain envelope permits a new attempt, not a claim of
        network exactly-once. Successful repeated callbacks are locally deduped.
        """
        with self.store._transaction():
            conn = self.store._conn
            row = conn.execute("SELECT state FROM router_ack_attempts WHERE envelope_id=? AND event_id=?", (envelope_id, event_id)).fetchone()
            if not row:
                raise RuntimeError("wire ACK has no prepared receipt")
            if row[0] == 'confirmed':
                return False
            conn.execute("UPDATE router_ack_attempts SET state='uncertain', attempts=attempts+1, updated_at=? WHERE envelope_id=?", (time.time(), envelope_id))
            conn.execute("UPDATE router_ack_receipts SET wire_state='uncertain', updated_at=? WHERE event_id=? AND wire_state != 'confirmed'", (time.time(), event_id))
        return True

    def confirm_wire(self, event_id, envelope_id):
        with self.store._transaction():
            conn = self.store._conn
            cursor = conn.execute("UPDATE router_ack_attempts SET state='confirmed', updated_at=? WHERE envelope_id=? AND event_id=? AND state='uncertain'", (time.time(), envelope_id, event_id))
            if not cursor.rowcount:
                raise RuntimeError("ACK confirmation without send intent")
            conn.execute("UPDATE router_ack_receipts SET wire_state='confirmed', updated_at=? WHERE event_id=?", (time.time(), event_id))

    def native(self, event_id, admitted):
        with self.store._transaction():
            self.store._conn.execute("""UPDATE router_ack_receipts
                SET native_state=?, reason=?, updated_at=?
                WHERE event_id=? AND native_state='pending'""",
                ('admitted' if admitted else 'rejected', '' if admitted else 'native_filtered', time.time(), event_id))
            if not admitted:
                self.store._conn.execute("UPDATE router_ack_receipts SET payload_json=NULL WHERE event_id=? AND native_state='rejected'", (event_id,))

    def ready(self):
        with self.store._lock:
            rows = self.store._conn.execute("""SELECT event_id FROM router_ack_receipts
                WHERE wire_state='confirmed' AND native_state='admitted'
                AND handoff_state='pending' AND payload_json IS NOT NULL""").fetchall()
        return [row[0] for row in rows]

    def promoted(self, event_id):
        with self.store._transaction():
            self.store._conn.execute("""UPDATE router_ack_receipts
                SET handoff_state='promoted', payload_json=NULL, updated_at=?
                WHERE event_id=? AND wire_state='confirmed' AND native_state='admitted'""", (time.time(), event_id))

    async def promote(self, router, event_id, *, fenced):
        # All attached receivers share the store journal and this lock. The
        # existing service preserves meeting/control special paths and inbox PK.
        async with self.lock('promote', event_id):
            if fenced():
                return
            row = self.get(event_id)
            if not row or (row['wire_state'], row['native_state'], row['handoff_state']) != ('confirmed', 'admitted', 'pending'):
                return
            event = CanonicalEvent.from_dict(json.loads(row['payload_json']))
            if not router.owns_event(event):
                return  # Current config may no longer authorize a recovered event.
            router.ack_gate.complete(event_id, success=True)
            await router.enqueue_after_ack(event)
            if not fenced():
                self.promoted(event_id)
