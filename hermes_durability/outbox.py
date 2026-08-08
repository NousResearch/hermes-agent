"""Outbox worker: exactly-once outbound delivery.

Rows are enqueued atomically with the journal transaction that produced
them (see JournalTransaction.enqueue_outbound).  The worker drains rows
per-session in enqueue order, passes each envelope through the guardrail,
sends via an adapter using outbox_id as the idempotency key, and records
OutboxDelivered in the journal in the SAME SQLite transaction that flips
the row to its terminal status — a crash can never observe one without
the other.

Concurrency/crash model:
  * A row is CLAIMED (status 'pending' -> 'inflight') before its send is
    attempted, so a background worker tick and a synchronous drain (e.g.
    recovery) can never both deliver the same row.  ``drain_once`` is
    additionally serialized by a lock.
  * A crash while a row is 'inflight' means the send outcome is unknown;
    startup resets 'inflight' -> 'pending' and retries.  The adapter-level
    idempotency key (outbox_id) makes that retry safe.

Exactly-once therefore reduces to: at-least-once delivery attempts +
adapter-level idempotency keyed on outbox_id.
"""

from __future__ import annotations

import json
import threading
import time
from typing import Callable, Optional, Protocol

from hermes_durability.guardrail import Envelope, Guardrail
from hermes_durability.journal import Journal, OUTBOX_DELIVERED, _canonical, _txn


class Adapter(Protocol):
    """Transport adapter. send() MUST be idempotent on idempotency_key
    (map it to Slack client_msg_id, Discord nonce, email Message-ID,
    webhook Idempotency-Key header...). Raise on failure."""

    def send(self, envelope: Envelope, idempotency_key: str) -> dict: ...


class RetryPolicy:
    def __init__(self, base_delay: float = 0.5, max_delay: float = 60.0,
                 max_attempts: int = 8):
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.max_attempts = max_attempts

    def next_delay(self, attempts: int) -> float:
        return min(self.max_delay, self.base_delay * (2 ** max(0, attempts - 1)))


class OutboxWorker:
    def __init__(self, journal: Journal, adapters: dict[str, Adapter],
                 guardrail: Optional[Guardrail] = None,
                 retry: Optional[RetryPolicy] = None,
                 on_dead_letter: Optional[Callable[[str, str], None]] = None):
        self.journal = journal
        self.adapters = adapters
        self.guardrail = guardrail
        self.retry = retry or RetryPolicy()
        self.on_dead_letter = on_dead_letter
        self._drain_lock = threading.Lock()
        self._wake = threading.Event()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    # -- lifecycle ---------------------------------------------------------
    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, daemon=True,
                                        name="hermes-durable-outbox")
        self._thread.start()

    def stop(self, timeout: float = 5.0) -> None:
        self._stop.set()
        self._wake.set()
        if self._thread:
            self._thread.join(timeout)

    def signal(self) -> None:
        """Called by the runtime after every commit that enqueued rows."""
        self._wake.set()

    def reset_inflight(self) -> int:
        """Startup: rows claimed by a crashed process go back to pending."""
        conn = self.journal._conn
        with self.journal._lock:
            with _txn(conn):
                cur = conn.execute(
                    "UPDATE outbox SET status = 'pending'"
                    " WHERE status = 'inflight'")
            return cur.rowcount

    # -- draining ----------------------------------------------------------
    def drain_once(self, now: Optional[float] = None) -> int:
        """Process all currently-eligible rows once. Returns rows delivered.
        Safe to call synchronously (used by recovery and tests); serialized
        against the background worker by ``_drain_lock`` and per-row claims.
        """
        with self._drain_lock:
            return self._drain_locked(now)

    def _drain_locked(self, now: Optional[float]) -> int:
        now = now if now is not None else time.time()
        conn = self.journal._conn
        with self.journal._lock:
            rows = conn.execute(
                "SELECT outbox_id, session_id, channel, payload, attempts"
                " FROM outbox WHERE status = 'pending' AND next_retry_at <= ?"
                " ORDER BY session_id, seq_hint", (now,)).fetchall()
        delivered = 0
        # Strict per-session ordering: once a session's row fails, skip the
        # rest of that session's rows this pass.
        blocked_sessions: set[str] = set()
        for outbox_id, session_id, channel, payload, attempts in rows:
            if session_id in blocked_sessions:
                continue
            if not self._claim(outbox_id):
                continue  # another drainer got it between SELECT and claim
            envelope = Envelope(session_id=session_id, channel=channel,
                                payload=json.loads(payload),
                                outbox_id=outbox_id)
            if not self._deliver(envelope, attempts):
                blocked_sessions.add(session_id)
            else:
                delivered += 1
        return delivered

    def _claim(self, outbox_id: str) -> bool:
        conn = self.journal._conn
        with self.journal._lock:
            with _txn(conn):
                cur = conn.execute(
                    "UPDATE outbox SET status = 'inflight'"
                    " WHERE outbox_id = ? AND status = 'pending'",
                    (outbox_id,))
            return cur.rowcount == 1

    def _deliver(self, envelope: Envelope, attempts: int) -> bool:
        outbox_id = envelope.outbox_id
        session_id = envelope.session_id

        if self.guardrail is not None:
            verdict = self.guardrail.evaluate(envelope)
            if verdict.action in ("block", "drop"):
                self._finalize(outbox_id, session_id,
                               status="blocked",
                               detail={"guardrail": verdict.action,
                                       "policies": verdict.matched_policies})
                return True
            envelope = verdict.envelope or envelope

        adapter = self.adapters.get(envelope.channel)
        if adapter is None:
            self._fail(outbox_id, session_id, envelope, attempts,
                       f"no adapter for channel {envelope.channel!r}")
            return False
        try:
            receipt = adapter.send(envelope, idempotency_key=outbox_id)
        except Exception as exc:  # noqa: BLE001 - transport errors are data
            self._fail(outbox_id, session_id, envelope, attempts, repr(exc))
            return False
        self._finalize(outbox_id, session_id, status="delivered",
                       detail={"receipt": receipt})
        return True

    def _finalize(self, outbox_id: str, session_id: str, status: str,
                  detail: dict) -> None:
        """Journal OutboxDelivered and mark the row terminal in ONE
        transaction (via Journal.append extra_sql) — a crash between the
        two would otherwise leave a journaled delivery with a still-pending
        row, and the retry would append a duplicate OutboxDelivered."""
        self.journal.append(
            session_id, OUTBOX_DELIVERED,
            {"outbox_id": outbox_id, "status": status, **detail},
            extra_sql=[(
                "UPDATE outbox SET status = ?, delivered_at = ?"
                " WHERE outbox_id = ?",
                (status, time.time(), outbox_id))])

    def _fail(self, outbox_id: str, session_id: str, envelope: Envelope,
              attempts: int, error: str) -> None:
        attempts += 1
        conn = self.journal._conn
        with self.journal._lock:
            if attempts >= self.retry.max_attempts:
                with _txn(conn):
                    conn.execute(
                        "INSERT OR REPLACE INTO dlq (outbox_id, session_id,"
                        " channel, payload, error, attempts, moved_at)"
                        " VALUES (?,?,?,?,?,?,?)",
                        (outbox_id, session_id, envelope.channel,
                         _canonical(envelope.payload), error, attempts,
                         time.time()))
                    conn.execute(
                        "UPDATE outbox SET status = 'deadletter', attempts = ?,"
                        " last_error = ? WHERE outbox_id = ?",
                        (attempts, error, outbox_id))
                if self.on_dead_letter:
                    self.on_dead_letter(outbox_id, error)
            else:
                delay = self.retry.next_delay(attempts)
                with _txn(conn):
                    conn.execute(
                        "UPDATE outbox SET status = 'pending', attempts = ?,"
                        " last_error = ?, next_retry_at = ?"
                        " WHERE outbox_id = ?",
                        (attempts, error, time.time() + delay, outbox_id))

    def replay_dead_letter(self, outbox_id: str) -> bool:
        conn = self.journal._conn
        with self.journal._lock:
            row = conn.execute(
                "SELECT outbox_id FROM dlq WHERE outbox_id = ?",
                (outbox_id,)).fetchone()
            if not row:
                return False
            with _txn(conn):
                conn.execute(
                    "UPDATE outbox SET status = 'pending', attempts = 0,"
                    " next_retry_at = 0, last_error = NULL WHERE outbox_id = ?",
                    (outbox_id,))
                conn.execute("DELETE FROM dlq WHERE outbox_id = ?", (outbox_id,))
        self.signal()
        return True

    def _run(self) -> None:
        while not self._stop.is_set():
            self._wake.wait(timeout=0.25)
            self._wake.clear()
            if self._stop.is_set():
                break
            try:
                self.drain_once()
            except Exception:  # pragma: no cover - worker must not die
                time.sleep(0.5)
