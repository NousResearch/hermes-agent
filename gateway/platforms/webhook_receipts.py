"""Durable webhook delivery receipts: one row per authenticated delivery, kept across restarts.

The receipt log answers two questions the in-process idempotency cache could not:

* **Was this delivery already accepted?** ``deliver_only``, ``cron_job`` and coalesced routes act
  before (or without) the session admission ledger, so their only replay guard used to be a 1-hour
  in-memory set. A claim row here (``claim = 1``, unique per profile/route/delivery id) survives a
  gateway restart, so a provider retry after a restart is answered as a duplicate instead of
  re-running the job or re-sending the message. A failed delivery releases its claim, so the
  provider's retry is processed.
* **What happened to it?** Every authenticated delivery (accepted, duplicate, ignored by the route's
  ``events``/``filters``/``script``, or failed) gets a row with the reason and a link to what it
  started (cron job, delivery target). Payload bodies are never stored. ``hermes webhook history``
  reads this file.

The database lives beside ``webhook_subscriptions.json`` in the gateway's home. Rows older than
``platforms.webhook.extra.receipt_retention_days`` (default 7) are pruned, and the history rows are
capped at ``MAX_HISTORY_ROWS`` so an authenticated flood cannot grow the file without bound.
"""

from __future__ import annotations

import json
import logging
from contextvars import ContextVar
import sqlite3
import threading
import time
from contextlib import closing, contextmanager
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

RECEIPTS_FILENAME = "webhook_receipts.db"
DEFAULT_RETENTION_DAYS = 7.0
MAX_HISTORY_ROWS = 10_000
_PRUNE_INTERVAL_SECONDS = 600.0

ACCEPTED, DUPLICATE, IGNORED, FAILED = "accepted", "duplicate", "ignored", "failed"

# The receipt of the POST being handled, so the route handlers can add the run they started (``ref``) or
# a reason the HTTP answer must not carry. Outside a POST (coalesced flush, replay) it is a throwaway.
CURRENT: ContextVar[Optional[dict]] = ContextVar("webhook_receipt", default=None)


def current() -> dict:
    receipt = CURRENT.get()
    return receipt if receipt is not None else {}

_SCHEMA = """
CREATE TABLE IF NOT EXISTS receipts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    profile TEXT NOT NULL,
    route TEXT NOT NULL,
    delivery_id TEXT NOT NULL,
    event_type TEXT NOT NULL DEFAULT '',
    outcome TEXT NOT NULL,
    reason TEXT NOT NULL DEFAULT '',
    ref TEXT NOT NULL DEFAULT '',
    received_at REAL NOT NULL,
    claim INTEGER NOT NULL DEFAULT 0
);
CREATE UNIQUE INDEX IF NOT EXISTS receipts_claim ON receipts(profile, route, delivery_id) WHERE claim = 1;
CREATE INDEX IF NOT EXISTS receipts_received ON receipts(received_at);
"""


def receipts_path(home: Path) -> Path:
    return Path(home) / RECEIPTS_FILENAME


class WebhookReceipts:
    """Synchronous SQLite store; the adapter calls it on a worker thread so commits never block the loop."""

    def __init__(self, path: Path, retention_days: float = DEFAULT_RETENTION_DAYS):
        self.path = Path(path)
        self.retention_seconds = max(float(retention_days), 0.0) * 86400.0
        self._next_prune_at = 0.0
        self._conn: Optional[sqlite3.Connection] = None
        # One connection, opened on first use and shared by the worker threads that call in: closing
        # the last connection to a WAL database checkpoints and fsyncs, which per delivery cost ~30 ms.
        self._lock = threading.Lock()

    @contextmanager
    def _connect(self):
        with self._lock:
            if self._conn is None:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                conn = sqlite3.connect(str(self.path), timeout=5.0, check_same_thread=False)
                # WAL + synchronous=NORMAL: a commit survives a process crash or restart (the case dedupe
                # is for) without an fsync per delivery; only an OS crash can lose the newest commits.
                conn.execute("PRAGMA journal_mode=WAL")
                conn.execute("PRAGMA synchronous=NORMAL")
                conn.executescript(_SCHEMA)
                self._conn = conn
            yield self._conn

    def claim(self, identity, event_type: str, now: float) -> Optional[int]:
        """Durably claim *identity*; the new row id, or None when a live claim already holds it."""
        with self._connect() as conn, conn:
            self._prune(conn, now)
            try:
                cursor = conn.execute(
                    "INSERT INTO receipts (profile, route, delivery_id, event_type, outcome, received_at, claim)"
                    " VALUES (?, ?, ?, ?, ?, ?, 1)",
                    (identity.profile, identity.route, identity.delivery_id, event_type, ACCEPTED, now))
            except sqlite3.IntegrityError:
                return None
            return cursor.lastrowid

    def is_claimed(self, identity) -> bool:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT 1 FROM receipts WHERE profile = ? AND route = ? AND delivery_id = ? AND claim = 1"
                " AND received_at >= ?",
                (identity.profile, identity.route, identity.delivery_id, time.time() - self.retention_seconds),
            ).fetchone()
        return row is not None

    def settle(self, row_id: int, outcome: str, reason: str, ref: str, *, keep_claim: bool) -> None:
        """Final outcome for a claimed row. A confirmed failure passes ``keep_claim=False`` so the provider's
        retry is processed; an unknown outcome (the send may have happened) keeps it."""
        with self._connect() as conn, conn:
            conn.execute("UPDATE receipts SET outcome = ?, reason = ?, ref = ?, claim = ? WHERE id = ?",
                         (outcome, reason, ref, int(keep_claim), row_id))

    def release(self, identity, reason: str) -> None:
        """Give back a claim whose work was dropped after the ack (a coalesced batch that was not admitted)."""
        with self._connect() as conn, conn:
            conn.execute(
                "UPDATE receipts SET outcome = ?, reason = ?, claim = 0"
                " WHERE profile = ? AND route = ? AND delivery_id = ? AND claim = 1",
                (FAILED, reason, identity.profile, identity.route, identity.delivery_id))

    def record(self, identity, event_type: str, outcome: str, reason: str, ref: str, now: float) -> None:
        with self._connect() as conn, conn:
            self._prune(conn, now)
            conn.execute(
                "INSERT INTO receipts (profile, route, delivery_id, event_type, outcome, reason, ref, received_at)"
                " VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (identity.profile, identity.route, identity.delivery_id, event_type, outcome, reason, ref, now))

    def _prune(self, conn: sqlite3.Connection, now: float) -> None:
        if now < self._next_prune_at:
            return
        self._next_prune_at = now + _PRUNE_INTERVAL_SECONDS
        conn.execute("DELETE FROM receipts WHERE received_at < ?", (now - self.retention_seconds,))
        # Live claims are bounded by the per-route rate limit over the retention window; the cap only
        # trims history rows, so trimming never re-opens a delivery to a replay.
        conn.execute(
            "DELETE FROM receipts WHERE claim = 0 AND id NOT IN"
            " (SELECT id FROM receipts WHERE claim = 0 ORDER BY id DESC LIMIT ?)", (MAX_HISTORY_ROWS,))


def read_history(path: Path, route: Optional[str] = None, limit: int = 20) -> list[dict[str, Any]]:
    """Newest-first receipts for ``hermes webhook history``; read-only, empty when nothing was recorded."""
    if not Path(path).exists():
        return []
    query = ("SELECT profile, route, delivery_id, event_type, outcome, reason, ref, received_at FROM receipts"
             + (" WHERE route = ?" if route else "") + " ORDER BY id DESC LIMIT ?")
    params = ((route,) if route else ()) + (max(int(limit), 1),)
    with closing(sqlite3.connect(f"{Path(path).resolve().as_uri()}?mode=ro", uri=True, timeout=5.0)) as conn:
        conn.row_factory = sqlite3.Row
        return [dict(row) for row in conn.execute(query, params)]


async def file_receipt(store: WebhookReceipts, receipt: dict, response) -> bool:
    """File the answer given to one authenticated delivery: settle its claim row, else add a history row.
    True when a claim was released (the caller drops its in-process copy too). History is best effort: a
    failed write is logged and never changes the answer already decided."""
    import asyncio
    from gateway.platforms.webhook import _WebhookDeliveryIdentity
    outcome, reason = outcome_for_response(response.status, getattr(response, "text", None) or "")
    reason, ref = receipt.get("reason") or reason, receipt.get("ref", "")
    try:
        if (row_id := receipt.get("claim_row")) is not None:
            keep_claim = outcome != FAILED or bool(receipt.get("outcome_unknown"))
            await asyncio.to_thread(store.settle, row_id, outcome, reason, ref, keep_claim=keep_claim)
            return not keep_claim
        identity = _WebhookDeliveryIdentity.from_parts(receipt["profile"], receipt["route"], receipt["delivery_id"])
        await asyncio.to_thread(store.record, identity, receipt.get("event_type", ""), outcome, reason, ref,
                                time.time())
    except sqlite3.Error:
        logger.warning("[webhook] Could not record receipt for delivery %s on route %s", receipt.get("delivery_id"),
                       receipt.get("route"), exc_info=True)
    return False


def outcome_for_response(status: int, body_text: str) -> tuple[str, str]:
    """``(outcome, reason)`` for the HTTP answer the adapter gave an authenticated delivery."""
    try:
        body = json.loads(body_text or "{}")
    except ValueError:
        body = {}
    if not isinstance(body, dict):
        body = {}
    state = body.get("status")
    if state == DUPLICATE:
        return DUPLICATE, ""
    if state == IGNORED:
        if body.get("reason"):
            return IGNORED, str(body["reason"])
        return IGNORED, f"event type '{body.get('event', '')}' is not in the route's events"
    if 200 <= status < 300:
        return ACCEPTED, state if state in {"coalesced", "delivered"} else ""
    return FAILED, str(body.get("error") or f"HTTP {status}")
