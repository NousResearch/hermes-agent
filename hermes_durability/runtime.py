"""DurableRuntime: the embeddable facade.

    rt = DurableRuntime("agent.db", adapters={"slack": SlackAdapter(...)})
    rt.recover()                      # crash-safe resume + drain pending sends
    with rt.transaction(session_id) as txn:
        txn.record(USER_MESSAGE, {...})
        txn.record(TOOL_CALL_INVOKED, {...})
        txn.enqueue_outbound("slack", {"text": reply})
    # commit fsyncs journal + outbox atomically; worker delivers with
    # guardrails + idempotency keys.

Startup integrity runs in __init__ BEFORE the background worker starts:
chain verification (torn-tail repair), orphaned-outbox cleanup, and
inflight-claim reset all complete before any thread can pick up a row, so
the worker can never deliver a row whose journal context is about to be
truncated. ``recover()`` then only drains and reports.
"""

from __future__ import annotations

import time
from typing import Optional

from hermes_durability.guardrail import Guardrail
from hermes_durability.journal import (Journal, JournalTransaction,
                                       OUTBOX_DELIVERED, TXN_COMMIT, _txn)
from hermes_durability.outbox import Adapter, OutboxWorker, RetryPolicy


class DurableRuntime:
    def __init__(self, db_path: str, adapters: Optional[dict[str, Adapter]] = None,
                 guardrail: Optional[Guardrail] = None,
                 retry: Optional[RetryPolicy] = None,
                 start_worker: bool = True):
        self.journal = Journal(db_path)
        self.guardrail = guardrail or Guardrail(audit_sink=self._audit_sink)
        if self.guardrail.audit_sink is None:
            self.guardrail.audit_sink = self._audit_sink
        self.worker = OutboxWorker(self.journal, adapters or {},
                                   guardrail=self.guardrail, retry=retry)
        self._startup_report = self._verify_and_clean()
        self._worker_started = False
        if start_worker:
            self.worker.start()
            self._worker_started = True

    # -- audit -------------------------------------------------------------
    def _audit_sink(self, payload_id: str, session_id: str, action: str,
                    policy_id: str, envelope_hash: bytes) -> None:
        conn = self.journal._conn
        with self.journal._lock:
            with _txn(conn):
                conn.execute(
                    "INSERT INTO audit_log (payload_id, session_id, action,"
                    " policy_id, envelope_hash, created_at)"
                    " VALUES (?,?,?,?,?,?)",
                    (payload_id, session_id, action, policy_id, envelope_hash,
                     time.time()))

    # -- API ---------------------------------------------------------------
    def transaction(self, session_id: str) -> JournalTransaction:
        txn = self.journal.begin(session_id)
        original_commit = txn.commit

        def commit_and_signal() -> list[int]:
            seqs = original_commit()
            self.worker.signal()
            return seqs

        txn.commit = commit_and_signal  # type: ignore[method-assign]
        return txn

    def _verify_and_clean(self) -> dict:
        """Chain verify/repair + outbox consistency, BEFORE any delivery.

        Orphan cleanup compares outbox rows against surviving OutboxEnqueued
        journal records by exact id (enqueue and journal are one SQLite
        transaction, so an orphan can only exist after a torn-tail
        truncation removed its journal context — delivering it would be a
        ghost send of a transaction that officially never committed).
        """
        ok, bad_seq = self.journal.verify_chain(repair=True)
        conn = self.journal._conn
        with self.journal._lock:
            orphaned = []
            if not ok:
                enqueued = self.journal.enqueued_outbox_ids()
                rows = conn.execute(
                    "SELECT outbox_id FROM outbox"
                    " WHERE status IN ('pending', 'inflight')").fetchall()
                orphaned = [oid for (oid,) in rows if oid not in enqueued]
                if orphaned:
                    with _txn(conn):
                        for oid in orphaned:
                            conn.execute(
                                "DELETE FROM outbox WHERE outbox_id = ?",
                                (oid,))
        reset = self.worker.reset_inflight()
        return {"chain_ok": ok, "truncated_at": bad_seq if not ok else None,
                "orphaned_outbox_discarded": len(orphaned),
                "inflight_reset": reset}

    def recover(self) -> dict:
        """Drain committed-but-undelivered sends and report startup state.

        Integrity repair already ran in __init__; idempotency keys make the
        redelivery of possibly-already-sent rows safe.
        """
        conn = self.journal._conn
        with self.journal._lock:
            pending = conn.execute(
                "SELECT COUNT(*) FROM outbox WHERE status = 'pending'"
            ).fetchone()[0]
        delivered = self.worker.drain_once() if pending else 0
        report = dict(self._startup_report)
        report.update({"pending_outbox": pending,
                       "delivered_on_recovery": delivered})
        return report

    def replay_state(self, session_id: str) -> dict:
        """Rebuild session state: latest complete snapshot + committed
        transactions after it. Uncommitted transactions are discarded."""
        snap = self.journal.latest_snapshot(session_id)
        state: dict = snap[1] if snap else {"messages": [], "delivered": []}
        base_seq = snap[0] if snap else 0
        committed = self.journal.committed_transactions(session_id)
        for rec in self.journal.records(session_id, after_seq=base_seq):
            if rec.record_type == TXN_COMMIT:
                continue
            if rec.transaction_id not in committed:
                continue  # uncommitted -> discard
            if rec.record_type == OUTBOX_DELIVERED:
                state.setdefault("delivered", []).append(rec.payload)
            elif rec.record_type not in ("CompactionSnapshot",
                                         "CompactionComplete"):
                state.setdefault("messages", []).append(
                    {"type": rec.record_type, **rec.payload})
        return state

    def close(self) -> None:
        if self._worker_started:
            self.worker.stop()
        self.journal.close()
