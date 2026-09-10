"""Private receipt/report recovery. No message send or operation execution."""

from __future__ import annotations

import time

from .client import WisdomError
from .delivery_outbox import DeliveryOutbox, MAX_SYNC_ATTEMPTS
from .mediation_store import MediationStore
from .operation_outbox import MAX_ATTEMPTS, OperationOutbox
from .preferences import WisdomPreferences


class WisdomSyncRecovery:
    def __init__(self, service, *, clock=time.time):
        self.service, self.store, self.clock = service, service.store, clock

    def _identity(self):
        org = self.store.active_org_id()
        return org, WisdomPreferences(self.service).identity(org)

    def _check_identity(self, org, user):
        if self._identity() != (org, user):
            raise WisdomError("The active account changed; reopen sync status")

    def _rows(self, db, org, user):
        MediationStore._check_org(db, org)
        deliveries = db.execute(
            """SELECT request_id,state,outcome,attempts,sync_until,last_error
            FROM wisdom_remote_delivery WHERE organization_id=? AND user_id=?
            AND (state!='settled' OR outcome='uncertain')""",
            (org, user),
        ).fetchall()
        operations = db.execute(
            """SELECT o.interaction_id,o.report_state,o.state,o.attempts,o.sync_until,o.last_error,
            (d.state='settled' AND d.outcome='acknowledged'
             AND d.event_id IS NOT NULL AND d.receipt_json IS NOT NULL) AS receipt_ready
            FROM wisdom_operation_outbox o LEFT JOIN wisdom_remote_delivery d
            ON d.request_id=o.request_id AND d.organization_id=o.organization_id
            AND d.user_id=o.user_id
            WHERE o.organization_id=? AND o.user_id=? AND o.state!='settled'""",
            (org, user),
        ).fetchall()
        return deliveries, operations

    @staticmethod
    def _state(row, kind, now):
        if row["sync_until"] is not None and row["sync_until"] > now:
            return "syncing"
        if kind == "delivery":
            if row["state"] == "settled":
                return "uncertain"
            if row["last_error"] == "settlement_conflict":
                return "conflict"
            if row["state"] == "outcome" and row["attempts"] >= MAX_SYNC_ATTEMPTS:
                return "retryable"
        else:
            if row["last_error"] == "outcome_conflict":
                return "conflict"
            if not row["receipt_ready"]:
                return "waiting_for_receipt"
            if row["attempts"] >= MAX_ATTEMPTS or (
                row["state"] == "failed" and row["last_error"] == "outcome_unavailable"
            ):
                return "retryable"
        return "pending"

    def status(self):
        org, user = self._identity()
        now = self.clock()
        groups = {}
        with self.store.transaction() as db:
            for kind, rows in zip(("delivery", "operation"), self._rows(db, org, user)):
                counts = dict.fromkeys(
                    (
                        "pending",
                        "syncing",
                        "retryable",
                        "conflict",
                        "uncertain",
                        "waiting_for_receipt",
                    ),
                    0,
                )
                for row in rows:
                    counts[self._state(row, kind, now)] += 1
                groups[kind] = counts
        # No private references, transport identifiers, raw errors or payloads.
        self._check_identity(org, user)
        return {**groups, "can_retry": any(g["retryable"] for g in groups.values())}

    def retry(self):
        org, user = self._identity()
        now = self.clock()
        with self.store.transaction() as db:
            deliveries, operations = self._rows(db, org, user)
            for row in deliveries:
                if self._state(row, "delivery", now) == "retryable":
                    db.execute(
                        """UPDATE wisdom_remote_delivery SET attempts=0,available_at=?,
                        sync_token=NULL,sync_until=NULL,last_error=NULL WHERE request_id=?""",
                        (now, row["request_id"]),
                    )
            for row in operations:
                if self._state(row, "operation", now) == "retryable":
                    db.execute(
                        """UPDATE wisdom_operation_outbox SET state='pending',attempts=0,available_at=?,
                        sync_token=NULL,sync_until=NULL,last_error=NULL WHERE interaction_id=? AND report_state=?""",
                        (now, row["interaction_id"], row["report_state"]),
                    )
        # Reconcile saved transport acceptance before reporting completed work.
        # Existing leases and immutable request keys make concurrent retries safe.
        self._check_identity(org, user)
        DeliveryOutbox(self.service, clock=self.clock).flush(org)
        self._check_identity(org, user)
        OperationOutbox(self.service, clock=self.clock).flush(org)
        self._check_identity(org, user)
        return self.status()
