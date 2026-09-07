"""Durable client outcome sync. This module never sends advice or applies work."""

from __future__ import annotations

import hashlib
import json
import logging
import uuid

from .client import WisdomConflict, WisdomError
from .client_outcome import ClientOperationOutcome, ClientOperationResponse
from .mediation_store import MediationStore
from .preferences import WisdomPreferences, suppression_key

logger = logging.getLogger(__name__)
MAX_ATTEMPTS = 5
SYNC_LEASE = 90


def create_schema(db):
    db.execute("""CREATE TABLE IF NOT EXISTS wisdom_operation_outbox (
      interaction_id TEXT PRIMARY KEY REFERENCES wisdom_consent(id),
      organization_id TEXT NOT NULL, user_id TEXT NOT NULL, request_id TEXT NOT NULL,
      outcome_json TEXT NOT NULL, state TEXT NOT NULL DEFAULT 'pending',
      attempts INTEGER NOT NULL DEFAULT 0, available_at REAL NOT NULL,
      sync_token TEXT, sync_until REAL, last_error TEXT)""")
    db.execute("""CREATE INDEX IF NOT EXISTS wisdom_operation_sync
      ON wisdom_operation_outbox(organization_id,user_id,state,available_at)""")


def _delivery(db, value):
    org = value["organization_id"]
    assessment_id = value["assessment_id"]
    direct = db.execute(
        "SELECT * FROM wisdom_remote_delivery WHERE organization_id=? AND assessment_id=?",
        (org, assessment_id),
    ).fetchone()
    if direct is not None:
        return direct
    # User-requested packaging does not reserve a second proactive delivery.
    # Follow only the exact backend-created job/parent consent link, never names.
    parent = db.execute(
        """SELECT c.* FROM wisdom_assessment a JOIN wisdom_consent c
        ON a.event_key='share-package:'||c.id AND a.organization_id=c.organization_id
        WHERE a.id=? AND a.organization_id=? AND c.operation='share'
        AND c.state='completed' AND json_extract(c.result_json,'$.assessment_id')=a.id""",
        (assessment_id, org),
    ).fetchone()
    if parent is None or value["operation"] != "publish":
        return None
    plan = json.loads(parent["plan_json"])
    if any(
        plan.get(key) != value["plan"].get(key) for key in ("event_id", "source_hash")
    ):
        return None
    return db.execute(
        "SELECT * FROM wisdom_remote_delivery WHERE organization_id=? AND assessment_id=?",
        (org, parent["assessment_id"]),
    ).fetchone()


def stage(db, value, state, result, now):
    """Commit with native consent completion. No auth/network or nested transaction."""
    delivery = _delivery(db, value)
    if delivery is None or delivery["outcome"] == "not_sent":
        return False  # Manual/fixed-mode actions do not manufacture delivery evidence.
    reference = json.loads(delivery["reference_json"])
    plan = value["plan"]
    if value["operation"] in {"share", "publish"}:
        if not plan.get("source_hash") or reference != {
            "kind": "candidate",
            "key": suppression_key({
                "kind": "candidate",
                "content_hash": plan["source_hash"],
            }),
        }:
            return False
    elif (
        reference.get("kind"),
        reference.get("skill_id"),
        reference.get("version"),
    ) != ("skill", plan.get("skill_id"), plan.get("version")):
        return False
    operation = value["operation"]
    if operation == "share" and state == "completed":
        if result.get("packaging_state") != "queued":
            return False
        state = "queued"
    identity = json.dumps(
        [
            "wisdom-operation-v1",
            value["organization_id"],
            delivery["user_id"],
            value["id"],
        ],
        separators=(",", ":"),
    )
    report = ClientOperationOutcome(
        request_id=delivery["request_id"],
        operation_key="sha256:" + hashlib.sha256(identity.encode()).hexdigest(),
        operation=operation,
        state=state,
    )
    payload = report.model_dump_json()
    existing = db.execute(
        "SELECT outcome_json FROM wisdom_operation_outbox WHERE interaction_id=?",
        (value["id"],),
    ).fetchone()
    if existing is not None:
        if existing["outcome_json"] != payload:
            raise WisdomConflict("The operation outcome is already recorded")
        return False
    db.execute(
        """INSERT INTO wisdom_operation_outbox(interaction_id,organization_id,user_id,
        request_id,outcome_json,available_at) VALUES(?,?,?,?,?,?)""",
        (
            value["id"],
            value["organization_id"],
            delivery["user_id"],
            delivery["request_id"],
            payload,
            now,
        ),
    )
    return True


class OperationOutbox:
    def __init__(self, service, *, clock):
        self.service, self.store, self.clock = service, service.store, clock

    def identity(self, org):
        return WisdomPreferences(self.service, clock=self.clock).identity(org)

    def flush(self, org):
        user = self.identity(org)
        for _ in range(8):
            now = self.clock()
            with self.store.transaction() as db:
                MediationStore._check_org(db, org)
                db.execute(
                    """UPDATE wisdom_operation_outbox SET state='failed',last_error='outcome_retry_exhausted',
                    sync_token=NULL,sync_until=NULL WHERE organization_id=? AND user_id=? AND state='pending'
                    AND attempts>=? AND (sync_until IS NULL OR sync_until<=?)""",
                    (org, user, MAX_ATTEMPTS, now),
                )
                row = db.execute(
                    """SELECT o.*,d.event_id FROM wisdom_operation_outbox o
                    JOIN wisdom_remote_delivery d ON d.request_id=o.request_id
                    AND d.organization_id=o.organization_id AND d.user_id=o.user_id
                    WHERE o.organization_id=? AND o.user_id=? AND o.state='pending'
                    AND o.attempts<? AND o.available_at<=?
                    AND (o.sync_until IS NULL OR o.sync_until<=?)
                    AND d.state='settled' AND d.outcome='acknowledged'
                    AND d.event_id IS NOT NULL AND d.receipt_json IS NOT NULL
                    ORDER BY o.available_at,o.interaction_id LIMIT 1""",
                    (org, user, MAX_ATTEMPTS, now, now),
                ).fetchone()
                if row is None:
                    return
                row = dict(row)
                token = str(uuid.uuid4())
                db.execute(
                    """UPDATE wisdom_operation_outbox SET sync_token=?,sync_until=?,attempts=attempts+1
                    WHERE interaction_id=?""",
                    (token, now + SYNC_LEASE, row["interaction_id"]),
                )
            state, failure = "settled", None
            try:
                if self.identity(org) != user:
                    return
                report = ClientOperationOutcome.model_validate_json(row["outcome_json"])
                response = self.service.client.report_operation_outcome(
                    row["event_id"], report.model_dump(mode="json")
                )
                if (
                    not isinstance(response, ClientOperationResponse)
                    or (response.org_id, response.recipient_user_id) != (org, user)
                    or response.event_id != row["event_id"]
                    or response.outcome != report
                ):
                    raise WisdomError("Invalid operation acknowledgement")
            except Exception as exc:
                failure = (
                    "outcome_conflict"
                    if getattr(exc, "status", None) == 409
                    else "outcome_unavailable"
                )
                state = (
                    "failed"
                    if failure == "outcome_conflict"
                    or row["attempts"] + 1 >= MAX_ATTEMPTS
                    else "pending"
                )
                logger.warning("Wisdom outcome sync deferred (%s)", failure)
            if self.identity(org) != user:
                return
            with self.store.transaction() as db:
                MediationStore._check_org(db, org)
                db.execute(
                    """UPDATE wisdom_operation_outbox SET state=?,last_error=?,available_at=?,
                    sync_token=NULL,sync_until=NULL WHERE interaction_id=? AND sync_token=?""",
                    (
                        state,
                        failure,
                        self.clock() + min(3600, 60 * 2 ** row["attempts"]),
                        row["interaction_id"],
                        token,
                    ),
                )
