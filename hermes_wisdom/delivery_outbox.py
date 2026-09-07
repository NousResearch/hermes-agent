"""Local send reservations and receipt recovery; never sends a message itself."""

from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime

from .client import WisdomError
from .client_delivery import ClientDeliveryResponse
from .delivery import DeliveryReceipt
from .mediation_store import MediationStore
from .preferences import WisdomPreferences, suppression_key

logger = logging.getLogger(__name__)
SEND_MARGIN = 10
SYNC_LEASE = 90
MAX_SYNC_ATTEMPTS = 5


def create_schema(db):
    db.execute("""CREATE TABLE IF NOT EXISTS wisdom_remote_delivery (
      request_id TEXT PRIMARY KEY, organization_id TEXT NOT NULL, user_id TEXT NOT NULL,
      assessment_id TEXT NOT NULL REFERENCES wisdom_assessment(id), reference_json TEXT NOT NULL,
      local_token TEXT NOT NULL, owner_session TEXT NOT NULL, platform TEXT NOT NULL,
      address_json TEXT NOT NULL, state TEXT NOT NULL DEFAULT 'claiming',
      event_id TEXT, lease_until REAL, outcome TEXT, receipt_json TEXT,
      introduced INTEGER NOT NULL DEFAULT 0, attempts INTEGER NOT NULL DEFAULT 0,
      available_at REAL NOT NULL, sync_token TEXT, sync_until REAL, last_error TEXT,
      UNIQUE(organization_id,user_id,assessment_id))""")


def begin_reserved(db, org, assessment_id, token, request_id, now):
    return bool(
        db.execute(
            """UPDATE wisdom_remote_delivery SET state='sending'
      WHERE organization_id=? AND assessment_id=? AND request_id=? AND local_token=?
      AND state='claimed' AND lease_until>? AND sync_token IS NULL""",
            (org, assessment_id, request_id, token, now + SEND_MARGIN),
        ).rowcount
    )


def cancel_reserved(db, org, assessment_id, token, now):
    db.execute(
        """UPDATE wisdom_remote_delivery SET state='outcome',outcome='not_sent',available_at=?
      WHERE organization_id=? AND assessment_id=? AND local_token=? AND state='sending'""",
        (now, org, assessment_id, token),
    )


def stage_outcome(
    db, org, assessment_id, token, now, *, receipt=None, introduced=False
):
    """Commit with the local receipt/uncertainty transaction, including late receipts."""
    row = db.execute(
        """SELECT * FROM wisdom_remote_delivery WHERE organization_id=?
      AND assessment_id=? AND local_token=? AND (state IN ('sending','outcome')
      OR (state='settled' AND outcome='uncertain'))""",
        (org, assessment_id, token),
    ).fetchone()
    if row is None or row["outcome"] in {"acknowledged", "not_sent"}:
        return False
    if receipt is not None:
        address = json.loads(row["address_json"])
        if not isinstance(receipt, DeliveryReceipt) or (
            receipt.platform != row["platform"]
            or receipt.destination != address.get("chat_id", "")
            or receipt.thread_id != address.get("thread_id", "")
            or receipt.scope_id != address.get("scope_id", "")
        ):
            raise ValueError("Receipt does not match the reserved sender")
    db.execute(
        """UPDATE wisdom_remote_delivery SET state='outcome',outcome=?,receipt_json=?,
      introduced=?,available_at=?,attempts=0,sync_token=NULL,sync_until=NULL,last_error=NULL
      WHERE request_id=?""",
        (
            "acknowledged" if receipt is not None else "uncertain",
            receipt.model_dump_json() if receipt is not None else None,
            int(introduced),
            now,
            row["request_id"],
        ),
    )
    return receipt is not None


class DeliveryOutbox:
    def __init__(self, service, *, clock):
        self.service = service
        self.store = service.store
        self.clock = clock

    def identity(self, org):
        return WisdomPreferences(self.service, clock=self.clock).identity(org)

    def reserve(self, org, job):
        """Persist intent before networking. Return only a currently usable reservation."""
        user = self.identity(org)
        source = job["reference"]
        if source["kind"] == "candidate":
            reference = {"kind": "candidate", "key": suppression_key(source)}
        else:
            installed = self.store.installation(source["skill_id"])
            reference = {
                "kind": "skill",
                "skill_id": source["skill_id"],
                "version": source["version"],
                "event_type": "update_available"
                if installed and installed["state"] == "active"
                else "teammate_published",
            }
        now = self.clock()
        with self.store.transaction() as db:
            MediationStore._check_org(db, org)
            owner = db.execute(
                """SELECT a.owner_session,s.platform,s.address_json FROM wisdom_assessment a
              JOIN wisdom_agent_session s ON s.organization_id=a.organization_id AND s.session_key=a.owner_session
              WHERE a.id=? AND a.organization_id=? AND a.lease_token=? AND a.lease_until>?
              AND a.state IN ('ready','fallback')""",
                (job["id"], org, job["lease_token"], now),
            ).fetchone()
            if owner is None:
                return None
            row = db.execute(
                """SELECT * FROM wisdom_remote_delivery
              WHERE organization_id=? AND user_id=? AND assessment_id=?""",
                (org, user, job["id"]),
            ).fetchone()
            if row is not None:
                if row["state"] == "settled" and row["outcome"] == "not_sent":
                    db.execute(
                        "DELETE FROM wisdom_remote_delivery WHERE request_id=?",
                        (row["request_id"],),
                    )
                    row = None
                elif row["state"] == "settled" and row["outcome"] == "acknowledged":
                    self._passive(db, org, job)
                    return None
                elif (
                    row["state"] not in {"claiming", "claimed"}
                    or row["local_token"] != job["lease_token"]
                    or row["sync_token"]
                ):
                    return None
            if row is None:
                request_id = str(uuid.uuid4())
                db.execute(
                    """INSERT INTO wisdom_remote_delivery(request_id,organization_id,user_id,assessment_id,
                  reference_json,local_token,owner_session,platform,address_json,available_at)
                  VALUES(?,?,?,?,?,?,?,?,?,?)""",
                    (
                        request_id,
                        org,
                        user,
                        job["id"],
                        json.dumps(reference),
                        job["lease_token"],
                        owner["owner_session"],
                        owner["platform"],
                        owner["address_json"],
                        now,
                    ),
                )
            else:
                request_id = row["request_id"]
                reference = json.loads(row["reference_json"])
        result = self.service.client.claim_notification_delivery(request_id, reference)
        if not isinstance(result, ClientDeliveryResponse) or self.identity(org) != user:
            raise WisdomError("Invalid notification reservation response")
        expiry = (
            datetime.fromisoformat(
                result.lease_until.replace("Z", "+00:00")
            ).timestamp()
            if result.lease_until
            else None
        )
        with self.store.transaction() as db:
            MediationStore._check_org(db, org)
            updated = db.execute(
                """UPDATE wisdom_remote_delivery SET event_id=?,lease_until=?,state=?,outcome=?
              WHERE request_id=? AND local_token=? AND state IN ('claiming','claimed') AND sync_token IS NULL""",
                (
                    result.event_id,
                    expiry,
                    "claimed" if result.state == "claimed" else "outcome",
                    None if result.state == "claimed" else "not_sent",
                    request_id,
                    job["lease_token"],
                ),
            ).rowcount
            if not updated:
                return None
            if result.state == "acknowledged":
                db.execute(
                    "UPDATE wisdom_remote_delivery SET state='settled',outcome='acknowledged' WHERE request_id=?",
                    (request_id,),
                )
                self._passive(db, org, job)
                return None
        return (
            request_id
            if result.state == "claimed" and expiry > self.clock() + SEND_MARGIN
            else None
        )

    @staticmethod
    def _passive(db, org, job):
        # Another client has already delivered. Keep advice and consent, not a second alert.
        db.execute(
            """UPDATE wisdom_assessment SET state='delivered',lease_token=NULL,lease_until=NULL
          WHERE organization_id=? AND id=? AND lease_token=? AND state IN ('ready','fallback')""",
            (org, job["id"], job["lease_token"]),
        )

    def not_sent(self, org, job):
        with self.store.transaction() as db:
            MediationStore._check_org(db, org)
            db.execute(
                """UPDATE wisdom_remote_delivery SET state='outcome',outcome='not_sent',available_at=?
              WHERE organization_id=? AND assessment_id=? AND local_token=? AND state IN ('claiming','claimed')""",
                (self.clock(), org, job["id"], job["lease_token"]),
            )

    def flush(self, org):
        """At most eight settlements per poll. Network retries never execute sends."""
        user = self.identity(org)
        now = self.clock()
        for _ in range(8):
            with self.store.transaction() as db:
                MediationStore._check_org(db, org)
                row = db.execute(
                    """SELECT r.* FROM wisdom_remote_delivery r
                  LEFT JOIN wisdom_assessment a ON a.id=r.assessment_id
                  WHERE r.organization_id=? AND r.user_id=? AND r.state!='settled'
                  AND r.attempts<? AND r.available_at<=? AND (r.sync_until IS NULL OR r.sync_until<=?)
                  AND (r.state='outcome' OR a.lease_token IS NULL OR a.lease_token!=r.local_token OR a.lease_until<=?)
                  ORDER BY r.available_at,r.request_id LIMIT 1""",
                    (org, user, MAX_SYNC_ATTEMPTS, now, now, now),
                ).fetchone()
                if row is None:
                    return
                row = dict(row)
                if row["state"] != "outcome":
                    row["outcome"] = (
                        "uncertain" if row["state"] == "sending" else "not_sent"
                    )
                token = str(uuid.uuid4())
                db.execute(
                    """UPDATE wisdom_remote_delivery SET state='outcome',outcome=?,sync_token=?,sync_until=?,attempts=attempts+1
                  WHERE request_id=?""",
                    (row["outcome"], token, now + SYNC_LEASE, row["request_id"]),
                )
            result = None
            failure = None
            settled_outcome = row["outcome"]
            try:
                if self.identity(org) != user:
                    return
                reference = json.loads(row["reference_json"])
                if row["event_id"] is None:
                    claimed = self.service.client.claim_notification_delivery(
                        row["request_id"], reference
                    )
                    row["event_id"] = claimed.event_id
                    if claimed.state == "acknowledged":
                        result = claimed
                if result is None:
                    if self.identity(org) != user:
                        return
                    receipt = (
                        DeliveryReceipt.model_validate_json(row["receipt_json"])
                        if row["receipt_json"]
                        else None
                    )
                    result = self.service.client.settle_notification_delivery(
                        row["event_id"],
                        row["request_id"],
                        reference,
                        outcome=row["outcome"],
                        receipt=receipt,
                    )
                if (
                    not isinstance(result, ClientDeliveryResponse)
                    or self.identity(org) != user
                ):
                    raise WisdomError("Invalid notification settlement response")
                settled_outcome = result.state
            except Exception as exc:
                # A definite local non-send can be retired even when another client
                # owns the remote event. A new attempt must still obtain a new claim.
                if getattr(exc, "status", None) != 409 or row["outcome"] != "not_sent":
                    failure = (
                        "settlement_conflict"
                        if getattr(exc, "status", None) == 409
                        else "settlement_unavailable"
                    )
                    logger.warning("Wisdom receipt sync deferred (%s)", failure)
            if self.identity(org) != user:
                return
            with self.store.transaction() as db:
                MediationStore._check_org(db, org)
                updated = db.execute(
                    """UPDATE wisdom_remote_delivery SET state=?,outcome=?,event_id=?,sync_token=NULL,sync_until=NULL,
                  available_at=?,last_error=? WHERE request_id=? AND sync_token=?""",
                    (
                        "settled" if failure is None else "outcome",
                        settled_outcome,
                        row["event_id"],
                        now + min(3600, 60 * 2 ** row["attempts"]),
                        failure,
                        row["request_id"],
                        token,
                    ),
                ).rowcount
                if (
                    updated
                    and failure is None
                    and settled_outcome == "acknowledged"
                    and row["receipt_json"]
                ):
                    self._recover_receipt(db, org, row, now)

    @staticmethod
    def _recover_receipt(db, org, row, now):
        changed = db.execute(
            """UPDATE wisdom_assessment SET state='delivered',delivered_at=?,lease_token=NULL,lease_until=NULL
          WHERE id=? AND organization_id=? AND owner_session=? AND
          ((state='delivery_uncertain' AND (lease_token IS NULL OR lease_token=?)) OR (state='delivering' AND lease_token=?))""",
            (
                now,
                row["assessment_id"],
                org,
                row["owner_session"],
                row["local_token"],
                row["local_token"],
            ),
        ).rowcount
        if changed:
            db.execute(
                "INSERT OR IGNORE INTO wisdom_delivery_receipt VALUES(?,?,?,?,?)",
                (
                    row["assessment_id"],
                    org,
                    row["owner_session"],
                    row["receipt_json"],
                    now,
                ),
            )
            if row["introduced"]:
                db.execute(
                    "INSERT OR IGNORE INTO wisdom_agent_introduction VALUES(?,?)",
                    (org, now),
                )
