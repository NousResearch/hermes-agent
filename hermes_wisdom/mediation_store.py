"""Durable, profile-local ownership of Wisdom advice and consent.

Delivery receipts and unread cursors remain in WisdomStore. These records
describe work, not whether the user has read or approved a notification.
"""

from __future__ import annotations

import json
import time
import uuid
from contextlib import nullcontext
from typing import TYPE_CHECKING, Any

from .delivery import DeliveryReceipt

if TYPE_CHECKING:
    import sqlite3

    from .store import WisdomStore

MAX_ATTEMPTS = 3
LEASE_SECONDS = 180
SESSION_TTL = 120
BATCH_SIZE = 8


def create_schema(db: sqlite3.Connection) -> None:
    # execute, not executescript: preserve the caller's migration transaction.
    for statement in (
        """CREATE TABLE IF NOT EXISTS wisdom_agent_session (
          organization_id TEXT NOT NULL, session_key TEXT NOT NULL,
          session_id TEXT NOT NULL, platform TEXT NOT NULL, actor_id TEXT NOT NULL,
          last_activity REAL NOT NULL, alive_until REAL NOT NULL,
          available INTEGER NOT NULL, address_json TEXT NOT NULL DEFAULT '{}',
          PRIMARY KEY(organization_id,session_key))""",
        """CREATE TABLE IF NOT EXISTS wisdom_assessment (
          id TEXT PRIMARY KEY, organization_id TEXT NOT NULL,
          event_key TEXT NOT NULL, origin_session TEXT,
          reference_json TEXT NOT NULL, state TEXT NOT NULL DEFAULT 'pending',
          owner_session TEXT, lease_token TEXT, lease_until REAL,
          attempts INTEGER NOT NULL DEFAULT 0, available_at REAL NOT NULL,
          advice_json TEXT, last_error TEXT, delivered_at REAL,
          created_at REAL NOT NULL, updated_at REAL NOT NULL,
          UNIQUE(organization_id,event_key))""",
        """CREATE INDEX IF NOT EXISTS wisdom_assessment_queue
          ON wisdom_assessment(organization_id,state,available_at)""",
        """CREATE TABLE IF NOT EXISTS wisdom_delivery_receipt (
          assessment_id TEXT PRIMARY KEY REFERENCES wisdom_assessment(id),
          organization_id TEXT NOT NULL, owner_session TEXT NOT NULL,
          receipt_json TEXT NOT NULL, recorded_at REAL NOT NULL)""",
        """CREATE TABLE IF NOT EXISTS wisdom_agent_introduction (
          organization_id TEXT PRIMARY KEY, delivered_at REAL NOT NULL)""",
        """CREATE TABLE IF NOT EXISTS wisdom_mediation_poll (
          organization_id TEXT PRIMARY KEY, next_poll REAL NOT NULL)""",
        """CREATE TABLE IF NOT EXISTS wisdom_consent (
          id TEXT PRIMARY KEY, organization_id TEXT NOT NULL,
          assessment_id TEXT NOT NULL REFERENCES wisdom_assessment(id),
          owner_session TEXT NOT NULL, actor_id TEXT NOT NULL,
          platform TEXT NOT NULL, operation TEXT NOT NULL,
          plan_json TEXT NOT NULL, state TEXT NOT NULL DEFAULT 'pending',
          expires_at REAL NOT NULL, result_json TEXT,
          created_at REAL NOT NULL, updated_at REAL NOT NULL)""",
        """CREATE TABLE IF NOT EXISTS wisdom_consent_defer (
          interaction_id TEXT NOT NULL REFERENCES wisdom_consent(id),
          surface TEXT NOT NULL, deferred_at REAL NOT NULL,
          PRIMARY KEY(interaction_id,surface))""",
        """CREATE TABLE IF NOT EXISTS wisdom_preference_outbox (
          organization_id TEXT NOT NULL, user_id TEXT NOT NULL, key TEXT NOT NULL,
          suppress_until REAL NOT NULL, created_at REAL NOT NULL,
          state TEXT NOT NULL DEFAULT 'pending', attempts INTEGER NOT NULL DEFAULT 0,
          available_at REAL NOT NULL, lease_token TEXT, lease_until REAL,
          last_error TEXT,
          PRIMARY KEY(organization_id,user_id,key))""",
        """CREATE TABLE IF NOT EXISTS wisdom_mute_outbox (
          organization_id TEXT NOT NULL, user_id TEXT NOT NULL,
          mutation_id TEXT NOT NULL, expected_revision INTEGER NOT NULL,
          duration TEXT, requested_at REAL NOT NULL, expires_at REAL NOT NULL,
          state TEXT NOT NULL DEFAULT 'pending', attempts INTEGER NOT NULL DEFAULT 0,
          available_at REAL NOT NULL, lease_token TEXT, lease_until REAL,
          last_error TEXT, PRIMARY KEY(organization_id,user_id))""",
        """CREATE TABLE IF NOT EXISTS wisdom_mute_control (
          id TEXT PRIMARY KEY, organization_id TEXT NOT NULL, user_id TEXT NOT NULL,
          expected_revision INTEGER NOT NULL, prior_mutation_id TEXT,
          expires_at REAL NOT NULL, selected_at REAL, duration TEXT)""",
        """CREATE TABLE IF NOT EXISTS wisdom_consent_outcome (
          interaction_id TEXT PRIMARY KEY REFERENCES wisdom_consent(id),
          organization_id TEXT NOT NULL, owner_session TEXT NOT NULL,
          result_json TEXT NOT NULL, delivered_at REAL)""",
    ):
        db.execute(statement)
    from .delivery_outbox import create_schema as create_delivery_schema

    create_delivery_schema(db)
    # Support development databases created before renewable consent IDs.
    definition = db.execute(
        "SELECT sql FROM sqlite_master WHERE name='wisdom_consent'"
    ).fetchone()[0]
    if "UNIQUE(assessment_id,operation)" in definition:
        db.execute("PRAGMA defer_foreign_keys=ON")
        db.execute(
            definition.replace("wisdom_consent", "wisdom_consent_new", 1).replace(
                ",\n          UNIQUE(assessment_id,operation)", ""
            )
        )
        db.execute("INSERT INTO wisdom_consent_new SELECT * FROM wisdom_consent")
        db.execute("DROP TABLE wisdom_consent")
        db.execute("ALTER TABLE wisdom_consent_new RENAME TO wisdom_consent")
    db.execute("""CREATE UNIQUE INDEX IF NOT EXISTS wisdom_consent_one_active
               ON wisdom_consent(assessment_id) WHERE state IN ('pending','applying')""")


def _decode(row: Any) -> dict[str, Any]:
    value = dict(row)
    for key in ("reference", "advice", "plan", "result"):
        raw = value.pop(f"{key}_json", None)
        if raw is not None:
            value[key] = json.loads(raw)
    return value


class MediationStore:
    def __init__(self, store: WisdomStore, *, clock=time.time) -> None:
        self.store = store
        self.clock = clock

    def _require_org(self, org: str) -> None:
        if not org or self.store.active_org_id() != org:
            raise ValueError("Wisdom organization is no longer active")

    @staticmethod
    def _check_org(db: sqlite3.Connection, org: str) -> None:
        row = db.execute(
            "SELECT verified_org_id FROM installation_identity WHERE singleton=1"
        ).fetchone()
        if row is None or row[0] != org:
            raise ValueError("Wisdom organization is no longer active")

    def register_session(
        self,
        org: str,
        *,
        session_key: str,
        session_id: str,
        platform: str,
        actor_id: str,
        private: bool,
        available: bool,
        user_activity: bool = False,
        address: dict[str, str] | None = None,
        activity_at: float | None = None,
    ) -> None:
        self._require_org(org)
        if not session_key or not actor_id or not private:
            return
        now = self.clock()
        with self.store.transaction() as db:
            self._check_org(db, org)
            db.execute(
                """INSERT INTO wisdom_agent_session VALUES(?,?,?,?,?,?,?,?,?)
                ON CONFLICT(organization_id,session_key) DO UPDATE SET
                  session_id=excluded.session_id, platform=excluded.platform,
                  actor_id=excluded.actor_id, alive_until=excluded.alive_until,
                  available=excluded.available,address_json=excluded.address_json,
                  last_activity=CASE WHEN ? THEN excluded.last_activity
                    ELSE wisdom_agent_session.last_activity END""",
                (
                    org,
                    session_key,
                    session_id,
                    platform,
                    actor_id,
                    min(activity_at or now, now) if user_activity else 0,
                    now + SESSION_TTL,
                    int(available),
                    json.dumps(address or {}),
                    int(user_activity),
                ),
            )

    def claim_refresh(self, org: str) -> bool:
        now = self.clock()
        with self.store.transaction() as db:
            self._check_org(db, org)
            return bool(
                db.execute(
                    """INSERT INTO wisdom_mediation_poll VALUES(?,?)
                ON CONFLICT(organization_id) DO UPDATE SET next_poll=excluded.next_poll
                WHERE wisdom_mediation_poll.next_poll<=?""",
                    (org, now + 60, now),
                ).rowcount
            )

    def retire_candidates(self, org: str, current_ids: set[str]) -> None:
        with self.store.transaction() as db:
            self._check_org(db, org)
            rows = db.execute(
                """SELECT id,event_key FROM wisdom_assessment WHERE organization_id=?
                AND event_key LIKE 'candidate:%' AND state NOT IN ('delivered','retired')""",
                (org,),
            ).fetchall()
            for row in rows:
                if row["event_key"].removeprefix("candidate:") not in current_ids:
                    db.execute(
                        "UPDATE wisdom_assessment SET state='retired',lease_token=NULL,lease_until=NULL WHERE id=?",
                        (row["id"],),
                    )
                    db.execute(
                        "UPDATE wisdom_consent SET state='stale' WHERE assessment_id=? AND state='pending'",
                        (row["id"],),
                    )

    def enqueue(
        self,
        org: str,
        event_key: str,
        reference: dict[str, Any],
        *,
        origin_session: str | None = None,
        _db: sqlite3.Connection | None = None,
    ) -> str:
        now = self.clock()
        with self.store.transaction() if _db is None else nullcontext(_db) as db:
            self._check_org(db, org)
            db.execute(
                """INSERT INTO wisdom_assessment
                (id,organization_id,event_key,origin_session,reference_json,
                 available_at,created_at,updated_at) VALUES(?,?,?,?,?,?,?,?)
                ON CONFLICT(organization_id,event_key) DO NOTHING""",
                (
                    uuid.uuid4().hex,
                    org,
                    event_key,
                    origin_session,
                    json.dumps(reference, sort_keys=True),
                    now,
                    now,
                    now,
                ),
            )
            return db.execute(
                "SELECT id FROM wisdom_assessment WHERE organization_id=? AND event_key=?",
                (org, event_key),
            ).fetchone()[0]

    def reconcile_feed(self, org: str, event_id: str, reference: dict[str, Any]) -> str:
        """Repair old recommendation/notice classification without another delivery."""
        with self.store.transaction() as db:
            identity = self.enqueue(org, f"feed:{event_id}", reference, _db=db)
            row = db.execute(
                "SELECT reference_json,state FROM wisdom_assessment WHERE id=?",
                (identity,),
            ).fetchone()
            if json.loads(row["reference_json"]).get("kind") == reference["kind"]:
                return identity
            # A send in flight is uncertain, not permission to send again.
            terminal = row["state"] in {
                "delivering",
                "delivered",
                "delivery_uncertain",
                "retired",
            }
            db.execute(
                """UPDATE wisdom_assessment SET reference_json=?,updated_at=?
                WHERE id=?""",
                (json.dumps(reference, sort_keys=True), self.clock(), identity),
            )
            if not terminal:
                db.execute(
                    """UPDATE wisdom_assessment SET state='pending',advice_json=NULL,
                    lease_token=NULL,lease_until=NULL,attempts=0,available_at=? WHERE id=?""",
                    (self.clock(), identity),
                )
            db.execute(
                "UPDATE wisdom_consent SET state='stale' WHERE assessment_id=? AND state='pending'",
                (identity,),
            )
            return identity

    def retire(self, org: str, job: dict[str, Any]) -> bool:
        """Fence terminal arrival invalidation against a replacement owner."""
        now = self.clock()
        with self.store.transaction() as db:
            self._check_org(db, org)
            changed = bool(
                db.execute(
                    """UPDATE wisdom_assessment SET state='retired',lease_token=NULL,
                lease_until=NULL,updated_at=? WHERE id=? AND organization_id=?
                AND lease_token=? AND lease_until>? AND state IN ('assessing','ready','fallback')""",
                    (now, job["id"], org, job["lease_token"], now),
                ).rowcount
            )
            if changed:
                db.execute(
                    "UPDATE wisdom_consent SET state='stale' WHERE assessment_id=? AND state='pending'",
                    (job["id"],),
                )
            return changed

    def check_claim(self, db, org: str, assessment_id: str, token: str) -> None:
        from .client import WisdomConflict

        self._check_org(db, org)
        if not db.execute(
            """SELECT 1 FROM wisdom_assessment WHERE id=? AND organization_id=?
            AND state='assessing' AND lease_token=? AND lease_until>?""",
            (assessment_id, org, token, self.clock()),
        ).fetchone():
            raise WisdomConflict("Wisdom assessment ownership changed")

    def claim(self, org: str, session_key: str) -> list[dict[str, Any]]:
        """Elect one eligible session, then fence every row with a fresh token."""
        self._require_org(org)
        now = self.clock()
        with self.store.transaction() as db:
            self._check_org(db, org)
            session = db.execute(
                """SELECT * FROM wisdom_agent_session WHERE organization_id=?
                AND session_key=? AND alive_until>? AND available=1
                AND last_activity>0""",
                (org, session_key, now),
            ).fetchone()
            if session is None:
                return []
            if db.execute(
                """SELECT 1 FROM wisdom_assessment WHERE organization_id=?
                AND lease_until>? AND lease_token IS NOT NULL
                AND state IN ('assessing','ready','fallback','delivering') LIMIT 1""",
                (org, now),
            ).fetchone():
                return []
            recent = db.execute(
                """SELECT session_key FROM wisdom_agent_session
                WHERE organization_id=? AND alive_until>? AND last_activity>0
                ORDER BY last_activity DESC,session_key LIMIT 1""",
                (org, now),
            ).fetchone()[0]
            # A send with an unknown outcome must not be blindly replayed.
            db.execute(
                """UPDATE wisdom_assessment SET state='delivery_uncertain',
                lease_token=NULL,lease_until=NULL WHERE organization_id=?
                AND state='delivering' AND lease_until<=?""",
                (org, now),
            )
            rows = db.execute(
                """SELECT * FROM wisdom_assessment WHERE organization_id=?
                AND available_at<=? AND (state IN ('pending','ready','fallback')
                  OR (state='assessing' AND lease_until<=?))
                AND ((state='ready' AND (owner_session=? OR
                  (origin_session IS NULL AND ?=? AND NOT EXISTS (
                    SELECT 1 FROM wisdom_agent_session owner WHERE owner.organization_id=wisdom_assessment.organization_id
                    AND owner.session_key=wisdom_assessment.owner_session AND owner.alive_until>?)))) OR (state!='ready' AND
                  ((origin_session IS NULL AND ?=?) OR origin_session IN (?,?))))
                ORDER BY created_at,id LIMIT ?""",
                (
                    org,
                    now,
                    now,
                    session_key,
                    session_key,
                    recent,
                    now,
                    session_key,
                    recent,
                    session_key,
                    session["session_id"],
                    BATCH_SIZE,
                ),
            ).fetchall()
            claimed = []
            for row in rows:
                if row["owner_session"] and row["owner_session"] != session_key:
                    db.execute(
                        "UPDATE wisdom_consent SET state='stale' WHERE assessment_id=? AND state='pending'",
                        (row["id"],),
                    )
                token = uuid.uuid4().hex
                state = row["state"]
                needs_assessment = state in {"pending", "assessing"}
                if needs_assessment and row["attempts"] >= MAX_ATTEMPTS:
                    state = "fallback"
                    needs_assessment = False
                else:
                    state = "assessing" if needs_assessment else state
                db.execute(
                    """UPDATE wisdom_assessment SET state=?,owner_session=?,
                    lease_token=?,lease_until=?,attempts=attempts+?,updated_at=?
                    WHERE id=?""",
                    (
                        state,
                        session_key,
                        token,
                        now + LEASE_SECONDS,
                        int(needs_assessment),
                        now,
                        row["id"],
                    ),
                )
                claimed.append(
                    _decode(
                        db.execute(
                            "SELECT * FROM wisdom_assessment WHERE id=?", (row["id"],)
                        ).fetchone()
                    )
                )
            return claimed

    def renew(self, org: str, assessment_id: str, token: str) -> bool:
        now = self.clock()
        with self.store.transaction() as db:
            self._check_org(db, org)
            return bool(
                db.execute(
                    """UPDATE wisdom_assessment SET lease_until=? WHERE id=?
                AND organization_id=? AND lease_token=? AND lease_until>?
                AND state IN ('assessing','ready','fallback')""",
                    (now + LEASE_SECONDS, assessment_id, org, token, now),
                ).rowcount
            )

    def defer_for_preferences(self, org: str, job: dict, until: float) -> None:
        """Release without spending a model attempt; retain advice and unread state."""
        now = self.clock()
        with self.store.transaction() as db:
            self._check_org(db, org)
            db.execute(
                """UPDATE wisdom_assessment SET
              state=CASE WHEN state='assessing' THEN 'pending' ELSE state END,
              attempts=MAX(0,attempts-CASE WHEN state='assessing' THEN 1 ELSE 0 END),
              available_at=?,lease_token=NULL,lease_until=NULL,updated_at=?
              WHERE id=? AND organization_id=? AND lease_token=? AND lease_until>?
              AND state IN ('assessing','ready','fallback')""",
                (max(now + 60, until), now, job["id"], org, job["lease_token"], now),
            )

    def save_advice(
        self, org: str, assessment_id: str, token: str, advice: dict[str, Any]
    ) -> bool:
        now = self.clock()
        with self.store.transaction() as db:
            self._check_org(db, org)
            return bool(
                db.execute(
                    """UPDATE wisdom_assessment SET advice_json=?,state='ready',
                updated_at=? WHERE id=? AND organization_id=? AND lease_token=?
                AND lease_until>? AND state='assessing'""",
                    (json.dumps(advice), now, assessment_id, org, token, now),
                ).rowcount
            )

    def fail(self, org: str, assessment_id: str, token: str, reason: str) -> bool:
        now = self.clock()
        with self.store.transaction() as db:
            self._check_org(db, org)
            return bool(
                db.execute(
                    """UPDATE wisdom_assessment SET
                state=CASE WHEN attempts>=? THEN 'fallback' ELSE 'pending' END,
                last_error=?,available_at=?,lease_token=NULL,lease_until=NULL,
                updated_at=? WHERE id=? AND organization_id=? AND lease_token=?
                AND lease_until>? AND state='assessing'""",
                    (
                        MAX_ATTEMPTS,
                        reason[:100],
                        now + 60,
                        now,
                        assessment_id,
                        org,
                        token,
                        now,
                    ),
                ).rowcount
            )

    def begin_delivery(
        self, org: str, assessment_id: str, token: str, *, request_id: str | None = None
    ) -> bool:
        now = self.clock()
        with self.store.transaction() as db:
            self._check_org(db, org)
            if request_id is not None:
                owned = db.execute(
                    """SELECT 1 FROM wisdom_assessment WHERE id=? AND organization_id=?
                  AND lease_token=? AND lease_until>? AND state IN ('ready','fallback')""",
                    (assessment_id, org, token, now),
                ).fetchone()
                if not owned:
                    return False
                from .delivery_outbox import begin_reserved

                if not begin_reserved(db, org, assessment_id, token, request_id, now):
                    return False
            return bool(
                db.execute(
                    """UPDATE wisdom_assessment SET state='delivering',updated_at=?
                WHERE id=? AND organization_id=? AND lease_token=? AND lease_until>?
                AND state IN ('ready','fallback')""",
                    (now, assessment_id, org, token, now),
                ).rowcount
            )

    def cancel_delivery(self, org: str, assessment_id: str, token: str) -> None:
        """Only before invoking the transport: preserve advice and release a known non-send."""
        with self.store.transaction() as db:
            self._check_org(db, org)
            from .delivery_outbox import cancel_reserved

            now = self.clock()
            cancel_reserved(db, org, assessment_id, token, now)
            db.execute(
                """UPDATE wisdom_assessment SET state=CASE WHEN advice_json IS NULL THEN 'fallback' ELSE 'ready' END,
              lease_token=NULL,lease_until=NULL,available_at=?,updated_at=?
              WHERE id=? AND organization_id=? AND lease_token=? AND state='delivering'""",
                (now + 60, now, assessment_id, org, token),
            )

    def delivery_ready(
        self, org: str, assessment_id: str, token: str, *, user_id: str | None = None
    ) -> bool:
        with self.store.transaction() as db:
            self._check_org(db, org)
            now = self.clock()
            owned = db.execute(
                """SELECT s.platform,s.address_json FROM wisdom_assessment a
              JOIN wisdom_agent_session s ON s.organization_id=a.organization_id AND s.session_key=a.owner_session
              WHERE a.organization_id=? AND a.id=? AND a.lease_token=? AND a.lease_until>?
              AND a.state='delivering'""",
                (org, assessment_id, token, now),
            ).fetchone()
            remote = db.execute(
                "SELECT * FROM wisdom_remote_delivery WHERE organization_id=? AND assessment_id=? AND local_token=?",
                (org, assessment_id, token),
            ).fetchone()
            return bool(
                owned
                and (
                    remote is None
                    or (
                        remote["state"] == "sending"
                        and remote["lease_until"] > now
                        and (user_id is None or remote["user_id"] == user_id)
                        and remote["platform"] == owned["platform"]
                        and json.loads(remote["address_json"])
                        == json.loads(owned["address_json"])
                    )
                )
            )

    def complete_delivery(
        self,
        org: str,
        assessment_id: str,
        token: str,
        *,
        receipt: DeliveryReceipt,
        introduced: bool = False,
    ) -> bool:
        now = self.clock()
        with self.store.transaction() as db:
            self._check_org(db, org)
            if not isinstance(receipt, DeliveryReceipt):
                raise ValueError("a validated delivery receipt is required")
            from .delivery_outbox import stage_outcome

            staged = stage_outcome(
                db,
                org,
                assessment_id,
                token,
                now,
                receipt=receipt,
                introduced=introduced,
            )
            owner = db.execute(
                """SELECT a.owner_session,s.platform,s.address_json
                FROM wisdom_assessment a JOIN wisdom_agent_session s
                  ON s.organization_id=a.organization_id AND s.session_key=a.owner_session
                WHERE a.id=? AND a.organization_id=? AND a.lease_token=?
                AND a.state='delivering' AND a.lease_until>?""",
                (assessment_id, org, token, now),
            ).fetchone()
            if owner is None:
                return False
            address = json.loads(owner["address_json"])
            if (
                receipt.platform != owner["platform"]
                or receipt.destination != address.get("chat_id", "")
                or receipt.thread_id != address.get("thread_id", "")
                or receipt.scope_id != address.get("scope_id", "")
            ):
                if staged:
                    # The session moved during the send. Recover the receipt using
                    # the immutable reservation address, not the new destination.
                    return False
                raise ValueError("delivery receipt does not match the owning session")
            changed = bool(
                db.execute(
                    """UPDATE wisdom_assessment SET state='delivered',delivered_at=?,
                updated_at=?,lease_token=NULL,lease_until=NULL
                WHERE id=? AND organization_id=? AND lease_token=?
                AND state='delivering' AND lease_until>?""",
                    (now, now, assessment_id, org, token, now),
                ).rowcount
            )
            if changed and introduced:
                db.execute(
                    "INSERT OR IGNORE INTO wisdom_agent_introduction VALUES(?,?)",
                    (org, now),
                )
            if changed:
                db.execute(
                    "INSERT INTO wisdom_delivery_receipt VALUES(?,?,?,?,?)",
                    (
                        assessment_id,
                        org,
                        owner["owner_session"],
                        receipt.model_dump_json(),
                        now,
                    ),
                )
                event_key = db.execute(
                    "SELECT event_key FROM wisdom_assessment WHERE id=?",
                    (assessment_id,),
                ).fetchone()[0]
                if event_key.startswith("outcome:"):
                    db.execute(
                        "UPDATE wisdom_consent_outcome SET delivered_at=? WHERE interaction_id=? AND organization_id=?",
                        (now, event_key.removeprefix("outcome:"), org),
                    )
            return changed

    def uncertain_delivery(self, org: str, assessment_id: str, token: str) -> bool:
        with self.store.transaction() as db:
            self._check_org(db, org)
            from .delivery_outbox import stage_outcome

            stage_outcome(db, org, assessment_id, token, self.clock())
            return bool(
                db.execute(
                    """UPDATE wisdom_assessment SET state='delivery_uncertain',
                lease_token=NULL,lease_until=NULL,updated_at=?
                WHERE id=? AND organization_id=? AND lease_token=? AND state='delivering'""",
                    (self.clock(), assessment_id, org, token),
                ).rowcount
            )

    def introduced(self, org: str) -> bool:
        with self.store.transaction() as db:
            self._check_org(db, org)
            return (
                db.execute(
                    "SELECT 1 FROM wisdom_agent_introduction WHERE organization_id=?",
                    (org,),
                ).fetchone()
                is not None
            )

    def assessments(self, org: str) -> list[dict[str, Any]]:
        with self.store.transaction() as db:
            self._check_org(db, org)
            return [
                _decode(row)
                for row in db.execute(
                    "SELECT * FROM wisdom_assessment WHERE organization_id=? ORDER BY created_at,id",
                    (org,),
                ).fetchall()
            ]
