"""Opaque, account-scoped recommendation preferences and durable suppression sync.

Only identifiers derived from exact references leave the profile. This queue
does not upload candidate names, paths, private usage, or review rationale.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
import uuid
from datetime import datetime, timezone

from .client import AgentLedPolicyResponse, WisdomError
from .mediation_store import MediationStore

logger = logging.getLogger(__name__)


def suppression_key(reference: dict) -> str:
    if reference.get("kind") == "candidate":
        content_hash = reference.get("content_hash")
        if not isinstance(content_hash, str) or not content_hash:
            raise ValueError("Candidate suppression requires an exact content hash")
        identity = ["candidate", content_hash]
    elif reference.get("kind") == "skill":
        skill_id, version = reference.get("skill_id"), reference.get("version")
        if (
            not isinstance(skill_id, str)
            or not skill_id
            or type(version) is not int
            or version < 1
        ):
            raise ValueError(
                "Published suppression requires an immutable skill version"
            )
        identity = ["skill", skill_id, version]
    else:
        raise ValueError("Only candidate and skill recommendations can be suppressed")
    return (
        "sha256:"
        + hashlib.sha256(
            json.dumps(
                ["wisdom-suppression-v1", *identity], separators=(",", ":")
            ).encode()
        ).hexdigest()
    )


def _timestamp(value: str) -> float:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise WisdomError("Preference expiry must include a timezone")
    return parsed.timestamp()


class WisdomPreferences:
    def __init__(self, service, *, clock=time.time):
        self.service = service
        self.store = service.store
        self.clock = clock

    def identity(self, org: str) -> str:
        MediationStore(self.store)._require_org(org)
        client = self.service.client
        owner = client.identity.get("owner")
        if (
            client.display_org_id != org
            or not isinstance(owner, str)
            or not owner
            or owner == "unknown"
        ):
            raise WisdomError(
                "Recommendation preferences require an authenticated organization member"
            )
        return owner

    def native_mute_command(self, choice: str = "status") -> dict:
        """Explicit command entrypoint, not available to automatic assessment tools."""
        self.service.require_setup()
        org = self.store.active_org_id()
        choices = {
            "1d": "1_day",
            "1w": "1_week",
            "30d": "30_days",
            "forever": "forever",
            "off": None,
        }
        if choice != "status":
            if choice not in choices:
                raise ValueError("Use mute status, 1d, 1w, 30d, forever, or off")
            self.request_mute(org, choices[choice])
        self.flush_mute(org)
        try:
            user = self.identity(org)
            current = self.service.client.recommendation_mute()
            if current.org_id != org or self.identity(org) != user:
                raise WisdomError("Preference identity changed")
            remote = current.model_dump(mode="json")
        except Exception:
            remote = None
        return {
            "organization_id": org,
            "gateway_available": remote is not None,
            "mute": remote,
            "sync": self.mute_status(org),
        }

    def stage_suppression(
        self, db, *, org: str, user: str, reference: dict, days: int = 30
    ) -> dict:
        """Called in the native consent transaction; no network under SQLite lock."""
        MediationStore._check_org(db, org)
        if type(days) is not int or not 1 <= days <= 365:
            raise ValueError("Invalid suppression duration")
        key, now = suppression_key(reference), self.clock()
        db.execute(
            """INSERT INTO wisdom_preference_outbox
          (organization_id,user_id,key,suppress_until,created_at,available_at)
          VALUES(?,?,?,?,?,?) ON CONFLICT(organization_id,user_id,key) DO UPDATE SET
          suppress_until=excluded.suppress_until,created_at=excluded.created_at,
          available_at=excluded.available_at,state='pending',attempts=0,
          lease_token=NULL,lease_until=NULL,last_error=NULL
          WHERE wisdom_preference_outbox.suppress_until<=excluded.created_at""",
            (org, user, key, now + days * 86400, now, now),
        )
        # Another explicit native click may retry a terminal sync failure;
        # automatic polling never resets the bounded attempt budget.
        db.execute(
            """UPDATE wisdom_preference_outbox SET state='pending',attempts=0,
          available_at=?,last_error=NULL WHERE organization_id=? AND user_id=?
          AND key=? AND state='failed'""",
            (now, org, user, key),
        )
        row = db.execute(
            "SELECT suppress_until,state FROM wisdom_preference_outbox WHERE organization_id=? AND user_id=? AND key=?",
            (org, user, key),
        ).fetchone()
        return {
            "suppression_key": key,
            "suppressed_until": row[0],
            "preference_sync": row[1],
        }

    def review_suppression_days(self, org: str) -> int:
        """Snapshot verified policy for a later offline Not Now click."""
        try:
            user = self.identity(org)
            policy = AgentLedPolicyResponse.model_validate(
                self.service.client.agent_led_policy()
            )
            if policy.org_id != org or self.identity(org) != user:
                raise WisdomError("Recommendation policy identity changed")
            return policy.not_now_suppression_days
        except Exception as exc:
            # A policy outage must not prevent inspection or local deferral.
            # The Gateway replaces this legacy fallback on successful sync.
            logger.debug("Review suppression policy unavailable (%s)", type(exc).__name__)
            return 30

    def request_mute(self, org: str, duration: str | None) -> dict:
        """Native user action only; fresh authority, then durable local intent."""
        self.service.require_setup()
        user = self.identity(org)
        current = self.service.client.recommendation_mute()
        if current.org_id != org or self.identity(org) != user:
            raise WisdomError("Preference identity changed")
        if "revision" not in current.model_fields_set:
            raise WisdomError("Gateway must be updated before shared mute is available")
        return self.stage_mute(org, duration, expected_revision=current.revision)

    def stage_mute(
        self, org: str, duration: str | None, *, expected_revision: int
    ) -> dict:
        if duration not in {None, "1_day", "1_week", "30_days", "forever"}:
            raise ValueError("Unsupported mute duration")
        if (
            type(expected_revision) is not int
            or not 0 <= expected_revision < 2_147_483_647
        ):
            raise ValueError("Invalid mute revision")
        user, now = self.identity(org), self.clock()
        with self.store.transaction() as db:
            MediationStore._check_org(db, org)
            self._stage_mute(
                db, org, user, duration, expected_revision, uuid.uuid4().hex, now
            )
        return self.mute_status(org)

    @staticmethod
    def _stage_mute(db, org, user, duration, expected_revision, mutation_id, now):
        db.execute(
            """INSERT INTO wisdom_mute_outbox
                (organization_id,user_id,mutation_id,expected_revision,duration,
                 requested_at,expires_at,available_at) VALUES(?,?,?,?,?,?,?,?)
                ON CONFLICT(organization_id,user_id) DO UPDATE SET
                mutation_id=excluded.mutation_id,expected_revision=excluded.expected_revision,
                duration=excluded.duration,requested_at=excluded.requested_at,
                expires_at=excluded.expires_at,available_at=excluded.available_at,
                state='pending',attempts=0,lease_token=NULL,lease_until=NULL,last_error=NULL""",
            (
                org,
                user,
                mutation_id,
                expected_revision,
                duration,
                now,
                now + 86400,
                now,
            ),
        )

    def prepare_mute_control(self, org: str) -> dict:
        """Read authoritative state before binding a private native choice menu."""
        self.service.require_setup()
        user = self.identity(org)
        current = self.service.client.recommendation_mute()
        if current.org_id != org or self.identity(org) != user:
            raise WisdomError("Preference identity changed")
        if "revision" not in current.model_fields_set:
            raise WisdomError("Gateway must be updated before shared mute is available")
        now, control_id = self.clock(), uuid.uuid4().hex
        with self.store.transaction() as db:
            MediationStore._check_org(db, org)
            db.execute("DELETE FROM wisdom_mute_control WHERE expires_at<=?", (now,))
            count = db.execute(
                "SELECT count(*) FROM wisdom_mute_control WHERE organization_id=? AND user_id=?",
                (org, user),
            ).fetchone()[0]
            if count >= 128:
                raise WisdomError(
                    "Too many open notification controls. Try again shortly."
                )
            previous = db.execute(
                "SELECT mutation_id FROM wisdom_mute_outbox WHERE organization_id=? AND user_id=?",
                (org, user),
            ).fetchone()
            db.execute(
                """INSERT INTO wisdom_mute_control
                (id,organization_id,user_id,expected_revision,prior_mutation_id,expires_at)
                VALUES(?,?,?,?,?,?)""",
                (
                    control_id,
                    org,
                    user,
                    current.revision,
                    previous[0] if previous else None,
                    now + 600,
                ),
            )
        return {
            "id": control_id,
            "expires_at": now + 600,
            "mute": current.model_dump(mode="json"),
        }

    def choose_mute_control(
        self, org: str, control_id: str, duration: str | None
    ) -> dict:
        """Commit one choice per menu; repeats never extend expiry or rebase it."""
        self.service.require_setup()
        if duration not in {None, "1_day", "1_week", "30_days", "forever"}:
            raise ValueError("Unsupported mute duration")
        user, now = self.identity(org), self.clock()
        with self.store.transaction() as db:
            MediationStore._check_org(db, org)
            control = db.execute(
                "SELECT * FROM wisdom_mute_control WHERE id=? AND organization_id=? AND user_id=?",
                (control_id, org, user),
            ).fetchone()
            if control is None or control["expires_at"] <= now:
                raise ValueError(
                    "This notification control expired. Open /wisdom mute again."
                )
            previous = db.execute(
                "SELECT mutation_id FROM wisdom_mute_outbox WHERE organization_id=? AND user_id=?",
                (org, user),
            ).fetchone()
            previous_id = previous[0] if previous else None
            if control["selected_at"] is not None:
                if control["duration"] != duration or previous_id != control_id:
                    raise ValueError(
                        "This notification choice was superseded. Open /wisdom mute again."
                    )
            else:
                if previous_id != control["prior_mutation_id"]:
                    raise ValueError(
                        "Your notification preference changed. Open /wisdom mute again."
                    )
                self._stage_mute(
                    db,
                    org,
                    user,
                    duration,
                    control["expected_revision"],
                    control_id,
                    now,
                )
                db.execute(
                    "UPDATE wisdom_mute_control SET selected_at=?,duration=? WHERE id=?",
                    (now, duration, control_id),
                )
        return self.mute_status(org)

    def mute_status(self, org: str) -> dict | None:
        user = self.identity(org)
        with self.store.transaction() as db:
            MediationStore._check_org(db, org)
            row = db.execute(
                "SELECT * FROM wisdom_mute_outbox WHERE organization_id=? AND user_id=?",
                (org, user),
            ).fetchone()
        if row is None:
            return None
        days = {"1_day": 1, "1_week": 7, "30_days": 30}.get(row["duration"])
        return {
            "mutation_id": row["mutation_id"],
            "requested_duration": row["duration"],
            "requested_until": row["requested_at"] + days * 86400 if days else None,
            "preference_sync": row["state"],
            "request_expires_at": row["expires_at"],
        }

    def flush_mute(self, org: str) -> None:
        user, now, token = self.identity(org), self.clock(), uuid.uuid4().hex
        with self.store.transaction() as db:
            MediationStore._check_org(db, org)
            db.execute(
                """UPDATE wisdom_mute_outbox SET state='expired',lease_token=NULL,lease_until=NULL
                WHERE organization_id=? AND user_id=? AND expires_at<=? AND state IN ('pending','syncing')""",
                (org, user, now),
            )
            db.execute(
                """UPDATE wisdom_mute_outbox SET state='failed',lease_token=NULL,lease_until=NULL
                WHERE organization_id=? AND user_id=? AND state='syncing' AND lease_until<=? AND attempts>=3""",
                (org, user, now),
            )
            row = db.execute(
                """SELECT * FROM wisdom_mute_outbox WHERE organization_id=? AND user_id=?
                AND expires_at>? AND available_at<=? AND attempts<3
                AND (state='pending' OR (state='syncing' AND lease_until<=?))""",
                (org, user, now, now, now),
            ).fetchone()
            if row is None:
                return
            db.execute(
                """UPDATE wisdom_mute_outbox SET state='syncing',attempts=attempts+1,lease_token=?,lease_until=?
                WHERE organization_id=? AND user_id=?""",
                (token, now + 60, org, user),
            )
        try:
            if self.identity(org) != user:
                raise WisdomError("Preference owner changed")
            requested_at = (
                datetime
                .fromtimestamp(row["requested_at"], timezone.utc)
                .isoformat(timespec="milliseconds")
                .replace("+00:00", "Z")
            )
            result = self.service.client.set_recommendation_mute(
                row["duration"],
                expected_revision=row["expected_revision"],
                mutation_id=row["mutation_id"],
                requested_at=requested_at,
            )
            if (
                result.org_id != org
                or result.mutation_id != row["mutation_id"]
                or result.revision != row["expected_revision"] + 1
                or result.duration != row["duration"]
            ):
                raise WisdomError("Gateway returned mismatched mute acknowledgement")
            days = {"1_day": 1, "1_week": 7, "30_days": 30}.get(row["duration"])
            expected_until = _timestamp(requested_at) + days * 86400 if days else None
            actual_until = (
                _timestamp(result.muted_until) if result.muted_until else None
            )
            forever = row["duration"] == "forever"
            if (
                actual_until != expected_until
                or result.forever != forever
                or result.muted
                != (
                    forever
                    or (expected_until is not None and expected_until > self.clock())
                )
            ):
                raise WisdomError("Gateway returned inconsistent mute state")
            state, error = "synced", None
        except Exception as exc:
            status = getattr(exc, "status", None)
            state = (
                "conflict"
                if status == 409
                else "expired"
                if status == 410
                else "failed"
                if row["attempts"] + 1 >= 3
                else "pending"
            )
            error = type(exc).__name__
        if self.identity(org) != user:
            raise WisdomError("Preference owner changed")
        with self.store.transaction() as db:
            MediationStore._check_org(db, org)
            db.execute(
                """UPDATE wisdom_mute_outbox SET state=?,last_error=?,available_at=?,lease_token=NULL,lease_until=NULL
                WHERE organization_id=? AND user_id=? AND mutation_id=? AND lease_token=?""",
                (
                    state,
                    error,
                    now + 60 * 2 ** row["attempts"],
                    org,
                    user,
                    row["mutation_id"],
                    token,
                ),
            )

    def flush(self, org: str, *, limit: int = 10) -> None:
        user = self.identity(org)
        self.flush_mute(org)
        for _ in range(min(max(limit, 0), 10)):
            now, token = self.clock(), uuid.uuid4().hex
            with self.store.transaction() as db:
                MediationStore._check_org(db, org)
                db.execute(
                    """UPDATE wisdom_preference_outbox SET state='expired',lease_token=NULL,lease_until=NULL
                    WHERE organization_id=? AND user_id=? AND suppress_until<=? AND state IN ('pending','syncing')""",
                    (org, user, now),
                )
                db.execute(
                    """UPDATE wisdom_preference_outbox SET state='failed',lease_token=NULL,lease_until=NULL
                    WHERE organization_id=? AND user_id=? AND state='syncing' AND lease_until<=? AND attempts>=3""",
                    (org, user, now),
                )
                row = db.execute(
                    """SELECT * FROM wisdom_preference_outbox
                  WHERE organization_id=? AND user_id=? AND suppress_until>?
                  AND attempts<3 AND available_at<=?
                  AND (state='pending' OR (state='syncing' AND lease_until<=?))
                  ORDER BY created_at LIMIT 1""",
                    (org, user, now, now, now),
                ).fetchone()
                if row is None:
                    return
                db.execute(
                    """UPDATE wisdom_preference_outbox SET state='syncing',
                  attempts=attempts+1,lease_token=?,lease_until=?
                  WHERE organization_id=? AND user_id=? AND key=?""",
                    (token, now + 60, org, user, row["key"]),
                )
            try:
                if self.identity(org) != user:
                    raise WisdomError("Preference owner changed")
                result = self.service.client.suppress_recommendation(row["key"])
                until = _timestamp(result.suppress_until)
                if (
                    result.key != row["key"]
                    or result.org_id != org
                    or not now < until <= now + 366 * 86400
                ):
                    raise WisdomError("Gateway returned invalid suppression")
                state, error = "synced", None
            except Exception as exc:
                until = row["suppress_until"]
                state = "failed" if row["attempts"] + 1 >= 3 else "pending"
                error = type(exc).__name__
            if self.identity(org) != user:
                raise WisdomError("Preference owner changed")
            with self.store.transaction() as db:
                MediationStore._check_org(db, org)
                db.execute(
                    """UPDATE wisdom_preference_outbox SET state=?,
                  suppress_until=?,last_error=?,available_at=?,lease_token=NULL,lease_until=NULL
                  WHERE organization_id=? AND user_id=? AND key=? AND lease_token=?""",
                    (
                        state,
                        until,
                        error,
                        now + 60 * 2 ** row["attempts"],
                        org,
                        user,
                        row["key"],
                        token,
                    ),
                )

    def check(self, org: str, references: list[dict]) -> dict:
        """Fresh authority before proactive work. Network failure is not consent to notify."""
        keys = list(dict.fromkeys(suppression_key(ref) for ref in references))
        if len(keys) > 100:
            raise ValueError("At most 100 preference keys per check")
        try:
            user = self.identity(org)
            self.flush(org)
            mute = self.service.client.recommendation_mute()
            rows = self.service.client.recommendation_suppressions(keys)
            if mute.org_id != org or self.identity(org) != user:
                raise WisdomError("Preference identity changed")
            now = self.clock()
            suppressed = {}
            for row in rows:
                until = _timestamp(row.suppress_until)
                if row.key not in keys or until <= now or until > now + 366 * 86400:
                    raise WisdomError("Invalid suppression response")
                suppressed[row.key] = until
            with self.store.transaction() as db:
                MediationStore._check_org(db, org)
                for row in db.execute(
                    """SELECT key,suppress_until FROM wisdom_preference_outbox
                  WHERE organization_id=? AND user_id=? AND suppress_until>?
                  AND state!='synced'""",
                    (org, user, now),
                ):
                    if row["key"] in keys:
                        suppressed[row["key"]] = max(
                            suppressed.get(row["key"], 0), row["suppress_until"]
                        )
            pending_mute = self.mute_status(org)
            locally_muted = bool(
                pending_mute
                and pending_mute["preference_sync"] in {"pending", "syncing", "failed"}
                and pending_mute["requested_duration"] is not None
                and pending_mute["request_expires_at"] > now
                and (
                    pending_mute["requested_until"] is None
                    or pending_mute["requested_until"] > now
                )
            )
            return {
                "available": True,
                "muted": mute.muted or locally_muted,
                "suppressed": suppressed,
                "mute_sync": pending_mute,
            }
        except Exception as exc:
            return {
                "available": False,
                "muted": True,
                "suppressed": {},
                "error": type(exc).__name__,
            }
