"""Opaque, account-scoped recommendation preferences and durable suppression sync.

Only identifiers derived from exact references leave the profile. This queue
does not upload candidate names, paths, private usage, or review rationale.
"""

from __future__ import annotations

import hashlib
import json
import time
import uuid
from datetime import datetime

from .client import WisdomError
from .mediation_store import MediationStore


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

    def flush(self, org: str, *, limit: int = 10) -> None:
        user = self.identity(org)
        for _ in range(min(max(limit, 0), 10)):
            now, token = self.clock(), uuid.uuid4().hex
            with self.store.transaction() as db:
                MediationStore._check_org(db, org)
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
            return {"available": True, "muted": mute.muted, "suppressed": suppressed}
        except Exception as exc:
            return {
                "available": False,
                "muted": True,
                "suppressed": {},
                "error": type(exc).__name__,
            }
