"""Origin-profile-scoped storage for guided-routing policy + decision receipts.

Design (plans/2026-09-15_141016-guided-model-routing.md §3.C, §12 "Canonical
storage and privacy"): immutable policy revisions, routing decisions and
append-only outcome events live in the origin profile's own state, never a
new cross-profile daemon/queue. This module is a small dedicated SQLite store
(``model_routing.db`` beside ``kanban.db`` under the same Hermes home) rather
than another lifecycle queue — Kanban tasks hold only a receipt id pointer.

Nothing here calls an LLM or a provider. Pure persistence + validation of the
shapes ``agent.model_selection`` already defines.
"""
from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path
from typing import Optional

from hermes_cli.sqlite_util import open_db, transaction

from agent.model_selection_types import RoutingBlocked

_SCHEMA = """
CREATE TABLE IF NOT EXISTS policy_revisions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    policy_id TEXT NOT NULL,
    revision INTEGER NOT NULL,
    approval_ref TEXT NOT NULL,
    content_json TEXT NOT NULL,
    content_hash TEXT NOT NULL,
    created_at INTEGER NOT NULL,
    active INTEGER NOT NULL DEFAULT 0,
    UNIQUE(policy_id, revision)
);
CREATE TABLE IF NOT EXISTS routing_receipts (
    id TEXT PRIMARY KEY,
    execution_kind TEXT NOT NULL,
    execution_id TEXT NOT NULL,
    attempt_id TEXT NOT NULL,
    slot_id TEXT NOT NULL DEFAULT '',
    policy_id TEXT NOT NULL,
    policy_revision INTEGER NOT NULL,
    decision_json TEXT NOT NULL,
    created_at INTEGER NOT NULL,
    UNIQUE(execution_kind, execution_id, attempt_id, slot_id)
);
CREATE TABLE IF NOT EXISTS routing_outcomes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    receipt_id TEXT NOT NULL,
    seq INTEGER NOT NULL,
    kind TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    created_at INTEGER NOT NULL,
    UNIQUE(receipt_id, seq)
);
CREATE TABLE IF NOT EXISTS route_revocations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    policy_id TEXT NOT NULL,
    route_id TEXT,
    event_type TEXT NOT NULL DEFAULT 'revoke',
    reason TEXT NOT NULL,
    approval_ref TEXT NOT NULL,
    created_at INTEGER NOT NULL
);
"""


def _canonical_json(obj) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _content_hash(obj) -> str:
    import hashlib

    return hashlib.sha256(_canonical_json(obj).encode("utf-8")).hexdigest()


def _db_path(hermes_home) -> Path:
    return Path(hermes_home) / "model_routing.db"


def _connect(hermes_home) -> sqlite3.Connection:
    return open_db(_db_path(hermes_home), db_label="model_routing", initialize=lambda c: c.executescript(_SCHEMA))


def publish_policy(hermes_home, policy: dict, *, approval_ref: str) -> dict:
    """Publish an immutable policy revision. Never overwrites an existing
    ``(policy_id, revision)`` — a revision referenced by retained runs must
    stay available (design §12). Returns the stored record.
    """
    if not approval_ref or not str(approval_ref).strip():
        raise RoutingBlocked("schema_invalid", "approval_ref is required to publish a policy")
    for field in ("schema_version", "policy_id", "revision", "routes"):
        if field not in policy:
            raise RoutingBlocked("schema_invalid", f"policy missing field: {field}")
    content = dict(policy)
    content["approval_ref"] = approval_ref
    content_hash = _content_hash(content)
    now = int(time.time())
    with transaction(_connect(hermes_home)) as conn:
        existing = conn.execute(
            "SELECT content_hash FROM policy_revisions WHERE policy_id=? AND revision=?",
            (policy["policy_id"], policy["revision"]),
        ).fetchone()
        if existing is not None:
            if existing["content_hash"] != content_hash:
                raise RoutingBlocked(
                    "schema_invalid",
                    f"policy_id={policy['policy_id']} revision={policy['revision']} "
                    "already published with different content",
                )
        else:
            conn.execute(
                "INSERT INTO policy_revisions "
                "(policy_id, revision, approval_ref, content_json, content_hash, created_at, active) "
                "VALUES (?, ?, ?, ?, ?, ?, 0)",
                (policy["policy_id"], policy["revision"], approval_ref, _canonical_json(content),
                 content_hash, now),
            )
    return {"policy_id": policy["policy_id"], "revision": policy["revision"],
            "approval_ref": approval_ref, "content_hash": content_hash}


def activate_policy(hermes_home, policy_id: str, revision: int) -> None:
    """Mark exactly one revision of ``policy_id`` as the active one used at
    claim/start time. Publishing does not imply activation (§3.A: admission
    requires explicit approval, separate from CLI proposal)."""
    with transaction(_connect(hermes_home)) as conn:
        row = conn.execute(
            "SELECT id FROM policy_revisions WHERE policy_id=? AND revision=?",
            (policy_id, revision),
        ).fetchone()
        if row is None:
            raise RoutingBlocked("schema_invalid", f"no such policy revision: {policy_id}/{revision}")
        conn.execute("UPDATE policy_revisions SET active=0 WHERE policy_id=?", (policy_id,))
        conn.execute("UPDATE policy_revisions SET active=1 WHERE id=?", (row["id"],))


def get_active_policy(hermes_home, policy_id: str) -> Optional[dict]:
    with transaction(_connect(hermes_home)) as conn:
        row = conn.execute(
            "SELECT content_json FROM policy_revisions WHERE policy_id=? AND active=1",
            (policy_id,),
        ).fetchone()
    return json.loads(row["content_json"]) if row is not None else None


def list_policy_revisions(hermes_home, policy_id: str) -> list[dict]:
    with transaction(_connect(hermes_home)) as conn:
        rows = conn.execute(
            "SELECT revision, approval_ref, content_hash, created_at, active "
            "FROM policy_revisions WHERE policy_id=? ORDER BY revision",
            (policy_id,),
        ).fetchall()
    return [dict(r) for r in rows]


def persist_receipt(hermes_home, decision: dict) -> str:
    """Persist an immutable decision as a receipt keyed by
    ``(execution_kind, execution_id, attempt_id, slot_id)`` (design §12).
    Idempotent: re-persisting the identical key/decision returns the same id;
    a conflicting decision for an existing key is rejected (never silently
    overwritten — an active decision is never mutated, §4 step 6)."""
    req = decision["requirements"]
    slot_id = str(req.get("slot_id", ""))
    key = (req["execution_kind"], req["execution_id"], req["attempt_id"], slot_id)
    receipt_id = "rr_" + _content_hash({"key": key, "policy_id": decision["policy_id"],
                                         "revision": decision["policy_revision"]})[:24]
    decision_json = _canonical_json(decision)
    now = int(time.time())
    with transaction(_connect(hermes_home)) as conn:
        existing = conn.execute(
            "SELECT id, decision_json FROM routing_receipts WHERE "
            "execution_kind=? AND execution_id=? AND attempt_id=? AND slot_id=?",
            key,
        ).fetchone()
        if existing is not None:
            if existing["decision_json"] != decision_json:
                raise RoutingBlocked(
                    "stale_or_revoked_decision",
                    f"a different decision is already receipted for {key}",
                )
            return existing["id"]
        conn.execute(
            "INSERT INTO routing_receipts "
            "(id, execution_kind, execution_id, attempt_id, slot_id, policy_id, policy_revision, "
            " decision_json, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (receipt_id, *key, decision["policy_id"], decision["policy_revision"], decision_json, now),
        )
    return receipt_id


def get_receipt(hermes_home, receipt_id: str) -> Optional[dict]:
    with transaction(_connect(hermes_home)) as conn:
        row = conn.execute(
            "SELECT decision_json FROM routing_receipts WHERE id=?", (receipt_id,),
        ).fetchone()
    return json.loads(row["decision_json"]) if row is not None else None


def get_receipt_created_at(hermes_home, receipt_id: str) -> Optional[int]:
    """The unix timestamp this receipt was first persisted at.

    Used to distinguish a routine policy edit (which must never affect an
    already-receipted, in-flight attempt) from an explicit emergency
    revocation issued AFTER this attempt was authorized -- only a
    revocation whose ``created_at`` is at or after this timestamp is
    relevant to this receipt (design §12: revocation is a distinct,
    auditable act, never inferred from an ordinary republish).
    """
    with transaction(_connect(hermes_home)) as conn:
        row = conn.execute(
            "SELECT created_at FROM routing_receipts WHERE id=?", (receipt_id,),
        ).fetchone()
    return int(row["created_at"]) if row is not None else None


def _record_revocation_event(
    hermes_home, policy_id: str, *, route_id: Optional[str], event_type: str,
    reason: str, approval_ref: str,
) -> dict:
    if not approval_ref or not str(approval_ref).strip():
        raise RoutingBlocked("schema_invalid", f"approval_ref is required to {event_type} a route")
    if not reason or not str(reason).strip():
        raise RoutingBlocked("schema_invalid", f"reason is required to {event_type} a route")
    now = int(time.time())
    with transaction(_connect(hermes_home)) as conn:
        cur = conn.execute(
            "INSERT INTO route_revocations (policy_id, route_id, event_type, reason, approval_ref, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (policy_id, route_id, event_type, reason, approval_ref, now),
        )
        event_id = cur.lastrowid
    return {
        "id": event_id, "policy_id": policy_id, "route_id": route_id, "event_type": event_type,
        "reason": reason, "approval_ref": approval_ref, "created_at": now,
    }


def revoke_route(
    hermes_home, policy_id: str, *, route_id: Optional[str] = None,
    reason: str, approval_ref: str,
) -> dict:
    """Explicit, auditable EMERGENCY revocation/suspension (design §12).

    Distinct from ``activate_policy``: publishing/activating a new policy
    revision is an ordinary edit that only affects NEW attempts (§12
    "Routine policy edits affect new attempts, not active conversations").
    This function is the ONLY mechanism that blocks an already-receipted,
    in-flight attempt -- it never happens as a side effect of publish/
    activate. ``route_id=None`` revokes every route of ``policy_id``
    (whole-policy emergency suspension); a specific ``route_id`` revokes
    only that route, leaving receipts pinned to other routes of the same
    policy unaffected.

    This is DURABLE authorization state, not a timestamp-scoped event: once
    recorded, it blocks every receipt referencing this route/policy --
    already-receipted in-flight runs AND any future receipt persisted after
    the revocation -- until an explicit, separately-auditable
    ``readmit_route`` call records readmission. A routine ``publish_policy``/
    ``activate_policy`` can never clear it, and a new receipt minted after
    the revocation gets no fresh grace period merely by being new (the bug
    this closes: a receipt timestamped after the revocation is not itself
    proof of authorization).

    Never substitutes another route: a revoked attempt is blocked and the
    caller must not reroute it (§12: "it never substitutes another model").
    Returns the persisted event record including a monotonic ``id`` (the
    autoincrement row id, used for ordering, never wall-clock time) for
    audit/CLI display.
    """
    return _record_revocation_event(
        hermes_home, policy_id, route_id=route_id, event_type="revoke",
        reason=reason, approval_ref=approval_ref,
    )


def readmit_route(
    hermes_home, policy_id: str, *, route_id: Optional[str] = None,
    reason: str, approval_ref: str,
) -> dict:
    """Explicit, auditable READMISSION of a previously emergency-revoked route
    (or whole policy) -- design §12/root-AGENTS binding: "Route/policy-wide
    revocations and explicit re-admission must be well-scoped".

    The ONLY way to clear a durable ``revoke_route`` block. Never implied by
    ``publish_policy``/``activate_policy``, never automatic/time-based expiry.
    ``is_route_revoked`` resolves current state as whichever applicable event
    -- route-specific or whole-policy -- is most recent by monotonic id, so a
    route-specific readmission clears a route-specific revocation, and also
    clears an earlier whole-policy revocation for that one route (a targeted
    readmission is itself the well-scoped act; a fresh whole-policy
    revocation recorded after it still wins because it is the newer event).
    """
    return _record_revocation_event(
        hermes_home, policy_id, route_id=route_id, event_type="readmit",
        reason=reason, approval_ref=approval_ref,
    )


def is_route_revoked(hermes_home, policy_id: str, route_id: str) -> Optional[dict]:
    """Current durable authorization state for ``route_id`` of ``policy_id``:
    the revocation event record if the route is CURRENTLY revoked, else
    ``None``.

    Resolves state from monotonic event ORDER (autoincrement row id), never
    wall-clock ``created_at`` -- two events recorded in the same second (or
    across a clock skew/restart) are still ordered correctly, and a route's
    state is durable authorization state rather than something computed
    relative to any particular receipt's timestamp. Considers every event
    whose ``route_id`` is either the exact route or ``NULL`` (whole-policy);
    whichever single event (route-specific or whole-policy) is most recent
    by id determines the CURRENT state -- a revoke with no later readmit
    (route-specific or whole-policy) blocks; a readmit with no later revoke
    does not. This applies uniformly to every receipt referencing this
    route regardless of when that receipt itself was persisted: revocation
    is state about the ROUTE, not an event scoped to any one receipt's
    creation time (the parent-reproduced bypass: a receipt minted after the
    revocation must not silently inherit an implicit clean slate).
    """
    with transaction(_connect(hermes_home)) as conn:
        row = conn.execute(
            "SELECT id, policy_id, route_id, event_type, reason, approval_ref, created_at "
            "FROM route_revocations WHERE policy_id=? AND (route_id=? OR route_id IS NULL) "
            "ORDER BY id DESC LIMIT 1",
            (policy_id, route_id),
        ).fetchone()
    if row is None or row["event_type"] != "revoke":
        return None
    return dict(row)


def find_active_revocation(hermes_home, policy_id: str, route_id: str) -> Optional[dict]:
    """Back-compat alias for ``is_route_revoked`` (same durable-state contract,
    no ``since_ts``/receipt-timestamp parameter -- revocation is authorization
    state about the route, not something scoped to when any one receipt was
    persisted)."""
    return is_route_revoked(hermes_home, policy_id, route_id)


def append_outcome(hermes_home, receipt_id: str, kind: str, payload: dict) -> None:
    """Append-only outcome event (design §3.C: never rewrite the original
    decision to match what happened)."""
    now = int(time.time())
    with transaction(_connect(hermes_home)) as conn:
        row = conn.execute(
            "SELECT COALESCE(MAX(seq), 0) + 1 AS n FROM routing_outcomes WHERE receipt_id=?",
            (receipt_id,),
        ).fetchone()
        seq = row["n"]
        conn.execute(
            "INSERT INTO routing_outcomes (receipt_id, seq, kind, payload_json, created_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (receipt_id, seq, kind, _canonical_json(payload), now),
        )


def list_outcomes(hermes_home, receipt_id: str) -> list[dict]:
    with transaction(_connect(hermes_home)) as conn:
        rows = conn.execute(
            "SELECT seq, kind, payload_json, created_at FROM routing_outcomes "
            "WHERE receipt_id=? ORDER BY seq", (receipt_id,),
        ).fetchall()
    return [{"seq": r["seq"], "kind": r["kind"], "payload": json.loads(r["payload_json"]),
             "created_at": r["created_at"]} for r in rows]
