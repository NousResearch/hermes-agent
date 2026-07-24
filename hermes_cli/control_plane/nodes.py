"""Durable managed-node identity and lifecycle registry.

This module deliberately has no inference-provider dependency.  It owns the
control-plane facts needed before any workload can be scheduled: stable node
identity, operator ownership, declared role, lifecycle state, capabilities,
optimistic concurrency, and a hash-chained audit history.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import secrets
import sqlite3
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

from hermes_cli.sqlite_util import write_txn
from hermes_constants import get_hermes_home

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS managed_nodes (
    id                TEXT PRIMARY KEY,
    enrollment_key    TEXT NOT NULL UNIQUE,
    role              TEXT NOT NULL,
    owner             TEXT NOT NULL,
    state             TEXT NOT NULL,
    capabilities_json TEXT NOT NULL,
    revision          INTEGER NOT NULL,
    created_at        INTEGER NOT NULL,
    updated_at        INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS managed_node_events (
    sequence       INTEGER PRIMARY KEY AUTOINCREMENT,
    node_id        TEXT NOT NULL REFERENCES managed_nodes(id),
    event_type     TEXT NOT NULL,
    actor          TEXT NOT NULL,
    from_state     TEXT,
    to_state       TEXT NOT NULL,
    node_revision  INTEGER NOT NULL,
    occurred_at    INTEGER NOT NULL,
    details_json   TEXT NOT NULL,
    previous_hash  TEXT,
    event_hash     TEXT NOT NULL UNIQUE
);

CREATE INDEX IF NOT EXISTS idx_managed_node_events_node_sequence
    ON managed_node_events(node_id, sequence);

CREATE TABLE IF NOT EXISTS managed_node_audit_head (
    singleton       INTEGER PRIMARY KEY CHECK (singleton = 1),
    event_sequence  INTEGER NOT NULL,
    event_hash      TEXT
);

CREATE TABLE IF NOT EXISTS managed_node_credentials (
    node_id        TEXT PRIMARY KEY REFERENCES managed_nodes(id),
    verifier       TEXT NOT NULL,
    revision       INTEGER NOT NULL,
    issued_at      INTEGER NOT NULL,
    rotated_at     INTEGER,
    revoked_at     INTEGER
);

CREATE TABLE IF NOT EXISTS managed_node_observations (
    node_id          TEXT NOT NULL REFERENCES managed_nodes(id),
    report_sequence  INTEGER NOT NULL,
    schema_version   INTEGER NOT NULL,
    observed_at      INTEGER NOT NULL,
    received_at      INTEGER NOT NULL,
    health_state     TEXT NOT NULL,
    capabilities_json TEXT NOT NULL,
    PRIMARY KEY (node_id, report_sequence)
);

CREATE TABLE IF NOT EXISTS managed_node_policies (
    node_id          TEXT PRIMARY KEY REFERENCES managed_nodes(id),
    schema_version   INTEGER NOT NULL,
    desired_health_state TEXT,
    capabilities_json TEXT NOT NULL,
    revision         INTEGER NOT NULL,
    updated_at       INTEGER NOT NULL
);
"""

INITIAL_STATE = "enrolled"
LIFECYCLE_STATES = frozenset({
    "enrolled",
    "active",
    "quarantined",
    "recovering",
    "retired",
})
ALLOWED_TRANSITIONS = {
    "enrolled": frozenset({"active", "quarantined", "retired"}),
    "active": frozenset({"quarantined", "retired"}),
    "quarantined": frozenset({"recovering", "retired"}),
    "recovering": frozenset({"active", "quarantined", "retired"}),
    "retired": frozenset(),
}
OBSERVATION_SCHEMA_VERSION = 1
POLICY_SCHEMA_VERSION = 1
HEALTH_STATES = frozenset({"healthy", "degraded", "unhealthy", "unknown"})
SECRET_FIELD_NAMES = frozenset({
    "api_key",
    "credential",
    "credentials",
    "password",
    "secret",
    "token",
})


class IdempotencyConflict(ValueError):
    """An enrollment key was reused for different node facts."""


class ConcurrencyConflict(RuntimeError):
    """The caller acted on a stale node revision."""


class InvalidTransition(ValueError):
    """The requested lifecycle transition is not allowed."""


class CredentialConflict(RuntimeError):
    """The caller acted on a stale credential revision."""


class ReportConflict(RuntimeError):
    """An observed report sequence is stale or has already been accepted."""


class PolicyConflict(RuntimeError):
    """The caller acted on a stale desired-policy revision."""


class AuthenticationFailed(PermissionError):
    """A managed node did not present its active credential."""


@dataclass(frozen=True)
class Node:
    id: str
    enrollment_key: str
    role: str
    owner: str
    state: str
    capabilities: dict[str, Any]
    revision: int
    created_at: int
    updated_at: int
    credential_revision: int
    credential_status: str
    credential_issued_at: int
    credential_rotated_at: int | None
    credential_revoked_at: int | None


@dataclass(frozen=True)
class CredentialIssuance:
    """One-time credential delivery paired with its non-secret node view."""

    node: Node
    credential: str | None


@dataclass(frozen=True)
class NodeEvent:
    sequence: int
    node_id: str
    event_type: str
    actor: str
    from_state: str | None
    to_state: str
    node_revision: int
    occurred_at: int
    details: dict[str, Any]
    previous_hash: str | None
    event_hash: str


@dataclass(frozen=True)
class ObservedReport:
    node_id: str
    schema_version: int
    report_sequence: int
    observed_at: int
    received_at: int
    health_state: str
    capabilities: dict[str, Any]


@dataclass(frozen=True)
class DesiredPolicy:
    node_id: str
    schema_version: int
    desired_health_state: str | None
    capabilities: dict[str, Any]
    revision: int
    updated_at: int


@dataclass(frozen=True)
class ReconciliationResult:
    node_id: str
    policy: DesiredPolicy | None
    observation: ObservedReport | None
    in_sync: bool
    drift: list[dict[str, Any]]


def control_plane_db_path() -> Path:
    return get_hermes_home() / "control-plane.db"


def _canonical_json(value: Mapping[str, Any]) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )


def _required_text(value: str, field: str) -> str:
    normalized = str(value).strip()
    if not normalized:
        raise ValueError(f"{field} must not be empty")
    return normalized


def _json_object(value: Mapping[str, Any], field: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field} must be a JSON object")
    result = dict(value)
    try:
        _canonical_json(result)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must contain JSON values") from exc
    return result


def _required_int(value: int, field: str, *, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        qualifier = "positive" if minimum == 1 else "non-negative"
        raise ValueError(f"{field} must be a {qualifier} integer")
    return value


def _reject_secret_fields(value: Any, field: str) -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            if str(key).lower() in SECRET_FIELD_NAMES:
                raise ValueError(f"{field} must not contain secret field {key!r}")
            _reject_secret_fields(child, field)
    elif isinstance(value, list):
        for child in value:
            _reject_secret_fields(child, field)


class NodeRegistry:
    """SQLite-backed authoritative registry for managed nodes."""

    def __init__(
        self,
        db_path: Path | None = None,
        *,
        clock: Callable[[], float] = time.time,
        credential_factory: Callable[[], str] | None = None,
    ) -> None:
        self.db_path = db_path or control_plane_db_path()
        self._clock = clock
        self._credential_factory = credential_factory or (
            lambda: f"hermes_node_{secrets.token_urlsafe(32)}"
        )

    def connect(self) -> sqlite3.Connection:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(self.db_path), timeout=10)
        try:
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA foreign_keys=ON")
            from hermes_state import apply_wal_with_fallback

            apply_wal_with_fallback(conn, db_label="control-plane.db")
            conn.executescript(SCHEMA_SQL)
            self._migrate_audit_head(conn)
            self._migrate_uncredentialed_nodes(conn)
        except Exception:
            conn.close()
            raise
        return conn

    @staticmethod
    def _migrate_audit_head(conn: sqlite3.Connection) -> None:
        """Idempotently anchor a legacy event chain outside its event rows."""
        schema_version = conn.execute("PRAGMA user_version").fetchone()[0]
        if schema_version >= 1:
            return
        if (
            conn.execute("SELECT 1 FROM managed_node_audit_head LIMIT 1").fetchone()
            is not None
        ):
            conn.execute("PRAGMA user_version = 1")
            return
        with write_txn(conn):
            latest = conn.execute(
                """
                SELECT sequence, event_hash
                FROM managed_node_events ORDER BY sequence DESC LIMIT 1
                """
            ).fetchone()
            sequence = latest["sequence"] if latest is not None else 0
            event_hash = latest["event_hash"] if latest is not None else None
            conn.execute(
                """
                INSERT INTO managed_node_audit_head (
                    singleton, event_sequence, event_hash
                ) VALUES (1, ?, ?)
                """,
                (sequence, event_hash),
            )
            conn.execute("PRAGMA user_version = 1")

    def _migrate_uncredentialed_nodes(self, conn: sqlite3.Connection) -> None:
        """Safely disable legacy nodes until an operator rotates a credential."""
        missing = conn.execute(
            """
            SELECT 1
            FROM managed_nodes
            LEFT JOIN managed_node_credentials
              ON managed_node_credentials.node_id = managed_nodes.id
            WHERE managed_node_credentials.node_id IS NULL
            LIMIT 1
            """
        ).fetchone()
        if missing is None:
            return
        now = int(self._clock())
        with write_txn(conn):
            rows = conn.execute(
                """
                SELECT managed_nodes.*
                FROM managed_nodes
                LEFT JOIN managed_node_credentials
                  ON managed_node_credentials.node_id = managed_nodes.id
                WHERE managed_node_credentials.node_id IS NULL
                ORDER BY managed_nodes.created_at, managed_nodes.id
                """
            ).fetchall()
            for row in rows:
                discarded_credential = self._new_credential()
                conn.execute(
                    """
                    INSERT INTO managed_node_credentials (
                        node_id, verifier, revision, issued_at, rotated_at, revoked_at
                    ) VALUES (?, ?, 1, ?, NULL, ?)
                    """,
                    (
                        row["id"],
                        self._credential_verifier(discarded_credential),
                        now,
                        now,
                    ),
                )
                self._append_event(
                    conn,
                    node_id=row["id"],
                    event_type="node.credential_migrated_revoked",
                    actor="system:migration",
                    from_state=row["state"],
                    to_state=row["state"],
                    revision=row["revision"],
                    occurred_at=now,
                    details={"credential_revision": 1},
                )

    def enroll(
        self,
        *,
        enrollment_key: str,
        role: str,
        owner: str,
        actor: str,
        capabilities: Mapping[str, Any] | None = None,
        node_id: str | None = None,
    ) -> CredentialIssuance:
        """Enroll once and return the new raw credential exactly once."""
        key = _required_text(enrollment_key, "enrollment_key")
        role = _required_text(role, "role")
        owner = _required_text(owner, "owner")
        actor = _required_text(actor, "actor")
        requested_node_id = (
            _required_text(node_id, "node_id") if node_id is not None else None
        )
        node_id = requested_node_id or str(uuid.uuid4())
        capabilities_value = _json_object(capabilities or {}, "capabilities")
        _reject_secret_fields(capabilities_value, "capabilities")
        capabilities_json = _canonical_json(capabilities_value)
        now = int(self._clock())

        with self.connect() as conn, write_txn(conn):
            existing = conn.execute(
                "SELECT * FROM managed_nodes WHERE enrollment_key = ?", (key,)
            ).fetchone()
            if existing is not None:
                if (
                    existing["role"] != role
                    or existing["owner"] != owner
                    or existing["capabilities_json"] != capabilities_json
                    or (
                        requested_node_id is not None
                        and existing["id"] != requested_node_id
                    )
                ):
                    raise IdempotencyConflict(
                        f"enrollment key {key!r} already identifies a different node"
                    )
                return CredentialIssuance(
                    node=self._node_with_credential(conn, existing),
                    credential=None,
                )

            if requested_node_id is not None:
                existing_id = conn.execute(
                    "SELECT enrollment_key FROM managed_nodes WHERE id = ?",
                    (requested_node_id,),
                ).fetchone()
                if existing_id is not None:
                    raise IdempotencyConflict(
                        f"node id {requested_node_id!r} is already assigned to "
                        "a different enrollment key"
                    )

            credential = self._new_credential()
            try:
                conn.execute(
                    """
                    INSERT INTO managed_nodes (
                        id, enrollment_key, role, owner, state, capabilities_json,
                        revision, created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, 1, ?, ?)
                    """,
                    (
                        node_id,
                        key,
                        role,
                        owner,
                        INITIAL_STATE,
                        capabilities_json,
                        now,
                        now,
                    ),
                )
            except sqlite3.IntegrityError as exc:
                raise IdempotencyConflict(
                    "enrollment identity conflicts with an existing node"
                ) from exc
            conn.execute(
                """
                INSERT INTO managed_node_credentials (
                    node_id, verifier, revision, issued_at, rotated_at, revoked_at
                ) VALUES (?, ?, 1, ?, NULL, NULL)
                """,
                (node_id, self._credential_verifier(credential), now),
            )
            self._append_event(
                conn,
                node_id=node_id,
                event_type="node.enrolled",
                actor=actor,
                from_state=None,
                to_state=INITIAL_STATE,
                revision=1,
                occurred_at=now,
                details={
                    "capabilities": json.loads(capabilities_json),
                    "owner": owner,
                    "role": role,
                    "credential_revision": 1,
                },
            )
            row = conn.execute(
                "SELECT * FROM managed_nodes WHERE id = ?", (node_id,)
            ).fetchone()
            return CredentialIssuance(
                node=self._node_with_credential(conn, row),
                credential=credential,
            )

    def authenticate(self, node_id: str, credential: str) -> bool:
        """Verify an active node credential without exposing its verifier."""
        node_id = _required_text(node_id, "node_id")
        candidate = _required_text(credential, "credential")
        with self.connect() as conn:
            row = conn.execute(
                """
                SELECT credentials.verifier, credentials.revoked_at, nodes.state
                FROM managed_node_credentials AS credentials
                JOIN managed_nodes AS nodes ON nodes.id = credentials.node_id
                WHERE credentials.node_id = ?
                """,
                (node_id,),
            ).fetchone()
        if row is None or row["revoked_at"] is not None or row["state"] == "retired":
            return False
        return hmac.compare_digest(
            row["verifier"],
            self._credential_verifier(candidate),
        )

    def submit_observation(
        self,
        node_id: str,
        *,
        credential: str,
        schema_version: int,
        report_sequence: int,
        observed_at: int,
        health_state: str,
        capabilities: Mapping[str, Any],
    ) -> ObservedReport:
        """Authenticate and append a strictly newer immutable observed report."""
        node_id = _required_text(node_id, "node_id")
        candidate = _required_text(credential, "credential")
        _required_int(schema_version, "schema_version", minimum=1)
        if schema_version != OBSERVATION_SCHEMA_VERSION:
            raise ValueError(
                f"unsupported observation schema_version: {schema_version}"
            )
        _required_int(report_sequence, "report_sequence", minimum=1)
        _required_int(observed_at, "observed_at", minimum=0)
        health_state = _required_text(health_state, "health_state")
        if health_state not in HEALTH_STATES:
            raise ValueError(f"unknown health_state: {health_state}")
        capabilities_value = _json_object(capabilities, "capabilities")
        _reject_secret_fields(capabilities_value, "capabilities")
        capabilities_json = _canonical_json(capabilities_value)
        received_at = int(self._clock())

        with self.connect() as conn, write_txn(conn):
            node = conn.execute(
                "SELECT * FROM managed_nodes WHERE id = ?", (node_id,)
            ).fetchone()
            if node is None:
                raise KeyError(node_id)
            auth = conn.execute(
                "SELECT verifier, revoked_at FROM managed_node_credentials WHERE node_id = ?",
                (node_id,),
            ).fetchone()
            if (
                auth is None
                or node["state"] == "retired"
                or auth["revoked_at"] is not None
                or not hmac.compare_digest(
                    auth["verifier"], self._credential_verifier(candidate)
                )
            ):
                raise AuthenticationFailed("node authentication failed")
            latest = conn.execute(
                """
                SELECT report_sequence FROM managed_node_observations
                WHERE node_id = ? ORDER BY report_sequence DESC LIMIT 1
                """,
                (node_id,),
            ).fetchone()
            if latest is not None and report_sequence <= latest["report_sequence"]:
                raise ReportConflict(
                    f"report_sequence must be greater than {latest['report_sequence']}"
                )
            conn.execute(
                """
                INSERT INTO managed_node_observations (
                    node_id, report_sequence, schema_version, observed_at,
                    received_at, health_state, capabilities_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    node_id,
                    report_sequence,
                    schema_version,
                    observed_at,
                    received_at,
                    health_state,
                    capabilities_json,
                ),
            )
            self._append_event(
                conn,
                node_id=node_id,
                event_type="node.observation_accepted",
                actor=f"node:{node_id}",
                from_state=node["state"],
                to_state=node["state"],
                revision=node["revision"],
                occurred_at=received_at,
                details={
                    "schema_version": schema_version,
                    "report_sequence": report_sequence,
                    "observed_at": observed_at,
                    "health_state": health_state,
                    "capabilities_hash": hashlib.sha256(
                        capabilities_json.encode("utf-8")
                    ).hexdigest(),
                },
            )
        return ObservedReport(
            node_id=node_id,
            schema_version=schema_version,
            report_sequence=report_sequence,
            observed_at=observed_at,
            received_at=received_at,
            health_state=health_state,
            capabilities=capabilities_value,
        )

    def latest_observation(self, node_id: str) -> ObservedReport | None:
        node_id = _required_text(node_id, "node_id")
        with self.connect() as conn:
            if (
                conn.execute(
                    "SELECT 1 FROM managed_nodes WHERE id = ?", (node_id,)
                ).fetchone()
                is None
            ):
                raise KeyError(node_id)
            row = conn.execute(
                """
                SELECT * FROM managed_node_observations
                WHERE node_id = ? ORDER BY report_sequence DESC LIMIT 1
                """,
                (node_id,),
            ).fetchone()
            return self._observation(row) if row is not None else None

    def set_policy(
        self,
        node_id: str,
        *,
        actor: str,
        schema_version: int,
        desired_health_state: str | None,
        capabilities: Mapping[str, Any],
        expected_revision: int,
    ) -> DesiredPolicy:
        """Create or replace the declarative policy using optimistic concurrency."""
        node_id = _required_text(node_id, "node_id")
        actor = _required_text(actor, "actor")
        _required_int(schema_version, "schema_version", minimum=1)
        if schema_version != POLICY_SCHEMA_VERSION:
            raise ValueError(f"unsupported policy schema_version: {schema_version}")
        if desired_health_state is not None:
            desired_health_state = _required_text(
                desired_health_state, "desired_health_state"
            )
            if desired_health_state not in HEALTH_STATES:
                raise ValueError(
                    f"unknown desired_health_state: {desired_health_state}"
                )
        capabilities_value = _json_object(capabilities, "capabilities")
        _reject_secret_fields(capabilities_value, "capabilities")
        capabilities_json = _canonical_json(capabilities_value)
        _required_int(expected_revision, "expected_revision", minimum=0)
        now = int(self._clock())
        with self.connect() as conn, write_txn(conn):
            node = conn.execute(
                "SELECT * FROM managed_nodes WHERE id = ?", (node_id,)
            ).fetchone()
            if node is None:
                raise KeyError(node_id)
            current = conn.execute(
                "SELECT * FROM managed_node_policies WHERE node_id = ?", (node_id,)
            ).fetchone()
            current_revision = current["revision"] if current is not None else 0
            if expected_revision != current_revision:
                raise PolicyConflict(
                    f"node {node_id} policy is at revision {current_revision}, "
                    f"not {expected_revision}"
                )
            revision = current_revision + 1
            conn.execute(
                """
                INSERT INTO managed_node_policies (
                    node_id, schema_version, desired_health_state,
                    capabilities_json, revision, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(node_id) DO UPDATE SET
                    schema_version=excluded.schema_version,
                    desired_health_state=excluded.desired_health_state,
                    capabilities_json=excluded.capabilities_json,
                    revision=excluded.revision,
                    updated_at=excluded.updated_at
                """,
                (
                    node_id,
                    schema_version,
                    desired_health_state,
                    capabilities_json,
                    revision,
                    now,
                ),
            )
            self._append_event(
                conn,
                node_id=node_id,
                event_type="node.policy_updated",
                actor=actor,
                from_state=node["state"],
                to_state=node["state"],
                revision=node["revision"],
                occurred_at=now,
                details={
                    "schema_version": schema_version,
                    "policy_revision": revision,
                    "desired_health_state": desired_health_state,
                    "capabilities_hash": hashlib.sha256(
                        capabilities_json.encode("utf-8")
                    ).hexdigest(),
                },
            )
        return DesiredPolicy(
            node_id=node_id,
            schema_version=schema_version,
            desired_health_state=desired_health_state,
            capabilities=capabilities_value,
            revision=revision,
            updated_at=now,
        )

    def get_policy(self, node_id: str) -> DesiredPolicy | None:
        node_id = _required_text(node_id, "node_id")
        with self.connect() as conn:
            if (
                conn.execute(
                    "SELECT 1 FROM managed_nodes WHERE id = ?", (node_id,)
                ).fetchone()
                is None
            ):
                raise KeyError(node_id)
            row = conn.execute(
                "SELECT * FROM managed_node_policies WHERE node_id = ?", (node_id,)
            ).fetchone()
            return self._policy(row) if row is not None else None

    def reconcile(self, node_id: str) -> ReconciliationResult:
        """Compare desired policy with the latest report without taking action."""
        node_id = _required_text(node_id, "node_id")
        with self.connect() as conn:
            conn.execute("BEGIN")
            if (
                conn.execute(
                    "SELECT 1 FROM managed_nodes WHERE id = ?", (node_id,)
                ).fetchone()
                is None
            ):
                raise KeyError(node_id)
            policy_row = conn.execute(
                "SELECT * FROM managed_node_policies WHERE node_id = ?", (node_id,)
            ).fetchone()
            observation_row = conn.execute(
                """
                SELECT * FROM managed_node_observations
                WHERE node_id = ? ORDER BY report_sequence DESC LIMIT 1
                """,
                (node_id,),
            ).fetchone()
            policy = self._policy(policy_row) if policy_row is not None else None
            observation = (
                self._observation(observation_row)
                if observation_row is not None
                else None
            )
        drift: list[dict[str, Any]] = []
        if policy is None:
            drift.append({"path": "policy", "reason": "missing_policy"})
        elif observation is None:
            drift.append({"path": "observation", "reason": "missing_observation"})
        else:
            if (
                policy.desired_health_state is not None
                and policy.desired_health_state != observation.health_state
            ):
                drift.append({
                    "path": "health_state",
                    "reason": "value_mismatch",
                    "desired": policy.desired_health_state,
                    "observed": observation.health_state,
                })
            self._capability_drift(
                policy.capabilities, observation.capabilities, "capabilities", drift
            )
        return ReconciliationResult(
            node_id=node_id,
            policy=policy,
            observation=observation,
            in_sync=not drift,
            drift=drift,
        )

    def rotate_credential(
        self,
        node_id: str,
        *,
        actor: str,
        expected_credential_revision: int,
    ) -> CredentialIssuance:
        """Invalidate the prior credential and issue a replacement once."""
        return self._change_credential(
            node_id,
            actor=actor,
            expected_credential_revision=expected_credential_revision,
            revoke=False,
        )

    def revoke_credential(
        self,
        node_id: str,
        *,
        actor: str,
        expected_credential_revision: int,
    ) -> Node:
        """Explicitly revoke a node credential."""
        return self._change_credential(
            node_id,
            actor=actor,
            expected_credential_revision=expected_credential_revision,
            revoke=True,
        ).node

    def transition(
        self,
        node_id: str,
        to_state: str,
        *,
        actor: str,
        expected_revision: int,
        reason: str,
    ) -> Node:
        """Apply one explicit lifecycle transition with compare-and-swap."""
        node_id = _required_text(node_id, "node_id")
        to_state = _required_text(to_state, "to_state")
        actor = _required_text(actor, "actor")
        reason = _required_text(reason, "reason")
        if to_state not in LIFECYCLE_STATES:
            raise InvalidTransition(f"unknown lifecycle state: {to_state}")

        now = int(self._clock())
        with self.connect() as conn, write_txn(conn):
            row = conn.execute(
                "SELECT * FROM managed_nodes WHERE id = ?", (node_id,)
            ).fetchone()
            if row is None:
                raise KeyError(node_id)
            if row["revision"] != expected_revision:
                raise ConcurrencyConflict(
                    f"node {node_id} is at revision {row['revision']}, "
                    f"not {expected_revision}"
                )
            from_state = row["state"]
            if to_state not in ALLOWED_TRANSITIONS[from_state]:
                raise InvalidTransition(f"cannot transition {from_state} -> {to_state}")

            revision = expected_revision + 1
            conn.execute(
                """
                UPDATE managed_nodes
                SET state = ?, revision = ?, updated_at = ?
                WHERE id = ? AND revision = ?
                """,
                (to_state, revision, now, node_id, expected_revision),
            )
            self._append_event(
                conn,
                node_id=node_id,
                event_type=f"node.{to_state}",
                actor=actor,
                from_state=from_state,
                to_state=to_state,
                revision=revision,
                occurred_at=now,
                details={"reason": reason},
            )
            if to_state == "retired":
                credential = conn.execute(
                    """
                    SELECT revision, revoked_at
                    FROM managed_node_credentials WHERE node_id = ?
                    """,
                    (node_id,),
                ).fetchone()
                if credential["revoked_at"] is None:
                    credential_revision = credential["revision"] + 1
                    conn.execute(
                        """
                        UPDATE managed_node_credentials
                        SET revision = ?, revoked_at = ?
                        WHERE node_id = ? AND revision = ? AND revoked_at IS NULL
                        """,
                        (
                            credential_revision,
                            now,
                            node_id,
                            credential["revision"],
                        ),
                    )
                    self._append_event(
                        conn,
                        node_id=node_id,
                        event_type="node.credential_revoked",
                        actor=actor,
                        from_state=to_state,
                        to_state=to_state,
                        revision=revision,
                        occurred_at=now,
                        details={
                            "credential_revision": credential_revision,
                            "reason": "node retired",
                        },
                    )
            updated = conn.execute(
                "SELECT * FROM managed_nodes WHERE id = ?", (node_id,)
            ).fetchone()
            return self._node_with_credential(conn, updated)

    def get(self, node_id: str) -> Node | None:
        with self.connect() as conn:
            row = conn.execute(
                "SELECT * FROM managed_nodes WHERE id = ?", (node_id,)
            ).fetchone()
            return self._node_with_credential(conn, row) if row is not None else None

    def list(self, *, state: str | None = None) -> list[Node]:
        if state is not None and state not in LIFECYCLE_STATES:
            raise ValueError(f"unknown lifecycle state: {state}")
        query = "SELECT * FROM managed_nodes"
        params: tuple[str, ...] = ()
        if state is not None:
            query += " WHERE state = ?"
            params = (state,)
        query += " ORDER BY created_at, id"
        with self.connect() as conn:
            return [
                self._node_with_credential(conn, row)
                for row in conn.execute(query, params)
            ]

    def history(self, node_id: str) -> list[NodeEvent]:
        with self.connect() as conn:
            rows = conn.execute(
                """
                SELECT * FROM managed_node_events
                WHERE node_id = ? ORDER BY sequence
                """,
                (node_id,),
            )
            return [self._event(row) for row in rows]

    def verify_audit_chain(self) -> bool:
        """Return whether every stored event still matches the global hash chain."""
        previous_hash: str | None = None
        with self.connect() as conn:
            heads = conn.execute(
                """
                SELECT singleton, event_sequence, event_hash
                FROM managed_node_audit_head
                """
            ).fetchall()
            if len(heads) != 1:
                return False
            head = heads[0]
            head_sequence = head["event_sequence"]
            head_hash = head["event_hash"]
            if (
                head["singleton"] != 1
                or isinstance(head_sequence, bool)
                or not isinstance(head_sequence, int)
                or head_sequence < 0
                or (head_sequence == 0 and head_hash is not None)
                or (
                    head_sequence > 0
                    and (
                        not isinstance(head_hash, str)
                        or len(head_hash) != 64
                        or any(char not in "0123456789abcdef" for char in head_hash)
                    )
                )
            ):
                return False
            last_sequence = 0
            for row in conn.execute(
                "SELECT * FROM managed_node_events ORDER BY sequence"
            ):
                if row["previous_hash"] != previous_hash:
                    return False
                try:
                    expected = self._event_hash(
                        previous_hash=previous_hash,
                        node_id=row["node_id"],
                        event_type=row["event_type"],
                        actor=row["actor"],
                        from_state=row["from_state"],
                        to_state=row["to_state"],
                        revision=row["node_revision"],
                        occurred_at=row["occurred_at"],
                        details_json=row["details_json"],
                    )
                except (TypeError, ValueError, UnicodeError):
                    return False
                if row["event_hash"] != expected:
                    return False
                previous_hash = row["event_hash"]
                last_sequence = row["sequence"]
        return last_sequence == head_sequence and previous_hash == head_hash

    @staticmethod
    def _node(row: sqlite3.Row, credential: sqlite3.Row) -> Node:
        return Node(
            id=row["id"],
            enrollment_key=row["enrollment_key"],
            role=row["role"],
            owner=row["owner"],
            state=row["state"],
            capabilities=json.loads(row["capabilities_json"]),
            revision=row["revision"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            credential_revision=credential["revision"],
            credential_status=(
                "revoked" if credential["revoked_at"] is not None else "active"
            ),
            credential_issued_at=credential["issued_at"],
            credential_rotated_at=credential["rotated_at"],
            credential_revoked_at=credential["revoked_at"],
        )

    def _node_with_credential(self, conn: sqlite3.Connection, row: sqlite3.Row) -> Node:
        credential = conn.execute(
            """
            SELECT revision, issued_at, rotated_at, revoked_at
            FROM managed_node_credentials WHERE node_id = ?
            """,
            (row["id"],),
        ).fetchone()
        if credential is None:
            raise RuntimeError(f"managed node {row['id']} has no credential")
        return self._node(row, credential)

    def _change_credential(
        self,
        node_id: str,
        *,
        actor: str,
        expected_credential_revision: int,
        revoke: bool,
    ) -> CredentialIssuance:
        node_id = _required_text(node_id, "node_id")
        actor = _required_text(actor, "actor")
        now = int(self._clock())
        raw_credential = None if revoke else self._new_credential()
        with self.connect() as conn, write_txn(conn):
            node_row = conn.execute(
                "SELECT * FROM managed_nodes WHERE id = ?", (node_id,)
            ).fetchone()
            if node_row is None:
                raise KeyError(node_id)
            if not revoke and node_row["state"] == "retired":
                raise InvalidTransition("cannot rotate a retired node credential")
            credential_row = conn.execute(
                "SELECT * FROM managed_node_credentials WHERE node_id = ?", (node_id,)
            ).fetchone()
            if credential_row["revision"] != expected_credential_revision:
                raise CredentialConflict(
                    f"node {node_id} credential is at revision "
                    f"{credential_row['revision']}, not "
                    f"{expected_credential_revision}"
                )
            if revoke and credential_row["revoked_at"] is not None:
                raise CredentialConflict(
                    f"node {node_id} credential is already revoked"
                )

            revision = expected_credential_revision + 1
            if revoke:
                conn.execute(
                    """
                    UPDATE managed_node_credentials
                    SET revision = ?, revoked_at = ?
                    WHERE node_id = ? AND revision = ?
                    """,
                    (revision, now, node_id, expected_credential_revision),
                )
                event_type = "node.credential_revoked"
            else:
                conn.execute(
                    """
                    UPDATE managed_node_credentials
                    SET verifier = ?, revision = ?, rotated_at = ?, revoked_at = NULL
                    WHERE node_id = ? AND revision = ?
                    """,
                    (
                        self._credential_verifier(raw_credential),
                        revision,
                        now,
                        node_id,
                        expected_credential_revision,
                    ),
                )
                event_type = "node.credential_rotated"
            self._append_event(
                conn,
                node_id=node_id,
                event_type=event_type,
                actor=actor,
                from_state=node_row["state"],
                to_state=node_row["state"],
                revision=node_row["revision"],
                occurred_at=now,
                details={"credential_revision": revision},
            )
            return CredentialIssuance(
                node=self._node_with_credential(conn, node_row),
                credential=raw_credential,
            )

    def _new_credential(self) -> str:
        credential = _required_text(self._credential_factory(), "credential")
        return credential

    @staticmethod
    def _credential_verifier(credential: str) -> str:
        return hashlib.sha256(credential.encode("utf-8")).hexdigest()

    @staticmethod
    def _event(row: sqlite3.Row) -> NodeEvent:
        return NodeEvent(
            sequence=row["sequence"],
            node_id=row["node_id"],
            event_type=row["event_type"],
            actor=row["actor"],
            from_state=row["from_state"],
            to_state=row["to_state"],
            node_revision=row["node_revision"],
            occurred_at=row["occurred_at"],
            details=json.loads(row["details_json"]),
            previous_hash=row["previous_hash"],
            event_hash=row["event_hash"],
        )

    @staticmethod
    def _observation(row: sqlite3.Row) -> ObservedReport:
        return ObservedReport(
            node_id=row["node_id"],
            schema_version=row["schema_version"],
            report_sequence=row["report_sequence"],
            observed_at=row["observed_at"],
            received_at=row["received_at"],
            health_state=row["health_state"],
            capabilities=json.loads(row["capabilities_json"]),
        )

    @staticmethod
    def _policy(row: sqlite3.Row) -> DesiredPolicy:
        return DesiredPolicy(
            node_id=row["node_id"],
            schema_version=row["schema_version"],
            desired_health_state=row["desired_health_state"],
            capabilities=json.loads(row["capabilities_json"]),
            revision=row["revision"],
            updated_at=row["updated_at"],
        )

    @classmethod
    def _capability_drift(
        cls,
        desired: Mapping[str, Any],
        observed: Mapping[str, Any],
        prefix: str,
        drift: list[dict[str, Any]],
    ) -> None:
        for key in sorted(desired):
            path = f"{prefix}.{key}"
            if key not in observed:
                drift.append({
                    "path": path,
                    "reason": "missing",
                    "desired": desired[key],
                })
            elif isinstance(desired[key], dict) and isinstance(observed[key], dict):
                cls._capability_drift(desired[key], observed[key], path, drift)
            elif desired[key] != observed[key]:
                drift.append({
                    "path": path,
                    "reason": "value_mismatch",
                    "desired": desired[key],
                    "observed": observed[key],
                })

    def _append_event(
        self,
        conn: sqlite3.Connection,
        *,
        node_id: str,
        event_type: str,
        actor: str,
        from_state: str | None,
        to_state: str,
        revision: int,
        occurred_at: int,
        details: Mapping[str, Any],
    ) -> None:
        head = conn.execute(
            """
            SELECT singleton, event_sequence, event_hash
            FROM managed_node_audit_head
            """
        ).fetchall()
        if len(head) != 1 or head[0]["singleton"] != 1:
            raise RuntimeError("managed-node audit head is missing or malformed")
        previous = conn.execute(
            """
            SELECT sequence, event_hash
            FROM managed_node_events ORDER BY sequence DESC LIMIT 1
            """
        ).fetchone()
        previous_hash = previous["event_hash"] if previous is not None else None
        previous_sequence = previous["sequence"] if previous is not None else 0
        if (
            head[0]["event_sequence"] != previous_sequence
            or head[0]["event_hash"] != previous_hash
        ):
            raise RuntimeError("managed-node audit head does not match event history")
        details_json = _canonical_json(dict(details))
        event_hash = self._event_hash(
            previous_hash=previous_hash,
            node_id=node_id,
            event_type=event_type,
            actor=actor,
            from_state=from_state,
            to_state=to_state,
            revision=revision,
            occurred_at=occurred_at,
            details_json=details_json,
        )
        cursor = conn.execute(
            """
            INSERT INTO managed_node_events (
                node_id, event_type, actor, from_state, to_state, node_revision,
                occurred_at, details_json, previous_hash, event_hash
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                node_id,
                event_type,
                actor,
                from_state,
                to_state,
                revision,
                occurred_at,
                details_json,
                previous_hash,
                event_hash,
            ),
        )
        conn.execute(
            """
            UPDATE managed_node_audit_head
            SET event_sequence = ?, event_hash = ?
            WHERE singleton = 1
            """,
            (cursor.lastrowid, event_hash),
        )

    @staticmethod
    def _event_hash(
        *,
        previous_hash: str | None,
        node_id: str,
        event_type: str,
        actor: str,
        from_state: str | None,
        to_state: str,
        revision: int,
        occurred_at: int,
        details_json: str,
    ) -> str:
        payload = _canonical_json({
            "actor": actor,
            "details": json.loads(details_json),
            "event_type": event_type,
            "from_state": from_state,
            "node_id": node_id,
            "node_revision": revision,
            "occurred_at": occurred_at,
            "previous_hash": previous_hash,
            "to_state": to_state,
        })
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()
