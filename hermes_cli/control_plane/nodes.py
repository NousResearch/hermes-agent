"""Durable managed-node identity and lifecycle registry.

This module deliberately has no inference-provider dependency.  It owns the
control-plane facts needed before any workload can be scheduled: stable node
identity, operator ownership, declared role, lifecycle state, capabilities,
optimistic concurrency, and a hash-chained audit history.
"""

from __future__ import annotations

import hashlib
import json
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


class IdempotencyConflict(ValueError):
    """An enrollment key was reused for different node facts."""


class ConcurrencyConflict(RuntimeError):
    """The caller acted on a stale node revision."""


class InvalidTransition(ValueError):
    """The requested lifecycle transition is not allowed."""


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


def control_plane_db_path() -> Path:
    return get_hermes_home() / "control-plane.db"


def _canonical_json(value: Mapping[str, Any]) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _required_text(value: str, field: str) -> str:
    normalized = str(value).strip()
    if not normalized:
        raise ValueError(f"{field} must not be empty")
    return normalized


class NodeRegistry:
    """SQLite-backed authoritative registry for managed nodes."""

    def __init__(
        self,
        db_path: Path | None = None,
        *,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self.db_path = db_path or control_plane_db_path()
        self._clock = clock

    def connect(self) -> sqlite3.Connection:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(self.db_path), timeout=10)
        try:
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA foreign_keys=ON")
            from hermes_state import apply_wal_with_fallback

            apply_wal_with_fallback(conn, db_label="control-plane.db")
            conn.executescript(SCHEMA_SQL)
        except Exception:
            conn.close()
            raise
        return conn

    def enroll(
        self,
        *,
        enrollment_key: str,
        role: str,
        owner: str,
        actor: str,
        capabilities: Mapping[str, Any] | None = None,
        node_id: str | None = None,
    ) -> Node:
        """Enroll once, returning the same node for an identical retry."""
        key = _required_text(enrollment_key, "enrollment_key")
        role = _required_text(role, "role")
        owner = _required_text(owner, "owner")
        actor = _required_text(actor, "actor")
        requested_node_id = (
            _required_text(node_id, "node_id") if node_id is not None else None
        )
        node_id = requested_node_id or str(uuid.uuid4())
        capabilities_json = _canonical_json(dict(capabilities or {}))
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
                return self._node(existing)

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
                },
            )
            row = conn.execute(
                "SELECT * FROM managed_nodes WHERE id = ?", (node_id,)
            ).fetchone()
            return self._node(row)

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
            updated = conn.execute(
                "SELECT * FROM managed_nodes WHERE id = ?", (node_id,)
            ).fetchone()
            return self._node(updated)

    def get(self, node_id: str) -> Node | None:
        with self.connect() as conn:
            row = conn.execute(
                "SELECT * FROM managed_nodes WHERE id = ?", (node_id,)
            ).fetchone()
            return self._node(row) if row is not None else None

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
            return [self._node(row) for row in conn.execute(query, params)]

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
            for row in conn.execute(
                "SELECT * FROM managed_node_events ORDER BY sequence"
            ):
                if row["previous_hash"] != previous_hash:
                    return False
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
                if row["event_hash"] != expected:
                    return False
                previous_hash = row["event_hash"]
        return True

    @staticmethod
    def _node(row: sqlite3.Row) -> Node:
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
        )

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
        previous = conn.execute(
            "SELECT event_hash FROM managed_node_events ORDER BY sequence DESC LIMIT 1"
        ).fetchone()
        previous_hash = previous["event_hash"] if previous is not None else None
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
        conn.execute(
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
