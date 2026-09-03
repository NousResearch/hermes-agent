"""Durable plan/event ledger, lease/fence lifecycle, and integrity backstops.

The store is deliberately append-only at the lifecycle layer.  Plans and node
specifications are immutable after sealing; lease/fence and budget state are
derived from hash-chained events written under ``BEGIN IMMEDIATE``.
"""

from __future__ import annotations

import json
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Iterator, Mapping, NamedTuple

from hermes_constants import get_hermes_home

from agent.phase2_budget import Phase2BudgetMixin
from agent.phase2_envelope import (
    _ALLOWED_SURFACES,
    _HASH_KEYS,
    _MIGRATION_SAMPLE_MAX,
    _bindable_fence,
    _canonical_hash,
    _canonical_identity,
    _canonical_json,
    _expiry_from_ttl,
    _nonnegative_number,
    _iso,
    _json_tree,
    _normalize_claimed_fence,
    _parse_time,
    _sealed_json,
    _utc,
    _valid_sha256,
    bind_sealed_envelope,
    validate_sealed_envelope,
)
from agent.phase2_errors import (
    AuthorityError,
    AuthorityMigrationRequired,
    MalformedEnvelopeFence,
    ResultRejection,
)
from agent.phase2_sqlite import (
    _DB_LOCK,
    _check_phase2_db_files,
    _open_phase2_sqlite,
)

_ACCEPTED_KINDS = ("RESULT_ACCEPTED", "NODE_COMPLETED")


class _IntegrityIndex(NamedTuple):
    """A partial unique index that backstops one authority invariant.

    ``scan_sql`` groups the rows the index would collapse, using exactly the
    index's own NULL semantics (a unique index treats every NULL as distinct,
    so rows with a NULL indexed column can never conflict and are excluded).
    Running the scan before ``CREATE UNIQUE INDEX`` turns a raw
    ``sqlite3.IntegrityError`` on a legacy database into a typed, actionable
    migration decision.
    """

    name: str
    create_sql: str
    invariant: str
    scan_sql: str
    remediation: str


_INTEGRITY_INDEXES: tuple[_IntegrityIndex, ...] = (
    _IntegrityIndex(
        name="one_accepted_result_per_node",
        create_sql=(
            "CREATE UNIQUE INDEX one_accepted_result_per_node "
            "ON authority_events(graph_id, node_id) WHERE kind = 'RESULT_ACCEPTED'"
        ),
        invariant="duplicate_accepted_result",
        scan_sql=(
            "SELECT graph_id || '/' || node_id AS violation_key, COUNT(*) AS events "
            "FROM authority_events "
            "WHERE kind = 'RESULT_ACCEPTED' AND node_id IS NOT NULL "
            "GROUP BY graph_id, node_id HAVING COUNT(*) > 1 "
            "ORDER BY graph_id, node_id"
        ),
        remediation=(
            "more than one RESULT_ACCEPTED exists for a node; exactly one result "
            "may be authoritative, so the correct acceptance must be chosen by an "
            "operator before this store can serve authority"
        ),
    ),
    _IntegrityIndex(
        name="one_terminal_result_per_node",
        create_sql=(
            "CREATE UNIQUE INDEX one_terminal_result_per_node "
            "ON authority_events(graph_id, node_id) "
            "WHERE kind IN ('RESULT_ACCEPTED', 'NODE_COMPLETED')"
        ),
        invariant="duplicate_terminal_result",
        scan_sql=(
            "SELECT graph_id || '/' || node_id AS violation_key, COUNT(*) AS events "
            "FROM authority_events "
            "WHERE kind IN ('RESULT_ACCEPTED', 'NODE_COMPLETED') AND node_id IS NOT NULL "
            "GROUP BY graph_id, node_id HAVING COUNT(*) > 1 "
            "ORDER BY graph_id, node_id"
        ),
        remediation=(
            "a node carries more than one terminal completion row (legacy stores "
            "permitted regrant after completion, so a node could be completed once "
            "per fence); the authoritative terminal event must be chosen by an "
            "operator before this store can serve authority"
        ),
    ),
    _IntegrityIndex(
        name="one_grant_per_attempt",
        create_sql=(
            "CREATE UNIQUE INDEX one_grant_per_attempt "
            "ON authority_events(attempt_id) WHERE kind = 'LEASE_GRANTED'"
        ),
        invariant="duplicate_grant_attempt_id",
        scan_sql=(
            "SELECT attempt_id AS violation_key, COUNT(*) AS events "
            "FROM authority_events "
            "WHERE kind = 'LEASE_GRANTED' AND attempt_id IS NOT NULL "
            "GROUP BY attempt_id HAVING COUNT(*) > 1 "
            "ORDER BY attempt_id"
        ),
        remediation=(
            "an attempt_id was granted more than once (legacy stores did not bind "
            "attempt_id globally, so the same id could be granted on another node "
            "or graph); attempt identity cannot be reconstructed automatically and "
            "must be resolved by an operator"
        ),
    ),
)


class _Reject(Exception):
    """Internal control-flow signal: one typed rejection must be recorded."""

    def __init__(self, reason: str, *, fence_hint: int | None = None) -> None:
        super().__init__(reason)
        self.reason = reason
        self.fence_hint = fence_hint


def default_db_path() -> Path:
    return get_hermes_home() / "phase2_authority.db"


# ── Durable authority store ────────────────────────────────────────────────────


class Phase2AuthorityStore(Phase2BudgetMixin):
    """Seal plans and issue node-scoped fences, leases, and budget reservations."""

    def __init__(self, db_path: Path | str | None = None) -> None:
        self.db_path = Path(db_path) if db_path is not None else default_db_path()

    def _connect(self) -> sqlite3.Connection:
        conn = _open_phase2_sqlite(self.db_path, timeout=5.0)
        try:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA busy_timeout=5000")
            conn.row_factory = sqlite3.Row
            self._ensure_schema(conn)
            _check_phase2_db_files(self.db_path, harden=True)
            return conn
        except BaseException:
            conn.close()
            raise

    def _ensure_schema(self, conn: sqlite3.Connection) -> None:
        self._ensure_base_schema(conn)
        self._ensure_integrity_indexes(conn, self.db_path)

    @staticmethod
    def _ensure_base_schema(conn: sqlite3.Connection) -> None:
        """Create the tables and append-only triggers."""

        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS sealed_plans (
                graph_id TEXT PRIMARY KEY,
                contract_version INTEGER NOT NULL,
                planner_hash TEXT NOT NULL,
                policy_hash TEXT NOT NULL,
                plan_json TEXT NOT NULL,
                sealed_utc TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS sealed_nodes (
                graph_id TEXT NOT NULL,
                node_id TEXT NOT NULL,
                node_json TEXT NOT NULL,
                PRIMARY KEY(graph_id, node_id),
                FOREIGN KEY(graph_id) REFERENCES sealed_plans(graph_id)
            );
            CREATE TABLE IF NOT EXISTS authority_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                event_id TEXT NOT NULL UNIQUE,
                ts_utc TEXT NOT NULL,
                kind TEXT NOT NULL,
                graph_id TEXT NOT NULL,
                node_id TEXT,
                attempt_id TEXT,
                fence INTEGER,
                payload_json TEXT NOT NULL,
                prev_event_hash TEXT,
                event_hash TEXT NOT NULL UNIQUE
            );
            CREATE TRIGGER IF NOT EXISTS sealed_plans_no_update
            BEFORE UPDATE ON sealed_plans BEGIN
                SELECT RAISE(ABORT, 'sealed_plans is immutable');
            END;
            CREATE TRIGGER IF NOT EXISTS sealed_plans_no_delete
            BEFORE DELETE ON sealed_plans BEGIN
                SELECT RAISE(ABORT, 'sealed_plans is immutable');
            END;
            CREATE TRIGGER IF NOT EXISTS sealed_nodes_no_update
            BEFORE UPDATE ON sealed_nodes BEGIN
                SELECT RAISE(ABORT, 'sealed_nodes is immutable');
            END;
            CREATE TRIGGER IF NOT EXISTS sealed_nodes_no_delete
            BEFORE DELETE ON sealed_nodes BEGIN
                SELECT RAISE(ABORT, 'sealed_nodes is immutable');
            END;
            CREATE TRIGGER IF NOT EXISTS authority_events_no_update
            BEFORE UPDATE ON authority_events BEGIN
                SELECT RAISE(ABORT, 'authority_events is append-only');
            END;
            CREATE TRIGGER IF NOT EXISTS authority_events_no_delete
            BEFORE DELETE ON authority_events BEGIN
                SELECT RAISE(ABORT, 'authority_events is append-only');
            END;
            """
        )

    @staticmethod
    def _normalize_schema_sql(value: str | None) -> str:
        return " ".join(str(value or "").casefold().split())

    @classmethod
    def _missing_integrity_indexes(
        cls, conn: sqlite3.Connection
    ) -> list[_IntegrityIndex]:
        present = {
            row[0]: cls._normalize_schema_sql(row[1])
            for row in conn.execute(
                "SELECT name, sql FROM sqlite_master WHERE type = 'index'"
            ).fetchall()
        }
        return [
            index
            for index in _INTEGRITY_INDEXES
            if present.get(index.name) != cls._normalize_schema_sql(index.create_sql)
        ]

    @staticmethod
    def _scan_integrity_violations(
        conn: sqlite3.Connection, indexes: list[_IntegrityIndex]
    ) -> list[dict[str, Any]]:
        """Report, deterministically, which rows would violate each index."""

        violations: list[dict[str, Any]] = []
        for index in indexes:
            groups = conn.execute(
                f"SELECT COUNT(*) FROM ({index.scan_sql})"  # noqa: S608
            ).fetchone()[0]
            if not groups:
                continue
            sample = conn.execute(
                f"{index.scan_sql} LIMIT ?",  # noqa: S608
                (_MIGRATION_SAMPLE_MAX,),
            ).fetchall()
            violations.append({
                "invariant": index.invariant,
                "index": index.name,
                "violating_groups": int(groups),
                "sample": [{"key": row[0], "events": int(row[1])} for row in sample],
                "remediation": index.remediation,
            })
        return violations

    @classmethod
    def _ensure_integrity_indexes(
        cls, conn: sqlite3.Connection, db_path: Path | str
    ) -> None:
        """Create the backstop unique indexes, or fail closed with a typed error."""

        if not cls._missing_integrity_indexes(conn):
            return
        conn.execute("BEGIN IMMEDIATE")
        try:
            missing = cls._missing_integrity_indexes(conn)
            violations = (
                cls._scan_integrity_violations(conn, missing) if missing else []
            )
            if violations:
                conn.execute("ROLLBACK")
                raise AuthorityMigrationRequired(violations, db_path)
            for index in missing:
                conn.execute(f'DROP INDEX IF EXISTS "{index.name}"')
                conn.execute(index.create_sql)
            conn.execute("COMMIT")
        except AuthorityMigrationRequired:
            raise
        except Exception:
            try:
                conn.execute("ROLLBACK")
            except Exception:
                pass
            raise

    @staticmethod
    def _append_event(
        conn: sqlite3.Connection,
        *,
        kind: str,
        graph_id: str,
        node_id: str | None,
        attempt_id: str | None,
        fence: int | None,
        payload: Mapping[str, Any],
        now: datetime,
        event_id: str | None = None,
    ) -> str:
        if not isinstance(graph_id, str):
            raise AuthorityError("event graph_id must be a string")
        if node_id is not None and not isinstance(node_id, str):
            raise AuthorityError("event node_id must be a string or null")
        if attempt_id is not None and not isinstance(attempt_id, str):
            raise AuthorityError("event attempt_id must be a string or null")
        if fence is not None:
            if isinstance(fence, bool) or not isinstance(fence, int):
                raise AuthorityError("event fence must be an integer or null")
            if not _bindable_fence(fence):
                raise AuthorityError("event fence must be a signed 64-bit integer")
        try:
            payload_json = _canonical_json(payload)
        except (TypeError, ValueError) as exc:
            raise AuthorityError(
                "event payload is not canonically serializable"
            ) from exc

        event_id = event_id or f"ev-{uuid.uuid4().hex}"
        previous = conn.execute(
            "SELECT event_hash FROM authority_events ORDER BY id DESC LIMIT 1"
        ).fetchone()
        prev_hash = previous["event_hash"] if previous else None
        row = {
            "event_id": event_id,
            "ts_utc": _iso(now),
            "kind": kind,
            "graph_id": graph_id,
            "node_id": node_id,
            "attempt_id": attempt_id,
            "fence": fence,
            "payload": json.loads(payload_json),
            "prev_event_hash": prev_hash,
        }
        event_hash = _canonical_hash(row)
        conn.execute(
            """INSERT INTO authority_events(
                   event_id, ts_utc, kind, graph_id, node_id, attempt_id, fence,
                   payload_json, prev_event_hash, event_hash
               ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                event_id,
                row["ts_utc"],
                kind,
                graph_id,
                node_id,
                attempt_id,
                fence,
                payload_json,
                prev_hash,
                event_hash,
            ),
        )
        return event_id

    @staticmethod
    def _validate_plan(
        plan: Mapping[str, Any], policy_hash: str
    ) -> tuple[str, list[dict[str, Any]]]:
        if plan.get("contract_version") != 2:
            raise AuthorityError("contract_version must be 2")
        graph_id = plan.get("graph_id")
        if not isinstance(graph_id, str) or not graph_id.strip():
            raise AuthorityError("graph_id is required")
        _valid_sha256(policy_hash, "policy_hash")
        nodes = plan.get("nodes")
        if not isinstance(nodes, list) or not nodes:
            raise AuthorityError("plan nodes must be a non-empty list")
        normalized: list[dict[str, Any]] = []
        seen: set[str] = set()
        seen_ikeys: set[str] = set()
        for raw in nodes:
            if not isinstance(raw, Mapping):
                raise AuthorityError("each plan node must be an object")
            node = json.loads(_sealed_json(raw))
            reserved = {
                "envelope_version",
                "graph_id",
                "attempt_id",
                "planner_hash",
                "policy_hash",
                "lease",
                "fence",
            }.intersection(node)
            if reserved:
                raise AuthorityError(
                    "plan node contains authority-owned fields: "
                    + ", ".join(sorted(reserved))
                )
            node_id = node.get("node_id")
            if not isinstance(node_id, str) or not node_id.strip():
                raise AuthorityError("node_id is required")
            if node_id in seen:
                raise AuthorityError(f"duplicate node_id: {node_id}")
            seen.add(node_id)
            ikey = node.get("idempotency_key")
            if isinstance(ikey, str) and ikey in seen_ikeys:
                raise AuthorityError(f"duplicate idempotency_key across nodes: {ikey}")
            if isinstance(ikey, str):
                seen_ikeys.add(ikey)
            surface = node.get("execution_surface")
            if surface not in _ALLOWED_SURFACES:
                raise AuthorityError(f"invalid execution_surface for {node_id}")
            for key in _HASH_KEYS:
                _valid_sha256(node.get(key), f"{key} for {node_id}")
            candidate_envelope = {
                "envelope_version": 2,
                "graph_id": graph_id,
                **node,
                "attempt_id": "seal-validation",
                "planner_hash": "0" * 64,
                "policy_hash": policy_hash,
                "lease": {
                    "holder": "seal-validation",
                    "granted_utc": "1970-01-01T00:00:00+00:00",
                    "ttl_s": 1,
                    "renewable": True,
                },
                "fence": 1,
            }
            errors = validate_sealed_envelope(candidate_envelope)
            if errors:
                raise AuthorityError(
                    f"invalid sealed plan node {node_id}: " + ", ".join(errors)
                )
            normalized.append(node)
        return graph_id, normalized

    def seal_plan(
        self,
        plan: Mapping[str, Any],
        *,
        policy_hash: str,
        now: datetime | None = None,
    ) -> str:
        """Persist an immutable plan and its nodes, returning the planner hash."""

        graph_id, nodes = self._validate_plan(plan, policy_hash)
        sealed_at = _utc(now)
        canonical_plan = json.loads(_sealed_json(plan))
        planner_hash = _canonical_hash(canonical_plan)
        with _DB_LOCK:
            conn = self._connect()
            try:
                conn.execute("BEGIN IMMEDIATE")
                if conn.execute(
                    "SELECT 1 FROM sealed_plans WHERE graph_id = ?", (graph_id,)
                ).fetchone():
                    raise AuthorityError(f"graph {graph_id} is already sealed")
                conn.execute(
                    """INSERT INTO sealed_plans(
                           graph_id, contract_version, planner_hash, policy_hash,
                           plan_json, sealed_utc
                       ) VALUES (?, 2, ?, ?, ?, ?)""",
                    (
                        graph_id,
                        planner_hash,
                        policy_hash,
                        _sealed_json(canonical_plan),
                        _iso(sealed_at),
                    ),
                )
                conn.executemany(
                    "INSERT INTO sealed_nodes(graph_id, node_id, node_json) VALUES (?, ?, ?)",
                    [(graph_id, node["node_id"], _sealed_json(node)) for node in nodes],
                )
                self._append_event(
                    conn,
                    kind="PLAN_SEALED",
                    graph_id=graph_id,
                    node_id=None,
                    attempt_id=None,
                    fence=None,
                    payload={"planner_hash": planner_hash, "policy_hash": policy_hash},
                    now=sealed_at,
                )
                conn.execute("COMMIT")
                return planner_hash
            except Exception:
                try:
                    conn.execute("ROLLBACK")
                except Exception:
                    pass
                raise
            finally:
                conn.close()

    @staticmethod
    def _load_plan_node(
        conn: sqlite3.Connection, graph_id: str, node_id: str
    ) -> tuple[sqlite3.Row, dict[str, Any]]:
        plan = conn.execute(
            "SELECT * FROM sealed_plans WHERE graph_id = ?", (graph_id,)
        ).fetchone()
        row = conn.execute(
            "SELECT node_json FROM sealed_nodes WHERE graph_id = ? AND node_id = ?",
            (graph_id, node_id),
        ).fetchone()
        if plan is None or row is None:
            raise AuthorityError("unknown sealed graph node")
        return plan, json.loads(row["node_json"])

    @staticmethod
    def _latest_grant(
        conn: sqlite3.Connection, graph_id: str, node_id: str
    ) -> sqlite3.Row | None:
        return conn.execute(
            """SELECT * FROM authority_events
               WHERE graph_id = ? AND node_id = ? AND kind = 'LEASE_GRANTED'
               ORDER BY id DESC LIMIT 1""",
            (graph_id, node_id),
        ).fetchone()

    @classmethod
    def _node_authority(
        cls, conn: sqlite3.Connection, graph_id: str, node_id: str
    ) -> dict[str, Any] | None:
        """Fold the current fence's lifecycle events into one authority view."""

        grant = cls._latest_grant(conn, graph_id, node_id)
        if grant is None:
            return None
        fence = int(grant["fence"])
        envelope = json.loads(grant["payload_json"])["envelope"]
        rows = conn.execute(
            """SELECT kind, payload_json FROM authority_events
               WHERE graph_id = ? AND node_id = ? AND fence = ?
                 AND kind IN ('LEASE_RENEWED', 'LEASE_REVOKED',
                              'RESULT_ACCEPTED', 'NODE_COMPLETED')
               ORDER BY id""",
            (graph_id, node_id, fence),
        ).fetchall()
        revoked = False
        completed = False
        accepted_result_hash: str | None = None
        new_expires_utc: str | None = None
        for row in rows:
            payload = json.loads(row["payload_json"])
            if row["kind"] == "LEASE_RENEWED":
                new_expires_utc = payload["new_expires_utc"]
            elif row["kind"] == "LEASE_REVOKED":
                revoked = True
            elif row["kind"] in _ACCEPTED_KINDS:
                completed = True
                accepted_result_hash = payload.get("result_hash")
        base_expiry = cls._lease_expiry(envelope)
        effective_expiry = (
            _parse_time(new_expires_utc) if new_expires_utc is not None else base_expiry
        )
        return {
            "fence": fence,
            "envelope": envelope,
            "granted_expires": base_expiry,
            "effective_expires": effective_expiry,
            "revoked": revoked,
            "completed": completed,
            "accepted_result_hash": accepted_result_hash,
        }

    @staticmethod
    def _lease_expiry(envelope: Mapping[str, Any]) -> datetime:
        lease = envelope["lease"]
        return _expiry_from_ttl(
            _parse_time(lease["granted_utc"]),
            lease["ttl_s"],
        )

    @staticmethod
    def _terminal_completed(
        conn: sqlite3.Connection, graph_id: str, node_id: str
    ) -> bool:
        """True if the node has any terminal acceptance across ALL fences."""

        return (
            conn.execute(
                """SELECT 1 FROM authority_events
                   WHERE graph_id = ? AND node_id = ?
                     AND kind IN ('RESULT_ACCEPTED', 'NODE_COMPLETED')
                   LIMIT 1""",
                (graph_id, node_id),
            ).fetchone()
            is not None
        )

    @staticmethod
    def _attempt_seen(conn: sqlite3.Connection, attempt_id: str) -> bool:
        """True if the attempt_id was ever granted anywhere in this authority store."""

        return (
            conn.execute(
                """SELECT 1 FROM authority_events
                   WHERE kind = 'LEASE_GRANTED' AND attempt_id = ?
                   LIMIT 1""",
                (attempt_id,),
            ).fetchone()
            is not None
        )

    def grant_node(
        self,
        graph_id: str,
        node_id: str,
        *,
        holder: str,
        ttl_s: float,
        attempt_id: str,
        now: datetime | None = None,
    ) -> dict[str, Any]:
        """Atomically grant the next monotonic fence and return its sealed envelope.

        A node closed by terminal completion can never be regranted. Attempt IDs
        are single-use across the whole graph/node history and are never reused,
        including after revocation or natural expiry.
        """

        granted_at = _utc(now)
        if not isinstance(holder, str) or not holder.strip():
            raise AuthorityError("lease holder is required")
        if not isinstance(attempt_id, str) or not attempt_id.strip():
            raise AuthorityError("attempt_id is required")
        ttl = _nonnegative_number(ttl_s, "ttl_s")
        if ttl == 0:
            raise AuthorityError("ttl_s must be greater than zero")
        with _DB_LOCK:
            conn = self._connect()
            try:
                conn.execute("BEGIN IMMEDIATE")
                plan, node = self._load_plan_node(conn, graph_id, node_id)
                deadline = _parse_time(node["deadline_utc"])
                expires = _expiry_from_ttl(granted_at, ttl)
                if granted_at >= deadline:
                    raise AuthorityError("expired deadline cannot grant")
                if expires >= deadline:
                    raise AuthorityError(
                        "grant expiry must be strictly before the node deadline"
                    )
                if self._terminal_completed(conn, graph_id, node_id):
                    raise AuthorityError("node is terminal and cannot be regranted")
                if self._attempt_seen(conn, attempt_id):
                    raise AuthorityError(f"attempt_id {attempt_id} was already used")
                authority = self._node_authority(conn, graph_id, node_id)
                if authority is not None:
                    if not authority["revoked"] and not authority["completed"]:
                        if granted_at < authority["effective_expires"]:
                            raise AuthorityError("node already has an active lease")
                    fence = authority["fence"] + 1
                else:
                    fence = 1
                envelope = {
                    "envelope_version": 2,
                    "graph_id": graph_id,
                    **node,
                    "attempt_id": attempt_id,
                    "planner_hash": plan["planner_hash"],
                    "policy_hash": plan["policy_hash"],
                    "lease": {
                        "holder": holder,
                        "granted_utc": _iso(granted_at),
                        "ttl_s": ttl,
                        "renewable": True,
                    },
                    "fence": fence,
                }
                errors = validate_sealed_envelope(envelope)
                if errors:
                    raise AuthorityError(
                        "invalid sealed node envelope: " + ", ".join(errors)
                    )
                self._append_event(
                    conn,
                    kind="LEASE_GRANTED",
                    graph_id=graph_id,
                    node_id=node_id,
                    attempt_id=attempt_id,
                    fence=fence,
                    payload={"envelope": envelope},
                    now=granted_at,
                )
                conn.execute("COMMIT")
                return envelope
            except Exception:
                try:
                    conn.execute("ROLLBACK")
                except Exception:
                    pass
                raise
            finally:
                conn.close()

    def _authority_gate(
        self,
        conn: sqlite3.Connection,
        envelope: Mapping[str, Any],
        *,
        action: str,
    ) -> tuple[str, str, dict[str, Any]]:
        """Shared exact-authority gate binding holder and canonical envelope identity."""

        graph_id = str(envelope.get("graph_id"))
        node_id = str(envelope.get("node_id"))
        authority = self._node_authority(conn, graph_id, node_id)
        if authority is None or envelope.get("fence") != authority["fence"]:
            raise AuthorityError(f"stale fence cannot {action}")
        authoritative = authority["envelope"]
        if authoritative.get("attempt_id") != envelope.get("attempt_id"):
            raise AuthorityError(f"stale attempt cannot {action}")
        env_lease = envelope.get("lease")
        if not isinstance(env_lease, Mapping) or env_lease.get(
            "holder"
        ) != authoritative["lease"].get("holder"):
            raise AuthorityError(f"lease holder cannot {action}")
        if _canonical_identity(envelope) != _canonical_json(authoritative):
            raise AuthorityError(f"non-authoritative envelope cannot {action}")
        return graph_id, node_id, authority

    def _load_live_authority(
        self,
        conn: sqlite3.Connection,
        envelope: Mapping[str, Any],
        current: datetime,
        *,
        action: str,
    ) -> tuple[str, str, int, dict[str, Any]]:
        """Return the authoritative envelope for a lifecycle transition or fail."""

        graph_id, node_id, authority = self._authority_gate(
            conn, envelope, action=action
        )
        if authority["completed"]:
            raise AuthorityError("node already completed")
        if authority["revoked"]:
            raise AuthorityError(f"revoked lease cannot {action}")
        if current >= authority["effective_expires"]:
            raise AuthorityError(f"expired lease cannot {action}")
        if current >= _parse_time(authority["envelope"]["deadline_utc"]):
            raise AuthorityError(f"expired deadline cannot {action}")
        return graph_id, node_id, authority["fence"], authority["envelope"]

    def renew_node(
        self,
        envelope: Mapping[str, Any],
        *,
        ttl_s: float,
        now: datetime | None = None,
    ) -> dict[str, Any]:
        """Atomically extend the active lease via external bounded metadata.

        The sealed ``LEASE_GRANTED`` envelope is immutable forever. ``ttl_s`` is a
        duration measured from the renewal moment, and the new effective expiry
        must not reach or exceed the node deadline. Renewal must occur strictly
        before the current effective expiry. It never revives an expired, revoked,
        or terminal lease.
        """

        renewed_at = _utc(now)
        ttl = _nonnegative_number(ttl_s, "ttl_s")
        if ttl == 0:
            raise AuthorityError("ttl_s must be greater than zero")
        with _DB_LOCK:
            conn = self._connect()
            try:
                conn.execute("BEGIN IMMEDIATE")
                graph_id, node_id, authority = self._authority_gate(
                    conn, envelope, action="renew"
                )
                if authority["completed"]:
                    raise AuthorityError("node already completed")
                if authority["revoked"]:
                    raise AuthorityError("revoked lease cannot renew")
                authoritative = authority["envelope"]
                deadline = _parse_time(authoritative["deadline_utc"])
                new_expires = _expiry_from_ttl(renewed_at, ttl)
                if new_expires <= authority["effective_expires"]:
                    raise AuthorityError(
                        "renewal must extend the effective lease expiry"
                    )
                if new_expires >= deadline:
                    raise AuthorityError(
                        "renewal effective expiry must not exceed the node deadline"
                    )
                if renewed_at >= authority["effective_expires"]:
                    raise AuthorityError("expired lease cannot renew")
                if renewed_at >= deadline:
                    raise AuthorityError("expired deadline cannot renew")
                fence = authority["fence"]
                self._append_event(
                    conn,
                    kind="LEASE_RENEWED",
                    graph_id=graph_id,
                    node_id=node_id,
                    attempt_id=str(envelope["attempt_id"]),
                    fence=fence,
                    payload={
                        "holder": authoritative["lease"]["holder"],
                        "prior_expires_utc": _iso(authority["effective_expires"]),
                        "new_expires_utc": _iso(new_expires),
                    },
                    now=renewed_at,
                )
                conn.execute("COMMIT")
                return {
                    "graph_id": graph_id,
                    "node_id": node_id,
                    "attempt_id": str(envelope["attempt_id"]),
                    "fence": fence,
                    "holder": authoritative["lease"]["holder"],
                    "envelope": authoritative,
                    "prior_expires_utc": _iso(authority["effective_expires"]),
                    "new_expires_utc": _iso(new_expires),
                    "effective_expires_utc": _iso(new_expires),
                }
            except Exception:
                try:
                    conn.execute("ROLLBACK")
                except Exception:
                    pass
                raise
            finally:
                conn.close()

    def revoke_node(
        self, envelope: Mapping[str, Any], *, now: datetime | None = None
    ) -> dict[str, Any]:
        """Cancel the current lease so the node can be immediately regranted.

        Uses the shared exact authority gate: the supplied envelope must be the
        byte-identical authoritative sealed envelope. After revocation the old
        envelope validates ``lease_revoked``/``stale_fence`` and can never
        authorize execution.
        """

        revoked_at = _utc(now)
        with _DB_LOCK:
            conn = self._connect()
            try:
                conn.execute("BEGIN IMMEDIATE")
                graph_id, node_id, authority = self._authority_gate(
                    conn, envelope, action="revoke"
                )
                if authority["completed"]:
                    raise AuthorityError("node already completed")
                if authority["revoked"]:
                    raise AuthorityError("lease already revoked")
                authoritative = authority["envelope"]
                self._append_event(
                    conn,
                    kind="LEASE_REVOKED",
                    graph_id=graph_id,
                    node_id=node_id,
                    attempt_id=str(authoritative["attempt_id"]),
                    fence=authority["fence"],
                    payload={"holder": authoritative["lease"]["holder"]},
                    now=revoked_at,
                )
                conn.execute("COMMIT")
                return {
                    "graph_id": graph_id,
                    "node_id": node_id,
                    "attempt_id": str(authoritative["attempt_id"]),
                    "fence": authority["fence"],
                }
            except Exception:
                try:
                    conn.execute("ROLLBACK")
                except Exception:
                    pass
                raise
            finally:
                conn.close()

    def complete_node(
        self,
        envelope: Mapping[str, Any],
        *,
        result_hash: str,
        now: datetime | None = None,
    ) -> dict[str, Any]:
        """Accept the current unexpired holder/attempt/fence exactly once.

        Result acceptance requires the caller's ``result_hash`` (a SHA-256 over
        the canonical result; raw results are never stored). A redelivery of the
        SAME accepted ``result_hash`` is an idempotent lookup. A stale, expired,
        revoked, superseded, conflicting-duplicate, or forged completion appends
        exactly one typed ``RESULT_REJECTED`` event, then raises
        ``ResultRejection``. A completion whose ``fence`` is not a bindable
        integer raises ``MalformedEnvelopeFence``.
        """

        result_hash = _valid_sha256(result_hash, "result_hash")
        completed_at = _utc(now)
        with _DB_LOCK:
            conn = self._connect()
            try:
                conn.execute("BEGIN IMMEDIATE")
                try:
                    record = self._accept_result(
                        conn, envelope, result_hash, completed_at
                    )
                except _Reject as reject:
                    fence_audit = self._append_rejection(
                        conn,
                        envelope,
                        reject.reason,
                        result_hash,
                        completed_at,
                        fence_hint=reject.fence_hint,
                    )
                    conn.execute("COMMIT")
                    if fence_audit["claimed_fence_malformed"]:
                        raise MalformedEnvelopeFence(
                            reject.reason,
                            result_hash=result_hash,
                            claimed_fence_reason=fence_audit["claimed_fence_reason"],
                            claimed_fence_type=fence_audit["claimed_fence_type"],
                            claimed_fence_repr=fence_audit["claimed_fence_repr"],
                        ) from None
                    raise ResultRejection(
                        reject.reason, result_hash=result_hash
                    ) from None
                conn.execute("COMMIT")
                return record
            except Exception:
                try:
                    conn.execute("ROLLBACK")
                except Exception:
                    pass
                raise
            finally:
                conn.close()

    def _accept_result(
        self,
        conn: sqlite3.Connection,
        envelope: Mapping[str, Any],
        result_hash: str,
        current: datetime,
    ) -> dict[str, Any]:
        """Fold the acceptance decision; append RESULT_ACCEPTED or raise _Reject."""

        graph_id = str(envelope.get("graph_id"))
        node_id = str(envelope.get("node_id"))
        self._load_plan_node(conn, graph_id, node_id)  # unknown node -> AuthorityError
        authority = self._node_authority(conn, graph_id, node_id)

        def reject(reason: str, *, fence_hint: int | None = None) -> None:
            raise _Reject(reason, fence_hint=fence_hint)

        if authority is None:
            reject("no_grant")
        assert authority is not None
        _, fence_audit = _normalize_claimed_fence(envelope.get("fence"))
        if fence_audit["claimed_fence_malformed"]:
            reject("malformed_fence", fence_hint=authority["fence"])
        if envelope.get("fence") != authority["fence"]:
            reject("stale_fence", fence_hint=authority["fence"])
        authoritative = authority["envelope"]
        if authoritative.get("attempt_id") != envelope.get("attempt_id"):
            reject("stale_attempt", fence_hint=authority["fence"])
        env_lease = envelope.get("lease")
        if not isinstance(env_lease, Mapping) or env_lease.get(
            "holder"
        ) != authoritative["lease"].get("holder"):
            reject("holder_mismatch", fence_hint=authority["fence"])
        if _canonical_identity(envelope) != _canonical_json(authoritative):
            reject("envelope_not_authoritative", fence_hint=authority["fence"])
        if authority["completed"]:
            if authority["accepted_result_hash"] == result_hash:
                return {
                    "graph_id": graph_id,
                    "node_id": node_id,
                    "attempt_id": str(authoritative["attempt_id"]),
                    "fence": authority["fence"],
                    "holder": authoritative["lease"]["holder"],
                    "result_hash": result_hash,
                    "idempotent": True,
                }
            reject("node_completed", fence_hint=authority["fence"])
        if authority["revoked"]:
            reject("lease_revoked", fence_hint=authority["fence"])
        if current >= authority["effective_expires"]:
            reject("lease_expired", fence_hint=authority["fence"])
        if current >= _parse_time(authoritative["deadline_utc"]):
            reject("deadline_expired", fence_hint=authority["fence"])

        self._append_event(
            conn,
            kind="RESULT_ACCEPTED",
            graph_id=graph_id,
            node_id=node_id,
            attempt_id=str(authoritative["attempt_id"]),
            fence=authority["fence"],
            payload={
                "holder": authoritative["lease"]["holder"],
                "result_hash": result_hash,
            },
            now=current,
        )
        return {
            "graph_id": graph_id,
            "node_id": node_id,
            "attempt_id": str(authoritative["attempt_id"]),
            "fence": authority["fence"],
            "holder": authoritative["lease"]["holder"],
            "result_hash": result_hash,
        }

    def _append_rejection(
        self,
        conn: sqlite3.Connection,
        envelope: Mapping[str, Any],
        reason: str,
        result_hash: str | None,
        now: datetime,
        *,
        fence_hint: int | None = None,
    ) -> dict[str, Any]:
        """Append one typed RESULT_REJECTED audit event in the decision transaction."""

        claimed_fence, fence_audit = _normalize_claimed_fence(envelope.get("fence"))
        self._append_event(
            conn,
            kind="RESULT_REJECTED",
            graph_id=str(envelope.get("graph_id")),
            node_id=str(envelope.get("node_id")),
            attempt_id=str(envelope.get("attempt_id"))
            if envelope.get("attempt_id")
            else None,
            fence=claimed_fence,
            payload={
                "reason": reason,
                "result_hash": result_hash,
                "current_fence": fence_hint,
                **fence_audit,
            },
            now=now,
        )
        return fence_audit

    def current_fence(self, graph_id: str, node_id: str) -> int | None:
        conn = self._connect()
        try:
            row = self._latest_grant(conn, graph_id, node_id)
            return int(row["fence"]) if row is not None else None
        finally:
            conn.close()

    def current_authority(self, graph_id: str, node_id: str) -> dict[str, Any] | None:
        """Return the folded current authority view for a node, or ``None``."""

        conn = self._connect()
        try:
            authority = self._node_authority(conn, graph_id, node_id)
            if authority is None:
                return None
            return {
                "fence": authority["fence"],
                "envelope": authority["envelope"],
                "granted_expires_utc": _iso(authority["granted_expires"]),
                "effective_expires_utc": _iso(authority["effective_expires"]),
                "revoked": authority["revoked"],
                "completed": authority["completed"],
                "accepted_result_hash": authority["accepted_result_hash"],
            }
        finally:
            conn.close()

    @staticmethod
    def _validate_current_snapshot(
        conn: sqlite3.Connection,
        envelope: Mapping[str, Any],
        current: datetime,
    ) -> tuple[list[str], int | None]:
        """Validate and return the matching fence from one SQLite snapshot."""

        errors = list(validate_sealed_envelope(envelope))
        try:
            plan, _ = Phase2AuthorityStore._load_plan_node(
                conn, str(envelope.get("graph_id")), str(envelope.get("node_id"))
            )
        except AuthorityError:
            return sorted(set(errors + ["unknown_graph_node"])), None
        authority = Phase2AuthorityStore._node_authority(
            conn, str(envelope.get("graph_id")), str(envelope.get("node_id"))
        )
        if authority is None or envelope.get("fence") != authority["fence"]:
            errors.append("stale_fence")
        if authority is not None:
            try:
                authoritative = authority["envelope"]
                if _canonical_identity(envelope) != _canonical_json(authoritative):
                    errors.append("envelope_not_authoritative")
            except (KeyError, TypeError, ValueError, RecursionError):
                errors.append("authoritative_envelope_invalid")
            if authority["revoked"]:
                errors.append("lease_revoked")
            if authority["completed"]:
                errors.append("node_terminal")
            if current >= authority["effective_expires"]:
                errors.append("lease_expired")
            if current >= _parse_time(authority["envelope"]["deadline_utc"]):
                errors.append("deadline_expired")
        else:
            lease = envelope.get("lease")
            try:
                if not isinstance(lease, Mapping):
                    raise AuthorityError("lease must be an object")
                granted = _parse_time(lease.get("granted_utc"))
                expires = _expiry_from_ttl(granted, lease.get("ttl_s"))
                if current >= expires:
                    errors.append("lease_expired")
                if current >= _parse_time(envelope.get("deadline_utc")):
                    errors.append("deadline_expired")
            except (AuthorityError, TypeError, ValueError):
                errors.append("lease_or_deadline_invalid")
        if envelope.get("planner_hash") != plan["planner_hash"]:
            errors.append("planner_hash")
        if envelope.get("policy_hash") != plan["policy_hash"]:
            errors.append("policy_hash")
        fence = int(authority["fence"]) if authority is not None else None
        return sorted(set(errors)), fence

    def validate_current(
        self, envelope: Mapping[str, Any], *, now: datetime | None = None
    ) -> list[str]:
        current = _utc(now)
        conn = self._connect()
        try:
            conn.execute("BEGIN")
            errors, _ = self._validate_current_snapshot(conn, envelope, current)
            conn.execute("COMMIT")
            return errors
        except Exception:
            try:
                conn.execute("ROLLBACK")
            except Exception:
                pass
            raise
        finally:
            conn.close()

    @contextmanager
    def bind_current(
        self, envelope: Mapping[str, Any], *, now: datetime | None = None
    ) -> Iterator[None]:
        """Bind authority valid at one atomic durable snapshot.

        The envelope and fence are read and validated in the same SQLite read
        transaction. Later revocation still requires consumers to recheck before
        each effect; this context only attests to authority at bind time.
        """

        try:
            snapshot = _json_tree(envelope)
        except (TypeError, ValueError, RecursionError) as exc:
            raise AuthorityError("envelope must be JSON serializable") from exc
        current = _utc(now)
        conn = self._connect()
        try:
            conn.execute("BEGIN")
            errors, fence = self._validate_current_snapshot(conn, snapshot, current)
            conn.execute("COMMIT")
        except Exception:
            try:
                conn.execute("ROLLBACK")
            except Exception:
                pass
            raise
        finally:
            conn.close()
        if errors:
            raise AuthorityError("invalid current node authority: " + ", ".join(errors))
        if fence is None:
            raise AuthorityError("missing authoritative current fence")
        with bind_sealed_envelope(snapshot, current_fence=fence):
            yield

    def recover(self, *, now: datetime | None = None) -> dict[str, int]:
        """Verify the durable chain and poison expired, unreconciled reservations.

        Verifies every lifecycle event's hash chain, folds lifecycle state, and
        reconciles orphaned reservations with ``actual_usd=None`` once their
        owning lease is dead. Any malformed payload or hash-chain tamper fails
        closed.
        """

        current = _utc(now)
        reconciled_count = 0
        with _DB_LOCK:
            conn = self._connect()
            try:
                conn.execute("BEGIN IMMEDIATE")
                rows = conn.execute(
                    "SELECT * FROM authority_events ORDER BY id"
                ).fetchall()
                previous_hash: str | None = None
                reservations: dict[str, sqlite3.Row] = {}
                reconciled: set[str] = set()
                grants: dict[tuple[str, str, int], dict[str, Any]] = {}
                renewals: dict[tuple[str, str, int], dict[str, Any]] = {}
                dead_leases: set[tuple[str, str, int]] = set()
                for row in rows:
                    try:
                        payload = json.loads(row["payload_json"])
                    except (TypeError, ValueError) as exc:
                        raise AuthorityError(
                            "authority event hash chain is invalid"
                        ) from exc
                    expected = _canonical_hash({
                        "event_id": row["event_id"],
                        "ts_utc": row["ts_utc"],
                        "kind": row["kind"],
                        "graph_id": row["graph_id"],
                        "node_id": row["node_id"],
                        "attempt_id": row["attempt_id"],
                        "fence": row["fence"],
                        "payload": payload,
                        "prev_event_hash": row["prev_event_hash"],
                    })
                    if (
                        row["prev_event_hash"] != previous_hash
                        or row["event_hash"] != expected
                    ):
                        raise AuthorityError("authority event hash chain is invalid")
                    previous_hash = row["event_hash"]
                    if row["fence"] is None:
                        continue
                    key = (row["graph_id"], row["node_id"], int(row["fence"]))
                    if row["kind"] == "LEASE_GRANTED":
                        grants[key] = payload
                    elif row["kind"] == "LEASE_RENEWED":
                        renewals[key] = payload
                    elif row["kind"] in ("LEASE_REVOKED",) + _ACCEPTED_KINDS:
                        dead_leases.add(key)
                    elif row["kind"] == "BUDGET_RESERVED":
                        reservations[row["event_id"]] = row
                    elif row["kind"] == "BUDGET_RECONCILED":
                        reconciled.add(str(payload.get("reservation_id")))

                for reservation_id, reservation in reservations.items():
                    if reservation_id in reconciled:
                        continue
                    key = (
                        reservation["graph_id"],
                        reservation["node_id"],
                        int(reservation["fence"]),
                    )
                    grant = grants.get(key)
                    if grant is None:
                        raise AuthorityError(
                            "orphaned reservation has no authoritative lease"
                        )
                    if key not in dead_leases:
                        try:
                            envelope = grant["envelope"]
                            lease = envelope["lease"]
                            base_expires = _expiry_from_ttl(
                                _parse_time(lease["granted_utc"]),
                                lease["ttl_s"],
                            )
                            renewal = renewals.get(key)
                            expires = (
                                _parse_time(renewal["new_expires_utc"])
                                if renewal is not None
                                else base_expires
                            )
                        except (KeyError, TypeError, ValueError, AuthorityError) as exc:
                            raise AuthorityError(
                                "orphaned reservation lease is invalid"
                            ) from exc
                        if current < expires:
                            continue
                    self._append_event(
                        conn,
                        kind="BUDGET_RECONCILED",
                        graph_id=reservation["graph_id"],
                        node_id=reservation["node_id"],
                        attempt_id=reservation["attempt_id"],
                        fence=reservation["fence"],
                        payload={
                            "reservation_id": reservation_id,
                            "actual_tokens": 0,
                            "actual_usd": None,
                            "recovery": "expired_lease_unknown_spend",
                        },
                        now=current,
                    )
                    reconciled_count += 1
                conn.execute("COMMIT")
            except Exception:
                try:
                    conn.execute("ROLLBACK")
                except Exception:
                    pass
                raise
            finally:
                conn.close()
        return {"orphaned_reservations_reconciled": reconciled_count}

    def migration_report(self) -> dict[str, Any]:
        """Read-only compatibility diagnostic; never writes, creates, or deletes."""

        report: dict[str, Any] = {
            "db_path": str(self.db_path),
            "exists": self.db_path.exists(),
            "compatible": True,
            "missing_indexes": [],
            "violations": [],
        }
        if not self.db_path.exists():
            return report
        conn = _open_phase2_sqlite(self.db_path, readonly=True)
        try:
            table = conn.execute(
                "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'authority_events'"
            ).fetchone()
            missing = self._missing_integrity_indexes(conn)
            report["missing_indexes"] = [index.name for index in missing]
            if table is None:
                return report
            violations = self._scan_integrity_violations(conn, missing)
            report["violations"] = violations
            report["compatible"] = not violations
            return report
        finally:
            conn.close()

    def get_planner_hash(self, graph_id: str) -> str | None:
        conn = self._connect()
        try:
            row = conn.execute(
                "SELECT planner_hash FROM sealed_plans WHERE graph_id = ?", (graph_id,)
            ).fetchone()
            return row["planner_hash"] if row else None
        finally:
            conn.close()

    def read_events(self, graph_id: str) -> list[dict[str, Any]]:
        if not self.db_path.exists():
            return []
        conn = _open_phase2_sqlite(self.db_path, readonly=True)
        conn.row_factory = sqlite3.Row
        try:
            rows = conn.execute(
                "SELECT * FROM authority_events WHERE graph_id = ? ORDER BY id",
                (graph_id,),
            )
            result = []
            for row in rows:
                record = dict(row)
                try:
                    record["payload"] = json.loads(record["payload_json"])
                except (TypeError, ValueError, KeyError):
                    record["payload"] = None
                result.append(record)
            return result
        finally:
            conn.close()
