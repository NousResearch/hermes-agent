"""Durable lifecycle records for Phase 2 mutating logical operations.

The idempotency key identifies one logical effect. Attempts and fences are
recorded separately so a crash cannot turn an in-progress claim into evidence
that no effect occurred. Same-intent replay returns the durable current state;
a different intent under the same key is rejected.

This store is idempotency evidence, never authorization or containment.
Runtime integrations must still revalidate live authority and resource fencing
at the effect boundary. Reconciliation is read-only; compensation uses a new
logical operation. Raw arguments are never stored. The deterministic SHA-256
hash permits offline confirmation of guessed arguments and is not a
confidentiality boundary.

For Axl Ibiza (andrexibiza)'s replay/refusal design and September 3 corrective
review, see the shared "Design provenance" note in
:mod:`agent.phase2_authority`. Implementation authorship is separate.
"""

from __future__ import annotations

import hashlib
import json
import re
import sqlite3
import threading
from contextlib import suppress
from dataclasses import dataclass
from datetime import datetime, timezone, UTC
from enum import Enum
from pathlib import Path
from typing import Any, Mapping

from agent.phase2_sqlite import _check_phase2_db_files, _open_phase2_sqlite
from hermes_constants import get_hermes_home

_DB_LOCK = threading.Lock()
_HASH_RE = re.compile(r"^[0-9a-f]{64}$")
_SQLITE_INT_MAX = 2**63 - 1


class MutationClaimError(ValueError):
    """The caller supplied an invalid mutation lifecycle transition."""


class MutationState(str, Enum):
    """Durable state of the latest admitted attempt for a logical operation."""

    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED_PROVEN_NO_EFFECT = "failed_proven_no_effect"
    CANCELLED_BEFORE_EFFECT = "cancelled_before_effect"
    INDETERMINATE = "indeterminate"


_TERMINAL_STATES = frozenset({
    MutationState.COMPLETED,
    MutationState.FAILED_PROVEN_NO_EFFECT,
    MutationState.CANCELLED_BEFORE_EFFECT,
    MutationState.INDETERMINATE,
})
_RETRYABLE_STATES = frozenset({
    MutationState.FAILED_PROVEN_NO_EFFECT,
    MutationState.CANCELLED_BEFORE_EFFECT,
})


@dataclass(frozen=True)
class MutationClaimResult:
    """The durable attempt selected for a claim request."""

    idempotency_key: str
    attempt_id: str
    fence: int
    state: MutationState
    created: bool


def _required_text(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise MutationClaimError(f"{field} must be a non-empty string")
    return value


def _required_identity(envelope: Mapping[str, Any], field: str) -> str:
    return _required_text(envelope.get(field), field)


def _claim_fence(envelope: Mapping[str, Any]) -> int:
    value = envelope.get("fence")
    if (
        not isinstance(value, int)
        or isinstance(value, bool)
        or not 1 <= value <= _SQLITE_INT_MAX
    ):
        raise MutationClaimError("fence must be a positive signed 64-bit integer")
    return value


def default_db_path() -> Path:
    return get_hermes_home() / "phase2_idempotency.db"


def _canonical_hash(value: Mapping[str, Any]) -> str:
    if not isinstance(value, Mapping):
        raise MutationClaimError("args must be a mapping")

    def json_tree(item: Any) -> Any:
        if isinstance(item, Mapping):
            normalized: dict[str, Any] = {}
            for key, child in item.items():
                if not isinstance(key, str):
                    raise TypeError("JSON object keys must be strings")
                normalized[key] = json_tree(child)
            return normalized
        if isinstance(item, (list, tuple)):
            return [json_tree(child) for child in item]
        if item is None or isinstance(item, (str, bool, int, float)):
            return item
        raise TypeError(f"value of type {type(item).__name__} is not JSON serializable")

    try:
        encoded = json.dumps(
            json_tree(value),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError, RecursionError) as exc:
        raise MutationClaimError("args must be canonically JSON serializable") from exc
    return hashlib.sha256(encoded).hexdigest()


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


class MutationClaimStore:
    """Own durable logical-operation, attempt, settlement, and readmission rows."""

    def __init__(self, db_path: Path | str | None = None) -> None:
        self.db_path = (Path(db_path) if db_path is not None else default_db_path()).expanduser()

    def _connect(self) -> sqlite3.Connection:
        conn = _open_phase2_sqlite(self.db_path)
        try:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA busy_timeout=5000")
            conn.row_factory = sqlite3.Row
            legacy = conn.execute(
                "SELECT 1 FROM sqlite_master WHERE type = 'table' "
                "AND name = 'mutation_claims'"
            ).fetchone()
            if legacy is not None:
                raise MutationClaimError(
                    "legacy mutation_claims schema requires explicit migration"
                )
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS mutation_operations (
                    idempotency_key TEXT PRIMARY KEY,
                    graph_id TEXT NOT NULL,
                    node_id TEXT NOT NULL,
                    tool_name TEXT NOT NULL,
                    args_hash TEXT NOT NULL,
                    created_utc TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS mutation_attempts (
                    idempotency_key TEXT NOT NULL,
                    attempt_id TEXT NOT NULL,
                    fence INTEGER NOT NULL,
                    claimed_utc TEXT NOT NULL,
                    PRIMARY KEY(idempotency_key, attempt_id),
                    UNIQUE(idempotency_key, fence),
                    FOREIGN KEY(idempotency_key)
                        REFERENCES mutation_operations(idempotency_key)
                );
                CREATE TABLE IF NOT EXISTS mutation_settlements (
                    idempotency_key TEXT NOT NULL,
                    attempt_id TEXT NOT NULL,
                    state TEXT NOT NULL CHECK(state IN (
                        'completed',
                        'failed_proven_no_effect',
                        'cancelled_before_effect',
                        'indeterminate'
                    )),
                    outcome_hash TEXT,
                    settled_utc TEXT NOT NULL,
                    PRIMARY KEY(idempotency_key, attempt_id),
                    FOREIGN KEY(idempotency_key, attempt_id)
                        REFERENCES mutation_attempts(idempotency_key, attempt_id)
                );
                CREATE TABLE IF NOT EXISTS mutation_readmissions (
                    idempotency_key TEXT NOT NULL,
                    readmission_id TEXT NOT NULL,
                    prior_attempt_id TEXT NOT NULL,
                    reason TEXT NOT NULL,
                    recorded_utc TEXT NOT NULL,
                    PRIMARY KEY(idempotency_key, readmission_id),
                    FOREIGN KEY(idempotency_key, prior_attempt_id)
                        REFERENCES mutation_attempts(idempotency_key, attempt_id)
                );
                CREATE TABLE IF NOT EXISTS mutation_readmission_uses (
                    idempotency_key TEXT NOT NULL,
                    readmission_id TEXT NOT NULL,
                    attempt_id TEXT NOT NULL,
                    used_utc TEXT NOT NULL,
                    PRIMARY KEY(idempotency_key, readmission_id),
                    UNIQUE(idempotency_key, attempt_id),
                    FOREIGN KEY(idempotency_key, readmission_id)
                        REFERENCES mutation_readmissions(idempotency_key, readmission_id),
                    FOREIGN KEY(idempotency_key, attempt_id)
                        REFERENCES mutation_attempts(idempotency_key, attempt_id)
                );
                """
            )
            for table in (
                "mutation_operations",
                "mutation_attempts",
                "mutation_settlements",
                "mutation_readmissions",
                "mutation_readmission_uses",
            ):
                conn.execute(
                    f"""
                    CREATE TRIGGER IF NOT EXISTS {table}_no_update
                    BEFORE UPDATE ON {table}
                    BEGIN
                        SELECT RAISE(ABORT, '{table} is append-only');
                    END
                    """
                )
                conn.execute(
                    f"""
                    CREATE TRIGGER IF NOT EXISTS {table}_no_delete
                    BEFORE DELETE ON {table}
                    BEGIN
                        SELECT RAISE(ABORT, '{table} is append-only');
                    END
                    """
                )
            _check_phase2_db_files(self.db_path, harden=True)
            return conn
        except BaseException:
            conn.close()
            raise

    @staticmethod
    def _attempt_state(
        conn: sqlite3.Connection, idempotency_key: str, attempt_id: str
    ) -> MutationState:
        row = conn.execute(
            """
            SELECT state FROM mutation_settlements
            WHERE idempotency_key = ? AND attempt_id = ?
            """,
            (idempotency_key, attempt_id),
        ).fetchone()
        return MutationState.IN_PROGRESS if row is None else MutationState(row["state"])

    @classmethod
    def _result(
        cls,
        conn: sqlite3.Connection,
        idempotency_key: str,
        attempt: sqlite3.Row,
        *,
        created: bool,
    ) -> MutationClaimResult:
        return MutationClaimResult(
            idempotency_key=idempotency_key,
            attempt_id=str(attempt["attempt_id"]),
            fence=int(attempt["fence"]),
            state=cls._attempt_state(conn, idempotency_key, str(attempt["attempt_id"])),
            created=created,
        )

    def try_claim(
        self,
        envelope: Mapping[str, Any],
        *,
        tool_name: str,
        args: Mapping[str, Any],
        readmission_id: str | None = None,
    ) -> MutationClaimResult:
        """Claim or replay one logical operation without guessing effect outcome.

        A request suppressed by an in-progress, completed, or indeterminate
        attempt returns that durable attempt and state. It never manufactures a
        new attempt. A later attempt is admitted only after a proven non-effect
        settlement or by consuming a separately recorded readmission.
        """

        if not isinstance(envelope, Mapping):
            raise MutationClaimError("envelope must be a mapping")
        idempotency_key = _required_identity(envelope, "idempotency_key")
        if not _HASH_RE.fullmatch(idempotency_key):
            raise MutationClaimError(
                "idempotency_key must be a lowercase SHA-256 digest"
            )
        tool_name = _required_text(tool_name, "tool_name")
        graph_id = _required_identity(envelope, "graph_id")
        node_id = _required_identity(envelope, "node_id")
        attempt_id = _required_identity(envelope, "attempt_id")
        fence = _claim_fence(envelope)
        args_hash = _canonical_hash(args)
        if readmission_id is not None:
            readmission_id = _required_text(readmission_id, "readmission_id")

        with _DB_LOCK:
            conn = self._connect()
            try:
                conn.execute("BEGIN IMMEDIATE")
                conn.execute(
                    """
                    INSERT INTO mutation_operations(
                        idempotency_key, graph_id, node_id, tool_name,
                        args_hash, created_utc
                    ) VALUES (?, ?, ?, ?, ?, ?)
                    ON CONFLICT(idempotency_key) DO NOTHING
                    """,
                    (
                        idempotency_key,
                        graph_id,
                        node_id,
                        tool_name,
                        args_hash,
                        _utc_now(),
                    ),
                )
                operation = conn.execute(
                    """
                    SELECT graph_id, node_id, tool_name, args_hash
                    FROM mutation_operations WHERE idempotency_key = ?
                    """,
                    (idempotency_key,),
                ).fetchone()
                if operation is None:
                    raise MutationClaimError(
                        f"idempotency_key collision: key {idempotency_key!r} "
                        "conflicted without a readable operation"
                    )
                identity = (graph_id, node_id, tool_name, args_hash)
                prior_identity = tuple(operation[field] for field in (
                    "graph_id",
                    "node_id",
                    "tool_name",
                    "args_hash",
                ))
                if prior_identity != identity:
                    raise MutationClaimError(
                        f"idempotency_key collision: key {idempotency_key!r} "
                        "was already bound to a different logical effect"
                    )

                exact = conn.execute(
                    """
                    SELECT attempt_id, fence FROM mutation_attempts
                    WHERE idempotency_key = ? AND attempt_id = ?
                    """,
                    (idempotency_key, attempt_id),
                ).fetchone()
                if exact is not None:
                    if int(exact["fence"]) != fence:
                        raise MutationClaimError(
                            f"attempt_id {attempt_id!r} cannot be reused at a new fence"
                        )
                    result = self._result(
                        conn, idempotency_key, exact, created=False
                    )
                    conn.execute("COMMIT")
                    return result

                latest = conn.execute(
                    """
                    SELECT attempt_id, fence FROM mutation_attempts
                    WHERE idempotency_key = ?
                    ORDER BY fence DESC LIMIT 1
                    """,
                    (idempotency_key,),
                ).fetchone()
                if latest is not None:
                    if fence <= int(latest["fence"]):
                        raise MutationClaimError(
                            "new attempt fence must be greater than the durable fence"
                        )
                    latest_state = self._attempt_state(
                        conn, idempotency_key, str(latest["attempt_id"])
                    )
                    admitted = latest_state in _RETRYABLE_STATES
                    consume_readmission = False
                    if not admitted and readmission_id is not None:
                        readmission = conn.execute(
                            """
                            SELECT prior_attempt_id FROM mutation_readmissions
                            WHERE idempotency_key = ? AND readmission_id = ?
                            """,
                            (idempotency_key, readmission_id),
                        ).fetchone()
                        if (
                            readmission is None
                            or readmission["prior_attempt_id"] != latest["attempt_id"]
                        ):
                            raise MutationClaimError(
                                "readmission does not authorize the latest attempt"
                            )
                        used = conn.execute(
                            """
                            SELECT 1 FROM mutation_readmission_uses
                            WHERE idempotency_key = ? AND readmission_id = ?
                            """,
                            (idempotency_key, readmission_id),
                        ).fetchone()
                        if used is not None:
                            raise MutationClaimError("readmission was already consumed")
                        admitted = True
                        consume_readmission = True
                    if not admitted:
                        result = self._result(
                            conn, idempotency_key, latest, created=False
                        )
                        conn.execute("COMMIT")
                        return result

                conn.execute(
                    """
                    INSERT INTO mutation_attempts(
                        idempotency_key, attempt_id, fence, claimed_utc
                    ) VALUES (?, ?, ?, ?)
                    """,
                    (idempotency_key, attempt_id, fence, _utc_now()),
                )
                if latest is not None and consume_readmission:
                    conn.execute(
                        """
                        INSERT INTO mutation_readmission_uses(
                            idempotency_key, readmission_id, attempt_id, used_utc
                        ) VALUES (?, ?, ?, ?)
                        """,
                        (idempotency_key, readmission_id, attempt_id, _utc_now()),
                    )
                inserted = conn.execute(
                    """
                    SELECT attempt_id, fence FROM mutation_attempts
                    WHERE idempotency_key = ? AND attempt_id = ?
                    """,
                    (idempotency_key, attempt_id),
                ).fetchone()
                assert inserted is not None
                result = self._result(
                    conn, idempotency_key, inserted, created=True
                )
                conn.execute("COMMIT")
                return result
            except Exception:
                with suppress(Exception):
                    conn.execute("ROLLBACK")
                raise
            finally:
                conn.close()

    def settle(
        self,
        envelope: Mapping[str, Any],
        *,
        state: MutationState | str,
        outcome_hash: str | None = None,
    ) -> MutationClaimResult:
        """Append the terminal state of one exact attempt."""

        if not isinstance(envelope, Mapping):
            raise MutationClaimError("envelope must be a mapping")
        idempotency_key = _required_identity(envelope, "idempotency_key")
        attempt_id = _required_identity(envelope, "attempt_id")
        fence = _claim_fence(envelope)
        try:
            terminal = MutationState(state)
        except ValueError as exc:
            raise MutationClaimError("unknown mutation settlement state") from exc
        if terminal not in _TERMINAL_STATES:
            raise MutationClaimError("settlement must be terminal")
        if outcome_hash is not None and (
            not isinstance(outcome_hash, str) or not _HASH_RE.fullmatch(outcome_hash)
        ):
            raise MutationClaimError("outcome_hash must be a lowercase SHA-256 digest")

        with _DB_LOCK:
            conn = self._connect()
            try:
                conn.execute("BEGIN IMMEDIATE")
                attempt = conn.execute(
                    """
                    SELECT attempt_id, fence FROM mutation_attempts
                    WHERE idempotency_key = ? AND attempt_id = ?
                    """,
                    (idempotency_key, attempt_id),
                ).fetchone()
                if attempt is None or int(attempt["fence"]) != fence:
                    raise MutationClaimError("settlement does not match a durable attempt")
                prior = conn.execute(
                    """
                    SELECT state, outcome_hash FROM mutation_settlements
                    WHERE idempotency_key = ? AND attempt_id = ?
                    """,
                    (idempotency_key, attempt_id),
                ).fetchone()
                if prior is not None:
                    if prior["state"] != terminal.value or prior["outcome_hash"] != outcome_hash:
                        raise MutationClaimError("attempt already has a different settlement")
                    result = self._result(
                        conn, idempotency_key, attempt, created=False
                    )
                    conn.execute("COMMIT")
                    return result
                conn.execute(
                    """
                    INSERT INTO mutation_settlements(
                        idempotency_key, attempt_id, state, outcome_hash, settled_utc
                    ) VALUES (?, ?, ?, ?, ?)
                    """,
                    (
                        idempotency_key,
                        attempt_id,
                        terminal.value,
                        outcome_hash,
                        _utc_now(),
                    ),
                )
                result = self._result(conn, idempotency_key, attempt, created=True)
                conn.execute("COMMIT")
                return result
            except Exception:
                with suppress(Exception):
                    conn.execute("ROLLBACK")
                raise
            finally:
                conn.close()

    def record_readmission(
        self,
        idempotency_key: str,
        *,
        prior_attempt_id: str,
        readmission_id: str,
        reason: str,
    ) -> bool:
        """Persist an explicit operator decision authorizing one later attempt."""

        idempotency_key = _required_text(idempotency_key, "idempotency_key")
        if not _HASH_RE.fullmatch(idempotency_key):
            raise MutationClaimError(
                "idempotency_key must be a lowercase SHA-256 digest"
            )
        prior_attempt_id = _required_text(prior_attempt_id, "prior_attempt_id")
        readmission_id = _required_text(readmission_id, "readmission_id")
        reason = _required_text(reason, "reason")
        with _DB_LOCK:
            conn = self._connect()
            try:
                conn.execute("BEGIN IMMEDIATE")
                prior = conn.execute(
                    """
                    SELECT 1 FROM mutation_attempts
                    WHERE idempotency_key = ? AND attempt_id = ?
                    """,
                    (idempotency_key, prior_attempt_id),
                ).fetchone()
                if prior is None:
                    raise MutationClaimError("readmission prior attempt does not exist")
                existing = conn.execute(
                    """
                    SELECT prior_attempt_id, reason FROM mutation_readmissions
                    WHERE idempotency_key = ? AND readmission_id = ?
                    """,
                    (idempotency_key, readmission_id),
                ).fetchone()
                if existing is not None:
                    if (
                        existing["prior_attempt_id"] != prior_attempt_id
                        or existing["reason"] != reason
                    ):
                        raise MutationClaimError(
                            "readmission_id is already bound to another decision"
                        )
                    conn.execute("COMMIT")
                    return False
                conn.execute(
                    """
                    INSERT INTO mutation_readmissions(
                        idempotency_key, readmission_id, prior_attempt_id,
                        reason, recorded_utc
                    ) VALUES (?, ?, ?, ?, ?)
                    """,
                    (
                        idempotency_key,
                        readmission_id,
                        prior_attempt_id,
                        reason,
                        _utc_now(),
                    ),
                )
                conn.execute("COMMIT")
                return True
            except Exception:
                with suppress(Exception):
                    conn.execute("ROLLBACK")
                raise
            finally:
                conn.close()

    def read_attempts(self, idempotency_key: str) -> list[dict[str, Any]]:
        """Read one operation's lifecycle without mutating reconciliation state."""

        if not self.db_path.exists():
            return []
        conn = _open_phase2_sqlite(self.db_path, readonly=True)
        conn.row_factory = sqlite3.Row
        try:
            return [
                dict(row)
                for row in conn.execute(
                    """
                    SELECT a.idempotency_key, a.attempt_id, a.fence, a.claimed_utc,
                           COALESCE(s.state, 'in_progress') AS state,
                           s.outcome_hash, s.settled_utc
                    FROM mutation_attempts AS a
                    LEFT JOIN mutation_settlements AS s
                      ON s.idempotency_key = a.idempotency_key
                     AND s.attempt_id = a.attempt_id
                    WHERE a.idempotency_key = ?
                    ORDER BY a.fence
                    """,
                    (idempotency_key,),
                )
            ]
        finally:
            conn.close()

    def read_all(self) -> list[dict[str, Any]]:
        """Read all attempts joined with logical-operation identity."""

        if not self.db_path.exists():
            return []
        conn = _open_phase2_sqlite(self.db_path, readonly=True)
        conn.row_factory = sqlite3.Row
        try:
            return [
                dict(row)
                for row in conn.execute(
                    """
                    SELECT o.idempotency_key, o.graph_id, o.node_id,
                           a.attempt_id, a.fence, o.tool_name, o.args_hash,
                           a.claimed_utc,
                           COALESCE(s.state, 'in_progress') AS state,
                           s.outcome_hash, s.settled_utc
                    FROM mutation_operations AS o
                    JOIN mutation_attempts AS a USING(idempotency_key)
                    LEFT JOIN mutation_settlements AS s
                      ON s.idempotency_key = a.idempotency_key
                     AND s.attempt_id = a.attempt_id
                    ORDER BY o.rowid, a.fence
                    """
                )
            ]
        finally:
            conn.close()


__all__ = [
    "MutationClaimError",
    "MutationClaimResult",
    "MutationClaimStore",
    "MutationState",
    "default_db_path",
]
