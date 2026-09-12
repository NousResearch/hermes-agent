"""Atomic persistent claims for Phase 2 mutating logical operations.

A mutating tool call is keyed by the idempotency key from the sealed task
envelope. The first writer atomically claims it; every subsequent attempt
with the same key loses without side effects. Raw arguments are never stored.
The persisted deterministic SHA-256 hash still permits offline confirmation
of guessed arguments; it is an audit identity, not a confidentiality boundary.
"""

from __future__ import annotations

import hashlib
import json
import re
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from agent.phase2_sqlite import _check_phase2_db_files, _open_phase2_sqlite
from hermes_constants import get_hermes_home

_DB_LOCK = threading.Lock()
_HASH_RE = re.compile(r"^[0-9a-f]{64}$")
_SQLITE_INT_MAX = 2**63 - 1


class MutationClaimError(ValueError):
    """The caller supplied an invalid mutation claim."""


def _required_identity(envelope: Mapping[str, Any], field: str) -> str:
    value = envelope.get(field)
    if not isinstance(value, str) or not value.strip():
        raise MutationClaimError(f"{field} must be a non-empty string")
    return value


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


class MutationClaimStore:
    """Own the durable first-writer claim for a mutating idempotency key."""

    def __init__(self, db_path: Path | str | None = None) -> None:
        self.db_path = Path(db_path) if db_path is not None else default_db_path()

    def _connect(self) -> sqlite3.Connection:
        conn = _open_phase2_sqlite(self.db_path)
        try:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA busy_timeout=5000")
            conn.row_factory = sqlite3.Row
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS mutation_claims (
                    idempotency_key TEXT PRIMARY KEY,
                    graph_id TEXT NOT NULL,
                    node_id TEXT NOT NULL,
                    attempt_id TEXT NOT NULL,
                    fence INTEGER NOT NULL,
                    tool_name TEXT NOT NULL,
                    args_hash TEXT NOT NULL,
                    claimed_utc TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TRIGGER IF NOT EXISTS mutation_claims_no_update
                BEFORE UPDATE ON mutation_claims
                BEGIN
                    SELECT RAISE(ABORT, 'mutation_claims is append-only');
                END
                """
            )
            conn.execute(
                """
                CREATE TRIGGER IF NOT EXISTS mutation_claims_no_delete
                BEFORE DELETE ON mutation_claims
                BEGIN
                    SELECT RAISE(ABORT, 'mutation_claims is append-only');
                END
                """
            )
            _check_phase2_db_files(self.db_path, harden=True)
            return conn
        except BaseException:
            conn.close()
            raise

    def try_claim(
        self,
        envelope: Mapping[str, Any],
        *,
        tool_name: str,
        args: Mapping[str, Any],
    ) -> bool:
        """Atomically claim one logical mutating operation.

        Returns ``True`` if this caller is the first writer.  Returns ``False``
        only when the same idempotency key was already claimed *for the same
        logical effect* (same graph_id, node_id, attempt_id, fence, tool_name,
        and args_hash) — a safe at-least-once redelivery.  Raises
        ``MutationClaimError`` if the key was already claimed for a *different*
        logical effect; callers MUST treat this as a hard error and must not
        proceed.  The deterministic args hash omits raw payloads but permits
        offline confirmation of guessed inputs.
        """

        if not isinstance(envelope, Mapping):
            raise MutationClaimError("envelope must be a mapping")
        idempotency_key = _required_identity(envelope, "idempotency_key")
        if not _HASH_RE.fullmatch(idempotency_key):
            raise MutationClaimError(
                "idempotency_key must be a lowercase SHA-256 digest"
            )
        if not isinstance(tool_name, str) or not tool_name.strip():
            raise MutationClaimError("tool_name must be a non-empty string")
        row = (
            idempotency_key,
            _required_identity(envelope, "graph_id"),
            _required_identity(envelope, "node_id"),
            _required_identity(envelope, "attempt_id"),
            _claim_fence(envelope),
            tool_name,
            _canonical_hash(args),
            datetime.now(timezone.utc).isoformat(),
        )
        with _DB_LOCK:
            conn = self._connect()
            try:
                conn.execute("BEGIN IMMEDIATE")
                inserted = conn.execute(
                    """
                    INSERT INTO mutation_claims(
                        idempotency_key, graph_id, node_id, attempt_id, fence,
                        tool_name, args_hash, claimed_utc
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(idempotency_key) DO NOTHING
                    """,
                    row,
                ).rowcount
                if inserted == 1:
                    conn.execute("COMMIT")
                    return True
                # Key already claimed.  Read the prior record to determine
                # whether this is a safe same-effect replay (False) or an
                # idempotency-key collision with a different effect (error).
                prior = conn.execute(
                    """
                    SELECT graph_id, node_id, attempt_id, fence,
                           tool_name, args_hash
                    FROM mutation_claims
                    WHERE idempotency_key = ?
                    """,
                    (idempotency_key,),
                ).fetchone()
                conn.execute("COMMIT")
                # Identity: every field that distinguishes one logical effect
                # from another.  graph_id/node_id/attempt_id anchor the
                # provenance; fence captures the exact execution epoch;
                # tool_name and args_hash capture the operation itself.
                if prior is None:
                    raise MutationClaimError(
                        f"idempotency_key collision: key {idempotency_key!r} "
                        "conflicted without a readable prior claim"
                    )
                if (
                    prior["graph_id"] == row[1]
                    and prior["node_id"] == row[2]
                    and prior["attempt_id"] == row[3]
                    and prior["fence"] == row[4]
                    and prior["tool_name"] == row[5]
                    and prior["args_hash"] == row[6]
                ):
                    return False
                raise MutationClaimError(
                    f"idempotency_key collision: key {idempotency_key!r} was "
                    "already claimed for a different logical effect"
                )
            except MutationClaimError:
                try:
                    conn.execute("ROLLBACK")
                except Exception:
                    pass
                raise
            except Exception:
                try:
                    conn.execute("ROLLBACK")
                except Exception:
                    pass
                raise
            finally:
                conn.close()

    def read_all(self) -> list[dict[str, Any]]:
        if not self.db_path.exists():
            return []
        conn = _open_phase2_sqlite(self.db_path, readonly=True)
        conn.row_factory = sqlite3.Row
        try:
            return [
                dict(row)
                for row in conn.execute("SELECT * FROM mutation_claims ORDER BY rowid")
            ]
        finally:
            conn.close()


__all__ = ["MutationClaimError", "MutationClaimStore", "default_db_path"]
