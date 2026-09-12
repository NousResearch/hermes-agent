"""Operator-settled, one-shot proofs for security-policy config writes."""

from __future__ import annotations

import contextlib
import contextvars
import hashlib
import secrets
import sqlite3
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from hermes_constants import get_hermes_home

MAX_PROOF_TTL_SECONDS = 60
POLICY_CONFIG_KEYS = frozenset({"approvals.", "security.", "command_allowlist", "yolo", "persistent.yolo"})
_OPERATOR_SETTLEMENT: contextvars.ContextVar[object | None] = contextvars.ContextVar(
    "operator_settlement", default=None,
)


@contextlib.contextmanager
def _operator_settlement_scope():
    """Mark the existing operator-input surface while its handler is executing."""
    marker = object()
    token = _OPERATOR_SETTLEMENT.set(marker)
    try:
        yield
    finally:
        _OPERATOR_SETTLEMENT.reset(token)


class PolicyMutationDenied(PermissionError):
    """A policy mutation has no valid operator-settled proof."""


@dataclass(frozen=True)
class PolicyMutationRequest:
    request_id: str
    session_id: str
    target_key: str
    operation: str
    policy_digest: str
    nonce: str


@dataclass(frozen=True)
class PolicyMutationProof:
    token: str
    policy_digest: str
    target_key: str
    operation: str
    session_id: str
    nonce: str
    expires_at: float


def is_policy_config_key(key: str) -> bool:
    """Use one classification table at every config mutation boundary."""
    normalized = str(key or "").strip().lower()
    return (
        normalized == "command_allowlist"
        or normalized.startswith("command_allowlist.")
        or normalized == "yolo"
        or normalized == "persistent.yolo"
        or normalized.startswith("approvals.")
        or normalized.startswith("security.")
    )


def policy_config_digest(config_path: Path | None = None) -> str:
    path = config_path or (get_hermes_home() / "config.yaml")
    try:
        payload = path.read_bytes()
    except OSError:
        payload = b""
    return hashlib.sha256(payload).hexdigest()


class PolicyMutationBroker:
    """Approval-layer broker for request, operator settlement, and one-shot consume."""

    def __init__(self, *, db_path: Path | None = None, ttl_seconds: int = 60):
        self.db_path = Path(db_path or (get_hermes_home() / "state.db"))
        self.ttl_seconds = max(1, min(int(ttl_seconds), MAX_PROOF_TTL_SECONDS))
        self._lock = threading.RLock()
        self._ensure_schema()

    @property
    def policy_digest(self) -> str:
        return policy_config_digest(self.db_path.parent / "config.yaml")

    def _ensure_schema(self) -> None:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.db_path) as db:
            db.execute(
                """CREATE TABLE IF NOT EXISTS policy_mutation_proofs (
                    nonce TEXT PRIMARY KEY, request_id TEXT NOT NULL,
                    session_id TEXT NOT NULL, target_key TEXT NOT NULL,
                    operation TEXT NOT NULL, policy_digest TEXT NOT NULL,
                    token_hash TEXT, expires_at REAL NOT NULL DEFAULT 0,
                    consumed_at REAL, settled INTEGER NOT NULL DEFAULT 0
                )"""
            )
            db.commit()

    def request(self, session_id: str, target_key: str, operation: str) -> PolicyMutationRequest:
        """Create a pending request; this never grants authority."""
        if not is_policy_config_key(target_key):
            raise PolicyMutationDenied("ordinary config keys do not require a policy proof")
        if operation not in {"set", "unset"}:
            raise ValueError("operation must be set or unset")
        nonce = secrets.token_urlsafe(24)
        request = PolicyMutationRequest(
            request_id=uuid.uuid4().hex, session_id=str(session_id),
            target_key=str(target_key), operation=operation,
            policy_digest=self.policy_digest, nonce=nonce,
        )
        with sqlite3.connect(self.db_path) as db:
            db.execute(
                "INSERT INTO policy_mutation_proofs "
                "(nonce,request_id,session_id,target_key,operation,policy_digest) "
                "VALUES (?,?,?,?,?,?)",
                (nonce, request.request_id, request.session_id, request.target_key,
                 request.operation, request.policy_digest),
            )
            db.commit()
        return request

    def operator_confirm(self, request_id: str) -> PolicyMutationProof:
        """Mint only while the existing operator-input surface is settling a choice."""
        if _OPERATOR_SETTLEMENT.get() is None:
            raise PolicyMutationDenied("operator settlement required")
        now = time.time()
        with self._lock, sqlite3.connect(self.db_path) as db:
            row = db.execute(
                "SELECT nonce,session_id,target_key,operation,policy_digest FROM "
                "policy_mutation_proofs WHERE request_id=? AND settled=0",
                (request_id,),
            ).fetchone()
            if row is None:
                raise PolicyMutationDenied("request is not pending")
            nonce, session_id, target_key, operation, digest = row
            expires_at = now + self.ttl_seconds
            token = secrets.token_urlsafe(32)
            updated = db.execute(
                "UPDATE policy_mutation_proofs SET settled=1,token_hash=?,expires_at=? "
                "WHERE nonce=? AND settled=0",
                (hashlib.sha256(token.encode()).hexdigest(), expires_at, nonce),
            ).rowcount
            if updated != 1:
                raise PolicyMutationDenied("request is not pending")
            db.commit()
        return PolicyMutationProof(token, digest, target_key, operation, session_id, nonce, expires_at)

    def consume(self, proof: PolicyMutationProof, target_key: str, operation: str, session_id: str) -> bool:
        now = time.time()
        if proof.expires_at <= now:
            raise PolicyMutationDenied("proof expired")
        if (proof.target_key, proof.operation, proof.session_id) != (target_key, operation, session_id):
            raise PolicyMutationDenied("proof binding mismatch")
        if proof.policy_digest != self.policy_digest:
            raise PolicyMutationDenied("policy digest mismatch")
        token_hash = hashlib.sha256(proof.token.encode()).hexdigest()
        with self._lock, sqlite3.connect(self.db_path) as db:
            row = db.execute(
                "SELECT token_hash,expires_at,consumed_at FROM policy_mutation_proofs WHERE nonce=?",
                (proof.nonce,),
            ).fetchone()
            if row is None or row[0] != token_hash or row[2] is not None:
                raise PolicyMutationDenied("proof replay or unknown nonce")
            if row[1] <= now:
                raise PolicyMutationDenied("proof expired")
            updated = db.execute(
                "UPDATE policy_mutation_proofs SET consumed_at=? WHERE nonce=? AND consumed_at IS NULL",
                (now, proof.nonce),
            ).rowcount
            if updated != 1:
                raise PolicyMutationDenied("proof replay or unknown nonce")
            db.commit()
        return True


def require_policy_proof(key: str, operation: str, proof: Any = None, *, session_id: str = "local") -> None:
    if not is_policy_config_key(key):
        return
    if not isinstance(proof, PolicyMutationProof):
        raise PolicyMutationDenied("operator confirmation proof required")
    PolicyMutationBroker().consume(proof, key, operation, session_id)
