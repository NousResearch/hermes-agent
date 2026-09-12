"""One-shot, fail-closed proofs for all security-policy config writes.

Every policy mutation is digest-bound, key/operation/session-bound, and covered by
this broker or the shared ``save_config`` chokepoint. The SQLite ledger is the
one-shot authority and failures are refusals. Settlement remains an in-process
receipt: it is not cryptographically non-forgeable. Production hardening for
#59293 requires an OS/process boundary, such as a credential-manager-held broker
key or a separate approval process; that is the remaining contract item for
closure.
"""

from __future__ import annotations

import contextlib
import contextvars
import hashlib
import hmac
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
def _operator_settlement_scope(request_id: str | None = None, confirmation_id: str | None = None):
    """Bind the actual confirm-event receipt while its handler is executing."""
    marker = (str(request_id), str(confirmation_id)) if request_id and confirmation_id else None
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
        try:
            self._ensure_schema()
        except (OSError, sqlite3.Error) as exc:
            raise PolicyMutationDenied("policy ledger unavailable") from exc

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
                    consumed_at REAL, settled INTEGER NOT NULL DEFAULT 0,
                    confirmation_id TEXT
                )"""
            )
            columns = {row[1] for row in db.execute("PRAGMA table_info(policy_mutation_proofs)")}
            if "confirmation_id" not in columns:
                db.execute("ALTER TABLE policy_mutation_proofs ADD COLUMN confirmation_id TEXT")
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

    def record_settlement(self, request_id: str, confirmation_id: str) -> None:
        """Record the confirm-event receipt before its handler mints authority."""
        if not request_id or not confirmation_id:
            raise PolicyMutationDenied("settlement receipt required")
        with self._lock, sqlite3.connect(self.db_path) as db:
            updated = db.execute(
                "UPDATE policy_mutation_proofs SET confirmation_id=? "
                "WHERE request_id=? AND settled=0 AND confirmation_id IS NULL",
                (str(confirmation_id), str(request_id)),
            ).rowcount
            if updated != 1:
                raise PolicyMutationDenied("request is not pending")
            db.commit()

    def operator_confirm(self, request_id: str) -> PolicyMutationProof:
        """Mint only for a pending request with the actual confirm-event receipt."""
        settlement = _OPERATOR_SETTLEMENT.get()
        if not settlement or settlement[0] != str(request_id):
            raise PolicyMutationDenied("settlement receipt required")
        now = time.time()
        with self._lock, sqlite3.connect(self.db_path) as db:
            row = db.execute(
                "SELECT nonce,request_id,session_id,target_key,operation,policy_digest,confirmation_id FROM "
                "policy_mutation_proofs WHERE request_id=? AND settled=0",
                (request_id,),
            ).fetchone()
            if row is None:
                raise PolicyMutationDenied("request is not pending")
            nonce, stored_request_id, session_id, target_key, operation, digest, confirmation_id = row
            if not hmac.compare_digest(str(stored_request_id), str(request_id)):
                raise PolicyMutationDenied("settlement receipt required")
            if not confirmation_id or not hmac.compare_digest(str(confirmation_id), str(settlement[1])):
                raise PolicyMutationDenied("settlement receipt required")
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
        """Atomically verify every binding and consume exactly once.

        The caller must perform the final digest CAS immediately before its write while
        holding the same process-wide writer lock; this method never consumes on refusal.
        """
        now = time.time()
        token_hash = hashlib.sha256(proof.token.encode()).hexdigest()
        with self._lock, sqlite3.connect(self.db_path) as db:
            row = db.execute(
                "SELECT token_hash,expires_at,consumed_at,policy_digest,target_key,operation,session_id "
                "FROM policy_mutation_proofs WHERE nonce=?",
                (proof.nonce,),
            ).fetchone()
            if row is None or not hmac.compare_digest(str(row[0] or ""), token_hash) or row[2] is not None:
                raise PolicyMutationDenied("proof replay or unknown nonce")
            if row[1] <= now or proof.expires_at <= now:
                raise PolicyMutationDenied("proof expired")
            stored_digest, stored_key, stored_operation, stored_session = row[3:]
            bindings = (
                (proof.policy_digest, stored_digest), (proof.target_key, stored_key),
                (proof.operation, stored_operation), (proof.session_id, stored_session),
                (target_key, stored_key), (operation, stored_operation), (session_id, stored_session),
            )
            if not all(hmac.compare_digest(str(left), str(right)) for left, right in bindings):
                raise PolicyMutationDenied("proof binding mismatch")
            if not hmac.compare_digest(proof.policy_digest, self.policy_digest):
                raise PolicyMutationDenied("policy digest mismatch")
            updated = db.execute(
                "UPDATE policy_mutation_proofs SET consumed_at=? WHERE nonce=? AND consumed_at IS NULL",
                (now, proof.nonce),
            ).rowcount
            if updated != 1:
                raise PolicyMutationDenied("proof replay or unknown nonce")
            db.commit()
        return True

    def consume_for_write(self, proof: PolicyMutationProof, target_key: str, operation: str,
                          session_id: str, write) -> bool:
        """Serialize consume, final digest verification, and the guarded write."""
        with self._lock:
            now_digest = self.policy_digest
            if not hmac.compare_digest(proof.policy_digest, now_digest):
                raise PolicyMutationDenied("policy digest mismatch")
            with sqlite3.connect(self.db_path) as db:
                row = db.execute(
                    "SELECT token_hash,expires_at,consumed_at,policy_digest,target_key,operation,session_id "
                    "FROM policy_mutation_proofs WHERE nonce=?",
                    (proof.nonce,),
                ).fetchone()
                if (
                    row is None
                    or not hmac.compare_digest(
                        str(row[0] or ""), hashlib.sha256(proof.token.encode()).hexdigest()
                    )
                    or row[2] is not None
                ):
                    raise PolicyMutationDenied("proof replay or unknown nonce")
                if row[1] <= time.time() or proof.expires_at <= time.time():
                    raise PolicyMutationDenied("proof expired")
                if not all(hmac.compare_digest(str(a), str(b)) for a, b in (
                    (proof.policy_digest, row[3]), (proof.target_key, row[4]),
                    (proof.operation, row[5]), (proof.session_id, row[6]),
                    (target_key, row[4]), (operation, row[5]), (session_id, row[6]),
                )):
                    raise PolicyMutationDenied("proof binding mismatch")
                if not hmac.compare_digest(proof.policy_digest, self.policy_digest):
                    raise PolicyMutationDenied("policy digest mismatch")
                write()
                updated = db.execute(
                    "UPDATE policy_mutation_proofs SET consumed_at=? WHERE nonce=? AND consumed_at IS NULL",
                    (time.time(), proof.nonce),
                ).rowcount
                if updated != 1:
                    raise PolicyMutationDenied("proof replay or unknown nonce")
                db.commit()
        return True



def _policy_payload_keys(value: Any, prefix: str = "") -> list[str]:
    if not isinstance(value, dict):
        return [prefix] if prefix else []
    keys = []
    for key, child in value.items():
        dotted = f"{prefix}.{key}" if prefix else str(key)
        keys.extend(_policy_payload_keys(child, dotted))
    return keys


def policy_changed_keys(before: Any, after: Any) -> list[str]:
    """Return policy leaves whose effective values changed in a full-document save."""
    paths = set(_policy_payload_keys(before)) | set(_policy_payload_keys(after))
    changed = []
    for key in paths:
        if not is_policy_config_key(key):
            continue
        def get(node):
            for part in key.split("."):
                if not isinstance(node, dict) or part not in node:
                    return object()
                node = node[part]
            return node
        if get(before) != get(after):
            changed.append(key)
    return sorted(changed)


def require_policy_proof_for_payload(config: Any, proof: Any = None, *, session_id: str = "local",
                                     before: Any = None, consume: bool = True) -> None:
    """Guard the shared full-document writer without affecting ordinary saves."""
    policy_keys = policy_changed_keys(before, config) if before is not None else [
        key for key in _policy_payload_keys(config) if is_policy_config_key(key)
    ]
    if not policy_keys:
        return
    if not isinstance(proof, PolicyMutationProof):
        raise PolicyMutationDenied("operator confirmation proof required")
    if len(policy_keys) != 1:
        raise PolicyMutationDenied("policy payload requires a single-key proof")
    if proof.target_key != policy_keys[0]:
        raise PolicyMutationDenied("proof binding mismatch")
    if consume:
        require_policy_proof(policy_keys[0], "set", proof, session_id=session_id)


def require_policy_proof(key: str, operation: str, proof: Any = None, *, session_id: str = "local") -> None:
    if not is_policy_config_key(key):
        return
    if not isinstance(proof, PolicyMutationProof):
        raise PolicyMutationDenied("operator confirmation proof required")
    PolicyMutationBroker().consume(proof, key, operation, session_id)
