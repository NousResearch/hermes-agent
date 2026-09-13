"""Fail-closed, single-attempt proofs for security-policy config writes.

The ledger binds a proof to the policy-file digest, key, operation, session and
nonce. It is an in-process approval receipt, not a non-forgeable operator
principal: an actor able to execute arbitrary Python or write state.db is inside
this trust boundary. The requested replacement value is not bound by this API.
Closing #59293 requires an independently protected approval/execution boundary.

``consume_for_write`` durably spends the proof BEFORE invoking the writer, then
holds SQLite's cross-process writer reservation through a final digest check and
the callback. A failed or interrupted callback is not automatically replayable.
``completed_at`` distinguishes a recorded callback return from a spent proof with
an unknown outcome; neither is a filesystem postcondition receipt. SQLite and
config.yaml are not one atomic resource.

``consume`` only spends authority; existing per-key callers still own their
subsequent write and do not gain transactionality from it. Generic file writers,
nonparticipating config writers, external filesystem races and direct ledger
writes remain outside this broker's serialization boundary.
"""

from __future__ import annotations

import contextlib
import contextvars
import hashlib
import hmac
import inspect
import math
import secrets
import sqlite3
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from hermes_constants import get_hermes_home

MAX_PROOF_TTL_SECONDS = 60
POLICY_CONFIG_KEYS = frozenset({
    "approvals.", "security.", "command_allowlist", "yolo", "persistent.yolo",
})
_POLICY_ROOTS = tuple(key.rstrip(".") for key in sorted(POLICY_CONFIG_KEYS))
_OPERATOR_SETTLEMENT: contextvars.ContextVar[tuple[str, str] | None] = contextvars.ContextVar(
    "operator_settlement", default=None,
)
_MISSING = object()


@contextlib.contextmanager
def _operator_settlement_scope(request_id: str | None = None, confirmation_id: str | None = None):
    """Bind the actual confirmation receipt for the duration of its handler."""
    marker = (request_id, confirmation_id) if (
        isinstance(request_id, str) and request_id
        and isinstance(confirmation_id, str) and confirmation_id
    ) else None
    token = _OPERATOR_SETTLEMENT.set(marker)
    try:
        yield
    finally:
        _OPERATOR_SETTLEMENT.reset(token)


class PolicyMutationDenied(PermissionError):
    """A policy mutation has no valid operator-settled proof."""


class PolicyMutationIndeterminate(RuntimeError):
    """The writer started, but its outcome was not recorded; never auto-retry."""


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


def _same_text(left: Any, right: Any) -> bool:
    """Compare real text bindings without coercion or an ASCII-only assumption."""
    if not isinstance(left, str) or not isinstance(right, str):
        return False
    try:
        return hmac.compare_digest(left.encode("utf-8"), right.encode("utf-8"))
    except UnicodeError:
        return False


def is_policy_config_key(key: str) -> bool:
    """Guard protected roots, descendants, and destructive ancestor replacements."""
    normalized = str(key or "").strip().lower()
    return bool(normalized) and any(
        normalized == root or normalized.startswith(root + ".")
        or root.startswith(normalized + ".")
        for root in _POLICY_ROOTS
    )


def policy_config_digest(config_path: Path | None = None) -> str:
    path = config_path or (get_hermes_home() / "config.yaml")
    try:
        payload = path.read_bytes()
    except FileNotFoundError:
        payload = b""
    except OSError as exc:
        raise PolicyMutationDenied("policy config unavailable") from exc
    return hashlib.sha256(payload).hexdigest()


class PolicyMutationBroker:
    """Receipt minting and durable, cross-instance single-attempt reservation."""

    def __init__(self, *, db_path: Path | None = None, ttl_seconds: int = 60):
        self.ttl_seconds = max(1, min(int(ttl_seconds), MAX_PROOF_TTL_SECONDS))
        try:
            self.db_path = Path(db_path or (get_hermes_home() / "state.db")).absolute()
            self._ensure_schema()
        except (OSError, sqlite3.Error) as exc:
            raise PolicyMutationDenied("policy ledger unavailable") from exc

    @property
    def policy_digest(self) -> str:
        return policy_config_digest(self.db_path.parent / "config.yaml")

    @contextlib.contextmanager
    def _transaction(self):
        """Reserve the database before reading authority, across processes too."""
        db = None
        try:
            db = sqlite3.connect(self.db_path, timeout=5, isolation_level=None)
            db.execute("BEGIN IMMEDIATE")
            yield db
            db.commit()
        except sqlite3.Error as exc:
            raise PolicyMutationDenied("policy ledger unavailable") from exc
        finally:
            if db is not None:
                with contextlib.suppress(sqlite3.Error):
                    db.rollback()
                db.close()

    def _ensure_schema(self) -> None:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._transaction() as db:
            db.execute(
                """CREATE TABLE IF NOT EXISTS policy_mutation_proofs (
                    nonce TEXT PRIMARY KEY, request_id TEXT NOT NULL,
                    session_id TEXT NOT NULL, target_key TEXT NOT NULL,
                    operation TEXT NOT NULL, policy_digest TEXT NOT NULL,
                    token_hash TEXT, expires_at REAL NOT NULL DEFAULT 0,
                    consumed_at REAL, settled INTEGER NOT NULL DEFAULT 0,
                    confirmation_id TEXT, completed_at REAL
                )"""
            )
            columns = {row[1] for row in db.execute("PRAGMA table_info(policy_mutation_proofs)")}
            for column, kind in (("confirmation_id", "TEXT"), ("completed_at", "REAL")):
                if column not in columns:
                    db.execute(f"ALTER TABLE policy_mutation_proofs ADD COLUMN {column} {kind}")

    def request(self, session_id: str, target_key: str, operation: str) -> PolicyMutationRequest:
        """Create a pending request; this never grants authority."""
        if not isinstance(target_key, str) or not is_policy_config_key(target_key):
            raise PolicyMutationDenied("ordinary config keys do not require a policy proof")
        if operation not in {"set", "unset"}:
            raise ValueError("operation must be set or unset")
        if not isinstance(session_id, str) or not session_id:
            raise PolicyMutationDenied("session binding required")
        with self._transaction() as db:
            request = PolicyMutationRequest(
                uuid.uuid4().hex, session_id, target_key, operation,
                self.policy_digest, secrets.token_urlsafe(24),
            )
            db.execute(
                "INSERT INTO policy_mutation_proofs "
                "(nonce,request_id,session_id,target_key,operation,policy_digest) "
                "VALUES (?,?,?,?,?,?)",
                (request.nonce, request.request_id, request.session_id, request.target_key,
                 request.operation, request.policy_digest),
            )
        return request

    def record_settlement(self, request_id: str, confirmation_id: str) -> None:
        """Record the actual confirmation-event receipt before minting."""
        if not all(isinstance(value, str) and value for value in (request_id, confirmation_id)):
            raise PolicyMutationDenied("settlement receipt required")
        with self._transaction() as db:
            updated = db.execute(
                "UPDATE policy_mutation_proofs SET confirmation_id=? "
                "WHERE request_id=? AND settled=0 AND confirmation_id IS NULL",
                (confirmation_id, request_id),
            ).rowcount
            if updated != 1:
                raise PolicyMutationDenied("request is not pending")

    def operator_confirm(self, request_id: str) -> PolicyMutationProof:
        """Mint only for a pending request with the matching recorded receipt."""
        settlement = _OPERATOR_SETTLEMENT.get()
        if not settlement or not _same_text(settlement[0], request_id):
            raise PolicyMutationDenied("settlement receipt required")
        with self._transaction() as db:
            row = db.execute(
                "SELECT nonce,request_id,session_id,target_key,operation,policy_digest,confirmation_id "
                "FROM policy_mutation_proofs WHERE request_id=? AND settled=0",
                (request_id,),
            ).fetchone()
            if row is None:
                raise PolicyMutationDenied("request is not pending")
            nonce, stored_request_id, session_id, target_key, operation, digest, confirmation_id = row
            if not (_same_text(stored_request_id, request_id)
                    and confirmation_id and _same_text(confirmation_id, settlement[1])):
                raise PolicyMutationDenied("settlement receipt required")
            if not _same_text(digest, self.policy_digest):
                raise PolicyMutationDenied("policy digest mismatch")
            now = time.time()
            if not math.isfinite(now):
                raise PolicyMutationDenied("policy clock unavailable")
            expires_at = now + self.ttl_seconds
            token = secrets.token_urlsafe(32)
            updated = db.execute(
                "UPDATE policy_mutation_proofs SET settled=1,token_hash=?,expires_at=? "
                "WHERE nonce=? AND settled=0",
                (hashlib.sha256(token.encode()).hexdigest(), expires_at, nonce),
            ).rowcount
            if updated != 1:
                raise PolicyMutationDenied("request is not pending")
        return PolicyMutationProof(token, digest, target_key, operation, session_id, nonce, expires_at)

    def _verify(self, db, proof, target_key, operation, session_id, *, reserved=False):
        if not isinstance(proof, PolicyMutationProof):
            raise PolicyMutationDenied("operator confirmation proof required")
        if not all(isinstance(value, str) and value for value in (
            proof.token, proof.nonce, proof.policy_digest, proof.target_key,
            proof.operation, proof.session_id, target_key, operation, session_id,
        )):
            raise PolicyMutationDenied("proof binding mismatch")
        try:
            token_hash = hashlib.sha256(proof.token.encode("utf-8")).hexdigest()
        except UnicodeError as exc:
            raise PolicyMutationDenied("proof binding mismatch") from exc
        row = db.execute(
            "SELECT token_hash,expires_at,consumed_at,policy_digest,target_key,operation,session_id,settled "
            "FROM policy_mutation_proofs WHERE nonce=?", (proof.nonce,),
        ).fetchone()
        if (row is None or not _same_text(row[0], token_hash) or row[7] != 1
                or ((row[2] is not None) != reserved)):
            raise PolicyMutationDenied("proof replay or unknown nonce")
        now = time.time()
        for expiry in (row[1], proof.expires_at):
            if type(expiry) not in (int, float) or not math.isfinite(expiry):
                raise PolicyMutationDenied("proof expired or invalid expiry")
        if proof.expires_at != row[1]:
            raise PolicyMutationDenied("proof binding mismatch")
        if not math.isfinite(now) or not now < row[1] <= now + MAX_PROOF_TTL_SECONDS:
            raise PolicyMutationDenied("proof expired or invalid expiry")
        if not all(_same_text(a, b) for a, b in (
            (proof.policy_digest, row[3]), (proof.target_key, row[4]),
            (proof.operation, row[5]), (proof.session_id, row[6]),
            (target_key, row[4]), (operation, row[5]), (session_id, row[6]),
        )):
            raise PolicyMutationDenied("proof binding mismatch")
        if not _same_text(proof.policy_digest, self.policy_digest):
            raise PolicyMutationDenied("policy digest mismatch")
        return now

    @staticmethod
    def _spend(db, proof, now):
        updated = db.execute(
            "UPDATE policy_mutation_proofs SET consumed_at=? WHERE nonce=? AND consumed_at IS NULL",
            (now, proof.nonce),
        ).rowcount
        if updated != 1:
            raise PolicyMutationDenied("proof replay or unknown nonce")

    def consume(self, proof: PolicyMutationProof, target_key: str, operation: str, session_id: str) -> bool:
        """Spend authority once; this alone does not serialize a later file write."""
        with self._transaction() as db:
            self._spend(db, proof, self._verify(db, proof, target_key, operation, session_id))
        return True

    def consume_for_write(self, proof: PolicyMutationProof, target_key: str, operation: str,
                          session_id: str, write: Callable[[], Any]) -> bool:
        """Spend before effect; serialize final CAS and a synchronous callback.

        After reservation, failure requires a new operator decision, not a retry
        of the old proof. A writer can have effects before raising; rolling back
        its SQLite transaction must never resurrect the authority to repeat them.
        """
        if (not callable(write) or inspect.iscoroutinefunction(write)
                or inspect.iscoroutinefunction(getattr(write, "__call__", None))):
            raise TypeError("policy writer must be synchronous")
        started = False
        try:
            with self._transaction() as db:
                self._spend(db, proof, self._verify(db, proof, target_key, operation, session_id))
                db.commit()  # Durable even if the process dies in the callback.
                db.execute("BEGIN IMMEDIATE")
                # Another writer can win between the two transactions; recheck
                # under the reacquired reservation rather than using stale CAS.
                self._verify(db, proof, target_key, operation, session_id, reserved=True)
                started = True
                result = write()
                if inspect.isawaitable(result):
                    if inspect.iscoroutine(result):
                        result.close()
                    raise TypeError("policy writer returned an awaitable")
                db.execute(
                    "UPDATE policy_mutation_proofs SET completed_at=? WHERE nonce=?",
                    (time.time(), proof.nonce),
                )
        except Exception as exc:
            if started:
                raise PolicyMutationIndeterminate(
                    "policy write outcome indeterminate; proof spent; reconcile before a new approval"
                ) from exc
            raise
        return True


def _policy_leaves(value: Any, prefix: str = "", ancestors: frozenset[int] = frozenset()) -> dict[str, Any]:
    if prefix and not is_policy_config_key(prefix):
        return {}
    if not isinstance(value, dict) or not value:
        return {prefix: value} if prefix else {}
    if id(value) in ancestors or len(ancestors) >= 64:
        raise PolicyMutationDenied("cyclic or excessively nested policy payload")
    seen = ancestors | {id(value)}
    leaves = {}
    for key, child in value.items():
        if not isinstance(key, str):
            if prefix:
                raise PolicyMutationDenied("policy payload keys must be strings")
            continue
        escaped = key.replace("\\", "\\\\").replace(".", "\\.")
        dotted = f"{prefix}.{escaped}" if prefix else escaped
        leaves.update(_policy_leaves(child, dotted, seen))
    return leaves


def _policy_payload_keys(value: Any, prefix: str = "") -> list[str]:
    return list(_policy_leaves(value, prefix))


def _same_value(left: Any, right: Any, depth: int = 0) -> bool:
    if depth >= 64:
        raise PolicyMutationDenied("cyclic or excessively nested policy payload")
    if type(left) is not type(right):
        return False
    if isinstance(left, dict):
        return left.keys() == right.keys() and all(
            _same_value(value, right[key], depth + 1) for key, value in left.items()
        )
    if isinstance(left, (list, tuple)):
        return len(left) == len(right) and all(
            _same_value(a, b, depth + 1) for a, b in zip(left, right)
        )
    return left == right


def policy_changed_keys(before: Any, after: Any) -> list[str]:
    """Compare actual policy leaves, including empty maps and escaped key names."""
    old, new = _policy_leaves(before), _policy_leaves(after)
    return sorted(key for key in old.keys() | new.keys()
                  if not _same_value(old.get(key, _MISSING), new.get(key, _MISSING)))


def require_policy_proof_for_payload(config: Any, proof: Any = None, *, session_id: str = "local",
                                     before: Any = None, consume: bool = True) -> None:
    """Guard a full-document payload; callers must supply the final write shape."""
    policy_keys = policy_changed_keys(before, config) if before is not None else _policy_payload_keys(config)
    if not policy_keys:
        return
    if not isinstance(proof, PolicyMutationProof):
        raise PolicyMutationDenied("operator confirmation proof required")
    if len(policy_keys) != 1:
        raise PolicyMutationDenied("policy payload requires a single-key proof")
    if not _same_text(proof.target_key, policy_keys[0]):
        raise PolicyMutationDenied("proof binding mismatch")
    if consume:
        require_policy_proof(policy_keys[0], "set", proof, session_id=session_id)


def require_policy_proof(key: str, operation: str, proof: Any = None, *, session_id: str = "local") -> None:
    if not is_policy_config_key(key):
        return
    if not isinstance(proof, PolicyMutationProof):
        raise PolicyMutationDenied("operator confirmation proof required")
    PolicyMutationBroker().consume(proof, key, operation, session_id)
