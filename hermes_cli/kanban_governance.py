"""Governance Control Plane — machine-enforced budget, mutation policies, and denial ledger.

Enforces physical governance constraints (task budgets, dynamic wave limits, mutation tokens)
at the persistence boundary inside kanban_db, independent of prompt instructions or tool wrappers.
"""

from __future__ import annotations

import base64
from dataclasses import dataclass
import hashlib
import hmac
import io
import json
import logging
import os
from pathlib import Path
import secrets
import sqlite3
import time
from typing import Any, Optional

from ruamel.yaml import YAML

logger = logging.getLogger(__name__)


def load_governance_yaml(raw: bytes) -> Any:
    return YAML(typ="safe").load(raw)


def dump_governance_yaml(data: dict) -> str:
    stream = io.StringIO()
    YAML(typ="safe").dump(data, stream)
    return stream.getvalue()


class GovernancePolicyError(RuntimeError):
    """Base exception for policy loading, validation, and integrity failures. Fails closed."""

    def __init__(
        self,
        message: str,
        *,
        code: str = "GOVERNANCE_POLICY_INVALID",
        reason_code: str = "POLICY_INVALID",
        semantic_state: str = "GOVERNANCE_LOCKED",
        policy_id: str = "",
        policy_hash: str = "",
        policy_path: Optional[str] = None,
        event_id: str = "",
    ):
        super().__init__(message)
        self.message = message
        self.code = code
        self.reason_code = reason_code
        self.semantic_state = semantic_state
        self.policy_id = policy_id
        self.policy_hash = policy_hash
        self.policy_path = policy_path
        self.event_id = event_id
        self.graph_mutation_committed = False


class GovernancePolicyMissing(GovernancePolicyError):
    def __init__(self, message: str, *, policy_path: Optional[str] = None):
        super().__init__(
            message,
            reason_code="POLICY_MISSING",
            policy_path=policy_path,
        )


class GovernancePolicyMalformed(GovernancePolicyError):
    def __init__(self, message: str, *, policy_path: Optional[str] = None):
        super().__init__(
            message,
            reason_code="POLICY_MALFORMED",
            policy_path=policy_path,
        )


class GovernancePolicyHashMismatch(GovernancePolicyError):
    def __init__(
        self,
        message: str,
        *,
        policy_path: Optional[str] = None,
        expected_hash: str = "",
        actual_hash: str = "",
    ):
        super().__init__(
            message,
            reason_code="POLICY_HASH_MISMATCH",
            policy_hash=actual_hash,
            policy_path=policy_path,
        )
        self.expected_hash = expected_hash
        self.actual_hash = actual_hash


class GovernancePolicyUnsupportedVersion(GovernancePolicyError):
    def __init__(self, message: str, *, policy_path: Optional[str] = None, version: str = ""):
        super().__init__(
            message,
            reason_code="POLICY_UNSUPPORTED_VERSION",
            policy_path=policy_path,
        )
        self.unsupported_version = version


class GovernanceBudgetExceeded(ValueError):
    """Raised when a task or graph mutation exceeds machine-enforced governance limits."""

    def __init__(
        self,
        message: str,
        *,
        code: str = "BUDGET_EXHAUSTED",
        policy_id: str = "",
        policy_hash: str = "",
        actor: str = "",
        ingress: str = "",
        observed_task_count: int = 0,
        configured_limit: int = 0,
        permitted_next_actions: Optional[list] = None,
        semantic_state: str = "BLOCKED_BUDGET_EXHAUSTED",
        event_id: str = "",
    ):
        super().__init__(message)
        self.message = message
        self.code = code
        self.reason_code = code
        self.policy_id = policy_id
        self.policy_hash = policy_hash
        self.actor = actor
        self.ingress = ingress
        self.observed_task_count = observed_task_count
        self.configured_limit = configured_limit
        self.semantic_state = semantic_state
        self.permitted_next_actions = permitted_next_actions or [
            {
                "action": "record_unresolved_risk",
                "contract": "Write UNRESOLVED_RISK_REGISTER.md and GOVERNANCE_HANDOFF.json to workspace documenting unresolved material gaps, reproduction, impact, and marking additional_work_required=true and additional_work_authorized=false.",
            },
            {
                "action": "attach_evidence",
                "contract": "Generate EVIDENCE_INDEX.json and SHA256SUMS covering all evaluation and verification artifacts.",
            },
            {
                "action": "block_budget_exhausted",
                "contract": "Call kanban_block(kind='capability', reason='BUDGET_EXHAUSTED: <summary>') to freeze task in BLOCKED_BUDGET_EXHAUSTED state.",
            },
            {
                "action": "request_reauthorization",
                "contract": "Record recommended next budget delta (tasks/waves) in GOVERNANCE_HANDOFF.json for human review.",
            },
        ]
        self.event_id = event_id


def ensure_governance_schema(conn: sqlite3.Connection) -> None:
    """Ensure the durable governance ledger table exists."""
    conn.execute("""
        CREATE TABLE IF NOT EXISTS governance_events (
            id                   INTEGER PRIMARY KEY AUTOINCREMENT,
            event_id             TEXT UNIQUE NOT NULL,
            policy_id            TEXT NOT NULL,
            policy_hash          TEXT NOT NULL,
            actor                TEXT NOT NULL,
            ingress              TEXT NOT NULL,
            requested_operation  TEXT NOT NULL,
            decision             TEXT NOT NULL,
            reason_code          TEXT NOT NULL,
            observed_task_count  INTEGER NOT NULL,
            configured_limit     INTEGER NOT NULL,
            details_json         TEXT,
            timestamp            INTEGER NOT NULL
        )
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_gov_events_policy ON governance_events(policy_id)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_gov_events_decision ON governance_events(decision)")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS governance_reauth_requests (
            request_id           TEXT PRIMARY KEY,
            board_slug           TEXT NOT NULL,
            originating_denial_event TEXT,
            current_policy_hash  TEXT NOT NULL,
            requested_delta      INTEGER NOT NULL,
            reason               TEXT NOT NULL,
            requester            TEXT NOT NULL,
            status               TEXT NOT NULL,
            created_at           INTEGER NOT NULL
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS governance_reauth_grants (
            grant_id             TEXT PRIMARY KEY,
            request_id           TEXT,
            board_slug           TEXT NOT NULL,
            previous_policy_hash TEXT NOT NULL,
            authorized_delta     INTEGER NOT NULL,
            principal_id         TEXT NOT NULL,
            auth_method          TEXT NOT NULL,
            approval_nonce       TEXT UNIQUE NOT NULL,
            signature_proof      TEXT NOT NULL,
            status               TEXT NOT NULL,
            issued_at            INTEGER NOT NULL,
            expires_at           INTEGER NOT NULL,
            used_at              INTEGER
        )
    """)


@dataclass
class TrustedPrincipalContext:
    principal_id: str
    principal_type: str  # "human" or "agent" or "system"
    auth_method: str     # e.g., "TRUSTED_CONTROL_SURFACE", "ED25519_SIGNATURE"
    session_id: Optional[str] = None
    proof_token: Optional[str] = None

    def is_authenticated_human(self, action: str = "*", target: str = "*") -> bool:
        if str(self.principal_type).lower() != "human":
            return False
        if not self.auth_method or str(self.auth_method).lower() in {"self_asserted", "unverified", "payload", "none", "arbitrary-label"}:
            return False
        if not self.proof_token:
            return False
        valid, _ = verify_human_proof_token(self.proof_token, self.principal_id, action=action, target=target)
        return valid

    def verify_human_authority(self, action: str = "*", target: str = "*") -> None:
        if str(self.principal_type).lower() != "human":
            raise PermissionError(
                f"GOVERNANCE DENIAL: Principal '{self.principal_id}' of type '{self.principal_type}' is unauthorized (requires human)."
            )
        if not self.auth_method or str(self.auth_method).lower() in {"self_asserted", "unverified", "payload", "none", "arbitrary-label"}:
            raise PermissionError(
                f"GOVERNANCE DENIAL: Principal '{self.principal_id}' is unauthorized (uses unverified/arbitrary auth method '{self.auth_method}')."
            )
        if not self.proof_token:
            raise PermissionError(
                f"GOVERNANCE DENIAL: Human principal '{self.principal_id}' lacks mandatory cryptographic proof token."
            )
        valid, reason = verify_human_proof_token(self.proof_token, self.principal_id, action=action, target=target)
        if not valid:
            raise PermissionError(
                f"GOVERNANCE DENIAL: Proof token verification failed for '{self.principal_id}': {reason}."
            )


@dataclass
class ReauthorizationRequest:
    request_id: str
    board_slug: str
    originating_denial_event: Optional[str]
    current_policy_hash: str
    requested_delta: int
    reason: str
    requester: str
    status: str
    created_at: int


@dataclass
class ReauthorizationGrant:
    grant_id: str
    request_id: Optional[str]
    board_slug: str
    previous_policy_hash: str
    authorized_delta: int
    principal_id: str
    auth_method: str
    approval_nonce: str
    signature_proof: str
    status: str
    issued_at: int
    expires_at: int
    used_at: Optional[int] = None


_GOVERNANCE_KEY_CACHE: dict[str, bytes] = {}


def get_governance_secret() -> bytes:
    """Retrieve the authoritative governance master key.

    Persistence priority:
    1. HERMES_KANBAN_GOVERNANCE_SECRET / HERMES_GOVERNANCE_SECRET env var.
    2. Persisted file at <kanban_home>/.governance_authority/master.key (mode 0600).
    3. Cached in memory per directory/env scope.
    Never exposed to worker subprocesses or written inside task workspaces.
    """
    env_sec = os.environ.get("HERMES_KANBAN_GOVERNANCE_SECRET") or os.environ.get("HERMES_GOVERNANCE_SECRET")
    if env_sec:
        cache_key = f"env:{env_sec}"
        if cache_key in _GOVERNANCE_KEY_CACHE:
            return _GOVERNANCE_KEY_CACHE[cache_key]
        sec = hashlib.sha256(env_sec.encode("utf-8")).digest()
        _GOVERNANCE_KEY_CACHE[cache_key] = sec
        return sec

    home_dir = os.environ.get("HERMES_KANBAN_HOME") or os.environ.get("HERMES_HOME")
    if home_dir:
        base = Path(home_dir).expanduser()
    else:
        try:
            from hermes_constants import get_default_hermes_root
            base = get_default_hermes_root()
        except Exception:
            base = Path.home() / ".hermes"

    cache_key = f"dir:{base}"
    if cache_key in _GOVERNANCE_KEY_CACHE:
        return _GOVERNANCE_KEY_CACHE[cache_key]

    auth_dir = base / ".governance_authority"
    key_file = auth_dir / "master.key"

    if key_file.is_file():
        try:
            raw = key_file.read_bytes()
            if len(raw) >= 32:
                sec = raw[:32]
                _GOVERNANCE_KEY_CACHE[cache_key] = sec
                return sec
        except Exception:
            pass

    auth_dir.mkdir(parents=True, exist_ok=True)
    try:
        os.chmod(auth_dir, 0o700)
    except Exception:
        pass

    new_sec = secrets.token_bytes(32)
    tmp_path = auth_dir / f"tmp_{secrets.token_hex(6)}"
    tmp_path.write_bytes(new_sec)
    try:
        os.chmod(tmp_path, 0o600)
    except Exception:
        pass
    tmp_path.replace(key_file)
    try:
        os.chmod(key_file, 0o600)
    except Exception:
        pass

    _GOVERNANCE_KEY_CACHE[cache_key] = new_sec
    return new_sec


def issue_human_proof_token(
    principal_id: str,
    action: str = "*",
    target: str = "*",
    ttl_seconds: int = 3600,
) -> str:
    """Issue a cryptographically signed human proof token.

    Uses base64url-encoded JSON payload so targets containing dots, paths,
    or spaces never break token parsing.
    Only callable by control plane surface possessing the master secret.
    """
    now = int(time.time())
    expires_at = now + ttl_seconds
    nonce = secrets.token_hex(8)
    data = {
        "v": 2,
        "principal_id": principal_id,
        "action": action,
        "target": target,
        "expires_at": expires_at,
        "nonce": nonce,
    }
    payload_bytes = json.dumps(data, sort_keys=True, separators=(",", ":")).encode("utf-8")
    payload_b64 = base64.urlsafe_b64encode(payload_bytes).decode("ascii").rstrip("=")
    sig = hmac.new(get_governance_secret(), payload_bytes, hashlib.sha256).hexdigest()
    return f"hproof_v2.{payload_b64}.{sig}"


def verify_human_proof_token(
    token: Optional[str],
    principal_id: str,
    action: str = "*",
    target: str = "*",
) -> tuple[bool, str]:
    """Verify proof token signature, expiration, principal, action, and target."""
    if not token or not isinstance(token, str):
        return False, "Missing proof token"
    parts = token.split(".")
    if len(parts) != 3 or parts[0] not in {"hproof_v1", "hproof_v2"}:
        return False, "Malformed proof token format"
    version, payload_str, sig = parts[0], parts[1], parts[2]

    if version == "hproof_v2":
        try:
            padded = payload_str + "=" * ((4 - len(payload_str) % 4) % 4)
            payload_bytes = base64.urlsafe_b64decode(padded.encode("ascii"))
            data = json.loads(payload_bytes.decode("utf-8"))
        except Exception:
            return False, "Malformed proof token payload"

        expected_sig = hmac.new(get_governance_secret(), payload_bytes, hashlib.sha256).hexdigest()
        if not hmac.compare_digest(expected_sig, sig):
            return False, "Invalid proof token cryptographic signature"

        t_principal = str(data.get("principal_id", ""))
        t_action = str(data.get("action", ""))
        t_target = str(data.get("target", ""))
        t_expires = int(data.get("expires_at", 0))

        if t_principal != principal_id and t_principal != "*":
            return False, f"Token principal mismatch: expected {principal_id}, got {t_principal}"
        now = int(time.time())
        if now > t_expires:
            return False, "Proof token has expired"
        if t_action != "*" and action != "*" and t_action != action:
            prefix = t_action.rstrip("*").rstrip("_")
            if not (t_action.endswith("*") and (action == prefix or action.startswith(prefix))):
                return False, f"Token action mismatch: expected {action}, got {t_action}"
        if t_target != "*" and target != "*" and t_target != target:
            t_prefix = t_target.rstrip("*").rstrip("_")
            if not (t_target.endswith("*") and (target == t_prefix or target.startswith(t_prefix))):
                return False, f"Token target mismatch: expected {target}, got {t_target}"
        return True, "Valid"

    # v1 fallback
    payload_parts = payload_str.split("|")
    if len(payload_parts) != 5:
        return False, "Malformed proof token payload"
    t_principal, t_action, t_target, t_expires, t_nonce = payload_parts
    if t_principal != principal_id and t_principal != "*":
        return False, f"Token principal mismatch: expected {principal_id}, got {t_principal}"
    now = int(time.time())
    try:
        if now > int(t_expires):
            return False, "Proof token has expired"
    except ValueError:
        return False, "Invalid expiration in proof token"
    if t_action != "*" and action != "*" and t_action != action:
        prefix = t_action.rstrip("*").rstrip("_")
        if not (t_action.endswith("*") and (action == prefix or action.startswith(prefix))):
            return False, f"Token action mismatch: expected {action}, got {t_action}"
    if t_target != "*" and target != "*" and t_target != target:
        t_prefix = t_target.rstrip("*").rstrip("_")
        if not (t_target.endswith("*") and (target == t_prefix or target.startswith(t_prefix))):
            return False, f"Token target mismatch: expected {target}, got {t_target}"

    expected_sig = hmac.new(get_governance_secret(), payload_str.encode("utf-8"), hashlib.sha256).hexdigest()
    if not hmac.compare_digest(expected_sig, sig):
        return False, "Invalid proof token cryptographic signature"
    return True, "Valid"


def _compute_grant_proof(
    grant_id: str,
    request_id: Optional[str],
    board_slug: str,
    previous_policy_hash: str,
    authorized_delta: int,
    approval_nonce: str,
    expires_at: int,
) -> str:
    msg = f"{grant_id}|{request_id or ''}|{board_slug}|{previous_policy_hash}|{authorized_delta}|{approval_nonce}|{expires_at}".encode("utf-8")
    return hmac.new(get_governance_secret(), msg, hashlib.sha256).hexdigest()




def ensure_governance_budget_schema(conn: sqlite3.Connection) -> None:
    """Ensure cumulative budget and mutation idempotency tables exist."""
    conn.execute("""
        CREATE TABLE IF NOT EXISTS governance_budget_ledger (
            board_slug       TEXT PRIMARY KEY,
            cumulative_tasks INTEGER NOT NULL DEFAULT 0,
            updated_at       INTEGER NOT NULL
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS governance_mutations (
            mutation_id      TEXT PRIMARY KEY,
            board_slug       TEXT NOT NULL,
            task_count       INTEGER NOT NULL,
            created_task_ids TEXT NOT NULL,
            created_at       INTEGER NOT NULL
        )
    """)



def is_governance_enabled(
    conn: Optional[sqlite3.Connection] = None,
    board_slug: Optional[str] = None,
) -> bool:
    """Determine whether governance enforcement is required for this board/connection.

    Explicitly distinguishes between non-governed boards and governed boards with missing/broken policies.
    """
    req_env = os.environ.get("HERMES_KANBAN_GOVERNANCE_REQUIRED")
    if req_env is not None and req_env.strip().lower() not in {"0", "false", "no", "off", ""}:
        return True

    if os.environ.get("HERMES_KANBAN_GOVERNANCE_POLICY"):
        return True

    if os.environ.get("HERMES_KANBAN_GOVERNANCE_POLICY_HASH"):
        return True

    if board_slug:
        try:
            from hermes_cli import kanban_db as kb
            meta = kb.read_board_metadata(board_slug)
            gov_meta = meta.get("governance")
            if isinstance(gov_meta, dict):
                if gov_meta.get("enabled") is False:
                    return False
                if gov_meta.get("enabled") is True:
                    return True
            if meta.get("governed") is True or meta.get("governance_required") is True:
                return True
        except Exception:
            pass

    if conn is not None:
        try:
            rows = conn.execute("PRAGMA database_list").fetchall()
            for row in rows:
                if row[1] == "main" and row[2]:
                    db_dir = Path(row[2]).parent
                    for fname in ("governance_policy.yaml", "governance_policy.yml", "governance_policy.json"):
                        if (db_dir / fname).is_file():
                            return True
                    meta_p = db_dir / "board.json"
                    if meta_p.is_file():
                        meta = json.loads(meta_p.read_text(encoding="utf-8"))
                        if isinstance(meta, dict):
                            gov_cfg = meta.get("governance")
                            if isinstance(gov_cfg, dict) and gov_cfg.get("enabled") is False:
                                return False
                            if isinstance(gov_cfg, dict) and gov_cfg.get("enabled") is True:
                                return True
                            if meta.get("governed") is True:
                                return True
        except Exception:
            pass

    if board_slug:
        try:
            from hermes_cli import kanban_db as kb
            b_dir = kb.board_dir(board_slug)
            for fname in ("governance_policy.yaml", "governance_policy.yml", "governance_policy.json"):
                if (b_dir / fname).is_file():
                    return True
        except Exception:
            pass

    if conn is not None:
        try:
            row = conn.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name='governance_events'"
            ).fetchone()
            if row:
                count_row = conn.execute("SELECT COUNT(*) FROM governance_events").fetchone()
                if count_row and count_row[0] > 0:
                    return True
        except Exception:
            pass

    return False


def resolve_governance_policy_path(
    conn: Optional[sqlite3.Connection] = None,
    board_slug: Optional[str] = None,
) -> Optional[Path]:
    """Find the governance policy file path for the current connection / board.

    Returns the resolved Path if configured or discoverable (even if the file does not
    currently exist on disk, so callers can differentiate between 'governance not configured'
    and 'governance policy missing').
    """
    env_override = os.environ.get("HERMES_KANBAN_GOVERNANCE_POLICY")
    if env_override:
        return Path(env_override).expanduser()

    if conn is not None:
        try:
            rows = conn.execute("PRAGMA database_list").fetchall()
            for row in rows:
                if row[1] == "main" and row[2]:
                    db_dir = Path(row[2]).parent
                    for fname in ("governance_policy.yaml", "governance_policy.yml", "governance_policy.json"):
                        candidate = db_dir / fname
                        if candidate.is_file():
                            return candidate
                    meta_p = db_dir / "board.json"
                    if meta_p.is_file():
                        try:
                            meta = json.loads(meta_p.read_text(encoding="utf-8"))
                            if isinstance(meta, dict) and (
                                meta.get("governance", {}).get("enabled") is True
                                or meta.get("governed") is True
                                or meta.get("governance_required") is True
                            ):
                                return db_dir / "governance_policy.yaml"
                        except Exception:
                            pass
        except Exception:
            pass

    if board_slug:
        try:
            from hermes_cli import kanban_db as kb
            b_dir = kb.board_dir(board_slug)
            for fname in ("governance_policy.yaml", "governance_policy.yml", "governance_policy.json"):
                candidate = b_dir / fname
                if candidate.is_file():
                    return candidate
            meta = kb.read_board_metadata(board_slug)
            if (
                meta.get("governance", {}).get("enabled") is True
                or meta.get("governed") is True
                or meta.get("governance_required") is True
            ):
                return b_dir / "governance_policy.yaml"
        except Exception:
            pass

    return None


def load_governance_policy(
    conn: Optional[sqlite3.Connection] = None,
    board_slug: Optional[str] = None,
    *,
    expected_hash: Optional[str] = None,
) -> Optional[tuple[dict[str, Any], str, Path]]:
    """Load policy dict, sha256 hash, and path.

    Fail-closed:
    - Missing required policy -> GovernancePolicyMissing
    - Unreadable policy -> GovernancePolicyMissing
    - Malformed YAML -> GovernancePolicyMalformed
    - Hash mismatch -> GovernancePolicyHashMismatch
    - Unsupported version -> GovernancePolicyUnsupportedVersion
    - Un-governed board with no policy -> returns None
    """
    enabled = is_governance_enabled(conn=conn, board_slug=board_slug)
    policy_path = resolve_governance_policy_path(conn=conn, board_slug=board_slug)

    if not policy_path or not policy_path.is_file():
        if enabled or policy_path is not None:
            raise GovernancePolicyMissing(
                f"GOVERNANCE POLICY MISSING: Required policy not found at {policy_path or 'unresolved path'}",
                policy_path=str(policy_path) if policy_path else None,
            )
        return None

    try:
        raw_bytes = policy_path.read_bytes()
    except Exception as exc:
        raise GovernancePolicyMissing(
            f"GOVERNANCE POLICY UNREADABLE: Could not read policy at {policy_path}: {exc}",
            policy_path=str(policy_path),
        ) from exc

    policy_hash = hashlib.sha256(raw_bytes).hexdigest()

    # Hash verification
    target_expected_hash = expected_hash or os.environ.get("HERMES_KANBAN_GOVERNANCE_POLICY_HASH")
    if not target_expected_hash and board_slug:
        try:
            from hermes_cli import kanban_db as kb
            meta = kb.read_board_metadata(board_slug)
            gov_cfg = meta.get("governance")
            if isinstance(gov_cfg, dict):
                target_expected_hash = gov_cfg.get("policy_hash") or gov_cfg.get("expected_hash")
            if not target_expected_hash:
                target_expected_hash = meta.get("policy_hash")
        except Exception:
            pass

    if target_expected_hash and policy_hash != target_expected_hash:
        raise GovernancePolicyHashMismatch(
            f"GOVERNANCE POLICY HASH MISMATCH: expected {target_expected_hash}, got {policy_hash}",
            policy_path=str(policy_path),
            expected_hash=target_expected_hash,
            actual_hash=policy_hash,
        )

    try:
        data = load_governance_yaml(raw_bytes)
        if not isinstance(data, dict):
            raise ValueError(f"Policy YAML must parse to a dict, got {type(data).__name__}")
    except Exception as exc:
        logger.warning("Failed to parse governance policy %s: %s", policy_path, exc)
        raise GovernancePolicyMalformed(
            f"GOVERNANCE POLICY MALFORMED: Invalid YAML in {policy_path}: {exc}",
            policy_path=str(policy_path),
        ) from exc

    # Version validation
    version = data.get("version") or data.get("policy_version") or data.get("p0_spec_version")
    if version is not None and str(version) not in {"1", "2", "3", "4", "5", "v1", "v2", "v3", "level5b-p0-v1"}:
        raise GovernancePolicyUnsupportedVersion(
            f"GOVERNANCE POLICY UNSUPPORTED VERSION: version {version} is not supported",
            policy_path=str(policy_path),
            version=str(version),
        )

    return data, policy_hash, policy_path


def _record_policy_denial(
    conn: sqlite3.Connection,
    *,
    reason_code: str,
    policy_id: str = "INVALID_POLICY",
    policy_hash: str = "INVALID_HASH",
    actor: str = "unknown",
    ingress: str = "db_api",
    operation: str = "kanban_create",
    details: Optional[dict[str, Any]] = None,
) -> Optional[str]:
    """Durably record a policy failure denial event into the governance ledger."""
    try:
        ensure_governance_schema(conn)
        event_id = f"gov_policy_denial_{secrets.token_hex(6)}"
        now = int(time.time())
        from hermes_cli.kanban_db_connect import write_txn

        with write_txn(conn):
            conn.execute(
                """
                INSERT INTO governance_events (
                    event_id, policy_id, policy_hash, actor, ingress,
                    requested_operation, decision, reason_code,
                    observed_task_count, configured_limit, details_json, timestamp
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    event_id,
                    str(policy_id),
                    str(policy_hash),
                    str(actor),
                    str(ingress),
                    str(operation),
                    "DENY",
                    str(reason_code),
                    0,
                    0,
                    json.dumps(details or {}, ensure_ascii=False),
                    now,
                ),
            )
        return event_id
    except Exception as exc:
        logger.warning("Could not record policy denial event: %s", exc)
        return None


def request_reauthorization(
    conn: sqlite3.Connection,
    *,
    board_slug: str,
    current_policy_hash: str,
    requested_delta: int,
    reason: str,
    requester: str = "p0-orchestrator",
    originating_denial_event: Optional[str] = None,
) -> ReauthorizationRequest:
    """Agent / P0 capability: create an auditable reauthorization request.

    Does NOT grant authorization or alter budget. Status is strictly PENDING.
    """
    ensure_governance_schema(conn)
    request_id = f"rr_{secrets.token_hex(6)}"
    now = int(time.time())

    from hermes_cli.kanban_db_connect import write_txn

    with write_txn(conn):
        conn.execute(
            """
            INSERT INTO governance_reauth_requests (
                request_id, board_slug, originating_denial_event,
                current_policy_hash, requested_delta, reason, requester, status, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, 'PENDING', ?)
            """,
            (
                request_id,
                board_slug,
                originating_denial_event,
                current_policy_hash,
                requested_delta,
                reason,
                requester,
                now,
            ),
        )
        conn.execute(
            """
            INSERT INTO governance_events (
                event_id, policy_id, policy_hash, actor, ingress,
                requested_operation, decision, reason_code,
                observed_task_count, configured_limit, details_json, timestamp
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                f"gov_req_{secrets.token_hex(6)}",
                "governance_policy",
                current_policy_hash,
                requester,
                "agent_request",
                "reauthorization_request",
                "REQUESTED",
                "REAUTHORIZATION_REQUESTED",
                0,
                0,
                json.dumps(
                    {
                        "event_type": "REAUTHORIZATION_REQUESTED",
                        "request_id": request_id,
                        "board_slug": board_slug,
                        "requested_delta": requested_delta,
                        "reason": reason,
                        "originating_denial_event": originating_denial_event,
                    },
                    ensure_ascii=False,
                ),
                now,
            ),
        )

    return ReauthorizationRequest(
        request_id=request_id,
        board_slug=board_slug,
        originating_denial_event=originating_denial_event,
        current_policy_hash=current_policy_hash,
        requested_delta=requested_delta,
        reason=reason,
        requester=requester,
        status="PENDING",
        created_at=now,
    )


def create_reauthorization_grant(
    *,
    board_slug: str,
    previous_policy_hash: str,
    authorized_delta: int,
    principal_context: TrustedPrincipalContext,
    request_id: Optional[str] = None,
    ttl_seconds: int = 3600,
    conn: Optional[sqlite3.Connection] = None,
) -> ReauthorizationGrant:
    """Human-only capability: issue a cryptographically bound single-use grant capability.

    Agents and unauthenticated principals are strictly forbidden from issuing grants.
    """
    principal_context.verify_human_authority(action="reauthorization_grant", target=board_slug)

    grant_id = f"rg_{secrets.token_hex(8)}"
    approval_nonce = secrets.token_hex(16)
    now = int(time.time())
    expires_at = now + ttl_seconds

    sig_proof = _compute_grant_proof(
        grant_id,
        request_id,
        board_slug,
        previous_policy_hash,
        authorized_delta,
        approval_nonce,
        expires_at,
    )

    grant = ReauthorizationGrant(
        grant_id=grant_id,
        request_id=request_id,
        board_slug=board_slug,
        previous_policy_hash=previous_policy_hash,
        authorized_delta=authorized_delta,
        principal_id=principal_context.principal_id,
        auth_method=principal_context.auth_method,
        approval_nonce=approval_nonce,
        signature_proof=sig_proof,
        status="ISSUED",
        issued_at=now,
        expires_at=expires_at,
    )

    if conn is not None:
        ensure_governance_schema(conn)
        from hermes_cli.kanban_db_connect import write_txn

        with write_txn(conn):
            conn.execute(
                """
                INSERT INTO governance_reauth_grants (
                    grant_id, request_id, board_slug, previous_policy_hash,
                    authorized_delta, principal_id, auth_method, approval_nonce,
                    signature_proof, status, issued_at, expires_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'ISSUED', ?, ?)
                """,
                (
                    grant_id,
                    request_id,
                    board_slug,
                    previous_policy_hash,
                    authorized_delta,
                    principal_context.principal_id,
                    principal_context.auth_method,
                    approval_nonce,
                    sig_proof,
                    now,
                    expires_at,
                ),
            )

    return grant


def apply_reauthorization_grant(
    conn: sqlite3.Connection,
    board_slug: str,
    grant: ReauthorizationGrant,
    principal_context: TrustedPrincipalContext,
    *,
    attempted_delta: Optional[int] = None,
) -> tuple[dict[str, Any], str]:
    """Apply an authenticated grant via Compare-and-Swap (CAS).

    Validates:
    - Trusted human principal required
    - Cryptographic signature proof
    - Target board matches grant scope
    - Attempted delta does not exceed grant authorized delta
    - Grant has not expired
    - Grant has not already been used (replay protection)
    - Active policy hash matches grant.previous_policy_hash (stale policy check)
    """
    ensure_governance_schema(conn)

    # 1. Trusted principal check
    principal_context.verify_human_authority(action="reauthorization_apply", target=board_slug)

    # 2. Cryptographic signature proof verification
    expected_proof = _compute_grant_proof(
        grant.grant_id,
        grant.request_id,
        grant.board_slug,
        grant.previous_policy_hash,
        grant.authorized_delta,
        grant.approval_nonce,
        grant.expires_at,
    )
    if not hmac.compare_digest(expected_proof, grant.signature_proof):
        raise PermissionError("GOVERNANCE DENIAL: Invalid authorization proof signature.")

    # 3. Scope binding: board check
    if grant.board_slug != board_slug:
        raise PermissionError(
            f"GOVERNANCE DENIAL: Scope escalation. Grant is bound to board '{grant.board_slug}', "
            f"cannot be applied to board '{board_slug}'."
        )

    # 4. Scope binding: delta check
    delta_to_apply = attempted_delta if attempted_delta is not None else grant.authorized_delta
    if delta_to_apply > grant.authorized_delta:
        raise PermissionError(
            f"GOVERNANCE DENIAL: Scope escalation. Attempted delta (+{delta_to_apply}) "
            f"exceeds authorized delta (+{grant.authorized_delta})."
        )

    # 5. Expiration check
    now = int(time.time())
    if now > grant.expires_at:
        raise PermissionError("GOVERNANCE DENIAL: Reauthorization grant has expired.")

    from hermes_cli.kanban_db_connect import write_txn

    with write_txn(conn):
        # 6. Replay protection
        row = conn.execute(
            "SELECT status FROM governance_reauth_grants WHERE grant_id = ?",
            (grant.grant_id,),
        ).fetchone()
        if row and row[0] == "USED":
            raise PermissionError("GOVERNANCE DENIAL: Reauthorization replay. Grant has already been used.")

        # 7. Compare-And-Swap (CAS) against current active policy
        loaded = load_governance_policy(conn=conn, board_slug=board_slug)
        if loaded is None:
            raise GovernancePolicyMissing(f"No policy found for board '{board_slug}' to reauthorize.")

        curr_policy, curr_hash, policy_path = loaded
        if curr_hash != grant.previous_policy_hash:
            raise PermissionError(
                f"GOVERNANCE DENIAL: STALE_AUTHORIZATION. Active policy hash {curr_hash[:12]}... "
                f"does not match grant previous hash {grant.previous_policy_hash[:12]}..."
            )

        # Update policy in memory & on disk
        old_limit = int(curr_policy.get("budget", {}).get("max_total_tasks", 0))
        new_limit = old_limit + delta_to_apply
        curr_policy.setdefault("budget", {})["max_total_tasks"] = new_limit
        curr_version = curr_policy.get("version", 1)
        curr_policy["version"] = (int(curr_version) + 1) if isinstance(curr_version, int) else 2
        curr_policy["last_reauthorization"] = {
            "grant_id": grant.grant_id,
            "principal_id": principal_context.principal_id,
            "delta": delta_to_apply,
            "timestamp": now,
        }

        policy_path.write_text(dump_governance_yaml(curr_policy), encoding="utf-8")
        new_policy_hash = hashlib.sha256(policy_path.read_bytes()).hexdigest()

        # Update grant status to USED
        conn.execute(
            """
            INSERT INTO governance_reauth_grants (
                grant_id, request_id, board_slug, previous_policy_hash,
                authorized_delta, principal_id, auth_method, approval_nonce,
                signature_proof, status, issued_at, expires_at, used_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'USED', ?, ?, ?)
            ON CONFLICT(grant_id) DO UPDATE SET
                status = 'USED',
                used_at = excluded.used_at
            """,
            (
                grant.grant_id,
                grant.request_id,
                grant.board_slug,
                grant.previous_policy_hash,
                grant.authorized_delta,
                principal_context.principal_id,
                principal_context.auth_method,
                grant.approval_nonce,
                grant.signature_proof,
                grant.issued_at,
                grant.expires_at,
                now,
            ),
        )

        if grant.request_id:
            conn.execute(
                "UPDATE governance_reauth_requests SET status = 'APPROVED' WHERE request_id = ?",
                (grant.request_id,),
            )

        # Autonomous Wake-up: transition tasks blocked on BUDGET_EXHAUSTED to ready
        blocked_tasks = conn.execute(
            """
            SELECT id FROM tasks
            WHERE status = 'blocked'
              AND (
                  id IN (
                      SELECT task_id FROM task_events
                      WHERE kind = 'blocked' AND (payload LIKE '%BUDGET_EXHAUSTED%' OR payload LIKE '%budget%')
                  )
                  OR last_failure_error LIKE '%BUDGET_EXHAUSTED%'
              )
            """
        ).fetchall()
        for b_row in blocked_tasks:
            b_tid = b_row["id"]
            conn.execute("UPDATE tasks SET status = 'ready' WHERE id = ?", (b_tid,))
            from hermes_cli.kanban_db import _append_event
            _append_event(
                conn,
                b_tid,
                "reauthorized_wakeup",
                {
                    "grant_id": grant.grant_id,
                    "request_id": grant.request_id,
                    "previous_state": "BLOCKED_BUDGET_EXHAUSTED",
                    "new_status": "ready",
                    "authorized_delta": delta_to_apply,
                },
            )

        # Record durable REAUTHORIZATION_GRANTED audit event
        record_reauthorization_event(
            conn,
            previous_policy={"budget": {"max_total_tasks": old_limit}},
            new_policy={"budget": {"max_total_tasks": new_limit}, "policy_id": curr_policy.get("policy_id", "governance_policy"), "policy_hash": new_policy_hash},
            principal_context=principal_context,
            added_capacity=delta_to_apply,
            originating_denial_event=grant.request_id,
            reason=f"Human grant {grant.grant_id} applied (+{delta_to_apply} tasks)",
        )

    return curr_policy, new_policy_hash


def record_reauthorization_event(
    conn: sqlite3.Connection,
    *,
    previous_policy: dict[str, Any],
    new_policy: dict[str, Any],
    actor: str = "human",
    actor_class: str = "HUMAN",
    reason: str = "approved remediation continuation",
    originating_denial_event: Optional[str] = None,
    added_capacity: int = 3,
    principal_context: Optional[TrustedPrincipalContext] = None,
) -> str:
    """Durably record a human reauthorization event in the governance ledger.

    Security Boundary:
    Caller-supplied `actor_class="HUMAN"` is NOT authoritative.
    If `principal_context` is missing or not an authenticated human principal,
    this operation is DENIED (raises PermissionError) and logged as a violation.
    """
    ensure_governance_schema(conn)

    is_trusted_human = False
    if principal_context is not None:
        try:
            principal_context.verify_human_authority(action="reauthorization", target="*")
            is_trusted_human = True
        except PermissionError:
            is_trusted_human = False

    if not is_trusted_human:
        now = int(time.time())
        denial_id = f"gov_denial_{secrets.token_hex(6)}"
        from hermes_cli.kanban_db_connect import write_txn

        with write_txn(conn, allow_nested=True):
            conn.execute(
                """
                INSERT INTO governance_events (\n                    event_id, policy_id, policy_hash, actor, ingress,
                    requested_operation, decision, reason_code,
                    observed_task_count, configured_limit, details_json, timestamp
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    denial_id,
                    new_policy.get("policy_id", "governance_policy"),
                    new_policy.get("policy_hash", ""),
                    str(actor),
                    "governance_console",
                    "policy_reauthorization",
                    "DENY",
                    "UNAUTHORIZED_PRINCIPAL",
                    int(previous_policy.get("budget", {}).get("max_total_tasks", 0)),
                    int(new_policy.get("budget", {}).get("max_total_tasks", 0)),
                    json.dumps(
                        {
                            "event_type": "REAUTHORIZATION_DENIED",
                            "attempted_actor": actor,
                            "attempted_actor_class": actor_class,
                            "reason": "Self-asserted human identity is not authoritative; requires trusted human principal context",
                            "originating_denial_event": originating_denial_event,
                        },
                        ensure_ascii=False,
                    ),
                    now,
                ),
            )
        raise PermissionError(
            f"GOVERNANCE DENIAL: actor '{actor}' self-asserting actor_class='{actor_class}' is unauthorized. "
            "Reauthorization requires an authenticated human principal context."
        )

    assert principal_context is not None
    event_id = f"gov_reauth_{secrets.token_hex(6)}"
    now = int(time.time())
    from hermes_cli.kanban_db_connect import write_txn

    with write_txn(conn, allow_nested=True):
        conn.execute(
            """
            INSERT INTO governance_events (
                event_id, policy_id, policy_hash, actor, ingress,
                requested_operation, decision, reason_code,
                observed_task_count, configured_limit, details_json, timestamp
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                event_id,
                new_policy.get("policy_id", "governance_policy"),
                new_policy.get("policy_hash", ""),
                str(principal_context.principal_id),
                "governance_console",
                "policy_reauthorization",
                "ALLOW",
                "REAUTHORIZED",
                int(previous_policy.get("budget", {}).get("max_total_tasks", 0)),
                int(new_policy.get("budget", {}).get("max_total_tasks", 0)),
                json.dumps(
                    {
                        "event_type": "REAUTHORIZATION_GRANTED",
                        "principal": {
                            "id": principal_context.principal_id,
                            "type": "HUMAN",
                            "auth_method": principal_context.auth_method,
                            "derived_from_trusted_context": True,
                        },
                        "reason": reason,
                        "originating_denial_event": originating_denial_event,
                        "added_task_capacity": added_capacity,
                        "previous_policy": previous_policy,
                        "new_policy": new_policy,
                    },
                    ensure_ascii=False,
                ),
                now,
            ),
        )
    return event_id


def check_governance_mutation(
    conn: sqlite3.Connection,
    *,
    operation: str = "kanban_create",
    actor: str = "unknown",
    ingress: str = "db_api",
    board: Optional[str] = None,
    requested_count: int = 1,
    details: Optional[dict[str, Any]] = None,
) -> None:
    """PEP: evaluate immutable policy before committing any graph mutation.

    On violation: durably records DenialEvent to SQLite ledger and raises GovernanceBudgetExceeded or GovernancePolicyError.
    Zero task rows or graph edges are ever committed on denial.
    Supports atomic batch reservations: if observed_count + requested_count > max_total_tasks,
    the entire batch is denied.
    """
    if not is_governance_enabled(conn=conn, board_slug=board):
        return

    try:
        loaded = load_governance_policy(conn=conn, board_slug=board)
    except GovernancePolicyError as exc:
        eid = _record_policy_denial(
            conn,
            reason_code=exc.reason_code,
            policy_id=exc.policy_id or "INVALID_POLICY",
            policy_hash=exc.policy_hash or "INVALID_HASH",
            actor=actor,
            ingress=ingress,
            operation=operation,
            details={"error": str(exc), "semantic_state": exc.semantic_state},
        )
        exc.event_id = eid or ""
        raise

    if loaded is None:
        raise GovernancePolicyMissing("Governance is required but no policy could be resolved.")

    policy, policy_hash, _ = loaded
    policy_id = str(policy.get("policy_id", "governance_policy"))
    budget = policy.get("budget") or {}
    max_total_tasks = budget.get("max_total_tasks")

    if max_total_tasks is None:
        return

    from hermes_cli.kanban_db_connect import write_txn

    denied = False
    denial_details: dict[str, Any] = {}

    with write_txn(conn):
        ensure_governance_budget_schema(conn)
        target_board = board or "default"
        ledger_row = conn.execute(
            "SELECT cumulative_tasks FROM governance_budget_ledger WHERE board_slug = ?",
            (target_board,),
        ).fetchone()
        persisted = conn.execute("SELECT COUNT(*) FROM tasks").fetchone()[0]
        observed_count = max(persisted, ledger_row[0] if ledger_row else 0)

        if observed_count + requested_count > int(max_total_tasks):
            ensure_governance_schema(conn)
            event_id = f"gov_denial_{secrets.token_hex(6)}"
            now = int(time.time())

            merged_details = dict(details or {})
            merged_details["requested_count"] = requested_count

            conn.execute(
                """
                INSERT INTO governance_events (
                    event_id, policy_id, policy_hash, actor, ingress,
                    requested_operation, decision, reason_code,
                    observed_task_count, configured_limit, details_json, timestamp
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    event_id,
                    policy_id,
                    policy_hash,
                    str(actor),
                    str(ingress),
                    str(operation),
                    "DENY",
                    "BUDGET_EXHAUSTED",
                    observed_count,
                    int(max_total_tasks),
                    json.dumps(merged_details, ensure_ascii=False),
                    now,
                ),
            )
            denied = True
            denial_details = {
                "observed_count": observed_count,
                "event_id": event_id,
                "policy_id": policy_id,
                "policy_hash": policy_hash,
                "max_total_tasks": int(max_total_tasks),
            }

    if denied:
        permitted_next = policy.get("enforcement", {}).get("permitted_next_actions")
        semantic_state = policy.get("enforcement", {}).get("semantic_state", "BLOCKED_BUDGET_EXHAUSTED")

        raise GovernanceBudgetExceeded(
            f"GOVERNANCE DENIAL: budget exhausted ({denial_details['observed_count']} + {requested_count} > {denial_details['max_total_tasks']} tasks)",
            code="BUDGET_EXHAUSTED",
            policy_id=denial_details["policy_id"],
            policy_hash=denial_details["policy_hash"],
            actor=str(actor),
            ingress=str(ingress),
            observed_task_count=denial_details["observed_count"],
            configured_limit=denial_details["max_total_tasks"],
            permitted_next_actions=permitted_next,
            semantic_state=semantic_state,
            event_id=denial_details["event_id"],
        )
