"""Hermes cross-profile control-plane database.

This module is deliberately small and stdlib-only.  It is the durable local
control plane for cross-profile messages, approvals, dispatches, route policy,
and mirror/outbox work.  Discord and other gateways are projections/ingress;
they are not authoritative state.
"""
from __future__ import annotations

import contextlib
import hashlib
import hmac
import json
import os
import re
import secrets
import sqlite3
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Literal

from hermes_constants import get_default_hermes_root

SCHEMA_VERSION = 3
AuthorityMode = Literal["legacy", "shadow", "control_db"]

_SCHEMA_LOCK_PATHS: set[str] = set()
_SCHEMA_INIT_LOCK = threading.RLock()

_SECRET_KEY_PATTERN = r"(?:api[_-]?key|access[_-]?token|refresh[_-]?token|id[_-]?token|auth[_-]?token|token|secret|password|passwd|private[_-]?key|aws[_-]?secret[_-]?access[_-]?key|database[_-]?url)"
_SECRET_PATTERNS = [
    re.compile(r"(?is)-----BEGIN [^-]*PRIVATE KEY-----.*?-----END [^-]*PRIVATE KEY-----"),
    # Quoted assignments may contain spaces/newlines; redact the full quoted value.
    re.compile(rf"(?is)(\b{_SECRET_KEY_PATTERN}\b\s*[:=]\s*)\"(?:\\.|[^\"\\])*\""),
    re.compile(rf"(?is)(\b{_SECRET_KEY_PATTERN}\b\s*[:=]\s*)'(?:\\.|[^'\\])*'"),
    # Redact whole auth headers before generic assignment matching; otherwise
    # "Authorization: Bearer *** can leave the token behind.
    re.compile(r"(?i)(authorization\s*[:=]\s*)(?:bearer\s+)?[^\s,;\]}]+"),
    re.compile(r"(?i)(bearer\s+)[A-Za-z0-9._~+/=-]{12,}"),
    re.compile(rf"(?i)(\b{_SECRET_KEY_PATTERN}\b\s*[:=]\s*)([^\s,;\]}}]+)"),
    re.compile(r"(?i)((?:postgres|postgresql|mysql|mariadb|redis|mongodb|amqp|https?)://[^\s:/@]+:)[^\s/@]+(@)"),
    re.compile(r"(?i)(-u\s+[^\s:]+:)[^\s]+"),
    re.compile(r"(?i)(sk-[A-Za-z0-9]{12,})"),
]
_APPROVAL_DECISIONS = {"approved", "denied"}

_TEXT_COLUMNS = {
    "body",
    "metadata_json",
    "payload_json",
    "event_json",
    "command_preview",
    "tool_args_preview",
    "decision_reason",
    "last_error",
    "summary",
}

_ADMIN_ACTOR_TYPES = {"admin", "bootstrap"}
_DISPATCH_ADVANCE_STATUSES = {"running", "completed", "failed", "blocked"}
_CLAIMABLE_DISPATCH_STATUSES = {"pending", "retry_ready"}
ACTIVE_MESSAGE_STATUSES = {"pending", "delivered"}
TERMINAL_MESSAGE_STATUSES = {"acknowledged", "resolved", "superseded", "cancelled"}
MESSAGE_STATUSES = ACTIVE_MESSAGE_STATUSES | TERMINAL_MESSAGE_STATUSES
BOOTSTRAP_INSTANCE_LEASE_MS = 3_600_000


class ControlPlaneError(RuntimeError):
    pass


class RouteDenied(ControlPlaneError):
    pass


class RedactionFailed(ControlPlaneError):
    pass


def now_ms() -> int:
    return int(time.time() * 1000)


def _ts(now: int | None = None) -> int:
    return now_ms() if now is None else int(now)


def control_db_dir(root: Path | None = None) -> Path:
    return (root or get_default_hermes_root()) / "control-plane"


def control_db_path(root: Path | None = None) -> Path:
    return control_db_dir(root) / "control.db"


def _pepper_path(root: Path | None = None) -> Path:
    return control_db_dir(root) / ".pepper"


def _secure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True, mode=0o700)
    with contextlib.suppress(OSError):
        os.chmod(path, 0o700)


def _secure_file(path: Path) -> None:
    with contextlib.suppress(OSError):
        os.chmod(path, 0o600)


def _reject_symlink(path: Path) -> None:
    try:
        if path.is_symlink():
            raise ControlPlaneError(f"refusing symlinked control-plane path: {path}")
    except OSError as exc:
        raise ControlPlaneError(f"could not stat control-plane path {path}: {exc}") from exc


def _read_valid_pepper(path: Path, *, wait: bool = False) -> bytes | None:
    deadline = time.time() + 5.0
    while True:
        _reject_symlink(path)
        _secure_file(path)
        data = path.read_bytes().strip()
        if re.fullmatch(rb"[0-9a-fA-F]{64}", data):
            return data.lower()
        if not wait or time.time() >= deadline:
            if data:
                raise ControlPlaneError(f"invalid control-plane pepper at {path}")
            return None
        time.sleep(0.02)


def _load_or_create_pepper(root: Path | None = None) -> bytes:
    path = _pepper_path(root)
    _secure_dir(path.parent)
    if path.exists():
        existing = _read_valid_pepper(path, wait=True)
        if existing:
            return existing
    pepper = secrets.token_hex(32).encode("ascii")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        fd = os.open(path, flags, 0o600)
    except FileExistsError:
        existing = _read_valid_pepper(path, wait=True)
        if existing:
            return existing
        raise ControlPlaneError(f"empty control-plane pepper at {path}")
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(pepper + b"\n")
            f.flush()
            os.fsync(f.fileno())
    except Exception:
        with contextlib.suppress(OSError):
            path.unlink()
        raise
    _secure_file(path)
    return pepper


def hmac_id(value: str, *, root: Path | None = None) -> str:
    pepper = _load_or_create_pepper(root)
    return hmac.new(pepper, str(value).encode("utf-8"), hashlib.sha256).hexdigest()


def redact_text(text: str) -> str:
    if text is None:
        return ""
    if not isinstance(text, str):
        raise RedactionFailed(f"redact_text requires str, got {type(text).__name__}")
    redacted = text
    for pattern in _SECRET_PATTERNS:
        def repl(match: re.Match[str]) -> str:
            if match.lastindex and match.lastindex >= 2:
                suffix = match.group(match.lastindex) if match.group(match.lastindex) == "@" else ""
                return f"{match.group(1)}***{suffix}"
            if match.lastindex and match.lastindex >= 1:
                return f"{match.group(1)}***"
            return "***"
        redacted = pattern.sub(repl, redacted)
    # Fail closed if obvious secrets remain after redaction.
    lowered = redacted.lower()
    if re.search(rf"(?is)\b{_SECRET_KEY_PATTERN}\b\s*[:=]\s*(?!\*+(?:[\s,;\]}}]|$))[^\s,;\]}}]+", redacted):
        raise RedactionFailed("secret-like assignment survived redaction")
    if re.search(r"(?is)-----BEGIN [^-]*PRIVATE KEY-----.*?-----END [^-]*PRIVATE KEY-----", redacted):
        raise RedactionFailed("private key survived redaction")
    if "bearer " in lowered and "bearer ***" not in lowered:
        raise RedactionFailed("bearer credential survived redaction")
    if re.search(r"(?i)(?:postgres|postgresql|mysql|mariadb|redis|mongodb|amqp|https?)://[^\s:/@]+:[^*\s/@]+@", redacted):
        raise RedactionFailed("credential-bearing URL survived redaction")
    return redacted


def redact_jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, str):
        return redact_text(value)
    if isinstance(value, list):
        return [redact_jsonable(v) for v in value]
    if isinstance(value, dict):
        out: dict[str, Any] = {}
        for key, val in value.items():
            k = str(key)
            if re.search(rf"(?i){_SECRET_KEY_PATTERN}|authorization", k):
                out[k] = "***"
            else:
                out[k] = redact_jsonable(val)
        return out
    return redact_text(str(value))


def dumps_redacted(value: Any) -> str:
    return json.dumps(redact_jsonable(value), sort_keys=True, separators=(",", ":"))


def connect(path: Path | None = None, *, root: Path | None = None) -> sqlite3.Connection:
    db_path = path or control_db_path(root)
    _secure_dir(db_path.parent)
    if db_path.exists():
        _reject_symlink(db_path)
    if not db_path.exists():
        try:
            fd = os.open(db_path, os.O_RDWR | os.O_CREAT | os.O_EXCL, 0o600)
        except FileExistsError:
            pass
        else:
            os.close(fd)
    _secure_file(db_path)
    conn = sqlite3.connect(str(db_path), timeout=30.0, isolation_level=None)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys=ON")
    conn.execute("PRAGMA busy_timeout=30000")
    with _SCHEMA_INIT_LOCK:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        init_schema(conn)
    return conn


def _conn_root(conn: sqlite3.Connection) -> Path | None:
    row = conn.execute("PRAGMA database_list").fetchone()
    if not row or not row[2]:
        return None
    db_file = Path(row[2]).resolve()
    if db_file.name == "control.db" and db_file.parent.name == "control-plane":
        return db_file.parent.parent
    return db_file.parent


@contextlib.contextmanager
def transaction(conn: sqlite3.Connection) -> Iterator[sqlite3.Connection]:
    conn.execute("BEGIN IMMEDIATE")
    try:
        yield conn
    except Exception:
        conn.rollback()
        raise
    else:
        conn.commit()


def init_schema(conn: sqlite3.Connection) -> None:
    db_file = conn.execute("PRAGMA database_list").fetchone()[2]
    resolved = str(Path(db_file).resolve()) if db_file else f":memory:{id(conn)}"
    def _validate_existing() -> bool:
        existing_tables = {
            row[0]
            for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name LIKE 'cp_%'").fetchall()
        }
        existing_version = None
        if "cp_meta" in existing_tables:
            row = conn.execute("SELECT value FROM cp_meta WHERE key='schema_version'").fetchone()
            existing_version = row[0] if row else None
        if existing_tables and existing_version not in {"1", "2", str(SCHEMA_VERSION)}:
            raise ControlPlaneError(f"unsupported control DB schema_version={existing_version!r}; expected {SCHEMA_VERSION}")
        base_required = {"cp_meta", "cp_profiles", "cp_profile_instances", "cp_route_policies", "cp_messages", "cp_dispatches", "cp_approvals", "cp_outbox"}
        v3_required = base_required | {"cp_status_events", "cp_blockers", "cp_supervision_runs", "cp_runtime_mappings"}
        if "cp_meta" in existing_tables and not base_required.issubset(existing_tables):
            missing = sorted(base_required - existing_tables)
            raise ControlPlaneError(f"partial control DB schema missing tables: {missing}")
        if existing_version == str(SCHEMA_VERSION) and not v3_required.issubset(existing_tables):
            missing = sorted(v3_required - existing_tables)
            raise ControlPlaneError(f"partial control DB schema missing tables: {missing}")
        return bool(existing_tables) and existing_version == str(SCHEMA_VERSION)

    if resolved in _SCHEMA_LOCK_PATHS:
        if _validate_existing():
            return
    with _SCHEMA_INIT_LOCK:
        if resolved in _SCHEMA_LOCK_PATHS:
            if _validate_existing():
                return
        _validate_existing()
        with transaction(conn):
            _create_schema(conn)
            conn.execute(
                "INSERT INTO cp_meta(key,value,updated_at_ms) VALUES('schema_version',?,?) "
                "ON CONFLICT(key) DO UPDATE SET value=excluded.value, updated_at_ms=excluded.updated_at_ms",
                (str(SCHEMA_VERSION), now_ms()),
            )
            conn.execute(
                "INSERT OR IGNORE INTO cp_meta(key,value,updated_at_ms) VALUES('authority_mode','shadow',?)",
                (now_ms(),),
            )
        _SCHEMA_LOCK_PATHS.add(resolved)


def _create_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS cp_meta(
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL,
            updated_at_ms INTEGER NOT NULL
        );
        CREATE TABLE IF NOT EXISTS cp_profiles(
            profile_id TEXT PRIMARY KEY,
            role TEXT NOT NULL DEFAULT 'worker',
            display_name TEXT,
            created_at_ms INTEGER NOT NULL,
            updated_at_ms INTEGER NOT NULL
        );
        CREATE TABLE IF NOT EXISTS cp_profile_instances(
            instance_id TEXT PRIMARY KEY,
            profile_id TEXT NOT NULL REFERENCES cp_profiles(profile_id),
            pid INTEGER,
            host TEXT,
            started_at_ms INTEGER NOT NULL,
            heartbeat_at_ms INTEGER NOT NULL,
            lease_expires_at_ms INTEGER,
            status TEXT NOT NULL DEFAULT 'online',
            metadata_json TEXT NOT NULL DEFAULT '{}'
        );
        CREATE TABLE IF NOT EXISTS cp_route_policies(
            policy_id TEXT PRIMARY KEY,
            priority INTEGER NOT NULL DEFAULT 0,
            effect TEXT NOT NULL CHECK(effect IN ('allow','deny')),
            sender_profile TEXT NOT NULL DEFAULT '*',
            receiver_profile TEXT NOT NULL DEFAULT '*',
            kind TEXT NOT NULL DEFAULT '*',
            capability TEXT NOT NULL DEFAULT '*',
            created_by TEXT NOT NULL,
            created_by_type TEXT NOT NULL,
            created_at_ms INTEGER NOT NULL,
            UNIQUE(priority,effect,sender_profile,receiver_profile,kind,capability)
        );
        CREATE TABLE IF NOT EXISTS cp_messages(
            message_id TEXT PRIMARY KEY,
            kind TEXT NOT NULL,
            sender_profile TEXT NOT NULL,
            receiver_profile TEXT NOT NULL,
            capability TEXT NOT NULL DEFAULT 'message',
            body TEXT NOT NULL,
            metadata_json TEXT NOT NULL DEFAULT '{}',
            status TEXT NOT NULL DEFAULT 'pending',
            created_at_ms INTEGER NOT NULL,
            updated_at_ms INTEGER NOT NULL,
            idempotency_key TEXT UNIQUE
        );
        CREATE TABLE IF NOT EXISTS cp_message_events(
            event_id TEXT PRIMARY KEY,
            message_id TEXT NOT NULL REFERENCES cp_messages(message_id),
            event_type TEXT NOT NULL,
            event_json TEXT NOT NULL DEFAULT '{}',
            created_at_ms INTEGER NOT NULL
        );
        CREATE TABLE IF NOT EXISTS cp_dispatches(
            dispatch_id TEXT PRIMARY KEY,
            message_id TEXT REFERENCES cp_messages(message_id),
            sender_profile TEXT NOT NULL,
            receiver_profile TEXT NOT NULL,
            capability TEXT NOT NULL DEFAULT 'dispatch',
            status TEXT NOT NULL DEFAULT 'pending',
            payload_json TEXT NOT NULL DEFAULT '{}',
            parent_dispatch_id TEXT REFERENCES cp_dispatches(dispatch_id),
            dispatch_schema TEXT,
            max_wall_time_ms INTEGER,
            attempt INTEGER NOT NULL DEFAULT 0,
            blocked_at_ms INTEGER,
            completed_at_ms INTEGER,
            lease_instance_id TEXT REFERENCES cp_profile_instances(instance_id),
            lease_epoch INTEGER NOT NULL DEFAULT 0,
            lease_expires_at_ms INTEGER,
            attempts INTEGER NOT NULL DEFAULT 0,
            max_attempts INTEGER NOT NULL DEFAULT 3,
            last_error TEXT,
            created_at_ms INTEGER NOT NULL,
            updated_at_ms INTEGER NOT NULL,
            idempotency_key TEXT UNIQUE
        );
        CREATE TABLE IF NOT EXISTS cp_dispatch_events(
            event_id TEXT PRIMARY KEY,
            dispatch_id TEXT NOT NULL REFERENCES cp_dispatches(dispatch_id),
            event_type TEXT NOT NULL,
            event_json TEXT NOT NULL DEFAULT '{}',
            created_at_ms INTEGER NOT NULL
        );
        CREATE TABLE IF NOT EXISTS cp_artifacts(
            artifact_id TEXT PRIMARY KEY,
            dispatch_id TEXT REFERENCES cp_dispatches(dispatch_id),
            instance_id TEXT,
            lease_epoch INTEGER,
            path TEXT NOT NULL,
            summary TEXT,
            metadata_json TEXT NOT NULL DEFAULT '{}',
            created_at_ms INTEGER NOT NULL
        );
        CREATE TABLE IF NOT EXISTS cp_dispatch_results(
            dispatch_id TEXT NOT NULL REFERENCES cp_dispatches(dispatch_id),
            lease_epoch INTEGER NOT NULL,
            instance_id TEXT NOT NULL REFERENCES cp_profile_instances(instance_id),
            result_json TEXT NOT NULL,
            created_at_ms INTEGER NOT NULL,
            PRIMARY KEY(dispatch_id, lease_epoch)
        );
        CREATE TABLE IF NOT EXISTS cp_approvals(
            approval_id TEXT PRIMARY KEY,
            requester_profile TEXT NOT NULL,
            requester_instance_id TEXT NOT NULL,
            approver_profile TEXT NOT NULL,
            dispatch_id TEXT,
            lease_epoch INTEGER,
            cwd TEXT,
            affected_paths_json TEXT NOT NULL DEFAULT '[]',
            operation_class TEXT,
            risk_classification TEXT,
            normalized_command_preview TEXT,
            reason_requested TEXT,
            request_context_json TEXT NOT NULL DEFAULT '{}',
            decision_by_instance_id TEXT,
            decision_at_ms INTEGER,
            version INTEGER NOT NULL DEFAULT 1,
            status TEXT NOT NULL DEFAULT 'pending',
            command_preview TEXT NOT NULL,
            tool_args_preview TEXT,
            decision TEXT,
            decision_reason TEXT,
            consumed_by_instance_id TEXT,
            consumed_at_ms INTEGER,
            expires_at_ms INTEGER NOT NULL,
            created_at_ms INTEGER NOT NULL,
            updated_at_ms INTEGER NOT NULL
        );
        CREATE TABLE IF NOT EXISTS cp_outbox(
            outbox_id TEXT PRIMARY KEY,
            subject_type TEXT NOT NULL,
            subject_id TEXT NOT NULL,
            subject_version INTEGER NOT NULL,
            event_id TEXT,
            target_platform TEXT,
            target_ref_hmac TEXT,
            payload_json TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'pending',
            attempts INTEGER NOT NULL DEFAULT 0,
            last_error TEXT,
            created_at_ms INTEGER NOT NULL,
            updated_at_ms INTEGER NOT NULL
        );
        CREATE TABLE IF NOT EXISTS cp_inbound_receipts(
            receipt_id TEXT PRIMARY KEY,
            platform TEXT NOT NULL,
            external_id_hmac TEXT NOT NULL,
            subject_type TEXT,
            subject_id TEXT,
            payload_json TEXT NOT NULL,
            created_at_ms INTEGER NOT NULL,
            UNIQUE(platform, external_id_hmac)
        );
        CREATE TABLE IF NOT EXISTS cp_mirror_state(
            subject_type TEXT NOT NULL,
            subject_id TEXT NOT NULL,
            target_platform TEXT NOT NULL,
            target_ref_hmac TEXT NOT NULL,
            subject_version INTEGER NOT NULL,
            external_id_hmac TEXT,
            status TEXT NOT NULL,
            updated_at_ms INTEGER NOT NULL,
            PRIMARY KEY(subject_type, subject_id, target_platform, target_ref_hmac)
        );
        CREATE TABLE IF NOT EXISTS cp_runtime_mappings(
            control_profile_id TEXT PRIMARY KEY REFERENCES cp_profiles(profile_id),
            runtime_profile TEXT NOT NULL,
            role TEXT NOT NULL DEFAULT 'worker',
            enabled INTEGER NOT NULL DEFAULT 1,
            created_at_ms INTEGER NOT NULL,
            updated_at_ms INTEGER NOT NULL
        );
        CREATE TABLE IF NOT EXISTS cp_status_events(
            event_id TEXT PRIMARY KEY,
            profile_id TEXT NOT NULL REFERENCES cp_profiles(profile_id),
            instance_id TEXT NOT NULL REFERENCES cp_profile_instances(instance_id),
            dispatch_id TEXT REFERENCES cp_dispatches(dispatch_id),
            parent_dispatch_id TEXT REFERENCES cp_dispatches(dispatch_id),
            status TEXT NOT NULL CHECK(status IN ('starting','claimed','running','waiting_approval','blocked','verifying','completed','failed','cancelled','stalled')),
            summary TEXT NOT NULL,
            details_json TEXT NOT NULL DEFAULT '{}',
            created_at_ms INTEGER NOT NULL,
            version INTEGER NOT NULL DEFAULT 1
        );
        CREATE TABLE IF NOT EXISTS cp_blockers(
            blocker_id TEXT PRIMARY KEY,
            dispatch_id TEXT NOT NULL REFERENCES cp_dispatches(dispatch_id),
            profile_id TEXT NOT NULL REFERENCES cp_profiles(profile_id),
            instance_id TEXT NOT NULL REFERENCES cp_profile_instances(instance_id),
            severity TEXT NOT NULL CHECK(severity IN ('info','warning','blocked','critical')),
            kind TEXT NOT NULL CHECK(kind IN ('approval_needed','missing_context','test_failure','review_failure','dependency','auth','policy','runtime','other')),
            summary TEXT NOT NULL,
            details_json TEXT NOT NULL DEFAULT '{}',
            requires_response INTEGER NOT NULL DEFAULT 1,
            response_profile TEXT REFERENCES cp_profiles(profile_id),
            response_deadline_at_ms INTEGER,
            status TEXT NOT NULL CHECK(status IN ('open','acknowledged','resolved','superseded','cancelled')) DEFAULT 'open',
            resolution_json TEXT,
            created_at_ms INTEGER NOT NULL,
            updated_at_ms INTEGER NOT NULL
        );
        CREATE TABLE IF NOT EXISTS cp_supervision_runs(
            run_id TEXT PRIMARY KEY,
            started_at_ms INTEGER NOT NULL,
            finished_at_ms INTEGER,
            actor_profile TEXT NOT NULL REFERENCES cp_profiles(profile_id),
            actor_instance_id TEXT NOT NULL REFERENCES cp_profile_instances(instance_id),
            scope_json TEXT NOT NULL,
            findings_json TEXT NOT NULL DEFAULT '[]',
            actions_json TEXT NOT NULL DEFAULT '[]',
            status TEXT NOT NULL CHECK(status IN ('running','completed','failed'))
        );
        """
    )
    _ensure_column(conn, "cp_artifacts", "instance_id", "TEXT")
    _ensure_column(conn, "cp_artifacts", "lease_epoch", "INTEGER")
    _ensure_column(conn, "cp_artifacts", "metadata_json", "TEXT NOT NULL DEFAULT '{}'")
    for column, definition in {
        "parent_dispatch_id": "TEXT REFERENCES cp_dispatches(dispatch_id)",
        "dispatch_schema": "TEXT",
        "max_wall_time_ms": "INTEGER",
        "attempt": "INTEGER NOT NULL DEFAULT 0",
        "blocked_at_ms": "INTEGER",
        "completed_at_ms": "INTEGER",
    }.items():
        _ensure_column(conn, "cp_dispatches", column, definition)
    for column, definition in {
        "dispatch_id": "TEXT",
        "lease_epoch": "INTEGER",
        "cwd": "TEXT",
        "affected_paths_json": "TEXT NOT NULL DEFAULT '[]'",
        "operation_class": "TEXT",
        "risk_classification": "TEXT",
        "normalized_command_preview": "TEXT",
        "reason_requested": "TEXT",
        "request_context_json": "TEXT NOT NULL DEFAULT '{}'",
        "decision_by_instance_id": "TEXT",
        "decision_at_ms": "INTEGER",
        "version": "INTEGER NOT NULL DEFAULT 1",
    }.items():
        _ensure_column(conn, "cp_approvals", column, definition)
    conn.executescript(
        """
        CREATE INDEX IF NOT EXISTS idx_cp_messages_receiver_status ON cp_messages(receiver_profile,status,created_at_ms);
        CREATE INDEX IF NOT EXISTS idx_cp_dispatches_receiver_status ON cp_dispatches(receiver_profile,status,lease_expires_at_ms);
        CREATE INDEX IF NOT EXISTS idx_cp_dispatch_results_dispatch ON cp_dispatch_results(dispatch_id, lease_epoch DESC);
        CREATE INDEX IF NOT EXISTS idx_cp_approvals_status ON cp_approvals(status,expires_at_ms);
        CREATE INDEX IF NOT EXISTS idx_cp_status_events_dispatch ON cp_status_events(dispatch_id,created_at_ms);
        CREATE INDEX IF NOT EXISTS idx_cp_status_events_profile ON cp_status_events(profile_id,created_at_ms);
        CREATE INDEX IF NOT EXISTS idx_cp_blockers_status ON cp_blockers(status,response_deadline_at_ms);
        CREATE INDEX IF NOT EXISTS idx_cp_blockers_dispatch ON cp_blockers(dispatch_id,status);
        CREATE INDEX IF NOT EXISTS idx_cp_blockers_response ON cp_blockers(response_profile,status);
        CREATE INDEX IF NOT EXISTS idx_cp_dispatches_parent_status ON cp_dispatches(parent_dispatch_id,status);
        CREATE INDEX IF NOT EXISTS idx_cp_dispatches_lease ON cp_dispatches(lease_instance_id,lease_epoch);
        CREATE INDEX IF NOT EXISTS idx_cp_outbox_status ON cp_outbox(status,created_at_ms);
        """
    )


def _ensure_column(conn: sqlite3.Connection, table: str, column: str, definition: str) -> None:
    cols = {row[1] for row in conn.execute(f"PRAGMA table_info({table})").fetchall()}
    if column not in cols:
        conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {definition}")


def get_authority_mode(conn: sqlite3.Connection) -> AuthorityMode:
    row = conn.execute("SELECT value FROM cp_meta WHERE key='authority_mode'").fetchone()
    value = (row[0] if row else "shadow").strip()
    if value not in {"legacy", "shadow", "control_db"}:
        return "shadow"
    return value  # type: ignore[return-value]


def _require_admin_actor(conn: sqlite3.Connection, *, actor_profile: str | None, actor_instance_id: str | None, actor_type: str) -> None:
    if actor_type == "bootstrap":
        return
    if actor_type != "admin":
        raise PermissionError("operation requires admin/bootstrap actor")
    if not actor_profile or not actor_instance_id:
        raise PermissionError("admin actor requires actor_profile and live actor_instance_id")
    row = conn.execute(
        "SELECT p.role, i.profile_id, i.status, i.lease_expires_at_ms "
        "FROM cp_profiles p JOIN cp_profile_instances i ON i.profile_id=p.profile_id "
        "WHERE p.profile_id=? AND i.instance_id=?",
        (actor_profile, actor_instance_id),
    ).fetchone()
    if not row or row["role"] != "admin" or row["profile_id"] != actor_profile or row["status"] != "online" or int(row["lease_expires_at_ms"] or 0) <= now_ms():
        raise PermissionError("admin actor instance is not live/authorized")


def set_authority_mode(conn: sqlite3.Connection, mode: AuthorityMode, *, actor_type: str = "worker", actor_profile: str | None = None, actor_instance_id: str | None = None) -> None:
    with transaction(conn):
        _require_admin_actor(conn, actor_profile=actor_profile, actor_instance_id=actor_instance_id, actor_type=actor_type)
        conn.execute(
            "INSERT INTO cp_meta(key,value,updated_at_ms) VALUES('authority_mode',?,?) "
            "ON CONFLICT(key) DO UPDATE SET value=excluded.value, updated_at_ms=excluded.updated_at_ms",
            (mode, now_ms()),
        )


def register_profile(conn: sqlite3.Connection, profile_id: str, *, role: str = "worker", display_name: str | None = None, actor_type: str = "worker", actor_profile: str | None = None, actor_instance_id: str | None = None) -> None:
    ts = now_ms()
    with transaction(conn):
        if role != "worker":
            _require_admin_actor(conn, actor_profile=actor_profile, actor_instance_id=actor_instance_id, actor_type=actor_type)
        conn.execute(
            "INSERT INTO cp_profiles(profile_id,role,display_name,created_at_ms,updated_at_ms) VALUES(?,?,?,?,?) "
            "ON CONFLICT(profile_id) DO UPDATE SET role=CASE WHEN excluded.role != 'worker' THEN excluded.role ELSE cp_profiles.role END, display_name=COALESCE(excluded.display_name, cp_profiles.display_name), updated_at_ms=excluded.updated_at_ms",
            (profile_id, role, display_name, ts, ts),
        )


def register_instance(
    conn: sqlite3.Connection,
    profile_id: str,
    *,
    instance_id: str | None = None,
    pid: int | None = None,
    host: str | None = None,
    lease_ms: int = 120_000,
    metadata: dict[str, Any] | None = None,
    actor_type: str = "worker",
    actor_profile: str | None = None,
    actor_instance_id: str | None = None,
) -> str:
    inst = instance_id or f"{profile_id}:{uuid.uuid4().hex}"
    ts = now_ms()
    with transaction(conn):
        profile = conn.execute("SELECT role FROM cp_profiles WHERE profile_id=?", (profile_id,)).fetchone()
        if profile is None:
            conn.execute(
                "INSERT INTO cp_profiles(profile_id,role,display_name,created_at_ms,updated_at_ms) VALUES(?,?,?,?,?)",
                (profile_id, "worker", None, ts, ts),
            )
            role = "worker"
        else:
            role = str(profile["role"])
        if role != "worker":
            _require_admin_actor(conn, actor_profile=actor_profile, actor_instance_id=actor_instance_id, actor_type=actor_type)
        existing = conn.execute("SELECT profile_id FROM cp_profile_instances WHERE instance_id=?", (inst,)).fetchone()
        if existing and existing["profile_id"] != profile_id:
            raise ControlPlaneError(f"instance_id {inst!r} already belongs to profile {existing['profile_id']!r}")
        conn.execute(
            "INSERT INTO cp_profile_instances(instance_id,profile_id,pid,host,started_at_ms,heartbeat_at_ms,lease_expires_at_ms,status,metadata_json) "
            "VALUES(?,?,?,?,?,?,?,?,?) ON CONFLICT(instance_id) DO UPDATE SET pid=excluded.pid, host=excluded.host, heartbeat_at_ms=excluded.heartbeat_at_ms, lease_expires_at_ms=excluded.lease_expires_at_ms, status='online', metadata_json=excluded.metadata_json",
            (inst, profile_id, pid or os.getpid(), host or os.uname().nodename, ts, ts, ts + lease_ms, "online", dumps_redacted(metadata or {})),
        )
    return inst


def heartbeat_instance(conn: sqlite3.Connection, instance_id: str, *, lease_ms: int = 120_000) -> bool:
    ts = now_ms()
    with transaction(conn):
        cur = conn.execute(
            "UPDATE cp_profile_instances SET heartbeat_at_ms=?, lease_expires_at_ms=?, status='online' WHERE instance_id=?",
            (ts, ts + lease_ms, instance_id),
        )
    return cur.rowcount == 1


def renew_admin_bootstrap_instance_lease(
    conn: sqlite3.Connection,
    *,
    profile_id: str,
    instance_id: str,
    lease_ms: int = 120_000,
) -> dict[str, Any]:
    """Refresh an existing seeded admin bootstrap lease without rewriting history.

    This is deliberately narrower than ``register_instance``. It only revives an
    existing ``<admin-profile>:bootstrap`` row owned by an admin profile. It does
    not create profiles, create new admin instances, or authorize PM/worker
    bootstrap actors.
    """
    if lease_ms <= 0 or lease_ms > 120_000:
        raise ValueError("admin bootstrap lease_ms must be between 1 and 120000")
    if instance_id != f"{profile_id}:bootstrap":
        raise PermissionError("admin lease renewal is limited to the profile bootstrap instance")
    ts = now_ms()
    lease_expires_at_ms = ts + lease_ms
    with transaction(conn):
        row = conn.execute(
            "SELECT i.instance_id, i.profile_id, i.status, i.lease_expires_at_ms, i.metadata_json, p.role "
            "FROM cp_profile_instances i JOIN cp_profiles p ON p.profile_id=i.profile_id "
            "WHERE i.instance_id=? AND i.profile_id=?",
            (instance_id, profile_id),
        ).fetchone()
        if row is None:
            raise PermissionError(f"unknown admin bootstrap instance: {instance_id}")
        if row["role"] != "admin":
            raise PermissionError("admin lease renewal requires an admin profile")
        if row["status"] != "online":
            raise PermissionError("admin lease renewal refuses offline admin bootstrap instances")
        try:
            metadata = json.loads(row["metadata_json"] or "{}")
        except json.JSONDecodeError:
            metadata = {}
        if not bool(metadata.get("seeded_by_bootstrap")):
            raise PermissionError("admin lease renewal requires a seeded bootstrap instance")
        conn.execute(
            "UPDATE cp_profile_instances SET heartbeat_at_ms=?, lease_expires_at_ms=?, status='online' WHERE instance_id=?",
            (ts, lease_expires_at_ms, instance_id),
        )
    return {
        "instance_id": instance_id,
        "profile_id": profile_id,
        "heartbeat_at_ms": ts,
        "previous_lease_expires_at_ms": int(row["lease_expires_at_ms"]) if row["lease_expires_at_ms"] is not None else None,
        "lease_expires_at_ms": lease_expires_at_ms,
        "status": "online",
    }


def mark_instance_offline(conn: sqlite3.Connection, instance_id: str, *, now_ms_value: int | None = None) -> bool:
    """Mark a finite-lived profile instance offline without deleting audit state."""
    ts = _ts(now_ms_value)
    with transaction(conn):
        cur = conn.execute(
            "UPDATE cp_profile_instances SET heartbeat_at_ms=?, lease_expires_at_ms=?, status='offline' WHERE instance_id=? AND status='online'",
            (ts, ts, instance_id),
        )
    return cur.rowcount == 1


def mark_expired_worker_instances_offline(conn: sqlite3.Connection, *, limit: int = 100, now_ms_value: int | None = None) -> list[str]:
    """Mark expired worker-role instances offline without deleting rows.

    This is intentionally conservative: it only touches rows whose owning
    profile has role ``worker``, whose instance status is still ``online``, and
    whose lease has already expired. Admin/PM/bootstrap liveness semantics are
    policy decisions and are not inferred here.
    """
    ts = _ts(now_ms_value)
    with transaction(conn):
        rows = conn.execute(
            """
            SELECT i.instance_id
            FROM cp_profile_instances i
            JOIN cp_profiles p ON p.profile_id=i.profile_id
            WHERE p.role='worker'
              AND i.status='online'
              AND i.lease_expires_at_ms IS NOT NULL
              AND i.lease_expires_at_ms < ?
            ORDER BY i.lease_expires_at_ms, i.instance_id
            LIMIT ?
            """,
            (ts, limit),
        ).fetchall()
        ids = [str(row["instance_id"]) for row in rows]
        if ids:
            placeholders = ",".join("?" for _ in ids)
            conn.execute(
                f"UPDATE cp_profile_instances SET heartbeat_at_ms=?, lease_expires_at_ms=?, status='offline' WHERE instance_id IN ({placeholders})",
                (ts, ts, *ids),
            )
    return ids


def get_instance(conn: sqlite3.Connection, instance_id: str) -> sqlite3.Row | None:
    return conn.execute(
        "SELECT i.*, p.role FROM cp_profile_instances i JOIN cp_profiles p ON p.profile_id=i.profile_id WHERE i.instance_id=?",
        (instance_id,),
    ).fetchone()


def _require_live_instance(conn: sqlite3.Connection, instance_id: str, *, expected_profile: str | None = None, now: int | None = None) -> sqlite3.Row:
    ts = _ts(now)
    row = get_instance(conn, instance_id)
    if not row:
        raise PermissionError(f"unknown control-plane instance: {instance_id}")
    if expected_profile is not None and row["profile_id"] != expected_profile:
        raise PermissionError(f"instance {instance_id!r} does not belong to profile {expected_profile!r}")
    if row["status"] != "online" or int(row["lease_expires_at_ms"] or 0) <= ts:
        raise PermissionError(f"control-plane instance {instance_id!r} is not online/live")
    return row


def _require_dispatch_lease(conn: sqlite3.Connection, *, dispatch_id: str, instance_id: str, lease_epoch: int, now: int | None = None) -> sqlite3.Row:
    ts = _ts(now)
    row = conn.execute("SELECT * FROM cp_dispatches WHERE dispatch_id=?", (dispatch_id,)).fetchone()
    if not row:
        raise ControlPlaneError(f"unknown dispatch: {dispatch_id}")
    _require_live_instance(conn, instance_id, expected_profile=row["receiver_profile"], now=ts)
    if row["lease_instance_id"] != instance_id or int(row["lease_epoch"]) != int(lease_epoch) or int(row["lease_expires_at_ms"] or 0) <= ts:
        raise PermissionError("dispatch lease is not held by this instance/epoch")
    return row


def add_route_policy(conn: sqlite3.Connection, *, effect: Literal["allow", "deny"], sender_profile: str = "*", receiver_profile: str = "*", kind: str = "*", capability: str = "*", priority: int = 0, created_by: str = "unknown", created_by_type: str = "worker", created_by_instance_id: str | None = None) -> str:
    policy_id = f"pol_{uuid.uuid4().hex}"
    with transaction(conn):
        _require_admin_actor(conn, actor_profile=None if created_by == "unknown" else created_by, actor_instance_id=created_by_instance_id, actor_type=created_by_type)
        conn.execute(
            "INSERT INTO cp_route_policies(policy_id,priority,effect,sender_profile,receiver_profile,kind,capability,created_by,created_by_type,created_at_ms) VALUES(?,?,?,?,?,?,?,?,?,?)",
            (policy_id, priority, effect, sender_profile, receiver_profile, kind, capability, created_by, created_by_type, now_ms()),
        )
    return policy_id


def remove_route_policy(conn: sqlite3.Connection, policy_id: str, *, actor_profile: str | None = None, actor_instance_id: str | None = None, actor_type: str = "worker") -> bool:
    with transaction(conn):
        _require_admin_actor(conn, actor_profile=actor_profile, actor_instance_id=actor_instance_id, actor_type=actor_type)
        cur = conn.execute("DELETE FROM cp_route_policies WHERE policy_id=?", (policy_id,))
    return cur.rowcount == 1


def _matches(pattern: str, value: str) -> bool:
    return pattern == "*" or pattern == value


def route_allowed(conn: sqlite3.Connection, *, sender_profile: str, receiver_profile: str, kind: str, capability: str) -> bool:
    rows = conn.execute(
        "SELECT * FROM cp_route_policies ORDER BY priority DESC, effect ASC"
    ).fetchall()
    matches = [r for r in rows if _matches(r["sender_profile"], sender_profile) and _matches(r["receiver_profile"], receiver_profile) and _matches(r["kind"], kind) and _matches(r["capability"], capability)]
    if not matches:
        return False
    top_priority = max(r["priority"] for r in matches)
    top = [r for r in matches if r["priority"] == top_priority]
    if any(r["effect"] == "deny" for r in top):
        return False
    return any(r["effect"] == "allow" for r in top)


def _require_route(conn: sqlite3.Connection, *, sender_profile: str, receiver_profile: str, kind: str, capability: str) -> None:
    if not route_allowed(conn, sender_profile=sender_profile, receiver_profile=receiver_profile, kind=kind, capability=capability):
        raise RouteDenied(f"route denied: {sender_profile}->{receiver_profile} kind={kind} capability={capability}")


def _event_id(prefix: str = "evt") -> str:
    return f"{prefix}_{uuid.uuid4().hex}"


def _enqueue_outbox(conn: sqlite3.Connection, *, subject_type: str, subject_id: str, subject_version: int, event_id: str | None, payload: dict[str, Any], target_platform: str | None = None, target_ref: str | None = None) -> str:
    outbox_id = f"out_{uuid.uuid4().hex}"
    ts = now_ms()
    conn.execute(
        "INSERT INTO cp_outbox(outbox_id,subject_type,subject_id,subject_version,event_id,target_platform,target_ref_hmac,payload_json,status,created_at_ms,updated_at_ms) VALUES(?,?,?,?,?,?,?,?,?,?,?)",
        (outbox_id, subject_type, subject_id, subject_version, event_id, target_platform, hmac_id(target_ref, root=_conn_root(conn)) if target_ref else None, dumps_redacted(payload), "pending", ts, ts),
    )
    return outbox_id


def create_message(conn: sqlite3.Connection, *, sender_profile: str, receiver_profile: str, kind: str, body: str, capability: str = "message", metadata: dict[str, Any] | None = None, idempotency_key: str | None = None) -> str:
    mid = f"msg_{uuid.uuid4().hex}"
    evt = _event_id()
    ts = now_ms()
    redacted_body = redact_text(body)
    redacted_metadata = dumps_redacted(metadata or {})
    with transaction(conn):
        _require_route(conn, sender_profile=sender_profile, receiver_profile=receiver_profile, kind=kind, capability=capability)
        if idempotency_key:
            row = conn.execute("SELECT * FROM cp_messages WHERE idempotency_key=?", (idempotency_key,)).fetchone()
            if row:
                if (row["sender_profile"], row["receiver_profile"], row["kind"], row["capability"], row["body"], row["metadata_json"]) != (sender_profile, receiver_profile, kind, capability, redacted_body, redacted_metadata):
                    raise ControlPlaneError("idempotency key reused with different message request")
                return row["message_id"]
        conn.execute(
            "INSERT INTO cp_messages(message_id,kind,sender_profile,receiver_profile,capability,body,metadata_json,status,created_at_ms,updated_at_ms,idempotency_key) VALUES(?,?,?,?,?,?,?,?,?,?,?)",
            (mid, kind, sender_profile, receiver_profile, capability, redacted_body, redacted_metadata, "pending", ts, ts, idempotency_key),
        )
        conn.execute(
            "INSERT INTO cp_message_events(event_id,message_id,event_type,event_json,created_at_ms) VALUES(?,?,?,?,?)",
            (evt, mid, "created", dumps_redacted({"kind": kind, "status": "pending"}), ts),
        )
        _enqueue_outbox(conn, subject_type="message", subject_id=mid, subject_version=1, event_id=evt, payload={"event": "message.created", "message_id": mid, "kind": kind})
    return mid


def create_message_from_instance(conn: sqlite3.Connection, *, sender_instance_id: str, receiver_profile: str, kind: str, body: str, capability: str = "message", metadata: dict[str, Any] | None = None, idempotency_key: str | None = None, now_ms: int | None = None) -> str:
    mid = f"msg_{uuid.uuid4().hex}"
    evt = _event_id()
    ts = now_ms if now_ms is not None else globals()["now_ms"]()
    redacted_body = redact_text(body)
    redacted_metadata = dumps_redacted(metadata or {})
    with transaction(conn):
        sender = _require_live_instance(conn, sender_instance_id, now=now_ms)
        sender_profile = sender["profile_id"]
        _require_route(conn, sender_profile=sender_profile, receiver_profile=receiver_profile, kind=kind, capability=capability)
        if idempotency_key:
            row = conn.execute("SELECT * FROM cp_messages WHERE idempotency_key=?", (idempotency_key,)).fetchone()
            if row:
                if (row["sender_profile"], row["receiver_profile"], row["kind"], row["capability"], row["body"], row["metadata_json"]) != (sender_profile, receiver_profile, kind, capability, redacted_body, redacted_metadata):
                    raise ControlPlaneError("idempotency key reused with different message request")
                return row["message_id"]
        conn.execute(
            "INSERT INTO cp_messages(message_id,kind,sender_profile,receiver_profile,capability,body,metadata_json,status,created_at_ms,updated_at_ms,idempotency_key) VALUES(?,?,?,?,?,?,?,?,?,?,?)",
            (mid, kind, sender_profile, receiver_profile, capability, redacted_body, redacted_metadata, "pending", ts, ts, idempotency_key),
        )
        conn.execute(
            "INSERT INTO cp_message_events(event_id,message_id,event_type,event_json,created_at_ms) VALUES(?,?,?,?,?)",
            (evt, mid, "created", dumps_redacted({"kind": kind, "status": "pending"}), ts),
        )
        _enqueue_outbox(conn, subject_type="message", subject_id=mid, subject_version=1, event_id=evt, payload={"event": "message.created", "message_id": mid, "kind": kind})
    return mid


def _is_live_default_root(conn: sqlite3.Connection) -> bool:
    root = _conn_root(conn)
    if root is None:
        return False
    try:
        return root.resolve() == get_default_hermes_root().resolve()
    except OSError:
        return False


def _require_message_transition_actor(
    conn: sqlite3.Connection,
    *,
    message: sqlite3.Row,
    actor_type: str,
    actor_profile: str | None,
    actor_instance_id: str | None,
    now: int,
) -> None:
    if actor_type == "bootstrap":
        if _is_live_default_root(conn):
            raise PermissionError("bootstrap message transitions are refused against the live control DB")
        return
    if actor_type == "admin":
        _require_admin_actor(conn, actor_profile=actor_profile, actor_instance_id=actor_instance_id, actor_type="admin")
        return
    if actor_type != "receiver":
        raise PermissionError("message transition actor_type must be receiver, admin, or bootstrap")
    if not actor_instance_id:
        raise PermissionError("receiver message transitions require actor_instance_id")
    _require_live_instance(conn, actor_instance_id, expected_profile=message["receiver_profile"], now=now)


def transition_message_status(
    conn: sqlite3.Connection,
    message_id: str,
    *,
    status: str,
    actor_instance_id: str | None = None,
    actor_profile: str | None = None,
    actor_type: str = "receiver",
    reason: str | None = None,
    metadata: dict[str, Any] | None = None,
    now_ms: int | None = None,
) -> dict[str, Any]:
    """Move a message to a terminal status with append-only audit.

    `superseded` is also an audited archival override for stale terminal
    messages; other terminal-to-different-terminal rewrites are rejected.
    """
    if status not in TERMINAL_MESSAGE_STATUSES:
        raise ControlPlaneError(f"invalid terminal message status: {status}")
    ts = _ts(now_ms)
    with transaction(conn):
        row = conn.execute("SELECT * FROM cp_messages WHERE message_id=?", (message_id,)).fetchone()
        if not row:
            raise ControlPlaneError(f"unknown message: {message_id}")
        _require_message_transition_actor(
            conn,
            message=row,
            actor_type=actor_type,
            actor_profile=actor_profile,
            actor_instance_id=actor_instance_id,
            now=ts,
        )
        old_status = str(row["status"])
        if old_status in TERMINAL_MESSAGE_STATUSES and old_status != status and status != "superseded":
            raise ControlPlaneError(f"message {message_id} is already terminal: {old_status}")
        if old_status not in MESSAGE_STATUSES:
            raise ControlPlaneError(f"message {message_id} has unknown status: {old_status}")
        changed = old_status != status
        if changed:
            conn.execute("UPDATE cp_messages SET status=?, updated_at_ms=? WHERE message_id=?", (status, ts, message_id))
        event_id = _event_id()
        event = {
            "old_status": old_status,
            "new_status": status,
            "changed": changed,
            "actor_type": actor_type,
            "actor_profile": actor_profile,
            "actor_instance_id": actor_instance_id,
            "reason": redact_text(reason) if reason else None,
            "metadata": metadata or {},
        }
        conn.execute(
            "INSERT INTO cp_message_events(event_id,message_id,event_type,event_json,created_at_ms) VALUES(?,?,?,?,?)",
            (event_id, message_id, "status_transition", dumps_redacted(event), ts),
        )
        _enqueue_outbox(
            conn,
            subject_type="message",
            subject_id=message_id,
            subject_version=2,
            event_id=event_id,
            payload={"event": "message.status_transition", "message_id": message_id, "status": status, "changed": changed},
        )
        return {"message_id": message_id, "old_status": old_status, "status": status, "changed": changed, "event_id": event_id}


def create_dispatch(conn: sqlite3.Connection, *, sender_profile: str, receiver_profile: str, payload: dict[str, Any], capability: str = "dispatch", message_id: str | None = None, idempotency_key: str | None = None, max_attempts: int = 3, parent_dispatch_id: str | None = None, dispatch_schema: str | None = None, max_wall_time_ms: int | None = None) -> str:
    did = f"disp_{uuid.uuid4().hex}"
    evt = _event_id()
    ts = now_ms()
    redacted_payload = dumps_redacted(payload)
    with transaction(conn):
        _require_route(conn, sender_profile=sender_profile, receiver_profile=receiver_profile, kind="dispatch", capability=capability)
        if idempotency_key:
            row = conn.execute("SELECT * FROM cp_dispatches WHERE idempotency_key=?", (idempotency_key,)).fetchone()
            if row:
                if (row["sender_profile"], row["receiver_profile"], row["capability"], row["message_id"], row["payload_json"], row["max_attempts"]) != (sender_profile, receiver_profile, capability, message_id, redacted_payload, max_attempts):
                    raise ControlPlaneError("idempotency key reused with different dispatch request")
                return row["dispatch_id"]
        conn.execute(
            "INSERT INTO cp_dispatches(dispatch_id,message_id,sender_profile,receiver_profile,capability,status,payload_json,parent_dispatch_id,dispatch_schema,max_wall_time_ms,attempts,max_attempts,created_at_ms,updated_at_ms,idempotency_key) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (did, message_id, sender_profile, receiver_profile, capability, "pending", redacted_payload, parent_dispatch_id, dispatch_schema, max_wall_time_ms, 0, max_attempts, ts, ts, idempotency_key),
        )
        conn.execute(
            "INSERT INTO cp_dispatch_events(event_id,dispatch_id,event_type,event_json,created_at_ms) VALUES(?,?,?,?,?)",
            (evt, did, "created", dumps_redacted({"status": "pending"}), ts),
        )
        _enqueue_outbox(conn, subject_type="dispatch", subject_id=did, subject_version=1, event_id=evt, payload={"event": "dispatch.created", "dispatch_id": did})
    return did


def create_dispatch_from_instance(conn: sqlite3.Connection, *, sender_instance_id: str, receiver_profile: str, payload: dict[str, Any], capability: str = "dispatch", message_id: str | None = None, idempotency_key: str | None = None, max_attempts: int = 3, parent_dispatch_id: str | None = None, dispatch_schema: str | None = None, max_wall_time_ms: int | None = None, now_ms: int | None = None) -> str:
    did = f"disp_{uuid.uuid4().hex}"
    evt = _event_id()
    ts = now_ms if now_ms is not None else globals()["now_ms"]()
    redacted_payload = dumps_redacted(payload)
    with transaction(conn):
        sender = _require_live_instance(conn, sender_instance_id, now=now_ms)
        sender_profile = sender["profile_id"]
        _require_route(conn, sender_profile=sender_profile, receiver_profile=receiver_profile, kind="dispatch", capability=capability)
        if idempotency_key:
            row = conn.execute("SELECT * FROM cp_dispatches WHERE idempotency_key=?", (idempotency_key,)).fetchone()
            if row:
                if (row["sender_profile"], row["receiver_profile"], row["capability"], row["message_id"], row["payload_json"], row["max_attempts"]) != (sender_profile, receiver_profile, capability, message_id, redacted_payload, max_attempts):
                    raise ControlPlaneError("idempotency key reused with different dispatch request")
                return row["dispatch_id"]
        conn.execute(
            "INSERT INTO cp_dispatches(dispatch_id,message_id,sender_profile,receiver_profile,capability,status,payload_json,parent_dispatch_id,dispatch_schema,max_wall_time_ms,attempts,max_attempts,created_at_ms,updated_at_ms,idempotency_key) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (did, message_id, sender_profile, receiver_profile, capability, "pending", redacted_payload, parent_dispatch_id, dispatch_schema, max_wall_time_ms, 0, max_attempts, ts, ts, idempotency_key),
        )
        conn.execute(
            "INSERT INTO cp_dispatch_events(event_id,dispatch_id,event_type,event_json,created_at_ms) VALUES(?,?,?,?,?)",
            (evt, did, "created", dumps_redacted({"status": "pending"}), ts),
        )
        _enqueue_outbox(conn, subject_type="dispatch", subject_id=did, subject_version=1, event_id=evt, payload={"event": "dispatch.created", "dispatch_id": did})
    return did


def claim_dispatch(conn: sqlite3.Connection, dispatch_id: str, *, instance_id: str, lease_ms: int = 300_000, now_ms: int | None = None) -> tuple[bool, int | None]:
    return claim_dispatch_by_id(conn, dispatch_id=dispatch_id, instance_id=instance_id, lease_ms=lease_ms, now_ms=now_ms)


def claim_dispatch_by_id(conn: sqlite3.Connection, *, dispatch_id: str, instance_id: str, lease_ms: int = 300_000, now_ms: int | None = None) -> tuple[bool, int | None]:
    ts = _ts(now_ms)
    with transaction(conn):
        inst = _require_live_instance(conn, instance_id, now=ts)
        cur = conn.execute(
            "UPDATE cp_dispatches SET status='claimed', lease_instance_id=?, lease_epoch=lease_epoch+1, lease_expires_at_ms=?, attempts=attempts+1, updated_at_ms=? "
            "WHERE dispatch_id=? AND receiver_profile=? AND status IN ('pending','retry_ready') AND (lease_expires_at_ms IS NULL OR lease_expires_at_ms < ?)",
            (instance_id, ts + lease_ms, ts, dispatch_id, inst["profile_id"], ts),
        )
        if cur.rowcount != 1:
            return False, None
        epoch = conn.execute("SELECT lease_epoch FROM cp_dispatches WHERE dispatch_id=?", (dispatch_id,)).fetchone()[0]
        evt = _event_id()
        conn.execute(
            "INSERT INTO cp_dispatch_events(event_id,dispatch_id,event_type,event_json,created_at_ms) VALUES(?,?,?,?,?)",
            (evt, dispatch_id, "claimed", dumps_redacted({"instance_id": instance_id, "lease_epoch": epoch}), ts),
        )
        _enqueue_outbox(conn, subject_type="dispatch", subject_id=dispatch_id, subject_version=epoch, event_id=evt, payload={"event": "dispatch.claimed", "dispatch_id": dispatch_id, "lease_epoch": epoch})
        return True, int(epoch)


def claim_next_for_profile(conn: sqlite3.Connection, *, receiver_profile: str, instance_id: str, lease_ms: int = 300_000, now_ms: int | None = None) -> tuple[str, int] | None:
    ts = _ts(now_ms)
    with transaction(conn):
        _require_live_instance(conn, instance_id, expected_profile=receiver_profile, now=ts)
        row = conn.execute(
            "SELECT dispatch_id FROM cp_dispatches "
            "WHERE receiver_profile=? AND status IN ('pending','retry_ready') AND (lease_expires_at_ms IS NULL OR lease_expires_at_ms < ?) "
            "ORDER BY created_at_ms ASC LIMIT 1",
            (receiver_profile, ts),
        ).fetchone()
    if not row:
        return None
    ok, epoch = claim_dispatch_by_id(conn, dispatch_id=row["dispatch_id"], instance_id=instance_id, lease_ms=lease_ms, now_ms=ts)
    if not ok or epoch is None:
        return None
    return row["dispatch_id"], epoch


def extend_dispatch_lease(conn: sqlite3.Connection, dispatch_id: str, *, instance_id: str, lease_epoch: int, lease_ms: int = 300_000, now_ms: int | None = None) -> bool:
    """Extend an active dispatch lease for its current fenced owner."""
    ts = _ts(now_ms)
    with transaction(conn):
        cur = conn.execute(
            "UPDATE cp_dispatches SET lease_expires_at_ms=?, updated_at_ms=? "
            "WHERE dispatch_id=? AND lease_instance_id=? AND lease_epoch=? AND lease_expires_at_ms IS NOT NULL AND lease_expires_at_ms > ? AND status IN ('claimed','running')",
            (ts + lease_ms, ts, dispatch_id, instance_id, lease_epoch, ts),
        )
    return cur.rowcount == 1


def advance_dispatch(conn: sqlite3.Connection, dispatch_id: str, *, instance_id: str, lease_epoch: int, status: str, last_error: str | None = None, now_ms: int | None = None) -> bool:
    if status not in _DISPATCH_ADVANCE_STATUSES:
        raise ControlPlaneError(f"invalid dispatch status: {status}")
    ts = _ts(now_ms)
    with transaction(conn):
        cur = conn.execute(
            "UPDATE cp_dispatches SET status=?, last_error=?, updated_at_ms=? "
            "WHERE dispatch_id=? AND lease_instance_id=? AND lease_epoch=? AND lease_expires_at_ms IS NOT NULL AND lease_expires_at_ms > ? AND status IN ('claimed','running')",
            (status, redact_text(last_error) if last_error else None, ts, dispatch_id, instance_id, lease_epoch, ts),
        )
        if cur.rowcount != 1:
            return False
        evt = _event_id()
        conn.execute(
            "INSERT INTO cp_dispatch_events(event_id,dispatch_id,event_type,event_json,created_at_ms) VALUES(?,?,?,?,?)",
            (evt, dispatch_id, status, dumps_redacted({"status": status}), ts),
        )
        _enqueue_outbox(conn, subject_type="dispatch", subject_id=dispatch_id, subject_version=lease_epoch, event_id=evt, payload={"event": f"dispatch.{status}", "dispatch_id": dispatch_id})
    return True


def supersede_dispatch(conn: sqlite3.Connection, dispatch_id: str, *, actor_instance_id: str, actor_profile: str | None = None, reason: str | None = None, metadata: dict[str, Any] | None = None, now_ms: int | None = None) -> bool:
    """Mark a non-running dispatch as superseded with an audit event.

    This is for preserving stale incident rows without leaving them in active
    queue scans. It intentionally refuses live runnable states; claim/advance or
    the watchdog should handle pending/running work.
    """
    ts = _ts(now_ms)
    event_metadata = dict(metadata or {})
    if reason:
        event_metadata["reason"] = redact_text(reason)
    if actor_profile:
        event_metadata["actor_profile"] = actor_profile
    event_metadata["actor_instance_id"] = actor_instance_id
    with transaction(conn):
        row = conn.execute("SELECT status, lease_epoch FROM cp_dispatches WHERE dispatch_id=?", (dispatch_id,)).fetchone()
        if not row:
            return False
        if row["status"] in {"pending", "claimed", "running", "retry_ready"}:
            raise ControlPlaneError(f"dispatch {dispatch_id} is still runnable status={row['status']}")
        if row["status"] == "superseded":
            return True
        cur = conn.execute(
            "UPDATE cp_dispatches SET status='superseded', last_error=?, completed_at_ms=COALESCE(completed_at_ms, ?), updated_at_ms=? WHERE dispatch_id=?",
            (redact_text(reason) if reason else row["status"], ts, ts, dispatch_id),
        )
        if cur.rowcount != 1:
            return False
        evt = _event_id()
        conn.execute(
            "INSERT INTO cp_dispatch_events(event_id,dispatch_id,event_type,event_json,created_at_ms) VALUES(?,?,?,?,?)",
            (evt, dispatch_id, "superseded", dumps_redacted(event_metadata), ts),
        )
        _enqueue_outbox(conn, subject_type="dispatch", subject_id=dispatch_id, subject_version=int(row["lease_epoch"] or 0), event_id=evt, payload={"event": "dispatch.superseded", "dispatch_id": dispatch_id})
    return True


def record_artifact(conn: sqlite3.Connection, *, dispatch_id: str, instance_id: str, lease_epoch: int, path: str, summary: str | None = None, metadata: dict[str, Any] | None = None, now_ms: int | None = None) -> str:
    ts = _ts(now_ms)
    artifact_id = f"art_{uuid.uuid4().hex}"
    with transaction(conn):
        _require_dispatch_lease(conn, dispatch_id=dispatch_id, instance_id=instance_id, lease_epoch=lease_epoch, now=ts)
        conn.execute(
            "INSERT INTO cp_artifacts(artifact_id,dispatch_id,instance_id,lease_epoch,path,summary,metadata_json,created_at_ms) VALUES(?,?,?,?,?,?,?,?)",
            (artifact_id, dispatch_id, instance_id, lease_epoch, redact_text(path), redact_text(summary) if summary else None, dumps_redacted(metadata or {}), ts),
        )
    return artifact_id


def record_result(conn: sqlite3.Connection, *, dispatch_id: str, instance_id: str, lease_epoch: int, result: dict[str, Any], now_ms: int | None = None) -> None:
    ts = _ts(now_ms)
    result_json = dumps_redacted(result)
    with transaction(conn):
        _require_dispatch_lease(conn, dispatch_id=dispatch_id, instance_id=instance_id, lease_epoch=lease_epoch, now=ts)
        row = conn.execute(
            "SELECT instance_id,result_json FROM cp_dispatch_results WHERE dispatch_id=? AND lease_epoch=?",
            (dispatch_id, lease_epoch),
        ).fetchone()
        if row:
            if row["instance_id"] != instance_id or row["result_json"] != result_json:
                raise ControlPlaneError("conflicting dispatch result for lease epoch")
            return
        conn.execute(
            "INSERT INTO cp_dispatch_results(dispatch_id,lease_epoch,instance_id,result_json,created_at_ms) VALUES(?,?,?,?,?)",
            (dispatch_id, lease_epoch, instance_id, result_json, ts),
        )


def get_latest_dispatch_result(conn: sqlite3.Connection, dispatch_id: str) -> dict[str, Any] | None:
    row = conn.execute(
        "SELECT * FROM cp_dispatch_results WHERE dispatch_id=? ORDER BY lease_epoch DESC LIMIT 1",
        (dispatch_id,),
    ).fetchone()
    if not row:
        return None
    data = dict(row)
    data["result"] = json.loads(row["result_json"])
    return data


def list_dispatch_results(conn: sqlite3.Connection, dispatch_id: str) -> list[dict[str, Any]]:
    rows = conn.execute(
        "SELECT * FROM cp_dispatch_results WHERE dispatch_id=? ORDER BY lease_epoch ASC",
        (dispatch_id,),
    ).fetchall()
    out: list[dict[str, Any]] = []
    for row in rows:
        data = dict(row)
        data["result"] = json.loads(row["result_json"])
        out.append(data)
    return out


def list_artifacts(conn: sqlite3.Connection, dispatch_id: str) -> list[dict[str, Any]]:
    return [
        dict(r)
        for r in conn.execute(
            "SELECT artifact_id,dispatch_id,instance_id,lease_epoch,path,summary,metadata_json,created_at_ms FROM cp_artifacts WHERE dispatch_id=? ORDER BY created_at_ms",
            (dispatch_id,),
        ).fetchall()
    ]


def reap_expired_dispatches(conn: sqlite3.Connection, *, limit: int = 100, now_ms: int | None = None) -> int:
    ts = _ts(now_ms)
    with transaction(conn):
        rows = conn.execute(
            "SELECT dispatch_id, attempts, max_attempts FROM cp_dispatches WHERE status IN ('claimed','running') AND lease_expires_at_ms IS NOT NULL AND lease_expires_at_ms < ? LIMIT ?",
            (ts, limit),
        ).fetchall()
        for row in rows:
            status = "retry_ready" if row["attempts"] < row["max_attempts"] else "dead_letter"
            conn.execute(
                "UPDATE cp_dispatches SET status=?, lease_instance_id=NULL, lease_expires_at_ms=NULL, updated_at_ms=?, last_error=? WHERE dispatch_id=?",
                (status, ts, "lease expired", row["dispatch_id"]),
            )
            evt = _event_id()
            conn.execute(
                "INSERT INTO cp_dispatch_events(event_id,dispatch_id,event_type,event_json,created_at_ms) VALUES(?,?,?,?,?)",
                (evt, row["dispatch_id"], status, dumps_redacted({"reason": "lease_expired"}), ts),
            )
            _enqueue_outbox(conn, subject_type="dispatch", subject_id=row["dispatch_id"], subject_version=row["attempts"] + 1, event_id=evt, payload={"event": f"dispatch.{status}", "dispatch_id": row["dispatch_id"]})
    return len(rows)


def create_approval(conn: sqlite3.Connection, *, requester_profile: str, requester_instance_id: str, approver_profile: str, command_preview: str, tool_args_preview: str | None = None, ttl_ms: int = 900_000, approval_id: str | None = None, dispatch_id: str | None = None, lease_epoch: int | None = None, cwd: str | None = None, affected_paths: list[str] | None = None, operation_class: str | None = None, risk_classification: str | None = None, normalized_command_preview: str | None = None, reason_requested: str | None = None, request_context: dict[str, Any] | None = None) -> str:
    aid = approval_id or f"appr_{uuid.uuid4().hex}"
    ts = now_ms()
    with transaction(conn):
        conn.execute(
            "INSERT INTO cp_approvals(approval_id,requester_profile,requester_instance_id,approver_profile,dispatch_id,lease_epoch,cwd,affected_paths_json,operation_class,risk_classification,normalized_command_preview,reason_requested,request_context_json,version,status,command_preview,tool_args_preview,expires_at_ms,created_at_ms,updated_at_ms) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (aid, requester_profile, requester_instance_id, approver_profile, dispatch_id, lease_epoch, redact_text(cwd) if cwd else None, dumps_redacted(affected_paths or []), operation_class, risk_classification, redact_text(normalized_command_preview or command_preview), redact_text(reason_requested) if reason_requested else None, dumps_redacted(request_context or {}), 1, "pending", redact_text(command_preview), redact_text(tool_args_preview) if tool_args_preview else None, ts + ttl_ms, ts, ts),
        )
        _enqueue_outbox(conn, subject_type="approval", subject_id=aid, subject_version=1, event_id=None, payload={"event": "approval.pending", "approval_id": aid, "status": "pending"})
    return aid


def decide_approval(conn: sqlite3.Connection, approval_id: str, *, approver_profile: str, decision: Literal["approved", "denied"], reason: str | None = None, approver_instance_id: str | None = None, actor_type: str = "admin") -> bool:
    if decision not in _APPROVAL_DECISIONS:
        raise ControlPlaneError(f"invalid approval decision: {decision}")
    ts = now_ms()
    with transaction(conn):
        _require_admin_actor(conn, actor_profile=approver_profile, actor_instance_id=approver_instance_id, actor_type=actor_type)
        cur = conn.execute(
            "UPDATE cp_approvals SET status=?, decision=?, decision_reason=?, decision_by_instance_id=?, decision_at_ms=?, updated_at_ms=? WHERE approval_id=? AND approver_profile=? AND status='pending' AND expires_at_ms > ?",
            (decision, decision, redact_text(reason) if reason else None, approver_instance_id, ts, ts, approval_id, approver_profile, ts),
        )
        if cur.rowcount != 1:
            return False
        _enqueue_outbox(conn, subject_type="approval", subject_id=approval_id, subject_version=2, event_id=None, payload={"event": f"approval.{decision}", "approval_id": approval_id, "status": decision})
    return True


def consume_approval(conn: sqlite3.Connection, approval_id: str, *, requester_instance_id: str, requester_profile: str | None = None, dispatch_id: str | None = None, lease_epoch: int | None = None) -> bool:
    ts = now_ms()
    with transaction(conn):
        row = conn.execute("SELECT requester_profile, requester_instance_id, dispatch_id, lease_epoch FROM cp_approvals WHERE approval_id=?", (approval_id,)).fetchone()
        if not row:
            return False
        if requester_profile is not None and row["requester_profile"] != requester_profile:
            return False
        if row["requester_instance_id"] != requester_instance_id:
            return False
        if row["dispatch_id"] is not None and row["dispatch_id"] != dispatch_id:
            return False
        if row["lease_epoch"] is not None and int(row["lease_epoch"]) != int(lease_epoch or -1):
            return False
        cur = conn.execute(
            "UPDATE cp_approvals SET status='consumed', consumed_by_instance_id=?, consumed_at_ms=?, updated_at_ms=? "
            "WHERE approval_id=? AND status='approved' AND consumed_at_ms IS NULL AND expires_at_ms > ? AND requester_instance_id=?",
            (requester_instance_id, ts, ts, approval_id, ts, requester_instance_id),
        )
        if cur.rowcount != 1:
            return False
        _enqueue_outbox(conn, subject_type="approval", subject_id=approval_id, subject_version=3, event_id=None, payload={"event": "approval.consumed", "approval_id": approval_id, "status": "consumed"})
    return True


def expire_approvals(conn: sqlite3.Connection) -> int:
    ts = now_ms()
    with transaction(conn):
        rows = conn.execute("SELECT approval_id FROM cp_approvals WHERE status='pending' AND expires_at_ms <= ?", (ts,)).fetchall()
        for row in rows:
            conn.execute("UPDATE cp_approvals SET status='expired', updated_at_ms=? WHERE approval_id=?", (ts, row["approval_id"]))
            _enqueue_outbox(conn, subject_type="approval", subject_id=row["approval_id"], subject_version=2, event_id=None, payload={"event": "approval.expired", "approval_id": row["approval_id"], "status": "expired"})
    return len(rows)


def emit_status(conn: sqlite3.Connection, *, instance_id: str, status: str, summary: str, dispatch_id: str | None = None, parent_dispatch_id: str | None = None, details: dict[str, Any] | None = None, now_ms: int | None = None) -> str:
    allowed = {"starting", "claimed", "running", "waiting_approval", "blocked", "verifying", "completed", "failed", "cancelled", "stalled"}
    if status not in allowed:
        raise ControlPlaneError(f"invalid status: {status}")
    ts = _ts(now_ms)
    event_id = _event_id()
    with transaction(conn):
        inst = _require_live_instance(conn, instance_id, now=ts)
        if dispatch_id is not None:
            row = conn.execute("SELECT receiver_profile, lease_instance_id, lease_epoch, lease_expires_at_ms, parent_dispatch_id FROM cp_dispatches WHERE dispatch_id=?", (dispatch_id,)).fetchone()
            if not row:
                raise ControlPlaneError(f"unknown dispatch: {dispatch_id}")
            if inst["role"] not in {"admin", "pm"}:
                if row["receiver_profile"] != inst["profile_id"] or row["lease_instance_id"] != instance_id or int(row["lease_expires_at_ms"] or 0) <= ts:
                    raise PermissionError("status emitter does not hold dispatch lease")
            parent_dispatch_id = parent_dispatch_id or row["parent_dispatch_id"]
        conn.execute(
            "INSERT INTO cp_status_events(event_id,profile_id,instance_id,dispatch_id,parent_dispatch_id,status,summary,details_json,created_at_ms,version) VALUES(?,?,?,?,?,?,?,?,?,?)",
            (event_id, inst["profile_id"], instance_id, dispatch_id, parent_dispatch_id, status, redact_text(summary), dumps_redacted(details or {}), ts, 1),
        )
        _enqueue_outbox(conn, subject_type="status", subject_id=event_id, subject_version=1, event_id=event_id, payload={"event": "status.emitted", "event_id": event_id, "status": status})
    return event_id


def open_blocker(conn: sqlite3.Connection, *, dispatch_id: str, instance_id: str, severity: str, kind: str, summary: str, details: dict[str, Any] | None = None, requires_response: bool = True, response_profile: str | None = None, response_deadline_at_ms: int | None = None, now_ms: int | None = None) -> str:
    blocker_id = f"blk_{uuid.uuid4().hex}"
    ts = _ts(now_ms)
    with transaction(conn):
        disp = _require_dispatch_lease(conn, dispatch_id=dispatch_id, instance_id=instance_id, lease_epoch=int(conn.execute("SELECT lease_epoch FROM cp_dispatches WHERE dispatch_id=?", (dispatch_id,)).fetchone()[0]), now=ts)
        inst = _require_live_instance(conn, instance_id, expected_profile=disp["receiver_profile"], now=ts)
        conn.execute(
            "INSERT INTO cp_blockers(blocker_id,dispatch_id,profile_id,instance_id,severity,kind,summary,details_json,requires_response,response_profile,response_deadline_at_ms,status,created_at_ms,updated_at_ms) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (blocker_id, dispatch_id, inst["profile_id"], instance_id, severity, kind, redact_text(summary), dumps_redacted(details or {}), 1 if requires_response else 0, response_profile, response_deadline_at_ms, "open", ts, ts),
        )
        conn.execute("UPDATE cp_dispatches SET status='blocked', blocked_at_ms=?, updated_at_ms=? WHERE dispatch_id=?", (ts, ts, dispatch_id))
        _enqueue_outbox(conn, subject_type="blocker", subject_id=blocker_id, subject_version=1, event_id=None, payload={"event": "blocker.opened", "blocker_id": blocker_id})
    return blocker_id


def resolve_blocker(conn: sqlite3.Connection, blocker_id: str, *, resolver_instance_id: str, resolution: dict[str, Any] | None = None, now_ms: int | None = None) -> bool:
    ts = _ts(now_ms)
    with transaction(conn):
        inst = _require_live_instance(conn, resolver_instance_id, now=ts)
        row = conn.execute("SELECT * FROM cp_blockers WHERE blocker_id=?", (blocker_id,)).fetchone()
        if not row:
            return False
        allowed = inst["role"] in {"admin", "pm"} or row["response_profile"] == inst["profile_id"] or row["profile_id"] == inst["profile_id"]
        if not allowed:
            raise PermissionError("blocker resolution requires response profile, PM, admin, or opener")
        cur = conn.execute("UPDATE cp_blockers SET status='resolved', resolution_json=?, updated_at_ms=? WHERE blocker_id=? AND status IN ('open','acknowledged')", (dumps_redacted(resolution or {}), ts, blocker_id))
        if cur.rowcount != 1:
            return False
        _enqueue_outbox(conn, subject_type="blocker", subject_id=blocker_id, subject_version=2, event_id=None, payload={"event": "blocker.resolved", "blocker_id": blocker_id})
    return True


def start_supervision_run(conn: sqlite3.Connection, *, actor_instance_id: str, scope: dict[str, Any] | None = None, now_ms: int | None = None) -> str:
    ts = _ts(now_ms)
    run_id = f"sup_{uuid.uuid4().hex}"
    with transaction(conn):
        inst = _require_live_instance(conn, actor_instance_id, now=ts)
        if inst["role"] not in {"admin", "pm", "observer"}:
            raise PermissionError("supervision run requires admin/pm/observer instance")
        conn.execute("INSERT INTO cp_supervision_runs(run_id,started_at_ms,finished_at_ms,actor_profile,actor_instance_id,scope_json,findings_json,actions_json,status) VALUES(?,?,?,?,?,?,?,?,?)", (run_id, ts, None, inst["profile_id"], actor_instance_id, dumps_redacted(scope or {}), "[]", "[]", "running"))
    return run_id


def finish_supervision_run(conn: sqlite3.Connection, run_id: str, *, status: Literal["completed", "failed"], findings: list[dict[str, Any]] | None = None, actions: list[dict[str, Any]] | None = None, now_ms: int | None = None) -> bool:
    ts = _ts(now_ms)
    with transaction(conn):
        cur = conn.execute("UPDATE cp_supervision_runs SET finished_at_ms=?, findings_json=?, actions_json=?, status=? WHERE run_id=? AND status='running'", (ts, dumps_redacted(findings or []), dumps_redacted(actions or []), status, run_id))
    return cur.rowcount == 1


def set_runtime_mapping(conn: sqlite3.Connection, *, control_profile_id: str, runtime_profile: str, role: str = "worker", enabled: bool = True, actor_instance_id: str | None = None) -> None:
    ts = now_ms()
    with transaction(conn):
        if actor_instance_id:
            inst = _require_live_instance(conn, actor_instance_id, now=ts)
            if inst["role"] not in {"admin", "pm"}:
                raise PermissionError("runtime mapping update requires admin/pm instance")
        conn.execute(
            "INSERT INTO cp_runtime_mappings(control_profile_id,runtime_profile,role,enabled,created_at_ms,updated_at_ms) VALUES(?,?,?,?,?,?) "
            "ON CONFLICT(control_profile_id) DO UPDATE SET runtime_profile=excluded.runtime_profile, role=excluded.role, enabled=excluded.enabled, updated_at_ms=excluded.updated_at_ms",
            (control_profile_id, runtime_profile, role, 1 if enabled else 0, ts, ts),
        )


def get_runtime_mapping(conn: sqlite3.Connection, control_profile_id: str) -> dict[str, Any] | None:
    row = conn.execute("SELECT * FROM cp_runtime_mappings WHERE control_profile_id=?", (control_profile_id,)).fetchone()
    return dict(row) if row else None


def list_status_events(conn: sqlite3.Connection, *, dispatch_id: str | None = None, profile_id: str | None = None, limit: int = 50) -> list[dict[str, Any]]:
    clauses = []
    params: list[Any] = []
    if dispatch_id:
        clauses.append("dispatch_id=?")
        params.append(dispatch_id)
    if profile_id:
        clauses.append("profile_id=?")
        params.append(profile_id)
    where = "WHERE " + " AND ".join(clauses) if clauses else ""
    rows = conn.execute(f"SELECT * FROM cp_status_events {where} ORDER BY created_at_ms DESC LIMIT ?", (*params, int(limit))).fetchall()
    return [dict(r) for r in rows]


def list_blockers(conn: sqlite3.Connection, *, dispatch_id: str | None = None, status: str | None = None, response_profile: str | None = None, limit: int = 50) -> list[dict[str, Any]]:
    clauses = []
    params: list[Any] = []
    if dispatch_id:
        clauses.append("dispatch_id=?")
        params.append(dispatch_id)
    if status:
        clauses.append("status=?")
        params.append(status)
    if response_profile:
        clauses.append("response_profile=?")
        params.append(response_profile)
    where = "WHERE " + " AND ".join(clauses) if clauses else ""
    rows = conn.execute(f"SELECT * FROM cp_blockers {where} ORDER BY created_at_ms DESC LIMIT ?", (*params, int(limit))).fetchall()
    return [dict(r) for r in rows]


def list_supervision_runs(conn: sqlite3.Connection, *, status: str | None = None, limit: int = 50) -> list[dict[str, Any]]:
    clauses = []
    params: list[Any] = []
    if status:
        clauses.append("status=?")
        params.append(status)
    where = "WHERE " + " AND ".join(clauses) if clauses else ""
    rows = conn.execute(f"SELECT * FROM cp_supervision_runs {where} ORDER BY started_at_ms DESC LIMIT ?", (*params, int(limit))).fetchall()
    return [dict(r) for r in rows]


@dataclass(frozen=True)
class DoctorIssue:
    level: str
    code: str
    detail: str


def doctor(conn: sqlite3.Connection, *, root: Path | None = None) -> list[DoctorIssue]:
    issues: list[DoctorIssue] = []
    mode = get_authority_mode(conn)
    if mode == "legacy":
        issues.append(DoctorIssue("warn", "authority_legacy", "control DB authority is disabled"))
    dead_outbox = conn.execute("SELECT COUNT(*) FROM cp_outbox WHERE status='dead_letter'").fetchone()[0]
    if dead_outbox:
        issues.append(DoctorIssue("error", "dead_outbox", f"{dead_outbox} outbox rows are dead_letter"))
    ts = now_ms()
    stale = conn.execute("SELECT COUNT(*) FROM cp_profile_instances WHERE status='online' AND lease_expires_at_ms IS NOT NULL AND lease_expires_at_ms < ?", (ts,)).fetchone()[0]
    if stale:
        issues.append(DoctorIssue("warn", "stale_instances", f"{stale} profile instances missed heartbeat"))
        stale_bootstrap = conn.execute(
            "SELECT COUNT(*) FROM cp_profile_instances WHERE instance_id IN ('default:bootstrap','statutepm:bootstrap') AND status='online' AND lease_expires_at_ms IS NOT NULL AND lease_expires_at_ms < ?",
            (ts,),
        ).fetchone()[0]
        stale_spawned = conn.execute(
            "SELECT COUNT(*) FROM cp_profile_instances WHERE instance_id NOT IN ('default:bootstrap','statutepm:bootstrap') AND status='online' AND lease_expires_at_ms IS NOT NULL AND lease_expires_at_ms < ?",
            (ts,),
        ).fetchone()[0]
        if stale_bootstrap:
            issues.append(DoctorIssue("warn", "stale_bootstrap_instances", f"{stale_bootstrap} seeded bootstrap leases expired"))
        if stale_spawned:
            issues.append(DoctorIssue("warn", "stale_spawned_instances", f"{stale_spawned} spawned worker/admin instances missed heartbeat"))
    roots = conn.execute("PRAGMA database_list").fetchall()
    db_file = Path(roots[0][2]).resolve() if roots and roots[0][2] else None
    expected = control_db_path(root).resolve()
    if db_file and db_file != expected:
        issues.append(DoctorIssue("error", "wrong_root", f"control DB path {db_file} != expected {expected}"))
    for path, expected_mode, code in ((expected.parent, 0o700, "permissive_control_dir"), (expected, 0o600, "permissive_control_db"), (_pepper_path(root), 0o600, "permissive_control_pepper")):
        if path.exists():
            mode_bits = path.stat().st_mode & 0o777
            if mode_bits != expected_mode:
                issues.append(DoctorIssue("warn", code, f"{path} mode {oct(mode_bits)} should be {oct(expected_mode)}"))
    return issues


def bootstrap_default_policies(conn: sqlite3.Connection, *, admin_profile: str = "default") -> None:
    register_profile(conn, admin_profile, role="admin", display_name="Default/Galt", actor_type="bootstrap")
    # Idempotent defaults: default can route operative messages/dispatches to anyone;
    # workers may only send status/action_required/decision_request back to default.
    policies = [
        (100, "allow", admin_profile, "*", "*", "message"),
        (100, "allow", admin_profile, "*", "dispatch", "dispatch"),
        (90, "allow", "*", admin_profile, "status", "message"),
        (90, "allow", "*", admin_profile, "action_required", "message"),
        (90, "allow", "*", admin_profile, "decision_request", "message"),
    ]
    with transaction(conn):
        for priority, effect, sender, receiver, kind, capability in policies:
            conn.execute(
                "INSERT OR IGNORE INTO cp_route_policies(policy_id,priority,effect,sender_profile,receiver_profile,kind,capability,created_by,created_by_type,created_at_ms) VALUES(?,?,?,?,?,?,?,?,?,?)",
                (f"pol_{hashlib.sha256('|'.join(map(str, (priority,effect,sender,receiver,kind,capability))).encode()).hexdigest()[:24]}", priority, effect, sender, receiver, kind, capability, "bootstrap", "bootstrap", now_ms()),
            )


def _stable_policy_id(priority: int, effect: str, sender: str, receiver: str, kind: str, capability: str) -> str:
    return f"pol_{hashlib.sha256('|'.join(map(str, (priority, effect, sender, receiver, kind, capability))).encode()).hexdigest()[:24]}"


def bootstrap_statutepm_policies(
    conn: sqlite3.Connection,
    *,
    admin_profile: str = "default",
    pm_profile: str = "statutepm",
    worker_profile: str = "statute-worker",
    actor: str = "bootstrap",
    seed_instances: bool = False,
    instance_lease_ms: int = BOOTSTRAP_INSTANCE_LEASE_MS,
) -> dict[str, Any]:
    ts = now_ms()
    profile_specs = [
        (admin_profile, "admin", "Galt/default"),
        (pm_profile, "pm", "Statute PM"),
        (worker_profile, "worker", "Statute Worker"),
    ]
    route_specs = [
        (120, "allow", admin_profile, pm_profile, "dispatch", "dispatch"),
        (120, "allow", admin_profile, pm_profile, "instruction", "message"),
        (110, "allow", pm_profile, worker_profile, "dispatch", "dispatch"),
        (110, "allow", pm_profile, worker_profile, "instruction", "message"),
        (100, "allow", worker_profile, pm_profile, "status", "message"),
        (100, "allow", worker_profile, pm_profile, "artifact", "message"),
        (100, "allow", worker_profile, pm_profile, "action_required", "message"),
        (100, "allow", worker_profile, pm_profile, "decision_request", "message"),
        (100, "allow", pm_profile, admin_profile, "status", "message"),
        (100, "allow", pm_profile, admin_profile, "artifact", "message"),
        (100, "allow", pm_profile, admin_profile, "action_required", "message"),
        (100, "allow", pm_profile, admin_profile, "decision_request", "message"),
    ]
    result: dict[str, Any] = {
        "profiles": {},
        "instances": {admin_profile: None, pm_profile: None},
        "routes": [],
    }
    with transaction(conn):
        for profile_id, role, display in profile_specs:
            existing = conn.execute("SELECT profile_id FROM cp_profiles WHERE profile_id=?", (profile_id,)).fetchone()
            conn.execute(
                "INSERT INTO cp_profiles(profile_id,role,display_name,created_at_ms,updated_at_ms) VALUES(?,?,?,?,?) "
                "ON CONFLICT(profile_id) DO UPDATE SET role=excluded.role, display_name=excluded.display_name, updated_at_ms=excluded.updated_at_ms",
                (profile_id, role, display, ts, ts),
            )
            result["profiles"][profile_id] = {"role": role, "display": display, "status": "existing" if existing else "created"}
        for priority, effect, sender, receiver, kind, capability in route_specs:
            policy_id = _stable_policy_id(priority, effect, sender, receiver, kind, capability)
            existing = conn.execute("SELECT policy_id FROM cp_route_policies WHERE policy_id=?", (policy_id,)).fetchone()
            conn.execute(
                "INSERT OR IGNORE INTO cp_route_policies(policy_id,priority,effect,sender_profile,receiver_profile,kind,capability,created_by,created_by_type,created_at_ms) VALUES(?,?,?,?,?,?,?,?,?,?)",
                (policy_id, priority, effect, sender, receiver, kind, capability, actor, "bootstrap", ts),
            )
            result["routes"].append(
                {
                    "policy_id": policy_id,
                    "priority": priority,
                    "effect": effect,
                    "sender": sender,
                    "receiver": receiver,
                    "kind": kind,
                    "capability": capability,
                    "status": "existing" if existing else "created",
                }
            )
        if seed_instances:
            for profile_id in (admin_profile, pm_profile):
                inst = f"{profile_id}:bootstrap"
                conn.execute(
                    "INSERT INTO cp_profile_instances(instance_id,profile_id,pid,host,started_at_ms,heartbeat_at_ms,lease_expires_at_ms,status,metadata_json) "
                    "VALUES(?,?,?,?,?,?,?,?,?) ON CONFLICT(instance_id) DO UPDATE SET heartbeat_at_ms=excluded.heartbeat_at_ms, lease_expires_at_ms=excluded.lease_expires_at_ms, status='online', metadata_json=excluded.metadata_json",
                    (inst, profile_id, os.getpid(), os.uname().nodename, ts, ts, ts + instance_lease_ms, "online", dumps_redacted({"seeded_by_bootstrap": True})),
                )
                result["instances"][profile_id] = inst
    return result
