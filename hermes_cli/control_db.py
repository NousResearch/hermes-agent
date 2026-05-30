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

SCHEMA_VERSION = 1
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
_DISPATCH_ADVANCE_STATUSES = {"running", "completed", "failed"}


class ControlPlaneError(RuntimeError):
    pass


class RouteDenied(ControlPlaneError):
    pass


class RedactionFailed(ControlPlaneError):
    pass


def now_ms() -> int:
    return int(time.time() * 1000)


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
        if existing_tables and existing_version != str(SCHEMA_VERSION):
            raise ControlPlaneError(f"unsupported control DB schema_version={existing_version!r}; expected {SCHEMA_VERSION}")
        required = {"cp_meta", "cp_profiles", "cp_profile_instances", "cp_route_policies", "cp_messages", "cp_dispatches", "cp_approvals", "cp_outbox"}
        if "cp_meta" in existing_tables and not required.issubset(existing_tables):
            missing = sorted(required - existing_tables)
            raise ControlPlaneError(f"partial control DB schema missing tables: {missing}")
        return bool(existing_tables)

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
            path TEXT NOT NULL,
            summary TEXT,
            created_at_ms INTEGER NOT NULL
        );
        CREATE TABLE IF NOT EXISTS cp_approvals(
            approval_id TEXT PRIMARY KEY,
            requester_profile TEXT NOT NULL,
            requester_instance_id TEXT NOT NULL,
            approver_profile TEXT NOT NULL,
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
        CREATE INDEX IF NOT EXISTS idx_cp_messages_receiver_status ON cp_messages(receiver_profile,status,created_at_ms);
        CREATE INDEX IF NOT EXISTS idx_cp_dispatches_receiver_status ON cp_dispatches(receiver_profile,status,lease_expires_at_ms);
        CREATE INDEX IF NOT EXISTS idx_cp_approvals_status ON cp_approvals(status,expires_at_ms);
        CREATE INDEX IF NOT EXISTS idx_cp_outbox_status ON cp_outbox(status,created_at_ms);
        """
    )


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


def register_instance(conn: sqlite3.Connection, profile_id: str, *, instance_id: str | None = None, pid: int | None = None, host: str | None = None, lease_ms: int = 120_000, metadata: dict[str, Any] | None = None) -> str:
    register_profile(conn, profile_id)
    inst = instance_id or f"{profile_id}:{uuid.uuid4().hex}"
    ts = now_ms()
    with transaction(conn):
        existing = conn.execute("SELECT profile_id FROM cp_profile_instances WHERE instance_id=?", (inst,)).fetchone()
        if existing and existing["profile_id"] != profile_id:
            raise ControlPlaneError(f"instance_id {inst!r} already belongs to profile {existing['profile_id']!r}")
        conn.execute(
            "INSERT INTO cp_profile_instances(instance_id,profile_id,pid,host,started_at_ms,heartbeat_at_ms,lease_expires_at_ms,status,metadata_json) "
            "VALUES(?,?,?,?,?,?,?,?,?) ON CONFLICT(instance_id) DO UPDATE SET heartbeat_at_ms=excluded.heartbeat_at_ms, lease_expires_at_ms=excluded.lease_expires_at_ms, status='online', metadata_json=excluded.metadata_json",
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


def add_route_policy(conn: sqlite3.Connection, *, effect: Literal["allow", "deny"], sender_profile: str = "*", receiver_profile: str = "*", kind: str = "*", capability: str = "*", priority: int = 0, created_by: str = "unknown", created_by_type: str = "worker", created_by_instance_id: str | None = None) -> str:
    policy_id = f"pol_{uuid.uuid4().hex}"
    with transaction(conn):
        _require_admin_actor(conn, actor_profile=None if created_by == "unknown" else created_by, actor_instance_id=created_by_instance_id, actor_type=created_by_type)
        conn.execute(
            "INSERT INTO cp_route_policies(policy_id,priority,effect,sender_profile,receiver_profile,kind,capability,created_by,created_by_type,created_at_ms) VALUES(?,?,?,?,?,?,?,?,?,?)",
            (policy_id, priority, effect, sender_profile, receiver_profile, kind, capability, created_by, created_by_type, now_ms()),
        )
    return policy_id


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


def create_dispatch(conn: sqlite3.Connection, *, sender_profile: str, receiver_profile: str, payload: dict[str, Any], capability: str = "dispatch", message_id: str | None = None, idempotency_key: str | None = None, max_attempts: int = 3) -> str:
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
            "INSERT INTO cp_dispatches(dispatch_id,message_id,sender_profile,receiver_profile,capability,status,payload_json,attempts,max_attempts,created_at_ms,updated_at_ms,idempotency_key) VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",
            (did, message_id, sender_profile, receiver_profile, capability, "pending", redacted_payload, 0, max_attempts, ts, ts, idempotency_key),
        )
        conn.execute(
            "INSERT INTO cp_dispatch_events(event_id,dispatch_id,event_type,event_json,created_at_ms) VALUES(?,?,?,?,?)",
            (evt, did, "created", dumps_redacted({"status": "pending"}), ts),
        )
        _enqueue_outbox(conn, subject_type="dispatch", subject_id=did, subject_version=1, event_id=evt, payload={"event": "dispatch.created", "dispatch_id": did})
    return did


def claim_dispatch(conn: sqlite3.Connection, dispatch_id: str, *, instance_id: str, lease_ms: int = 300_000) -> tuple[bool, int | None]:
    ts = now_ms()
    with transaction(conn):
        cur = conn.execute(
            "UPDATE cp_dispatches SET status='claimed', lease_instance_id=?, lease_epoch=lease_epoch+1, lease_expires_at_ms=?, attempts=attempts+1, updated_at_ms=? "
            "WHERE dispatch_id=? AND status IN ('pending','retry_ready') AND (lease_expires_at_ms IS NULL OR lease_expires_at_ms < ?)",
            (instance_id, ts + lease_ms, ts, dispatch_id, ts),
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


def advance_dispatch(conn: sqlite3.Connection, dispatch_id: str, *, instance_id: str, lease_epoch: int, status: str, last_error: str | None = None) -> bool:
    if status not in _DISPATCH_ADVANCE_STATUSES:
        raise ControlPlaneError(f"invalid dispatch status: {status}")
    ts = now_ms()
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


def reap_expired_dispatches(conn: sqlite3.Connection, *, limit: int = 100) -> int:
    ts = now_ms()
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


def create_approval(conn: sqlite3.Connection, *, requester_profile: str, requester_instance_id: str, approver_profile: str, command_preview: str, tool_args_preview: str | None = None, ttl_ms: int = 900_000, approval_id: str | None = None) -> str:
    aid = approval_id or f"appr_{uuid.uuid4().hex}"
    ts = now_ms()
    with transaction(conn):
        conn.execute(
            "INSERT INTO cp_approvals(approval_id,requester_profile,requester_instance_id,approver_profile,status,command_preview,tool_args_preview,expires_at_ms,created_at_ms,updated_at_ms) VALUES(?,?,?,?,?,?,?,?,?,?)",
            (aid, requester_profile, requester_instance_id, approver_profile, "pending", redact_text(command_preview), redact_text(tool_args_preview) if tool_args_preview else None, ts + ttl_ms, ts, ts),
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
            "UPDATE cp_approvals SET status=?, decision=?, decision_reason=?, updated_at_ms=? WHERE approval_id=? AND approver_profile=? AND status='pending' AND expires_at_ms > ?",
            (decision, decision, redact_text(reason) if reason else None, ts, approval_id, approver_profile, ts),
        )
        if cur.rowcount != 1:
            return False
        _enqueue_outbox(conn, subject_type="approval", subject_id=approval_id, subject_version=2, event_id=None, payload={"event": f"approval.{decision}", "approval_id": approval_id, "status": decision})
    return True


def consume_approval(conn: sqlite3.Connection, approval_id: str, *, requester_instance_id: str, requester_profile: str | None = None) -> bool:
    ts = now_ms()
    with transaction(conn):
        profile_clause = " AND requester_profile=?" if requester_profile is not None else ""
        params: tuple[Any, ...] = (requester_instance_id, ts, ts, approval_id, ts, requester_instance_id)
        if requester_profile is not None:
            params = params + (requester_profile,)
        cur = conn.execute(
            "UPDATE cp_approvals SET status='consumed', consumed_by_instance_id=?, consumed_at_ms=?, updated_at_ms=? "
            "WHERE approval_id=? AND status='approved' AND consumed_at_ms IS NULL AND expires_at_ms > ? AND requester_instance_id=?" + profile_clause,
            params,
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
    stale = conn.execute("SELECT COUNT(*) FROM cp_profile_instances WHERE status='online' AND lease_expires_at_ms IS NOT NULL AND lease_expires_at_ms < ?", (now_ms(),)).fetchone()[0]
    if stale:
        issues.append(DoctorIssue("warn", "stale_instances", f"{stale} profile instances missed heartbeat"))
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
