"""Bounded Bridge-to-Hermes Issue 22 control and pending-intake admission."""
from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import re
import sqlite3
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone, UTC
from pathlib import Path
from typing import Callable, Iterator

from github_bridge.auth import GitHubAppAuth
from github_bridge.config import BridgeConfig, load_config
from hermes_cli import active_sessions, kanban_db, kanban_db_connect
from hermes_cli.active_sessions import ActiveSessionRegistryError

ISSUE_NUMBER = 22
MAX_BODY_BYTES = 1024
MAX_COMMENT_PAGE = 100
MAX_REQUEST_TIMEOUT_SECONDS = 5
MAX_INTENT_TTL_SECONDS = 900
MAX_SESSION_FENCE_WAIT_SECONDS = 1.0
MAX_DB_BUSY_TIMEOUT_MS = 200
MAX_RECEIPTS_PER_ISSUE = 100
MAX_PENDING_INTAKES_PER_ISSUE = 100
_ISSUE_SEAL = object()
_TASK_RESULT_PUBLISH_LOCK = threading.Lock()


class AdmissionRefused(RuntimeError):
    pass


@dataclass(frozen=True)
class TrustPins:
    repository: str
    repository_id: int
    app_id: int
    installation_id: int
    actor_id: int
    issue_number: int = ISSUE_NUMBER
    result_files: tuple[str, ...] = ()


@dataclass(frozen=True, init=False)
class AuthenticatedIssue:
    repository: str
    repository_id: int
    app_id: int
    installation_id: int
    issue_number: int
    issue_state: str

    def __init__(self, seal, *, repository, repository_id, app_id, installation_id,
                 issue_number, issue_state):
        if seal is not _ISSUE_SEAL:
            raise TypeError("AuthenticatedIssue is created only by the Bridge REST adapter")
        object.__setattr__(self, "repository", repository)
        object.__setattr__(self, "repository_id", repository_id)
        object.__setattr__(self, "app_id", app_id)
        object.__setattr__(self, "installation_id", installation_id)
        object.__setattr__(self, "issue_number", issue_number)
        object.__setattr__(self, "issue_state", issue_state)


@dataclass(frozen=True, init=False)
class AuthenticatedIntent:
    repository: str
    repository_id: int
    app_id: int
    installation_id: int
    issue_number: int
    issue_state: str
    comment_id: int
    actor_id: int
    actor_type: str
    created_at: int
    kind: str
    nonce: str
    expires_at: int
    body_sha256: str
    task_text: str | None

    def __init__(self, seal, *, repository, repository_id, app_id, installation_id,
                 issue_number, issue_state, comment_id, actor_id, actor_type,
                 created_at, kind, nonce, expires_at, body_sha256, task_text=None):
        if seal is not _ISSUE_SEAL:
            raise TypeError("AuthenticatedIntent is created only by the Bridge REST adapter")
        for name, value in locals().copy().items():
            if name not in {"self", "seal"}:
                object.__setattr__(self, name, value)


@dataclass(frozen=True)
class AdmissionReceipt:
    verdict: str
    kind: str
    repository_id: int
    issue_number: int
    task_id: str
    session_id: str
    nonce: str
    body_sha256: str

    def to_dict(self):
        return dataclasses.asdict(self)


@dataclass(frozen=True)
class TaskLeaseFence:
    task_id: str
    run_id: int
    claim_lock: str
    claim_expires: int

    def assert_held(self, conn: sqlite3.Connection, now: int) -> None:
        if not conn.in_transaction:
            raise AdmissionRefused("TASK_FENCE_OUTSIDE_TRANSACTION")
        row = conn.execute(
            "SELECT t.status,t.claim_lock,t.claim_expires,t.current_run_id,"
            "r.status AS run_status,r.claim_lock AS run_lock,r.claim_expires AS run_expires "
            "FROM tasks t JOIN task_runs r ON r.id=t.current_run_id "
            "WHERE t.id=? AND r.id=? AND r.task_id=t.id",
            (self.task_id, self.run_id),
        ).fetchone()
        if (not row or row["status"] != "running" or row["run_status"] != "running"
                or row["claim_lock"] != self.claim_lock or row["run_lock"] != self.claim_lock
                or int(row["claim_expires"] or 0) != self.claim_expires
                or int(row["run_expires"] or 0) != self.claim_expires
                or row["current_run_id"] != self.run_id or self.claim_expires <= now):
            raise AdmissionRefused("TASK_FENCE_LOST")


class _LegacyGlobalSessionFence:
    """Assertion for Hermes versions with only the established global registry lock.

    ``active_session_liveness_guard`` holds the same registry ``_FileLock`` used by
    every installed lease mutation for the whole context. Admission also holds its
    short bounded SQLite write transaction in that context, so the lease cannot be
    replaced between validation and receipt commit. This is intentionally compatible
    with the installed lock protocol; it serializes brief admission with other lease
    updates across sessions.
    """

    def assert_current(self) -> None:
        return None


@contextmanager
def _session_owner_fence(session_id: str):
    """Prefer the candidate generation fence; fall back to Hermes' installed global guard."""
    candidate_fence = getattr(active_sessions, "session_owner_fence", None)
    if callable(candidate_fence):
        with candidate_fence(
            session_id, timeout=MAX_SESSION_FENCE_WAIT_SECONDS,
        ) as fence:
            yield fence
        return

    guard = getattr(active_sessions, "active_session_liveness_guard", None)
    if not callable(guard):
        raise ActiveSessionRegistryError("Hermes owner has no compatible session guard")
    with guard(session_id) as live:
        if live is not True:
            raise ActiveSessionRegistryError("session lease missing, stale, or ambiguous")
        yield _LegacyGlobalSessionFence()


def _pins_from_config(cfg: BridgeConfig) -> TrustPins:
    cfg.validate()
    expected = (
        cfg.issue_ingress_repository_id,
        cfg.issue_ingress_app_id,
        cfg.issue_ingress_installation_id,
        cfg.issue_ingress_actor_id,
    )
    if any(type(value) is not int or value <= 0 for value in expected):
        raise AdmissionRefused("TRUSTED_INGRESS_PINS_MISSING")
    if cfg.app_id != cfg.issue_ingress_app_id:
        raise AdmissionRefused("APP_PIN_MISMATCH")
    if cfg.installation_id != cfg.issue_ingress_installation_id:
        raise AdmissionRefused("INSTALLATION_PIN_MISMATCH")
    if cfg.full_repo != "ISVGitHub/autonomous-entity":
        raise AdmissionRefused("REPOSITORY_NAME_PIN_MISMATCH")
    return TrustPins(
        cfg.full_repo, cfg.issue_ingress_repository_id, cfg.issue_ingress_app_id,
        cfg.issue_ingress_installation_id, cfg.issue_ingress_actor_id,
        cfg.issue_ingress_issue_number,
        cfg.issue_ingress_result_files,
    )


def validate_ingress_config(config_path: Path) -> TrustPins:
    """Validate local target pins before opening the owner database or doing HTTP."""
    return _pins_from_config(load_config(Path(config_path)))


def _open_authenticated_bridge(config_path: Path):
    cfg = load_config(config_path)
    pins = _pins_from_config(cfg)
    # At most one token exchange plus three fixed reads, each bounded to 5 seconds.
    bounded_cfg = dataclasses.replace(
        cfg, request_timeout=min(cfg.request_timeout, MAX_REQUEST_TIMEOUT_SECONDS)
    )
    auth = GitHubAppAuth(bounded_cfg)
    # Keep the installation token in memory for this one bounded call. The
    # Bridge's generic get_token() persists a cache file; ingress does not
    # need that side effect and must not widen its writable state surface.
    token = auth.fetch_installation_token()
    if token.permissions.get("issues") not in {"read", "write"}:
        raise AdmissionRefused("BRIDGE_ISSUES_READ_NOT_GRANTED")
    auth.get_token = lambda: token
    return cfg, pins, auth


def _fetch_issue(config_path: Path):
    cfg, pins, auth = _open_authenticated_bridge(config_path)
    repository = auth.get_repository(cfg.owner, cfg.repo)
    if (not isinstance(repository, dict) or type(repository.get("id")) is not int
            or repository["id"] != pins.repository_id
            or str(repository.get("full_name") or "").casefold() != pins.repository.casefold()):
        raise AdmissionRefused("REST_REPOSITORY_ID_MISMATCH")
    issue = auth.get_issue(cfg.owner, cfg.repo, pins.issue_number)
    if (not isinstance(issue, dict) or type(issue.get("number")) is not int
            or issue["number"] != pins.issue_number or issue.get("state") != "open"
            or "pull_request" in issue):
        raise AdmissionRefused("REST_ISSUE_SCOPE_MISMATCH")
    return cfg, pins, repository, issue, auth


def read_authenticated_issue(config_path: Path) -> AuthenticatedIssue:
    """Fetch issue/repository metadata through the configured GitHub App client."""
    _cfg, pins, _repository, issue, _auth = _fetch_issue(Path(config_path))
    return AuthenticatedIssue(
        _ISSUE_SEAL, repository=pins.repository, repository_id=pins.repository_id,
        app_id=pins.app_id, installation_id=pins.installation_id,
        issue_number=pins.issue_number, issue_state=issue["state"],
    )


def _timestamp(value) -> int:
    if not isinstance(value, str) or not value:
        raise AdmissionRefused("REST_TIMESTAMP_INVALID")
    try:
        parsed = datetime.fromisoformat(value)
        if parsed.tzinfo is None:
            raise ValueError("timezone required")
        return int(parsed.astimezone(UTC).timestamp())
    except (ValueError, OverflowError) as exc:
        raise AdmissionRefused("REST_TIMESTAMP_INVALID") from exc


def _reject_duplicate_pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise AdmissionRefused("DUPLICATE_JSON_KEY")
        result[key] = value
    return result


def read_authenticated_intent(
    config_path: Path, *, comment_id: int, now: int | None = None,
) -> AuthenticatedIntent:
    """Fetch one comment and return only a sanitized, sealed typed intent.

    The caller supplies only a numeric comment selector; all body and identity
    fields come from authenticated Bridge REST responses and trusted config.
    The call has a fixed upper bound: one repository, one issue, one comments
    page (100 items), with no writes or follow-up HTTP requests.
    """
    if type(comment_id) is not int or comment_id <= 0:
        raise AdmissionRefused("COMMENT_SELECTOR_INVALID")
    cfg, pins, _repository, issue, auth = _fetch_issue(Path(config_path))
    comments = auth.list_issue_comments(cfg.owner, cfg.repo, pins.issue_number,
                                        per_page=MAX_COMMENT_PAGE)
    if not isinstance(comments, list) or len(comments) > MAX_COMMENT_PAGE:
        raise AdmissionRefused("REST_COMMENT_PAGE_INVALID")
    comment = next((row for row in comments
                    if isinstance(row, dict) and type(row.get("id")) is int
                    and row["id"] == comment_id), None)
    if comment is None:
        raise AdmissionRefused("REST_COMMENT_NOT_FOUND")
    user = comment.get("user")
    actor_id = user.get("id") if isinstance(user, dict) else None
    actor_type = user.get("type") if isinstance(user, dict) else None
    created_at = _timestamp(comment.get("created_at"))
    updated_at = _timestamp(comment.get("updated_at"))
    body = comment.get("body")
    if type(actor_id) is not int or actor_id != pins.actor_id or actor_type != "User":
        raise AdmissionRefused("REST_ACTOR_PIN_MISMATCH")
    if created_at != updated_at:
        raise AdmissionRefused("EDITED_COMMENT_REFUSED")
    if not isinstance(body, str) or len(body.encode("utf-8")) > getattr(cfg, "issue_ingress_max_body_bytes", MAX_BODY_BYTES):
        raise AdmissionRefused("BODY_BOUND")
    try:
        payload = json.loads(body, object_pairs_hook=_reject_duplicate_pairs)
    except (ValueError, TypeError) as exc:
        if isinstance(exc, AdmissionRefused):
            raise
        raise AdmissionRefused("TYPED_JSON_REQUIRED") from exc
    if (not isinstance(payload, dict)
            or not {"kind", "nonce", "expires_at"}.issubset(payload)):
        raise AdmissionRefused("PAYLOAD_SCHEMA")
    kind, nonce, expires_at = payload["kind"], payload["nonce"], payload["expires_at"]
    current = int(time.time()) if now is None else int(now)
    expected_keys = {"kind", "nonce", "expires_at"}
    if kind == "TASK":
        expected_keys.add("task")
        task_text = payload.get("task")
        if (not isinstance(task_text, str) or not task_text.strip()
                or len(task_text.encode("utf-8")) > getattr(cfg, "issue_ingress_max_body_bytes", MAX_BODY_BYTES) - 128):
            raise AdmissionRefused("TASK_TEXT_BOUND")
    elif kind not in {"PING", "STATUS"}:
        raise AdmissionRefused("CAPABILITY_DENIED")
    if set(payload) != expected_keys:
        raise AdmissionRefused("PAYLOAD_SCHEMA")
    if not isinstance(nonce, str) or not re.fullmatch(r"[A-Za-z0-9_-]{16,64}", nonce):
        raise AdmissionRefused("NONCE_SCHEMA")
    if (type(expires_at) is not int or not created_at <= current < expires_at
            or expires_at > created_at + MAX_INTENT_TTL_SECONDS):
        raise AdmissionRefused("EXPIRED_OR_EXCESSIVE_TTL")
    return AuthenticatedIntent(
        _ISSUE_SEAL, repository=pins.repository, repository_id=pins.repository_id,
        app_id=pins.app_id, installation_id=pins.installation_id,
        issue_number=pins.issue_number, issue_state=issue["state"],
        comment_id=comment_id, actor_id=actor_id, actor_type=actor_type,
        created_at=created_at, kind=kind, nonce=nonce, expires_at=expires_at,
        body_sha256=hashlib.sha256(body.encode("utf-8")).hexdigest(),
        task_text=task_text if kind == "TASK" else None,
    )


def _verify_issue_object(issue: AuthenticatedIssue, cfg: BridgeConfig) -> None:
    pins = _pins_from_config(cfg)
    if (type(issue) is not AuthenticatedIssue or issue.repository != pins.repository
            or issue.repository_id != pins.repository_id or issue.app_id != pins.app_id
            or issue.installation_id != pins.installation_id or issue.issue_number != pins.issue_number
            or issue.issue_state != "open"):
        raise AdmissionRefused("ISSUE_BINDING_AUTHENTICATION")


def create_task_with_issue_binding(
    conn: sqlite3.Connection, *, issue: AuthenticatedIssue, config_path: Path,
    title: str, body: str | None, session_id: str, created_by: str,
    now: int | None = None,
) -> str:
    """Compose the canonical create_task path and Issue binding in one transaction.

    Task content is supplied by the trusted local mission creator, never the
    GitHub comment. New bound tasks start blocked and unassigned; admission text
    cannot create or dispatch work. Existing `create_task` composes by savepoint.
    """
    cfg = load_config(Path(config_path))
    _verify_issue_object(issue, cfg)
    if not title or not title.strip() or not session_id or not created_by:
        raise AdmissionRefused("LOCAL_TASK_FIELDS_REQUIRED")
    recorded_at = int(time.time()) if now is None else int(now)
    if conn.in_transaction:
        raise AdmissionRefused("TASK_CREATE_OUTER_TRANSACTION_REQUIRED")
    try:
        with _session_owner_fence(session_id) as session_fence:
            with kanban_db_connect.write_txn(conn):
                task_id = kanban_db.create_task(
                    conn, title=title, body=body, created_by=created_by, session_id=session_id,
                    initial_status="blocked", assignee=None, workspace_kind="scratch",
                    project_id="", board=None,
                )
                existing = conn.execute(
                    "SELECT task_id,payload FROM task_events WHERE kind='external_issue_binding'"
                ).fetchall()
                for row in existing:
                    data = json.loads(row["payload"] or "{}")
                    if (data.get("repository_id"), data.get("issue_number")) == (
                        issue.repository_id, issue.issue_number
                    ):
                        raise AdmissionRefused("ISSUE_ALREADY_BOUND")
                payload = {
                    "repository": issue.repository, "repository_id": issue.repository_id,
                    "app_id": issue.app_id, "installation_id": issue.installation_id,
                    "issue_number": issue.issue_number, "session_id": session_id,
                    "recorded_at": recorded_at, "verdict": "BOUND",
                }
                encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
                conn.execute(
                    "INSERT INTO task_events(task_id,run_id,kind,payload,created_at) "
                    "VALUES (?,NULL,'external_issue_binding',?,?)",
                    (task_id, encoded, recorded_at),
                )
                conn.execute(
                    "INSERT INTO task_events(task_id,run_id,kind,payload,created_at) "
                    "VALUES (?,NULL,'issue_binding_receipt',?,?)",
                    (task_id, encoded, recorded_at),
                )
                session_fence.assert_current()
                return task_id
    except ActiveSessionRegistryError as exc:
        raise AdmissionRefused("SESSION_FENCE_REFUSED") from exc
    except sqlite3.Error as exc:
        raise AdmissionRefused("TASK_BINDING_TRANSACTION_FAILED") from exc


@contextmanager
def _owner_write_txn(conn: sqlite3.Connection):
    if conn.in_transaction:
        raise AdmissionRefused("INGRESS_TRANSACTION_ALREADY_OPEN")
    row = conn.execute("PRAGMA busy_timeout").fetchone()
    old_timeout = int(row[0]) if row else 0
    conn.execute(f"PRAGMA busy_timeout={MAX_DB_BUSY_TIMEOUT_MS}")
    try:
        with kanban_db_connect.write_txn(conn):
            yield
    finally:
        if conn.in_transaction:
            conn.rollback()
        conn.execute(f"PRAGMA busy_timeout={old_timeout}")


def _resolve_binding(conn: sqlite3.Connection, intent: AuthenticatedIntent):
    bindings = []
    rows = conn.execute(
        "SELECT task_id,payload FROM task_events WHERE kind='external_issue_binding'"
    ).fetchall()
    for row in rows:
        try:
            binding = json.loads(row["payload"] or "{}")
        except (TypeError, ValueError) as exc:
            raise AdmissionRefused("BINDING_CORRUPT") from exc
        if (binding.get("repository_id"), binding.get("issue_number")) == (
            intent.repository_id, intent.issue_number
        ):
            bindings.append((str(row["task_id"]), binding))
    if len(bindings) != 1:
        raise AdmissionRefused("BINDING_MISSING_OR_AMBIGUOUS")
    task_id, binding = bindings[0]
    session_id = binding.get("session_id")
    if not isinstance(session_id, str) or not session_id:
        raise AdmissionRefused("BINDING_CORRUPT")
    return task_id, session_id


def admit_intent(
    conn: sqlite3.Connection, *, intent: AuthenticatedIntent, now: int | None = None,
) -> AdmissionReceipt:
    """Persist one bounded control intent or pending task pointer.

    PING/STATUS stay control-plane receipts. TASK stores only the authenticated
    comment pointer and body hash in a pending queue; the task text remains on
    GitHub, and no task/session is created or dispatched here.
    """
    if type(intent) is not AuthenticatedIntent:
        raise AdmissionRefused("AUTHENTICATED_INTENT_REQUIRED")
    clock = (lambda: int(time.time())) if now is None else (lambda: int(now))
    current = clock()
    if current >= intent.expires_at:
        raise AdmissionRefused("INTENT_EXPIRED_BEFORE_ADMISSION")
    try:
        with _owner_write_txn(conn):
            current = clock()
            if current >= intent.expires_at:
                raise AdmissionRefused("INTENT_EXPIRED_DURING_ADMISSION")
            # A nonce names one immutable intent across control and task kinds.
            other_table = "issue22_control_receipts" if intent.kind == "TASK" else "issue22_pending_intakes"
            if conn.execute("SELECT 1 FROM sqlite_master WHERE name=?", (other_table,)).fetchone():
                if conn.execute(f"SELECT 1 FROM {other_table} WHERE repository_id=? AND issue_number=? AND nonce=?",
                                (intent.repository_id, intent.issue_number, intent.nonce)).fetchone():
                    raise AdmissionRefused("NONCE_REUSED_ACROSS_KINDS")
            if intent.kind == "TASK":
                conn.execute(
                    "CREATE TABLE IF NOT EXISTS issue22_pending_intakes ("
                    "repository_id INTEGER NOT NULL, issue_number INTEGER NOT NULL, "
                    "nonce TEXT NOT NULL, body_sha256 TEXT NOT NULL, comment_id INTEGER NOT NULL, "
                    "actor_id INTEGER NOT NULL, received_at INTEGER NOT NULL, expires_at INTEGER NOT NULL, "
                    "state TEXT NOT NULL CHECK(state='awaiting_binding'), "
                    "result_comment_id INTEGER, dispatch_state TEXT NOT NULL DEFAULT 'awaiting_binding', "
                    "task_id TEXT, session_id TEXT, executor_profile TEXT, bound_at INTEGER, "
                    "task_result_comment_id INTEGER, "
                    "PRIMARY KEY(repository_id, issue_number, nonce))"
                )
                _ensure_pending_intake_columns(conn)
                prior = conn.execute(
                    "SELECT body_sha256 FROM issue22_pending_intakes "
                    "WHERE repository_id=? AND issue_number=? AND nonce=?",
                    (intent.repository_id, intent.issue_number, intent.nonce),
                ).fetchone()
                if prior:
                    if prior["body_sha256"] != intent.body_sha256:
                        raise AdmissionRefused("NONCE_REUSED_WITH_DIFFERENT_BODY")
                    return AdmissionReceipt(
                        "ACK", "TASK", intent.repository_id, intent.issue_number,
                        "", "", intent.nonce, intent.body_sha256,
                    )
                count = conn.execute(
                    "SELECT COUNT(*) FROM issue22_pending_intakes "
                    "WHERE repository_id=? AND issue_number=?",
                    (intent.repository_id, intent.issue_number),
                ).fetchone()[0]
                if count >= MAX_PENDING_INTAKES_PER_ISSUE:
                    raise AdmissionRefused("PENDING_INTAKE_CAPACITY_PRESERVE_AUDIT")
                recorded_at = clock()
                if recorded_at >= intent.expires_at:
                    raise AdmissionRefused("INTENT_EXPIRED_DURING_ADMISSION")
                conn.execute(
                    "INSERT INTO issue22_pending_intakes "
                    "(repository_id,issue_number,nonce,body_sha256,comment_id,actor_id,"
                    "received_at,expires_at,state) VALUES (?,?,?,?,?,?,?,?,'awaiting_binding')",
                    (intent.repository_id, intent.issue_number, intent.nonce,
                     intent.body_sha256, intent.comment_id, intent.actor_id, recorded_at,
                     intent.expires_at),
                )
                return AdmissionReceipt(
                    "ACK", "TASK", intent.repository_id, intent.issue_number,
                    "", "", intent.nonce, intent.body_sha256,
                )
            conn.execute(
                "CREATE TABLE IF NOT EXISTS issue22_control_receipts ("
                "repository_id INTEGER NOT NULL, issue_number INTEGER NOT NULL, "
                "nonce TEXT NOT NULL, body_sha256 TEXT NOT NULL, comment_id INTEGER NOT NULL, "
                "kind TEXT NOT NULL, recorded_at INTEGER NOT NULL, result_comment_id INTEGER, "
                "PRIMARY KEY(repository_id, issue_number, nonce))"
            )
            prior = conn.execute(
                "SELECT body_sha256,kind FROM issue22_control_receipts "
                "WHERE repository_id=? AND issue_number=? AND nonce=?",
                (intent.repository_id, intent.issue_number, intent.nonce),
            ).fetchone()
            if prior:
                if prior["body_sha256"] != intent.body_sha256:
                    raise AdmissionRefused("NONCE_REUSED_WITH_DIFFERENT_BODY")
                return AdmissionReceipt(
                    "ACK", prior["kind"], intent.repository_id, intent.issue_number,
                    "", "", intent.nonce, intent.body_sha256,
                )
            count = conn.execute(
                "SELECT COUNT(*) FROM issue22_control_receipts "
                "WHERE repository_id=? AND issue_number=?",
                (intent.repository_id, intent.issue_number),
            ).fetchone()[0]
            if count >= MAX_RECEIPTS_PER_ISSUE:
                raise AdmissionRefused("RECEIPT_CAPACITY_PRESERVE_AUDIT")
            recorded_at = clock()
            if recorded_at >= intent.expires_at:
                raise AdmissionRefused("INTENT_EXPIRED_DURING_ADMISSION")
            conn.execute(
                "INSERT INTO issue22_control_receipts "
                "(repository_id,issue_number,nonce,body_sha256,comment_id,kind,recorded_at) "
                "VALUES (?,?,?,?,?,?,?)",
                (intent.repository_id, intent.issue_number, intent.nonce,
                 intent.body_sha256, intent.comment_id, intent.kind, recorded_at),
            )
            return AdmissionReceipt(
                "ACK", intent.kind, intent.repository_id, intent.issue_number,
                "", "", intent.nonce, intent.body_sha256,
            )
    except sqlite3.Error as exc:
        raise AdmissionRefused("RECEIPT_TRANSACTION_FAILED") from exc


def admit_github_comment(
    conn: sqlite3.Connection, *, config_path: Path, comment_id: int,
    now: int | None = None,
) -> AdmissionReceipt:
    """Fetch one authenticated Bridge comment and admit it through the Hermes owner."""
    intent = read_authenticated_intent(
        Path(config_path), comment_id=comment_id, now=now,
    )
    return admit_intent(conn, intent=intent, now=now)


def read_pending_intake_task(
    conn: sqlite3.Connection, *, config_path: Path, nonce: str,
) -> str:
    """Re-fetch one pending TASK body for local review without storing or executing it.

    Admission freshness is checked at the recorded intake time. The later review
    still requires the same pinned actor, immutable comment, nonce and body hash.
    """
    if not isinstance(nonce, str) or not re.fullmatch(r"[A-Za-z0-9_-]{16,64}", nonce):
        raise AdmissionRefused("PENDING_NONCE_INVALID")
    pins = validate_ingress_config(Path(config_path))
    row = conn.execute(
        "SELECT comment_id,body_sha256,received_at FROM issue22_pending_intakes "
        "WHERE repository_id=? AND issue_number=? AND nonce=? AND state='awaiting_binding'",
        (pins.repository_id, pins.issue_number, nonce),
    ).fetchone()
    if row is None:
        raise AdmissionRefused("PENDING_INTAKE_NOT_FOUND")
    intent = read_authenticated_intent(
        Path(config_path), comment_id=row["comment_id"], now=int(row["received_at"]),
    )
    if (intent.kind != "TASK" or intent.nonce != nonce
            or intent.body_sha256 != row["body_sha256"] or intent.task_text is None):
        raise AdmissionRefused("PENDING_COMMENT_CHANGED")
    return intent.task_text


def _ensure_pending_intake_columns(conn: sqlite3.Connection) -> None:
    """Add dispatch/outbox fields to legacy pending rows without rewriting them."""
    with kanban_db_connect.write_txn(conn, allow_nested=True):
        conn.execute(
            "CREATE TABLE IF NOT EXISTS issue22_pending_intakes ("
            "repository_id INTEGER NOT NULL, issue_number INTEGER NOT NULL, "
            "nonce TEXT NOT NULL, body_sha256 TEXT NOT NULL, comment_id INTEGER NOT NULL, "
            "actor_id INTEGER NOT NULL, received_at INTEGER NOT NULL, expires_at INTEGER NOT NULL, "
            "state TEXT NOT NULL CHECK(state='awaiting_binding'), result_comment_id INTEGER, "
            "dispatch_state TEXT NOT NULL DEFAULT 'awaiting_binding', task_id TEXT, "
            "session_id TEXT, executor_profile TEXT, bound_at INTEGER, task_result_comment_id INTEGER, "
            "PRIMARY KEY(repository_id, issue_number, nonce))"
        )
        columns = {row["name"] for row in conn.execute("PRAGMA table_info(issue22_pending_intakes)")}
        additions = {
            "dispatch_state": "TEXT NOT NULL DEFAULT 'awaiting_binding'",
            "task_id": "TEXT",
            "session_id": "TEXT",
            "executor_profile": "TEXT",
            "bound_at": "INTEGER",
            "task_result_comment_id": "INTEGER",
        }
        for name, declaration in additions.items():
            if name not in columns:
                conn.execute(f"ALTER TABLE issue22_pending_intakes ADD COLUMN {name} {declaration}")


def _issue22_executor(config_path: Path) -> str:
    """Resolve the configured Hermes profile; Gateway owns the run session."""
    cfg = load_config(config_path)
    profile = (cfg.issue_ingress_executor_profile or "").strip()
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}", profile):
        raise AdmissionRefused("TASK_EXECUTOR_NOT_CONFIGURED")
    try:
        from hermes_cli.profiles import resolve_profile_env
        profile_home = Path(resolve_profile_env(profile))
        if not profile_home.is_dir():
            raise FileNotFoundError(profile)
    except Exception as exc:
        raise AdmissionRefused("TASK_EXECUTOR_PROFILE_UNAVAILABLE") from exc
    return profile


def issue22_task_is_active(conn: sqlite3.Connection) -> bool:
    """Fail closed while a bound Issue 22 card is not terminal."""
    _ensure_pending_intake_columns(conn)
    rows = conn.execute(
        "SELECT task_id FROM issue22_pending_intakes "
        "WHERE dispatch_state IN ('queued','running')"
    )
    for row in rows:
        task = conn.execute("SELECT status FROM tasks WHERE id=?", (row["task_id"],)).fetchone()
        if not task or task["status"] in {"ready", "todo", "running", "review", "scheduled"}:
            return True
    return False


def bind_next_pending_task(
    conn: sqlite3.Connection, *, config_path: Path, now: int | None = None,
) -> dict | None:
    """Atomically turn one fresh pending pointer into one assigned Kanban task."""
    current = int(time.time()) if now is None else int(now)
    _ensure_pending_intake_columns(conn)
    pins = validate_ingress_config(Path(config_path))
    pending = conn.execute(
        "SELECT comment_id,nonce,body_sha256,received_at,expires_at,task_id,dispatch_state "
        "FROM issue22_pending_intakes WHERE repository_id=? AND issue_number=? "
        "AND expires_at>? AND dispatch_state='awaiting_binding' "
        "ORDER BY received_at,comment_id LIMIT 1",
        (pins.repository_id, pins.issue_number, current),
    ).fetchone()
    if pending is None:
        return None
    intent = read_authenticated_intent(
        Path(config_path), comment_id=pending["comment_id"], now=current,
    )
    if (intent.kind != "TASK" or intent.nonce != pending["nonce"]
            or intent.body_sha256 != pending["body_sha256"] or not intent.task_text):
        raise AdmissionRefused("PENDING_COMMENT_CHANGED")
    title = next((line.strip(" #\t") for line in intent.task_text.splitlines() if line.strip()), "")
    title = title[:160] or f"Issue 22 task {intent.nonce[:12]}"
    try:
        # Durable admission/binding is independent of a live turn. The existing
        # Kanban dispatcher claims this ready task once; its worker resumes the
        # pinned session and uses Hermes' atomic active-session lease before work.
        with _owner_write_txn(conn):
            row = conn.execute(
                "SELECT dispatch_state,task_id,expires_at,body_sha256 FROM issue22_pending_intakes "
                "WHERE repository_id=? AND issue_number=? AND nonce=?",
                (pins.repository_id, pins.issue_number, intent.nonce),
            ).fetchone()
            if not row:
                raise AdmissionRefused("PENDING_INTAKE_NOT_FOUND")
            if row["dispatch_state"] != "awaiting_binding":
                if row["task_id"]:
                    return {"verdict": "ACK", "state": "queued", "task_id": row["task_id"]}
                raise AdmissionRefused("PENDING_INTAKE_STATE_INVALID")
            if int(row["expires_at"]) <= current:
                raise AdmissionRefused("INTENT_EXPIRED_BEFORE_BINDING")
            if row["body_sha256"] != intent.body_sha256:
                raise AdmissionRefused("PENDING_COMMENT_CHANGED")
            if issue22_task_is_active(conn):
                return {"verdict": "BUSY", "reason": "TASK_EXECUTOR_OCCUPIED"}
            profile = _issue22_executor(config_path)
            binding_clock = current if now is not None else int(time.time())
            if int(row["expires_at"]) <= binding_clock:
                raise AdmissionRefused("INTENT_EXPIRED_BEFORE_BINDING")
            task_id = kanban_db.create_task(
                conn, title=title, body=intent.task_text,
                created_by=f"github:{pins.repository}#{pins.issue_number}",
                assignee=profile, workspace_kind="scratch", project_id="",
                initial_status="ready", board="missions",
                idempotency_key=f"github-issue:{pins.repository_id}:{pins.issue_number}:{intent.nonce}",
                max_runtime_seconds=900, max_retries=1,
            )
            binding = {
                "repository": pins.repository, "repository_id": pins.repository_id,
                "issue_number": pins.issue_number, "comment_id": intent.comment_id,
                "app_id": pins.app_id, "installation_id": pins.installation_id, "actor_id": pins.actor_id,
                "nonce": intent.nonce, "executor_profile": profile, "board": "missions",
                "recorded_at": current,
                "task_sha256": hashlib.sha256(intent.task_text.encode("utf-8")).hexdigest(),
                "result_files": list(pins.result_files),
            }
            conn.execute(
                "INSERT INTO task_events(task_id,run_id,kind,payload,created_at) "
                "VALUES (?,NULL,'github_task_binding',?,?)",
                (task_id, json.dumps(binding, sort_keys=True, separators=(",", ":")), current),
            )
            updated = conn.execute(
                "UPDATE issue22_pending_intakes SET dispatch_state='queued',task_id=?, "
                "session_id=NULL,executor_profile=?,bound_at=? WHERE repository_id=? "
                "AND issue_number=? AND nonce=? AND dispatch_state='awaiting_binding'",
                (task_id, profile, current, pins.repository_id, pins.issue_number, intent.nonce),
            )
            if updated.rowcount != 1:
                raise AdmissionRefused("PENDING_INTAKE_RACE")
            return {"verdict": "ACK", "state": "queued", "task_id": task_id}
    except sqlite3.Error as exc:
        raise AdmissionRefused("TASK_BINDING_TRANSACTION_FAILED") from exc


def publish_control_result(
    config_path: Path, *, comment_id: int, kind: str,
    state: str = "awaiting_binding",
) -> int:
    """Publish/read back a deterministic, sanitized result for one accepted command."""
    if type(comment_id) is not int or comment_id <= 0 or kind not in {"PING", "STATUS", "TASK"}:
        raise AdmissionRefused("RESULT_REQUEST_INVALID")
    cfg, pins, _repository, _issue, auth = _fetch_issue(Path(config_path))
    token = auth.get_token()
    if token.permissions.get("issues") != "write":
        raise AdmissionRefused("BRIDGE_ISSUES_WRITE_NOT_GRANTED")
    marker = f"<!-- hermes-issue22-result:{comment_id} -->"
    if kind == "TASK":
        if state == "queued":
            message = "Task is bound to the configured Hermes executor and queued in Kanban. Final RESULT follows after completion."
        else:
            message = "Task is recorded as pending and awaits a trusted local Hermes session binding."
        body = f"Hermes intake {comment_id}: {message}\n\n{marker}"
    else:
        body = (
            f"Hermes control request {comment_id}: {kind} accepted. "
            "Gateway is responsive; no task was created or dispatched.\n\n" + marker
        )
    from hermes_cli.issue_result import publish_verified_comment
    return publish_verified_comment(auth, cfg, pins, marker, body)


def publish_pending_task_result(
    conn: sqlite3.Connection, *, config_path: Path,
) -> dict | None:
    """Publish one terminal Kanban result with marker-based crash recovery."""
    if not _TASK_RESULT_PUBLISH_LOCK.acquire(blocking=False):
        raise AdmissionRefused("TASK_RESULT_PUBLISH_BUSY")
    try:
        db_path = conn.execute("PRAGMA database_list").fetchone()[2]
        if db_path:
            with active_sessions._FileLock(Path(str(db_path) + ".issue-result.lock"), timeout=1.0) as fence:
                result = _publish_pending_task_result(conn, config_path=config_path, fence=fence)
                fence.assert_current()
                return result
        return _publish_pending_task_result(conn, config_path=config_path)
    finally:
        _TASK_RESULT_PUBLISH_LOCK.release()


def _publish_pending_task_result(
    conn: sqlite3.Connection, *, config_path: Path, fence=None,
) -> dict | None:
    pins = validate_ingress_config(config_path)
    _ensure_pending_intake_columns(conn)
    pending = conn.execute(
        "SELECT repository_id,issue_number,nonce,comment_id,task_id,session_id,executor_profile "
        "FROM issue22_pending_intakes WHERE dispatch_state IN ('queued','running') "
        "AND task_id IS NOT NULL AND task_result_comment_id IS NULL "
        "AND repository_id=? AND issue_number=? ORDER BY bound_at,comment_id LIMIT 1",
        (pins.repository_id, pins.issue_number),
    ).fetchone()
    if pending is None:
        return None
    task = conn.execute(
        "SELECT status,result FROM tasks WHERE id=?", (pending["task_id"],)
    ).fetchone()
    if not task:
        raise AdmissionRefused("BOUND_TASK_MISSING")
    if task["status"] not in {"running", "done", "blocked"}:
        return None
    run = conn.execute(
        "SELECT id,status,outcome,summary,metadata FROM task_runs WHERE task_id=? ORDER BY id DESC LIMIT 1",
        (pending["task_id"],),
    ).fetchone()
    if task["status"] == "done" and (not run or run["outcome"] != "completed"):
        raise AdmissionRefused("TASK_COMPLETION_EVIDENCE_MISSING")

    cfg, pins, _repository, _issue, auth = _fetch_issue(Path(config_path))
    if pins.repository_id != pending["repository_id"]:
        raise AdmissionRefused("RESULT_TRUST_PINS_CHANGED")
    from hermes_cli.issue_delivery import _binding
    binding = _binding(conn, pending["task_id"])
    if any(binding.get(key) != getattr(pins, key) for key in (
        "repository_id", "issue_number", "app_id", "installation_id", "actor_id",
    )) or binding.get("comment_id") != pending["comment_id"]:
        raise AdmissionRefused("RESULT_TRUST_PINS_CHANGED")
    owner = None
    if run:
        from hermes_cli.issue_delivery import canonical_owner_admission
        try:
            owner = canonical_owner_admission(conn, pending["task_id"], run["id"], binding)
        except ValueError as exc:
            if task["status"] == "running":
                return None
            if task["status"] == "done":
                raise AdmissionRefused("TASK_OWNER_EVIDENCE_MISSING") from exc
    session_id = owner["session_id"] if owner else None
    token = auth.get_token()
    if token.permissions.get("issues") != "write":
        raise AdmissionRefused("BRIDGE_ISSUES_WRITE_NOT_GRANTED")
    nonce = pending["nonce"]
    marker = f"<!-- hermes-issue22-task-result:{pending['comment_id']}:{nonce} -->"
    from hermes_cli.issue_delivery import evidence_summary
    summary = (evidence_summary(run["metadata"], pending["task_id"], session_id, pending["comment_id"])
               if task["status"] == "done" else "Task blocked; review local run evidence.")
    if task["status"] == "running":
        marker = f"<!-- hermes-issue-task-running:{pending['comment_id']}:{run['id']} -->"
        summary = "Gateway recorded canonical owner admission for this run; no completion claimed."
    body = "\n".join((
        "## Hermes TASK RESULT",
        f"issue_number: {pins.issue_number}",
        f"task_id: {json.dumps(pending['task_id'], ensure_ascii=False)}",
        f"status: {task['status']}",
        f"outcome: {json.dumps((run['outcome'] if run else None) or 'no_run')}",
        f"executor_profile: {json.dumps(pending['executor_profile'])}",
        f"session_id: {json.dumps(session_id, ensure_ascii=False)}",
        f"summary: {json.dumps(summary, ensure_ascii=False)}",
        marker,
    ))
    from hermes_cli.issue_result import publish_verified_comment
    result_id = publish_verified_comment(auth, cfg, pins, marker, body)

    if task["status"] == "running":
        with _owner_write_txn(conn):
            if not conn.execute("SELECT 1 FROM task_events WHERE task_id=? AND run_id=? AND kind='github_running_status'",
                                (pending["task_id"], run["id"])).fetchone():
                kanban_db._append_event(conn, pending["task_id"], "github_running_status",
                                       {"comment_id": result_id, "session_id": session_id}, run_id=run["id"])
            conn.execute(
                "UPDATE issue22_pending_intakes SET session_id=? WHERE task_id=? AND session_id IS NULL",
                (session_id, pending["task_id"]),
            )
            if fence is not None:
                fence.assert_current()
        return {"verdict": "ACK", "kind": "TASK", "state": "running", "issue_number": pins.issue_number,
                "task_id": pending["task_id"], "session_id": session_id, "comment_id": result_id}

    terminal_state = "done" if task["status"] == "done" else "blocked"
    if fence is not None:
        fence.assert_current()
    with _owner_write_txn(conn):
        row = conn.execute(
            "SELECT task_result_comment_id FROM issue22_pending_intakes "
            "WHERE repository_id=? AND issue_number=? AND nonce=?",
            (pending["repository_id"], pins.issue_number, nonce),
        ).fetchone()
        if not row:
            raise AdmissionRefused("PENDING_INTAKE_NOT_FOUND")
        prior = row["task_result_comment_id"]
        if prior is not None and prior != result_id:
            raise AdmissionRefused("RESULT_COMMENT_CONFLICT")
        conn.execute(
            "UPDATE issue22_pending_intakes SET task_result_comment_id=?,dispatch_state=?,session_id=COALESCE(session_id,?) "
            "WHERE repository_id=? AND issue_number=? AND nonce=?",
            (result_id, terminal_state, session_id, pending["repository_id"], pins.issue_number, nonce),
        )
        if fence is not None:
            fence.assert_current()
    return {
        "verdict": "ACK", "kind": "TASK", "state": "result_published",
        "issue_number": pins.issue_number, "comment_id": result_id,
    }


def record_control_result(
    conn: sqlite3.Connection, *, repository_id: int, issue_number: int,
    nonce: str, result_comment_id: int,
) -> None:
    """Durably link the accepted request receipt to its idempotent GitHub result."""
    if (type(repository_id) is not int or repository_id <= 0
            or type(issue_number) is not int or issue_number not in {22, 27}
            or not isinstance(nonce, str)
            or type(result_comment_id) is not int or result_comment_id <= 0):
        raise AdmissionRefused("RESULT_BINDING_INVALID")
    try:
        with _owner_write_txn(conn):
            matches = []
            for table in ("issue22_control_receipts", "issue22_pending_intakes"):
                if conn.execute(
                    "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (table,),
                ).fetchone() is None:
                    continue
                row = conn.execute(
                    f"SELECT result_comment_id FROM {table} "
                    "WHERE repository_id=? AND issue_number=? AND nonce=?",
                    (repository_id, issue_number, nonce),
                ).fetchone()
                if row is not None:
                    matches.append((table, row))
            if not matches:
                raise AdmissionRefused("RESULT_RECEIPT_NOT_FOUND")
            if len(matches) != 1:
                raise AdmissionRefused("RESULT_RECEIPT_AMBIGUOUS")
            table, row = matches[0]
            prior = row["result_comment_id"]
            if prior is not None and prior != result_comment_id:
                raise AdmissionRefused("RESULT_COMMENT_CONFLICT")
            conn.execute(
                f"UPDATE {table} SET result_comment_id=? "
                "WHERE repository_id=? AND issue_number=? AND nonce=?",
                (result_comment_id, repository_id, issue_number, nonce),
            )
    except sqlite3.Error as exc:
        raise AdmissionRefused("RESULT_RECEIPT_WRITE_FAILED") from exc
