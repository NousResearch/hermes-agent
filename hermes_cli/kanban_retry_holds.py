"""Fenced retry holds for dispatcher-proven protocol failures.

This module owns the durable hold and resolution boundary. It is deliberately
policy-neutral: callers supply only task/run/event identities and bounded,
trusted metadata; no prompt, tool arguments, provider credentials, or grade data
belong here.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import sqlite3
import time
from collections.abc import Mapping
from typing import Any
from typing import Optional


_TOKEN_RE = re.compile(r"[0-9a-f]{64}\Z")
_PROVIDER_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}\Z")
_MODEL_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_./:-]{0,127}\Z")
_SAFE_RECEIPT_VALUE_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,255}\Z")
_METADATA_KEY_RE = re.compile(r"[a-z][a-z0-9_]{0,63}\Z")
_ALLOWED_RECEIPT_KEYS = frozenset({
    "source", "schema_sha256", "generation_id", "source_generation_id", "actual_route_sha256",
    "receipt_id", "evidence_id", "event_id", "ticket_id", "issue_id", "run_id", "decision_id",
    "workflow_id", "batch_id", "task_id", "board_id",
})
_REASON_CODE_RE = re.compile(r"[a-z][a-z0-9_.:-]{0,79}\Z")
_BLOCKED_METADATA_TERMS = (
    "prompt", "argument", "args", "payload", "body", "transcript", "content",
    "response", "result", "error", "secret", "credential", "api_key", "base_url",
    "grade", "score", "price", "cost", "answer",
)
_AUTHORITY_EVENT_KINDS = (
    "blocked", "unblocked", "gave_up", "assigned", "status", "reclaimed", "stale",
    "timed_out", "crashed", "spawn_failed", "rate_limited", "promoted", "promoted_manual",
    "claimed", "review_requested", "changes_requested", "review_reopened", "completed",
    "archived", "scheduled", "protocol_violation", "edited", "specified",
    "descendant_invalidated", "dependency_wait", "block_loop_detected", "retry_hold_resolved",
    "retry_retired", "protocol_retry_held",
    "reclaim_deferred", "reconciled", "spawned", "worker_registered", "claim_rejected",
    "skipped_nonspawnable", "respawn_guarded", "commented", "attached", "attachment_removed",
    "reprioritized", "linked", "unlinked", "archive_worker_termination", "pr_acceptance",
)
_PROTOCOL_VIOLATION_SCAN_LIMIT = 50


def _protocol_violation_actions() -> Optional[dict[str, dict[str, str]]]:
    """Strictly parse opt-in per-board/per-assignee actions; invalid config fails closed."""
    from hermes_cli.config import load_config_readonly
    raw = (load_config_readonly() or {}).get("kanban", {}).get("protocol_violation_actions", {})
    if raw is None:
        return {}
    if not isinstance(raw, Mapping):
        _kb._log.warning("kanban: invalid protocol_violation_actions shape; keeping sticky-block behavior")
        return None
    actions: dict[str, dict[str, str]] = {}
    for board_slug, assignees in raw.items():
        if not isinstance(board_slug, str) or not board_slug.strip() or not isinstance(assignees, Mapping):
            _kb._log.warning("kanban: invalid protocol_violation_actions entry; keeping sticky-block behavior")
            return None
        parsed: dict[str, str] = {}
        for assignee, action in assignees.items():
            if (not isinstance(assignee, str) or not assignee.strip()
                    or not isinstance(action, str) or action not in {"block", "defer"}):
                _kb._log.warning("kanban: invalid protocol_violation_actions value; keeping sticky-block behavior")
                return None
            parsed[assignee] = action
        actions[board_slug] = parsed
    return actions


def _protocol_violation_action(conn: sqlite3.Connection, task_id: str, board: Optional[str]) -> str:
    """Resolve one board+assignee action; every absent/invalid entry keeps the existing block."""
    actions = _protocol_violation_actions()
    if actions is None:
        return "block"
    board_slug = str(board or _kb.get_current_board())
    task = conn.execute("SELECT assignee FROM tasks WHERE id = ?", (task_id,)).fetchone()
    assignee = task["assignee"] if task else None
    if not assignee:
        return "block"
    return actions.get(board_slug, {}).get(str(assignee), "block")


def _protocol_violation_run_boundary(conn: sqlite3.Connection, task_id: str) -> int:
    """Return the failed run released by the newest fenced resume, or 0.

    An ordinary ``unblock_task`` is deliberately not a boundary: the default
    (non-opt-in) path keeps counting the same trailing streak, so a manual
    unblock still buys exactly one more attempt before the sticky block.
    """
    boundary = conn.execute(
        "SELECT failed_run_id FROM task_retry_hold_resolutions "
        "WHERE task_id = ? AND action = 'resume' "
        "ORDER BY resolution_event_id DESC LIMIT 1", (task_id,),
    ).fetchone()
    return int(boundary["failed_run_id"]) if boundary else 0


def _protocol_violation_streak(conn: sqlite3.Connection, task_id: str) -> int:
    """Count trailing clean-exit protocol violations after the latest explicit release."""
    from hermes_cli import kanban_db_dispatch as _kbd  # late import: the dispatcher owns the output separator
    streak = 0
    rows = conn.execute(
        "SELECT id, outcome, error, metadata FROM task_runs "
        "WHERE task_id = ? AND ended_at IS NOT NULL AND id > ? "
        "ORDER BY id DESC LIMIT ?",
        (task_id, _protocol_violation_run_boundary(conn, task_id), _PROTOCOL_VIOLATION_SCAN_LIMIT),
    ).fetchall()
    for row in rows:
        outcome = row["outcome"] or ""
        if outcome == "rate_limited":
            continue
        head = (row["error"] or "").partition(_kbd.WORKER_OUTPUT_SEP)[0]  # never the worker's own words
        if outcome == "crashed" and (
            _kb._json_dict(row["metadata"]).get("protocol_violation")
            or "protocol violation" in head
        ):
            streak += 1
            continue
        break
    return streak


def _maybe_hold_protocol_failure(
    conn: sqlite3.Connection, task_id: str, run_id: int, event_id: int,
    board: Optional[str], default_limit: int,
) -> bool:
    """Create an opted-in exhausted hold inside the worker-reap transaction."""
    streak = _protocol_violation_streak(conn, task_id)
    task = conn.execute("SELECT max_retries FROM tasks WHERE id = ?", (task_id,)).fetchone()
    if task is None:
        return False
    override = _kb._row_get(task, "max_retries")
    limit = int(override) if override is not None else default_limit
    if streak < limit or _protocol_violation_action(conn, task_id, board) != "defer":
        return False
    try:
        return _kb.create_protocol_retry_hold(
            conn, task_id,
            expected_failed_run_id=run_id,
            expected_failure_event_id=event_id,
            protocol_violations=streak,
            protocol_violation_limit=limit,
            allow_nested=True,
        )
    except sqlite3.Error:
        _kb._log.error(
            "kanban: could not atomically create protocol retry hold for %s; "
            "existing sticky-block fallback remains active",
            task_id, exc_info=True,
        )
        return False


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _body_sha256(body: Any) -> str:
    return hashlib.sha256((body or "").encode("utf-8")).hexdigest()


def _positive_int(value: Any) -> Optional[int]:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value if value > 0 else None
    if isinstance(value, str) and value.isdecimal() and len(value) <= 20:
        parsed = int(value)
        return parsed if parsed > 0 else None
    return None


def _hold_snapshot(conn: sqlite3.Connection, task_id: str) -> Optional[dict]:
    """Read active hold and all token inputs in one SQLite statement/snapshot."""
    row = conn.execute(
        """
        SELECT h.task_id, h.failed_run_id, h.source_failure_event_id, h.created_at,
               h.metadata_json, t.status, t.current_run_id, t.assignee, t.body,
               (SELECT r.id FROM task_runs r WHERE r.task_id = t.id ORDER BY r.id DESC LIMIT 1)
                   AS latest_run_id,
               (SELECT COALESCE(MAX(e.id), 0) FROM task_events e WHERE e.task_id = t.id)
                   AS max_task_event_id
          FROM task_retry_holds h
          JOIN tasks t ON t.id = h.task_id
         WHERE h.task_id = ?
        """,
        (task_id,),
    ).fetchone()
    if row is None:
        return None
    raw_metadata = row["metadata_json"]
    try:
        metadata = json.loads(raw_metadata)
    except (TypeError, ValueError):
        metadata = None
    metadata_valid = isinstance(metadata, dict)
    if not metadata_valid:
        metadata = {}
    body_sha256 = _body_sha256(row["body"])
    identity = {
        "task_id": row["task_id"],
        "failed_run_id": int(row["failed_run_id"]),
        "source_failure_event_id": int(row["source_failure_event_id"]),
        "created_at": int(row["created_at"]),
        "metadata_json": raw_metadata,
    }
    state = {
        "status": row["status"],
        "current_run_id": row["current_run_id"],
        "latest_run_id": row["latest_run_id"],
        "assignee": row["assignee"],
        "body_sha256": body_sha256,
        "max_task_event_id": int(row["max_task_event_id"]),
    }
    token = hashlib.sha256(_canonical_json({"hold": identity, "task": state}).encode("utf-8")).hexdigest()
    return {
        "task_id": row["task_id"],
        "failed_run_id": int(row["failed_run_id"]),
        "source_failure_event_id": int(row["source_failure_event_id"]),
        "created_at": int(row["created_at"]),
        "hold_token": token,
        "body_sha256": body_sha256,
        "assignee": row["assignee"],
        "status": row["status"],
        "current_run_id": row["current_run_id"],
        "latest_run_id": row["latest_run_id"],
        "max_task_event_id": int(row["max_task_event_id"]),
        "metadata": metadata,
        "metadata_valid": metadata_valid,
    }


def _public_hold(snapshot: Optional[dict]) -> Optional[dict]:
    if snapshot is None:
        return None
    return {
        key: snapshot[key]
        for key in (
            "task_id", "failed_run_id", "source_failure_event_id", "hold_token",
            "body_sha256", "assignee", "status", "metadata",
        )
    }


def get_retry_hold(conn: sqlite3.Connection, task_id: str) -> Optional[dict]:
    """Return the active hold plus a snapshot-bound SHA-256 concurrency token."""
    return _public_hold(_hold_snapshot(conn, str(task_id)))


def _retry_hold_exists(conn: sqlite3.Connection, task_id: str) -> bool:
    return conn.execute(
        "SELECT 1 FROM task_retry_holds WHERE task_id = ?", (task_id,),
    ).fetchone() is not None


def _append_retry_event(conn: sqlite3.Connection, task_id: str, kind: str, payload: dict) -> int:
    cursor = conn.execute(
        "INSERT INTO task_events (task_id, run_id, kind, payload, created_at) "
        "VALUES (?, NULL, ?, ?, ?)",
        (task_id, kind, _canonical_json(payload), int(time.time())),
    )
    return int(cursor.lastrowid)


def _refusal(reason: str, **details: Any) -> dict:
    return {"ok": False, "reason": reason, **details}


def _record_protocol_retry_hold(
    conn: sqlite3.Connection,
    task_id: str,
    *,
    expected_failed_run_id: int,
    expected_failure_event_id: int,
    protocol_violations: int,
    protocol_violation_limit: int,
    allow_nested: bool = False,
) -> bool:
    """Atomically park one just-reaped protocol failure; caller owns policy lookup."""
    if protocol_violations < protocol_violation_limit:
        return False
    with _kb.write_txn(conn, allow_nested=allow_nested):
        task = conn.execute(
            "SELECT status, current_run_id, claim_lock, claim_expires, worker_pid "
            "FROM tasks WHERE id = ?", (task_id,),
        ).fetchone()
        if task is None or task["status"] not in {"ready", "review"}:
            return False
        if any(task[key] is not None for key in ("current_run_id", "claim_lock", "claim_expires", "worker_pid")):
            return False
        if _retry_hold_exists(conn, task_id):
            return False
        run = conn.execute(
            "SELECT id, outcome, ended_at, metadata FROM task_runs "
            "WHERE task_id = ? ORDER BY id DESC LIMIT 1", (task_id,),
        ).fetchone()
        if (run is None or int(run["id"]) != expected_failed_run_id
                or run["outcome"] != "crashed" or run["ended_at"] is None
                or not _kb._json_dict(run["metadata"]).get("protocol_violation")):
            return False
        source = conn.execute(
            "SELECT payload FROM task_events WHERE id = ? AND task_id = ? "
            "AND run_id = ? AND kind = 'protocol_violation'",
            (expected_failure_event_id, task_id, expected_failed_run_id),
        ).fetchone()
        source_payload = _kb._json_dict(source["payload"]) if source else {}
        if not source_payload.get("protocol_violation"):
            return False
        latest_protocol = conn.execute(
            "SELECT id FROM task_events WHERE task_id = ? AND run_id = ? "
            "AND kind = 'protocol_violation' ORDER BY id DESC LIMIT 1",
            (task_id, expected_failed_run_id),
        ).fetchone()
        latest_authority = conn.execute(
            "SELECT id FROM task_events WHERE task_id = ? "
            f"AND kind IN ({_authority_event_kinds_sql()}) ORDER BY id DESC LIMIT 1",
            (task_id, *_AUTHORITY_EVENT_KINDS),
        ).fetchone()
        if (latest_protocol is None or int(latest_protocol["id"]) != expected_failure_event_id
                or latest_authority is None or int(latest_authority["id"]) != expected_failure_event_id):
            return False
        resume_status = task["status"]
        created_at = int(time.time())
        metadata = {
            "version": 1,
            "reason": "protocol_violation_limit",
            "protocol_violations": int(protocol_violations),
            "protocol_violation_limit": int(protocol_violation_limit),
            "resume_status": resume_status,
        }
        updated = conn.execute(
            "UPDATE tasks SET status = 'scheduled' "
            "WHERE id = ? AND status = ? AND current_run_id IS NULL "
            "AND claim_lock IS NULL AND claim_expires IS NULL AND worker_pid IS NULL",
            (task_id, resume_status),
        )
        if updated.rowcount != 1:
            return False
        conn.execute(
            "INSERT INTO task_retry_holds "
            "(task_id, failed_run_id, source_failure_event_id, created_at, metadata_json) "
            "VALUES (?, ?, ?, ?, ?)",
            (task_id, expected_failed_run_id, expected_failure_event_id, created_at,
             _canonical_json(metadata)),
        )
        _kb._append_event(
            conn, task_id, "protocol_retry_held",
            {
                "version": 1,
                "reason": "protocol_violation_limit",
                "failed_run_id": expected_failed_run_id,
                "source_failure_event_id": expected_failure_event_id,
                "protocol_violations": int(protocol_violations),
                "protocol_violation_limit": int(protocol_violation_limit),
                "status": "scheduled",
                "resume_status": resume_status,
            },
            run_id=expected_failed_run_id,
        )
    return True


def _authority_event_kinds_sql() -> str:
    return ", ".join("?" for _ in _AUTHORITY_EVENT_KINDS)


def adopt_protocol_failure(
    conn: sqlite3.Connection,
    task_id: str,
    *,
    expected_failed_run_id: int,
    expected_failure_event_id: int,
    expected_body_sha256: str,
) -> dict:
    """CAS-adopt only a still-authoritative historical protocol ``gave_up``."""
    if not _TOKEN_RE.fullmatch(str(expected_body_sha256)):
        return _refusal("invalid_body_sha256")
    failed_run_id = _positive_int(expected_failed_run_id)
    failure_event_id = _positive_int(expected_failure_event_id)
    if failed_run_id is None or failure_event_id is None:
        return _refusal("invalid_failure_identity")
    with _kb.write_txn(conn):
        existing = _hold_snapshot(conn, str(task_id))
        if existing is not None:
            if (existing["failed_run_id"] == failed_run_id
                    and existing["source_failure_event_id"] == failure_event_id
                    and existing["body_sha256"] == expected_body_sha256
                    and existing["metadata_valid"]):
                return _public_hold(existing)
            return _refusal("different_hold_already_active")
        task = conn.execute(
            "SELECT status, current_run_id, claim_lock, claim_expires, worker_pid, body "
            "FROM tasks WHERE id = ?", (task_id,),
        ).fetchone()
        if task is None:
            return _refusal("task_not_found")
        if task["status"] != "blocked":
            return _refusal("task_not_blocked")
        if any(task[key] is not None for key in ("current_run_id", "claim_lock", "claim_expires", "worker_pid")):
            return _refusal("claim_not_released")
        body_sha256 = _body_sha256(task["body"])
        if body_sha256 != expected_body_sha256:
            return _refusal("body_changed", body_sha256=body_sha256)
        run = conn.execute(
            "SELECT id, outcome, ended_at, error, metadata FROM task_runs "
            "WHERE task_id = ? ORDER BY id DESC LIMIT 1", (task_id,),
        ).fetchone()
        if (run is None or int(run["id"]) != failed_run_id or run["outcome"] != "crashed"
                or run["ended_at"] is None
                or not _kb._json_dict(run["metadata"]).get("protocol_violation")):
            return _refusal("latest_run_not_protocol_failure")
        source = conn.execute(
            "SELECT payload FROM task_events WHERE id = ? AND task_id = ? "
            "AND run_id = ? AND kind = 'protocol_violation'",
            (failure_event_id, task_id, failed_run_id),
        ).fetchone()
        source_payload = _kb._json_dict(source["payload"]) if source else {}
        if not source_payload.get("protocol_violation"):
            return _refusal("source_event_not_protocol_failure")
        latest_protocol = conn.execute(
            "SELECT id FROM task_events WHERE task_id = ? AND run_id = ? "
            "AND kind = 'protocol_violation' ORDER BY id DESC LIMIT 1",
            (task_id, failed_run_id),
        ).fetchone()
        if latest_protocol is None or int(latest_protocol["id"]) != failure_event_id:
            return _refusal("source_event_not_latest_protocol_failure")
        latest_authority = conn.execute(
            "SELECT id, kind, payload FROM task_events WHERE task_id = ? "
            f"AND kind IN ({_authority_event_kinds_sql()}) ORDER BY id DESC LIMIT 1",
            (task_id, *_AUTHORITY_EVENT_KINDS),
        ).fetchone()
        if latest_authority is None or latest_authority["kind"] != "gave_up":
            return _refusal("newer_lifecycle_authority")
        gave_up = _kb._json_dict(latest_authority["payload"])
        violations = gave_up.get("protocol_violations")
        limit = gave_up.get("protocol_violation_limit")
        if (gave_up.get("sticky") is not True or gave_up.get("trigger_outcome") != "crashed"
                or not isinstance(violations, int) or isinstance(violations, bool)
                or not isinstance(limit, int) or isinstance(limit, bool)
                or limit < 1 or violations < limit):
            return _refusal("gave_up_not_protocol_limit")
        resume_status = source_payload.get("retry_status")
        if resume_status not in {"ready", "review"}:
            resume_status = "ready"
        created_at = int(time.time())
        hold_metadata = {
            "version": 1,
            "reason": "legacy_protocol_failure_adopted",
            "protocol_violations": violations,
            "protocol_violation_limit": limit,
            "resume_status": resume_status,
        }
        changed = conn.execute(
            "UPDATE tasks SET status = 'scheduled' "
            "WHERE id = ? AND status = 'blocked' AND current_run_id IS NULL "
            "AND claim_lock IS NULL AND claim_expires IS NULL AND worker_pid IS NULL",
            (task_id,),
        )
        if changed.rowcount != 1:
            return _refusal("task_changed")
        conn.execute(
            "INSERT INTO task_retry_holds "
            "(task_id, failed_run_id, source_failure_event_id, created_at, metadata_json) "
            "VALUES (?, ?, ?, ?, ?)",
            (task_id, failed_run_id, failure_event_id, created_at, _canonical_json(hold_metadata)),
        )
        _kb._append_event(
            conn, task_id, "protocol_retry_held",
            {
                "version": 1,
                "reason": "legacy_protocol_failure_adopted",
                "failed_run_id": failed_run_id,
                "source_failure_event_id": failure_event_id,
                "protocol_violations": violations,
                "protocol_violation_limit": limit,
                "status": "scheduled",
                "resume_status": resume_status,
            },
            run_id=failed_run_id,
        )
        hold = _hold_snapshot(conn, str(task_id))
    return _public_hold(hold) or _refusal("hold_creation_failed")


def _validated_resolution_metadata(metadata: Any) -> tuple[Optional[dict], Optional[str]]:
    if metadata is None:
        metadata = {}
    if not isinstance(metadata, dict) or len(metadata) > 32:
        return None, "metadata_must_be_small_object"
    safe: dict[str, Any] = {}
    for key, value in metadata.items():
        if not isinstance(key, str) or not _METADATA_KEY_RE.fullmatch(key) or key not in _ALLOWED_RECEIPT_KEYS:
            return None, "invalid_metadata_key"
        lowered = key.lower()
        if any(term in lowered for term in _BLOCKED_METADATA_TERMS):
            return None, "sensitive_metadata_key"
        if isinstance(value, str):
            if not _SAFE_RECEIPT_VALUE_RE.fullmatch(value):
                return None, "invalid_metadata_string"
            if key in {"schema_sha256", "actual_route_sha256"} and not _TOKEN_RE.fullmatch(value):
                return None, "invalid_metadata_digest"
            if key in {"generation_id", "source_generation_id"} and not re.fullmatch(r"gen-[0-9a-f]{32}", value):
                return None, "invalid_metadata_generation"
        elif isinstance(value, float) and not math.isfinite(value):
            return None, "invalid_metadata_number"
        elif value is not None and not isinstance(value, (bool, int, float)):
            return None, "metadata_values_must_be_scalar"
        safe[key] = value
    if len(_canonical_json(safe).encode("utf-8")) > 4096:
        return None, "metadata_too_large"
    return safe, None


def _load_resolution(conn: sqlite3.Connection, hold_token: str) -> Optional[dict]:
    row = conn.execute(
        "SELECT task_id, action, receipt_json FROM task_retry_hold_resolutions "
        "WHERE hold_token = ?", (hold_token,),
    ).fetchone()
    if row is None:
        return None
    try:
        receipt = json.loads(row["receipt_json"])
    except (TypeError, ValueError):
        return _refusal("resolution_receipt_corrupt")
    if not isinstance(receipt, dict):
        return _refusal("resolution_receipt_corrupt")
    return {"task_id": row["task_id"], "action": row["action"], "receipt": receipt}


def resolve_retry_hold(
    conn: sqlite3.Connection,
    task_id: str,
    *,
    expected_hold_token: str,
    action: str,
    reason: str,
    metadata: Optional[dict] = None,
) -> dict:
    """CAS-resume or retire one active hold and retain an idempotent receipt."""
    if not isinstance(expected_hold_token, str) or not _TOKEN_RE.fullmatch(expected_hold_token):
        return _refusal("invalid_hold_token")
    if not isinstance(action, str) or action not in {"resume", "retire"}:
        return _refusal("invalid_action")
    if (not isinstance(reason, str) or not _REASON_CODE_RE.fullmatch(reason)
            or any(term in reason.lower() for term in _BLOCKED_METADATA_TERMS)):
        return _refusal("invalid_reason")
    safe_metadata, metadata_error = _validated_resolution_metadata(metadata)
    if metadata_error:
        return _refusal(metadata_error)
    cleanup = False
    with _kb.write_txn(conn):
        prior = _load_resolution(conn, expected_hold_token)
        if prior is not None:
            if prior.get("ok") is False:
                return prior
            if prior.get("task_id") != task_id or prior.get("action") != action:
                return _refusal("hold_token_already_resolved", receipt=prior.get("receipt"))
            return {**prior["receipt"], "idempotent": True}
        hold = _hold_snapshot(conn, str(task_id))
        if hold is None:
            return _refusal("active_hold_not_found")
        if not hold["metadata_valid"]:
            return _refusal("hold_metadata_invalid")
        if hold["hold_token"] != expected_hold_token:
            return _refusal("stale_hold_token", current_hold=hold)
        if (hold["status"] != "scheduled" or hold["current_run_id"] is not None
                or conn.execute(
                    "SELECT 1 FROM tasks WHERE id = ? AND claim_lock IS NULL "
                    "AND claim_expires IS NULL AND worker_pid IS NULL", (task_id,),
                ).fetchone() is None):
            return _refusal("task_not_quiescent")
        resume_status = hold["metadata"].get("resume_status")
        if resume_status not in {"ready", "review"}:
            return _refusal("hold_resume_status_invalid")
        if action == "retire":
            dependents = conn.execute(
                "SELECT 1 FROM task_links l JOIN tasks child ON child.id = l.child_id "
                "WHERE l.parent_id = ? AND child.status NOT IN ('done', 'archived') LIMIT 1",
                (task_id,),
            ).fetchone()
            if dependents is not None:
                return _refusal("live_dependents_would_be_released")
            target_status = "archived"
            outcome = "retry_retired_non_grade"
            cleanup = True
        else:
            target_status = resume_status if _kb._parents_satisfied(conn, task_id) else "todo"
            outcome = "retry_resumed"
        changed = conn.execute(
            "UPDATE tasks SET status = ?, claim_lock = NULL, claim_expires = NULL, "
            "worker_pid = NULL, worker_started_at = NULL, current_run_id = NULL "
            "WHERE id = ? AND status = 'scheduled' AND current_run_id IS NULL "
            "AND claim_lock IS NULL AND claim_expires IS NULL AND worker_pid IS NULL",
            (target_status, task_id),
        )
        if changed.rowcount != 1:
            return _refusal("task_changed")
        event_payload = {
            "version": 1,
            "action": action,
            "outcome": outcome,
            "status": target_status,
            "resume_status": resume_status,
            "hold_token": expected_hold_token,
            "failed_run_id": hold["failed_run_id"],
            "source_failure_event_id": hold["source_failure_event_id"],
            "reason": reason,
            "metadata": safe_metadata,
        }
        event_kind = "retry_retired" if action == "retire" else "retry_hold_resolved"
        event_id = _append_retry_event(conn, task_id, event_kind, event_payload)
        receipt = {
            "ok": True,
            "task_id": task_id,
            "failed_run_id": hold["failed_run_id"],
            "source_failure_event_id": hold["source_failure_event_id"],
            "hold_token": expected_hold_token,
            "resolution_event_id": event_id,
            "action": action,
            "outcome": outcome,
            "status": target_status,
            "resume_status": resume_status,
            "reason": reason,
            "metadata": safe_metadata,
        }
        conn.execute(
            "INSERT INTO task_retry_hold_resolutions "
            "(hold_token, task_id, failed_run_id, resolution_event_id, action, receipt_json) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (expected_hold_token, task_id, hold["failed_run_id"], event_id, action,
             _canonical_json(receipt)),
        )
        # The hold row was read in this same write transaction, so it is deleted
        # together with the resolution receipt and the release event.
        conn.execute("DELETE FROM task_retry_holds WHERE task_id = ?", (task_id,))
    if cleanup:
        _kb._cleanup_workspace(conn, task_id)
    return receipt


def unblock_task(conn: sqlite3.Connection, task_id: str) -> bool:
    """Resume a task unless a fenced retry hold still owns its release authority."""
    now = int(time.time())
    with _kb.write_txn(conn):
        if _retry_hold_exists(conn, task_id):
            # Only resolve_retry_hold has the hold-token CAS and audited release boundary.
            return False
        resume_status = (
            _kb._resume_status_from_events(conn, task_id)
            if _kb._task_status(conn, task_id) == "blocked"
            else "ready"
        )
        _kb._reclaim_dangling_run(
            conn, task_id, statuses=("blocked", "scheduled"), now=now,
            note="invariant recovery on unblock",
        )
        # Re-gate on parent completion before restoring the source phase.
        landing_status = _kb._landing_status_after_parents(conn, task_id)
        new_status = (
            "review"
            if landing_status == "ready" and resume_status == "review"
            else landing_status
        )
        # Preserve block recurrence history while resetting only the explicit retry budget.
        cur = conn.execute(
            "UPDATE tasks SET status = ?, current_run_id = NULL, "
            "consecutive_failures = 0, last_failure_error = NULL "
            "WHERE id = ? AND status IN ('blocked', 'scheduled')", (new_status, task_id),
        )
        if cur.rowcount != 1:
            return False
        _kb._append_event(
            conn, task_id, "unblocked",
            (
                {"status": new_status, "resume_status": resume_status}
                if new_status != "ready" or resume_status != "ready"
                else None
            ),
        )
        return True


def _validated_run_provenance(value: Any) -> Optional[dict]:
    if not isinstance(value, dict) or set(value) != {
        "version", "worker_session_id", "source_generation_id", "schema_sha256", "actual_routes", "unknown_fields",
    }:
        return None
    if value.get("version") != 1 or isinstance(value.get("version"), bool):
        return None
    session_id = value.get("worker_session_id")
    if session_id is not None and (
        not isinstance(session_id, str) or not session_id or len(session_id) > 128
        or any(ord(char) < 32 for char in session_id)
    ):
        return None
    generation = value.get("source_generation_id")
    if generation is not None and (not isinstance(generation, str) or not re.fullmatch(r"gen-[0-9a-f]{32}", generation)):
        return None
    schema_sha256 = value.get("schema_sha256")
    if schema_sha256 is not None and (
        not isinstance(schema_sha256, str) or not _TOKEN_RE.fullmatch(schema_sha256)
    ):
        return None
    routes = value.get("actual_routes")
    if not isinstance(routes, list) or len(routes) > 32:
        return None
    route_pairs = set()
    for route in routes:
        if not isinstance(route, dict) or set(route) != {"provider", "model"}:
            return None
        provider, model = route["provider"], route["model"]
        if (not isinstance(provider, str) or not _PROVIDER_ID_RE.fullmatch(provider)
                or "@" in provider or not isinstance(model, str) or not _MODEL_ID_RE.fullmatch(model)
                or "://" in model or "@" in model):
            return None
        route_pairs.add((provider, model))
    allowed_unknown = {"worker_session_id", "source_generation_id", "schema_sha256", "actual_routes"}
    unknown = value.get("unknown_fields")
    if (not isinstance(unknown, list) or any(
        not isinstance(item, str) or item not in allowed_unknown for item in unknown
    )):
        return None
    unknown_set = set(unknown)
    for key, item in (
        ("worker_session_id", session_id),
        ("source_generation_id", generation),
        ("schema_sha256", schema_sha256),
    ):
        if item is None:
            unknown_set.add(key)
        else:
            unknown_set.discard(key)
    if route_pairs:
        unknown_set.discard("actual_routes")
    else:
        unknown_set.add("actual_routes")
    return {
        "version": 1,
        "worker_session_id": session_id,
        "source_generation_id": generation,
        "schema_sha256": schema_sha256,
        "actual_routes": [
            {"provider": provider, "model": model}
            for provider, model in sorted(route_pairs)
        ],
        "unknown_fields": sorted(unknown_set),
    }


def carry_run_provenance(conn: sqlite3.Connection, task_id: str, metadata: dict) -> dict:
    """Keep the worker's bounded route projection when the reaper closes its run.

    ``_end_run`` replaces the closed run's metadata wholesale, so the reaper
    passes its replacement metadata through here to retain a valid projection.
    """
    row = conn.execute(
        "SELECT r.metadata FROM tasks t JOIN task_runs r ON r.id = t.current_run_id "
        "WHERE t.id = ?", (task_id,),
    ).fetchone()
    if row is None or not row["metadata"]:
        return metadata
    try:
        prior = json.loads(row["metadata"])
    except (TypeError, ValueError):
        return metadata
    if not isinstance(prior, dict):
        return metadata
    provenance = _validated_run_provenance(prior.get("retry_provenance"))
    return {**metadata, "retry_provenance": provenance} if provenance is not None else metadata


def record_retry_run_provenance(
    conn: sqlite3.Connection,
    task_id: str,
    *,
    expected_run_id: int,
    provenance: dict,
) -> bool:
    """Persist a bounded runtime-owned route projection for the matching active run only."""
    safe = _validated_run_provenance(provenance)
    if safe is None:
        return False
    run_id = _positive_int(expected_run_id)
    if run_id is None:
        return False
    with _kb.write_txn(conn):
        task = conn.execute(
            "SELECT status, current_run_id FROM tasks WHERE id = ?", (task_id,),
        ).fetchone()
        if task is None or task["status"] != "running" or task["current_run_id"] != run_id:
            return False
        run = conn.execute(
            "SELECT metadata FROM task_runs WHERE id = ? AND task_id = ? AND ended_at IS NULL",
            (run_id, task_id),
        ).fetchone()
        if run is None:
            return False
        raw_metadata = run["metadata"]
        if raw_metadata:
            try:
                metadata = json.loads(raw_metadata)
            except (TypeError, ValueError):
                return False
            if not isinstance(metadata, dict):
                return False
        else:
            metadata = {}
        existing = metadata.get("retry_provenance")
        if existing is not None:
            prior = _validated_run_provenance(existing)
            if prior is None:
                return False
            for key in ("source_generation_id", "schema_sha256"):
                if prior[key] is not None and safe[key] is not None and prior[key] != safe[key]:
                    return False
            merged_routes = {
                (route["provider"], route["model"])
                for route in (*prior["actual_routes"], *safe["actual_routes"])
            }
            safe["actual_routes"] = [
                {"provider": provider, "model": model}
                for provider, model in sorted(merged_routes)
            ]
            safe["unknown_fields"] = sorted(
                set(prior["unknown_fields"]).union(safe["unknown_fields"])
                - ({"actual_routes"} if safe["actual_routes"] else set())
            )
            for key in ("worker_session_id", "source_generation_id", "schema_sha256"):
                if safe[key] is None:
                    safe[key] = prior[key]
            safe = _validated_run_provenance(safe)
            if safe is None:
                return False
        metadata["retry_provenance"] = safe
        serialized = _canonical_json(metadata)
        if len(serialized.encode("utf-8")) > 8192:
            return False
        updated = conn.execute(
            "UPDATE task_runs SET metadata = ? WHERE id = ? AND task_id = ? AND ended_at IS NULL",
            (serialized, run_id, task_id),
        )
        return updated.rowcount == 1


# Called by the dispatcher only after the existing protocol-violation streak reaches
# its configured limit. Kept below the public service APIs to make the policy edge
# explicit and independently testable.
def create_protocol_retry_hold(
    conn: sqlite3.Connection,
    task_id: str,
    *,
    expected_failed_run_id: int,
    expected_failure_event_id: int,
    protocol_violations: int,
    protocol_violation_limit: int,
    allow_nested: bool = False,
) -> bool:
    """Internal dispatcher entry point for a newly exhausted protocol streak."""
    failed_run_id = _positive_int(expected_failed_run_id)
    failure_event_id = _positive_int(expected_failure_event_id)
    violations = _positive_int(protocol_violations)
    limit = _positive_int(protocol_violation_limit)
    if any(value is None for value in (failed_run_id, failure_event_id, violations, limit)):
        return False
    return _record_protocol_retry_hold(
        conn, task_id,
        expected_failed_run_id=failed_run_id,
        expected_failure_event_id=failure_event_id,
        protocol_violations=violations,
        protocol_violation_limit=limit,
        allow_nested=allow_nested,
    )


# Late binding keeps the facade/sibling import cycle in the established direction.
from hermes_cli import kanban_db as _kb
