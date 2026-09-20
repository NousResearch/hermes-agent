"""Persisted human prerequisites, delivery receipts, and restart-safe readiness supervision."""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
import sqlite3
import time
from typing import Any, Mapping, Optional

_FIELDS = (
    "incomplete_status", "reason", "execution_location", "action",
    "expected_success", "automatic_continuation",
)


@dataclass(frozen=True)
class UserActionState:
    task_id: str
    payload: dict[str, str]
    readiness_probe: dict[str, Any]
    fingerprint: str
    created_at: int
    resolved_at: Optional[int]


def _clean(value: Any) -> str:
    return " ".join(str(value or "").split())


def fallback_payload(task_id: str, reason: str) -> dict[str, str]:
    return {
        "incomplete_status": "The task needs user action before it can continue.",
        "reason": _clean(reason) or "The worker needs information from the task owner.",
        "execution_location": f"Reply on or unblock Kanban task {task_id} in its originating chat or dashboard.",
        "action": f"Reply to Kanban task {task_id} with the requested information, then unblock it.",
        "expected_success": f"Kanban task {task_id} becomes ready and a worker is dispatched.",
        "automatic_continuation": (
            f"No continue response is needed; the persisted task_unblocked trigger for {task_id} "
            "automatically resumes and dispatches the task."
        ),
    }


def normalize_payload(task_id: str, reason: str, payload: Optional[Mapping[str, Any]]) -> dict[str, str]:
    source = payload if isinstance(payload, Mapping) else fallback_payload(task_id, reason)
    normalized = {name: _clean(source.get(name)) for name in _FIELDS}
    missing = [name for name, value in normalized.items() if not value]
    if missing:
        raise ValueError("user_action missing required field(s): " + ", ".join(missing))
    return normalized


def normalize_probe(task_id: str, probe: Optional[Mapping[str, Any]]) -> dict[str, Any]:
    if not isinstance(probe, Mapping):
        return {"kind": "task_unblocked", "task_id": task_id}
    kind = _clean(probe.get("kind"))
    allowed = {"env_present", "path_exists", "task_unblocked", "task_status"}
    if kind not in allowed:
        raise ValueError(f"readiness_probe.kind must be one of {sorted(allowed)}")
    clean = {str(k): v for k, v in probe.items() if v is not None}
    clean["kind"] = kind
    required = {
        "env_present": "name", "path_exists": "path", "task_unblocked": "task_id",
        "task_status": "task_id",
    }[kind]
    if not _clean(clean.get(required)):
        clean[required] = task_id if required == "task_id" else ""
    if not _clean(clean.get(required)):
        raise ValueError(f"readiness_probe.{required} is required")
    return clean


def material_fingerprint(payload: Mapping[str, str], probe: Mapping[str, Any]) -> str:
    material = {"payload": {name: _clean(payload[name]) for name in _FIELDS}, "prerequisite": probe}
    encoded = json.dumps(material, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def persist_user_action(
    conn: sqlite3.Connection, task_id: str, reason: str, payload: Optional[Mapping[str, Any]],
    readiness_probe: Optional[Mapping[str, Any]], *, now: Optional[int] = None,
) -> UserActionState:
    normalized = normalize_payload(task_id, reason, payload)
    probe = normalize_probe(task_id, readiness_probe)
    fingerprint = material_fingerprint(normalized, probe)
    when = int(time.time()) if now is None else int(now)
    conn.execute(
        "INSERT INTO kanban_user_actions "
        "(task_id, payload, readiness_probe, material_fingerprint, created_at, resolved_at) "
        "VALUES (?, ?, ?, ?, ?, NULL) "
        "ON CONFLICT(task_id) DO UPDATE SET payload=excluded.payload, "
        "readiness_probe=excluded.readiness_probe, material_fingerprint=excluded.material_fingerprint, "
        "created_at=excluded.created_at, resolved_at=NULL",
        (task_id, json.dumps(normalized, sort_keys=True), json.dumps(probe, sort_keys=True), fingerprint, when),
    )
    return UserActionState(task_id, normalized, probe, fingerprint, when, None)


def get_user_action(conn: sqlite3.Connection, task_id: str) -> Optional[UserActionState]:
    row = conn.execute("SELECT * FROM kanban_user_actions WHERE task_id = ?", (task_id,)).fetchone()
    if row is None:
        return None
    return UserActionState(
        task_id=row["task_id"], payload=json.loads(row["payload"]),
        readiness_probe=json.loads(row["readiness_probe"]), fingerprint=row["material_fingerprint"],
        created_at=int(row["created_at"]), resolved_at=row["resolved_at"],
    )


def _destination_key(destination: str) -> str:
    return hashlib.sha256(str(destination).encode("utf-8")).hexdigest()


def claim_delivery(
    conn: sqlite3.Connection, task_id: str, provider: str, destination: str, fingerprint: str,
) -> Optional[dict[str, Any]]:
    """Claim one material delivery. Acknowledged rows dedup; failed/in-flight rows retry."""
    from hermes_cli import kanban_db as kb
    now = int(time.time())
    destination_key = _destination_key(destination)
    with kb.write_txn(conn):
        row = conn.execute(
            "SELECT * FROM kanban_user_action_deliveries WHERE task_id=? AND destination_key=? "
            "AND material_fingerprint=?",
            (task_id, destination_key, fingerprint),
        ).fetchone()
        if row is not None and row["result"] == "acknowledged":
            return None
        if row is None:
            cur = conn.execute(
                "INSERT INTO kanban_user_action_deliveries "
                "(task_id, provider, destination_key, material_fingerprint, attempts, result, updated_at) "
                "VALUES (?, ?, ?, ?, 1, 'pending', ?)",
                (task_id, provider, destination_key, fingerprint, now),
            )
            assert cur.lastrowid is not None
            delivery_id, attempts = int(cur.lastrowid), 1
        else:
            delivery_id, attempts = int(row["id"]), int(row["attempts"]) + 1
            conn.execute(
                "UPDATE kanban_user_action_deliveries SET attempts=?, result='pending', updated_at=? WHERE id=?",
                (attempts, now, delivery_id),
            )
    return {"id": delivery_id, "attempts": attempts, "destination_key": destination_key}


def record_delivery_result(
    conn: sqlite3.Connection, delivery_id: int, *, acknowledged: bool, provider: str,
    message_id: Optional[str] = None, error: Optional[str] = None,
) -> None:
    from agent.redact import redact_sensitive_text
    from hermes_cli import kanban_db as kb
    now = int(time.time())
    safe_error = redact_sensitive_text(str(error or ""), force=True)[:500] or None
    with kb.write_txn(conn):
        conn.execute(
            "UPDATE kanban_user_action_deliveries SET provider=?, provider_message_id=?, "
            "delivered_at=?, result=?, error_metadata=?, updated_at=? WHERE id=?",
            (
                provider, str(message_id) if acknowledged and message_id is not None else None,
                now if acknowledged else None, "acknowledged" if acknowledged else "delivery_failed",
                json.dumps({"error": safe_error}) if safe_error else None, now, int(delivery_id),
            ),
        )


def readiness_satisfied(conn: sqlite3.Connection, state: UserActionState) -> bool:
    probe = state.readiness_probe
    kind = probe["kind"]
    if kind == "env_present":
        return bool(os.environ.get(str(probe["name"])))
    if kind == "path_exists":
        return Path(str(probe["path"])).exists()
    if kind == "task_status":
        expected = str(probe.get("status") or "done")
        row = conn.execute("SELECT status FROM tasks WHERE id=?", (str(probe["task_id"]),)).fetchone()
        return bool(row and row["status"] == expected)
    if kind == "task_unblocked":
        row = conn.execute("SELECT status FROM tasks WHERE id=?", (str(probe["task_id"]),)).fetchone()
        return bool(row and row["status"] not in {"needs_user_action", "blocked"})
    return False


def supervise_user_actions(conn: sqlite3.Connection) -> list[str]:
    """Resolve satisfied persisted probes. Safe to call repeatedly and after DB reopen."""
    from hermes_cli import kanban_db as kb
    resolved: list[str] = []
    rows = conn.execute(
        "SELECT task_id FROM kanban_user_actions WHERE resolved_at IS NULL ORDER BY created_at, task_id"
    ).fetchall()
    for row in rows:
        state = get_user_action(conn, row["task_id"])
        if state is None or not readiness_satisfied(conn, state):
            continue
        now = int(time.time())
        with kb.write_txn(conn):
            task = conn.execute("SELECT status FROM tasks WHERE id=?", (state.task_id,)).fetchone()
            if task is None or task["status"] != "needs_user_action":
                conn.execute(
                    "UPDATE kanban_user_actions SET resolved_at=COALESCE(resolved_at, ?) WHERE task_id=?",
                    (now, state.task_id),
                )
                continue
            landing = "ready" if kb._parents_satisfied(conn, state.task_id) else "todo"
            conn.execute(
                "UPDATE tasks SET status=?, current_run_id=NULL, claim_lock=NULL, claim_expires=NULL, "
                "worker_pid=NULL, worker_started_at=NULL WHERE id=? AND status='needs_user_action'",
                (landing, state.task_id),
            )
            conn.execute("UPDATE kanban_user_actions SET resolved_at=? WHERE task_id=?", (now, state.task_id))
            kb._append_event(conn, state.task_id, "user_action_ready", {"status": landing})
            resolved.append(state.task_id)
    return resolved


def classify_capability_failure(error: str) -> dict[str, str]:
    text = _clean(error).lower()
    if "nonewprivileges" in text or "no new privileges" in text or ("root" in text and "sudo" in text):
        cls = "privilege"
    elif "credential" in text or "api key" in text or "token" in text or "auth" in text:
        cls = "credential"
    elif "approval" in text or "consent" in text:
        cls = "approval"
    elif "physical" in text or "press the" in text or "plug in" in text:
        cls = "physical_action"
    else:
        cls = "provider_or_tool"
    return {"class": cls, "reason": _clean(error)}
