"""Task/run-scoped approval grants for dispatcher-owned Kanban workers.

Kanban workers are single-query processes with no human at their stdin.  A
normal ``approvals.single_query_mode: deny`` therefore blocks recoverable tool
warnings deterministically.  This module provides a narrower alternative to a
profile-wide ``approve`` or ``--yolo``: an operator can attach a short-lived,
operation-allowlisted grant to one task.  Each use is validated against the
authoritative task + run rows and recorded as an event.

The environment contains only the expected grant id.  It is not authority: a
worker cannot gain approval by fabricating environment variables, stale run
ids, or task text because every consumption re-reads the board database and
checks task, run, claim, actor, action class, operation list, and expiry.

This follows Kanban's documented trusted-local-user threat model.  It is a
workflow authorization boundary, not an OS sandbox: processes sharing the same
OS identity can ultimately edit the same SQLite database.  Deployments that
need hostile-code isolation must also use separate identities/sandboxes.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import time
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Freeze the dispatcher-owned worker marker at import so code in the same
# process cannot gain operator powers by mutating os.environ before calling the
# DB API. The live check also covers modules imported before test/runtime
# context is installed.
_KANBAN_WORKER_CONTEXT_FROZEN = bool(os.environ.get("HERMES_KANBAN_TASK"))

MAX_APPROVAL_TTL_SECONDS = 1_800
MAX_APPROVAL_OPERATIONS = 20
GRANTABLE_TASK_STATUSES = frozenset({"triage", "todo", "ready", "blocked", "review"})

_ID_RE = re.compile(r"^(?:apr|chg)-[a-z0-9-]{6,64}$")
VALID_ACTION_CLASSES = frozenset(
    {"read", "write", "command", "deploy", "admin", "recovery", "provider", "trust"}
)
_REQUIRED_FIELDS = frozenset(
    {
        "version",
        "approval_id",
        "change_id",
        "approver",
        "actor",
        "target",
        "segment_id",
        "action_class",
        "allowed_operations",
        "valid_from",
        "expires_at",
        "rollback_ref",
    }
)
_PUBLIC_ALLOWED_FIELDS = _REQUIRED_FIELDS | {"task_id", "scope_digest"}
_RUNTIME_BINDING_FIELDS = frozenset({"bound_run_id", "bound_claim_lock"})

_TASK_SCOPE_COLUMNS = (
    "id, title, body, assignee, tenant, workspace_kind, workspace_path, branch_name, "
    "project_id, skills, model_override, provider_override, reasoning_effort, "
    "max_retries, goal_mode, goal_max_turns, workflow_template_id, current_step_key, "
    "max_runtime_seconds"
)
_TASK_SCOPE_QUERY = "SELECT " + _TASK_SCOPE_COLUMNS + " FROM tasks WHERE id = ?"


def _epoch(value: int | None) -> int:
    return int(time.time()) if value is None else int(value)


def _assert_operator_context() -> None:
    """Keep grant mutation outside dispatcher/delegation child processes."""
    if _KANBAN_WORKER_CONTEXT_FROZEN or os.environ.get("HERMES_KANBAN_TASK"):
        raise PermissionError("Kanban worker contexts cannot manage approval grants")
    try:
        from agent.delegation_context import is_delegated_child_process_context

        delegated = is_delegated_child_process_context()
    except Exception:
        delegated = bool(os.environ.get("HERMES_DELEGATED_CHILD_CONTEXT"))
    if delegated:
        raise PermissionError("delegated worker contexts cannot manage approval grants")


def _nonempty_string(grant: Mapping[str, Any], field: str, *, maximum: int = 200) -> str:
    value = grant.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"approval grant {field} must be a non-empty string")
    value = value.strip()
    if len(value) > maximum:
        raise ValueError(f"approval grant {field} exceeds {maximum} characters")
    return value


def _integer(grant: Mapping[str, Any], field: str) -> int:
    value = grant.get(field)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"approval grant {field} must be an integer epoch")
    return value


def _normalize_operation(value: Any) -> str:
    """Accept exact Hermes pattern/rule keys, never control-bearing text."""
    if not isinstance(value, str):
        raise ValueError("approval grant allowed_operations entries must be strings")
    operation = value.strip()
    if not 3 <= len(operation) <= 128:
        raise ValueError(f"invalid approval operation {operation!r}")
    if any(not char.isprintable() or char in "\r\n" for char in operation):
        raise ValueError(f"invalid approval operation {operation!r}")
    return operation


def normalize_grant(
    grant: Mapping[str, Any],
    *,
    task_id: str,
    assignee: str,
    scope_digest: str | None = None,
    allow_runtime_binding: bool = False,
    now: int | None = None,
) -> dict[str, Any]:
    """Validate and bind an operator-supplied grant to one task/assignee.

    Unknown fields are rejected so a misspelled scope field cannot look
    effective while being ignored.  ``valid_from <= now < expires_at`` and a
    maximum 30-minute identity TTL are required at issuance.
    """
    if not isinstance(grant, Mapping):
        raise ValueError("approval grant must be an object")

    keys = set(grant)
    missing = sorted(_REQUIRED_FIELDS - keys)
    allowed_fields = _PUBLIC_ALLOWED_FIELDS
    if allow_runtime_binding:
        allowed_fields = allowed_fields | _RUNTIME_BINDING_FIELDS
    unknown = sorted(keys - allowed_fields)
    if missing:
        raise ValueError(f"approval grant missing fields: {', '.join(missing)}")
    if unknown:
        raise ValueError(f"approval grant has unknown fields: {', '.join(unknown)}")
    if grant.get("version") != 1:
        raise ValueError("approval grant version must be 1")

    approval_id = _nonempty_string(grant, "approval_id", maximum=68)
    change_id = _nonempty_string(grant, "change_id", maximum=68)
    if not _ID_RE.fullmatch(approval_id) or not approval_id.startswith("apr-"):
        raise ValueError("approval grant approval_id must match apr-[a-z0-9-]{6,64}")
    if not _ID_RE.fullmatch(change_id) or not change_id.startswith("chg-"):
        raise ValueError("approval grant change_id must match chg-[a-z0-9-]{6,64}")

    approver = _nonempty_string(grant, "approver", maximum=80)
    actor = _nonempty_string(grant, "actor", maximum=80)
    target = _nonempty_string(grant, "target", maximum=120)
    segment_id = _nonempty_string(grant, "segment_id", maximum=80)
    action_class = _nonempty_string(grant, "action_class", maximum=24).lower()
    rollback_ref = _nonempty_string(grant, "rollback_ref", maximum=200)

    if not assignee:
        raise ValueError("approval grant requires an assigned task")
    if actor != assignee:
        raise ValueError(
            f"approval grant actor {actor!r} does not match task assignee {assignee!r}"
        )
    existing_task_id = grant.get("task_id")
    if existing_task_id is not None and existing_task_id != task_id:
        raise ValueError("approval grant task_id does not match target task")
    if action_class not in VALID_ACTION_CLASSES:
        raise ValueError(
            "approval grant action_class must be one of "
            + ", ".join(sorted(VALID_ACTION_CLASSES))
        )

    supplied_scope_digest = grant.get("scope_digest")
    if supplied_scope_digest is not None and (
        not isinstance(supplied_scope_digest, str)
        or not re.fullmatch(r"sha256:[0-9a-f]{64}", supplied_scope_digest)
    ):
        raise ValueError("approval grant scope_digest must be sha256:<64 lowercase hex>")
    if scope_digest is not None and supplied_scope_digest not in {None, scope_digest}:
        raise ValueError("approval grant scope_digest does not match current task scope")
    bound_scope_digest = scope_digest or supplied_scope_digest
    if bound_scope_digest is None:
        raise ValueError("approval grant requires a bound scope_digest")

    raw_operations = grant.get("allowed_operations")
    if not isinstance(raw_operations, list) or not raw_operations:
        raise ValueError("approval grant allowed_operations must be a non-empty list")
    if len(raw_operations) > MAX_APPROVAL_OPERATIONS:
        raise ValueError(
            f"approval grant allowed_operations exceeds {MAX_APPROVAL_OPERATIONS} entries"
        )
    operations: list[str] = []
    for raw in raw_operations:
        operations.append(_normalize_operation(raw))
    if len(set(operations)) != len(operations):
        raise ValueError("approval grant allowed_operations must be unique")

    valid_from = _integer(grant, "valid_from")
    expires_at = _integer(grant, "expires_at")
    current = _epoch(now)
    if valid_from > current:
        raise ValueError("approval grant is not active yet")
    if expires_at <= current:
        raise ValueError("approval grant is already expired")
    ttl = expires_at - valid_from
    if ttl <= 0 or ttl > MAX_APPROVAL_TTL_SECONDS:
        raise ValueError(
            f"approval grant TTL must be between 1 and {MAX_APPROVAL_TTL_SECONDS} seconds"
        )

    bound_run_id = grant.get("bound_run_id")
    bound_claim_lock = grant.get("bound_claim_lock")
    if (bound_run_id is None) != (bound_claim_lock is None):
        raise ValueError("approval grant runtime binding must include run and claim")
    runtime_binding: dict[str, Any] = {}
    if bound_run_id is not None:
        if isinstance(bound_run_id, bool) or not isinstance(bound_run_id, int):
            raise ValueError("approval grant bound_run_id must be a positive integer")
        if bound_run_id <= 0:
            raise ValueError("approval grant bound_run_id must be a positive integer")
        if not isinstance(bound_claim_lock, str) or not bound_claim_lock.strip():
            raise ValueError("approval grant bound_claim_lock must be a non-empty string")
        if len(bound_claim_lock) > 200:
            raise ValueError("approval grant bound_claim_lock exceeds 200 characters")
        runtime_binding = {
            "bound_run_id": bound_run_id,
            "bound_claim_lock": bound_claim_lock.strip(),
        }

    normalized = {
        "version": 1,
        "approval_id": approval_id,
        "change_id": change_id,
        "approver": approver,
        "actor": actor,
        "target": target,
        "segment_id": segment_id,
        "action_class": action_class,
        "allowed_operations": operations,
        "valid_from": valid_from,
        "expires_at": expires_at,
        "rollback_ref": rollback_ref,
        "task_id": task_id,
        "scope_digest": bound_scope_digest,
    }
    normalized.update(runtime_binding)
    return normalized


def _sha256_file(path_value: str) -> str:
    """Hash an attachment without trusting mutable DB size metadata."""
    try:
        digest = hashlib.sha256()
        with Path(path_value).open("rb") as handle:
            while chunk := handle.read(1024 * 1024):
                digest.update(chunk)
        return f"sha256:{digest.hexdigest()}"
    except (OSError, ValueError):
        return "unreadable"


def task_scope_digest(conn, task_id: str) -> str:
    """Hash every durable input that becomes worker instruction/context.

    Runtime status, the active claim/run, task events, and the approval row itself
    are excluded because dispatcher bookkeeping changes them after issuance.
    Task fields, comments, attachment bytes, prior attempts, parent handoffs, and
    role history are included so a post-grant context edit invalidates the
    capability.
    """
    from hermes_cli import kanban_db as kb

    task = conn.execute(
        _TASK_SCOPE_QUERY,
        (task_id,),
    ).fetchone()
    if task is None:
        raise ValueError(f"unknown task {task_id}")
    task_payload = {key: task[key] for key in task.keys()}
    comments = [
        {key: row[key] for key in row.keys()}
        for row in conn.execute(
            "SELECT id, author, body, created_at FROM task_comments "
            "WHERE task_id = ? ORDER BY id",
            (task_id,),
        ).fetchall()
    ]
    parents = [
        {key: row[key] for key in row.keys()}
        for row in conn.execute(
            "SELECT p.id, p.title, p.status, p.result "
            "FROM task_links l JOIN tasks p ON p.id = l.parent_id "
            "WHERE l.child_id = ? ORDER BY p.id",
            (task_id,),
        ).fetchall()
    ]
    attachments = []
    for row in conn.execute(
        "SELECT id, filename, stored_path, content_type, size, uploaded_by, created_at "
        "FROM task_attachments WHERE task_id = ? ORDER BY id",
        (task_id,),
    ).fetchall():
        attachment = {key: row[key] for key in row.keys()}
        attachment["content_digest"] = _sha256_file(str(row["stored_path"]))
        attachments.append(attachment)
    payload = {
        "task": task_payload,
        "comments": comments,
        "parents": parents,
        "attachments": attachments,
        "worker_context_history": kb.worker_context_history_snapshot(conn, task_id),
    }
    canonical = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return f"sha256:{hashlib.sha256(canonical).hexdigest()}"


def grant_task_approval(
    conn,
    task_id: str,
    grant: Mapping[str, Any],
    *,
    now: int | None = None,
) -> dict[str, Any]:
    """Attach a validated grant to an idle task and emit ``approval_granted``."""
    from hermes_cli import kanban_db as kb

    _assert_operator_context()
    with kb.write_txn(conn):
        row = conn.execute(
            "SELECT id, assignee, status, approval_grant FROM tasks WHERE id = ?",
            (task_id,),
        ).fetchone()
        if row is None:
            raise ValueError(f"unknown task {task_id}")
        if row["status"] not in GRANTABLE_TASK_STATUSES:
            raise ValueError(
                f"cannot grant approval while task status is {row['status']!r}"
            )
        digest = task_scope_digest(conn, task_id)
        bound = normalize_grant(
            grant,
            task_id=task_id,
            assignee=row["assignee"] or "",
            scope_digest=digest,
            now=now,
        )
        if row["approval_grant"]:
            raise ValueError(
                "task already has an approval grant; revoke it before replacing"
            )
        payload = {
            "approval_id": bound["approval_id"],
            "change_id": bound["change_id"],
            "actor": bound["actor"],
            "action_class": bound["action_class"],
            "allowed_operations": list(bound["allowed_operations"]),
            "expires_at": bound["expires_at"],
            "scope_digest": bound["scope_digest"],
        }
        encoded = json.dumps(bound, sort_keys=True, separators=(",", ":"))
        updated = conn.execute(
            "UPDATE tasks SET approval_grant = ? "
            "WHERE id = ? AND status != 'running' AND approval_grant IS NULL",
            (encoded, task_id),
        )
        if updated.rowcount != 1:
            raise ValueError("task became active before approval could be granted")
        kb._append_event(conn, task_id, "approval_granted", payload)
    return bound


def bind_task_approval_to_run(
    conn,
    task_id: str,
    *,
    run_id: int,
    claim_lock: str,
    now: int | None = None,
) -> str | None:
    """Bind an unbound grant to the first claimed run inside its claim txn."""
    from hermes_cli import kanban_db as kb

    if isinstance(run_id, bool) or not isinstance(run_id, int) or run_id <= 0:
        return None
    if not isinstance(claim_lock, str) or not claim_lock:
        return None
    row = conn.execute(
        "SELECT assignee, status, current_run_id, claim_lock, approval_grant "
        "FROM tasks WHERE id = ?",
        (task_id,),
    ).fetchone()
    if (
        row is None
        or row["status"] != "running"
        or int(row["current_run_id"] or 0) != run_id
        or row["claim_lock"] != claim_lock
        or not row["approval_grant"]
    ):
        return None
    encoded_before = row["approval_grant"]
    try:
        grant = normalize_grant(
            json.loads(encoded_before),
            task_id=task_id,
            assignee=row["assignee"] or "",
            scope_digest=task_scope_digest(conn, task_id),
            allow_runtime_binding=True,
            now=now,
        )
    except (TypeError, ValueError, json.JSONDecodeError):
        return None

    existing_run = grant.get("bound_run_id")
    existing_claim = grant.get("bound_claim_lock")
    if existing_run is not None or existing_claim is not None:
        if existing_run == run_id and existing_claim == claim_lock:
            return str(grant["approval_id"])
        return None

    grant["bound_run_id"] = run_id
    grant["bound_claim_lock"] = claim_lock
    encoded_after = json.dumps(grant, sort_keys=True, separators=(",", ":"))
    updated = conn.execute(
        "UPDATE tasks SET approval_grant = ? "
        "WHERE id = ? AND status = 'running' AND current_run_id = ? "
        "AND claim_lock = ? AND approval_grant = ?",
        (encoded_after, task_id, run_id, claim_lock, encoded_before),
    )
    if updated.rowcount != 1:
        return None
    kb._append_event(
        conn,
        task_id,
        "approval_bound",
        {
            "approval_id": grant["approval_id"],
            "change_id": grant["change_id"],
            "run_id": run_id,
        },
        run_id=run_id,
    )
    return str(grant["approval_id"])


def revoke_task_approval(
    conn,
    task_id: str,
    *,
    approval_id: str,
    revoked_by: str,
    reason: str,
    now: int | None = None,
) -> bool:
    """Revoke one matching grant. Active workers observe it on their next call."""
    from hermes_cli import kanban_db as kb

    _assert_operator_context()
    row = conn.execute(
        "SELECT approval_grant, current_run_id FROM tasks WHERE id = ?", (task_id,)
    ).fetchone()
    if row is None or not row["approval_grant"]:
        return False
    try:
        current = json.loads(row["approval_grant"])
    except (TypeError, ValueError):
        return False
    if current.get("approval_id") != approval_id:
        return False

    actor = str(revoked_by or "").strip()
    why = str(reason or "").strip()
    if not actor or not why:
        raise ValueError("revoked_by and reason are required")
    payload = {
        "approval_id": approval_id,
        "change_id": current.get("change_id"),
        "revoked_by": actor,
        "reason": why,
        "revoked_at": _epoch(now),
    }
    with kb.write_txn(conn):
        updated = conn.execute(
            "UPDATE tasks SET approval_grant = NULL "
            "WHERE id = ? AND approval_grant = ?",
            (task_id, row["approval_grant"]),
        )
        if updated.rowcount != 1:
            return False
        kb._append_event(
            conn,
            task_id,
            "approval_revoked",
            payload,
            run_id=(int(row["current_run_id"]) if row["current_run_id"] else None),
        )
    return True


def _parse_operation_keys(operation_keys: Iterable[str]) -> list[str] | None:
    if isinstance(operation_keys, (str, bytes)):
        return None
    operations: list[str] = []
    for raw in operation_keys:
        try:
            operation = _normalize_operation(raw)
        except ValueError:
            return None
        if operation not in operations:
            operations.append(operation)
    return operations or None


def _worker_binding_from_env() -> dict[str, Any] | None:
    task_id = os.environ.get("HERMES_KANBAN_TASK", "").strip()
    run_raw = os.environ.get("HERMES_KANBAN_RUN_ID", "").strip()
    claim_lock = os.environ.get("HERMES_KANBAN_CLAIM_LOCK", "").strip()
    actor = os.environ.get("HERMES_PROFILE", "").strip()
    approval_id = os.environ.get("HERMES_KANBAN_APPROVAL_ID", "").strip()
    db_path = os.environ.get("HERMES_KANBAN_DB", "").strip()
    if not all((task_id, run_raw, claim_lock, actor, approval_id, db_path)):
        return None
    try:
        run_id = int(run_raw)
    except ValueError:
        return None
    if run_id <= 0:
        return None
    return {
        "task_id": task_id,
        "run_id": run_id,
        "claim_lock": claim_lock,
        "actor": actor,
        "approval_id": approval_id,
        "db_path": db_path,
    }


def active_grant_id_for_task(task: Any, *, now: int | None = None) -> str | None:
    """Return a spawn-safe grant id, or ``None`` for missing/invalid/expired grants."""
    grant = getattr(task, "approval_grant", None)
    if not isinstance(grant, Mapping):
        return None
    try:
        normalized = normalize_grant(
            grant,
            task_id=str(task.id),
            assignee=str(task.assignee or ""),
            allow_runtime_binding=True,
            now=now,
        )
    except (TypeError, ValueError):
        return None
    if getattr(task, "status", None) != "running":
        return None
    if not getattr(task, "current_run_id", None) or not getattr(task, "claim_lock", None):
        return None
    if normalized.get("bound_run_id") != task.current_run_id:
        return None
    if normalized.get("bound_claim_lock") != task.claim_lock:
        return None
    return normalized["approval_id"]


def consume_task_approval(
    operation_keys: Iterable[str],
    *,
    action_class: str,
    now: int | None = None,
) -> dict[str, Any] | None:
    """Atomically validate and receipt an exact operation subset.

    Any missing/malformed/stale/mismatched state returns ``None``. Authorization
    is granted only when the audit event is written in the same transaction.
    """
    from hermes_cli import kanban_db as kb

    binding = _worker_binding_from_env()
    operations = _parse_operation_keys(operation_keys)
    requested_class = str(action_class or "").strip().lower()
    if binding is None or operations is None or requested_class not in VALID_ACTION_CLASSES:
        return None

    try:
        conn = kb.connect(Path(binding["db_path"]))
    except Exception:
        logger.warning("Kanban approval validation could not open the pinned board")
        return None

    try:
        with kb.write_txn(conn):
            row = conn.execute(
                "SELECT id, assignee, status, claim_lock, current_run_id, approval_grant "
                "FROM tasks WHERE id = ?",
                (binding["task_id"],),
            ).fetchone()
            if row is None or row["status"] != "running" or not row["approval_grant"]:
                return None
            if (
                row["assignee"] != binding["actor"]
                or row["claim_lock"] != binding["claim_lock"]
                or int(row["current_run_id"] or 0) != binding["run_id"]
            ):
                return None
            run = conn.execute(
                "SELECT profile, status, claim_lock FROM task_runs WHERE id = ? AND task_id = ?",
                (binding["run_id"], binding["task_id"]),
            ).fetchone()
            if (
                run is None
                or run["status"] != "running"
                or run["profile"] != binding["actor"]
                or run["claim_lock"] != binding["claim_lock"]
            ):
                return None
            try:
                raw_grant = json.loads(row["approval_grant"])
                grant = normalize_grant(
                    raw_grant,
                    task_id=binding["task_id"],
                    assignee=binding["actor"],
                    allow_runtime_binding=True,
                    now=now,
                )
            except (TypeError, ValueError, json.JSONDecodeError):
                return None
            if grant["approval_id"] != binding["approval_id"]:
                return None
            if grant.get("bound_run_id") != binding["run_id"]:
                return None
            if grant.get("bound_claim_lock") != binding["claim_lock"]:
                return None
            if grant["action_class"] != requested_class:
                return None
            if not set(operations).issubset(set(grant["allowed_operations"])):
                return None
            if grant["scope_digest"] != task_scope_digest(conn, binding["task_id"]):
                return None

            receipt = {
                "approval_id": grant["approval_id"],
                "change_id": grant["change_id"],
                "task_id": binding["task_id"],
                "run_id": binding["run_id"],
                "operations": operations,
            }
            kb._append_event(
                conn,
                binding["task_id"],
                "approval_consumed",
                receipt,
                run_id=binding["run_id"],
            )
        return receipt
    except Exception:
        logger.warning("Kanban approval validation/receipt failed closed", exc_info=True)
        return None
    finally:
        conn.close()
