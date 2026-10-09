"""Exact packet approval gates for irreversible controlled execution."""
from __future__ import annotations

import json
import re
import sqlite3
import time
from typing import Any

_SHA256 = re.compile(r"[a-f0-9]{64}")
_ACTOR_FIELDS = ("user_id", "email", "display_name", "provider")


class PacketApprovalRequired(ValueError):
    """A generic lifecycle action tried to satisfy a packet approval gate."""


class PacketApprovalMismatch(ValueError):
    """The current packet or execution identity differs from the approved one."""


def _digest(value: str) -> str:
    value = str(value or "").strip().lower()
    if not _SHA256.fullmatch(value):
        raise ValueError("packet_sha256 must be exactly 64 lowercase hexadecimal characters")
    return value


def _canonical_identity(value: dict[str, Any]) -> str:
    if not isinstance(value, dict) or not value:
        raise ValueError("execution_identity must be a non-empty object")
    missing = [field for field in ("host", "project", "commit", "operations", "rollback") if field not in value]
    if missing:
        raise ValueError(
            "execution_identity must include host, project, commit, operations, and rollback; "
            f"missing: {', '.join(missing)}"
        )
    if not isinstance(value["host"], str) or not value["host"].strip():
        raise ValueError("execution_identity.host must be a non-empty string")
    if not isinstance(value["project"], str) or not value["project"].strip():
        raise ValueError("execution_identity.project must be a non-empty string")
    commit = value["commit"]
    if not isinstance(commit, str) or not re.fullmatch(r"(?:[a-f0-9]{40}|[a-f0-9]{64})", commit):
        raise ValueError("execution_identity.commit must be a full lowercase Git object id")
    operations = value["operations"]
    if (
        not isinstance(operations, list)
        or not operations
        or not all(isinstance(item, str) and item.strip() for item in operations)
    ):
        raise ValueError("execution_identity.operations must be a non-empty list of strings")
    rollback = value["rollback"]
    if not isinstance(rollback, str) or not rollback.strip():
        raise ValueError("execution_identity.rollback must be a non-empty string")
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _canonical_actor(value: dict[str, Any]) -> str:
    if not isinstance(value, dict):
        raise ValueError("approval actor must be an authenticated identity")
    actor = {field: str(value.get(field) or "").strip() for field in _ACTOR_FIELDS}
    if not actor["user_id"] or not actor["provider"]:
        raise ValueError("approval actor must include authenticated user_id and provider")
    return json.dumps(actor, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def is_packet_approval_gate(conn: sqlite3.Connection, task_id: str) -> bool:
    return conn.execute(
        "SELECT 1 FROM packet_approval_gates WHERE task_id = ?", (task_id,),
    ).fetchone() is not None


def reject_generic_mutation(conn: sqlite3.Connection, task_id: str, action: str) -> None:
    if is_packet_approval_gate(conn, task_id):
        raise PacketApprovalRequired(
            f"{task_id} is an exact packet approval gate; generic {action} cannot satisfy it"
        )


def configure_packet_approval(
    conn: sqlite3.Connection,
    task_id: str,
    *,
    packet_sha256: str,
    execution_identity: dict[str, Any],
) -> dict[str, Any]:
    """Bind a blocked card to one packet hash and one execution identity."""
    from hermes_cli.kanban_db import _append_event, write_txn

    digest = _digest(packet_sha256)
    identity_json = _canonical_identity(execution_identity)
    now = int(time.time())
    with write_txn(conn):
        task = conn.execute(
            "SELECT status FROM tasks WHERE id = ?", (task_id,),
        ).fetchone()
        if task is None:
            raise ValueError(f"unknown task: {task_id}")
        if task["status"] != "blocked":
            raise ValueError("packet approval gate must be configured while the task is blocked")
        if is_packet_approval_gate(conn, task_id):
            raise ValueError(f"packet approval gate already configured for {task_id}")
        conn.execute(
            """
            INSERT INTO packet_approval_gates
                (task_id, packet_sha256, execution_identity, created_at)
            VALUES (?, ?, ?, ?)
            """,
            (task_id, digest, identity_json, now),
        )
        _append_event(
            conn,
            task_id,
            "packet_approval_configured",
            {
                "packet_sha256": digest,
                "execution_identity": json.loads(identity_json),
            },
        )
    return get_packet_approval(conn, task_id)


def get_packet_approval(conn: sqlite3.Connection, task_id: str) -> dict[str, Any] | None:
    row = conn.execute(
        "SELECT * FROM packet_approval_gates WHERE task_id = ?", (task_id,),
    ).fetchone()
    if row is None:
        return None
    return _gate_from_row(row)


def _gate_from_row(row: sqlite3.Row) -> dict[str, Any]:
    return {
        "task_id": row["task_id"],
        "packet_sha256": row["packet_sha256"],
        "execution_identity": json.loads(row["execution_identity"]),
        "created_at": int(row["created_at"]),
        "approved_packet_sha256": row["approved_packet_sha256"],
        "approved_execution_identity": (
            json.loads(row["approved_execution_identity"])
            if row["approved_execution_identity"] else None
        ),
        "actor": json.loads(row["approved_actor"]) if row["approved_actor"] else None,
        "approved_at": int(row["approved_at"]) if row["approved_at"] is not None else None,
    }


def _assert_exact(
    gate: dict[str, Any], *, packet_sha256: str, execution_identity: dict[str, Any], approved: bool,
) -> tuple[str, str]:
    digest = _digest(packet_sha256)
    identity_json = _canonical_identity(execution_identity)
    wanted_digest = gate["approved_packet_sha256"] if approved else gate["packet_sha256"]
    wanted_identity = (
        gate["approved_execution_identity"] if approved else gate["execution_identity"]
    )
    if digest != wanted_digest:
        raise PacketApprovalMismatch("packet hash does not match the exact approved packet")
    if json.loads(identity_json) != wanted_identity:
        raise PacketApprovalMismatch("execution identity does not match the exact approved execution")
    return digest, identity_json


def approve_packet(
    conn: sqlite3.Connection,
    task_id: str,
    *,
    packet_sha256: str,
    execution_identity: dict[str, Any],
    actor: dict[str, Any],
) -> dict[str, Any]:
    """Approve exactly one configured packet as an authenticated dashboard actor."""
    from hermes_cli.kanban_db import (
        _append_event,
        _fire_task_hook,
        _synthesize_ended_run,
        get_task,
        recompute_ready,
        write_txn,
    )

    actor_json = _canonical_actor(actor)
    now = int(time.time())
    with write_txn(conn):
        gate = get_packet_approval(conn, task_id)
        if gate is None:
            raise PacketApprovalRequired(f"{task_id} is not a configured packet approval gate")
        if gate["approved_at"] is not None:
            raise PacketApprovalRequired(f"{task_id} was already approved")
        digest, identity_json = _assert_exact(
            gate,
            packet_sha256=packet_sha256,
            execution_identity=execution_identity,
            approved=False,
        )
        task = conn.execute(
            "SELECT status FROM tasks WHERE id = ?", (task_id,),
        ).fetchone()
        if task is None or task["status"] != "blocked":
            raise PacketApprovalRequired("packet approval card is not blocked and pending")
        conn.execute(
            """
            UPDATE packet_approval_gates
               SET approved_packet_sha256 = ?,
                   approved_execution_identity = ?,
                   approved_actor = ?,
                   approved_at = ?
             WHERE task_id = ? AND approved_at IS NULL
            """,
            (digest, identity_json, actor_json, now, task_id),
        )
        changed = conn.execute(
            """
            UPDATE tasks
               SET status = 'done', completed_at = ?, block_kind = NULL,
                   block_recurrences = 0, current_run_id = NULL,
                   claim_lock = NULL, claim_expires = NULL, worker_pid = NULL
             WHERE id = ? AND status = 'blocked'
            """,
            (now, task_id),
        ).rowcount
        if changed != 1:
            raise PacketApprovalRequired("packet approval card changed while approval was recorded")
        receipt = {
            "packet_sha256": digest,
            "execution_identity": json.loads(identity_json),
            "actor": json.loads(actor_json),
            "approved_at": now,
        }
        run_id = _synthesize_ended_run(
            conn,
            task_id,
            outcome="completed",
            summary="Exact packet approved by authenticated dashboard actor.",
            metadata=receipt,
            profile=receipt["actor"]["user_id"],
        )
        _append_event(conn, task_id, "packet_approved", receipt, run_id=run_id)
        _append_event(
            conn,
            task_id,
            "completed",
            {"summary": "Exact packet approved.", "packet_approval": receipt},
            run_id=run_id,
        )
    recompute_ready(conn)
    done_task = get_task(conn, task_id)
    _fire_task_hook(
        "kanban_task_completed",
        done_task,
        task_id,
        run_id,
        summary="Exact packet approved by authenticated dashboard actor.",
    )
    return receipt


def require_packet_approval(
    conn: sqlite3.Connection,
    task_id: str,
    *,
    packet_sha256: str,
    execution_identity: dict[str, Any],
) -> dict[str, Any]:
    """Permanent fail-closed check for the controlled-write entry point."""
    # A single statement gives approval and lifecycle status one snapshot.
    # Separate reads could accept an approval retired by concurrent renewal.
    row = conn.execute(
        """SELECT g.*, t.status AS task_status FROM packet_approval_gates g
           JOIN tasks t ON t.id = g.task_id WHERE g.task_id = ?""", (task_id,),
    ).fetchone()
    gate = _gate_from_row(row) if row is not None else None
    if gate is None or gate["approved_at"] is None:
        raise PacketApprovalRequired("exact packet approval is absent")
    if row['task_status'] != "done":
        raise PacketApprovalRequired("approval card is not complete")
    _assert_exact(
        gate,
        packet_sha256=packet_sha256,
        execution_identity=execution_identity,
        approved=True,
    )
    return gate
