"""PROFILE-GATE resolver: human-approved profile create / delete.

Ties the kanban approval primitive (``kanban_db``) to the profile lifecycle
primitives (``profiles``). Autonomous create/delete blocks its task and writes
a pending approval; Sahil approves or rejects on Discord; on approve the
resolver runs the op here (execute-on-approve) with the one-shot authorisation
token and marks the task done.

Kept in its own module so ``kanban_db`` never imports ``profiles`` (avoids a
heavy import cycle); the Discord bridge imports only this module.
"""
from __future__ import annotations

import json
import logging
from typing import Optional

from hermes_cli import kanban_db as kb
from hermes_cli import profiles as profiles_mod

logger = logging.getLogger("hermes_cli.profile_lifecycle_gate")


def summarise_blast_radius(
    conn, op: str, profile: str, args: Optional[dict] = None
) -> str:
    """One-line, human-readable blast summary for the Discord prompt.

    Deliberately cheap: counts open tasks routed to the profile for a
    delete, and states the clone source for a create. Not a full EditGuard
    analysis (that governs config edits, not lifecycle).
    """
    canon = profiles_mod.normalize_profile_name(profile)
    if op == "delete":
        try:
            open_tasks = conn.execute(
                "SELECT COUNT(*) FROM tasks WHERE assignee = ? "
                "AND status NOT IN ('done', 'archived')",
                (canon,),
            ).fetchone()[0]
        except Exception:
            open_tasks = "?"
        exists = profiles_mod.get_profile_dir(canon).is_dir()
        return (
            f"DELETE profile '{canon}'"
            f"{'' if exists else ' (MISSING on disk)'}; "
            f"{open_tasks} open task(s) routed to it"
        )
    clone = (args or {}).get("clone_from")
    return (
        f"CREATE profile '{canon}'"
        + (f" cloned from '{clone}'" if clone else " (fresh)")
    )


def submit_lifecycle_request(
    conn,
    task_id: str,
    *,
    op: str,
    profile: str,
    args: Optional[dict] = None,
    requested_by: Optional[str] = None,
    expected_run_id: Optional[int] = None,
) -> str:
    """Entry point for an autonomous lifecycle request.

    Computes the blast summary, blocks the task, and records the pending
    approval. Returns the approval id. The Discord bridge picks it up and
    posts the approve/reject prompt.
    """
    blast = summarise_blast_radius(conn, op, profile, args)
    approval_id = kb.request_profile_lifecycle_approval(
        conn, task_id,
        op=op, profile=profile, args=args,
        requested_by=requested_by, blast_summary=blast,
        expected_run_id=expected_run_id,
    )
    logger.info(
        "profile-gate: %s %s requested approval %s (task %s)",
        op, profile, approval_id, task_id,
    )
    return approval_id


def _run_op(op: str, profile: str, args: dict, token: str) -> None:
    """Execute the approved op under the approval's one-shot authorisation."""
    # args may arrive as None when a request stored no extra args (e.g. a
    # create with no clone_from, serialised as JSON null). Coerce so the
    # **kwargs splat never throws on the approve path.
    args = args or {}
    with profiles_mod.lifecycle_authorised(token):
        if op == "create":
            profiles_mod.create_profile(profile, **args)
        elif op == "delete":
            profiles_mod.delete_profile(profile, yes=True)
        else:  # pragma: no cover - validated upstream
            raise ValueError(f"unknown lifecycle op {op!r}")


def _close_task(conn, task_id: str, *, result: str) -> None:
    """Move the blocked requester task to a terminal state with a note.

    Tries ``done`` (unblock then complete); falls back to ``archived``. If
    BOTH fail the task is left actionable while its approval is already
    resolved, so log loudly rather than swallow it.

    Exception-safe: this runs AFTER the (irreversible) op has executed, so a
    DB error here must not propagate and mask the completed side effect; it
    is logged instead.
    """
    try:
        kb.unblock_task(conn, task_id)
        if kb.complete_task(conn, task_id, result=result, summary=result):
            return
        if kb.archive_task(conn, task_id):
            return
        logger.error(
            "profile-gate: could not move task %s to a terminal state after "
            "resolving its approval; manual cleanup needed", task_id,
        )
    except Exception:
        logger.exception(
            "profile-gate: error closing task %s after approval resolution; "
            "the op already ran, manual cleanup may be needed", task_id,
        )


def approve(conn, approval_id: str, *, resolved_by: str) -> dict:
    """Approve and execute a pending lifecycle op.

    Returns a result dict ``{ok, op, profile, error}``. Raises ValueError if
    the approval is unknown or not pending (so a double-click is a no-op).
    """
    row = kb.resolve_profile_lifecycle_approval(
        conn, approval_id, decision="approve", resolved_by=resolved_by
    )
    op, profile = row["op"], row["profile"]
    args = json.loads(row.get("args_json") or "{}")
    try:
        _run_op(op, profile, args, row["token"])
    except Exception as exc:
        logger.exception("profile-gate: %s %s failed", op, profile)
        kb.mark_profile_lifecycle_executed(
            conn, approval_id, ok=False, error=str(exc)
        )
        _close_task(
            conn, row["task_id"],
            result=f"profile {op} {profile} APPROVED but FAILED: {exc}",
        )
        return {"ok": False, "op": op, "profile": profile, "error": str(exc)}
    kb.mark_profile_lifecycle_executed(conn, approval_id, ok=True)
    _close_task(
        conn, row["task_id"],
        result=f"profile {op} {profile} approved by {resolved_by} and executed",
    )
    logger.info("profile-gate: %s %s executed (approval %s)", op, profile, approval_id)
    return {"ok": True, "op": op, "profile": profile, "error": None}


def reject(conn, approval_id: str, *, resolved_by: str) -> dict:
    """Reject a pending lifecycle op; the op never runs."""
    row = kb.resolve_profile_lifecycle_approval(
        conn, approval_id, decision="reject", resolved_by=resolved_by
    )
    op, profile = row["op"], row["profile"]
    _close_task(
        conn, row["task_id"],
        result=f"profile {op} {profile} REJECTED by {resolved_by}",
    )
    logger.info("profile-gate: %s %s rejected (approval %s)", op, profile, approval_id)
    return {"ok": True, "op": op, "profile": profile, "rejected": True}
