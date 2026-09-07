"""Reject model-authored CI completion without a durable exact-dispatch receipt.

This uses the native pre_tool_call policy hook. Native hooks select the first
block/approve directive, so an earlier approval hook can mask this veto. It is
not a replacement for the controller's independently validated merge boundary.
"""
from __future__ import annotations

import json
import os
import sqlite3
from contextlib import closing
from datetime import datetime
from functools import partial
from pathlib import Path

from .ci_runner import CIAuditReceipt
from .controller import _local_ci_feedback_id
from .ledger import FeedbackLedger


def _has_receipt(connection, binding) -> bool:
    repository, number, feedback_id, head, claimed_at = binding
    claimed = datetime.fromisoformat(claimed_at)
    if claimed.tzinfo is None:
        return False
    rows = connection.execute(
        "SELECT evidence_json FROM ci_audit_receipts WHERE repository = ? AND pr_number = ? "
        "AND head_sha = ? ORDER BY completed_at DESC",
        (repository, number, head),
    )
    for (payload,) in rows:
        receipt = CIAuditReceipt.from_payload(json.loads(payload))
        if (receipt.status == "passed" and receipt.identity.repository == repository and receipt.identity.pr_number == number
                and receipt.identity.head_sha == head
                and _local_ci_feedback_id(receipt.identity) == feedback_id
                and receipt.started_at >= claimed):
            return True
    return False


def guard_completion(ctx, *, tool_name: str = "", args=None, **_kwargs):
    """Model tools need ledger evidence; deterministic audit-pr uses its own CLI transition."""
    if tool_name != "kanban_complete":
        return None
    worker_task = os.environ.get("HERMES_KANBAN_TASK", "").strip()
    if not worker_task and ctx.get_config("enabled", default=False) is not True:
        return None
    args = args if isinstance(args, dict) else {}
    target = str(args.get("task_id") or worker_task or "").strip()
    if not target:
        return None
    try:
        if worker_task:
            from hermes_constants import get_default_hermes_root

            control_home = os.environ.get("HERMES_CONTROL_HOME", "").strip()
            root = Path(control_home) if control_home else get_default_hermes_root()
            path = root / "github-pr-feedback" / "ledger.sqlite3"
        else:
            path = FeedbackLedger.current_profile_path()
        with closing(sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True, timeout=1)) as connection:
            # One snapshot covers both task ownership and evidence; never create/migrate a ledger here.
            connection.execute("BEGIN")
            bindings = connection.execute(
                "SELECT repository, pr_number, feedback_id, head_sha, claimed_at "
                "FROM feedback_receipts WHERE task_id = ? AND feedback_kind = 'pr_local_ci'",
                (target,),
            ).fetchall()
            if not bindings:
                return None
            if all(_has_receipt(connection, binding) for binding in bindings):
                return None
        reason = "no typed passing durable CI receipt matches this task's exact PR head/base and dispatch"
    except (OSError, sqlite3.Error, ValueError, TypeError, KeyError):
        reason = "the durable CI task binding or receipt could not be verified"
    return {"action": "block", "message": (
        f"CI completion rejected: {reason}. Run the governed github-pr-feedback audit-pr command; "
        "its deterministic receipt and handoff own completion. If audit cannot run, use kanban_block "
        "with the exact blocker. A summary or claimed command is not CI evidence."
    )}


def register_completion_guard(ctx) -> None:
    register_hook = getattr(ctx, "register_hook", None)
    if callable(register_hook):
        register_hook("pre_tool_call", partial(guard_completion, ctx))
