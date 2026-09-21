"""Require task-bound repair acknowledgement before Kanban reports completion."""

import os
import sqlite3
from contextlib import closing
from functools import partial
from pathlib import Path

from .ledger import FeedbackLedger


def guard_repair_completion(ctx, *, task_id, **_kwargs):
    worker_task = os.environ.get("HERMES_KANBAN_TASK", "").strip()
    if not worker_task and ctx.get_config("enabled", default=False) is not True:
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
            rows = connection.execute(
                "SELECT status, action_status FROM feedback_receipts WHERE task_id = ? "
                "AND feedback_kind IN ('review_comment', 'issue_comment', 'review', 'pr_repair') "
                "AND NOT (feedback_kind = 'pr_repair' AND feedback_id LIKE 'report:%')",
                (task_id,),
            ).fetchall()
            if all(status == "completed" and action in {"completed", "superseded"}
                   for status, action in rows):
                return None
        reason = "this task still has an unacknowledged feedback dispatch"
    except (OSError, sqlite3.Error, ValueError, TypeError):
        reason = "the task's durable feedback completion contract could not be verified"
    return {"action": "block", "message": (
        f"Feedback completion rejected: {reason}. Finish the authorized push and factual reply, "
        "then run the exact governed complete-feedback command; use retire-feedback only for its "
        "verified closed-PR case. If the contract cannot be completed, use kanban_block with the "
        "actual blocker. A summary or local commit is not an acknowledgement."
    )}


def register_repair_completion_policy(ctx):
    register_hook = getattr(ctx, "register_hook", None)
    if callable(register_hook):
        register_hook("pre_kanban_complete", partial(guard_repair_completion, ctx))
        register_hook("pre_kanban_review", partial(guard_repair_completion, ctx))
