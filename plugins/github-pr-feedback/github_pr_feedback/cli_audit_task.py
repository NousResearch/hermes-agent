"""Bind audit-owned task transitions to the durable exact-head dispatch."""
import os

from .controller import _local_ci_feedback_id
from .policy import FeedbackReceipt


def owns_current_audit_task(ledger, identity) -> bool:
    task_id = os.environ.get("HERMES_KANBAN_TASK", "").strip()
    if not task_id:
        return False
    receipt = FeedbackReceipt(
        identity.repository, identity.pr_number, "pr_local_ci",
        _local_ci_feedback_id(identity), identity.head_sha,
    )
    binding = ledger.exact_pending_task_binding(receipt)
    return binding is not None and binding.task_id == task_id
