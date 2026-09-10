"""Retire inactive duplicate dispatches after their exact feedback was acknowledged."""

from .github_client import GitHubClientError


def reconcile_inactive_actioned_duplicates(ledger, kanban, github, current, *, board):
    rows = ledger._connection.execute(
        "SELECT pending.feedback_kind, pending.feedback_id, pending.head_sha, "
        "pending.task_id, pending.lease_version FROM feedback_receipts AS pending "
        "WHERE pending.repository = ? AND pending.pr_number = ? AND pending.head_sha != ? "
        "AND pending.feedback_kind IN ('review_comment', 'issue_comment', 'review') "
        "AND pending.status = 'completed' AND pending.action_status = 'pending' "
        "AND EXISTS (SELECT 1 FROM feedback_receipts AS actioned "
        "WHERE actioned.repository = pending.repository AND actioned.pr_number = pending.pr_number "
        "AND actioned.feedback_kind = pending.feedback_kind AND actioned.feedback_id = pending.feedback_id "
        "AND actioned.status = 'completed' AND actioned.action_status = 'completed')",
        (current.base_repository, current.number, current.head_sha),
    ).fetchall()
    if not rows:
        return 0
    try:
        inactive = [row for row in rows if row[3] and kanban.task_status(board, row[3]) in {"done", "archived"}]
        if not inactive:
            return 0
        canonical = github.get_merge_state(current.base_repository, current.number)
    except (RuntimeError, GitHubClientError):
        return 0
    if (canonical.repository, canonical.number, canonical.head_sha, canonical.base_sha) != (
        current.base_repository, current.number, current.head_sha, current.base_sha
    ) or canonical.state != "OPEN" or canonical.merged:
        return 0
    count = 0
    with ledger._transaction():
        for kind, identity, old_head, task_id, lease_version in inactive:
            result = ledger._connection.execute(
                "UPDATE feedback_receipts AS pending SET action_status = 'superseded', last_error = ? "
                "WHERE repository = ? AND pr_number = ? AND feedback_kind = ? AND feedback_id = ? "
                "AND head_sha = ? AND task_id = ? AND lease_version = ? "
                "AND status = 'completed' AND action_status = 'pending' "
                "AND EXISTS (SELECT 1 FROM feedback_receipts AS actioned "
                "WHERE actioned.repository = pending.repository AND actioned.pr_number = pending.pr_number "
                "AND actioned.feedback_kind = pending.feedback_kind AND actioned.feedback_id = pending.feedback_id "
                "AND actioned.status = 'completed' AND actioned.action_status = 'completed')",
                ("Inactive duplicate dispatch superseded by an acknowledged feedback identity; no CI evidence created",
                 current.base_repository, current.number, kind, identity, old_head, task_id, lease_version),
            )
            count += result.rowcount
    return count
