"""Retire obsolete conflict observations without acknowledging code findings."""


def reconcile_inactive_conflicts(ledger, kanban, github, pull, *, board):
    if pull.state != "OPEN" or pull.merged or not pull.mergeable or pull.merge_state_status == "DIRTY":
        return 0
    rows = ledger._connection.execute(
        "SELECT feedback_id, head_sha, task_id FROM feedback_receipts "
        "WHERE repository = ? AND pr_number = ? AND feedback_kind = 'pr_repair' "
        "AND feedback_id LIKE 'repair:merge_conflict:target-base:%' "
        "AND head_sha != ? AND status = 'completed' AND action_status = 'pending'",
        (pull.repository, pull.number, pull.head_sha),
    ).fetchall()
    inactive = [row for row in rows if row[2] and kanban.task_status(board, row[2]) in {"done", "archived"}]
    if not inactive:
        return 0
    current = github.get_merge_state(pull.repository, pull.number)
    if (current.repository, current.number, current.head_sha, current.base_sha) != (
        pull.repository, pull.number, pull.head_sha, pull.base_sha
    ) or current.state != "OPEN" or current.merged or not current.mergeable or current.merge_state_status == "DIRTY":
        return 0
    count = 0
    with ledger._transaction():
        for feedback_id, old_head, task_id in inactive:
            result = ledger._connection.execute(
                "UPDATE feedback_receipts SET action_status = 'superseded', last_error = ? "
                "WHERE repository = ? AND pr_number = ? AND feedback_kind = 'pr_repair' "
                "AND feedback_id = ? AND head_sha = ? AND task_id = ? "
                "AND status = 'completed' AND action_status = 'pending'",
                (f"Inactive conflict dispatch superseded by conflict-free head {current.head_sha} "
                 f"against base {current.base_sha}; no code findings or CI acknowledged",
                 current.repository, current.number, feedback_id, old_head, task_id),
            )
            count += result.rowcount
    return count
