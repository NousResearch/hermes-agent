from pathlib import Path


def test_pr_feedback_task_body_includes_resolved_receipt_worktree() -> None:
    from github_pr_feedback.cli import _kanban_create_argv
    from github_pr_feedback.controller import KanbanTask

    task = KanbanTask(
        title="GitHub PR feedback: acme/widgets#17",
        instructions="Run the bounded repair.",
        board="tradingbot-burndown",
        assignee="hermes-maintenance-steward",
        repository_path=Path("/Users/example/.hermes/github-pr-feedback/worktrees/receipt"),
        head_sha="a" * 40,
        branch="hermes/github-pr-feedback/receipt",
        idempotency_key="feedback:test",
        evidence={"repository": "acme/widgets", "pr_number": 17},
    )

    argv = _kanban_create_argv(task)
    body = argv[argv.index("--body") + 1]

    assert "Canonical receipt worktree:" in body
    assert str(task.repository_path) in body
    assert "Do not search for the worktree" in body
