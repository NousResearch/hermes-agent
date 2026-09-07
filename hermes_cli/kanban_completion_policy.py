"""Apply completion contracts before durable Kanban transitions."""


class CompletionPolicyError(ValueError):
    """A registered completion contract could not be satisfied."""


def enforce_completion_policies(*, task_id, board, assignee, summary):
    from hermes_cli.plugins import invoke_hook

    for result in invoke_hook(
        "pre_kanban_complete", task_id=task_id, board=board,
        assignee=assignee, summary=summary,
    ):
        if not isinstance(result, dict) or result.get("action") not in {"block", "allow", "approve"}:
            raise CompletionPolicyError("Kanban completion policy returned an invalid decision")
        if result["action"] == "block":
            raise CompletionPolicyError(result.get("message") or "Kanban completion policy rejected this transition")
