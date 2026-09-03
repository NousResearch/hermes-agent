"""Exit-status policy for Kanban worker processes."""

import os


def worker_exit_code(result: object) -> int:
    """Return the supervisor-visible exit code for a worker result."""
    if not isinstance(result, dict):
        return 0

    if (
        os.environ.get("HERMES_KANBAN_TASK")
        and result.get("failure_reason") in ("rate_limit", "billing")
    ):
        try:
            from hermes_cli.kanban_db import KANBAN_RATE_LIMIT_EXIT_CODE

            return KANBAN_RATE_LIMIT_EXIT_CODE
        except Exception:
            return 1

    return 1 if result.get("failed") else 0