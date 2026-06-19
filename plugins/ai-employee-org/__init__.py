"""AI Employee Org — five Hermes profiles as an autonomous company."""

from __future__ import annotations

from .cli import ai_employees_command, register_cli
from .slash import handle_slash


def register(ctx) -> None:
    """Register CLI, slash command, and lifecycle hooks."""
    ctx.register_cli_command(
        name="ai-employees",
        help="Bootstrap and manage the five-role AI employee organization",
        setup_fn=register_cli,
        handler_fn=ai_employees_command,
        description=(
            "Create secretary/job-recruiter/job-seeker/self-improver/delivery-worker "
            "profiles, kanban board ai-company, bundled skill, operator stack, and "
            "role crons with Telegram delivery."
        ),
    )
    ctx.register_command(
        "ai-employees",
        handler=handle_slash,
        description="AI employee org status and quick setup hints.",
        args_hint="[status|install]",
    )
