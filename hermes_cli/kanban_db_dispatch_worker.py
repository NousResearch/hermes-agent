"""Worker command construction for the Kanban dispatcher."""

from __future__ import annotations

from typing import Optional
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from hermes_cli.kanban_db import Task


def worker_argv(task: Task, profile_arg: str, hermes_home: Optional[str]) -> list[str]:
    """Build the ``hermes -p <profile> --cli ... chat -q ...`` worker command."""
    from hermes_cli import kanban_db_dispatch as dispatch

    cmd = [
        *dispatch._resolve_hermes_argv(),
        "-p", profile_arg,
        # A worker must never boot the interactive TUI: its no-TTY bail-out
        # exits 0 without doing the task.
        "--cli",
        # Use the profile-scoped shell-hook allowlist.
        "--accept-hooks",
    ]
    for skill in task.skills or ():
        if skill:
            cmd.extend(["--skills", skill])
    if task.model_override:
        cmd.extend(["-m", task.model_override])
        if task.provider_override:
            cmd.extend(["--provider", task.provider_override])
    if task.reasoning_effort:
        cmd.extend(["--reasoning", task.reasoning_effort])
    worker_toolsets = dispatch._resolve_worker_cli_toolsets(hermes_home)
    if worker_toolsets:
        cmd.extend(["--toolsets", ",".join(worker_toolsets)])
    cmd.extend(["chat", "-q", f"work kanban task {task.id}"])
    return cmd
