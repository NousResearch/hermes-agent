"""High-risk tool routing — terminal and execute_code via Cube microVM."""

from __future__ import annotations

from typing import Any, Optional

from .config import select_tier
from .token_client import cube_credentials


def run_terminal(
    *,
    command: str,
    background: bool = False,
    timeout: Optional[int] = None,
    task_id: Optional[str] = None,
    workdir: Optional[str] = None,
    pty: bool = False,
    notify_on_complete: bool = False,
    watch_patterns: Optional[list] = None,
) -> str:
    """Execute a shell command inside the Cube microVM."""
    from tools.terminal_tool import terminal_tool

    tier = select_tier(task_id)
    with cube_credentials(task_id, tier):
        return terminal_tool(
            command=command,
            background=background,
            timeout=timeout,
            task_id=task_id,
            workdir=workdir,
            pty=pty,
            notify_on_complete=notify_on_complete,
            watch_patterns=watch_patterns,
        )


def run_execute_code(
    *,
    code: str,
    task_id: Optional[str] = None,
    enabled_tools: Optional[list[str]] = None,
) -> str:
    """Execute Python inside the Cube microVM sandbox."""
    from tools.code_execution_tool import execute_code

    tier = select_tier(task_id)
    with cube_credentials(task_id, tier):
        return execute_code(
            code=code,
            task_id=task_id,
            enabled_tools=enabled_tools,
        )
