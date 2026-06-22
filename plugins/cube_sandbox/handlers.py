"""Plugin tool handlers — override built-in terminal / execute_code."""

from __future__ import annotations

from typing import Any

from .router import run_execute_code, run_terminal


def handle_terminal(args: dict[str, Any], **kw: Any) -> str:
    return run_terminal(
        command=args.get("command", ""),
        background=args.get("background", False),
        timeout=args.get("timeout"),
        task_id=kw.get("task_id"),
        workdir=args.get("workdir"),
        pty=args.get("pty", False),
        notify_on_complete=args.get("notify_on_complete", False),
        watch_patterns=args.get("watch_patterns"),
    )


def handle_execute_code(args: dict[str, Any], **kw: Any) -> str:
    return run_execute_code(
        code=args.get("code", ""),
        task_id=kw.get("task_id"),
        enabled_tools=kw.get("enabled_tools"),
    )
