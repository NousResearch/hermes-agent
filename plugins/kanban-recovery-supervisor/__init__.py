"""Opt-in, bounded recovery supervisor for durable Kanban block events."""

from __future__ import annotations

from typing import Any

from .supervisor import on_task_blocked


def register(ctx: Any) -> None:
    """Register the post-commit observer through Hermes' public plugin API."""
    ctx.register_hook("kanban_task_blocked", on_task_blocked)
