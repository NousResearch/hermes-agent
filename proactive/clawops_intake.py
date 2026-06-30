"""Hermes-owned intake for ClawOps runtime work.

This module deliberately stops at kanban task creation. Hermes remains the
planner and user-facing owner; ClawOps/OpenClaw workers only receive queued
execution work and report results back through the existing kanban notifier.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Any, Mapping, Optional

from hermes_cli import kanban_db as kb


DEFAULT_ASSIGNEE = "default"
DEFAULT_CREATED_BY = "hermes-clawops-intake"
DEFAULT_MAX_RUNTIME_SECONDS = 1800


@dataclass(frozen=True)
class ClawOpsTask:
    task_id: str
    status: str
    assignee: str
    title: str
    body: str
    board: Optional[str] = None


def resolve_clawops_assignee(config: Optional[Mapping[str, Any]] = None) -> str:
    """Resolve the worker profile that should claim ClawOps tasks."""
    env_value = os.getenv("HERMES_CLAWOPS_ASSIGNEE", "").strip()
    if env_value:
        return env_value

    cfg = config or {}
    for section_name, key in (
        ("clawops", "default_assignee"),
        ("kanban", "clawops_assignee"),
        ("proactive", "clawops_assignee"),
    ):
        section = cfg.get(section_name)
        if isinstance(section, Mapping):
            value = section.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()

    return DEFAULT_ASSIGNEE


def create_clawops_task(
    objective: str,
    *,
    source: Optional[Mapping[str, Any]] = None,
    assignee: Optional[str] = None,
    board: Optional[str] = None,
    created_by: str = DEFAULT_CREATED_BY,
    priority: int = 0,
    max_runtime_seconds: int = DEFAULT_MAX_RUNTIME_SECONDS,
    config: Optional[Mapping[str, Any]] = None,
) -> ClawOpsTask:
    """Create a Hermes-owned ClawOps task in the existing kanban queue."""
    clean_objective = (objective or "").strip()
    if not clean_objective:
        raise ValueError("objective is required")

    resolved_assignee = (assignee or "").strip() or resolve_clawops_assignee(config)
    title = _title_from_objective(clean_objective)
    body = _body_from_objective(clean_objective, source=source)

    with kb.connect_closing(board=board) as conn:
        task_id = kb.create_task(
            conn,
            title=title,
            body=body,
            assignee=resolved_assignee,
            created_by=created_by,
            priority=priority,
            workspace_kind="scratch",
            max_runtime_seconds=max_runtime_seconds,
        )
        row = kb.get_task(conn, task_id)
        status = str(row.status if row else "ready")

    return ClawOpsTask(
        task_id=task_id,
        status=status,
        assignee=resolved_assignee,
        title=title,
        body=body,
        board=board,
    )


def subscribe_clawops_task(
    task_id: str,
    *,
    platform: str,
    chat_id: str,
    thread_id: Optional[str] = None,
    user_id: Optional[str] = None,
    notifier_profile: Optional[str] = None,
    board: Optional[str] = None,
) -> bool:
    """Subscribe the originating Hermes channel to terminal task updates."""
    clean_platform = (platform or "").strip().lower()
    clean_chat_id = (chat_id or "").strip()
    if not task_id or not clean_platform or not clean_chat_id:
        return False

    with kb.connect_closing(board=board) as conn:
        kb.add_notify_sub(
            conn,
            task_id=task_id,
            platform=clean_platform,
            chat_id=clean_chat_id,
            thread_id=(thread_id or "").strip() or None,
            user_id=(user_id or "").strip() or None,
            notifier_profile=(notifier_profile or "").strip() or None,
        )
    return True


def _title_from_objective(objective: str) -> str:
    single_line = " ".join(objective.split())
    if len(single_line) <= 96:
        return f"ClawOps: {single_line}"
    return f"ClawOps: {single_line[:93].rstrip()}..."


def _body_from_objective(
    objective: str,
    *,
    source: Optional[Mapping[str, Any]] = None,
) -> str:
    lines = [
        "ClawOps runtime task created by Hermes.",
        "",
        "Control boundary:",
        "- Hermes remains the primary agent and user-facing decision owner.",
        "- ClawOps/OpenClaw may execute only delegated work in this queued task.",
        "- Results must return to Hermes for audit, review, and user-facing summary.",
        "",
        "Objective:",
        objective,
    ]
    if source:
        lines.extend(["", "Source:"])
        for key in sorted(source):
            value = source.get(key)
            if value is None or value == "":
                continue
            lines.append(f"- {key}: {value}")
    return "\n".join(lines)
