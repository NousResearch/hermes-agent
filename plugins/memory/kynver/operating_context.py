"""Resolve AgentOS parent context (goal/project/plan/task/progress row)."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Mapping

from hermes_cli.config import cfg_get

from .agentos_bridge import KynverAgentOSClient, KynverAgentOSError


@dataclass(frozen=True)
class OperatingContext:
    goal_id: str = ""
    project_id: str = ""
    plan_id: str = ""
    plan_version_id: str = ""
    task_id: str = ""
    progress_row_key: str = ""
    intake_required: bool = False
    intake_reason: str = ""

    @property
    def has_plan_anchor(self) -> bool:
        return bool(self.plan_id)

    @property
    def has_task_anchor(self) -> bool:
        return bool(self.task_id)


def _pick(env: Mapping[str, str], config: Mapping[str, Any] | None, *keys: str) -> str:
    for key in keys:
        val = (env.get(key) or "").strip()
        if val:
            return val
    if config:
        for key in keys:
            short = key.replace("KYNVER_", "").lower()
            nested = cfg_get(config, "kynver", short, default="") or ""
            if str(nested).strip():
                return str(nested).strip()
    return ""


def load_operating_context(
    env: Mapping[str, str] | None = None,
    config: Mapping[str, Any] | None = None,
) -> OperatingContext:
    merged = dict(env or os.environ)
    plan_id = _pick(merged, config, "KYNVER_PLAN_ID")
    plan_version_id = _pick(merged, config, "KYNVER_PLAN_VERSION_ID")
    task_id = _pick(merged, config, "KYNVER_TASK_ID", "KYNVER_AGENTOS_TASK_ID")
    progress_row_key = _pick(merged, config, "KYNVER_PROGRESS_ROW_KEY", "KYNVER_PLAN_PROGRESS_ROW_KEY")
    goal_id = _pick(merged, config, "KYNVER_GOAL_ID")
    project_id = _pick(merged, config, "KYNVER_PROJECT_ID")

    intake_required = False
    intake_reason = ""
    if not plan_id and not task_id:
        intake_required = True
        intake_reason = "missing_plan_and_task_anchor"

    return OperatingContext(
        goal_id=goal_id,
        project_id=project_id,
        plan_id=plan_id,
        plan_version_id=plan_version_id,
        task_id=task_id,
        progress_row_key=progress_row_key,
        intake_required=intake_required,
        intake_reason=intake_reason,
    )


def ensure_intake_task(
    client: KynverAgentOSClient,
    ctx: OperatingContext,
    *,
    hermes_session_id: str = "",
) -> OperatingContext:
    """Create explicit intake/classification AgentTask when parent context is missing."""

    if not ctx.intake_required:
        return ctx
    idem = f"hermes-forge:intake:{hermes_session_id or 'unknown'}"
    body: dict[str, Any] = {
        "title": "Classify unparented Forge work",
        "description": (
            "Hermes Forge attempted operating work without a resolved AgentOS plan/task anchor. "
            "Link this task to the correct goal, project, plan, and progress row before normal work proceeds."
        ),
        "priority": "high",
        "executor": "manual",
        "idempotencyKey": idem,
        "metadata": {"intakeReason": ctx.intake_reason, "hermesSessionId": hermes_session_id},
    }
    try:
        created = client.post("/tasks", body)
        task_id = ""
        if isinstance(created, dict):
            task_id = str(created.get("id") or created.get("taskId") or "")
        return OperatingContext(
            goal_id=ctx.goal_id,
            project_id=ctx.project_id,
            plan_id=ctx.plan_id,
            plan_version_id=ctx.plan_version_id,
            task_id=task_id or ctx.task_id,
            progress_row_key=ctx.progress_row_key,
            intake_required=True,
            intake_reason=ctx.intake_reason,
        )
    except KynverAgentOSError:
        return ctx


def todo_idempotency_key(todo_id: str) -> str:
    return f"hermes-forge:todo:{(todo_id or '').strip()}"
