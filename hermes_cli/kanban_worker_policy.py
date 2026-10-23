"""Assignee controls for persisted-card worker capabilities."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from hermes_cli.kanban_db import Task


@dataclass(frozen=True)
class WorkerPolicy:
    """Assignee-owned choices for card-supplied worker capabilities."""

    task_skills: Optional[frozenset[str]] = None
    review_skill: Optional[str] = "sdlc-review"
    lifecycle_toolset: Optional[str] = "kanban"
    allow_task_model_override: bool = True
    allow_task_reasoning_override: bool = True


def _worker_policy(hermes_home: Optional[str]) -> WorkerPolicy:
    """Load an assignee's opt-in worker policy, preserving legacy defaults."""
    if not hermes_home:
        return WorkerPolicy()
    from hermes_cli.config_effective import load_user_config_effective

    from hermes_cli.kanban_db_dispatch import _worker_profile_scope

    with _worker_profile_scope(hermes_home):
        config = load_user_config_effective(fail_closed=True)
    return _parse_worker_policy(config)


def _parse_worker_policy(config: dict) -> WorkerPolicy:
    """Only omitted fields inherit grants; explicit malformed policy is rejected."""
    kanban = config.get("kanban", {})
    if not isinstance(kanban, dict):
        raise ValueError("invalid kanban.worker_policy configuration")
    raw = kanban.get("worker_policy", {})
    if not isinstance(raw, dict):
        raise ValueError("invalid kanban.worker_policy mapping")

    task_skills = raw.get("task_skills", "all")
    if task_skills == "none":
        allowed_skills: Optional[frozenset[str]] = frozenset()
    elif isinstance(task_skills, list):
        if not all(isinstance(skill, str) and skill.strip() for skill in task_skills):
            raise ValueError("invalid kanban.worker_policy.task_skills")
        allowed_skills = frozenset(task_skills)
    elif task_skills == "all":
        allowed_skills = None
    else:
        raise ValueError("invalid kanban.worker_policy.task_skills")

    def _named_policy_value(name: str, default: str) -> Optional[str]:
        value = raw.get(name, default)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"invalid kanban.worker_policy.{name}")
        return None if value == "none" else value

    def _allowed(name: str) -> bool:
        value = raw.get(name, "allow")
        if value not in ("allow", "deny"):
            raise ValueError(f"invalid kanban.worker_policy.{name}")
        return value == "allow"

    return WorkerPolicy(
        task_skills=allowed_skills,
        review_skill=_named_policy_value("review_skill", "sdlc-review"),
        lifecycle_toolset=_named_policy_value("lifecycle_toolset", "kanban"),
        allow_task_model_override=_allowed("task_model_override"),
        allow_task_reasoning_override=_allowed("task_reasoning_override"),
    )


def _worker_profile_home(profile_arg: Optional[str]) -> Optional[str]:
    if not profile_arg:
        return None
    try:
        from hermes_cli.profiles import normalize_profile_name, resolve_profile_env

        return resolve_profile_env(normalize_profile_name(profile_arg))
    except FileNotFoundError:
        return None


def _apply_dispatcher_policy(conn, task: Task, lane: str) -> None:
    """Keep custom spawn callbacks and durable dispatcher evidence under the same policy."""
    from hermes_cli import kanban_db as kb

    _, refused = _apply_worker_policy(task, lane, _worker_profile_home(task.assignee))
    if refused:
        with kb.write_txn(conn):
            kb._append_event(conn, task.id, "worker_policy_refused", {"refused": refused},
                             run_id=task.current_run_id)


def _apply_worker_policy(task: Task, lane: str, hermes_home: Optional[str], *, policy: Optional[WorkerPolicy] = None) -> tuple[WorkerPolicy, list[str]]:
    """Filter dispatcher injections and return the refused capability names."""
    policy = policy if policy is not None else _worker_policy(hermes_home)
    refused: list[str] = []
    original_skills = list(task.skills or ())
    if policy.task_skills is not None:
        task.skills = [skill for skill in original_skills if skill in policy.task_skills]
        if len(task.skills) != len(original_skills):
            refused.append("task_skills")
    if lane == "review":
        if policy.review_skill:
            task.skills = list(dict.fromkeys([*(task.skills or []), policy.review_skill]))
        else:
            refused.append("review_skill")
    if (task.model_override or task.provider_override) and not policy.allow_task_model_override:
        task.model_override = None
        task.provider_override = None
        refused.append("task_model_override")
    if task.reasoning_effort and not policy.allow_task_reasoning_override:
        task.reasoning_effort = None
        refused.append("task_reasoning_override")
    if policy.lifecycle_toolset is None:
        refused.append("lifecycle_toolset")
    return policy, refused
