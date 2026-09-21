"""Goal-to-Kanban routing helpers.

This module deliberately keeps routing as a thin explicit layer on top of the
existing /goal loop and Kanban create tool. It does not mutate the goal judge,
change dispatch ownership, or auto-dispatch destructive work.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable


DEFAULT_ASSIGNEE = "orchestrator"
DEFAULT_THEME = "goal-routing"


@dataclass(frozen=True)
class GoalKanbanTask:
    title: str
    body: str
    assignee: str
    triage: bool
    initial_status: str
    priority: int
    theme: str
    session_id: str | None
    idempotency_key: str
    parents: tuple[str, ...] = ()

    def to_kanban_create_args(self) -> dict[str, Any]:
        args: dict[str, Any] = {
            "title": self.title,
            "body": self.body,
            "assignee": self.assignee,
            "triage": self.triage,
            "initial_status": self.initial_status,
            "priority": self.priority,
            "theme": self.theme,
            "idempotency_key": self.idempotency_key,
        }
        if self.parents:
            args["parents"] = list(self.parents)
        if self.session_id:
            args["session_id"] = self.session_id
        return args


@dataclass(frozen=True)
class GoalKanbanRouteResult:
    task: GoalKanbanTask
    task_id: str | None
    status: str | None
    error: str | None = None
    child_task_ids: tuple[str, ...] = ()

    @property
    def ok(self) -> bool:
        return self.error is None and bool(self.task_id)


def _current_profile_count() -> int | None:
    profiles_dir = Path.home() / ".hermes" / "profiles"
    try:
        return sum(1 for child in profiles_dir.iterdir() if child.is_dir())
    except Exception:
        return None


def _normalise_subgoals(subgoals: Iterable[str] | None) -> list[str]:
    if not subgoals:
        return []
    return [str(item).strip() for item in subgoals if str(item).strip()]


def _idempotency_key(goal: str, subgoals: list[str], session_id: str | None) -> str:
    payload = json.dumps(
        {"goal": goal, "subgoals": subgoals, "session_id": session_id or ""},
        sort_keys=True,
        ensure_ascii=False,
    )
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]
    return f"goal-route:{session_id or 'no-session'}:{digest}"


def build_goal_kanban_task(
    goal: str,
    *,
    subgoals: Iterable[str] | None = None,
    session_id: str | None = None,
    assignee: str = DEFAULT_ASSIGNEE,
) -> GoalKanbanTask:
    """Build the parent Kanban task for a routed /goal.

    The body intentionally carries open-risk language. Prior architecture
    reviews found recurring drift where broken items were described as
    resolved. Keeping these risks in every routed parent task forces the
    owner/reviewer to actively account for them before execution.
    """
    clean_goal = str(goal or "").strip()
    if not clean_goal:
        raise ValueError("goal text is required")

    clean_subgoals = _normalise_subgoals(subgoals)
    observed_count = _current_profile_count()
    count_text = (
        f"current observed count: {observed_count}"
        if observed_count is not None
        else "current observed count unavailable"
    )

    subgoal_block = "\n".join(
        f"- [ ] {idx}. {text}" for idx, text in enumerate(clean_subgoals, start=1)
    ) or "- [ ] No sub-goals supplied yet; Orchestrator must decompose before execution."

    body = f"""Source: /goal Kanban route
Session: {session_id or 'unknown'}
Goal: {clean_goal}

Sub-goals / acceptance criteria:
{subgoal_block}

Routing:
- KENSEI owns final judgement and cron/Kanban dispatch policy.
- Orchestrator decomposes; leads own lanes; workers execute; Quan/KENSEI Review gates quality.

Open risks to account for:
- Fleet hygiene: open risk. Baseline live count 51; {count_text}. Avoid duplicate/variant routing drift.
- Knowledge accumulation: open risk. GitNexus/LadybugDB/GBrain remain unreliable unless separately verified.
- Webhook autonomy: open risk. Use Kanban + HMAC/allowlist; no direct destructive webhook action.
- Skill cross-pollination: open risk until logged ritual output exists.
- Dispatcher safety: Kensei owns cron/Kanban dispatch; specialists must not own dispatcher loops.
- Execution wording: Do not describe broken items as resolved.

Definition of done:
- Child tasks have owner, assignee, reviewer/QA path, AC, and risks/gates.
- External/researched skills go through `skill-research` rewrite before use.
- Dispatcher safety checked before/after routing.
"""

    return GoalKanbanTask(
        title=f"Goal: {clean_goal}",
        body=body,
        assignee=assignee,
        triage=True,
        initial_status="backlog",
        priority=1,
        theme=DEFAULT_THEME,
        session_id=session_id,
        idempotency_key=_idempotency_key(clean_goal, clean_subgoals, session_id),
    )


def _default_create_task(args: dict[str, Any]) -> str:
    from tools.kanban_tools import _handle_create

    return _handle_create(args)


def _assignee_for_subgoal(text: str) -> str:
    lower = text.lower()
    if any(k in lower for k in ("test", "qa", "review", "verify", "quality")):
        return "quan-code"
    if any(k in lower for k in ("content", "draft", "post", "copy", "social")):
        return "ceecee-writer"
    if any(k in lower for k in ("research", "market", "scan", "signal")):
        return "remii-deep"
    if any(k in lower for k in ("ops", "gateway", "cron", "backup", "infra")):
        return "wesker-ops"
    if any(k in lower for k in ("design", "ux", "ui", "component", "brand")):
        return "dezzy-ux-prototype"
    if any(k in lower for k in ("skill", "profile", "governance", "audit")):
        return "denji-reviewer"
    if any(k in lower for k in ("frontend", "ui code", "react", "screen")):
        return "octacon-frontend"
    if any(k in lower for k in ("backend", "api", "database", "db", "route")):
        return "octacon-backend"
    return "orchestrator"


def build_goal_child_tasks(
    *,
    parent_task_id: str,
    goal: str,
    subgoals: Iterable[str] | None,
    session_id: str | None = None,
) -> list[GoalKanbanTask]:
    clean_subgoals = _normalise_subgoals(subgoals)
    tasks: list[GoalKanbanTask] = []
    for idx, subgoal in enumerate(clean_subgoals, start=1):
        assignee = _assignee_for_subgoal(subgoal)
        body = f"""Source: /goal child decomposition
Parent goal: {goal}
Parent task: {parent_task_id}
Session: {session_id or 'unknown'}

Sub-goal:
{subgoal}

Acceptance criteria:
- Complete this sub-goal only; do not broaden scope.
- Respect approval gates for destructive/provider/auth/publishing actions.
- If external/researched skills are needed, block for `skill-research` rewrite first.
- Keep GitNexus/LadybugDB/GBrain claims marked open unless separately verified.
- Call kanban_complete or kanban_block with concrete evidence.
"""
        tasks.append(
            GoalKanbanTask(
                title=f"Sub-goal {idx}: {subgoal}",
                body=body,
                assignee=assignee,
                triage=False,
                initial_status="backlog",
                priority=1,
                theme=DEFAULT_THEME,
                session_id=session_id,
                idempotency_key=_idempotency_key(f"{goal}::{idx}::{subgoal}", [], session_id),
                parents=(parent_task_id,),
            )
        )
    return tasks


def _parse_create_response(raw: str) -> dict[str, Any]:
    try:
        parsed = json.loads(raw)
    except Exception:
        return {"ok": False, "error": f"kanban_create returned non-JSON response: {raw}"}
    if not isinstance(parsed, dict):
        return {"ok": False, "error": "kanban_create returned non-object JSON"}
    return parsed


def route_goal_to_kanban(
    goal: str,
    *,
    subgoals: Iterable[str] | None = None,
    session_id: str | None = None,
    assignee: str = DEFAULT_ASSIGNEE,
    create_task: Callable[[dict[str, Any]], str] | None = None,
    decompose: bool = False,
) -> GoalKanbanRouteResult:
    clean_subgoals = _normalise_subgoals(subgoals)
    task = build_goal_kanban_task(
        goal,
        subgoals=clean_subgoals,
        session_id=session_id,
        assignee=assignee,
    )
    creator = create_task or _default_create_task
    response = _parse_create_response(creator(task.to_kanban_create_args()))
    if response.get("ok") is False:
        return GoalKanbanRouteResult(
            task=task,
            task_id=None,
            status=None,
            error=str(response.get("error") or "kanban_create failed"),
        )
    task_id = response.get("task_id")
    if not task_id:
        return GoalKanbanRouteResult(
            task=task,
            task_id=None,
            status=response.get("status"),
            error="kanban_create did not return a task_id",
        )

    child_ids: list[str] = []
    if decompose and clean_subgoals:
        for child in build_goal_child_tasks(
            parent_task_id=str(task_id),
            goal=str(goal).strip(),
            subgoals=clean_subgoals,
            session_id=session_id,
        ):
            child_response = _parse_create_response(creator(child.to_kanban_create_args()))
            if child_response.get("ok") is False or not child_response.get("task_id"):
                return GoalKanbanRouteResult(
                    task=task,
                    task_id=str(task_id),
                    status=str(response.get("status") or ""),
                    error=str(child_response.get("error") or "child kanban_create failed"),
                    child_task_ids=tuple(child_ids),
                )
            child_ids.append(str(child_response["task_id"]))

    return GoalKanbanRouteResult(
        task=task,
        task_id=str(task_id),
        status=str(response.get("status") or ""),
        child_task_ids=tuple(child_ids),
    )
