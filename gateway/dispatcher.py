"""Minimal workflow dispatcher for PM-style Feishu collaboration."""

from __future__ import annotations

import re
import json
from typing import Any, Dict, List, Optional

from gateway import task_state
from gateway.message_renderer import render_task_message
from gateway.message_validator import validate_message_events
from gateway.workflow_profile import WorkflowProfile, WorkflowRole


_PM_PREFIX_RE = re.compile(r"^\s*@?\s*pm\b[\s:：,，-]*", re.IGNORECASE)
_SEPARATE_TARGET_RE = re.compile(r"分别发给(?P<targets>.+?)(?:，|,|。|$)")


def _strip_pm_prefix(text: str) -> str:
    return _PM_PREFIX_RE.sub("", str(text or "").strip(), count=1).strip()


def is_pm_dispatch_message(text: str) -> bool:
    return bool(_PM_PREFIX_RE.match(str(text or "")))


def _role_by_text(profile: WorkflowProfile, token: str) -> Optional[WorkflowRole]:
    normalized = str(token or "").strip().lower()
    for role in profile.roles:
        values = [role.role_id, role.name, *role.aliases]
        if any(str(value or "").strip().lower() == normalized for value in values):
            return role
    return None


def _parse_separate_targets(text: str, profile: WorkflowProfile) -> List[WorkflowRole]:
    match = _SEPARATE_TARGET_RE.search(text)
    if not match:
        return []
    raw_targets = re.split(r"[/、,，\s]+", match.group("targets"))
    roles: List[WorkflowRole] = []
    seen: set[str] = set()
    for raw in raw_targets:
        role = _role_by_text(profile, raw)
        if role and role.role_id not in seen and role.role_id != profile.dispatcher_role:
            seen.add(role.role_id)
            roles.append(role)
    return roles


def _message_for_task(profile: WorkflowProfile, role: WorkflowRole, task: Dict[str, Any]) -> str:
    return render_task_message(profile, role, task)


def _event_for_task(profile: WorkflowProfile, role: WorkflowRole, task: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "content": _message_for_task(profile, role, task),
        "to_role": role.role_id,
        "task_type": task["task_type"],
        "deliverable": task["deliverable"],
        "return_to": task["return_to"],
        "reviewer_role": task.get("reviewer_role", ""),
        "deliver_to_role": task.get("deliver_to_role", task["return_to"]),
        "deliver_after_review_to_role": task.get("deliver_after_review_to_role", task["return_to"]),
        "upstream_role": task.get("upstream_role", ""),
        "upstream_summary": task.get("upstream_summary", ""),
        "send_mode": "real_send",
        "real_sent": False,
        "metadata": {
            "workflow_id": task["workflow_id"],
            "task_id": task["task_id"],
            "profile_id": profile.profile_id,
            "to_role": role.role_id,
            "target_role": role.role_id,
            "target_profile": role.profile,
            "role_code": role.role_id,
            "task_type": task["task_type"],
            "deliverable": task["deliverable"],
            "return_to": task["return_to"],
            "reviewer_role": task.get("reviewer_role", ""),
            "deliver_to_role": task.get("deliver_to_role", task["return_to"]),
            "deliver_after_review_to_role": task.get("deliver_after_review_to_role", task["return_to"]),
            "upstream_role": task.get("upstream_role", ""),
            "upstream_summary": task.get("upstream_summary", ""),
            "deliverable_format": task.get("deliverable_format", task["deliverable"]),
            "send_mode": "real_send",
        },
    }


def _make_workflow_with_tasks(
    *,
    profile: WorkflowProfile,
    title: str,
    roles: List[WorkflowRole],
    task_type: str,
    deliverable: str,
    instruction: str,
    return_to: str,
    current_node: str,
    next_action: str,
    reviewer_role: str = "",
    deliver_to_role: str = "",
    upstream_role: str = "",
    upstream_summary: str = "",
    deliverable_format: str = "",
) -> Dict[str, Any]:
    workflow = task_state.create_workflow(
        profile_id=profile.profile_id,
        title=title,
        dispatcher_role=profile.dispatcher_role,
        current_node=current_node,
        next_action=next_action,
    )
    tasks = [
        task_state.create_task(
            workflow_id=workflow["workflow_id"],
            to_role=role.role_id,
            task_type=task_type,
            deliverable=deliverable,
            instruction=instruction,
            return_to=return_to,
            reviewer_role=reviewer_role,
            deliver_to_role=deliver_to_role or return_to,
            deliver_after_review_to_role=deliver_to_role or return_to,
            upstream_role=upstream_role,
            upstream_summary=upstream_summary,
            deliverable_format=deliverable_format or deliverable,
        )
        for role in roles
    ]
    task_state.add_pending_tasks(workflow["workflow_id"], tasks)
    events = [_event_for_task(profile, role, task) for role, task in zip(roles, tasks)]
    validation = validate_message_events(events, profile)
    if validation["ok"]:
        for task in tasks:
            task_state.mark_task_validated(workflow_id=workflow["workflow_id"], task_id=task["task_id"])
            task_state.mark_task_send_pending(workflow_id=workflow["workflow_id"], task_id=task["task_id"])
            task["status"] = "send_pending"
    else:
        validation_error = json.dumps(validation, ensure_ascii=False)
        for task in tasks:
            task_state.mark_task_failed(
                workflow_id=workflow["workflow_id"],
                task_id=task["task_id"],
                error=validation_error,
            )
            task["status"] = "failed"
    return {
        "handled": True,
        "result": "dispatch",
        "profile": profile,
        "workflow": workflow,
        "tasks": tasks,
        "message_events": events,
        "validator": "pass" if validation["ok"] else "fail",
        "validation": validation,
        "message_event_count": len(events),
        "task_count": len(tasks),
        "next_state": current_node,
    }


def _make_tasks_for_existing_workflow(
    *,
    profile: WorkflowProfile,
    workflow: Dict[str, Any],
    roles: List[WorkflowRole],
    task_type: str,
    deliverable: str,
    instruction: str,
    return_to: str,
    current_node: str,
    next_action: str,
    reviewer_role: str = "",
    deliver_to_role: str = "",
    upstream_role: str = "",
    upstream_summary: str = "",
    deliverable_format: str = "",
) -> Dict[str, Any]:
    workflow_id = str(workflow.get("workflow_id") or "")
    tasks = [
        task_state.create_task(
            workflow_id=workflow_id,
            to_role=role.role_id,
            task_type=task_type,
            deliverable=deliverable,
            instruction=instruction,
            return_to=return_to,
            reviewer_role=reviewer_role,
            deliver_to_role=deliver_to_role or return_to,
            deliver_after_review_to_role=deliver_to_role or return_to,
            upstream_role=upstream_role,
            upstream_summary=upstream_summary,
            deliverable_format=deliverable_format or deliverable,
        )
        for role in roles
    ]
    task_state.add_pending_tasks(workflow_id, tasks)
    events = [_event_for_task(profile, role, task) for role, task in zip(roles, tasks)]
    validation = validate_message_events(events, profile)
    if validation["ok"]:
        for task in tasks:
            task_state.mark_task_validated(workflow_id=workflow_id, task_id=task["task_id"])
            task_state.mark_task_send_pending(workflow_id=workflow_id, task_id=task["task_id"])
            task["status"] = "send_pending"
    else:
        validation_error = json.dumps(validation, ensure_ascii=False)
        for task in tasks:
            task_state.mark_task_failed(
                workflow_id=workflow_id,
                task_id=task["task_id"],
                error=validation_error,
            )
            task["status"] = "failed"
    task_state.update_workflow_progress(
        workflow_id=workflow_id,
        current_node=current_node,
        next_action=next_action,
    )
    workflow = task_state.get_workflow(workflow_id) or workflow
    return {
        "handled": True,
        "result": "dispatch",
        "profile": profile,
        "workflow": workflow,
        "tasks": tasks,
        "message_events": events,
        "validator": "pass" if validation["ok"] else "fail",
        "validation": validation,
        "message_event_count": len(events),
        "task_count": len(tasks),
        "next_state": current_node,
    }


def dispatch_workflow_delivery(envelope: Dict[str, Any], profile: Optional[WorkflowProfile] = None) -> Dict[str, Any]:
    """Handle a completed upstream task delivered back to the dispatcher."""
    if profile is None or not isinstance(envelope, dict):
        return {"handled": False}
    if str(envelope.get("relay_type") or "").strip() != "workflow_task_return":
        return {"handled": False}
    to_role = str(envelope.get("to_role") or "").strip().lower()
    if to_role and to_role != profile.dispatcher_role:
        return {"handled": False}

    workflow_id = str(envelope.get("workflow_id") or "").strip()
    upstream_task_id = str(envelope.get("upstream_task_id") or envelope.get("task_id") or "").strip()
    workflow = task_state.get_workflow(workflow_id) if workflow_id else None
    upstream_task = task_state.get_task(workflow_id=workflow_id, task_id=upstream_task_id) if workflow_id and upstream_task_id else None
    if not workflow or not upstream_task:
        return {
            "handled": True,
            "result": "workflow_delivery_waiting_dispatcher",
            "profile": profile,
            "summary": "上游结果已经回来了，但没有找到对应的 workflow 任务。当前等待 PM 人工确认下一步。",
            "message_events": [],
            "tasks": [],
        }

    upstream_summary = str(
        envelope.get("upstream_summary")
        or upstream_task.get("result_summary")
        or ""
    ).strip()
    upstream_role = str(envelope.get("from_role") or upstream_task.get("to_role") or "").strip().lower()
    action_role = profile.find_role_by_capability("action_design")
    if upstream_task.get("task_type") == "direction_decision" and action_role and action_role.role_id != upstream_role:
        result = _make_tasks_for_existing_workflow(
            profile=profile,
            workflow=workflow,
            roles=[action_role],
            task_type="action_design",
            deliverable="动作结构",
            instruction="基于上游方向判断，拆动作结构和冲突节奏。",
            return_to=profile.dispatcher_role,
            current_node=f"waiting_for_{action_role.role_id}",
            next_action=f"wait_for_{action_role.role_id}_return",
            upstream_role=upstream_role,
            upstream_summary=upstream_summary,
            deliverable_format="动作结构和冲突节奏",
        )
        result.update(
            {
                "intent": "followup_from_upstream",
                "upstream_task_id": upstream_task_id,
                "upstream_role": upstream_role,
            }
        )
        return result

    return {
        "handled": True,
        "result": "workflow_delivery_waiting_dispatcher",
        "profile": profile,
        "workflow": workflow,
        "summary": (
            "上游结果已经回来了。\n\n"
            "我先收一下他的判断，当前等待 PM 决定下一步派给谁。"
        ),
        "message_events": [],
        "tasks": [],
        "upstream_task_id": upstream_task_id,
        "upstream_role": upstream_role,
        "upstream_summary": upstream_summary,
    }


def dispatch_pm_message(text: str, profile: Optional[WorkflowProfile] = None) -> Dict[str, Any]:
    """Return a dispatch decision for the minimal PM workflow path."""
    if profile is None or not is_pm_dispatch_message(text):
        return {"handled": False}
    body = _strip_pm_prefix(text)

    if body in {"继续", "继续。", "继续！"}:
        active = task_state.get_active_workflow(profile.profile_id)
        if not active or not active.get("next_action"):
            return {
                "handled": True,
                "result": "need_judgement",
                "reason": "没有明确当前工作流，不能盲目继续",
                "message_events": [],
                "tasks": [],
            }
        return {
            "handled": True,
            "result": "continue_dispatch",
            "workflow": active,
            "next_action": active.get("next_action"),
            "validator": "pass",
            "message_events": [],
            "tasks": [],
        }

    separate_roles = _parse_separate_targets(body, profile)
    if separate_roles:
        result = _make_workflow_with_tasks(
            profile=profile,
            title=body[:80] or "多角色分别派单",
            roles=separate_roles,
            task_type="parallel_specialist_dispatch",
            deliverable="专项意见",
            instruction="请按你的专业职责给出专项处理意见，不要合并到其他角色回复里。",
            return_to=profile.dispatcher_role,
            current_node="waiting_for_" + "_".join(role.role_id for role in separate_roles),
            next_action="wait_for_parallel_returns",
        )
        result.update(
            {
                "intent": "multi_target_separate_dispatch",
                "target_roles": [role.role_id for role in separate_roles],
                "merged_message": False,
            }
        )
        return result

    direction_role = profile.find_role_by_capability("direction_decision")
    if direction_role:
        result = _make_workflow_with_tasks(
            profile=profile,
            title=body[:80] or "粗颗粒创作需求",
            roles=[direction_role],
            task_type="direction_decision",
            deliverable="方向判断",
            instruction=body or "请先判断创作方向和下一步归口。",
            return_to=profile.dispatcher_role,
            current_node=f"waiting_for_{direction_role.role_id}",
            next_action=f"wait_for_{direction_role.role_id}_return",
        )
        result.update(
            {
                "intent": "rough_creative_request",
                "matched_capability": "direction_decision",
                "matched_role": direction_role.role_id,
            }
        )
        return result

    return {
        "handled": True,
        "result": "need_judgement",
        "reason": "workflow profile 中没有 direction_decision capability，不能派发。",
        "message_events": [],
        "tasks": [],
    }
