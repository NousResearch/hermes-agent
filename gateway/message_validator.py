"""Validation rules for workflow dispatch message events."""

from __future__ import annotations

import re
from typing import Any, Dict, List

from gateway.workflow_profile import WorkflowProfile


_INTERNAL_PATTERNS = (
    re.compile(r"\bhermes\s+(-|[a-z])", re.IGNORECASE),
    re.compile(r"\btool\.started\b", re.IGNORECASE),
    re.compile(r"\bterminal\s+args\b", re.IGNORECASE),
    re.compile(r"\bpython\s+traceback\b", re.IGNORECASE),
    re.compile(r"正在执行命令"),
)


def _error(code: str, message: str) -> Dict[str, str]:
    return {"code": code, "message": message}


def _addressed_roles(content: str, profile: WorkflowProfile) -> List[str]:
    """Only the first addressed line determines which role the message drives."""
    for line in (content or "").splitlines():
        stripped = line.strip()
        if stripped:
            return [role.role_id for role in profile.find_roles_in_text(stripped)]
    return []


def validate_message_event(event: Dict[str, Any], profile: WorkflowProfile) -> Dict[str, Any]:
    """Validate a single outbound workflow message event."""
    errors: List[Dict[str, str]] = []
    content = str(event.get("content") or "")
    metadata = event.get("metadata") if isinstance(event.get("metadata"), dict) else {}
    to_role = str(event.get("to_role") or metadata.get("to_role") or "").strip().lower()

    if not to_role:
        errors.append(_error("MISSING_TARGET_ROLE", "缺少目标角色。"))
    elif not profile.get_role(to_role):
        errors.append(_error("UNKNOWN_TARGET_ROLE", f"目标角色不在 workflow profile 中：{to_role}"))

    distinct_roles = set(_addressed_roles(content, profile))
    if len(distinct_roles) > 1:
        errors.append(
            _error(
                "MULTI_TARGET_IN_ONE_MESSAGE",
                "检测到多个目标角色被合并在同一条消息中，必须拆成独立 message event。",
            )
        )

    if not str(event.get("deliverable") or metadata.get("deliverable") or "").strip():
        errors.append(_error("MISSING_DELIVERABLE", "派单缺 deliverable。"))

    if profile.message_rules.get("require_return_to", True):
        if not str(event.get("return_to") or metadata.get("return_to") or "").strip():
            errors.append(_error("MISSING_RETURN_TO", "派单缺 return_to。"))

    dispatcher = profile.dispatcher_role
    reviewer = str(metadata.get("reviewer_role") or metadata.get("final_reviewer") or "").strip().lower()
    task_type = str(event.get("task_type") or metadata.get("task_type") or "").strip().lower()
    deliverable = str(event.get("deliverable") or metadata.get("deliverable") or "").strip()
    if reviewer == dispatcher or (
        to_role == dispatcher and ("review" in task_type or "审核" in deliverable or "验收" in deliverable)
    ):
        errors.append(_error("DISPATCHER_CANNOT_REVIEW", "PM / dispatcher 不得作为内容审核人。"))

    send_mode = str(metadata.get("send_mode") or event.get("send_mode") or "real_send").strip()
    if bool(event.get("real_sent") or metadata.get("real_sent")) and send_mode != "real_send":
        errors.append(_error("DRAFT_MARKED_REAL_SENT", "只写文案不能标记为 real_sent。"))

    for pattern in _INTERNAL_PATTERNS:
        if pattern.search(content):
            errors.append(_error("INTERNAL_COMMAND_EXPOSED", "消息中包含底层命令或工具日志，不能发送给飞书用户。"))
            break

    if str(event.get("status") or metadata.get("status") or "").strip() == "completed":
        if not reviewer and not metadata.get("next_node"):
            errors.append(_error("COMPLETED_WITHOUT_REVIEW_OR_NEXT", "缺 final reviewer 或下一步节点时不能标记完成。"))

    return {"ok": not errors, "errors": errors}


def validate_message_events(events: List[Dict[str, Any]], profile: WorkflowProfile) -> Dict[str, Any]:
    errors: List[Dict[str, Any]] = []
    if len(events) > 1:
        for event in events:
            content = str(event.get("content") or "")
            if len(set(_addressed_roles(content, profile))) > 1:
                errors.append(
                    _error("MULTI_TARGET_BATCH_NOT_SPLIT", "多角色派单中存在未拆分的 message event。")
                )
                break
    for index, event in enumerate(events):
        result = validate_message_event(event, profile)
        for error in result["errors"]:
            errors.append({**error, "event_index": index})
    return {"ok": not errors, "errors": errors}
