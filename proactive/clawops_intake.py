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
from proactive.hubops_routing import route_clawops_objective


DEFAULT_ASSIGNEE = "default"
DEFAULT_CREATED_BY = "hermes-clawops-intake"
DEFAULT_MAX_RUNTIME_SECONDS = 1800
KNOWN_PROJECTS = {"hub_ops", "hahow_course", "course_marketing", "secondhand_commerce", "ingrids_marketing"}


EXTERNAL_BROWSER_TARGET_TERMS = (
    "facebook",
    "fb marketplace",
    "marketplace",
    "社團",
    "交流團",
    "group",
    "商品表單",
)
EXTERNAL_BROWSER_ACTION_TERMS = (
    "刊登",
    "發布",
    "發佈",
    "貼文",
    "post",
    "publish",
    "listing",
    "join",
    "加入",
    "next",
    "上傳",
    "照片",
    "檢查",
    "核對",
    "只讀",
)
AUTO_PUBLISH_APPROVAL_TERMS = (
    "已確認",
    "已核准",
    "已同意",
    "已經傳給我確認",
    "文案已確認",
    "文案已核准",
    "之前發佈文案",
    "之前發布文案",
    "自動發佈",
    "自動發布",
    "自動 publish",
    "auto publish",
    "approved copy",
    "preapproved",
)


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

    enriched_source = infer_clawops_metadata(clean_objective, source=source)
    hubops_envelope = _route_hubops_if_requested(clean_objective, enriched_source)
    if hubops_envelope and hubops_envelope.get("status") == "blocked":
        raise ValueError(str(hubops_envelope.get("blocked_reason") or "HubOps routing blocked this task."))

    assigned_worker = ""
    runtime_profile = ""
    if hubops_envelope:
        assignment = hubops_envelope.get("assignment")
        if isinstance(assignment, Mapping):
            assigned_worker = str(assignment.get("assigned_worker") or "").strip()
            runtime_profile = str(assignment.get("runtime_profile") or "").strip()

    configured_assignee = (assignee or "").strip() or resolve_clawops_assignee(config)
    resolved_assignee = (
        configured_assignee
        if configured_assignee and configured_assignee != DEFAULT_ASSIGNEE
        else runtime_profile or assigned_worker or configured_assignee
    )
    title = _title_from_objective(clean_objective)
    body = _body_from_objective(clean_objective, source=enriched_source, hubops_envelope=hubops_envelope)

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
    hubops_envelope: Optional[Mapping[str, Any]] = None,
) -> str:
    needs_external_browser = requires_external_browser_capabilities(objective, source=source)
    publish_preapproved = auto_publish_preapproved(objective, source=source)
    execution_owner = (
        "Browser-capable Hermes/ClawOps runtime may execute only the delegated browser work in this queued task."
        if needs_external_browser and publish_preapproved
        else "ClawOps/OpenClaw may execute only delegated work in this queued task."
    )
    lines = [
        "ClawOps runtime task created by Hermes.",
        "",
        "Control boundary:",
        "- Hermes remains the primary agent and user-facing decision owner.",
        f"- {execution_owner}",
        "- Results must return to Hermes for audit, review, and user-facing summary.",
        "",
        "Objective:",
        objective,
    ]
    if hubops_envelope:
        lines.extend(_hubops_routing_contract(hubops_envelope))
    if needs_external_browser:
        lines.extend(
            _external_browser_capability_contract_for(
                auto_publish_preapproved=publish_preapproved,
            )
        )
    if source:
        lines.extend(["", "Source:"])
        for key in sorted(source):
            value = source.get(key)
            if value is None or value == "":
                continue
            lines.append(f"- {key}: {value}")
    return "\n".join(lines)


def _route_hubops_if_requested(
    objective: str,
    source: Optional[Mapping[str, Any]],
) -> Optional[dict[str, Any]]:
    if not source or not any(key in source for key in ("project", "task_type", "risk_level", "approved")):
        return None
    return route_clawops_objective(
        objective,
        project=str(source.get("project") or "hub_ops"),
        task_type=str(source.get("task_type") or "ops"),
        risk_level=str(source.get("risk_level") or "low"),
        approved=_read_bool(source.get("approved")),
    )


def infer_clawops_metadata(
    objective: str,
    *,
    source: Optional[Mapping[str, Any]] = None,
) -> dict[str, Any]:
    """Infer project/task metadata so /clawops natural language uses YAML routing."""
    inferred: dict[str, Any] = dict(source or {})
    haystack = " ".join(
        [objective or "", *[str(v) for v in inferred.values() if v is not None]]
    ).lower()

    if not str(inferred.get("project") or "").strip():
        inferred["project"] = _infer_project(haystack)
    if not str(inferred.get("task_type") or "").strip():
        inferred["task_type"] = _infer_task_type(haystack, str(inferred.get("project") or ""))
    if not str(inferred.get("risk_level") or "").strip():
        inferred["risk_level"] = "medium" if inferred.get("task_type") == "browser_publish" else "low"
    if auto_publish_preapproved(objective, source=inferred):
        inferred.setdefault("auto_publish_preapproved", "true")
        inferred.setdefault("previous_copy_confirmed", "true")
        inferred.setdefault("approved", "true")
    return inferred


def _infer_project(haystack: str) -> str:
    if any(term in haystack for term in ("二手", "secondhand", "咖啡機", "facebook", "marketplace", "社團", "商品")):
        return "secondhand_commerce"
    if any(term in haystack for term in ("hahow", "課綱", "課程大綱", "課程設計", "proposal")):
        return "hahow_course"
    if any(term in haystack for term in ("招生", "crm", "lead", "名單", "課程行銷", "course marketing")):
        return "course_marketing"
    if any(term in haystack for term in ("ingrids", "seo", "trading", "投資", "交易")):
        return "ingrids_marketing"
    if any(term in haystack for term in ("openclaw", "hermes", "bridge", "health", "runtime", "kanban", "hubops", "clawops")):
        return "hub_ops"
    return "hub_ops"


def _infer_task_type(haystack: str, project: str) -> str:
    if requires_external_browser_capabilities(haystack):
        return "browser_publish" if any(term in haystack for term in ("發佈", "發布", "刊登", "post", "publish", "listing")) else "browser_ops"
    if any(term in haystack for term in ("health", "bridge", "runtime", "修正", "fix", "deploy", "部署")):
        return "devops"
    if project == "ingrids_marketing":
        return "product_marketing"
    if any(term in haystack for term in ("研究", "research", "分析", "競品")):
        return "research"
    if any(term in haystack for term in ("內容", "文案", "draft", "草稿")):
        return "content_draft"
    if project == "hahow_course":
        return "course_design"
    if project == "course_marketing" and any(term in haystack for term in ("crm", "名單", "私訊", "followup")):
        return "crm"
    if project == "course_marketing":
        return "campaign"
    return "ops"


def _read_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "approved"}
    return False


def _hubops_routing_contract(envelope: Mapping[str, Any]) -> list[str]:
    assignment = envelope.get("assignment")
    assignment = assignment if isinstance(assignment, Mapping) else {}
    agent_assignment = envelope.get("agent_assignment")
    agent_assignment = agent_assignment if isinstance(agent_assignment, Mapping) else {}
    return [
        "",
        "HubOps routing:",
        f"- status: {envelope.get('status', '')}",
        f"- assigned_agent: {agent_assignment.get('assigned_agent', '')}",
        f"- assigned_worker: {assignment.get('assigned_worker', '')}",
        f"- runtime_profile: {assignment.get('runtime_profile', '')}",
        f"- risk_level_limit: {assignment.get('risk_level_limit', '')}",
        f"- approval_required: {assignment.get('approval_required', '')}",
        f"- approval_checklist: {envelope.get('approval_checklist', '')}",
        f"- output_schema: {envelope.get('output_schema', {})}",
    ]


def requires_external_browser_capabilities(
    objective: str,
    *,
    source: Optional[Mapping[str, Any]] = None,
) -> bool:
    """Return True for delegated work that must run in ClawOps browser mode."""
    haystack_parts = [objective or ""]
    if source:
        haystack_parts.extend(str(v) for v in source.values() if v is not None)
    haystack = " ".join(haystack_parts).lower()
    return any(term in haystack for term in EXTERNAL_BROWSER_TARGET_TERMS) and any(
        term in haystack for term in EXTERNAL_BROWSER_ACTION_TERMS
    )


def _external_browser_capability_contract() -> list[str]:
    return _external_browser_capability_contract_for(auto_publish_preapproved=False)


def _external_browser_capability_contract_for(
    *,
    auto_publish_preapproved: bool,
) -> list[str]:
    final_action_policy = (
        "- Approved-copy auto-publish: if the task uses the exact Hermes-confirmed "
        "copy/assets and the page state matches that approved payload, the worker may "
        "click final Post/Publish/Submit without asking KJ again."
        if auto_publish_preapproved
        else "- For Facebook listing flows without explicit final-publish approval, stop before Post/Publish/Submit and report page state, remaining required fields, and visible final action buttons."
    )
    if auto_publish_preapproved:
        execution_boundary = [
            "- This task must be executed by the browser-capable Hermes/ClawOps runtime with audit evidence.",
            "- Do not route this task through the OpenClaw dry-run bridge; that bridge cannot execute Facebook/external browser side effects.",
        ]
    else:
        execution_boundary = [
            "- This task must be executed by ClawOps/OpenClaw, not by Hermes directly.",
            "- Hermes must not perform this browser UI work directly; Hermes only owns intake, approvals, monitoring, and final user-facing summary.",
        ]

    return [
        "",
        "External browser capability contract:",
        *execution_boundary,
        "- Required capabilities: logged-in browser CDP session via BROWSER_CDP_URL, browser_cdp, browser_snapshot/browser_navigate as needed, and browser_upload_files for local file inputs.",
        "- If BROWSER_CDP_URL is absent, Facebook login/checkpoint appears, browser_upload_files is unavailable, or required local files cannot be accessed, call kanban_block with block_kind=capability and a concrete missing-capability reason.",
        "- Safety boundary: do not Join groups, send messages, comment, pay, promote, or change account settings unless the objective contains explicit user approval for that exact action.",
        final_action_policy,
        "- If visible Facebook content differs from the confirmed copy/assets, required fields are ambiguous, or the flow adds new destinations not covered by the task, stop and call kanban_block with block_kind=approval.",
        "- After an automatic publish/post, capture URL/screenshot/page state and report the published destination, timestamp, visible status, and any Facebook warning or review state.",
    ]


def auto_publish_preapproved(
    objective: str,
    *,
    source: Optional[Mapping[str, Any]] = None,
) -> bool:
    """Return True only when the task explicitly carries prior copy approval."""
    if source:
        for key in (
            "auto_publish_preapproved",
            "previous_copy_confirmed",
            "approved_copy",
            "copy_approved",
            "final_publish_approved",
        ):
            if _read_bool(source.get(key)):
                return True

    haystack_parts = [objective or ""]
    if source:
        haystack_parts.extend(str(v) for v in source.values() if v is not None)
    haystack = " ".join(haystack_parts).lower()
    return any(term.lower() in haystack for term in AUTO_PUBLISH_APPROVAL_TERMS)
