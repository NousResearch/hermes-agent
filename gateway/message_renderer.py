"""User-facing rendering and filtering for workflow dispatch."""

from __future__ import annotations

import os
import re
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import yaml


_INTERNAL_LINE_PATTERNS = (
    re.compile(r"\bhermes\s+(-|[a-z])", re.IGNORECASE),
    re.compile(r"\.hermes(?:/|\\)", re.IGNORECASE),
    re.compile(r"\bworkflow_profiles(?:/|\\)", re.IGNORECASE),
    re.compile(r"\bworkflow_state(?:/|\\)", re.IGNORECASE),
    re.compile(r"\brelay\s+inbox\b", re.IGNORECASE),
    re.compile(r"\bcross_bot_relay\b", re.IGNORECASE),
    re.compile(r"\bprofile\s+(?:path|file)\b", re.IGNORECASE),
    re.compile(r"\btool\.started\b", re.IGNORECASE),
    re.compile(r"\btool\s+args\b", re.IGNORECASE),
    re.compile(r"\bterminal\s+args\b", re.IGNORECASE),
    re.compile(r"\bterminal\s+command\b", re.IGNORECASE),
    re.compile(r"\bpython\s+traceback\b", re.IGNORECASE),
    re.compile(r"\btraceback\b", re.IGNORECASE),
    re.compile(r"\bcurrent_tool\b", re.IGNORECASE),
    re.compile(r"\biteration\s*[:=]?\s*\d+(?:\s*/\s*\d+)?\b", re.IGNORECASE),
    re.compile(r"\b(?:cd|source|python(?:3)?|pytest|git|npm|pnpm|yarn|uv)\b.+(?:&&| -m |/|\\)", re.IGNORECASE),
    re.compile(r'"\s*(?:command|args|current_tool|iteration|traceback|terminal|tool_name|tool)\s*"\s*:', re.IGNORECASE),
    re.compile(r"^\s*File\s+\".*\",\s+line\s+\d+", re.IGNORECASE),
    re.compile(r"^\s*[A-Za-z_][\w.]*?(?:Error|Exception):\s+"),
    re.compile(r"正在执行命令"),
    re.compile(r"\b(?:waiting_for|returned_to)_[a-z0-9_]+\b", re.IGNORECASE),
    re.compile(r"\brelay_enqueued\b", re.IGNORECASE),
    re.compile(r"\breview_required\s*=", re.IGNORECASE),
    re.compile(r"\bdeliver(?:_after_review)?_to_role\s*=", re.IGNORECASE),
)

_DIRECTION_BULLETS = (
    "这段片子走什么气质",
    "关系冲突是什么",
    "哪些方向不要碰",
)


class _FormatVars(dict):
    def __missing__(self, key: str) -> str:
        return ""


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _template_path() -> Path:
    configured = os.getenv("HERMES_WORKFLOW_REPLY_TEMPLATE_PATH", "").strip()
    return Path(configured).expanduser() if configured else _repo_root() / "workflow_reply_templates.yaml"


@lru_cache(maxsize=8)
def _load_reply_templates(path_text: str = "") -> Dict[str, Any]:
    path = Path(path_text).expanduser() if path_text else _template_path()
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except FileNotFoundError:
        return {}
    if not isinstance(payload, dict):
        return {}
    return payload


def _render_template(
    event_type: str,
    variables: Dict[str, Any],
    *,
    display_mode: str = "human",
) -> str:
    payload = _load_reply_templates(str(_template_path()))
    templates = payload.get("templates") if isinstance(payload.get("templates"), dict) else {}
    entry = templates.get(event_type) if isinstance(templates, dict) else None
    if isinstance(entry, dict):
        template = entry.get(str(display_mode or "human").strip().lower()) or entry.get("human")
    else:
        template = entry
    if not isinstance(template, str) or not template.strip():
        return ""
    values = _FormatVars({key: "" if value is None else str(value) for key, value in variables.items()})
    return template.format_map(values).strip()


def render_workflow_template(
    event_type: str,
    variables: Dict[str, Any],
    *,
    display_mode: str = "human",
) -> str:
    return _render_template(event_type, variables, display_mode=display_mode)


def _render_snippet(name: str, variables: Dict[str, Any]) -> str:
    payload = _load_reply_templates(str(_template_path()))
    snippets = payload.get("snippets") if isinstance(payload.get("snippets"), dict) else {}
    template = snippets.get(name) if isinstance(snippets, dict) else None
    if not isinstance(template, str) or not template.strip():
        return ""
    values = _FormatVars({key: "" if value is None else str(value) for key, value in variables.items()})
    return template.format_map(values).strip()


def sanitize_user_visible_message(text: str, *, fallback: Optional[str] = None) -> Optional[str]:
    """Remove command/tool details from text destined for chat display.

    Returns ``None`` when every line is internal and no safe fallback is provided.
    This lets Feishu progress senders skip backend-only updates instead of
    replacing them with a misleading user-visible message.
    """
    lines: List[str] = []
    removed = False
    in_traceback = False
    for line in str(text or "").splitlines():
        if "Traceback (most recent call last)" in line:
            in_traceback = True
            removed = True
            continue
        if in_traceback:
            if line.strip() and not line.startswith((" ", "\t")) and not re.match(r"^[A-Za-z_][\w.]*?(?:Error|Exception):", line):
                in_traceback = False
            else:
                removed = True
                continue
        if any(pattern.search(line) for pattern in _INTERNAL_LINE_PATTERNS):
            removed = True
            continue
        lines.append(line)
    cleaned = "\n".join(lines).strip()
    if cleaned:
        return cleaned
    if removed and fallback:
        return fallback
    return None


def render_validation_errors(errors: Iterable[Dict[str, Any]]) -> str:
    parts = ["派单已拦截："]
    for error in errors:
        parts.append(f"- {error.get('message') or error.get('code')}")
    return "\n".join(parts)


def _normalize_role_id(value: Any) -> str:
    return str(value or "").strip().lower()


def _role_name(profile: Any, role_id: Any) -> str:
    normalized = _normalize_role_id(role_id)
    if not normalized:
        return ""
    try:
        role = profile.get_role(normalized)
    except Exception:
        role = None
    return str(getattr(role, "name", "") or normalized.upper()).strip()


def _task_reviewer_text(profile: Any, task: Dict[str, Any]) -> str:
    reviewer_role = _normalize_role_id(task.get("reviewer_role"))
    deliver_to_role = _normalize_role_id(
        task.get("deliver_after_review_to_role")
        or task.get("deliver_to_role")
        or task.get("return_to")
        or getattr(profile, "dispatcher_role", "")
    )
    deliver_to = _role_name(profile, deliver_to_role) or "PM"
    if not reviewer_role:
        return _render_snippet("no_review", {"deliver_to_name": deliver_to})
    reviewer = _role_name(profile, reviewer_role)
    return _render_snippet(
        "review_flow",
        {"reviewer_name": reviewer, "deliver_to_name": deliver_to},
    )


def _task_action(task: Dict[str, Any]) -> str:
    task_type = str(task.get("task_type") or "").strip()
    deliverable = str(task.get("deliverable") or "").strip()
    if task_type == "direction_decision" or "方向" in deliverable:
        return "定方向"
    if task_type == "parallel_specialist_dispatch":
        return "给一版专项判断"
    return deliverable or "处理这一段"


def _task_focus_lines(task: Dict[str, Any]) -> List[str]:
    task_type = str(task.get("task_type") or "").strip()
    if task_type == "action_design":
        return [
            "动作结构怎么起承转合",
            "肢体冲突和关系冲突怎么咬住",
            "节奏上哪些地方需要收、哪些地方需要放",
        ]
    if task_type == "direction_decision":
        return list(_DIRECTION_BULLETS)
    return [
        "基于自己的专业职责先判断这一段怎么成立",
        "把你认为最关键的处理意见说清楚",
        "如果后续需要别人接力，也顺手指出来",
    ]


def _task_focus_text(task: Dict[str, Any]) -> str:
    return "\n".join(f"- {line}" for line in _task_focus_lines(task))


def _debug_task_message(profile: Any, role: Any, task: Dict[str, Any]) -> str:
    return "\n".join(
        [
            f"task_id：{task.get('task_id')}",
            f"执行者：{getattr(role, 'role_id', '')}",
            f"任务：{task.get('instruction')}",
            f"上游输入：{task.get('upstream_summary') or '无'}",
            f"验收者：{task.get('reviewer_role') or '无'}",
            f"完成后交付给：{task.get('deliver_to_role') or task.get('return_to')}",
            f"交付格式：{task.get('deliverable_format') or task.get('deliverable')}",
            f"当前状态：{task.get('status')}",
        ]
    )


def render_task_message(
    profile: Any,
    role: Any,
    task: Dict[str, Any],
    *,
    display_mode: str = "human",
) -> str:
    """Render one outbound workflow task for chat.

    The task object still carries structured fields; this function only controls
    what humans see in Feishu.
    """
    if str(display_mode or "human").strip().lower() == "debug":
        return _debug_task_message(profile, role, task)

    role_name = str(getattr(role, "name", "") or getattr(role, "role_id", "") or "这位同事").strip()
    instruction = str(task.get("instruction") or "").strip()
    upstream_summary = str(task.get("upstream_summary") or "").strip()
    upstream_role = _role_name(profile, task.get("upstream_role")) or "上游"

    if upstream_summary:
        context = (
            f"前面{upstream_role}已经给过一轮判断：\n"
            f"{upstream_summary}\n\n"
            "你这一步基于这个结果继续往下拆。"
        )
    else:
        context = f"用户这次的需求是：{instruction or '先判断这一轮需求该怎么推进'}。"

    return _render_template(
        "dispatch_to_role",
        {
            "task_id": task.get("task_id"),
            "target_role": task.get("to_role"),
            "target_role_name": role_name,
            "task_type": task.get("task_type"),
            "task_goal": _task_action(task),
            "user_request": context,
            "focus_points": _task_focus_text(task),
            "reviewer_name": _role_name(profile, task.get("reviewer_role")),
            "deliver_to_name": _role_name(profile, task.get("deliver_to_role") or task.get("return_to")) or "PM",
            "review_flow": _task_reviewer_text(profile, task),
            "status": task.get("status"),
        },
        display_mode=display_mode,
    )


def render_task_completion_delivery(
    profile: Any,
    task: Dict[str, Any],
    *,
    from_role: str,
    to_role: str,
    result_summary: str,
) -> str:
    from_name = _role_name(profile, from_role) or str(from_role or "上游").upper()
    to_name = _role_name(profile, to_role) or str(to_role or "PM").upper()
    deliverable = str(task.get("deliverable") or "这一步").strip()
    return _render_template(
        "role_result_returned",
        {
            "from_role": from_role,
            "from_role_name": from_name,
            "to_role": to_role,
            "deliver_to_name": to_name,
            "task_goal": deliverable,
            "upstream_summary": str(result_summary or "").strip() or "他已经给出了一轮判断。",
            "upstream_task_id": task.get("task_id"),
            "workflow_id": task.get("workflow_id"),
        },
    )


def render_review_request_message(
    profile: Any,
    task: Dict[str, Any],
    *,
    from_role: str,
    reviewer_role: str,
    result_summary: str,
) -> str:
    from_name = _role_name(profile, from_role) or str(from_role or "执行者").upper()
    reviewer_name = _role_name(profile, reviewer_role) or str(reviewer_role or "验收者").upper()
    deliver_to_role = task.get("deliver_after_review_to_role") or task.get("deliver_to_role") or task.get("return_to")
    deliver_to = _role_name(profile, deliver_to_role) or "PM"
    return _render_template(
        "review_request",
        {
            "from_role": from_role,
            "from_role_name": from_name,
            "reviewer_role": reviewer_role,
            "reviewer_name": reviewer_name,
            "deliver_to_role": deliver_to_role,
            "deliver_to_name": deliver_to,
            "upstream_summary": str(result_summary or "").strip() or "他已经给出了一轮处理结果。",
        },
    )


def _debug_dispatch_result(result: Dict[str, Any]) -> str:
    tasks = result.get("tasks") or []
    lines: List[str] = []
    for task in tasks:
        lines.append(
            " / ".join(
                [
                    f"task_id={task.get('task_id')}",
                    f"to_role={task.get('to_role')}",
                    f"status={task.get('status')}",
                    f"deliver_to_role={task.get('deliver_to_role') or task.get('return_to')}",
                ]
            )
        )
    if result.get("next_state"):
        lines.append(f"next_state={result.get('next_state')}")
    return "\n".join(lines) or str(result.get("summary") or "已更新协作状态。")


def _human_dispatch_result(result: Dict[str, Any]) -> str:
    if result.get("result") == "need_judgement":
        return str(result.get("reason") or "没有明确当前工作流，不能盲目继续")

    tasks = result.get("tasks") or []
    if not tasks:
        if result.get("result") == "workflow_delivery_waiting_dispatcher":
            profile = result.get("profile")
            return _render_template(
                "role_result_returned",
                {
                    "from_role": result.get("upstream_role"),
                    "from_role_name": _role_name(profile, result.get("upstream_role")) or "上游",
                    "upstream_task_id": result.get("upstream_task_id"),
                    "upstream_summary": str(result.get("upstream_summary") or "").strip() or "他已经给出了一轮判断。",
                    "to_role": getattr(profile, "dispatcher_role", "pm") if profile else "pm",
                    "deliver_to_name": _role_name(profile, getattr(profile, "dispatcher_role", "pm")) if profile else "PM",
                },
            )
        return str(result.get("summary") or "已更新协作状态。")

    profile = result.get("profile")
    role_names = [_role_name(profile, task.get("to_role")) for task in tasks]
    role_names = [name for name in role_names if name]

    if result.get("intent") == "rough_creative_request" and role_names:
        return _render_template(
            "workflow_waiting",
            {
                "target_role": tasks[0].get("to_role"),
                "target_role_name": role_names[0],
                "task_type": tasks[0].get("task_type"),
                "task_goal": "定一下方向",
                "focus_points": "\n".join(f"- {line}" for line in _DIRECTION_BULLETS),
                "review_flow": _task_reviewer_text(profile, tasks[0]),
                "next_state": result.get("next_state"),
            },
        )

    if len(role_names) > 1:
        joined = "、".join(role_names)
        return (
            f"我会分别交给{joined}，每个人单独看自己的部分。\n\n"
            "他们会各自给专项意见，不会合并成一条任务。\n"
            f"{_task_reviewer_text(profile, tasks[0])}\n"
            "我等这些结果回来后，再继续整合。"
        )

    role_name = role_names[0] if role_names else "对应角色"
    if result.get("intent") == "followup_from_upstream":
        upstream_role = _role_name(profile, tasks[0].get("upstream_role")) or "上游"
        upstream_summary = str(tasks[0].get("upstream_summary") or "").strip()
        return _render_template(
            "next_handoff",
            {
                "from_role": tasks[0].get("upstream_role"),
                "from_role_name": upstream_role,
                "next_role": tasks[0].get("to_role"),
                "next_role_name": role_name,
                "target_role_name": role_name,
                "upstream_task_id": result.get("upstream_task_id"),
                "upstream_summary": upstream_summary or "他已经给出了一轮判断。",
                "review_flow": _task_reviewer_text(profile, tasks[0]),
            },
        )

    return (
        f"我先交给{role_name}处理这一段。\n"
        f"{_task_reviewer_text(profile, tasks[0])}\n"
        "我等他回来后，再判断下一步。"
    )


def render_dispatch_result(
    result: Dict[str, Any],
    *,
    display_mode: str = "human",
) -> str:
    if str(display_mode or "human").strip().lower() == "debug":
        return _debug_dispatch_result(result)
    return _human_dispatch_result(result)
