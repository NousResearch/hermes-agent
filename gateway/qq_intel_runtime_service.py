"""Shared runtime helpers for QQ oral intel-worker shortcuts."""

from __future__ import annotations

from typing import Any, Callable, Iterable

from gateway.direct_tool_result_runtime_service import shortcut_tool_failure_text
from gateway.intel_control_request_platform_specs import (
    IntelControlRequestPlatformSpec,
    QQ_INTEL_CONTROL_REQUEST_PLATFORM_SPEC,
)


def match_admin_platform_intel_control_request(
    *,
    source: Any,
    body: str,
    admin_ids_configured: bool,
    is_admin_user: bool,
    looks_like_joined_group_list_query,
    known_worker_names: Iterable[str],
    report_target_resolver,
    request_spec: IntelControlRequestPlatformSpec,
) -> tuple[dict[str, Any] | None, str | None]:
    normalized_body = str(body or "").strip()
    if source is None or not normalized_body:
        return None, None
    return request_spec.request_matcher(
        source=source,
        body=normalized_body,
        admin_ids_configured=admin_ids_configured,
        is_admin_user=is_admin_user,
        looks_like_joined_group_list_query=looks_like_joined_group_list_query,
        extract_worker_name=request_spec.worker_name_extractor,
        looks_like_worker_context=request_spec.worker_context_checker,
        known_worker_names=known_worker_names,
        target_extractor=request_spec.target_extractor,
        report_target_resolver=report_target_resolver,
        hire_objective_extractor=request_spec.hire_objective_extractor,
    )


def match_admin_qq_intel_control_request(
    *,
    source: Any,
    body: str,
    admin_ids_configured: bool,
    is_admin_user: bool,
    looks_like_joined_group_list_query,
    known_worker_names: Iterable[str],
    report_target_resolver,
) -> tuple[dict[str, Any] | None, str | None]:
    return match_admin_platform_intel_control_request(
        source=source,
        body=body,
        admin_ids_configured=admin_ids_configured,
        is_admin_user=is_admin_user,
        looks_like_joined_group_list_query=looks_like_joined_group_list_query,
        known_worker_names=known_worker_names,
        report_target_resolver=report_target_resolver,
        request_spec=QQ_INTEL_CONTROL_REQUEST_PLATFORM_SPEC,
    )


def format_admin_qq_intel_control_reply(
    tool_args: dict[str, Any],
    result: dict[str, Any],
    *,
    status_label_formatter: Callable[[str], str],
    unique_report_targets_fn: Callable[[list[Any]], list[str]],
) -> str:
    action = str(tool_args.get("action") or "").strip().lower()
    if action == "list_joined_groups":
        groups = list(result.get("groups") or [])
        if not groups:
            return "当前还没查到已加入的 QQ 群。"
        lines = ["当前已加入的 QQ 群："]
        for item in groups[:20]:
            if not isinstance(item, dict):
                continue
            group_id = str(item.get("group_id") or "").strip()
            group_name = str(item.get("group_name") or group_id).strip()
            lines.append(f"- {group_name} ({group_id})")
        return "\n".join(lines)

    worker = result.get("worker") or {}
    worker_name = str(worker.get("worker_name") or tool_args.get("worker_name") or "").strip()
    status_label = status_label_formatter(str(worker.get("status") or ""))
    if action == "hire_worker":
        target_group = str(
            worker.get("target_group_id")
            or worker.get("target_group_ref")
            or tool_args.get("target_group")
            or ""
        ).replace("group:", "").strip()
        return f"已安排情报员 {worker_name} 去 QQ 群 {target_group} 执行任务。当前状态：{status_label}。"
    if action == "pause_worker":
        return f"情报员 {worker_name} 已暂停。当前状态：{status_label}。"
    if action == "resume_worker":
        return f"情报员 {worker_name} 已恢复任务。当前状态：{status_label}。"
    if action == "stop_worker":
        return f"情报员 {worker_name} 已停用。当前状态：{status_label}。"
    if action == "run_report_now":
        delivery = str(
            (result.get("delivery") or {}).get("target")
            or tool_args.get("manual_report_target")
            or ""
        ).strip()
        if delivery:
            return f"已让情报员 {worker_name} 立即汇报，发送到 {delivery}。"
        return f"已让情报员 {worker_name} 立即汇报。"

    group_id = str(worker.get("target_group_id") or "").strip()
    group_name = str(worker.get("target_group_name") or "").strip()
    objective = str(worker.get("objective") or "").strip()
    lines = [f"情报员 {worker_name} 当前状态：{status_label}。"]
    if group_id or group_name:
        label = group_name or group_id
        if group_id and group_name and group_id != group_name:
            label = f"{group_name} ({group_id})"
        lines.append(f"目标群：{label}")
    if objective:
        lines.append(f"任务：{objective}")
    daily_targets = unique_report_targets_fn([worker.get("daily_report_target")])
    manual_targets = unique_report_targets_fn([worker.get("manual_report_target")])
    if bool(worker.get("daily_report_enabled")) and daily_targets:
        lines.append(f"日报目标：{', '.join(daily_targets)}")
    if manual_targets:
        lines.append(f"立即汇报目标：{', '.join(manual_targets)}")
    last_error = str(worker.get("last_error") or "").strip()
    if last_error:
        lines.append(f"备注：{last_error}")
    return "\n".join(lines)


def run_admin_qq_intel_control_shortcut(
    *,
    tool_args: dict[str, Any] | None,
    shortcut_error: str | None,
    tool_runner: Callable[[dict[str, Any]], dict[str, Any]],
    reply_formatter: Callable[[dict[str, Any], dict[str, Any]], str],
    logger,
) -> str | None:
    if shortcut_error:
        return shortcut_error
    if not tool_args:
        return None

    try:
        result = tool_runner(tool_args)
    except Exception as exc:
        logger.warning("Admin QQ oral intel control shortcut failed: %s", exc)
        return f"QQ 情报员控制执行失败：{exc}"

    failure = shortcut_tool_failure_text(result, failure_prefix="QQ 情报员控制执行失败")
    if failure:
        return failure
    return reply_formatter(tool_args, result)
