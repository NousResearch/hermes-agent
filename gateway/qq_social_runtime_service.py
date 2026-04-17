"""Shared runtime helpers for QQ oral social-control shortcuts."""

from __future__ import annotations

from typing import Any, Callable

from gateway.qq_intents import (
    _looks_like_qq_social_policy_candidate,
    _looks_like_qq_social_request_list_query,
)
from gateway.qq_social_control_requests import (
    looks_like_qq_social_policy_query,
    match_qq_social_control_request,
    match_qq_social_request_type,
    qq_social_policy_notify_target,
)


def match_admin_qq_social_control_request(
    *,
    source: Any,
    body: str,
    admin_ids_configured: bool,
    is_admin_user: bool,
    admin_only_message: str,
) -> tuple[dict[str, Any] | None, str | None]:
    normalized_body = str(body or "").strip()
    if source is None or not normalized_body:
        return None, None
    return match_qq_social_control_request(
        source=source,
        body=normalized_body,
        admin_ids_configured=admin_ids_configured,
        is_admin_user=is_admin_user,
        admin_only_message=admin_only_message,
        looks_like_request_list_query=_looks_like_qq_social_request_list_query,
        looks_like_policy_candidate=_looks_like_qq_social_policy_candidate,
        looks_like_policy_query=looks_like_qq_social_policy_query,
        request_type_matcher=match_qq_social_request_type,
        notify_target_resolver=qq_social_policy_notify_target,
    )


def format_admin_qq_social_control_reply(tool_args: dict[str, Any], result: dict[str, Any]) -> str:
    action = str(tool_args.get("action") or "").strip().lower()
    if action == "list_requests":
        requests = list(result.get("requests") or [])
        request_type = str(tool_args.get("request_type") or "").strip().lower()
        if not requests:
            if request_type == "friend":
                return "当前没有待处理的 QQ 好友申请。"
            if request_type == "group":
                return "当前没有待处理的 QQ 加群/邀请申请。"
            return "当前没有待处理的 QQ 社交申请。"

        if request_type == "friend":
            lines = ["当前待处理的 QQ 好友申请："]
        elif request_type == "group":
            lines = ["当前待处理的 QQ 加群/邀请申请："]
        else:
            lines = ["当前待处理的 QQ 社交申请："]
        for item in requests[:10]:
            if not isinstance(item, dict):
                continue
            key = str(item.get("request_key") or "").strip()
            user_id = str(item.get("user_id") or "").strip()
            group_id = str(item.get("group_id") or "").strip()
            comment = str(item.get("comment") or "").strip()
            line = f"- {key}"
            if user_id:
                line += f" | 用户 {user_id}"
            if group_id:
                line += f" | 群 {group_id}"
            if comment:
                line += f" | 备注：{comment}"
            lines.append(line)
        return "\n".join(lines)

    policy = result.get("policy") or {}
    lines = ["QQ 社交自动处理策略已更新：" if action == "set_social_policy" else "QQ 社交自动处理策略："]
    enabled_label = "已开启" if action == "set_social_policy" else "开"
    disabled_label = "已关闭" if action == "set_social_policy" else "关"
    lines.append(
        f"- 好友申请自动通过：{enabled_label if bool(policy.get('auto_approve_friend_requests')) else disabled_label}"
    )
    lines.append(
        f"- 加群申请自动通过：{enabled_label if bool(policy.get('auto_approve_group_add_requests')) else disabled_label}"
    )
    lines.append(
        f"- 群邀请自动通过：{enabled_label if bool(policy.get('auto_approve_group_invites')) else disabled_label}"
    )
    notify_target = str(policy.get("notify_target") or "").strip()
    if notify_target:
        lines.append(f"- 通知目标：{notify_target}")
    return "\n".join(lines)


def run_admin_qq_social_control_shortcut(
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
        logger.warning("Admin QQ social shortcut failed: %s", exc)
        return f"QQ 社交控制执行失败：{exc}"

    if result.get("error"):
        return str(result["error"])
    return reply_formatter(tool_args, result)
