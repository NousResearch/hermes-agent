"""Shared runtime helpers for QQ oral group-moderation shortcuts."""

from __future__ import annotations

from typing import Any, Callable

from gateway.direct_tool_result_runtime_service import shortcut_tool_failure_text
from gateway.group_target_intents import extract_qq_group_target
from gateway.qq_group_moderation_requests import (
    extract_qq_oral_moderation_duration_seconds,
    extract_qq_oral_moderation_reason,
    extract_qq_oral_moderation_user_query,
    match_qq_group_moderation_action,
    match_qq_group_moderation_request,
)


def match_admin_qq_group_moderation_request(
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
    return match_qq_group_moderation_request(
        source=source,
        body=normalized_body,
        admin_ids_configured=admin_ids_configured,
        is_admin_user=is_admin_user,
        admin_only_message=admin_only_message,
        action_matcher=match_qq_group_moderation_action,
        target_extractor=extract_qq_group_target,
        user_query_extractor=extract_qq_oral_moderation_user_query,
        reason_extractor=extract_qq_oral_moderation_reason,
        duration_extractor=extract_qq_oral_moderation_duration_seconds,
    )


def format_admin_qq_group_moderation_reply(tool_args: dict[str, Any], result: dict[str, Any]) -> str:
    action = str(result.get("action") or tool_args.get("action") or "").strip().lower()
    group_id = str(result.get("group_id") or tool_args.get("target") or "").replace("group:", "").strip()
    member_name = str(
        result.get("member_name")
        or tool_args.get("user_query")
        or result.get("user_id")
        or "目标成员"
    ).strip()
    reason = str(result.get("reason") or tool_args.get("reason") or "").strip()
    if action == "mute_user":
        duration_seconds = int(result.get("duration_seconds") or tool_args.get("duration_seconds") or 0)
        line = f"已把 QQ 群 {group_id} 的 {member_name} 禁言 {duration_seconds} 秒。"
    else:
        line = f"已把 QQ 群 {group_id} 的 {member_name} 踢出。"
    if reason:
        line += f" 原因：{reason}。"
    return line


def run_admin_qq_group_moderation_shortcut(
    *,
    tool_args: dict[str, Any] | None,
    shortcut_error: str | None,
    tool_runner: Callable[[dict[str, Any]], dict[str, Any]],
    logger,
) -> str | None:
    if shortcut_error:
        return shortcut_error
    if not tool_args:
        return None

    try:
        result = tool_runner(tool_args)
    except Exception as exc:
        logger.warning("Admin QQ oral moderation shortcut failed: %s", exc)
        return f"QQ 群管理执行失败：{exc}"

    failure = shortcut_tool_failure_text(result, failure_prefix="QQ 群管理执行失败")
    if failure:
        return failure
    return format_admin_qq_group_moderation_reply(tool_args, result)
