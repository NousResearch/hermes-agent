"""Shared runtime helpers for oral group-moderation shortcuts."""

from __future__ import annotations

from typing import Any, Callable

from gateway.direct_control_platform_specs import (
    AdminGroupModerationPlatformSpec,
    QQ_ADMIN_GROUP_MODERATION_SPEC,
)
from gateway.direct_tool_result_runtime_service import shortcut_tool_failure_text
from gateway.group_moderation_request_platform_specs import (
    GroupModerationRequestPlatformSpec,
    QQ_GROUP_MODERATION_REQUEST_PLATFORM_SPEC,
    get_group_moderation_request_platform_spec,
)


def match_admin_platform_group_moderation_request(
    *,
    source: Any,
    body: str,
    admin_ids_configured: bool,
    is_admin_user: bool,
    admin_only_message: str,
    spec: AdminGroupModerationPlatformSpec,
    request_spec: GroupModerationRequestPlatformSpec | None = None,
) -> tuple[dict[str, Any] | None, str | None]:
    normalized_body = str(body or "").strip()
    if source is None or not normalized_body:
        return None, None
    request_spec = request_spec or get_group_moderation_request_platform_spec(spec.platform)
    return request_spec.request_matcher(
        source=source,
        body=normalized_body,
        admin_ids_configured=admin_ids_configured,
        is_admin_user=is_admin_user,
        admin_only_message=admin_only_message,
        action_matcher=request_spec.action_matcher,
        target_extractor=spec.target_extractor,
        user_query_extractor=request_spec.user_query_extractor,
        reason_extractor=request_spec.reason_extractor,
        duration_extractor=request_spec.duration_extractor,
        current_group_target_formatter=spec.current_group_target_formatter,
        missing_target_message=spec.missing_target_message,
    )


def match_admin_qq_group_moderation_request(
    *,
    source: Any,
    body: str,
    admin_ids_configured: bool,
    is_admin_user: bool,
    admin_only_message: str,
) -> tuple[dict[str, Any] | None, str | None]:
    return match_admin_platform_group_moderation_request(
        source=source,
        body=body,
        admin_ids_configured=admin_ids_configured,
        is_admin_user=is_admin_user,
        admin_only_message=admin_only_message,
        spec=QQ_ADMIN_GROUP_MODERATION_SPEC,
        request_spec=QQ_GROUP_MODERATION_REQUEST_PLATFORM_SPEC,
    )


def format_admin_platform_group_moderation_reply(
    tool_args: dict[str, Any],
    result: dict[str, Any],
    *,
    spec: AdminGroupModerationPlatformSpec,
) -> str:
    capability = str(result.get("capability") or "").strip().lower()
    if capability == "not_capable":
        detail = str(result.get("detail") or result.get("message") or "").strip()
        return detail or f"{spec.platform_label}暂不支持禁言/踢人。"

    action = str(result.get("action") or tool_args.get("action") or "").strip().lower()
    group_id = spec.reply_target_normalizer(
        result.get("group_id") or result.get("target") or tool_args.get("target") or ""
    )
    member_name = str(
        result.get("member_name")
        or result.get("subject_name")
        or tool_args.get("user_query")
        or result.get("user_id")
        or result.get("subject_id")
        or "目标成员"
    ).strip()
    reason = str(result.get("reason") or tool_args.get("reason") or "").strip()
    if action == "mute_user":
        duration_seconds = int(result.get("duration_seconds") or tool_args.get("duration_seconds") or 0)
        line = f"已把 {spec.platform_label} {group_id} 的 {member_name} 禁言 {duration_seconds} 秒。"
    else:
        line = f"已把 {spec.platform_label} {group_id} 的 {member_name} 踢出。"
    if reason:
        line += f" 原因：{reason}。"
    return line


def format_admin_qq_group_moderation_reply(tool_args: dict[str, Any], result: dict[str, Any]) -> str:
    return format_admin_platform_group_moderation_reply(
        tool_args,
        result,
        spec=QQ_ADMIN_GROUP_MODERATION_SPEC,
    )


def run_admin_platform_group_moderation_shortcut(
    *,
    tool_args: dict[str, Any] | None,
    shortcut_error: str | None,
    tool_runner: Callable[[dict[str, Any]], dict[str, Any]],
    logger,
    spec: AdminGroupModerationPlatformSpec,
) -> str | None:
    if shortcut_error:
        return shortcut_error
    if not tool_args:
        return None

    try:
        result = tool_runner(tool_args)
    except Exception as exc:
        logger.warning("Admin %s oral moderation shortcut failed: %s", spec.platform_label, exc)
        return f"{spec.error_prefix}：{exc}"

    if isinstance(result, dict) and str(result.get("capability") or "").strip().lower() == "not_capable":
        return format_admin_platform_group_moderation_reply(tool_args, result, spec=spec)

    failure = shortcut_tool_failure_text(result, failure_prefix=spec.error_prefix)
    if failure:
        return failure
    return format_admin_platform_group_moderation_reply(tool_args, result, spec=spec)


def run_admin_qq_group_moderation_shortcut(
    *,
    tool_args: dict[str, Any] | None,
    shortcut_error: str | None,
    tool_runner: Callable[[dict[str, Any]], dict[str, Any]],
    logger,
) -> str | None:
    return run_admin_platform_group_moderation_shortcut(
        tool_args=tool_args,
        shortcut_error=shortcut_error,
        tool_runner=tool_runner,
        logger=logger,
        spec=QQ_ADMIN_GROUP_MODERATION_SPEC,
    )
