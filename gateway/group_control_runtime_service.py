"""Shared runtime helpers for direct oral group-control shortcuts."""

from __future__ import annotations

from typing import Any, Callable

from gateway.direct_tool_result_runtime_service import shortcut_tool_failure_text
from gateway.group_control_requests import match_group_control_request


def match_admin_platform_group_control_request(
    *,
    source: Any,
    body: str,
    target_extractor,
    admin_ids_configured: bool,
    is_admin_user: bool,
    missing_target_message: str,
    admin_only_message: str,
    collect_only_action: str,
    report_target_resolver,
    unresolved_target_guard=None,
) -> tuple[dict[str, Any] | None, str | None]:
    normalized_body = str(body or "").strip()
    if source is None or not normalized_body:
        return None, None

    target = target_extractor(source, normalized_body)
    if unresolved_target_guard is not None and unresolved_target_guard(normalized_body) and not target:
        return None, None

    return match_group_control_request(
        source=source,
        body=normalized_body,
        target=target,
        admin_ids_configured=admin_ids_configured,
        is_admin_user=is_admin_user,
        missing_target_message=missing_target_message,
        admin_only_message=admin_only_message,
        collect_only_action=collect_only_action,
        report_target_resolver=report_target_resolver,
    )


def run_admin_group_control_shortcut(
    *,
    tool_args: dict[str, Any] | None,
    shortcut_error: str | None,
    tool_runner: Callable[[dict[str, Any]], dict[str, Any]],
    error_prefix: str,
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
        logger.warning("%s: %s", error_prefix, exc)
        return f"{error_prefix}：{exc}"

    failure = shortcut_tool_failure_text(result, failure_prefix=error_prefix)
    if failure:
        return failure
    return reply_formatter(tool_args, result)
