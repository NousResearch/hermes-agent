"""Pure request-parsing helpers for oral group control shortcuts."""

from __future__ import annotations

from gateway.group_control_plane import NormalizedGroupControlRequest
from gateway.group_control_intents import (
    has_followup_group_reference,
    looks_like_group_chat_enable_request,
    looks_like_group_listen_disable_request,
    looks_like_group_listen_enable_request,
    looks_like_group_report_disable_request,
    looks_like_group_report_enable_request,
    looks_like_group_report_now_request,
    wants_report_delivery_to_current_chat,
    wants_report_delivery_to_dm,
)


def match_group_control_request(
    *,
    source,
    body: str,
    target: str | None,
    admin_ids_configured: bool,
    is_admin_user: bool,
    missing_target_message: str,
    admin_only_message: str,
    collect_only_action: str,
    report_target_resolver,
) -> tuple[dict[str, object] | None, str | None]:
    normalized_body = str(body or "").strip()
    if not admin_ids_configured:
        return None, None

    allow_chat = looks_like_group_chat_enable_request(normalized_body)
    disable_listen = looks_like_group_listen_disable_request(normalized_body)
    enable_listen = not disable_listen and looks_like_group_listen_enable_request(normalized_body)
    disable_report = looks_like_group_report_disable_request(normalized_body)
    report_now = looks_like_group_report_now_request(normalized_body)
    explicit_report_delivery = wants_report_delivery_to_dm(normalized_body) or wants_report_delivery_to_current_chat(
        normalized_body
    )
    enable_report = looks_like_group_report_enable_request(normalized_body) or (
        enable_listen and "日报" in normalized_body and explicit_report_delivery
    )

    if not any((allow_chat, enable_listen, disable_listen, enable_report, disable_report, report_now)):
        return None, None

    if not is_admin_user:
        return None, admin_only_message
    if not target:
        if not has_followup_group_reference(normalized_body):
            return None, None
        return None, missing_target_message

    if report_now:
        request = NormalizedGroupControlRequest(
            action="deliver_report",
            target=target,
            delivery_target=report_target_resolver(
                source,
                normalized_body,
                prefer_dm=False,
            ),
        )
        return request.to_tool_args(), None

    action = ""
    if disable_listen:
        action = "resume_chat" if allow_chat else "disable_group"
    elif allow_chat:
        action = "resume_chat"
    elif enable_listen:
        action = collect_only_action

    request = NormalizedGroupControlRequest(
        action=action,
        target=target,
    )
    if enable_report:
        report_target = report_target_resolver(
            source,
            normalized_body,
            prefer_dm=True,
        )
        request = NormalizedGroupControlRequest(
            action=action,
            target=target,
            daily_report_enabled=True,
            daily_report_target=report_target,
            manual_report_target=report_target,
        )
    elif disable_report:
        request = NormalizedGroupControlRequest(
            action=action,
            target=target,
            daily_report_enabled=False,
        )

    if not request.action:
        return None, None
    return request.to_tool_args(), None
