"""Pure request-parsing helpers for QQ social-policy oral shortcuts."""

from __future__ import annotations

from gateway.group_control_intents import (
    wants_report_delivery_to_current_chat,
    wants_report_delivery_to_dm,
)
from gateway.qq_intents import (
    _QQ_SOCIAL_DISABLE_TERMS,
    _QQ_SOCIAL_ENABLE_TERMS,
    _QQ_SOCIAL_FRIEND_REQUEST_TERMS,
    _QQ_SOCIAL_GROUP_REQUEST_TERMS,
    _QQ_SOCIAL_POLICY_ALL_TERMS,
    _QQ_SOCIAL_POLICY_FRIEND_TERMS,
    _QQ_SOCIAL_POLICY_GROUP_ADD_TERMS,
    _QQ_SOCIAL_POLICY_GROUP_INVITE_TERMS,
    _QQ_SOCIAL_POLICY_QUERY_TERMS,
)


def match_qq_social_request_type(message_text: str) -> str | None:
    body = str(message_text or "").strip()
    if not body:
        return None
    if any(term in body for term in _QQ_SOCIAL_FRIEND_REQUEST_TERMS):
        return "friend"
    if any(term in body for term in _QQ_SOCIAL_GROUP_REQUEST_TERMS):
        return "group"
    return None


def looks_like_qq_social_policy_query(message_text: str) -> bool:
    body = str(message_text or "").strip()
    if not body:
        return False
    return any(term in body for term in _QQ_SOCIAL_POLICY_QUERY_TERMS) and any(
        term in body
        for term in (
            _QQ_SOCIAL_POLICY_FRIEND_TERMS
            + _QQ_SOCIAL_POLICY_GROUP_ADD_TERMS
            + _QQ_SOCIAL_POLICY_GROUP_INVITE_TERMS
            + _QQ_SOCIAL_POLICY_ALL_TERMS
        )
    )


def qq_social_policy_notify_target(source, message_text: str) -> str | None:
    body = str(message_text or "").strip()
    if any(term in body for term in ("别通知", "不要通知", "不必通知")):
        return "none"
    if wants_report_delivery_to_dm(body):
        return "current_user_dm"
    if wants_report_delivery_to_current_chat(body):
        return "current_chat"
    if getattr(source, "chat_type", "") == "dm":
        return None
    return None


def match_qq_social_control_request(
    *,
    source,
    body: str,
    admin_ids_configured: bool,
    is_admin_user: bool,
    admin_only_message: str,
    looks_like_request_list_query,
    looks_like_policy_candidate,
    looks_like_policy_query,
    request_type_matcher,
    notify_target_resolver,
) -> tuple[dict[str, object] | None, str | None]:
    normalized_body = str(body or "").strip()
    if not normalized_body:
        return None, None

    candidate = looks_like_request_list_query(normalized_body) or looks_like_policy_candidate(normalized_body)
    if not candidate:
        return None, None

    if not admin_ids_configured:
        return None, None
    if not is_admin_user:
        return None, admin_only_message

    if looks_like_request_list_query(normalized_body):
        tool_args: dict[str, object] = {"action": "list_requests", "status": "pending", "limit": 20}
        request_type = request_type_matcher(normalized_body)
        if request_type:
            tool_args["request_type"] = request_type
        return tool_args, None

    if looks_like_policy_query(normalized_body):
        return {"action": "get_social_policy"}, None

    enable = any(term in normalized_body for term in _QQ_SOCIAL_ENABLE_TERMS)
    disable = any(term in normalized_body for term in _QQ_SOCIAL_DISABLE_TERMS)
    if enable == disable:
        return {"action": "get_social_policy"}, None

    desired = True if enable else False
    tool_args: dict[str, object] = {"action": "set_social_policy"}
    touched = False

    if any(term in normalized_body for term in _QQ_SOCIAL_POLICY_FRIEND_TERMS):
        tool_args["auto_approve_friend_requests"] = desired
        touched = True
    if any(term in normalized_body for term in _QQ_SOCIAL_POLICY_GROUP_ADD_TERMS):
        tool_args["auto_approve_group_add_requests"] = desired
        touched = True
    if any(term in normalized_body for term in _QQ_SOCIAL_POLICY_GROUP_INVITE_TERMS):
        tool_args["auto_approve_group_invites"] = desired
        touched = True
    if any(term in normalized_body for term in _QQ_SOCIAL_POLICY_ALL_TERMS) and not touched:
        tool_args["auto_approve_friend_requests"] = desired
        tool_args["auto_approve_group_add_requests"] = desired
        tool_args["auto_approve_group_invites"] = desired
        touched = True

    notify_target = notify_target_resolver(source, normalized_body)
    if notify_target is not None:
        tool_args["notify_target"] = notify_target

    if not touched:
        return {"action": "get_social_policy"}, None
    return tool_args, None
