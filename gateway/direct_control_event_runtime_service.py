"""Shared event/context helpers for gateway direct-control shortcuts."""

from __future__ import annotations

from typing import Any

from gateway.shared_oral_intents import DIRECT_CONTROL_WRAPPER_PATTERNS


def normalize_direct_control_body(body: str) -> str:
    """Strip obvious imperative wrappers before direct-shortcut matching.

    This keeps downstream send/group/intel matchers focused on the actionable
    text instead of conversational prefixes such as ``我让你`` or ``帮我把``.
    """

    normalized = str(body or "").strip()
    if not normalized:
        return ""

    previous = None
    while normalized and normalized != previous:
        previous = normalized
        for pattern in DIRECT_CONTROL_WRAPPER_PATTERNS:
            normalized = pattern.sub("", normalized, count=1).strip()
    return normalized


def extract_platform_text_event_context(event: Any, *, platform) -> tuple[Any | None, str]:
    source = getattr(event, "source", None)
    if getattr(source, "platform", None) != platform:
        return None, ""
    if event.get_command():
        return None, ""
    if getattr(event, "message_type", None) is None:
        return None, ""
    message_type = getattr(event, "message_type", None)
    if str(getattr(message_type, "value", message_type)) != "text":
        return None, ""
    if getattr(event, "media_urls", None):
        return None, ""

    body = normalize_direct_control_body(str(getattr(event, "text", "") or "").strip())
    if not body:
        return None, ""
    return source, body


def build_admin_platform_text_context(
    event: Any,
    *,
    platform,
    configured_admin_user_ids_fn,
    is_admin_user_fn,
) -> dict[str, Any]:
    source, body = extract_platform_text_event_context(event, platform=platform)
    if source is None:
        return {
            "source": None,
            "body": "",
            "admin_ids_configured": False,
            "is_admin_user": False,
        }

    configured_admin_user_ids = configured_admin_user_ids_fn(getattr(source, "platform", None)) or []
    admin_ids_configured = bool(configured_admin_user_ids)
    is_admin_user = bool(is_admin_user_fn(source)) if admin_ids_configured else False
    return {
        "source": source,
        "body": body,
        "admin_ids_configured": admin_ids_configured,
        "is_admin_user": is_admin_user,
    }
