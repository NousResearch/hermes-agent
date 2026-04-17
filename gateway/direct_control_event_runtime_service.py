"""Shared event/context helpers for gateway direct-control shortcuts."""

from __future__ import annotations

from typing import Any


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

    body = str(getattr(event, "text", "") or "").strip()
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
