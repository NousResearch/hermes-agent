"""Opt-in, exact-ID permission checks for Slack and Microsoft Teams messages.

The receiving adapter supplies platform IDs from the authenticated event.  Text in a
message, display names, and model-generated IDs are never used as a grant source.
"""

from __future__ import annotations

from typing import Any


_KINDS = {
    "slack": {"user", "channel"},
    "teams": {"user", "group_chat", "channel"},
}


def _grants(extra: dict[str, Any]) -> list[dict]:
    settings = extra.get("enterprise_permissions")
    if not isinstance(settings, dict) or settings.get("enabled") is not True:
        return []
    grants = settings.get("grants")
    return grants if isinstance(grants, list) else []


def enforcement_enabled(extra: dict[str, Any]) -> bool:
    settings = extra.get("enterprise_permissions")
    # An explicitly configured but malformed policy must not turn protection off.
    return settings is not None and not (isinstance(settings, dict) and settings.get("enabled") is False)


def allows(extra: dict[str, Any], *, platform: str, scope_id: str, kind: str,
           resource_id: str, action: str) -> bool:
    """Exact scope/kind/id grant; malformed enabled policy fails closed.

    ``action`` is ``read`` or ``write``.  The platform is fixed by the adapter,
    not supplied by the message or by a tool argument.
    """
    if not enforcement_enabled(extra):
        return True
    if (platform not in _KINDS or kind not in _KINDS[platform]
            or action not in {"read", "write"} or not scope_id or not resource_id):
        return False
    return any(
        isinstance(grant, dict)
        and grant.get("scope_id") == scope_id
        and grant.get("kind") == kind
        and grant.get("id") == resource_id
        and grant.get(action) is True
        for grant in _grants(extra)
    )


def allows_inbound(extra: dict[str, Any], *, platform: str, scope_id: str,
                   user_id: str, chat_id: str, chat_kind: str) -> bool:
    """Both the sender and shared destination need read permission.

    Personal DMs are covered by the sender grant.  Slack MPIM is deliberately
    unsupported in enforced mode until it has an explicit resource type.
    """
    if not enforcement_enabled(extra):
        return True
    if not allows(extra, platform=platform, scope_id=scope_id, kind="user",
                  resource_id=user_id, action="read"):
        return False
    if chat_kind == "dm":
        return True
    if chat_kind not in _KINDS.get(platform, set()) - {"user"}:
        return False
    return allows(extra, platform=platform, scope_id=scope_id, kind=chat_kind,
                  resource_id=chat_id, action="read")
