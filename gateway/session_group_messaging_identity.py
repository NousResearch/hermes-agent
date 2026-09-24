"""Pure M73 person/private-source provenance, selectively retained.

Derived from group_home_identity.py and hosted_room_messaging.py at
73fbc700664c56eabd9d7a55f178320662ef0c47 (dokterdok/hermes-agent).
No Home enrollment, service resolution, mailbox or command authority is imported.
"""
from collections.abc import Mapping
from typing import Any


NATIVE_DISTINCT_DM_PLATFORMS = frozenset({
    "bluebubbles",
    "dingtalk",
    "email",
    "feishu",
    "mattermost",
    "qqbot",
    "sms",
    "wecom",
    "wecom_callback",
    "weixin",
    "whatsapp_cloud",
    "yuanbao",
})


def home_thread_from_source(source):
    """Ignore Slack's synthetic per-message session thread, not real threads."""
    thread = getattr(source, "thread_id", None)
    if not thread:
        return None
    platform = getattr(getattr(source, "platform", None), "value", "")
    if (
        platform == "slack"
        and getattr(source, "message_id", None)
        and str(thread) == str(source.message_id)
    ):
        return None
    return str(thread)


def is_private_source(source):
    if str(getattr(source, "chat_type", "") or "").casefold() not in {
        "dm",
        "direct",
        "private",
    }:
        return False
    platform = getattr(getattr(source, "platform", None), "value", "")
    return getattr(source, "is_one_to_one", None) is True or (
        getattr(source, "delivered_via_upstream_relay", False) is not True
        and platform in NATIVE_DISTINCT_DM_PLATFORMS
    )


def trusted_person(event):
    source = event.source
    platform = getattr(getattr(source, "platform", None), "value", "")
    user = str(getattr(source, "user_id", "") or "").strip()
    if (
        not user
        or user.casefold() in {"unknown", "anonymous", "none", "null", "channel"}
        or not getattr(source, "chat_id", None)
        or platform == "irc"
        or relay_provenance_is_unknown(event)
        or getattr(source, "profile_route_rejected", False) is True
        or is_machine_authored(event)
        or is_message_edit(event)
    ):
        return False
    if platform == "telegram":
        raw = getattr(event, "raw_message", None)
        if (
            str(source.chat_type).casefold() in {"channel", "broadcast"}
            or getattr(raw, "sender_chat", None) is not None
            or (isinstance(raw, dict) and raw.get("sender_chat") is not None)
            or user.startswith("-")
            or user == "1087968824"
        ):
            return False
    return True


def is_machine_authored(event: Any) -> bool:
    """Recognize native and relayed bot/webhook provenance defensively."""

    source = getattr(event, "source", None)
    if getattr(source, "is_bot", False):
        return True
    metadata = getattr(event, "metadata", None)
    if isinstance(metadata, Mapping) and any(
        metadata.get(key) is True
        for key in ("is_bot", "sender_is_bot", "webhook_sender")
    ):
        return True
    raw = getattr(event, "raw_message", None)
    if isinstance(raw, Mapping):
        if raw.get("bot_id") or raw.get("bot_profile"):
            return True
        if raw.get("subtype") in {"bot_message", "webhook_message"}:
            return True
    for owner_field in ("author", "user"):
        owner = getattr(raw, owner_field, None)
        if getattr(owner, "bot", False) or getattr(owner, "is_bot", False):
            return True
    return False


def is_message_edit(event: Any) -> bool:
    """Reject edited commands even when a platform redelivers them as messages."""

    source = getattr(event, "source", None)
    if getattr(source, "message_is_edit", False):
        return True
    metadata = getattr(event, "metadata", None)
    if isinstance(metadata, Mapping) and metadata.get("message_is_edit") is True:
        return True
    raw = getattr(event, "raw_message", None)
    if isinstance(raw, Mapping):
        if raw.get("editMessage") or raw.get("isEdited") is True:
            return True
        if raw.get("subtype") == "message_changed":
            return True
        relation = raw.get("m.relates_to")
        if isinstance(relation, Mapping) and relation.get("rel_type") == "m.replace":
            return True
    return bool(getattr(raw, "edit_date", None) or getattr(raw, "edited_at", None))


def relay_provenance_is_unknown(event: Any) -> bool:
    """Fail closed until a relay producer classifies the inbound author."""

    source = getattr(event, "source", None)
    if not getattr(source, "delivered_via_upstream_relay", False):
        return False
    metadata = getattr(event, "metadata", None)
    return not (
        isinstance(metadata, Mapping)
        and metadata.get("relay_author_classified") is True
        and metadata.get("relay_edit_classified") is True
    )
