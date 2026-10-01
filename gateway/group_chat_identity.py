"""Who sent a messaging command, and whether the chat is one-to-one with the Bot.

Pure predicates: no grants, rooms or storage. ``gateway.group_chat_access`` decides
what a sender may do with them.
"""
from collections.abc import Mapping
from typing import Any

# Native platforms whose "dm" chat is always one person and the Bot. Slack ("mpim") and
# Matrix (direct rooms) also call multi-person chats "dm", so they count as shared.
ONE_TO_ONE_DM_PLATFORMS = frozenset({
    "bluebubbles", "dingtalk", "discord", "email", "feishu", "mattermost", "qqbot", "signal",
    "sms", "telegram", "wecom", "wecom_callback", "weixin", "whatsapp", "whatsapp_cloud", "yuanbao",
})
_DM_CHAT_TYPES = frozenset({"dm", "direct", "private"})


def platform_name(source: Any) -> str:
    return str(getattr(getattr(source, "platform", None), "value", "") or "")


def is_dm_scope(source: Any) -> bool:
    """The slash-access scope: a DM chat type uses the DM admin list."""
    return str(getattr(source, "chat_type", "") or "").strip().casefold() in _DM_CHAT_TYPES


def is_private_source(source: Any) -> bool:
    """True only for a chat that is provably the sender and the Bot alone."""
    if not is_dm_scope(source):
        return False
    return (getattr(source, "is_one_to_one", None) is True
            or (getattr(source, "delivered_via_upstream_relay", False) is not True
                and platform_name(source) in ONE_TO_ONE_DM_PLATFORMS))


def trusted_person(event: Any) -> bool:
    """A real, unedited message from an identified person (never a bot, webhook or channel)."""
    source = event.source
    user = str(getattr(source, "user_id", "") or "").strip()
    if (not user
            or user.casefold() in {"unknown", "anonymous", "none", "null", "channel"}
            or not getattr(source, "chat_id", None)
            or platform_name(source) == "irc"
            or relay_provenance_is_unknown(event)
            or getattr(source, "profile_route_rejected", False) is True
            or is_machine_authored(event)
            or is_message_edit(event)):
        return False
    if platform_name(source) == "telegram":
        raw = getattr(event, "raw_message", None)
        # Channel posts and anonymous group admins speak as the chat, not as a person.
        if (str(source.chat_type).casefold() in {"channel", "broadcast"}
                or getattr(raw, "sender_chat", None) is not None
                or (isinstance(raw, dict) and raw.get("sender_chat") is not None)
                or user.startswith("-")
                or user == "1087968824"):
            return False
    return True


def is_machine_authored(event: Any) -> bool:
    """Recognize native and relayed bot/webhook provenance defensively."""
    source = getattr(event, "source", None)
    if getattr(source, "is_bot", False):
        return True
    metadata = getattr(event, "metadata", None)
    if isinstance(metadata, Mapping) and any(
            metadata.get(key) is True for key in ("is_bot", "sender_is_bot", "webhook_sender")):
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
    return not (isinstance(metadata, Mapping)
                and metadata.get("relay_author_classified") is True
                and metadata.get("relay_edit_classified") is True)
