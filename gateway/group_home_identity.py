"""Exact home locations and transport-authenticated Group Chat principals."""

from __future__ import annotations

import hashlib
import json


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
    from gateway.hosted_room_messaging import (
        is_machine_authored,
        is_message_edit,
        relay_provenance_is_unknown,
    )

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


def home_identity(home):
    return (
        home.platform.value,
        str(home.chat_id),
        str(home.thread_id or ""),
        str(home.user_id or ""),
        str(home.scope_id or ""),
        str(getattr(home, "selection_id", None) or ""),
    )


def acknowledgement(home):
    return hashlib.sha256(
        json.dumps(
            ["group-home-audience-v1", *home_identity(home)], separators=(",", ":")
        ).encode()
    ).hexdigest()


def audience_accepted(config, source):
    if is_private_source(source):
        return True
    from gateway.slash_access import is_home_control_source

    if not is_home_control_source(config, source, require_owner_identity=True):
        return False
    home = config.get_home_channel(source.platform)
    return home.group_audience_ack == acknowledgement(home)
