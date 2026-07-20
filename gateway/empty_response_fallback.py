"""User-facing fallbacks when the model yields empty / NO_REPLY responses."""

from __future__ import annotations

from typing import Any

from gateway.config import Platform
from gateway.platforms.base import MessageEvent
from gateway.qq_intents import (
    _QQ_VISIBLE_NAME_ALIASES,
    _looks_like_qq_group_request_text,
    _looks_like_qq_media_message,
    _looks_like_qq_runtime_short_query,
    _qq_group_has_visible_bot_address,
)
from gateway.session import SessionSource


def _qq_group_latest_admin_turn(raw_message: Any) -> bool:
    """Return True when QQ group metadata marks the latest turn as admin-owned."""
    if not isinstance(raw_message, dict):
        return False
    if not bool(raw_message.get("qq_group_batch")):
        return False
    return bool(raw_message.get("latest_is_admin"))


def _qq_group_no_reply_fallback(
    message: str,
    *,
    is_admin_user: bool = False,
    raw_message: Any = None,
) -> str:
    """Return a QQ-group fallback for explicit-address no-reply turns."""
    body = str(message or "").strip()
    if not body:
        return ""
    if _qq_group_latest_admin_turn(raw_message):
        return "刚才这轮接口空转了，消息我收到了。你再发一遍，或者我继续接着干。"
    if any(name in body for name in _QQ_VISIBLE_NAME_ALIASES):
        return "刚才我这轮空转了，但消息我收到了。你再发一遍，或者我继续接着干。"
    if _looks_like_qq_runtime_short_query(body):
        return "刚才我这轮空转了，但我还在。你再发一遍，或者我继续接着干。"
    if is_admin_user and _looks_like_qq_group_request_text(body):
        return "刚才这轮接口空转了，消息我收到了。你再发一遍，或者我继续接着干。"
    return ""


def _qq_group_empty_response_fallback(
    message: str,
    *,
    is_admin_user: bool = False,
    explicit_addressed: bool = False,
) -> str:
    """Return a QQ-group fallback when a provider/tool turn yielded no final text."""
    body = str(message or "").strip()
    if explicit_addressed and not body:
        return "我在，你继续说。"
    if not body:
        return ""
    if explicit_addressed:
        if _looks_like_qq_runtime_short_query(body):
            return "刚才我这轮空转了，但我还在。你再发一遍，或者我继续接着干。"
        return "刚才我这轮空转了，但消息我收到了。你再发一遍，或者我继续接着干。"
    if _qq_group_has_visible_bot_address(body):
        return "刚才我这轮空转了，但消息我收到了。你再发一遍，或者我继续接着干。"
    if _looks_like_qq_runtime_short_query(body):
        return "刚才我这轮空转了，但我还在。你再发一遍，或者我继续接着干。"
    if _looks_like_qq_media_message(body):
        return "刚才这条带图/附件的消息我这轮没读出来。你再发一次，或者补一句文字我继续接。"
    if _looks_like_qq_group_request_text(body):
        return "刚才这轮接口空转了，但消息我收到了。你再发一遍，或者我继续接着干。"
    if is_admin_user:
        return "刚才这轮接口空转了，消息我收到了。你再发一遍，或者我继续接着干。"
    return ""


def explicit_group_trigger_label(event: MessageEvent | None) -> str:
    """Return a short label when the group turn explicitly addressed the bot."""
    source = getattr(event, "source", None)
    if getattr(source, "chat_type", "") != "group":
        return ""

    metadata = getattr(event, "metadata", None) or {}
    explicit_trigger = bool(
        metadata.get("explicit_addressed")
        or metadata.get("requires_reply")
        or metadata.get("explicit_group_trigger")
    )
    explicit_reason = str(
        metadata.get("address_reason")
        or metadata.get("explicit_group_trigger_reason")
        or ""
    ).strip()
    trigger_reason = str(metadata.get("group_trigger_reason") or "").strip()
    if explicit_trigger:
        return explicit_reason or trigger_reason or "explicit_address"
    if trigger_reason in {"bot_mention", "reply_to_bot", "name_trigger"}:
        return trigger_reason
    return ""


def qq_busy_followup_ack(source: SessionSource, message: str = "") -> str:
    """Return a short visible QQ acknowledgement for queued follow-ups."""
    if getattr(source, "platform", None) != Platform.QQ_NAPCAT:
        return ""
    if getattr(source, "chat_type", "") == "dm":
        return "收到，这条我先排队，上一轮忙完马上接着回你。"

    body = str(message or "").strip()
    if any(name in body for name in _QQ_VISIBLE_NAME_ALIASES):
        return "收到，这条我先排队，上一轮忙完接着回你。"
    return ""


def empty_response_fallback(
    source: SessionSource,
    message: str = "",
    *,
    empty_kind: str = "empty",
    is_admin_user: bool = False,
    raw_message: Any = None,
    event: MessageEvent | None = None,
) -> str:
    """Return a user-facing fallback when the model yields no final text."""
    if getattr(source, "chat_type", "") == "dm":
        if getattr(source, "platform", None) == Platform.QQ_NAPCAT:
            return "刚才接口抽了，没吐出正文。你再发一条，或者我继续接着刚才的话题说。"
        return "I didn't get a usable response just now. Please send that again."
    if getattr(source, "chat_type", "") == "group":
        explicit_group_trigger = bool(explicit_group_trigger_label(event))
        if getattr(source, "platform", None) == Platform.QQ_NAPCAT:
            if empty_kind == "no_reply":
                if explicit_group_trigger:
                    return "收到，你继续说。"
                return _qq_group_no_reply_fallback(
                    message,
                    is_admin_user=is_admin_user,
                    raw_message=raw_message,
                )
            return _qq_group_empty_response_fallback(
                message,
                is_admin_user=is_admin_user,
                explicit_addressed=explicit_group_trigger,
            )
        if explicit_group_trigger:
            if empty_kind == "no_reply":
                return "收到，你继续说。"
            if not str(message or "").strip():
                return "我在，你继续说。"
            return "刚才这轮没吐出正文，但消息我收到了。你再发一遍，或者我继续接着干。"
    return ""


def explicit_group_reply_context_note(event: MessageEvent) -> str:
    """Inject a system note forcing a reply on explicitly-addressed group turns."""
    label = explicit_group_trigger_label(event)
    if not label:
        return ""
    return (
        "[Current group turn note: This message explicitly addressed you "
        f"(trigger reason: {label}). You must reply briefly to this turn. "
        "Do not return [[NO_REPLY]] or an empty response for this turn.]"
    )
