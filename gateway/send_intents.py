"""Shared oral send-message intent helpers."""

from __future__ import annotations

import re

from gateway.qq_intents import _QQ_SEND_INLINE_PATTERNS
from gateway.shared_oral_intents import (
    SEND_CONFIRM_TERMS,
    SEND_QUERY_TERMS,
)

_WEIXIN_SEND_INLINE_PATTERNS = (
    re.compile(
        r"往\s*(?:微信(?:群)?\s*)?([A-Za-z0-9._-]+@chatroom)\s*发[：:，,]?\s*(.+)$",
        re.IGNORECASE,
    ),
    re.compile(
        r"向\s*(?:微信(?:群)?\s*)?([A-Za-z0-9._-]+@chatroom)\s*发送[：:，,]?\s*(.+)$",
        re.IGNORECASE,
    ),
)


def looks_like_send_query(message_text: str) -> bool:
    body = str(message_text or "").strip()
    return bool(body) and any(term in body for term in SEND_QUERY_TERMS)


def looks_like_send_confirmation(message_text: str) -> bool:
    body = str(message_text or "").strip()
    return bool(body) and any(term in body for term in SEND_CONFIRM_TERMS)


def extract_send_confirmation_message(message_text: str) -> str:
    body = str(message_text or "").strip()
    if not body:
        return ""
    cleaned = body
    for term in SEND_CONFIRM_TERMS:
        cleaned = cleaned.replace(term, "")
    return cleaned.strip(" \n\r\t：:，,。；;")


def extract_qq_inline_send_target_and_message(message_text: str) -> tuple[str, str]:
    body = str(message_text or "").strip()
    if not body:
        return "", ""
    for pattern in _QQ_SEND_INLINE_PATTERNS:
        match = pattern.search(body)
        if not match:
            continue
        group_id = str(match.group(1) or "").strip()
        message = str(match.group(2) or "").strip().strip("：:，,。")
        if group_id and message:
            return f"group:{group_id}", message
    return "", ""


def extract_weixin_inline_send_target_and_message(message_text: str) -> tuple[str, str]:
    body = str(message_text or "").strip()
    if not body:
        return "", ""
    for pattern in _WEIXIN_SEND_INLINE_PATTERNS:
        match = pattern.search(body)
        if not match:
            continue
        chat_id = str(match.group(1) or "").strip()
        message = str(match.group(2) or "").strip().strip("：:，,。")
        if chat_id and message:
            return chat_id, message
    return "", ""
