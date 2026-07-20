"""Shared oral target-extraction helpers for group-capable messaging platforms."""

from __future__ import annotations

import re
from typing import Any, Callable, Iterable

from gateway.config import Platform
from gateway.group_control_intents import has_current_group_reference
from gateway.qq_intents import (
    _QQ_GROUP_ID_ANYWHERE_RE,
    _QQ_GROUP_ID_EXPLICIT_PATTERNS,
)


def resolve_current_group_target_reference(
    source,
    message_text: str,
    *,
    expected_platform,
    validator: Callable[[str], bool],
    formatter: Callable[[str], str],
) -> str | None:
    body = str(message_text or "").strip()
    if not body:
        return None
    if getattr(source, "platform", None) != expected_platform:
        return None
    if getattr(source, "chat_type", "") != "group":
        return None
    if not has_current_group_reference(body):
        return None
    current_group_id = str(getattr(source, "chat_id", "") or "").strip()
    if not validator(current_group_id):
        return None
    return formatter(current_group_id)


def extract_recent_target_from_history(
    source,
    conversation_history: Iterable[dict[str, Any]] | None,
    *,
    extractor: Callable[[Any, str], str | None],
    predicate: Callable[[dict[str, Any], str], bool] | None = None,
    limit: int = 6,
) -> str:
    items = list(conversation_history or [])
    for item in reversed(items[-limit:]):
        if not isinstance(item, dict):
            continue
        content = str(item.get("content") or "").strip()
        if not content:
            continue
        if predicate is not None and not predicate(item, content):
            continue
        target = extractor(source, content)
        if target:
            return target
    return ""


def extract_qq_group_target(source, message_text: str) -> str | None:
    body = str(message_text or "").strip()
    if not body:
        return None

    for pattern in _QQ_GROUP_ID_EXPLICIT_PATTERNS:
        match = pattern.search(body)
        if match:
            return f"group:{match.group(1)}"

    numeric_matches = {
        match.group(1)
        for match in _QQ_GROUP_ID_ANYWHERE_RE.finditer(body)
    }
    if len(numeric_matches) == 1:
        return f"group:{next(iter(numeric_matches))}"

    return resolve_current_group_target_reference(
        source,
        body,
        expected_platform=Platform.QQ_NAPCAT,
        validator=lambda chat_id: bool(str(chat_id or "").strip()),
        formatter=lambda chat_id: f"group:{str(chat_id).strip()}",
    )


def extract_weixin_group_target(source, message_text: str) -> str | None:
    body = str(message_text or "").strip()
    if not body:
        return None

    explicit_matches = re.findall(r"(?:(?:weixin:)?(?:group:)?)([A-Za-z0-9._-]+@chatroom)", body)
    unique_matches = {str(item or "").strip() for item in explicit_matches if str(item or "").strip()}
    if len(unique_matches) == 1:
        return next(iter(unique_matches))

    return resolve_current_group_target_reference(
        source,
        body,
        expected_platform=Platform.WEIXIN,
        validator=lambda chat_id: str(chat_id or "").strip().endswith("@chatroom"),
        formatter=lambda chat_id: str(chat_id).strip(),
    )
