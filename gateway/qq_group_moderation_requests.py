"""Pure request-parsing helpers for QQ group moderation oral shortcuts."""

from __future__ import annotations

from gateway.qq_intents import (
    _QQ_GROUP_ID_EXPLICIT_PATTERNS,
    _QQ_GROUP_MODERATION_COMMON_REASONS,
    _QQ_GROUP_MODERATION_DURATION_RE,
    _QQ_GROUP_MODERATION_KICK_TERMS,
    _QQ_GROUP_MODERATION_MUTE_TERMS,
    _QQ_GROUP_MODERATION_NEGATION_TERMS,
    _QQ_GROUP_MODERATION_REASON_RE,
    _QQ_GROUP_MODERATION_USER_PATTERNS,
)
from gateway.group_control_intents import (
    looks_like_group_listen_disable_request,
    strip_current_group_reference_terms,
)


def extract_qq_oral_moderation_duration_seconds(message_text: str) -> int | None:
    body = str(message_text or "").strip()
    if not body:
        return None
    match = _QQ_GROUP_MODERATION_DURATION_RE.search(body)
    if not match:
        return None
    value = int(match.group(1))
    unit = str(match.group(2) or "")
    if unit.startswith("秒"):
        return value
    if unit.startswith("分"):
        return value * 60
    if unit.startswith("小时") or unit == "时":
        return value * 3600
    if unit.startswith("天"):
        return value * 86400
    return None


def extract_qq_oral_moderation_reason(message_text: str) -> str:
    body = str(message_text or "").strip()
    if not body:
        return ""
    match = _QQ_GROUP_MODERATION_REASON_RE.search(body)
    if match:
        return str(match.group(1) or "").strip(" ，。,；;")
    for reason in _QQ_GROUP_MODERATION_COMMON_REASONS:
        if reason in body:
            return reason
    return ""


def extract_qq_oral_moderation_user_query(message_text: str) -> str:
    body = str(message_text or "").strip()
    if not body:
        return ""
    cleaned = _QQ_GROUP_MODERATION_REASON_RE.sub("", body)
    cleaned = _QQ_GROUP_MODERATION_DURATION_RE.sub("", cleaned)
    for pattern in _QQ_GROUP_ID_EXPLICIT_PATTERNS:
        cleaned = pattern.sub("", cleaned)
    cleaned = strip_current_group_reference_terms(cleaned)
    for pattern in _QQ_GROUP_MODERATION_USER_PATTERNS:
        match = pattern.search(cleaned)
        if match:
            return str(match.group(1) or "").strip().lstrip("@").strip()
    return ""


def match_qq_group_moderation_action(message_text: str) -> str:
    body = str(message_text or "").strip()
    if not body:
        return ""
    if any(term in body for term in _QQ_GROUP_MODERATION_NEGATION_TERMS):
        return ""
    if any(term in body for term in _QQ_GROUP_MODERATION_MUTE_TERMS):
        return "mute_user"
    if any(term in body for term in _QQ_GROUP_MODERATION_KICK_TERMS):
        return "kick_user"
    return ""


def match_qq_group_moderation_request(
    *,
    source,
    body: str,
    admin_ids_configured: bool,
    is_admin_user: bool,
    admin_only_message: str,
    action_matcher,
    target_extractor,
    user_query_extractor,
    reason_extractor,
    duration_extractor,
    current_group_target_formatter=None,
    missing_target_message: str = "要禁言/踢人，请直接说清群号，或者在目标群里明确说“这个群”。",
) -> tuple[dict[str, object] | None, str | None]:
    normalized_body = str(body or "").strip()
    if not normalized_body:
        return None, None
    if looks_like_group_listen_disable_request(normalized_body):
        return None, None

    action = str(action_matcher(normalized_body) or "").strip()
    if not action:
        return None, None

    if not admin_ids_configured:
        return None, None
    if not is_admin_user:
        return None, admin_only_message

    target = target_extractor(source, normalized_body)
    if not target and getattr(source, "chat_type", "") == "group":
        current_group_id = str(getattr(source, "chat_id", "") or "").strip()
        if current_group_id:
            formatter = current_group_target_formatter or (lambda chat_id: f"group:{chat_id}")
            target = formatter(current_group_id)
    if not target:
        return None, missing_target_message

    user_query = str(user_query_extractor(normalized_body) or "").strip()
    if not user_query:
        return None, "要禁言/踢人，请把对象说清楚。"

    reason = str(reason_extractor(normalized_body) or "").strip()
    if not reason:
        return None, "要禁言/踢人，请顺带把原因说清楚。"

    tool_args: dict[str, object] = {
        "action": action,
        "target": target,
        "user_query": user_query,
        "reason": reason,
    }
    if action == "mute_user":
        duration_seconds = duration_extractor(normalized_body)
        if not duration_seconds:
            return None, "要禁言请把时长说清楚，比如 10 分钟。"
        tool_args["duration_seconds"] = duration_seconds
    return tool_args, None
