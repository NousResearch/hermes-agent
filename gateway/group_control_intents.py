"""Shared oral-intent helpers for group policy/report control across platforms."""

from __future__ import annotations

from gateway.shared_oral_intents import (
    GROUP_CHAT_ENABLE_TERMS,
    GROUP_CURRENT_TARGET_TERMS,
    GROUP_LISTEN_DISABLE_PATTERNS,
    GROUP_LISTEN_DISABLE_TERMS,
    GROUP_LISTEN_ENABLE_PATTERNS,
    GROUP_LISTEN_ENABLE_TERMS,
    GROUP_LISTEN_HINT_TERMS,
    GROUP_REPORT_CURRENT_CHAT_TERMS,
    GROUP_REPORT_DISABLE_TERMS,
    GROUP_REPORT_DM_TERMS,
    GROUP_REPORT_ENABLE_TERMS,
    GROUP_REPORT_NOW_TERMS,
    GROUP_STATUS_QUERY_TERMS,
)

_FOLLOWUP_GROUP_REFERENCE_TERMS = (
    "这个群",
    "这群",
    "当前群",
    "本群",
    "那个群",
    "那群",
    "该群",
    "群里",
    "在群里",
)
_GROUP_CHAT_ENABLE_IMPERATIVE_TERMS = (
    "允许开始聊天",
    "允许聊天",
    "开始聊天",
    "恢复聊天",
    "正常聊天",
    "恢复回复",
    "允许回复",
)
_GENERIC_GROUP_RUNTIME_STATUS_TERMS = (
    "什么状态",
    "啥状态",
    "什么情况",
    "啥情况",
    "当前状态",
)


def has_current_group_reference(message_text: str) -> bool:
    body = str(message_text or "").strip()
    return bool(body) and any(term in body for term in GROUP_CURRENT_TARGET_TERMS)


def has_followup_group_reference(message_text: str) -> bool:
    body = str(message_text or "").strip()
    return bool(body) and (
        has_current_group_reference(body)
        or any(term in body for term in _FOLLOWUP_GROUP_REFERENCE_TERMS)
    )


def strip_current_group_reference_terms(message_text: str) -> str:
    cleaned = str(message_text or "")
    for term in GROUP_CURRENT_TARGET_TERMS:
        cleaned = cleaned.replace(term, "")
    return cleaned


def looks_like_group_chat_enable_request(message_text: str) -> bool:
    body = str(message_text or "").strip()
    return (
        bool(body)
        and any(term in body for term in GROUP_CHAT_ENABLE_TERMS)
        and (
            ("?" not in body and "？" not in body)
            or any(term in body for term in _GROUP_CHAT_ENABLE_IMPERATIVE_TERMS)
        )
    )


def looks_like_group_listen_disable_request(message_text: str) -> bool:
    body = str(message_text or "").strip()
    if not body:
        return False
    if any(term in body for term in GROUP_LISTEN_DISABLE_TERMS):
        return True
    if any(pattern.search(body) for pattern in GROUP_LISTEN_DISABLE_PATTERNS):
        return True
    return looks_like_group_chat_enable_request(body) and any(hint in body for hint in GROUP_LISTEN_HINT_TERMS)


def looks_like_group_listen_enable_request(message_text: str) -> bool:
    body = str(message_text or "").strip()
    if not body:
        return False
    if any(term in body for term in GROUP_LISTEN_ENABLE_TERMS):
        return True
    return any(pattern.search(body) for pattern in GROUP_LISTEN_ENABLE_PATTERNS)


def looks_like_group_runtime_status_query(message_text: str) -> bool:
    body = str(message_text or "").strip()
    return bool(body) and (
        any(term in body for term in GROUP_STATUS_QUERY_TERMS)
        or (has_followup_group_reference(body) and any(term in body for term in _GENERIC_GROUP_RUNTIME_STATUS_TERMS))
    )


def looks_like_group_report_enable_request(message_text: str) -> bool:
    body = str(message_text or "").strip()
    return bool(body) and any(term in body for term in GROUP_REPORT_ENABLE_TERMS)


def looks_like_group_report_disable_request(message_text: str) -> bool:
    body = str(message_text or "").strip()
    return bool(body) and any(term in body for term in GROUP_REPORT_DISABLE_TERMS)


def looks_like_group_report_now_request(message_text: str) -> bool:
    body = str(message_text or "").strip()
    return bool(body) and any(term in body for term in GROUP_REPORT_NOW_TERMS)


def wants_report_delivery_to_dm(message_text: str) -> bool:
    body = str(message_text or "").strip()
    return bool(body) and any(term in body for term in GROUP_REPORT_DM_TERMS)


def wants_report_delivery_to_current_chat(message_text: str) -> bool:
    body = str(message_text or "").strip()
    return bool(body) and any(term in body for term in GROUP_REPORT_CURRENT_CHAT_TERMS)


def resolve_oral_report_delivery_target(message_text: str, *, prefer_dm: bool) -> str:
    body = str(message_text or "").strip()
    if wants_report_delivery_to_dm(body):
        return "current_user_dm"
    if wants_report_delivery_to_current_chat(body):
        return "current_chat"
    if prefer_dm:
        return "current_user_dm"
    return "current_chat"
