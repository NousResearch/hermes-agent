"""Shared QQ/NapCat intent lexicon and regex definitions.

This module centralizes the QQ natural-language trigger words and regexes that
were previously duplicated across ``gateway/run.py`` and
``gateway/platforms/qq_napcat.py``.  The source of truth lives in
``gateway/data/qq_intents.json`` so policy vocabulary can evolve without
further bloating the gateway runner.
"""

from __future__ import annotations

import json
import logging
import re
from functools import lru_cache
from pathlib import Path
from typing import Any

from gateway.group_control_intents import (
    has_followup_group_reference,
    looks_like_group_runtime_status_query as looks_like_shared_group_runtime_status_query,
)

logger = logging.getLogger(__name__)

_DATA_PATH = Path(__file__).resolve().parent / "data" / "qq_intents.json"
_FLAG_MAP = {
    "IGNORECASE": re.IGNORECASE,
    "DOTALL": re.DOTALL,
    "MULTILINE": re.MULTILINE,
}


@lru_cache(maxsize=1)
def _load_qq_intents_data() -> dict[str, Any]:
    try:
        raw = json.loads(_DATA_PATH.read_text(encoding="utf-8"))
    except FileNotFoundError:
        logger.warning("QQ intents data file missing: %s", _DATA_PATH)
        return {"run_terms": {}, "run_patterns": {}, "napcat_terms": {}}
    except Exception as exc:
        logger.warning("Failed to load QQ intents data from %s: %s", _DATA_PATH, exc)
        return {"run_terms": {}, "run_patterns": {}, "napcat_terms": {}}
    if not isinstance(raw, dict):
        logger.warning("QQ intents data must be a dict: %s", _DATA_PATH)
        return {"run_terms": {}, "run_patterns": {}, "napcat_terms": {}}
    return raw


def _load_term_sequence(section: str, key: str) -> tuple[str, ...]:
    payload = (_load_qq_intents_data().get(section) or {}).get(key) or []
    if not isinstance(payload, list):
        return ()
    normalized: list[str] = []
    for item in payload:
        text = str(item or "").strip()
        if text:
            normalized.append(text)
    return tuple(normalized)


def _load_term_set(section: str, key: str) -> frozenset[str]:
    return frozenset(_load_term_sequence(section, key))


def _compile_pattern(spec: dict[str, Any]) -> re.Pattern[str]:
    pattern = str(spec.get("pattern") or "")
    flags = 0
    for flag_name in spec.get("flags") or []:
        flags |= _FLAG_MAP.get(str(flag_name or "").strip().upper(), 0)
    return re.compile(pattern, flags)


def _load_pattern_sequence(section: str, key: str) -> tuple[re.Pattern[str], ...]:
    payload = (_load_qq_intents_data().get(section) or {}).get(key)
    if isinstance(payload, list):
        return tuple(_compile_pattern(spec) for spec in payload if isinstance(spec, dict))
    if isinstance(payload, dict):
        return (_compile_pattern(payload),)
    return ()


def _load_single_pattern(section: str, key: str) -> re.Pattern[str]:
    compiled = _load_pattern_sequence(section, key)
    if compiled:
        return compiled[0]
    return re.compile(r"$^")


_QQ_VISIBLE_NAME_ALIASES = _load_term_sequence("run_terms", "_QQ_VISIBLE_NAME_ALIASES")
_QQ_MEDIA_PLACEHOLDER_MARKERS = _load_term_sequence("run_terms", "_QQ_MEDIA_PLACEHOLDER_MARKERS")
_QQ_GROUP_REQUEST_HINT_TERMS = _load_term_sequence("run_terms", "_QQ_GROUP_REQUEST_HINT_TERMS")
_QQ_GROUP_CURRENT_TARGET_TERMS = _load_term_sequence("run_terms", "_QQ_GROUP_CURRENT_TARGET_TERMS")
_QQ_GROUP_LISTEN_DISABLE_TERMS = _load_term_sequence("run_terms", "_QQ_GROUP_LISTEN_DISABLE_TERMS")
_QQ_GROUP_LISTEN_ENABLE_TERMS = _load_term_sequence("run_terms", "_QQ_GROUP_LISTEN_ENABLE_TERMS")
_QQ_GROUP_CHAT_ENABLE_TERMS = _load_term_sequence("run_terms", "_QQ_GROUP_CHAT_ENABLE_TERMS")
_QQ_GROUP_REPORT_ENABLE_TERMS = _load_term_sequence("run_terms", "_QQ_GROUP_REPORT_ENABLE_TERMS")
_QQ_GROUP_REPORT_DISABLE_TERMS = _load_term_sequence("run_terms", "_QQ_GROUP_REPORT_DISABLE_TERMS")
_QQ_GROUP_REPORT_NOW_TERMS = _load_term_sequence("run_terms", "_QQ_GROUP_REPORT_NOW_TERMS")
_QQ_GROUP_REPORT_DM_TERMS = _load_term_sequence("run_terms", "_QQ_GROUP_REPORT_DM_TERMS")
_QQ_GROUP_REPORT_CURRENT_CHAT_TERMS = _load_term_sequence("run_terms", "_QQ_GROUP_REPORT_CURRENT_CHAT_TERMS")
_QQ_SEND_QUERY_TERMS = _load_term_sequence("run_terms", "_QQ_SEND_QUERY_TERMS")
_QQ_SEND_CONFIRM_TERMS = _load_term_sequence("run_terms", "_QQ_SEND_CONFIRM_TERMS")
_QQ_BACKGROUND_STATUS_QUERY_TERMS = _load_term_sequence("run_terms", "_QQ_BACKGROUND_STATUS_QUERY_TERMS")
_QQ_GROUP_STATUS_QUERY_TERMS = _load_term_sequence("run_terms", "_QQ_GROUP_STATUS_QUERY_TERMS")
_QQ_JOINED_GROUP_LIST_TERMS = _load_term_sequence("run_terms", "_QQ_JOINED_GROUP_LIST_TERMS")
_QQ_RUNTIME_STATUS_QUERY_TERMS = _load_term_sequence("run_terms", "_QQ_RUNTIME_STATUS_QUERY_TERMS")
_QQ_RUNTIME_STATUS_SHORT_TERMS = _load_term_sequence("run_terms", "_QQ_RUNTIME_STATUS_SHORT_TERMS")
_QQ_INTEL_HIRE_TERMS = _load_term_sequence("run_terms", "_QQ_INTEL_HIRE_TERMS")
_QQ_INTEL_PAUSE_TERMS = _load_term_sequence("run_terms", "_QQ_INTEL_PAUSE_TERMS")
_QQ_INTEL_RESUME_TERMS = _load_term_sequence("run_terms", "_QQ_INTEL_RESUME_TERMS")
_QQ_INTEL_STOP_TERMS = _load_term_sequence("run_terms", "_QQ_INTEL_STOP_TERMS")
_QQ_INTEL_WORKER_STATUS_QUERY_TERMS = _load_term_sequence("run_terms", "_QQ_INTEL_WORKER_STATUS_QUERY_TERMS")
_QQ_INTEL_CONTEXT_TERMS = _load_term_sequence("run_terms", "_QQ_INTEL_CONTEXT_TERMS")
_QQ_INTEL_WORKER_NAME_STOPWORDS = _load_term_set("run_terms", "_QQ_INTEL_WORKER_NAME_STOPWORDS")
_QQ_GROUP_MODERATION_MUTE_TERMS = _load_term_sequence("run_terms", "_QQ_GROUP_MODERATION_MUTE_TERMS")
_QQ_GROUP_MODERATION_KICK_TERMS = _load_term_sequence("run_terms", "_QQ_GROUP_MODERATION_KICK_TERMS")
_QQ_GROUP_MODERATION_NEGATION_TERMS = _load_term_sequence("run_terms", "_QQ_GROUP_MODERATION_NEGATION_TERMS")
_QQ_GROUP_MODERATION_COMMON_REASONS = _load_term_sequence("run_terms", "_QQ_GROUP_MODERATION_COMMON_REASONS")
_QQ_SOCIAL_QUERY_HINT_TERMS = _load_term_sequence("run_terms", "_QQ_SOCIAL_QUERY_HINT_TERMS")
_QQ_SOCIAL_FRIEND_REQUEST_TERMS = _load_term_sequence("run_terms", "_QQ_SOCIAL_FRIEND_REQUEST_TERMS")
_QQ_SOCIAL_GROUP_REQUEST_TERMS = _load_term_sequence("run_terms", "_QQ_SOCIAL_GROUP_REQUEST_TERMS")
_QQ_SOCIAL_POLICY_FRIEND_TERMS = _load_term_sequence("run_terms", "_QQ_SOCIAL_POLICY_FRIEND_TERMS")
_QQ_SOCIAL_POLICY_GROUP_ADD_TERMS = _load_term_sequence("run_terms", "_QQ_SOCIAL_POLICY_GROUP_ADD_TERMS")
_QQ_SOCIAL_POLICY_GROUP_INVITE_TERMS = _load_term_sequence("run_terms", "_QQ_SOCIAL_POLICY_GROUP_INVITE_TERMS")
_QQ_SOCIAL_POLICY_ALL_TERMS = _load_term_sequence("run_terms", "_QQ_SOCIAL_POLICY_ALL_TERMS")
_QQ_SOCIAL_POLICY_QUERY_TERMS = _load_term_sequence("run_terms", "_QQ_SOCIAL_POLICY_QUERY_TERMS")
_QQ_SOCIAL_ENABLE_TERMS = _load_term_sequence("run_terms", "_QQ_SOCIAL_ENABLE_TERMS")
_QQ_SOCIAL_DISABLE_TERMS = _load_term_sequence("run_terms", "_QQ_SOCIAL_DISABLE_TERMS")

_QQ_GROUP_LISTEN_DISABLE_PATTERNS = _load_pattern_sequence("run_patterns", "_QQ_GROUP_LISTEN_DISABLE_PATTERNS")
_QQ_GROUP_LISTEN_ENABLE_PATTERNS = _load_pattern_sequence("run_patterns", "_QQ_GROUP_LISTEN_ENABLE_PATTERNS")
_QQ_SEND_INLINE_PATTERNS = _load_pattern_sequence("run_patterns", "_QQ_SEND_INLINE_PATTERNS")
_QQ_GROUP_ID_EXPLICIT_PATTERNS = _load_pattern_sequence("run_patterns", "_QQ_GROUP_ID_EXPLICIT_PATTERNS")
_QQ_GROUP_ID_ANYWHERE_RE = _load_single_pattern("run_patterns", "_QQ_GROUP_ID_ANYWHERE_RE")
_QQ_GROUP_MODERATION_DURATION_RE = _load_single_pattern("run_patterns", "_QQ_GROUP_MODERATION_DURATION_RE")
_QQ_GROUP_MODERATION_REASON_RE = _load_single_pattern("run_patterns", "_QQ_GROUP_MODERATION_REASON_RE")
_QQ_GROUP_MODERATION_USER_PATTERNS = _load_pattern_sequence("run_patterns", "_QQ_GROUP_MODERATION_USER_PATTERNS")
_QQ_JOINED_GROUP_LIST_AMBIGUOUS_TERMS = frozenset({"列一下群", "列出群"})
_QQ_JOINED_GROUP_LIST_EXACT_PATTERNS = (
    re.compile(r"^(?:给我|帮我|麻烦|请)?\s*(?:列一下|列出)\s*(?:加的|加入的|现在加的)?群(?:列表|名单)?[。！!？?\s]*$"),
)

_QQ_DEFAULT_TRIGGER_ALIASES = _load_term_sequence("napcat_terms", "_QQ_DEFAULT_TRIGGER_ALIASES")
_QQ_LOW_VALUE_IMAGE_HINTS = _load_term_sequence("napcat_terms", "_QQ_LOW_VALUE_IMAGE_HINTS")
_QQ_BUSY_SHORTCUT_MARKERS = _load_term_sequence("napcat_terms", "_QQ_BUSY_SHORTCUT_MARKERS")


def _looks_like_explicit_qq_intel_status_query(message_text: str) -> bool:
    body = str(message_text or "").strip()
    return bool(body) and any(marker in body for marker in ("情报员", "员工")) and any(
        term in body for term in _QQ_INTEL_WORKER_STATUS_QUERY_TERMS
    )


def _looks_like_explicit_qq_group_runtime_status_query(message_text: str) -> bool:
    body = str(message_text or "").strip()
    return bool(body) and has_followup_group_reference(body) and looks_like_shared_group_runtime_status_query(body)


def _looks_like_qq_background_status_query(message_text: str) -> bool:
    body = str(message_text or "").strip()
    if not body:
        return False
    if _looks_like_explicit_qq_intel_status_query(body):
        return False
    if _looks_like_explicit_qq_group_runtime_status_query(body):
        return False
    if any(term in body for term in _QQ_BACKGROUND_STATUS_QUERY_TERMS):
        return True
    if "还在吗" in body and "情报员" not in body and "员工" not in body:
        return True
    return False


def _looks_like_qq_runtime_status_query(message_text: str) -> bool:
    body = str(message_text or "").strip()
    if _looks_like_explicit_qq_intel_status_query(body):
        return False
    if _looks_like_explicit_qq_group_runtime_status_query(body):
        return False
    return bool(body) and any(term in body for term in _QQ_RUNTIME_STATUS_QUERY_TERMS)


def _looks_like_qq_joined_group_list_query(message_text: str) -> bool:
    body = str(message_text or "").strip()
    if not body:
        return False
    if any(term in body for term in _QQ_JOINED_GROUP_LIST_TERMS if term not in _QQ_JOINED_GROUP_LIST_AMBIGUOUS_TERMS):
        return True
    return any(pattern.search(body) for pattern in _QQ_JOINED_GROUP_LIST_EXACT_PATTERNS)


def _looks_like_qq_group_runtime_status_query(message_text: str) -> bool:
    body = str(message_text or "").strip()
    return bool(body) and any(term in body for term in _QQ_GROUP_STATUS_QUERY_TERMS)


def _looks_like_qq_group_moderation_candidate(message_text: str) -> bool:
    body = str(message_text or "").strip()
    if not body or any(term in body for term in _QQ_GROUP_MODERATION_NEGATION_TERMS):
        return False
    return any(term in body for term in _QQ_GROUP_MODERATION_MUTE_TERMS + _QQ_GROUP_MODERATION_KICK_TERMS)


def _looks_like_qq_social_request_list_query(message_text: str) -> bool:
    body = str(message_text or "").strip()
    if not body:
        return False
    has_query_hint = any(term in body for term in _QQ_SOCIAL_QUERY_HINT_TERMS)
    if "待处理申请" in body or "申请列表" in body:
        return True
    return has_query_hint and any(
        term in body for term in _QQ_SOCIAL_FRIEND_REQUEST_TERMS + _QQ_SOCIAL_GROUP_REQUEST_TERMS
    )


def _looks_like_qq_social_policy_candidate(message_text: str) -> bool:
    body = str(message_text or "").strip()
    if not body:
        return False
    if any(term in body for term in _QQ_SOCIAL_ENABLE_TERMS + _QQ_SOCIAL_DISABLE_TERMS):
        return any(
            term in body
            for term in (
                _QQ_SOCIAL_POLICY_FRIEND_TERMS
                + _QQ_SOCIAL_POLICY_GROUP_ADD_TERMS
                + _QQ_SOCIAL_POLICY_GROUP_INVITE_TERMS
                + _QQ_SOCIAL_POLICY_ALL_TERMS
            )
        )
    return any(term in body for term in _QQ_SOCIAL_POLICY_QUERY_TERMS) and any(
        term in body
        for term in (
            _QQ_SOCIAL_POLICY_FRIEND_TERMS
            + _QQ_SOCIAL_POLICY_GROUP_ADD_TERMS
            + _QQ_SOCIAL_POLICY_GROUP_INVITE_TERMS
            + _QQ_SOCIAL_POLICY_ALL_TERMS
        )
    )


def _looks_like_qq_active_session_inline_candidate(
    message_text: str,
    *,
    is_admin: bool,
    explicit_followup: bool,
) -> bool:
    body = str(message_text or "").strip()
    if not body:
        return False

    if _looks_like_qq_background_status_query(body) or _looks_like_qq_runtime_status_query(body):
        return explicit_followup or is_admin
    if explicit_followup and "还在吗" in body:
        return True

    if not is_admin:
        return False

    if _looks_like_qq_joined_group_list_query(body):
        return True
    if "情报员" in body or "员工" in body:
        return True
    if _looks_like_qq_group_runtime_status_query(body):
        return True
    if any(term in body for term in ("监听", "采集", "日报")):
        return True
    if _looks_like_qq_group_moderation_candidate(body):
        return True
    if _looks_like_qq_social_request_list_query(body) or _looks_like_qq_social_policy_candidate(body):
        return True
    return False


def _qq_group_has_visible_bot_address(message: str) -> bool:
    body = str(message or "").strip()
    return bool(body) and any(name in body for name in _QQ_VISIBLE_NAME_ALIASES)


def _looks_like_qq_group_request_text(message: str) -> bool:
    body = str(message or "").strip().lower()
    if not body:
        return False
    if any(token in body for token in ("?", "？")):
        return True
    return any(marker in body for marker in _QQ_GROUP_REQUEST_HINT_TERMS)


def _looks_like_qq_media_message(message: str) -> bool:
    body = str(message or "").strip().lower()
    if not body:
        return False
    return any(marker in body for marker in _QQ_MEDIA_PLACEHOLDER_MARKERS)


def _looks_like_qq_runtime_short_query(message_text: str) -> bool:
    body = str(message_text or "").strip()
    if not body:
        return False
    if body in _QQ_RUNTIME_STATUS_SHORT_TERMS:
        return True
    return body.endswith(("在吗", "在嘛", "挂了吗"))
