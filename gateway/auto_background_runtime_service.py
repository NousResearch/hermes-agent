"""Shared runtime helpers for auto-background gateway dispatch."""

from __future__ import annotations

import json
import logging
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Iterable, Optional

from gateway.group_visible_addressing_platform_specs import (
    get_group_visible_addressing_platform_spec,
)
from gateway.platforms.base import MessageType

logger = logging.getLogger(__name__)

_DATA_PATH = Path(__file__).resolve().parent / "data" / "auto_background_intents.json"
_DEFAULT_INTENTS_DATA = {
    "shortcuts": [
        "继续",
        "继续啊",
        "继续!",
        "继续！",
        "接着做",
        "继续处理",
        "按你建议做",
        "马上做",
        "整套",
    ],
    "action_terms": [
        "修复", "排查", "审查", "部署", "实现", "开发", "收尾", "调查",
        "分析", "制定", "设计", "重启", "同步", "处理", "看看日志",
        "上服务器", "查原因", "完整实施", "全部修复", "计划清单",
    ],
    "domain_terms": [
        "服务器", "日志", "配置", "代码", "接口", "模型", "端点", "并发",
        "群聊", "私聊", "gateway", "cron", "qq", "napcat", "服务", "任务",
        "问题", "故障",
    ],
    "worker_assignment": {
        "lead_markers": ["让", "叫", "安排", "交给", "给", "找", "请", "麻烦"],
        "tail_markers": [
            "继续",
            "去",
            "来",
            "做",
            "处理",
            "跟进",
            "修",
            "查",
            "看",
            "优化",
            "打磨",
            "润色",
            "改",
            "整",
        ],
    },
}


@lru_cache(maxsize=1)
def _load_auto_background_intents_data() -> dict[str, Any]:
    try:
        raw = json.loads(_DATA_PATH.read_text(encoding="utf-8"))
    except FileNotFoundError:
        logger.warning("Auto-background intents data file missing: %s", _DATA_PATH)
        return dict(_DEFAULT_INTENTS_DATA)
    except Exception as exc:
        logger.warning(
            "Failed to load auto-background intents from %s: %s",
            _DATA_PATH,
            exc,
        )
        return dict(_DEFAULT_INTENTS_DATA)
    if not isinstance(raw, dict):
        logger.warning("Auto-background intents data must be a dict: %s", _DATA_PATH)
        return dict(_DEFAULT_INTENTS_DATA)
    return raw


def _load_term_sequence(key: str) -> tuple[str, ...]:
    payload = _load_auto_background_intents_data().get(key)
    if not isinstance(payload, list):
        payload = _DEFAULT_INTENTS_DATA.get(key) or []
    normalized: list[str] = []
    for item in payload:
        text = str(item or "").strip()
        if text:
            normalized.append(text)
    return tuple(normalized)


def _load_nested_term_sequence(section: str, key: str) -> tuple[str, ...]:
    payload = (_load_auto_background_intents_data().get(section) or {}).get(key)
    if not isinstance(payload, list):
        payload = ((_DEFAULT_INTENTS_DATA.get(section) or {}).get(key) or [])
    normalized: list[str] = []
    for item in payload:
        text = str(item or "").strip()
        if text:
            normalized.append(text)
    return tuple(normalized)


def _auto_background_shortcuts() -> tuple[str, ...]:
    return _load_term_sequence("shortcuts")


def _auto_background_action_terms() -> tuple[str, ...]:
    return _load_term_sequence("action_terms")


def _auto_background_domain_terms() -> tuple[str, ...]:
    return _load_term_sequence("domain_terms")


def _worker_assignment_lead_markers() -> tuple[str, ...]:
    return _load_nested_term_sequence("worker_assignment", "lead_markers")


def _worker_assignment_tail_markers() -> tuple[str, ...]:
    return _load_nested_term_sequence("worker_assignment", "tail_markers")


def contains_any(text: str, terms: tuple[str, ...]) -> bool:
    """Return True when any term is present in the normalized text."""
    if not text:
        return False
    lowered = text.lower()
    return any(term in lowered for term in terms)


def is_auto_background_shortcut(text: str) -> bool:
    """Return True when the message is a bare follow-up shortcut like '继续'."""
    compact = str(text or "").strip().lower()
    return compact in _auto_background_shortcuts()


def looks_like_auto_background_work_request(text: str) -> bool:
    """Return True when the message itself looks like a real work assignment."""
    body = str(text or "").strip()
    if not body:
        return False

    compact = body.lower()
    action_hits = {term for term in _auto_background_action_terms() if term in body}
    domain_hits = {term for term in _auto_background_domain_terms() if term in compact}
    overlapping_hits = {term for term in action_hits if term in domain_hits}
    distinct_hits = (action_hits | domain_hits) - overlapping_hits
    has_structure = (
        len(body) >= 80
        or body.count("\n") >= 2
        or "```" in body
        or body.count("http://") + body.count("https://") > 0
    )
    has_action = bool(action_hits)

    if distinct_hits and has_action:
        return True
    if has_structure and has_action:
        return True
    if "全部修复" in body or "完整实施" in body or "上服务器" in body:
        return True
    return False


def looks_like_explicit_worker_assignment(text: str, worker_names: list[str]) -> bool:
    """Return True when a named employee is being explicitly assigned work."""
    body = str(text or "").strip()
    if not body:
        return False
    if any(mark in body for mark in ("?", "？")):
        return False
    if is_auto_background_shortcut(body) or looks_like_auto_background_work_request(body):
        return True

    lead_markers = _worker_assignment_lead_markers()
    tail_markers = _worker_assignment_tail_markers()
    for name in worker_names:
        candidate = str(name or "").strip()
        if not candidate or candidate not in body:
            continue
        before, _, after = body.partition(candidate)
        if any(marker in before[-4:] for marker in lead_markers):
            return True
        if any(after.startswith(marker) for marker in tail_markers):
            return True
    return False


def should_auto_background_message(
    *,
    auto_background_work_enabled: bool,
    event: Any,
    message_text: str,
) -> bool:
    """Heuristic: detach obvious work assignments to keep chat responsive."""
    if not auto_background_work_enabled:
        return False
    if event.get_command():
        return False
    if getattr(event, "message_type", None) != MessageType.TEXT:
        return False
    if getattr(event, "media_urls", None):
        return False

    body = str(message_text or "").strip()
    if not body:
        return False
    return looks_like_auto_background_work_request(body)


def history_suggests_auto_background_work(
    conversation_history: Optional[list[dict[str, Any]]] = None,
) -> bool:
    """Return True when recent history clearly looks like an ongoing work task."""
    recent_parts: list[str] = []
    for item in list(conversation_history or [])[-4:]:
        if not isinstance(item, dict):
            continue
        role = str(item.get("role") or "").strip().lower()
        if role and role != "user":
            continue
        content = item.get("content")
        if content:
            recent_parts.append(str(content))
    recent_text = "\n".join(recent_parts)
    if not recent_text.strip():
        return False

    lowered = recent_text.lower()
    action_hits = {term for term in _auto_background_action_terms() if term in recent_text}
    domain_hits = {term for term in _auto_background_domain_terms() if term in lowered}
    overlapping_hits = {term for term in action_hits if term in domain_hits}
    distinct_hits = (action_hits | domain_hits) - overlapping_hits
    return bool(action_hits) and bool(distinct_hits)


def resolve_employee_background_dispatch(
    message_text: str,
    *,
    employee_routes: Iterable[dict[str, Any]] | None,
    conversation_history: Optional[list[dict[str, Any]]] = None,
) -> dict[str, Any] | None:
    """Return worker-routing metadata for tasks that should run as employees."""
    body = str(message_text or "").strip()
    if not body:
        return None

    routes = list(employee_routes or [])
    if not routes:
        return None

    current_text = body.lower()
    recent_context_parts: list[str] = [body]
    for item in list(conversation_history or [])[-4:]:
        if not isinstance(item, dict):
            continue
        content = item.get("content")
        if content:
            recent_context_parts.append(str(content))
    recent_context = "\n".join(recent_context_parts).lower()
    shortcut_followup = is_auto_background_shortcut(body)

    for route in routes:
        route_names = [str(route.get("worker_name") or "").strip(), *list(route.get("aliases") or [])]
        match_modes = {str(mode or "").strip().lower() for mode in (route.get("match_modes") or ())}
        explicit_worker_mention = any(name and name in body for name in route_names)
        current_has_action = contains_any(current_text, tuple(route.get("action_terms") or ()))
        current_has_subject = contains_any(current_text, tuple(route.get("subject_terms") or ()))
        current_has_pain = contains_any(current_text, tuple(route.get("pain_terms") or ()))
        combined_has_action = contains_any(recent_context, tuple(route.get("action_terms") or ()))
        combined_has_subject = contains_any(recent_context, tuple(route.get("subject_terms") or ()))
        combined_has_pain = contains_any(recent_context, tuple(route.get("pain_terms") or ()))

        if "explicit" in match_modes and explicit_worker_mention and (
            shortcut_followup
            or current_has_action
            or current_has_subject
            or current_has_pain
            or looks_like_auto_background_work_request(body)
            or looks_like_explicit_worker_assignment(body, route_names)
        ):
            return {
                "worker_name": str(route["worker_name"]),
                "preloaded_skills": list(route.get("preloaded_skills") or []),
            }
        if "heuristic" not in match_modes:
            continue
        if current_has_subject and (current_has_action or current_has_pain):
            return {
                "worker_name": str(route["worker_name"]),
                "preloaded_skills": list(route.get("preloaded_skills") or []),
            }
        if (current_has_action or current_has_pain) and combined_has_subject:
            return {
                "worker_name": str(route["worker_name"]),
                "preloaded_skills": list(route.get("preloaded_skills") or []),
            }
        if shortcut_followup and combined_has_subject and (combined_has_action or combined_has_pain):
            return {
                "worker_name": str(route["worker_name"]),
                "preloaded_skills": list(route.get("preloaded_skills") or []),
            }
    return None


def resolve_auto_background_dispatch(
    event: Any,
    message_text: str,
    *,
    auto_background_work_enabled: bool,
    employee_routes: Iterable[dict[str, Any]] | None,
    conversation_history: Optional[list[dict[str, Any]]] = None,
    group_visible_address_checker: Callable[[str], bool] | None = None,
) -> dict[str, Any] | None:
    """Return auto-background metadata when this turn should detach."""
    if not auto_background_work_enabled:
        return None
    if event.get_command():
        return None
    if getattr(event, "message_type", None) != MessageType.TEXT:
        return None
    if getattr(event, "media_urls", None):
        return None

    body = str(message_text or "").strip()
    if not body:
        return None
    source = getattr(event, "source", None)
    if getattr(source, "chat_type", "") == "group":
        is_shortcut = is_auto_background_shortcut(body)
        visible_address_checker = (
            group_visible_address_checker
            or get_group_visible_addressing_platform_spec(getattr(source, "platform", None)).has_visible_bot_address
        )
        if not visible_address_checker(body) and not (
            is_shortcut and history_suggests_auto_background_work(conversation_history)
        ):
            return None

    dispatch = resolve_employee_background_dispatch(
        body,
        employee_routes=employee_routes,
        conversation_history=conversation_history,
    )
    if dispatch:
        return dispatch
    if is_auto_background_shortcut(body):
        if history_suggests_auto_background_work(conversation_history):
            return {
                "worker_name": "",
                "preloaded_skills": [],
            }
        return None
    if should_auto_background_message(
        auto_background_work_enabled=auto_background_work_enabled,
        event=event,
        message_text=body,
    ):
        return {
            "worker_name": "",
            "preloaded_skills": [],
        }
    return None


def format_auto_background_ack(prompt: str, task_id: str, *, worker_name: str = "") -> str:
    """Return the immediate acknowledgement for an auto-detached job."""
    preview = prompt[:60] + ("..." if len(prompt) > 60 else "")
    lead = f"🛠️ 这活我交给{worker_name}后台处理了。" if worker_name else "🛠️ 这事我转后台做了。"
    return (
        f"{lead}\n"
        f"任务ID：`{task_id}`\n"
        f"内容：{preview}\n"
        "你继续发消息就行，我做完会回来汇报。"
    )
