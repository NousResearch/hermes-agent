"""Pure request-parsing helpers for QQ intel-worker oral shortcuts."""

from __future__ import annotations

import re
from typing import Iterable

from gateway.group_control_intents import looks_like_group_report_now_request
from gateway.qq_intents import (
    _QQ_INTEL_CONTEXT_TERMS,
    _QQ_INTEL_HIRE_TERMS,
    _QQ_INTEL_PAUSE_TERMS,
    _QQ_INTEL_RESUME_TERMS,
    _QQ_INTEL_STOP_TERMS,
    _QQ_INTEL_WORKER_NAME_STOPWORDS,
    _QQ_INTEL_WORKER_STATUS_QUERY_TERMS,
    _QQ_VISIBLE_NAME_ALIASES,
)

_QQ_INTEL_WORKER_BOUNDARY_TERMS = tuple(
    sorted(
        {
            "现在",
            "马上",
            "立刻",
            "先",
            "暂停",
            "恢复",
            "停止",
            "继续",
            "汇报",
            "回报",
            "状态",
            "情况",
            "任务",
            "监听",
            "采集",
            "潜伏",
            "刺探",
            *_QQ_INTEL_WORKER_STATUS_QUERY_TERMS,
        },
        key=len,
        reverse=True,
    )
)
_QQ_INTEL_WORKER_BOUNDARY = (
    r"(?=(?:"
    + "|".join(re.escape(term) for term in _QQ_INTEL_WORKER_BOUNDARY_TERMS)
    + r"|，|,|。|；|;|\s|$))"
)
_QQ_INTEL_WORKER_SURROUNDING_TOKENS_RE = re.compile(r"^(?:现在|马上|立刻|先)+|(?:现在|马上|立刻|先)+$")
_QQ_INTEL_INVALID_WORKER_NAME_TERMS = frozenset(_QQ_INTEL_WORKER_NAME_STOPWORDS) | frozenset(
    _QQ_INTEL_WORKER_STATUS_QUERY_TERMS
)
_QQ_VISIBLE_BOT_ALIAS_NAMES = frozenset(
    alias.lstrip("@").strip() for alias in _QQ_VISIBLE_NAME_ALIASES if alias.lstrip("@").strip()
)


def _normalize_qq_worker_name_candidate(candidate: str) -> str:
    normalized = _QQ_INTEL_WORKER_SURROUNDING_TOKENS_RE.sub("", str(candidate or "").strip()).strip()
    if normalized in _QQ_INTEL_INVALID_WORKER_NAME_TERMS:
        return ""
    return normalized


def extract_qq_worker_name(message_text: str) -> str:
    body = str(message_text or "").strip()
    if not body:
        return ""
    patterns = (
        re.compile(
            r"(?:情报员|员工)\s*[：:，,\s]*"
            r"([A-Za-z0-9_\-\u4e00-\u9fff]{1,20}?)"
            + _QQ_INTEL_WORKER_BOUNDARY
        ),
        re.compile(
            r"让\s*([A-Za-z0-9_\-\u4e00-\u9fff]{1,20}?)"
            + _QQ_INTEL_WORKER_BOUNDARY
            + r"\s*(?:现在|马上|立刻|先)?(?:汇报|回报|暂停|恢复|停止|继续)"
        ),
    )
    for pattern in patterns:
        match = pattern.search(body)
        if not match:
            continue
        candidate = _normalize_qq_worker_name_candidate(match.group(1) or "")
        if not candidate:
            continue
        return candidate
    return ""


def looks_like_qq_intel_worker_context(message_text: str) -> bool:
    body = str(message_text or "").strip()
    if not body:
        return False
    return any(term in body for term in _QQ_INTEL_CONTEXT_TERMS)


def extract_qq_oral_intel_hire_objective(
    message_text: str,
    *,
    worker_name: str,
    target_group: str,
) -> str | None:
    body = str(message_text or "").strip()
    if not body:
        return None
    cleaned = body
    if worker_name:
        cleaned = re.sub(rf"(情报员|员工)\s*[：:，,\s]*{re.escape(worker_name)}", "", cleaned)
    cleaned = re.sub(r"(招一个|招个|招|安排一个|安排个|安排|派一个|派个|派|新增一个|新增个|新增)", "", cleaned)
    group_id = str(target_group or "").replace("group:", "").strip()
    if group_id:
        cleaned = re.sub(rf"去\s*{re.escape(group_id)}\s*(这个)?群", "", cleaned)
        cleaned = cleaned.replace(group_id, "")
    cleaned = cleaned.replace("私聊向我汇报", "")
    cleaned = cleaned.replace("私聊发我", "")
    cleaned = cleaned.replace("每天向我汇报", "")
    cleaned = cleaned.replace("每天给我汇报", "")
    cleaned = cleaned.replace("每天给我日报", "")
    cleaned = cleaned.replace("每天汇报", "")
    cleaned = cleaned.replace("，", " ").replace(",", " ")
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" 。；;")
    return cleaned or None


def match_qq_intel_control_request(
    *,
    source,
    body: str,
    admin_ids_configured: bool,
    is_admin_user: bool,
    looks_like_joined_group_list_query,
    extract_worker_name,
    looks_like_worker_context,
    known_worker_names: Iterable[str],
    target_extractor,
    report_target_resolver,
    hire_objective_extractor,
) -> tuple[dict[str, object] | None, str | None]:
    normalized_body = str(body or "").strip()
    if not normalized_body:
        return None, None
    if not admin_ids_configured:
        return None, None
    if not is_admin_user:
        return None, None

    if looks_like_joined_group_list_query(normalized_body):
        return {"action": "list_joined_groups"}, None

    explicit_worker_context = any(marker in normalized_body for marker in ("情报员", "员工"))
    worker_name = str(extract_worker_name(normalized_body) or "").strip()
    has_worker_status_query = any(term in normalized_body for term in _QQ_INTEL_WORKER_STATUS_QUERY_TERMS)
    has_intel_context = bool(looks_like_worker_context(normalized_body)) or (
        explicit_worker_context and has_worker_status_query
    )
    normalized_known_names = {
        str(name or "").strip()
        for name in (known_worker_names or [])
        if str(name or "").strip()
    }
    worker_reference_is_explicit = explicit_worker_context or (
        worker_name in normalized_known_names and worker_name not in _QQ_VISIBLE_BOT_ALIAS_NAMES
    )

    if worker_name and has_intel_context and worker_reference_is_explicit:
        if any(term in normalized_body for term in _QQ_INTEL_PAUSE_TERMS):
            return {"action": "pause_worker", "worker_name": worker_name}, None
        if any(term in normalized_body for term in _QQ_INTEL_STOP_TERMS):
            return {"action": "stop_worker", "worker_name": worker_name}, None
        if any(term in normalized_body for term in _QQ_INTEL_RESUME_TERMS):
            return {"action": "resume_worker", "worker_name": worker_name}, None
        if looks_like_group_report_now_request(normalized_body):
            return {
                "action": "run_report_now",
                "worker_name": worker_name,
                "manual_report_target": report_target_resolver(
                    source,
                    normalized_body,
                    prefer_dm=True,
                ),
            }, None
        if any(term in normalized_body for term in _QQ_INTEL_WORKER_STATUS_QUERY_TERMS):
            return {"action": "get_worker", "worker_name": worker_name}, None

    if "情报员" in normalized_body and any(term in normalized_body for term in _QQ_INTEL_HIRE_TERMS):
        if not worker_name:
            return None, "要招情报员，请把名字说清楚。"
        target_group = target_extractor(source, normalized_body)
        if not target_group:
            return None, "要安排情报员，请直接说清群号，或者在目标群里明确说“这个群”。"
        tool_args: dict[str, object] = {
            "action": "hire_worker",
            "worker_name": worker_name,
            "target_group": target_group,
            "daily_report_target": report_target_resolver(
                source,
                normalized_body,
                prefer_dm=True,
            ),
            "manual_report_target": report_target_resolver(
                source,
                normalized_body,
                prefer_dm=True,
            ),
        }
        objective = hire_objective_extractor(
            normalized_body,
            worker_name=worker_name,
            target_group=target_group,
        )
        if objective:
            tool_args["objective"] = objective
        return tool_args, None

    return None, None
