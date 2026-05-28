"""Rule-first intent recognition for RecruitmentSystem queries."""

from __future__ import annotations

import re

from .models import IntentResult

RECRUITING_JOB_LIST = "recruiting_job_list"
RECRUITING_JOB_DETAIL = "recruiting_job_detail"
MY_ATTENDANCE_RECENT_WEEK = "my_attendance_recent_week"
MY_ATTENDANCE_BY_TIME_RANGE = "my_attendance_by_time_range"
UNKNOWN_QUERY_INTENT = "unknown_query_intent"

_JOB_WORDS = ("岗位", "职位", "招聘", "在招", "招哪些", "开放招聘")
_JOB_LIST_WORDS = ("有哪些", "哪些", "列出", "当前", "现在", "还在招聘", "正在招聘", "开放")
_JOB_DETAIL_WORDS = ("要求", "职责", "薪资", "范围", "详情", "描述", "任职", "工作内容")
_ATTENDANCE_WORDS = ("考勤", "打卡", "迟到", "早退", "缺卡", "出勤")
_RECENT_WEEK_WORDS = ("最近一周", "最近七天", "最近7天", "近一周", "近7天", "这周", "本周")
_TIME_RANGE_WORDS = (
    "今天",
    "昨天",
    "上周",
    "本月",
    "上个月",
    "月份",
    "月",
    "到",
    "至",
)


def recognize_intent(question: str) -> IntentResult:
    text = (question or "").strip()
    compact = re.sub(r"\s+", "", text)
    if not compact:
        return IntentResult(UNKNOWN_QUERY_INTENT, 0.0)

    if _contains_any(compact, _ATTENDANCE_WORDS):
        if _contains_any(compact, _RECENT_WEEK_WORDS):
            return IntentResult(MY_ATTENDANCE_RECENT_WEEK, 0.92)
        if _looks_like_time_range(compact) or _contains_any(compact, _TIME_RANGE_WORDS):
            return IntentResult(MY_ATTENDANCE_BY_TIME_RANGE, 0.86)
        return IntentResult(MY_ATTENDANCE_RECENT_WEEK, 0.72)

    if _contains_any(compact, _JOB_WORDS):
        if _contains_any(compact, _JOB_DETAIL_WORDS):
            return IntentResult(
                RECRUITING_JOB_DETAIL,
                0.82,
                slots={"job_name": extract_job_name(compact)},
            )
        if _contains_any(compact, _JOB_LIST_WORDS):
            return IntentResult(RECRUITING_JOB_LIST, 0.9)

    return IntentResult(UNKNOWN_QUERY_INTENT, 0.0)


def extract_job_name(question: str) -> str | None:
    text = (question or "").strip()
    patterns = [
        r"(.+?)(?:这个)?岗位(?:要求|职责|薪资|详情|描述)",
        r"(.+?)(?:招聘)?(?:要求|职责|薪资范围|薪资|详情)",
    ]
    for pattern in patterns:
        match = re.search(pattern, text)
        if not match:
            continue
        candidate = match.group(1)
        candidate = re.sub(r"^(查询|查看|了解|请问|我想知道)", "", candidate)
        candidate = candidate.strip("，。,. 的")
        if candidate and candidate not in {"某个", "这个"}:
            return candidate
    return None


def _contains_any(text: str, terms: tuple[str, ...]) -> bool:
    return any(term in text for term in terms)


def _looks_like_time_range(text: str) -> bool:
    return bool(
        re.search(r"\d{4}-\d{1,2}-\d{1,2}", text)
        or re.search(r"\d{1,2}\s*月份?", text)
    )
