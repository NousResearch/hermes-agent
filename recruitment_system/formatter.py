"""Natural-language formatting for RecruitmentSystem query results."""

from __future__ import annotations

from typing import Any

from .intent import (
    MY_ATTENDANCE_BY_TIME_RANGE,
    MY_ATTENDANCE_RECENT_WEEK,
    RECRUITING_JOB_DETAIL,
    RECRUITING_JOB_LIST,
)
from .models import TimeRange


def format_answer(intent: str, rows: list[dict[str, Any]], *, time_range: TimeRange | None = None) -> str:
    if intent == RECRUITING_JOB_LIST:
        return _format_job_list(rows)
    if intent == RECRUITING_JOB_DETAIL:
        return _format_job_detail(rows)
    if intent in {MY_ATTENDANCE_RECENT_WEEK, MY_ATTENDANCE_BY_TIME_RANGE}:
        return _format_attendance(rows, time_range=time_range)
    return "暂时无法识别该查询，请尝试查询招聘岗位或我的考勤数据。"


def _format_job_list(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return "当前未查询到正在招聘的岗位。"

    names = [_string(row.get("job_name")) for row in rows if row.get("job_name")]
    if not names:
        return f"当前查询到 {len(rows)} 条正在招聘岗位记录。"

    answer = f"当前正在招聘的岗位有 {'、'.join(names)}。"
    first = rows[0]
    details = []
    if first.get("department_name"):
        details.append(f"{_string(first.get('job_name'))} 所属部门为 {_string(first.get('department_name'))}")
    if first.get("headcount") is not None:
        details.append(f"计划招聘 {first.get('headcount')} 人")
    if details:
        answer += "其中 " + "，".join(details) + "。"
    return answer


def _format_job_detail(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return "未查询到匹配的岗位详情。"
    row = rows[0]
    name = _string(row.get("job_name") or "该岗位")
    parts = [f"{name}"]
    if row.get("department_name"):
        parts.append(f"所属部门为 {_string(row.get('department_name'))}")
    if row.get("job_requirement"):
        parts.append(f"岗位要求：{_string(row.get('job_requirement'))}")
    if row.get("job_description"):
        parts.append(f"岗位职责：{_string(row.get('job_description'))}")
    if row.get("salary_min") is not None or row.get("salary_max") is not None:
        parts.append(f"薪资范围为 {_string(row.get('salary_min'))} - {_string(row.get('salary_max'))}")
    return "，".join(parts) + "。"


def _format_attendance(rows: list[dict[str, Any]], *, time_range: TimeRange | None) -> str:
    label = time_range.label if time_range else "该时间范围"
    if not rows:
        return f"{label}未查询到你的考勤记录。"

    total = len(rows)
    late_rows = [row for row in rows if _number(row.get("late_minutes")) > 0 or "迟到" in _string(row.get("attendance_status"))]
    early_rows = [
        row
        for row in rows
        if _number(row.get("early_leave_minutes")) > 0 or "早退" in _string(row.get("attendance_status"))
    ]
    missing_rows = [row for row in rows if "缺卡" in _string(row.get("attendance_status"))]
    normal_count = sum(
        1
        for row in rows
        if "正常" in _string(row.get("attendance_status"))
        and _number(row.get("late_minutes")) <= 0
        and _number(row.get("early_leave_minutes")) <= 0
    )

    answer = (
        f"{label}你共有 {total} 条考勤记录，其中正常 {normal_count} 天，"
        f"迟到 {len(late_rows)} 天，早退 {len(early_rows)} 天，缺卡 {len(missing_rows)} 天。"
    )
    if late_rows:
        row = late_rows[0]
        answer += (
            f"迟到发生在 {_string(row.get('attendance_date'))}，"
            f"上班打卡时间为 {_string(row.get('check_in_time'))}，"
            f"迟到 {_number(row.get('late_minutes')):g} 分钟。"
        )
    return answer


def _string(value: Any) -> str:
    if value is None:
        return ""
    return str(value)


def _number(value: Any) -> float:
    if value in (None, ""):
        return 0.0
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0
