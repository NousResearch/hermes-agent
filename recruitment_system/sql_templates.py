"""Template-first SQL generation for RecruitmentSystem queries."""

from __future__ import annotations

import re
from typing import Any

from .config import QuerySettings, TableMapping
from .intent import (
    MY_ATTENDANCE_BY_TIME_RANGE,
    MY_ATTENDANCE_RECENT_WEEK,
    RECRUITING_JOB_DETAIL,
    RECRUITING_JOB_LIST,
)
from .models import SQLPlan, TimeRange, UserContext

_IDENTIFIER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def build_sql_plan(
    *,
    intent: str,
    question: str,
    user: UserContext,
    mapping: TableMapping,
    settings: QuerySettings,
    time_range: TimeRange | None = None,
    slots: dict[str, Any] | None = None,
) -> SQLPlan:
    if intent == RECRUITING_JOB_LIST:
        return build_recruiting_job_list_sql(user=user, mapping=mapping, settings=settings)
    if intent == RECRUITING_JOB_DETAIL:
        job_name = (slots or {}).get("job_name")
        return build_recruiting_job_detail_sql(
            user=user,
            mapping=mapping,
            settings=settings,
            job_name=job_name,
        )
    if intent in {MY_ATTENDANCE_RECENT_WEEK, MY_ATTENDANCE_BY_TIME_RANGE}:
        if time_range is None:
            raise ValueError("Attendance SQL requires a time range")
        return build_attendance_sql(
            intent=intent,
            user=user,
            mapping=mapping,
            settings=settings,
            time_range=time_range,
        )
    raise ValueError(f"Unsupported intent: {intent}")


def build_recruiting_job_list_sql(
    *,
    user: UserContext,
    mapping: TableMapping,
    settings: QuerySettings,
) -> SQLPlan:
    fields = list(mapping.job_safe_fields)
    select_clause = _select_clause(mapping.job_fields, fields)
    where_parts = []
    params: dict[str, Any] = {}

    if "is_deleted" in mapping.job_fields:
        where_parts.append(f"{_q(mapping.job_fields['is_deleted'])} = 0")
    if "tenant_id" in mapping.job_fields and user.tenant_id:
        where_parts.append(f"{_q(mapping.job_fields['tenant_id'])} = %(tenant_id)s")
        params["tenant_id"] = user.tenant_id
    recruit_field = mapping.require_job_field("recruit_status")
    status_placeholders = []
    for idx, value in enumerate(mapping.recruiting_status_values):
        key = f"recruit_status_{idx}"
        status_placeholders.append(f"%({key})s")
        params[key] = value
    where_parts.append(f"{_q(recruit_field)} IN ({', '.join(status_placeholders)})")

    order_field = mapping.job_fields.get("update_time") or mapping.job_fields.get("create_time")
    order_clause = f" ORDER BY {_q(order_field)} DESC" if order_field else ""
    limit = min(settings.default_limit, settings.max_limit)
    sql = (
        f"SELECT {select_clause} FROM {_q_table(mapping.job_table)} "
        f"WHERE {' AND '.join(where_parts)}{order_clause} LIMIT {limit}"
    )
    return SQLPlan(sql=sql, params=params, intent=RECRUITING_JOB_LIST, logical_tables=("job",))


def build_recruiting_job_detail_sql(
    *,
    user: UserContext,
    mapping: TableMapping,
    settings: QuerySettings,
    job_name: str | None,
) -> SQLPlan:
    fields = list(mapping.job_detail_safe_fields)
    select_clause = _select_clause(mapping.job_fields, fields)
    where_parts = []
    params: dict[str, Any] = {}

    if "is_deleted" in mapping.job_fields:
        where_parts.append(f"{_q(mapping.job_fields['is_deleted'])} = 0")
    if "tenant_id" in mapping.job_fields and user.tenant_id:
        where_parts.append(f"{_q(mapping.job_fields['tenant_id'])} = %(tenant_id)s")
        params["tenant_id"] = user.tenant_id
    if job_name:
        where_parts.append(f"{_q(mapping.require_job_field('job_name'))} LIKE %(job_name)s")
        params["job_name"] = f"%{job_name}%"

    order_field = mapping.job_fields.get("update_time") or mapping.job_fields.get("create_time")
    order_clause = f" ORDER BY {_q(order_field)} DESC" if order_field else ""
    limit = min(10, settings.max_limit)
    sql = (
        f"SELECT {select_clause} FROM {_q_table(mapping.job_table)} "
        f"WHERE {' AND '.join(where_parts)}{order_clause} LIMIT {limit}"
    )
    return SQLPlan(sql=sql, params=params, intent=RECRUITING_JOB_DETAIL, logical_tables=("job",))


def build_attendance_sql(
    *,
    intent: str,
    user: UserContext,
    mapping: TableMapping,
    settings: QuerySettings,
    time_range: TimeRange,
) -> SQLPlan:
    fields = list(mapping.attendance_safe_fields)
    select_clause = _select_clause(mapping.attendance_fields, fields)
    params: dict[str, Any] = {
        "current_user_id": user.user_id,
        **time_range.as_params(),
    }
    where_parts = []
    if "is_deleted" in mapping.attendance_fields:
        where_parts.append(f"{_q(mapping.attendance_fields['is_deleted'])} = 0")
    if "tenant_id" in mapping.attendance_fields and user.tenant_id:
        where_parts.append(f"{_q(mapping.attendance_fields['tenant_id'])} = %(tenant_id)s")
        params["tenant_id"] = user.tenant_id
    where_parts.extend(
        [
            f"{_q(mapping.require_attendance_field('user_id'))} = %(current_user_id)s",
            f"{_q(mapping.require_attendance_field('attendance_date'))} >= %(start_date)s",
            f"{_q(mapping.require_attendance_field('attendance_date'))} < %(end_date)s",
        ]
    )
    limit = min(settings.default_limit, settings.max_limit)
    sql = (
        f"SELECT {select_clause} FROM {_q_table(mapping.attendance_table)} "
        f"WHERE {' AND '.join(where_parts)} "
        f"ORDER BY {_q(mapping.require_attendance_field('attendance_date'))} DESC LIMIT {limit}"
    )
    return SQLPlan(
        sql=sql,
        params=params,
        intent=intent,
        logical_tables=("attendance",),
        time_range=time_range,
    )


def _select_clause(field_mapping: dict[str, str], logical_fields: list[str]) -> str:
    return ", ".join(
        f"{_q(field_mapping[logical])} AS {_q(logical)}"
        for logical in logical_fields
    )


def _q(identifier: str) -> str:
    if not _IDENTIFIER_RE.match(identifier):
        raise ValueError(f"Unsafe SQL identifier: {identifier}")
    return f"`{identifier}`"


def _q_table(identifier: str) -> str:
    parts = identifier.split(".")
    if not parts or any(not _IDENTIFIER_RE.match(part) for part in parts):
        raise ValueError(f"Unsafe SQL table identifier: {identifier}")
    return ".".join(f"`{part}`" for part in parts)
