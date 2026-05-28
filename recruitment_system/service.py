"""High-level RecruitmentSystem AI query service."""

from __future__ import annotations

import json
import logging
import time
import uuid
from datetime import date
from typing import Any

from .config import QuerySettings, RecruitmentConfigError, TableMapping
from .db import MySQLQueryClient, QueryExecutor, RecruitmentQueryError
from .formatter import format_answer
from .intent import (
    MY_ATTENDANCE_BY_TIME_RANGE,
    MY_ATTENDANCE_RECENT_WEEK,
    UNKNOWN_QUERY_INTENT,
    recognize_intent,
)
from .models import QueryRequest, QueryResponse, QueryResult, UserContext
from .permissions import can_query_attendance
from .sql_guard import SQLGuard
from .sql_templates import build_sql_plan
from .time_ranges import parse_time_range

logger = logging.getLogger(__name__)
audit_logger = logging.getLogger("recruitment_system.audit")


def query_recruitment_system(
    request: QueryRequest,
    *,
    executor: QueryExecutor | None = None,
    mapping: TableMapping | None = None,
    settings: QuerySettings | None = None,
    today: date | None = None,
) -> QueryResponse:
    trace_id = uuid.uuid4().hex
    start = time.perf_counter()
    intent_name = UNKNOWN_QUERY_INTENT
    generated_sql = None
    safe = False
    row_count = 0
    error_message = None

    try:
        if not request.question.strip():
            return _error(trace_id, UNKNOWN_QUERY_INTENT, "INVALID_REQUEST", "question 不能为空")
        if not request.user_id.strip():
            return _error(trace_id, UNKNOWN_QUERY_INTENT, "INVALID_REQUEST", "user_id 不能为空")

        user = UserContext.from_values(
            user_id=request.user_id,
            tenant_id=request.tenant_id,
            org_id=request.org_id,
            roles=list(request.roles),
        )
        mapping = mapping or TableMapping.from_env()
        mapping.validate()
        settings = settings or QuerySettings.from_env()

        intent = recognize_intent(request.question)
        intent_name = intent.name
        if intent.name == UNKNOWN_QUERY_INTENT:
            return _error(
                trace_id,
                intent.name,
                "UNKNOWN_QUERY_INTENT",
                "暂时无法识别该查询，请尝试查询招聘岗位或我的考勤数据。",
            )

        time_range = None
        if intent.name in {MY_ATTENDANCE_RECENT_WEEK, MY_ATTENDANCE_BY_TIME_RANGE}:
            if not can_query_attendance(user, target_user_id=user.user_id):
                return _error(trace_id, intent.name, "PERMISSION_DENIED", "你只能查询自己的考勤数据。")
            range_text = request.time_range or request.question
            time_range = parse_time_range(range_text, today=today)

        plan = build_sql_plan(
            intent=intent.name,
            question=request.question,
            user=user,
            mapping=mapping,
            settings=settings,
            time_range=time_range,
            slots=intent.slots,
        )
        generated_sql = plan.sql

        guard = SQLGuard(mapping=mapping, settings=settings)
        guard_result = guard.validate(
            plan.sql,
            params=plan.params,
            current_user_id=user.user_id,
            tenant_id=user.tenant_id,
        )
        safe = guard_result.safe
        if not guard_result.safe:
            return _error(
                trace_id,
                intent.name,
                "SQL_GUARD_REJECTED",
                f"当前查询未通过安全校验：{guard_result.error}",
                sql=_maybe_sql(request, settings, guard_result.sql),
            )

        generated_sql = guard_result.sql
        query_executor = executor or MySQLQueryClient.from_env()
        result = query_executor.query(guard_result.sql, plan.params)
        row_count = len(result.rows)
        answer = format_answer(intent.name, result.rows, time_range=time_range)
        return QueryResponse(
            success=True,
            intent=intent.name,
            answer=answer,
            data=result.rows,
            safe=True,
            trace_id=trace_id,
            sql=_maybe_sql(request, settings, guard_result.sql),
        )
    except RecruitmentConfigError as exc:
        error_message = str(exc)
        logger.warning("RecruitmentSystem config error: %s", exc)
        return _error(trace_id, intent_name, "CONFIG_ERROR", str(exc))
    except RecruitmentQueryError as exc:
        error_message = str(exc)
        logger.warning("RecruitmentSystem database error: %s", exc)
        return _error(trace_id, intent_name, "DATABASE_ERROR", str(exc), sql=_maybe_sql(request, settings, generated_sql))
    except Exception as exc:
        error_message = f"{type(exc).__name__}: {exc}"
        logger.exception("RecruitmentSystem query failed")
        return _error(trace_id, intent_name, "QUERY_FAILED", error_message, sql=_maybe_sql(request, settings, generated_sql))
    finally:
        duration_ms = (time.perf_counter() - start) * 1000
        _audit(
            trace_id=trace_id,
            tenant_id=request.tenant_id,
            user_id=request.user_id,
            question=request.question,
            intent=intent_name,
            generated_sql=generated_sql,
            safe=safe,
            query_duration_ms=duration_ms,
            result_count=row_count,
            error_message=error_message,
        )


def _maybe_sql(request: QueryRequest, settings: QuerySettings | None, sql: str | None) -> str | None:
    if not sql:
        return None
    include_sql = request.include_sql
    if include_sql is None:
        include_sql = bool(settings and settings.include_sql)
    return sql if include_sql else None


def _error(
    trace_id: str,
    intent: str,
    error_code: str,
    message: str,
    *,
    sql: str | None = None,
) -> QueryResponse:
    return QueryResponse(
        success=False,
        intent=intent,
        answer=message,
        data=[],
        safe=False,
        trace_id=trace_id,
        sql=sql,
        error_code=error_code,
        message=message,
    )


def _audit(
    *,
    trace_id: str,
    tenant_id: str | None,
    user_id: str,
    question: str,
    intent: str,
    generated_sql: str | None,
    safe: bool,
    query_duration_ms: float,
    result_count: int,
    error_message: str | None,
) -> None:
    payload = {
        "trace_id": trace_id,
        "tenant_id": tenant_id,
        "user_id": user_id,
        "question": question,
        "intent": intent,
        "generated_sql": generated_sql,
        "sql_safe_result": safe,
        "query_duration_ms": round(query_duration_ms, 2),
        "result_count": result_count,
        "error_message": error_message,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    audit_logger.info(json.dumps(payload, ensure_ascii=False))
