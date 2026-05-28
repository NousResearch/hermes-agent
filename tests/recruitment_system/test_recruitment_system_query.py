from __future__ import annotations

from datetime import date

import pytest

from recruitment_system.config import QuerySettings, TableMapping
from recruitment_system.db import DatabaseExecutionError
from recruitment_system.formatter import format_answer
from recruitment_system.intent import (
    MY_ATTENDANCE_RECENT_WEEK,
    RECRUITING_JOB_LIST,
    recognize_intent,
)
from recruitment_system.models import QueryRequest, QueryResult, UserContext
from recruitment_system.service import query_recruitment_system
from recruitment_system.sql_guard import SQLGuard
from recruitment_system.sql_templates import build_attendance_sql, build_recruiting_job_list_sql
from recruitment_system.time_ranges import parse_time_range


class FakeExecutor:
    def __init__(self, rows=None, error=None):
        self.rows = rows or []
        self.error = error
        self.sql = None
        self.params = None

    def query(self, sql, params=None):
        self.sql = sql
        self.params = params or {}
        if self.error:
            raise self.error
        return QueryResult(rows=self.rows, duration_ms=2.5)


def test_intent_recognizer_recruiting_job_list():
    result = recognize_intent("当前正在招聘的岗位有哪些？")

    assert result.name == RECRUITING_JOB_LIST


def test_intent_recognizer_attendance_recent_week():
    result = recognize_intent("列出最近一周我的考勤数据。")

    assert result.name == MY_ATTENDANCE_RECENT_WEEK


def test_time_parser_recent_week():
    result = parse_time_range("最近一周", today=date(2026, 5, 23))

    assert result.start_date == date(2026, 5, 17)
    assert result.end_date == date(2026, 5, 24)
    assert result.label == "最近一周"


def test_job_query_template_generates_recruiting_sql():
    mapping = TableMapping.default()
    settings = QuerySettings(default_limit=50, max_limit=100)
    user = UserContext.from_values(user_id="u001", tenant_id="t001")

    plan = build_recruiting_job_list_sql(user=user, mapping=mapping, settings=settings)

    assert "FROM `recruitment_job`" in plan.sql
    assert "`recruit_status` IN" in plan.sql
    assert "`tenant_id` = %(tenant_id)s" in plan.sql
    assert "LIMIT 50" in plan.sql
    assert plan.params["tenant_id"] == "t001"


def test_attendance_query_template_binds_current_user():
    mapping = TableMapping.default()
    settings = QuerySettings(default_limit=20, max_limit=100)
    user = UserContext.from_values(user_id="u001", tenant_id="t001")
    time_range = parse_time_range("最近一周", today=date(2026, 5, 23))

    plan = build_attendance_sql(
        intent=MY_ATTENDANCE_RECENT_WEEK,
        user=user,
        mapping=mapping,
        settings=settings,
        time_range=time_range,
    )

    assert "`user_id` = %(current_user_id)s" in plan.sql
    assert plan.params["current_user_id"] == "u001"
    assert plan.params["start_date"] == "2026-05-17"
    assert plan.params["end_date"] == "2026-05-24"


def test_sql_guard_rejects_delete():
    guard = SQLGuard(mapping=TableMapping.default(), settings=QuerySettings())

    result = guard.validate("DELETE FROM attendance_record", current_user_id="u001")

    assert result.safe is False
    assert "SELECT" in result.error


def test_sql_guard_rejects_multi_statement():
    guard = SQLGuard(mapping=TableMapping.default(), settings=QuerySettings())

    result = guard.validate("SELECT `job_name` FROM `recruitment_job`; SELECT 1")

    assert result.safe is False
    assert "多语句" in result.error


def test_sql_guard_rejects_attendance_without_user_id():
    guard = SQLGuard(mapping=TableMapping.default(), settings=QuerySettings())
    sql = (
        "SELECT `attendance_date` AS `attendance_date` "
        "FROM `attendance_record` WHERE `is_deleted` = 0 LIMIT 20"
    )

    result = guard.validate(sql, current_user_id="u001")

    assert result.safe is False
    assert "user_id" in result.error


def test_sql_guard_auto_limits_query():
    guard = SQLGuard(mapping=TableMapping.default(), settings=QuerySettings(default_limit=50, max_limit=100))
    sql = (
        "SELECT `job_name` AS `job_name` "
        "FROM `recruitment_job` WHERE `is_deleted` = 0"
    )

    result = guard.validate(sql)

    assert result.safe is True
    assert result.sql.endswith("LIMIT 50")


def test_sql_guard_reduces_large_limit():
    guard = SQLGuard(mapping=TableMapping.default(), settings=QuerySettings(default_limit=50, max_limit=100))
    sql = (
        "SELECT `job_name` AS `job_name` "
        "FROM `recruitment_job` WHERE `is_deleted` = 0 LIMIT 500"
    )

    result = guard.validate(sql)

    assert result.safe is True
    assert result.sql.endswith("LIMIT 100")


def test_service_uses_mock_query_client_for_jobs():
    executor = FakeExecutor(
        rows=[
            {
                "job_name": "AI算法工程师",
                "department_name": "AI研发部",
                "headcount": 2,
                "hired_count": 0,
                "job_status": "open",
                "recruit_status": "正在招聘",
            }
        ]
    )
    request = QueryRequest(question="当前正在招聘的岗位有哪些？", user_id="u001", tenant_id="t001", include_sql=True)

    response = query_recruitment_system(
        request,
        executor=executor,
        mapping=TableMapping.default(),
        settings=QuerySettings(default_limit=50, max_limit=100),
    )

    assert response.success is True
    assert response.intent == RECRUITING_JOB_LIST
    assert "AI算法工程师" in response.answer
    assert response.sql is not None
    assert executor.params["tenant_id"] == "t001"


def test_job_rows_generate_natural_language_answer():
    answer = format_answer(
        RECRUITING_JOB_LIST,
        [
            {"job_name": "AI算法工程师", "department_name": "AI研发部", "headcount": 2},
            {"job_name": "高级Java工程师", "department_name": "平台部", "headcount": 1},
        ],
    )

    assert "当前正在招聘的岗位有 AI算法工程师、高级Java工程师" in answer
    assert "AI研发部" in answer


def test_attendance_rows_generate_summary():
    time_range = parse_time_range("最近一周", today=date(2026, 5, 23))
    answer = format_answer(
        MY_ATTENDANCE_RECENT_WEEK,
        [
            {
                "attendance_date": "2026-05-18",
                "check_in_time": "09:02:00",
                "check_out_time": "18:05:00",
                "attendance_status": "迟到",
                "late_minutes": 2,
                "early_leave_minutes": 0,
                "work_hours": 8.0,
            },
            {
                "attendance_date": "2026-05-19",
                "check_in_time": "08:58:00",
                "check_out_time": "18:00:00",
                "attendance_status": "正常",
                "late_minutes": 0,
                "early_leave_minutes": 0,
                "work_hours": 8.0,
            },
        ],
        time_range=time_range,
    )

    assert "最近一周你共有 2 条考勤记录" in answer
    assert "迟到 1 天" in answer
    assert "2026-05-18" in answer


def test_database_exception_returns_clear_error():
    request = QueryRequest(question="当前正在招聘的岗位有哪些？", user_id="u001")
    response = query_recruitment_system(
        request,
        executor=FakeExecutor(error=DatabaseExecutionError("connection refused")),
        mapping=TableMapping.default(),
        settings=QuerySettings(),
    )

    assert response.success is False
    assert response.error_code == "DATABASE_ERROR"
    assert "connection refused" in response.message


def test_empty_job_result_returns_friendly_answer():
    request = QueryRequest(question="当前正在招聘的岗位有哪些？", user_id="u001")
    response = query_recruitment_system(
        request,
        executor=FakeExecutor(rows=[]),
        mapping=TableMapping.default(),
        settings=QuerySettings(),
    )

    assert response.success is True
    assert response.answer == "当前未查询到正在招聘的岗位。"


def test_unknown_intent_returns_guidance():
    request = QueryRequest(question="帮我查一下天气", user_id="u001")
    response = query_recruitment_system(
        request,
        executor=FakeExecutor(),
        mapping=TableMapping.default(),
        settings=QuerySettings(),
    )

    assert response.success is False
    assert response.error_code == "UNKNOWN_QUERY_INTENT"
    assert "招聘岗位" in response.message


def test_sql_guard_rejects_wrong_attendance_user_param():
    mapping = TableMapping.default()
    user = UserContext.from_values(user_id="u001")
    plan = build_attendance_sql(
        intent=MY_ATTENDANCE_RECENT_WEEK,
        user=user,
        mapping=mapping,
        settings=QuerySettings(),
        time_range=parse_time_range("最近一周", today=date(2026, 5, 23)),
    )
    guard = SQLGuard(mapping=mapping, settings=QuerySettings())

    result = guard.validate(
        plan.sql,
        params={**plan.params, "current_user_id": "u999"},
        current_user_id="u001",
    )

    assert result.safe is False
    assert "当前用户" in result.error
