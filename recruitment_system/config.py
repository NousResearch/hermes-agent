"""Configuration and table mapping for RecruitmentSystem querying."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import yaml


class RecruitmentConfigError(ValueError):
    """Raised when RecruitmentSystem configuration is invalid."""


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw in (None, ""):
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise RecruitmentConfigError(f"{name} must be an integer") from exc


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw in (None, ""):
        return default
    try:
        return float(raw)
    except ValueError as exc:
        raise RecruitmentConfigError(f"{name} must be a number") from exc


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw in (None, ""):
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class RecruitmentDBConfig:
    host: str = ""
    port: int = 3306
    username: str = ""
    password: str = ""
    database: str = ""
    max_open_conns: int = 5
    max_idle_conns: int = 2
    conn_max_lifetime_seconds: int = 1800
    connect_timeout_seconds: float = 5.0
    query_timeout_seconds: float = 15.0

    @classmethod
    def from_env(cls) -> "RecruitmentDBConfig":
        return cls(
            host=os.getenv("RECRUITMENT_DB_HOST", ""),
            port=_env_int("RECRUITMENT_DB_PORT", 3306),
            username=os.getenv("RECRUITMENT_DB_USERNAME", ""),
            password=os.getenv("RECRUITMENT_DB_PASSWORD", ""),
            database=os.getenv("RECRUITMENT_DB_DATABASE", ""),
            max_open_conns=_env_int("RECRUITMENT_DB_MAX_OPEN_CONNS", 5),
            max_idle_conns=_env_int("RECRUITMENT_DB_MAX_IDLE_CONNS", 2),
            conn_max_lifetime_seconds=_env_int("RECRUITMENT_DB_CONN_MAX_LIFETIME", 1800),
            connect_timeout_seconds=_env_float("RECRUITMENT_DB_CONNECT_TIMEOUT_SECONDS", 5.0),
            query_timeout_seconds=_env_float("RECRUITMENT_DB_QUERY_TIMEOUT_SECONDS", 15.0),
        )

    def is_complete(self) -> bool:
        return bool(self.host and self.username and self.database)

    def missing_keys(self) -> list[str]:
        missing = []
        if not self.host:
            missing.append("RECRUITMENT_DB_HOST")
        if not self.username:
            missing.append("RECRUITMENT_DB_USERNAME")
        if not self.database:
            missing.append("RECRUITMENT_DB_DATABASE")
        return missing


@dataclass(frozen=True)
class QuerySettings:
    default_limit: int = 50
    max_limit: int = 100
    include_sql: bool = False

    @classmethod
    def from_env(cls) -> "QuerySettings":
        return cls(
            default_limit=_env_int("RECRUITMENT_DB_DEFAULT_LIMIT", 50),
            max_limit=_env_int("RECRUITMENT_DB_MAX_LIMIT", 100),
            include_sql=_env_bool("RECRUITMENT_QUERY_RETURN_SQL", False),
        )


DEFAULT_JOB_FIELDS = {
    "id": "id",
    "tenant_id": "tenant_id",
    "job_name": "job_name",
    "job_code": "job_code",
    "department_id": "department_id",
    "department_name": "department_name",
    "job_status": "job_status",
    "recruit_status": "recruit_status",
    "headcount": "headcount",
    "hired_count": "hired_count",
    "job_requirement": "job_requirement",
    "job_description": "job_description",
    "salary_min": "salary_min",
    "salary_max": "salary_max",
    "create_time": "create_time",
    "update_time": "update_time",
    "is_deleted": "is_deleted",
}

DEFAULT_ATTENDANCE_FIELDS = {
    "id": "id",
    "tenant_id": "tenant_id",
    "user_id": "user_id",
    "employee_id": "employee_id",
    "employee_name": "employee_name",
    "attendance_date": "attendance_date",
    "check_in_time": "check_in_time",
    "check_out_time": "check_out_time",
    "attendance_status": "attendance_status",
    "late_minutes": "late_minutes",
    "early_leave_minutes": "early_leave_minutes",
    "work_hours": "work_hours",
    "exception_reason": "exception_reason",
    "create_time": "create_time",
    "update_time": "update_time",
    "is_deleted": "is_deleted",
}


@dataclass(frozen=True)
class TableMapping:
    job_table: str = "recruitment_job"
    attendance_table: str = "attendance_record"
    job_fields: dict[str, str] = field(default_factory=lambda: dict(DEFAULT_JOB_FIELDS))
    attendance_fields: dict[str, str] = field(default_factory=lambda: dict(DEFAULT_ATTENDANCE_FIELDS))
    job_safe_fields: tuple[str, ...] = (
        "job_name",
        "department_name",
        "headcount",
        "hired_count",
        "job_status",
        "recruit_status",
    )
    job_detail_safe_fields: tuple[str, ...] = (
        "job_name",
        "department_name",
        "headcount",
        "hired_count",
        "job_status",
        "recruit_status",
        "job_requirement",
        "job_description",
        "salary_min",
        "salary_max",
    )
    attendance_safe_fields: tuple[str, ...] = (
        "attendance_date",
        "check_in_time",
        "check_out_time",
        "attendance_status",
        "late_minutes",
        "early_leave_minutes",
        "work_hours",
        "exception_reason",
    )
    recruiting_status_values: tuple[str, ...] = ("recruiting", "open", "正在招聘")

    @classmethod
    def default(cls) -> "TableMapping":
        return cls()

    @classmethod
    def from_env(cls) -> "TableMapping":
        path = os.getenv("RECRUITMENT_DB_TABLE_MAPPING_FILE", "").strip()
        if not path:
            return cls.default()
        return cls.from_file(Path(path))

    @classmethod
    def from_file(cls, path: Path) -> "TableMapping":
        if not path.exists():
            raise RecruitmentConfigError(f"Table mapping file does not exist: {path}")

        raw = path.read_text(encoding="utf-8")
        if path.suffix.lower() == ".json":
            payload = json.loads(raw)
        else:
            payload = yaml.safe_load(raw) or {}
        if not isinstance(payload, dict):
            raise RecruitmentConfigError("Table mapping must be a mapping")
        return cls.from_mapping(payload)

    @classmethod
    def from_mapping(cls, payload: dict[str, Any]) -> "TableMapping":
        mapping = cls.default()
        tables = payload.get("tables") or payload
        if not isinstance(tables, dict):
            raise RecruitmentConfigError("tables must be a mapping")

        job = tables.get("job") or tables.get("recruitment_job") or {}
        attendance = tables.get("attendance") or tables.get("attendance_record") or {}

        if job:
            mapping = replace(
                mapping,
                job_table=str(job.get("table_name") or job.get("table") or mapping.job_table),
                job_fields=_merge_fields(mapping.job_fields, job.get("fields")),
                job_safe_fields=_merge_tuple(mapping.job_safe_fields, job.get("safe_fields")),
                job_detail_safe_fields=_merge_tuple(
                    mapping.job_detail_safe_fields,
                    job.get("detail_safe_fields"),
                ),
                recruiting_status_values=_merge_tuple(
                    mapping.recruiting_status_values,
                    job.get("recruiting_status_values"),
                ),
            )
        if attendance:
            mapping = replace(
                mapping,
                attendance_table=str(
                    attendance.get("table_name") or attendance.get("table") or mapping.attendance_table
                ),
                attendance_fields=_merge_fields(mapping.attendance_fields, attendance.get("fields")),
                attendance_safe_fields=_merge_tuple(
                    mapping.attendance_safe_fields,
                    attendance.get("safe_fields"),
                ),
            )
        mapping.validate()
        return mapping

    def validate(self) -> None:
        for logical in self.job_safe_fields:
            self.require_job_field(logical)
        for logical in self.job_detail_safe_fields:
            self.require_job_field(logical)
        for logical in self.attendance_safe_fields:
            self.require_attendance_field(logical)
        self.require_attendance_field("user_id")
        self.require_attendance_field("attendance_date")

    def require_job_field(self, logical: str) -> str:
        try:
            return self.job_fields[logical]
        except KeyError as exc:
            raise RecruitmentConfigError(f"Missing job field mapping: {logical}") from exc

    def require_attendance_field(self, logical: str) -> str:
        try:
            return self.attendance_fields[logical]
        except KeyError as exc:
            raise RecruitmentConfigError(f"Missing attendance field mapping: {logical}") from exc

    def actual_tables(self) -> set[str]:
        return {self.job_table, self.attendance_table}

    def allowed_fields_by_table(self) -> dict[str, set[str]]:
        job_allowed = {self.job_fields[name] for name in self.job_detail_safe_fields}
        job_allowed.update(
            self.job_fields[name]
            for name in ("tenant_id", "is_deleted", "recruit_status", "job_name", "update_time", "create_time")
            if name in self.job_fields
        )
        attendance_allowed = {self.attendance_fields[name] for name in self.attendance_safe_fields}
        attendance_allowed.update(
            self.attendance_fields[name]
            for name in ("tenant_id", "user_id", "is_deleted", "attendance_date")
            if name in self.attendance_fields
        )
        return {
            self.job_table: job_allowed,
            self.attendance_table: attendance_allowed,
        }


def _merge_fields(default: dict[str, str], override: Any) -> dict[str, str]:
    merged = dict(default)
    if override is None:
        return merged
    if not isinstance(override, dict):
        raise RecruitmentConfigError("fields must be a mapping")
    for key, value in override.items():
        merged[str(key)] = str(value)
    return merged


def _merge_tuple(default: tuple[str, ...], override: Any) -> tuple[str, ...]:
    if override is None:
        return default
    if isinstance(override, str):
        return tuple(part.strip() for part in override.split(",") if part.strip())
    if isinstance(override, (list, tuple, set)):
        return tuple(str(item) for item in override)
    raise RecruitmentConfigError("tuple override must be a string or list")
