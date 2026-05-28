"""Shared models for RecruitmentSystem querying."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Any, Mapping


@dataclass(frozen=True)
class UserContext:
    user_id: str
    tenant_id: str | None = None
    org_id: str | None = None
    roles: frozenset[str] = field(default_factory=frozenset)

    @classmethod
    def from_values(
        cls,
        *,
        user_id: str,
        tenant_id: str | None = None,
        org_id: str | None = None,
        roles: list[str] | tuple[str, ...] | set[str] | None = None,
    ) -> "UserContext":
        normalized_roles = frozenset(str(role).lower() for role in (roles or ()))
        return cls(
            user_id=str(user_id),
            tenant_id=str(tenant_id) if tenant_id else None,
            org_id=str(org_id) if org_id else None,
            roles=normalized_roles,
        )


@dataclass(frozen=True)
class TimeRange:
    start_date: date
    end_date: date
    label: str

    def as_params(self) -> dict[str, str]:
        return {
            "start_date": self.start_date.isoformat(),
            "end_date": self.end_date.isoformat(),
        }


@dataclass(frozen=True)
class QueryRequest:
    question: str
    user_id: str
    tenant_id: str | None = None
    org_id: str | None = None
    session_id: str | None = None
    time_range: str | None = None
    roles: tuple[str, ...] = ()
    include_sql: bool | None = None

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "QueryRequest":
        roles = payload.get("roles") or ()
        if isinstance(roles, str):
            roles = tuple(part.strip() for part in roles.split(",") if part.strip())
        return cls(
            question=str(payload.get("question") or payload.get("user_question") or ""),
            user_id=str(payload.get("user_id") or ""),
            tenant_id=str(payload.get("tenant_id")) if payload.get("tenant_id") else None,
            org_id=str(payload.get("org_id")) if payload.get("org_id") else None,
            session_id=str(payload.get("session_id")) if payload.get("session_id") else None,
            time_range=str(payload.get("time_range")) if payload.get("time_range") else None,
            roles=tuple(str(role) for role in roles),
            include_sql=payload.get("include_sql"),
        )


@dataclass(frozen=True)
class IntentResult:
    name: str
    confidence: float = 0.0
    slots: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SQLPlan:
    sql: str
    params: dict[str, Any]
    intent: str
    logical_tables: tuple[str, ...]
    time_range: TimeRange | None = None


@dataclass(frozen=True)
class GuardResult:
    safe: bool
    sql: str
    error: str | None = None
    applied_limit: int | None = None


@dataclass(frozen=True)
class QueryResult:
    rows: list[dict[str, Any]]
    duration_ms: float


@dataclass(frozen=True)
class QueryResponse:
    success: bool
    intent: str
    answer: str
    data: list[dict[str, Any]] = field(default_factory=list)
    safe: bool = False
    trace_id: str = ""
    sql: str | None = None
    error_code: str | None = None
    message: str | None = None

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "success": self.success,
            "intent": self.intent,
            "answer": self.answer,
            "data": self.data,
            "safe": self.safe,
            "trace_id": self.trace_id,
        }
        if self.sql is not None:
            payload["sql"] = self.sql
        if self.error_code:
            payload["error_code"] = self.error_code
        if self.message:
            payload["message"] = self.message
        return payload
