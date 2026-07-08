"""Schemas for diagnostic and maintenance reports."""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, Field


class DiagnosticSeverity(StrEnum):
    """Supported issue severities."""

    ERROR = "error"
    WARNING = "warning"
    INFO = "info"


class DiagnosticIssue(BaseModel):
    """A single actionable issue discovered during validation."""

    code: str
    severity: DiagnosticSeverity = DiagnosticSeverity.ERROR
    path: str
    message: str
    note_id: str | None = None


class DiagnosticReport(BaseModel):
    """Structured validation output."""

    issues: list[DiagnosticIssue] = Field(default_factory=list)
    stats: dict[str, int | dict[str, int]] = Field(default_factory=dict)
    severity_counts: dict[str, int] = Field(default_factory=dict)
    recovery_workflow: list[str] = Field(default_factory=list)
