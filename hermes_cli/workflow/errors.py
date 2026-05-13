"""Workflow domain errors."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class WorkflowValidationIssue:
    """A structured policy/DAG/workflow validation issue."""

    code: str
    message: str
    path: str = ""
    severity: str = "error"
    data: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "code": self.code,
            "message": self.message,
            "path": self.path,
            "severity": self.severity,
        }
        if self.data:
            payload["data"] = self.data
        return payload


class WorkflowError(Exception):
    """Base class for workflow subsystem errors."""


class WorkflowValidationError(WorkflowError):
    """Raised when strict validation is requested and validation fails."""

    def __init__(self, issues: list[WorkflowValidationIssue]):
        self.issues = issues
        super().__init__("; ".join(issue.message for issue in issues) or "workflow validation failed")
