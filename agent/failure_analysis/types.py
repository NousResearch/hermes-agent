"""Core types for the failure-pattern analysis subsystem."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional


class FailureType(str, Enum):
    """Top-level failure categories."""
    EVAL = "eval"
    TOOL = "tool"
    POLICY = "policy"
    INFRA = "infra"
    MODEL = "model"
    UNKNOWN = "unknown"


class FailureSubtype(str, Enum):
    """Subtypes within each failure category."""
    # eval
    REGRESSION = "regression"
    FAILED_CHECK = "failed_check"
    # tool
    TIMEOUT = "timeout"
    EXECUTION = "execution"
    # policy
    APPROVAL_BLOCKED = "approval_blocked"
    # infra
    AUTH = "auth"
    ENVIRONMENT = "environment"
    # model
    OUTPUT_MALFORMED = "output_malformed"
    # unknown
    UNKNOWN = "unknown"


class Severity(str, Enum):
    """Failure severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class NormalizedFailure:
    """A single normalized failure record."""
    id: str
    created_at: float = field(default_factory=time.time)
    source_surface: str = ""          # cli, gateway, eval, tool
    eval_run_id: Optional[str] = None
    case_id: Optional[str] = None
    session_id: Optional[str] = None
    task_id: Optional[str] = None
    failure_type: str = "unknown"
    failure_subtype: str = "unknown"
    severity: str = "medium"
    tool_name: Optional[str] = None
    model: Optional[str] = None
    provider: Optional[str] = None
    summary: str = ""
    evidence_json: Optional[str] = None
    fingerprint: str = ""
