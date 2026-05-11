"""Core types for the Hermes eval subsystem."""

from __future__ import annotations

import uuid
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Optional


class CheckType(str, Enum):
    """Types of deterministic checks."""
    FILE_EXISTS = "file_exists"
    FILE_NOT_EXISTS = "file_not_exists"
    CONTENT_CONTAINS = "content_contains"
    CONTENT_NOT_CONTAINS = "content_not_contains"
    CONTENT_EQUALS = "content_equals"
    JSON_VALID = "json_valid"
    JSON_KEY_EXISTS = "json_key_exists"
    REGEX_MATCH = "regex_match"
    EXIT_CODE = "exit_code"


@dataclass(frozen=True)
class DeterministicCheck:
    """A single deterministic check to run against eval output."""
    check_type: CheckType
    target: str  # file path, key name, or description
    expected: Any = None  # expected value (string, int, pattern, etc.)
    weight: float = 1.0  # relative weight for scoring


class CaseCategory(str, Enum):
    """Categories of eval cases."""
    FILE_WORKSPACE = "file_workspace"
    TOOL_ORCHESTRATION = "tool_orchestration"
    RELIABILITY = "reliability"


@dataclass(frozen=True)
class EvalCase:
    """Definition of a single eval case."""
    id: str
    name: str
    category: CaseCategory
    prompt: str
    deterministic_checks: tuple[DeterministicCheck, ...]
    timeout_seconds: int = 60
    tags: tuple[str, ...] = ()
    setup: Optional[Callable[[str], None]] = None  # receives workdir path


class CaseStatus(str, Enum):
    """Status of a case result."""
    PASSED = "passed"
    FAILED = "failed"
    ERROR = "error"
    TIMEOUT = "timeout"
    SKIPPED = "skipped"


@dataclass
class CheckResult:
    """Result of one deterministic check."""
    check: DeterministicCheck
    passed: bool
    actual: Any = None
    message: str = ""


@dataclass
class CaseResult:
    """Result of running a single eval case."""
    run_id: str
    case_id: str
    category: str
    status: CaseStatus
    deterministic_score: float = 0.0
    total_score: float = 0.0
    duration_ms: int = 0
    failure_summary: str = ""
    check_results: list[CheckResult] = field(default_factory=list)
    raw_result: dict[str, Any] = field(default_factory=dict)


@dataclass
class RunSummary:
    """Summary of a complete eval run."""
    run_id: str
    suite_name: str
    created_at: float = field(default_factory=time.time)
    label: str = ""
    case_count: int = 0
    passed_count: int = 0
    failed_count: int = 0
    avg_score: float = 0.0
    case_results: list[CaseResult] = field(default_factory=list)

    @staticmethod
    def new_run_id() -> str:
        return uuid.uuid4().hex[:12]
