"""Harness domain state: tasks, features, execution, and the feature lock.

Pure data + transitions. The agent never writes these directly; the run
driver in :mod:`harness.loop` owns every mutation. All types serialize to
plain JSON dicts for :mod:`harness.store`.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class TaskStatus:
    CREATED = "CREATED"
    ANALYZING = "ANALYZING"
    PLANNING = "PLANNING"
    IMPLEMENTING = "IMPLEMENTING"
    VERIFYING = "VERIFYING"
    RECOVERING = "RECOVERING"
    COMPLETED = "COMPLETED"
    BLOCKED = "BLOCKED"
    FAILED = "FAILED"
    STOPPED = "STOPPED"
    BUDGET_EXHAUSTED = "BUDGET_EXHAUSTED"
    CANCELLED = "CANCELLED"


TERMINAL_STATUSES = frozenset({
    TaskStatus.COMPLETED,
    TaskStatus.FAILED,
    TaskStatus.STOPPED,
    TaskStatus.BUDGET_EXHAUSTED,
    TaskStatus.CANCELLED,
})


class FeatureStatus:
    PENDING = "pending"
    INVESTIGATING = "investigating"
    IMPLEMENTING = "implementing"
    VERIFYING = "verifying"
    BLOCKED = "blocked"
    COMPLETED = "completed"


class Outcome:
    """Runtime iteration decision. The model may hint; only the driver sets these."""

    CONTINUE = "CONTINUE"
    COMPLETED = "COMPLETED"
    BLOCKED = "BLOCKED"
    FAILED = "FAILED"
    STOPPED = "STOPPED"
    BUDGET_EXHAUSTED = "BUDGET_EXHAUSTED"


class StepStatus:
    """Agent-side hint for one turn. Advisory only — gates decide."""

    CONTINUE = "CONTINUE"
    DONE = "DONE"
    BLOCKED = "BLOCKED"
    FAILED = "FAILED"


TERMINAL_OUTCOMES = frozenset({
    Outcome.COMPLETED,
    Outcome.FAILED,
    Outcome.STOPPED,
    Outcome.BUDGET_EXHAUSTED,
})


class TaskType:
    BUG_FIX = "BUG_FIX"
    FEATURE = "FEATURE"
    REFACTOR = "REFACTOR"
    DATABASE_CHANGE = "DATABASE_CHANGE"
    DEPENDENCY_CHANGE = "DEPENDENCY_CHANGE"
    DEBUG = "DEBUG"
    INVESTIGATION = "INVESTIGATION"
    TEST = "TEST"
    DOCUMENTATION = "DOCUMENTATION"
    MIGRATION = "MIGRATION"


class RiskLevel:
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


@dataclass
class ExecutionBudget:
    max_context_tokens: int = 32000
    max_output_tokens: int = 8000
    max_tool_calls: int = 50
    max_retries: int = 5
    max_replans: int = 3
    max_iterations: int = 20
    max_elapsed_seconds: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ExecutionBudget":
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in data.items() if k in known})


@dataclass
class Task:
    id: str
    goal: str
    success_criteria: List[str] = field(default_factory=list)
    type: str = TaskType.FEATURE
    constraints: List[str] = field(default_factory=list)
    risk: str = RiskLevel.MEDIUM
    budget: ExecutionBudget = field(default_factory=ExecutionBudget)
    status: str = TaskStatus.CREATED
    feature_id: Optional[str] = None
    created_at: str = field(default_factory=_utcnow)
    updated_at: str = field(default_factory=_utcnow)

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["budget"] = self.budget.to_dict()
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Task":
        data = dict(data)
        if isinstance(data.get("budget"), dict):
            data["budget"] = ExecutionBudget.from_dict(data["budget"])
        return cls(**data)


@dataclass
class FeatureState:
    id: str
    task_id: str
    name: str
    status: str = FeatureStatus.PENDING
    confidence: float = 0.0
    relevant_files: List[str] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)
    open_questions: List[str] = field(default_factory=list)
    known_issues: List[str] = field(default_factory=list)
    hypothesis: Optional[str] = None
    verification_results: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FeatureState":
        return cls(**data)


@dataclass
class ExecutionState:
    task_id: str
    feature_id: str
    phase: str = TaskStatus.ANALYZING
    hypothesis: Optional[str] = None
    relevant_files: List[str] = field(default_factory=list)
    open_questions: List[str] = field(default_factory=list)
    known_issues: List[str] = field(default_factory=list)
    iteration: int = 0
    last_evidence: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ExecutionState":
        return cls(**data)


@dataclass
class ToolObservation:
    id: str
    tool: str
    success: bool
    summary: str
    evidence: List[str] = field(default_factory=list)
    artifacts: List[str] = field(default_factory=list)
    timestamp: str = field(default_factory=_utcnow)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ToolObservation":
        return cls(**data)


@dataclass
class VerificationCheck:
    name: str
    passed: bool
    detail: str = ""
    strength: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "VerificationCheck":
        return cls(**data)


@dataclass
class VerificationResult:
    passed: bool
    checks: List[VerificationCheck] = field(default_factory=list)
    failures: List[str] = field(default_factory=list)
    confidence: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "passed": self.passed,
            "checks": [c.to_dict() for c in self.checks],
            "failures": list(self.failures),
            "confidence": self.confidence,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "VerificationResult":
        return cls(
            passed=data.get("passed", False),
            checks=[VerificationCheck.from_dict(c) for c in data.get("checks", [])],
            failures=list(data.get("failures", [])),
            confidence=data.get("confidence", 0.0),
        )


class KnowledgeType:
    ARCHITECTURE_DECISION = "ARCHITECTURE_DECISION"
    PROJECT_INVARIANT = "PROJECT_INVARIANT"
    DEPENDENCY = "DEPENDENCY"
    KNOWN_BUG = "KNOWN_BUG"
    SOLUTION = "SOLUTION"
    FEATURE_FACT = "FEATURE_FACT"
    PROCEDURE = "PROCEDURE"


@dataclass
class KnowledgeItem:
    id: str
    type: str
    content: str
    scope: str = ""
    confidence: float = 0.5
    source: List[str] = field(default_factory=list)
    last_verified: str = field(default_factory=_utcnow)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "KnowledgeItem":
        return cls(**data)


@dataclass
class Checkpoint:
    id: str
    task_id: str
    feature_id: str
    state: ExecutionState = field(
        default_factory=lambda: ExecutionState(task_id="", feature_id="")
    )
    context_ref: Optional[str] = None
    reason: str = ""
    created_at: str = field(default_factory=_utcnow)

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["state"] = self.state.to_dict()
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Checkpoint":
        data = dict(data)
        if isinstance(data.get("state"), dict):
            data["state"] = ExecutionState.from_dict(data["state"])
        return cls(**data)


class ScopeReason:
    FEATURE_COMPLETE = "FEATURE_COMPLETE"
    DEPENDENCY_BLOCK = "DEPENDENCY_BLOCK"
    VERIFICATION_REQUIRES_CHANGE = "VERIFICATION_REQUIRES_CHANGE"
    USER_SCOPE_CHANGE = "USER_SCOPE_CHANGE"


_ALLOWED_SCOPE_REASONS = frozenset({
    ScopeReason.FEATURE_COMPLETE,
    ScopeReason.DEPENDENCY_BLOCK,
    ScopeReason.VERIFICATION_REQUIRES_CHANGE,
    ScopeReason.USER_SCOPE_CHANGE,
})


class ScopeRejected(Exception):
    """Raised when a feature switch lacks a justified reason."""


class FeatureLock:
    """One active feature per task. Switches require a recorded reason."""

    def __init__(self) -> None:
        self._active: Optional[FeatureState] = None
        self.transitions: List[Dict[str, Any]] = []

    @property
    def active(self) -> Optional[FeatureState]:
        return self._active

    def select(
        self,
        feature: FeatureState,
        reason: str = ScopeReason.USER_SCOPE_CHANGE,
        evidence: str = "",
    ) -> FeatureState:
        previous = self._active.id if self._active else None
        if previous is not None and previous != feature.id:
            if reason not in _ALLOWED_SCOPE_REASONS:
                raise ScopeRejected(
                    f"switch {previous} -> {feature.id} needs a scope reason"
                )
            self.transitions.append({
                "from": previous,
                "to": feature.id,
                "reason": reason,
                "evidence": evidence,
                "at": _utcnow(),
            })
        self._active = feature
        return feature
