#!/usr/bin/env python3
"""
Core types for the executor adapter layer.

Defines the AgentExecutorAdapter protocol (Python equivalent of the TypeScript
interface), event types, manifest types, and health types.

Design decisions:
  - Uses Protocol for duck-typed adapters (no hard base class dependency)
  - RunEvent is a simple dataclass; no inheritance tree needed yet
  - ExecutorManifest is separate from the adapter (health + capabilities are metadata)
"""

from __future__ import annotations

import abc
import dataclasses
import datetime
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import (
    Any,
    AsyncIterable,
    Awaitable,
    Callable,
    Dict,
    List,
    Optional,
    Protocol,
    Sequence,
    runtime_checkable,
)


# ---------------------------------------------------------------------------
# Executor identity
# ---------------------------------------------------------------------------

ExecutorId = str
"""Unique identifier for an executor backend.

Examples: ``"hermes-local"``, ``"claude-code"``, ``"codex"``, ``"deepseek-tui"``
"""


# ---------------------------------------------------------------------------
# Executor health
# ---------------------------------------------------------------------------

class ExecutorHealthStatus(Enum):
    """Tri-state health for executor availability."""
    AVAILABLE = "available"       # command found, ready to run
    UNAVAILABLE = "unavailable"   # command not found or broken
    UNKNOWN = "unknown"           # health check not yet performed


@dataclass
class ExecutorHealthResult:
    """Result of a single executor health check."""
    executor_id: ExecutorId
    status: ExecutorHealthStatus
    version: Optional[str] = None
    error: Optional[str] = None
    checked_at: datetime.datetime = field(default_factory=datetime.datetime.utcnow)


# ---------------------------------------------------------------------------
# Executor capabilities & manifest
# ---------------------------------------------------------------------------

@dataclass
class ExecutorCapabilities:
    """What an executor can do. UI uses this to enable/disable components."""
    structured_tool_calls: bool = False
    native_diff_events: bool = False
    reasoning_blocks: bool = False
    review_gate: bool = False
    streaming: str = "batch"  # "realtime" | "line-buffered" | "batch"


@dataclass
class ExecutorManifest:
    """Static metadata for a registered executor."""
    id: ExecutorId
    label: str                          # human-readable name for UI
    description: str                    # one-liner
    capabilities: ExecutorCapabilities = field(default_factory=ExecutorCapabilities)
    default_model: Optional[str] = None
    supports_worktree: bool = False
    ui_fidelity: str = "full"           # "full" | "low" — DeepSeek TUI is "low"


# ---------------------------------------------------------------------------
# Run events (normalized across executors)
# ---------------------------------------------------------------------------

class RunEventType(Enum):
    """Normalized event types. All executors map to these."""
    MESSAGE = "message"
    REASONING = "reasoning"
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"
    DIFF = "diff"
    APPROVAL_NEEDED = "approval_needed"
    COMPLETED = "completed"
    FAILED = "failed"
    LOG = "log"


@dataclass
class RunEvent:
    """A single normalized event emitted during an agent run."""
    type: RunEventType
    payload: Dict[str, Any] = field(default_factory=dict)
    seq: Optional[int] = None           # assigned by Orchestrator, not adapter
    timestamp: datetime.datetime = field(default_factory=datetime.datetime.utcnow)


# ---------------------------------------------------------------------------
# Run status
# ---------------------------------------------------------------------------

class RunStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


# ---------------------------------------------------------------------------
# AgentRun (data passed to adapter.start)
# ---------------------------------------------------------------------------

@dataclass
class AgentRun:
    """Metadata for a single agent run."""
    id: str
    executor_id: ExecutorId
    prompt: str
    workspace: Path
    model_ref: Optional[str] = None
    risk_level: str = "R0"
    display_name: Optional[str] = None
    env: Dict[str, str] = field(default_factory=dict)
    run_seq: int = 0                     # monotonic within thread, assigned by Orchestrator
    run_type: str = "main"               # "main" | "review" | "qa"


# ---------------------------------------------------------------------------
# Executor configuration (passed to adapter.start)
# ---------------------------------------------------------------------------

@dataclass
class ExecutorConfig:
    """Per-run configuration consumed by the adapter."""
    path: Optional[str] = None          # override command path
    extra: Dict[str, Any] = field(default_factory=dict)
    disabled_tools: List[str] = field(default_factory=list)  # for review/qa runs


# ---------------------------------------------------------------------------
# Adapter start result
# ---------------------------------------------------------------------------

@dataclass
class AdapterStartResult:
    """What the adapter returns after start()."""
    external_run_id: Optional[str] = None
    base_path: Optional[str] = None
    git_snapshot: Optional[str] = None  # git rev-parse HEAD, for diff generation


# ---------------------------------------------------------------------------
# AgentExecutorAdapter protocol
# ---------------------------------------------------------------------------

@runtime_checkable
class AgentExecutorAdapter(Protocol):
    """Protocol for executor adapters.

    Any object with these methods can be registered as an executor.
    No base class required — duck typing works via ExecutorRegistry.register().
    """

    def start(
        self, run: AgentRun, config: ExecutorConfig
    ) -> Awaitable[AdapterStartResult]:
        """Start a run. The adapter should begin execution and return immediately."""

    def stop(self, run_id: str) -> Awaitable[None]:
        """Send stop signal to the executor. Must be idempotent."""

    def stream_events(self, run_id: str) -> AsyncIterable[RunEvent]:
        """Yield normalized RunEvents.

        **Contract**: If the run fails, this must yield a ``RunEvent(type=FAILED)``
        before the iterator ends.  Silent termination is not allowed.
        """

    def get_status(self, run_id: str) -> Awaitable[RunStatus]:
        """Query current run status without side-effects."""

    def check_health(self) -> Awaitable[ExecutorHealthResult]:
        """Check if the executor binary is available and functional.

        Optional but recommended.  Returns UNKNOWN if not implemented.
        """


# ---------------------------------------------------------------------------
# Worktree types (v0.4)
# ---------------------------------------------------------------------------

class WorktreeStatus(Enum):
    """Lifecycle states of a git worktree allocation."""
    NOT_CREATED = "not_created"
    CREATING = "creating"
    READY = "ready"
    DIRTY = "dirty"
    MERGING = "merging"
    MERGED = "merged"
    DISCARDED = "discarded"
    FAILED = "failed"


@dataclass
class WorktreeAllocation:
    """Tracks a single git worktree bound to a task thread."""
    thread_id: str
    worktree_path: str                            # absolute path on disk
    branch_name: str                              # e.g. "hermes/a3f8c1b2/1"
    status: WorktreeStatus = WorktreeStatus.NOT_CREATED
    base_commit: Optional[str] = None             # git rev-parse HEAD at creation
    created_at: Optional[datetime.datetime] = None
    released_at: Optional[datetime.datetime] = None
    error: Optional[str] = None
    merge_commit_sha: Optional[str] = None
    changed_files_count: Optional[int] = None     # git diff --stat count


# ---------------------------------------------------------------------------
# Router types (v0.5)
# ---------------------------------------------------------------------------

@dataclass
class RouterRecommendation:
    """Output of the executor router for a task creation context."""
    recommended_executor: ExecutorId
    confidence: float                      # 0.0 – 1.0
    reason: str                            # human-readable justification
    alternatives: List[ExecutorId] = field(default_factory=list)  # other viable options
    source: str = "keyword"                # "keyword" | "user_override" | "health_fallback"
    override: bool = False                 # True if user manually selected after recommendation


@dataclass
class TaskCreateContext:
    """Input to the router: what we know at task creation time."""
    title: str = ""
    goal: str = ""
    project_path: Optional[str] = None
    available_executors: List[ExecutorId] = field(default_factory=list)
    prefer_worktree: bool = False


# ---------------------------------------------------------------------------
# Project context types (v0.6)
# ---------------------------------------------------------------------------

@dataclass
class AdrSummary:
    """An Architecture Decision Record summary."""
    id: str         # e.g. "ADR-001"
    title: str
    decision: str   # one-sentence summary


@dataclass
class CommandEntry:
    """A named command pair (e.g. build, lint, test)."""
    label: str
    command: str


@dataclass
class RecentTask:
    """Auto-maintained record of a recently completed task."""
    thread_id: str
    title: str
    executor: str
    status: str  # "done" | "failed"
    completed_at: str
    summary: Optional[str] = None


@dataclass
class ProjectContext:
    """Structured project context stored at .hermes/context.json."""
    project_overview: str = ""
    architecture_notes: str = ""
    adr_summaries: List[AdrSummary] = field(default_factory=list)
    current_sprint: str = ""
    common_commands: List[CommandEntry] = field(default_factory=list)
    test_commands: List[CommandEntry] = field(default_factory=list)
    forbidden_areas: List[str] = field(default_factory=list)
    coding_conventions: str = ""
    recent_tasks: List[RecentTask] = field(default_factory=list)

    # Injection control
    context_injection_enabled: bool = True

    # Per-field include flags (runtime, not persisted — set by user at run creation)
    include_flags: Dict[str, bool] = field(default_factory=dict)


@dataclass
class PromptSnapshot:
    """Immutable record of the full prompt sent to an executor."""
    user_prompt: str                           # user's original input
    injected_prompt: str                       # full prompt sent to executor
    context_sha: Optional[str] = None           # hash of context at injection time
    context_include_flags: Dict[str, bool] = field(default_factory=dict)
    estimated_tokens: int = 0
    generated_at: Optional[datetime.datetime] = None


# ---------------------------------------------------------------------------
# Review / QA types (v0.7)
# ---------------------------------------------------------------------------

class Severity(Enum):
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


class FindingCategory(Enum):
    CORRECTNESS = "correctness"
    SECURITY = "security"
    PERFORMANCE = "performance"
    MAINTAINABILITY = "maintainability"
    STYLE = "style"
    TEST_COVERAGE = "test_coverage"


class ReviewStatus(Enum):
    NOT_STARTED = "not_started"
    RUNNING = "running"
    COMPLETED = "completed"    # has findings
    PASSED = "passed"          # no findings
    FAILED = "failed"          # review run itself failed


class QAStatus(Enum):
    NOT_STARTED = "not_started"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class RunType(Enum):
    MAIN = "main"
    REVIEW = "review"
    QA = "qa"


@dataclass
class ReviewFinding:
    """A single finding from a review run."""
    id: str                                       # unique finding ID
    run_id: str                                   # review run ID
    severity: Severity = Severity.MEDIUM
    category: FindingCategory = FindingCategory.CORRECTNESS
    file_path: Optional[str] = None
    line_start: Optional[int] = None
    line_end: Optional[int] = None
    title: str = ""
    description: str = ""
    suggestion: Optional[str] = None
    dismissed: bool = False


@dataclass
class ReviewReport:
    """Summary of a review run."""
    review_run_id: str
    status: ReviewStatus = ReviewStatus.NOT_STARTED
    executor: str = ""
    total_findings: int = 0
    critical_count: int = 0
    high_count: int = 0
    medium_count: int = 0
    low_count: int = 0
    info_count: int = 0
    findings: List[ReviewFinding] = field(default_factory=list)
    started_at: Optional[datetime.datetime] = None
    completed_at: Optional[datetime.datetime] = None
    duration_seconds: float = 0.0
    error: Optional[str] = None


@dataclass
class QARisk:
    """A single risk identified by QA run."""
    severity: Severity = Severity.MEDIUM
    title: str = ""
    description: str = ""
    affected_areas: List[str] = field(default_factory=list)


@dataclass
class QAReport:
    """Summary of a QA run."""
    qa_run_id: str
    status: QAStatus = QAStatus.NOT_STARTED
    executor: str = ""
    test_passed: int = 0
    test_failed: int = 0
    test_skipped: int = 0
    test_output: str = ""
    risks: List[QARisk] = field(default_factory=list)
    coverage_delta: Optional[float] = None
    started_at: Optional[datetime.datetime] = None
    completed_at: Optional[datetime.datetime] = None
    duration_seconds: float = 0.0
    error: Optional[str] = None


# ---------------------------------------------------------------------------
# Inbox types (v0.8)
# ---------------------------------------------------------------------------

class InboxSource(Enum):
    DESKTOP = "desktop"
    CLI = "cli"
    FEISHU = "feishu"        # stub
    DISCORD = "discord"      # stub
    SCHEDULER = "scheduler"  # stub


class InboxStatus(Enum):
    PENDING = "pending"
    CONFIRMED = "confirmed"   # converted to task
    REJECTED = "rejected"
    ARCHIVED = "archived"
    EXPIRED = "expired"


@dataclass
class TaskDraft:
    """Structured draft extracted from an inbox item's raw payload."""
    title: str = ""
    suggested_prompt: str = ""
    suggested_executor: Optional[str] = None
    project_hint: Optional[str] = None
    priority: str = "normal"  # "high" | "normal" | "low"
    tags: List[str] = field(default_factory=list)
    user_edited: bool = False


@dataclass
class InboxItem:
    """An item in the external inbox."""
    id: str
    source: InboxSource = InboxSource.DESKTOP
    raw_payload: Dict[str, Any] = field(default_factory=dict)
    draft: TaskDraft = field(default_factory=TaskDraft)
    status: InboxStatus = InboxStatus.PENDING
    created_at: datetime.datetime = field(default_factory=lambda: datetime.datetime.now(datetime.timezone.utc))
    expires_at: Optional[datetime.datetime] = None
    linked_task_id: Optional[str] = None
    rejected_reason: Optional[str] = None


@dataclass
class InboxResultCallback:
    """Result written back to the source after a linked task completes."""
    inbox_item_id: str
    run_id: str
    status: str  # "done" | "failed"
    summary: str
    changed_files_count: int = 0
    review_decision: str = ""
    writeback_available: bool = False   # True if source supports writeback
