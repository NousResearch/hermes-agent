#!/usr/bin/env python3
"""
Executor adapter layer for Hermes Desktop.

Provides a uniform interface for launching, monitoring, and stopping
agent execution backends (Hermes Local, Claude Code CLI, Codex CLI,
OpenCode, DeepSeek TUI).

Architecture:
  UI -> ExecutorRegistry.get(executor_id).start(run) -> subprocess / HTTP / stub
  UI -> ExecutorRegistry.get(executor_id).stream_events() -> unified RunEvent stream
  UI <- ExecutorRegistry.check_all_health() -> health status for all registered executors
"""

from executors.types import (
    ExecutorId,
    ExecutorHealthStatus,
    ExecutorHealthResult,
    ExecutorCapabilities,
    ExecutorManifest,
    RunEventType,
    RunEvent,
    RunStatus,
    AgentRun,
    ExecutorConfig,
    AdapterStartResult,
    # v0.4+ worktree
    WorktreeStatus,
    WorktreeAllocation,
    # v0.5+ router
    RouterRecommendation,
    TaskCreateContext,
    # v0.6+ context
    AdrSummary,
    CommandEntry,
    RecentTask,
    ProjectContext,
    PromptSnapshot,
    # v0.7+ review/qa
    Severity,
    FindingCategory,
    ReviewStatus,
    QAStatus,
    RunType,
    ReviewFinding,
    ReviewReport,
    QARisk,
    QAReport,
    # v0.8+ inbox
    InboxSource,
    InboxStatus,
    InboxItem,
    TaskDraft,
    InboxResultCallback,
)
from executors.registry import ExecutorRegistry
from executors.health import (
    check_executor_health,
    check_all_executors_health,
    make_unknown_health,
    make_available_health,
    make_unavailable_health,
)
from executors.router import ExecutorRouter, create_default_router

__all__ = [
    "ExecutorId",
    "ExecutorHealthStatus",
    "ExecutorHealthResult",
    "ExecutorCapabilities",
    "ExecutorManifest",
    "RunEventType",
    "RunEvent",
    "RunStatus",
    "AgentRun",
    "ExecutorConfig",
    "AdapterStartResult",
    "WorktreeStatus",
    "WorktreeAllocation",
    "RouterRecommendation",
    "TaskCreateContext",
    "AdrSummary",
    "CommandEntry",
    "RecentTask",
    "ProjectContext",
    "PromptSnapshot",
    "Severity",
    "FindingCategory",
    "ReviewStatus",
    "QAStatus",
    "RunType",
    "ReviewFinding",
    "ReviewReport",
    "QARisk",
    "QAReport",
    "InboxSource",
    "InboxStatus",
    "InboxItem",
    "TaskDraft",
    "InboxResultCallback",
    "ExecutorRegistry",
    "check_executor_health",
    "check_all_executors_health",
    "make_unknown_health",
    "make_available_health",
    "make_unavailable_health",
    "ExecutorRouter",
    "create_default_router",
]
