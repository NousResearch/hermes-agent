#!/usr/bin/env python3
"""
IPC protocol definitions for Hermes Desktop Electron bridge.

Defines the TypeScript-compatible interface that the Electron main process
exposes to the renderer via preload.  These are the data models used by
window.hermesAPI.* calls.

The actual IPC wire format is JSON over stdin/stdout (for CLI testing)
or Electron contextBridge (for production).  Both use the same schema.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Run lifecycle
# ---------------------------------------------------------------------------


@dataclass
class CreateRunRequest:
    thread_id: str
    prompt: str
    executor_type: str
    run_type: str = "main"  # "main" | "review" | "qa"
    project_root: str = "."


@dataclass
class CreateRunResponse:
    run_id: str
    status: str  # "created"


@dataclass
class StopRunRequest:
    run_id: str


@dataclass
class ContinueRunRequest:
    thread_id: str
    prompt: str
    previous_run_id: str
    executor_type: str = ""


@dataclass
class ContinueRunResponse:
    run_id: str


@dataclass
class RetryRunRequest:
    thread_id: str
    prompt: str
    executor_type: str = ""


@dataclass
class RetryRunResponse:
    run_id: str
    run_seq: int


# ---------------------------------------------------------------------------
# Review / QA
# ---------------------------------------------------------------------------


@dataclass
class TriggerReviewRequest:
    main_run_id: str
    worktree_path: str
    diff_patch: str = ""
    task_goal: str = ""
    changed_files: List[str] = field(default_factory=list)
    # executor_type defaults to the review agent's recommendation
    executor_type: str = ""


@dataclass
class TriggerReviewResponse:
    review_run_id: str
    status: str


@dataclass
class TriggerQARequest:
    main_run_id: str
    worktree_path: str
    test_commands: List[str] = field(default_factory=list)
    task_goal: str = ""
    # executor_type defaults to the QA agent's recommendation
    executor_type: str = ""


@dataclass
class TriggerQAResponse:
    qa_run_id: str
    status: str


# ---------------------------------------------------------------------------
# Data reads
# ---------------------------------------------------------------------------


@dataclass
class GetChangedFilesRequest:
    run_id: str


@dataclass
class GetChangedFilesResponse:
    run_id: str
    files: List[Dict[str, Any]] = field(default_factory=list)
    # Each file: {path, status, additions, deletions, absolute_path}


@dataclass
class GetGatewayStatusResponse:
    connected: bool
    model: str = ""
    error: Optional[str] = None


# ---------------------------------------------------------------------------
# Event streaming
# ---------------------------------------------------------------------------


@dataclass
class RawRunEvent:
    """IPC-level event as received by the renderer."""
    event: str           # "tool.completed", "diff", "message.delta", ...
    run_id: str
    timestamp: float     # unix epoch
    payload: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Approval
# ---------------------------------------------------------------------------


@dataclass
class ResolveApprovalRequest:
    run_id: str
    decision: str  # "continue" | "accept" | "done" | "reject"
    comment: Optional[str] = None


# ---------------------------------------------------------------------------
# HermesAPI — the full preload interface (for documentation)
# ---------------------------------------------------------------------------

class HermesAPI:
    """Typed interface for window.hermesAPI in Electron.

    This class is NOT instantiated in Python.  It exists as documentation
    of the IPC contract between backend (Python) and frontend (TS/Electron).

    All methods return Promises.  streamRunEvents returns an unsubscribe function.
    """

    # Run lifecycle
    async def createRun(self, thread_id: str, prompt: str, executor_type: str) -> dict: ...
    async def stopRun(self, run_id: str) -> None: ...
    async def continueRun(self, thread_id: str, prompt: str, previous_run_id: str) -> dict: ...
    async def retryRun(self, thread_id: str, prompt: str) -> dict: ...

    # Event streaming
    def streamRunEvents(self, run_id: str, callback) -> callable: ...  # returns unsubscribe

    # Approval
    async def resolveApproval(self, run_id: str, decision: str, comment: str = None) -> None: ...

    # Review / QA
    async def triggerReview(self, main_run_id: str) -> dict: ...
    async def triggerQA(self, main_run_id: str) -> dict: ...

    # Data reads
    async def getTaskThreads(self, project_id: str) -> list: ...
    async def getChangedFiles(self, run_id: str) -> dict: ...
    async def getGatewayStatus(self) -> dict: ...


# ---------------------------------------------------------------------------
# IPC channel names (for Electron ipcMain/ipcRenderer)
# ---------------------------------------------------------------------------

IPC_CHANNELS = {
    "run:create": "CreateRunRequest → CreateRunResponse",
    "run:stop": "StopRunRequest → void",
    "run:continue": "ContinueRunRequest → ContinueRunResponse",
    "run:retry": "RetryRunRequest → RetryRunResponse",
    "run:events:subscribe": "run_id → stream of RawRunEvent",
    "run:events:unsubscribe": "run_id → void",
    "review:trigger": "TriggerReviewRequest → TriggerReviewResponse",
    "qa:trigger": "TriggerQARequest → TriggerQAResponse",
    "approval:resolve": "ResolveApprovalRequest → void",
    "data:changed-files": "GetChangedFilesRequest → GetChangedFilesResponse",
    "data:gateway-status": "void → GetGatewayStatusResponse",
}
