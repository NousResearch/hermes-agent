#!/usr/bin/env python3
"""
RunBridge — aggregates RunEvents into Logs/Diff/ChangedFiles views.

This is the bridge between the Python executor backend and the desktop UI
(Electron + IPC).  It consumes normalized RunEvents and produces the
structured views that the UI renders: LogsTab, DiffTab, ChangedFilesPanel.

v1.0 release blocker: makes the full main link functional:
  Task → Run → Logs → Changed Files → Diff → Continue/Retry → Review/QA → Done
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Set

from executors.types import (
    ExecutorId,
    ExecutorConfig,
    ProjectContext,
    PromptSnapshot,
    RunEvent,
    RunEventType,
    RunStatus,
    RunType,
)

# ---------------------------------------------------------------------------
# ChangedFile
# ---------------------------------------------------------------------------


@dataclass
class ChangedFile:
    """A single file changed during a run."""
    path: str
    status: str          # "added" | "modified" | "deleted"
    additions: int = 0
    deletions: int = 0
    diff_patch: str = ""


# ---------------------------------------------------------------------------
# RunResult — aggregated output of a completed run
# ---------------------------------------------------------------------------


@dataclass
class RunResult:
    """Complete aggregate of a run's events, produced by RunBridge."""
    run_id: str
    status: RunStatus = RunStatus.PENDING
    error_summary: Optional[str] = None

    # Logs
    logs: List[_LogEntry] = field(default_factory=list)
    gemini_app_logs: List[str] = field(default_factory=list)  # renamed from "tool logs"

    # Changed files (aggregated from diff events)
    changed_files: List[ChangedFile] = field(default_factory=list)

    # Diff events (raw patches)
    diff_patches: List[str] = field(default_factory=list)

    # Tool calls
    tool_calls: List[_ToolCall] = field(default_factory=list)

    # Message deltas
    message_parts: List[str] = field(default_factory=list)

    # Reasoning blocks
    reasoning_blocks: List[str] = field(default_factory=list)

    # Stats
    total_events: int = 0
    events_by_type: Dict[str, int] = field(default_factory=dict)

    def full_message(self) -> str:
        return "".join(self.message_parts)

    def has_changes(self) -> bool:
        return len(self.changed_files) > 0

    def has_diff(self) -> bool:
        return len(self.diff_patches) > 0

    def to_summary(self) -> str:
        """One-line summary for status bar / task card."""
        if self.status == RunStatus.RUNNING:
            return f"Running ({len(self.message_parts)} message parts, {len(self.tool_calls)} tool calls)"
        if self.status == RunStatus.COMPLETED:
            changed = f", {len(self.changed_files)} files changed" if self.changed_files else ""
            return f"Completed ({self.total_events} events{changed})"
        if self.status == RunStatus.FAILED:
            return f"Failed: {self.error_summary or 'unknown error'}"
        return self.status.value


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


@dataclass
class _LogEntry:
    tool: str
    message: str
    level: str = "info"


@dataclass
class _ToolCall:
    seq: int
    tool_name: str
    args: Optional[dict] = None
    duration: float = 0.0
    error: bool = False
    output: Optional[str] = None
    stdout: Optional[str] = None
    stderr: Optional[str] = None


# ---------------------------------------------------------------------------
# RunBridge
# ---------------------------------------------------------------------------


class RunBridge:
    """Aggregates a stream of RunEvents into structured views.

    Usage::

        bridge = RunBridge()
        for event in adapter.stream_events(run_id):
            bridge.ingest(event)
        result = bridge.finalize()
        print(result.changed_files)
        for log in result.logs:
            print(f"[{log.tool}] {log.message}")
    """

    def __init__(self, run_id: Optional[str] = None):
        self.run_id = run_id or ""
        self._run_id: Optional[str] = None  # set on first event
        self._status: RunStatus = RunStatus.PENDING
        self._error_summary: Optional[str] = None

        self._logs: List[_LogEntry] = []
        self._changed_files: List[ChangedFile] = []
        self._diff_patches: List[str] = []
        self._tool_calls: List[_ToolCall] = []
        self._message_parts: List[str] = []
        self._reasoning_blocks: List[str] = []

        self._total_events = 0
        self._events_by_type: Dict[str, int] = {}

        # Track tool calls by their seq for pairing started/completed
        self._pending_tools: Dict[int, _ToolCall] = {}

    # ------------------------------------------------------------------
    # Ingest
    # ------------------------------------------------------------------

    def ingest(self, event: RunEvent) -> None:
        """Process a single RunEvent."""
        if self._run_id is None:
            self._run_id = event.payload.get("run_id", self.run_id)

        self._total_events += 1
        et = event.type.value
        self._events_by_type[et] = self._events_by_type.get(et, 0) + 1

        handler = getattr(self, f"_handle_{event.type.value}", None)
        if handler:
            handler(event)
        else:
            # Unknown event type — log as generic
            self._logs.append(_LogEntry(
                tool="system",
                message=f"Unknown event: {event.type.value} payload={str(event.payload)[:200]}",
                level="warn",
            ))

    def _handle_message(self, event: RunEvent) -> None:
        content = event.payload.get("content", "")
        if content:
            self._message_parts.append(content)

    def _handle_reasoning(self, event: RunEvent) -> None:
        content = event.payload.get("content", "")
        if content:
            self._reasoning_blocks.append(content)

    def _handle_tool_call(self, event: RunEvent) -> None:
        tc = _ToolCall(
            seq=event.seq or 0,
            tool_name=event.payload.get("tool_name", "unknown"),
            args=event.payload.get("arguments"),
        )
        self._pending_tools[tc.seq] = tc

    def _handle_tool_result(self, event: RunEvent) -> None:
        # Try to pair with a pending tool call
        tool_call_id = event.payload.get("tool_call_id", "")
        content = event.payload.get("content", "")
        tool_name = event.payload.get("tool_name", "unknown")

        # Extract file changes from tool results
        self._extract_files_from_content(tool_name, content)

        # Pair with pending tool call (by seq or tool_call_id)
        tc = None
        for seq, pending in list(self._pending_tools.items()):
            if pending.tool_name in tool_name or tool_name in pending.tool_name:
                tc = pending
                del self._pending_tools[seq]
                break

        if tc is None:
            tc = _ToolCall(seq=event.seq or 0, tool_name=tool_name)

        tc.stdout = event.payload.get("stdout")
        tc.stderr = event.payload.get("stderr")
        tc.output = content[:1000] if content else None
        tc.duration = event.payload.get("duration", 0.0)
        tc.error = event.payload.get("is_error", False)
        self._tool_calls.append(tc)

        # Generate log entries from tool output
        if tc.stdout:
            for line in tc.stdout.split("\n")[:50]:
                if line.strip():
                    self._logs.append(_LogEntry(tool=tc.tool_name, message=line, level="info"))
        if tc.stderr:
            for line in tc.stderr.split("\n")[:20]:
                if line.strip():
                    self._logs.append(_LogEntry(tool=tc.tool_name, message=line, level="warn"))
        if tc.error:
            self._logs.append(_LogEntry(
                tool=tc.tool_name,
                message=f"Tool failed: {content[:200] if content else 'no output'}",
                level="error",
            ))

    def _handle_log(self, event: RunEvent) -> None:
        tool = event.payload.get("tool", event.payload.get("tool_name", "system"))
        message = event.payload.get("message", event.payload.get("line", ""))
        level = event.payload.get("level", "info")
        if message.strip():
            self._logs.append(_LogEntry(tool=tool, message=message[:500], level=level))

    def _handle_diff(self, event: RunEvent) -> None:
        patch = event.payload.get("patch", "")
        if patch:
            self._diff_patches.append(patch)
            files = self._parse_diff_patch(patch, self._run_id or "unknown")
            self._changed_files.extend(files)

    def _handle_completed(self, event: RunEvent) -> None:
        self._status = RunStatus.COMPLETED

    def _handle_failed(self, event: RunEvent) -> None:
        self._status = RunStatus.FAILED
        self._error_summary = event.payload.get("error_summary", "Run failed")

    # ------------------------------------------------------------------
    # Diff parsing
    # ------------------------------------------------------------------

    def _parse_diff_patch(self, patch: str, run_id: str) -> List[ChangedFile]:
        """Parse a unified diff into ChangedFile objects."""
        files: List[ChangedFile] = []
        blocks = re.split(r"^diff --git ", patch, flags=re.MULTILINE)[1:]

        for block in blocks:
            path_match = re.match(r"a/(.*?) b/(.*?)$", block, re.MULTILINE)
            if not path_match:
                continue
            path = path_match.group(1)

            additions = len(re.findall(r"^\+(?!\+\+)", block, re.MULTILINE))
            deletions = len(re.findall(r"^-(?!--)", block, re.MULTILINE))

            if additions > 0 and deletions == 0:
                status = "added"
            elif additions == 0 and deletions > 0:
                status = "deleted"
            else:
                status = "modified"

            files.append(ChangedFile(
                path=path,
                status=status,
                additions=additions,
                deletions=deletions,
                diff_patch=block[:5000],  # cap per-file diff
            ))

        return files

    def _extract_files_from_content(self, tool_name: str, content: str) -> None:
        """Heuristic: extract file paths from tool output (e.g. write_file results)."""
        # Look for patterns like "Wrote to /path/to/file.py" or "Created file /path/to/file.py"
        patterns = [
            r"(?:Wrote|Created|Modified|Updated)\s+(?:to\s+)?([/\w.\-]+\.(?:py|ts|js|tsx|jsx|json|yaml|yml|md|css|html))",
            r"File\s+([/\w.\-]+\.(?:py|ts|js|tsx|jsx|json|yaml|yml|md|css|html))\s+(?:written|created|modified)",
        ]
        for pat in patterns:
            matches = re.findall(pat, content, re.IGNORECASE)
            for m in matches:
                # Avoid duplicates
                if not any(cf.path == m for cf in self._changed_files):
                    self._changed_files.append(ChangedFile(
                        path=m,
                        status="modified",
                    ))

    # ------------------------------------------------------------------
    # Finalize
    # ------------------------------------------------------------------

    def finalize(self) -> RunResult:
        """Return the aggregated RunResult."""
        return RunResult(
            run_id=self._run_id or self.run_id,
            status=self._status,
            error_summary=self._error_summary,
            logs=self._logs,
            changed_files=self._changed_files,
            diff_patches=self._diff_patches,
            tool_calls=self._tool_calls,
            message_parts=self._message_parts,
            reasoning_blocks=self._reasoning_blocks,
            total_events=self._total_events,
            events_by_type=self._events_by_type,
        )


# ---------------------------------------------------------------------------
# IPC protocol models
# ---------------------------------------------------------------------------


@dataclass
class IPCEvent:
    """An event that can be sent over the IPC bridge to the Electron UI."""
    event: str           # e.g. "tool.completed", "diff", "message.delta"
    run_id: str
    timestamp: float
    payload: dict = field(default_factory=dict)


@dataclass
class IPCChangedFile:
    """Changed file data sent over IPC."""
    path: str
    status: str
    additions: int = 0
    deletions: int = 0
    absolute_path: str = ""


def result_to_ipc_events(result: RunResult) -> List[IPCEvent]:
    """Convert a RunResult to a list of IPC events for the UI."""
    events: List[IPCEvent] = []
    import time
    ts = time.time()

    # Logs
    for log in result.logs:
        events.append(IPCEvent(
            event="tool.log",
            run_id=result.run_id,
            timestamp=ts,
            payload={"tool": log.tool, "message": log.message, "level": log.level},
        ))

    # Diff
    for patch in result.diff_patches:
        events.append(IPCEvent(
            event="diff",
            run_id=result.run_id,
            timestamp=ts,
            payload={"patch": patch, "files_changed": len(result.changed_files)},
        ))

    # Changed files
    for cf in result.changed_files:
        events.append(IPCEvent(
            event="changed_file",
            run_id=result.run_id,
            timestamp=ts,
            payload={"path": cf.path, "status": cf.status, "additions": cf.additions, "deletions": cf.deletions},
        ))

    # Status
    events.append(IPCEvent(
        event="run.completed" if result.status == RunStatus.COMPLETED else "run.failed",
        run_id=result.run_id,
        timestamp=ts,
        payload={"status": result.status.value, "error_summary": result.error_summary},
    ))

    return events
