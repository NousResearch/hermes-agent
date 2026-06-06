#!/usr/bin/env python3
"""
Hermes Local adapter — runs agents in-process via the existing AIAgent.

This is the "built-in" executor.  It does not spawn a subprocess; instead it
imports ``run_agent.AIAgent`` and calls ``run_conversation()`` directly.

The adapter wraps the blocking agent call in an asyncio executor thread and
normalizes the message stream into ``RunEvent`` items.

Status: **full** — start, stop, stream_events, get_status, health.
"""

from __future__ import annotations

import asyncio
import datetime
import json
import logging
import os
import sys
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, AsyncIterable, Dict, List, Optional

from executors.types import (
    ExecutorId,
    AgentRun,
    ExecutorConfig,
    AdapterStartResult,
    RunEvent,
    RunEventType,
    RunStatus,
    ExecutorHealthResult,
    ExecutorHealthStatus,
    AgentExecutorAdapter,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# In-memory run state (simplest form; production would want persistence)
# ---------------------------------------------------------------------------

@dataclass
class _RunState:
    run_id: str
    status: RunStatus = RunStatus.PENDING
    events: List[RunEvent] = field(default_factory=list)
    error: Optional[str] = None
    cancel_event: Optional[asyncio.Event] = None
    task: Optional[asyncio.Task] = None


# ---------------------------------------------------------------------------
# Adapter
# ---------------------------------------------------------------------------

class HermesLocalAdapter:
    """Runs agents in-process via ``run_agent.AIAgent``."""

    def __init__(self, workspace: Optional[Path] = None):
        self._workspace = workspace or Path.cwd()
        self._runs: Dict[str, _RunState] = {}

    # ------------------------------------------------------------------
    # Health
    # ------------------------------------------------------------------

    async def check_health(self) -> ExecutorHealthResult:
        """Hermes Local is always available (it's the same process)."""
        try:
            import run_agent
            return ExecutorHealthResult(
                executor_id="hermes-local",
                status=ExecutorHealthStatus.AVAILABLE,
                version="built-in",
            )
        except ImportError as e:
            return ExecutorHealthResult(
                executor_id="hermes-local",
                status=ExecutorHealthStatus.UNAVAILABLE,
                error=f"run_agent import failed: {e}",
            )

    # ------------------------------------------------------------------
    # start
    # ------------------------------------------------------------------

    async def start(
        self, run: AgentRun, config: ExecutorConfig
    ) -> AdapterStartResult:
        """Create a background task that runs the agent to completion.

        Returns immediately with an AdapterStartResult so the caller can
        begin streaming events.
        """
        run_id = run.id or str(uuid.uuid4())
        cancel_event = asyncio.Event()

        state = _RunState(
            run_id=run_id,
            status=RunStatus.RUNNING,
            cancel_event=cancel_event,
        )
        self._runs[run_id] = state

        # Launch agent in thread (AIAgent is blocking/sync)
        task = asyncio.create_task(
            self._execute(run, config, state),
            name=f"hermes-local-run-{run_id[:8]}",
        )
        state.task = task

        # Get git snapshot for diff support
        git_snapshot = await self._get_git_snapshot(run.workspace)

        return AdapterStartResult(
            external_run_id=run_id,
            base_path=str(run.workspace),
            git_snapshot=git_snapshot,
        )

    async def _execute(
        self, run: AgentRun, config: ExecutorConfig, state: _RunState,
    ) -> None:
        """Run the agent in a thread, collecting events."""
        try:
            # Import here to avoid circular dependency at module level
            from run_agent import AIAgent

            workspace = str(run.workspace) if run.workspace else str(self._workspace)

            # Switch to workspace
            original_cwd = os.getcwd()
            os.chdir(workspace)

            try:
                agent = AIAgent(
                    model=run.model_ref or "deepseek-v4-pro",
                    # Propagate config extras that AIAgent understands
                    max_iterations=config.extra.get("max_iterations", 90),
                )

                # Run in thread to avoid blocking event loop
                result = await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: agent.run_conversation(
                        run.prompt,
                        task_id=run.id,
                    ),
                )

                # Normalize messages to events
                messages = result.get("messages", [])
                self._normalize_messages(state, messages)

                if result.get("completed", True):
                    self._append_event(state, RunEvent(
                        type=RunEventType.COMPLETED,
                        payload={"summary": result.get("summary", "")},
                    ))
                    state.status = RunStatus.COMPLETED
                else:
                    self._append_event(state, RunEvent(
                        type=RunEventType.FAILED,
                        payload={"error_summary": "Agent did not complete"},
                    ))
                    state.status = RunStatus.FAILED

            finally:
                os.chdir(original_cwd)

        except asyncio.CancelledError:
            state.status = RunStatus.CANCELLED
            self._append_event(state, RunEvent(
                type=RunEventType.FAILED,
                payload={"error_summary": "Run cancelled by user"},
            ))
        except Exception as e:
            logger.exception("Hermes Local run %s failed", run.id)
            state.status = RunStatus.FAILED
            state.error = str(e)
            self._append_event(state, RunEvent(
                type=RunEventType.FAILED,
                payload={"error_summary": str(e)},
            ))

    def _normalize_messages(self, state: _RunState, messages: List[Dict]) -> None:
        """Convert AIAgent message list to RunEvent stream."""
        for msg in messages:
            role = msg.get("role", "")
            content = msg.get("content", "")

            if role == "assistant":
                # Reasoning content
                reasoning = msg.get("reasoning", "")
                if reasoning:
                    self._append_event(state, RunEvent(
                        type=RunEventType.REASONING,
                        payload={"content": reasoning},
                    ))

                # Tool calls
                tool_calls = msg.get("tool_calls")
                if tool_calls:
                    for tc in tool_calls:
                        func = tc.get("function", {})
                        try:
                            args = json.loads(func.get("arguments", "{}"))
                        except (json.JSONDecodeError, TypeError):
                            args = {"raw": func.get("arguments", "")}
                        self._append_event(state, RunEvent(
                            type=RunEventType.TOOL_CALL,
                            payload={
                                "tool_name": func.get("name", "unknown"),
                                "arguments": args,
                            },
                        ))

                # Text content (skip if it's just tool call markers)
                if content and not tool_calls:
                    self._append_event(state, RunEvent(
                        type=RunEventType.MESSAGE,
                        payload={"content": content},
                    ))

            elif role == "tool":
                tool_call_id = msg.get("tool_call_id", "")
                self._append_event(state, RunEvent(
                    type=RunEventType.TOOL_RESULT,
                    payload={
                        "tool_call_id": tool_call_id,
                        "content": str(content)[:2000],  # truncate long results
                    },
                ))

    # ------------------------------------------------------------------
    # stop
    # ------------------------------------------------------------------

    async def stop(self, run_id: str) -> None:
        """Cancel the background task. Idempotent."""
        state = self._runs.get(run_id)
        if state is None:
            return  # already gone

        if state.status in (RunStatus.COMPLETED, RunStatus.FAILED, RunStatus.CANCELLED):
            return  # already finished

        if state.cancel_event:
            state.cancel_event.set()

        if state.task and not state.task.done():
            state.task.cancel()
            try:
                await state.task
            except asyncio.CancelledError:
                pass

        state.status = RunStatus.CANCELLED

    # ------------------------------------------------------------------
    # stream_events
    # ------------------------------------------------------------------

    async def stream_events(self, run_id: str) -> AsyncIterable[RunEvent]:
        """Yield events from the run as they arrive.

        Events are buffered in memory; the iterator polls for new events
        and yields them as they are appended by the background task.
        """
        state = self._runs.get(run_id)
        if state is None:
            yield RunEvent(
                type=RunEventType.FAILED,
                payload={"error_summary": f"Run {run_id} not found"},
            )
            return

        yielded = 0
        done_statuses = {RunStatus.COMPLETED, RunStatus.FAILED, RunStatus.CANCELLED}

        while True:
            # Yield any new events
            while yielded < len(state.events):
                yield state.events[yielded]
                yielded += 1

            # Check if done
            if state.status in done_statuses:
                # Ensure FAILED event contract: if run failed but no FAILED
                # event was emitted, synthesize one
                if state.status == RunStatus.FAILED:
                    has_failed_event = any(
                        e.type == RunEventType.FAILED
                        for e in state.events[yielded:]
                    )
                    # Check already-yielded events too
                    has_failed_event = has_failed_event or any(
                        e.type == RunEventType.FAILED
                        for e in state.events[:yielded]
                    )
                    if not has_failed_event:
                        yield RunEvent(
                            type=RunEventType.FAILED,
                            payload={"error_summary": state.error or "Unknown error"},
                        )
                return

            await asyncio.sleep(0.1)

    # ------------------------------------------------------------------
    # get_status
    # ------------------------------------------------------------------

    async def get_status(self, run_id: str) -> RunStatus:
        """Return the current status of a run."""
        state = self._runs.get(run_id)
        if state is None:
            return RunStatus.FAILED  # run doesn't exist — treat as terminal
        return state.status

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _append_event(state: _RunState, event: RunEvent) -> None:
        """Append an event to the state buffer."""
        event.seq = len(state.events)
        event.timestamp = datetime.datetime.utcnow()
        state.events.append(event)

    @staticmethod
    async def _get_git_snapshot(workspace: Path) -> Optional[str]:
        """Get the current git HEAD sha for diff tracking."""
        try:
            proc = await asyncio.create_subprocess_exec(
                "git", "rev-parse", "HEAD",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=str(workspace),
            )
            stdout, _ = await asyncio.wait_for(
                proc.communicate(), timeout=5.0
            )
            if proc.returncode == 0:
                return stdout.decode().strip()
        except Exception:
            pass
        return None
