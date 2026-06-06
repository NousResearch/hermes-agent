#!/usr/bin/env python3
"""
DeepSeek TUI adapter — stub implementation.

DeepSeek TUI is an interactive terminal UI that is NOT suitable for
programmatic driving.  It outputs ANSI escape sequences, not structured JSON.

Limitations (permanent architectural constraints):
  - Cannot yield structured ``tool_call`` events (ANSI output only)
  - Cannot distinguish ``reasoning`` from ``message``
  - No native diff support
  - No review gate
  - No health check beyond binary presence

v0.3 status: **STUB** — registered with ``ui_fidelity="low"``.
  ``check_health()`` reports ``UNAVAILABLE`` with a clear explanation.
  The adapter does NOT attempt to launch the TUI, because driving an
  interactive terminal UI from a subprocess is unreliable and would
  produce garbage events.

Future direction (post-v0.5):
  If DeepSeek ships a non-TUI (JSON-line) CLI, this adapter will be
  upgraded to match the ClaudeCodeAdapter pattern.
"""

from __future__ import annotations

import asyncio
import datetime
import logging
import os
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, AsyncIterable, Dict, List, Optional

from executors.types import (
    AgentRun,
    ExecutorConfig,
    AdapterStartResult,
    RunEvent,
    RunEventType,
    RunStatus,
    ExecutorHealthResult,
    ExecutorHealthStatus,
)
from executors.health import (
    check_command_exists,
    make_unavailable_health,
)

logger = logging.getLogger(__name__)

DEFAULT_COMMAND = "deepseek-tui"

# STUB: set to True to attempt TUI launch (experimental, not recommended)
_ALLOW_TUI_LAUNCH = False


@dataclass
class _RunState:
    run_id: str
    status: RunStatus = RunStatus.PENDING
    events: List[RunEvent] = field(default_factory=list)
    error: Optional[str] = None


class DeepSeekTuiAdapter:
    """
    STUB adapter for DeepSeek TUI.

    **Does not launch the TUI.**  Attempting to programmatically drive an
    interactive terminal UI is unreliable and would corrupt the event stream.

    If you need a low-cost DeepSeek worker, configure ``hermes-local`` with
    ``model_ref="deepseek-v4-flash"`` instead.
    """

    def __init__(self, workspace: Optional[Path] = None):
        self._workspace = workspace or Path.cwd()
        self._runs: Dict[str, _RunState] = {}

    # ------------------------------------------------------------------
    # Health — always UNAVAILABLE in v0.3
    # ------------------------------------------------------------------

    async def check_health(self) -> ExecutorHealthResult:
        """Report UNAVAILABLE with explanation.

        Even if ``deepseek-tui`` binary exists, we mark it unavailable
        because the TUI cannot be driven programmatically.
        """
        command = _resolve_command(None)
        found, path = await check_command_exists(command)
        if not found:
            return make_unavailable_health(
                "deepseek-tui",
                f"Command not found: '{command}'. "
                f"DeepSeek TUI adapter is a STUB in v0.3 — "
                f"use 'hermes-local' with model 'deepseek-v4-flash' instead.",
            )

        return make_unavailable_health(
            "deepseek-tui",
            f"DeepSeek TUI found at {path}, but this adapter is a STUB. "
            f"Interactive TUI cannot be driven programmatically. "
            f"Use 'hermes-local' with model 'deepseek-v4-flash' as a drop-in replacement.",
        )

    # ------------------------------------------------------------------
    # start — refuses to launch
    # ------------------------------------------------------------------

    async def start(
        self, run: AgentRun, config: ExecutorConfig
    ) -> AdapterStartResult:
        """Refuse to launch with a clear error."""
        run_id = run.id or str(uuid.uuid4())

        state = _RunState(
            run_id=run_id,
            status=RunStatus.FAILED,
            error=(
                "DeepSeek TUI adapter is a STUB in v0.3. "
                "The interactive TUI cannot be driven programmatically. "
                "To use DeepSeek, select 'hermes-local' and set model to "
                "'deepseek-v4-flash'."
            ),
        )
        self._runs[run_id] = state

        self._append_event(state, RunEvent(
            type=RunEventType.FAILED,
            payload={"error_summary": state.error},
        ))

        return AdapterStartResult(
            external_run_id=run_id,
            base_path=str(self._workspace),
        )

    # ------------------------------------------------------------------
    # stop (no-op for stub)
    # ------------------------------------------------------------------

    async def stop(self, run_id: str) -> None:
        state = self._runs.get(run_id)
        if state and state.status == RunStatus.PENDING:
            state.status = RunStatus.CANCELLED

    # ------------------------------------------------------------------
    # stream_events
    # ------------------------------------------------------------------

    async def stream_events(self, run_id: str) -> AsyncIterable[RunEvent]:
        state = self._runs.get(run_id)
        if state is None:
            yield RunEvent(
                type=RunEventType.FAILED,
                payload={"error_summary": f"Run {run_id} not found"},
            )
            return

        yield RunEvent(
            type=RunEventType.FAILED,
            payload={
                "error_summary": (
                    "DeepSeek TUI cannot be used. "
                    "Select a different executor (hermes-local, claude-code, or codex)."
                ),
                "help": "Use 'hermes-local' with model 'deepseek-v4-flash' for DeepSeek.",
            },
        )

    # ------------------------------------------------------------------
    # get_status
    # ------------------------------------------------------------------

    async def get_status(self, run_id: str) -> RunStatus:
        state = self._runs.get(run_id)
        if state is None:
            return RunStatus.FAILED
        return state.status

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _append_event(state: _RunState, event: RunEvent) -> None:
        event.seq = len(state.events)
        event.timestamp = datetime.datetime.utcnow()
        state.events.append(event)


def _resolve_command(override: Optional[str]) -> str:
    if override:
        return override
    env_cmd = os.environ.get("HERMES_DEEPSEEK_TUI_PATH")
    if env_cmd:
        return env_cmd
    return DEFAULT_COMMAND
