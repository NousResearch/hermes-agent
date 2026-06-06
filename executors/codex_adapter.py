#!/usr/bin/env python3
"""
Codex adapter — runs ``codex`` CLI as a subprocess.

Minimal v0.3 implementation:
  - Command launch via asyncio subprocess
  - stdout/stderr log collection
  - Exit status reporting
  - Health check via ``which codex && codex --version``

  If ``codex`` is not available: health reports ``UNAVAILABLE``, but the
  adapter still registers so the UI can show it as disabled with a reason.

Not implemented (v0.4+):
  - Cloud API mode (currently CLI subprocess only)
  - Structured JSON output parsing
  - Diff generation
  - Worktree support

Status: **minimal** — start, stop, log collection, exit status, health.
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
    check_version,
    make_unavailable_health,
    make_available_health,
)

logger = logging.getLogger(__name__)

DEFAULT_COMMAND = "codex"
DEFAULT_VERSION_ARGS = "--version"


@dataclass
class _RunState:
    run_id: str
    process: Optional[asyncio.subprocess.Process] = None
    status: RunStatus = RunStatus.PENDING
    events: List[RunEvent] = field(default_factory=list)
    error: Optional[str] = None
    stdout_lines: List[str] = field(default_factory=list)
    stderr_lines: List[str] = field(default_factory=list)


class CodexAdapter:
    """Runs Codex CLI as a subprocess.

    If ``codex`` is not installed, ``check_health()`` reports UNAVAILABLE
    and the adapter cannot start runs.
    """

    def __init__(self, workspace: Optional[Path] = None):
        self._workspace = workspace or Path.cwd()
        self._runs: Dict[str, _RunState] = {}

    # ------------------------------------------------------------------
    # Health
    # ------------------------------------------------------------------

    async def check_health(self) -> ExecutorHealthResult:
        """Check that ``codex`` exists and is runnable."""
        command = _resolve_command(None)
        found, path = await check_command_exists(command)
        if not found:
            return make_unavailable_health(
                "codex",
                f"Command not found: '{command}' — "
                f"install via Codex.app or set HERMES_CODEX_PATH",
            )

        version = await check_version(command, DEFAULT_VERSION_ARGS)
        return make_available_health("codex", version=version)

    # ------------------------------------------------------------------
    # start
    # ------------------------------------------------------------------

    async def start(
        self, run: AgentRun, config: ExecutorConfig
    ) -> AdapterStartResult:
        """Launch ``codex`` subprocess with the prompt.

        The command format is: ``codex -p "<prompt>"``
        """
        run_id = run.id or str(uuid.uuid4())
        command = _resolve_command(config.path)
        workspace = str(run.workspace) if run.workspace else str(self._workspace)

        state = _RunState(run_id=run_id, status=RunStatus.RUNNING)
        self._runs[run_id] = state

        # Codex CLI accepts prompts with -p flag
        cmd = [command, "-p", run.prompt]

        try:
            state.process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=workspace,
                env={**os.environ, **(run.env or {})},
            )

            asyncio.create_task(
                self._collect_output(state),
                name=f"codex-collect-{run_id[:8]}",
            )

        except FileNotFoundError:
            state.status = RunStatus.FAILED
            state.error = f"Command not found: {command}"
            self._append_event(state, RunEvent(
                type=RunEventType.FAILED,
                payload={"error_summary": state.error},
            ))
        except Exception as e:
            state.status = RunStatus.FAILED
            state.error = str(e)
            self._append_event(state, RunEvent(
                type=RunEventType.FAILED,
                payload={"error_summary": state.error},
            ))

        git_snapshot = await self._get_git_snapshot(workspace)

        return AdapterStartResult(
            external_run_id=run_id,
            base_path=workspace,
            git_snapshot=git_snapshot,
        )

    async def _collect_output(self, state: _RunState) -> None:
        """Read stdout/stderr, normalize to log events."""
        if state.process is None:
            return

        try:
            async def read_stream(stream, lines):
                while True:
                    line = await stream.readline()
                    if not line:
                        break
                    decoded = line.decode("utf-8", errors="replace").rstrip("\n")
                    lines.append(decoded)
                    self._append_event(state, RunEvent(
                        type=RunEventType.LOG,
                        payload={"line": decoded},
                    ))

            await asyncio.gather(
                read_stream(state.process.stdout, state.stdout_lines),
                read_stream(state.process.stderr, state.stderr_lines),
            )

            await state.process.wait()
            exit_code = state.process.returncode

            if exit_code == 0:
                state.status = RunStatus.COMPLETED
                self._append_event(state, RunEvent(
                    type=RunEventType.COMPLETED,
                    payload={"exit_code": exit_code},
                ))
            else:
                stderr_summary = "\n".join(state.stderr_lines[-20:])
                state.status = RunStatus.FAILED
                state.error = f"Exit code {exit_code}: {stderr_summary}"
                self._append_event(state, RunEvent(
                    type=RunEventType.FAILED,
                    payload={
                        "exit_code": exit_code,
                        "error_summary": stderr_summary[:500],
                    },
                ))

        except Exception as e:
            state.status = RunStatus.FAILED
            state.error = str(e)
            self._append_event(state, RunEvent(
                type=RunEventType.FAILED,
                payload={"error_summary": str(e)},
            ))

    # ------------------------------------------------------------------
    # stop
    # ------------------------------------------------------------------

    async def stop(self, run_id: str) -> None:
        """Send SIGTERM to subprocess. Idempotent."""
        state = self._runs.get(run_id)
        if state is None or state.process is None:
            return
        if state.status in (RunStatus.COMPLETED, RunStatus.FAILED, RunStatus.CANCELLED):
            return

        try:
            state.process.terminate()
            try:
                await asyncio.wait_for(state.process.wait(), timeout=10.0)
            except asyncio.TimeoutError:
                state.process.kill()
                await state.process.wait()
        except ProcessLookupError:
            pass

        state.status = RunStatus.CANCELLED

    # ------------------------------------------------------------------
    # stream_events / get_status (identical pattern to ClaudeCodeAdapter)
    # ------------------------------------------------------------------

    async def stream_events(self, run_id: str) -> AsyncIterable[RunEvent]:
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
            while yielded < len(state.events):
                yield state.events[yielded]
                yielded += 1

            if state.status in done_statuses:
                if state.status == RunStatus.FAILED:
                    has_failed = any(
                        e.type == RunEventType.FAILED
                        for e in state.events
                    )
                    if not has_failed:
                        yield RunEvent(
                            type=RunEventType.FAILED,
                            payload={"error_summary": state.error or "Unknown error"},
                        )
                return

            await asyncio.sleep(0.1)

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

    @staticmethod
    async def _get_git_snapshot(workspace: str) -> Optional[str]:
        try:
            proc = await asyncio.create_subprocess_exec(
                "git", "rev-parse", "HEAD",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=workspace,
            )
            stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=5.0)
            if proc.returncode == 0:
                return stdout.decode().strip()
        except Exception:
            pass
        return None


def _resolve_command(override: Optional[str]) -> str:
    if override:
        return override
    env_cmd = os.environ.get("HERMES_CODEX_PATH")
    if env_cmd:
        return env_cmd
    return DEFAULT_COMMAND
