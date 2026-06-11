"""Provider-neutral realtime voice tool bridge.

This module keeps provider protocol code out of Hermes tool execution. Realtime
providers emit neutral ``RealtimeToolCall`` events; the bridge validates the
requested tool against profile config, executes the configured Hermes action, and
returns the result to the active provider session via ``submit_tool_result``.
"""

from __future__ import annotations

import asyncio
import inspect
import json
import logging
import uuid
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Mapping

from gateway.realtime_voice.config import RealtimeVoiceConfig
from gateway.realtime_voice.session import RealtimeToolCall

logger = logging.getLogger(__name__)

AskAgentCallable = Callable[[str], Awaitable[str] | str]

_HERMES_TOOL_NAMES = {
    "ask_agent",
    "start_agent_task",
    "get_agent_task_status",
    "summarize_agent_task",
}


@dataclass(frozen=True)
class RealtimeToolDefinition:
    """Provider-neutral custom tool definition exposed to realtime providers."""

    name: str
    description: str
    parameters: dict[str, Any]


def hermes_realtime_tool_definitions(allow_tools: tuple[str, ...] | list[str] | None) -> list[RealtimeToolDefinition]:
    """Return neutral Hermes custom tool definitions allowed by config."""

    allowed = {_normalize_tool_name(name) for name in (allow_tools or ())}
    definitions = {
        "ask_agent": RealtimeToolDefinition(
            name="ask_agent",
            description="Ask the Hermes agent to answer a short realtime voice question. Use for factual/contextual answers, not long-running file or system work.",
            parameters={
                "type": "object",
                "properties": {
                    "prompt": {
                        "type": "string",
                        "description": "The concise question or request to send to Hermes.",
                    }
                },
                "required": ["prompt"],
                "additionalProperties": False,
            },
        ),
        "start_agent_task": RealtimeToolDefinition(
            name="start_agent_task",
            description="Start a bounded background Hermes task from realtime voice and return a task_id immediately.",
            parameters={
                "type": "object",
                "properties": {
                    "prompt": {
                        "type": "string",
                        "description": "The task for Hermes to work on in the background.",
                    }
                },
                "required": ["prompt"],
                "additionalProperties": False,
            },
        ),
        "get_agent_task_status": RealtimeToolDefinition(
            name="get_agent_task_status",
            description="Get the status of a realtime background Hermes task by task_id.",
            parameters={
                "type": "object",
                "properties": {
                    "task_id": {
                        "type": "string",
                        "description": "The task_id returned by start_agent_task.",
                    }
                },
                "required": ["task_id"],
                "additionalProperties": False,
            },
        ),
        "summarize_agent_task": RealtimeToolDefinition(
            name="summarize_agent_task",
            description="Return the current summary/result for a realtime background Hermes task by task_id.",
            parameters={
                "type": "object",
                "properties": {
                    "task_id": {
                        "type": "string",
                        "description": "The task_id returned by start_agent_task.",
                    }
                },
                "required": ["task_id"],
                "additionalProperties": False,
            },
        ),
    }
    return [definitions[name] for name in definitions if name in allowed]


@dataclass
class _RealtimeBackgroundTask:
    task_id: str
    prompt: str
    task: asyncio.Task
    status: str = "running"
    result: str | None = None
    error: str | None = None


class RealtimeToolBridge:
    """Execute allowed Hermes realtime tools and return results to a provider."""

    def __init__(
        self,
        config: RealtimeVoiceConfig,
        *,
        ask_agent: AskAgentCallable,
    ) -> None:
        self.config = config
        self.ask_agent = ask_agent
        self.allowed_tools = {
            _normalize_tool_name(name)
            for name in (getattr(config, "allow_tools", ()) or ())
        }
        self.max_background_tasks = max(0, int(getattr(config, "max_background_tasks", 0) or 0))
        self._tasks: dict[str, _RealtimeBackgroundTask] = {}

    async def handle_tool_call(self, provider_session: Any, tool_call: RealtimeToolCall) -> None:
        """Execute one provider tool call and submit exactly one provider result."""

        call_id = str(getattr(tool_call, "call_id", "") or "")
        name = _normalize_tool_name(getattr(tool_call, "name", ""))
        arguments = getattr(tool_call, "arguments", {})
        if not isinstance(arguments, Mapping):
            arguments = {}

        try:
            if name not in self.allowed_tools or name not in _HERMES_TOOL_NAMES:
                output = f"Tool {name or '<unknown>'!r} is not allowed in this realtime voice session."
            elif name == "ask_agent":
                output = await self._ask_agent(str(arguments.get("prompt") or arguments.get("question") or ""))
            elif name == "start_agent_task":
                output = await self._start_agent_task(str(arguments.get("prompt") or ""))
            elif name == "get_agent_task_status":
                output = self._task_status(str(arguments.get("task_id") or ""))
            elif name == "summarize_agent_task":
                output = self._task_summary(str(arguments.get("task_id") or ""))
            else:  # pragma: no cover - guarded by _HERMES_TOOL_NAMES
                output = f"Tool {name!r} is not implemented."
        except Exception:
            logger.warning("Realtime voice tool %s failed", name or "<unknown>", exc_info=True)
            output = "The Hermes realtime tool failed safely. Try again or ask in text if this needs deeper work."

        await _submit_tool_result(provider_session, call_id, output)

    async def close(self) -> None:
        """Cancel outstanding background tasks during provider/session cleanup."""

        tasks = [entry.task for entry in self._tasks.values() if not entry.task.done()]
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    async def _ask_agent(self, prompt: str) -> str:
        clean = prompt.strip()
        if not clean:
            return "No prompt was provided for ask_agent."
        result = self.ask_agent(clean)
        if inspect.isawaitable(result):
            result = await result
        return _safe_text(result)

    async def _start_agent_task(self, prompt: str) -> str:
        clean = prompt.strip()
        if not clean:
            return "No prompt was provided for start_agent_task."
        self._prune_finished_tasks()
        running = [entry for entry in self._tasks.values() if not entry.task.done()]
        if len(running) >= self.max_background_tasks:
            return f"Maximum realtime background tasks reached ({self.max_background_tasks}). Wait for one to finish before starting another."
        task_id = f"rt_{uuid.uuid4().hex[:10]}"
        task = asyncio.create_task(self._run_background_task(task_id, clean), name=f"realtime-voice-tool-{task_id}")
        self._tasks[task_id] = _RealtimeBackgroundTask(task_id=task_id, prompt=clean, task=task)
        return json.dumps({"task_id": task_id, "status": "running"})

    async def _run_background_task(self, task_id: str, prompt: str) -> None:
        entry = self._tasks[task_id]
        try:
            entry.result = await self._ask_agent(prompt)
            entry.status = "completed"
        except asyncio.CancelledError:
            entry.status = "cancelled"
            raise
        except Exception:
            logger.warning("Realtime voice background task %s failed", task_id, exc_info=True)
            entry.status = "failed"
            entry.error = "The Hermes realtime background task failed safely."

    def _task_status(self, task_id: str) -> str:
        entry = self._tasks.get(task_id.strip())
        if entry is None:
            return f"No realtime background task found for task_id {task_id!r}."
        if entry.task.done() and entry.status == "running":
            self._finalize_done_task(entry)
        return json.dumps({"task_id": entry.task_id, "status": entry.status})

    def _task_summary(self, task_id: str) -> str:
        entry = self._tasks.get(task_id.strip())
        if entry is None:
            return f"No realtime background task found for task_id {task_id!r}."
        if entry.task.done() and entry.status == "running":
            self._finalize_done_task(entry)
        if entry.status == "completed":
            return entry.result or "Task completed with no text result."
        if entry.status == "failed":
            return entry.error or "The realtime background task failed."
        return json.dumps({"task_id": entry.task_id, "status": entry.status})

    def _prune_finished_tasks(self) -> None:
        # Keep completed entries available for summary; only remove cancelled old entries.
        for task_id, entry in list(self._tasks.items()):
            if entry.task.done() and entry.status == "running":
                self._finalize_done_task(entry)
            if entry.status == "cancelled":
                self._tasks.pop(task_id, None)

    def _finalize_done_task(self, entry: _RealtimeBackgroundTask) -> None:
        try:
            exc = entry.task.exception()
        except asyncio.CancelledError:
            entry.status = "cancelled"
            return
        if exc is not None:
            entry.status = "failed"
            entry.error = "The Hermes realtime background task failed safely."
        elif entry.status == "running":
            entry.status = "completed"


def _normalize_tool_name(name: Any) -> str:
    return str(name or "").strip().lower()


def _safe_text(value: Any) -> str:
    if value is None:
        return ""
    text = str(value)
    return text[:8000]


async def _submit_tool_result(provider_session: Any, call_id: str, output: str) -> None:
    submit = getattr(provider_session, "submit_tool_result", None)
    if not callable(submit):
        logger.warning("Realtime provider session cannot accept tool results")
        return
    result = submit(call_id, _safe_text(output))
    if inspect.isawaitable(result):
        await result
