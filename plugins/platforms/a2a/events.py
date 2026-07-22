"""Translate AIAgent callbacks into A2A ``TaskUpdater`` events.

``AIAgent.run_conversation`` runs synchronously in a worker thread, but the A2A
event queue lives on the server's asyncio loop. These callback factories marshal
each agent event back onto that loop with ``run_coroutine_threadsafe`` and block
briefly on it — so updates preserve order relative to the agent's own progress,
and all working-status updates land before the final artifact.

Best-effort: a failed status update is logged and swallowed, never aborting the
turn. This module intentionally imports nothing from Hermes, so the adapter is
unit-testable on its own.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import json
import logging
from typing import Any, Callable

from a2a.server.tasks import TaskUpdater
from a2a.types import Part, TaskState, TextPart

logger = logging.getLogger(__name__)

# How long a worker-thread callback waits for the loop to enqueue its event.
# Bounded (and modest) so a slow or disconnected SSE consumer can't throttle the
# agent loop to a crawl — each progress update is best-effort. Matches the ACP
# adapter's 5s ceiling.
_SCHEDULE_TIMEOUT = 5.0
_RESULT_PREVIEW_LIMIT = 2000


def _tool_call_metadata(
    name: str | None, args: Any, tool_io_mode: str = "preview"
) -> dict[str, Any]:
    """Metadata for a ``tool.started`` status update, honoring the I/O mode."""
    metadata: dict[str, Any] = {"hermes/kind": "tool-call", "hermes/tool": name}
    if tool_io_mode == "none":
        return metadata
    metadata["hermes/args"] = (
        _json_safe(args) if tool_io_mode == "full" else _bounded(args)
    )
    return metadata


def _tool_result_metadata(
    name: str | None, result: Any, tool_io_mode: str = "preview"
) -> dict[str, Any]:
    """Metadata for a completed-tool status update, honoring the I/O mode."""
    metadata: dict[str, Any] = {"hermes/kind": "tool-result", "hermes/tool": name}
    if tool_io_mode == "none":
        return metadata
    metadata["hermes/result"] = (
        _json_safe(result) if tool_io_mode == "full" else _bounded(result)
    )
    return metadata


def _schedule(loop: asyncio.AbstractEventLoop, coro: Any) -> None:
    """Run *coro* on *loop* from a worker thread and wait for it (bounded)."""
    try:
        future = asyncio.run_coroutine_threadsafe(coro, loop)
    except RuntimeError:
        # Loop already closed — drop the update.
        return
    try:
        future.result(timeout=_SCHEDULE_TIMEOUT)
    except concurrent.futures.TimeoutError:
        # Cancel so the orphaned coroutine can't run later and emit a status
        # update after the task has already reached a terminal state.
        future.cancel()
        logger.debug("A2A status update timed out; cancelled")
    except Exception:
        logger.debug("A2A status update failed", exc_info=True)


def _emit_working(
    updater: TaskUpdater,
    loop: asyncio.AbstractEventLoop,
    text: str,
    metadata: dict[str, Any] | None = None,
) -> None:
    message = (
        updater.new_agent_message([Part(root=TextPart(text=text))]) if text else None
    )
    _schedule(
        loop,
        updater.update_status(TaskState.working, message=message, metadata=metadata),
    )


def make_stream_delta_cb(
    updater: TaskUpdater, loop: asyncio.AbstractEventLoop
) -> Callable[[str], None]:
    """Stream incremental agent text as working-status message chunks."""

    def _cb(text: str) -> None:
        if text:
            _emit_working(updater, loop, text)

    return _cb


def make_reasoning_cb(
    updater: TaskUpdater, loop: asyncio.AbstractEventLoop
) -> Callable[[str], None]:
    """Surface provider/model reasoning, tagged so clients can render it apart."""

    def _cb(text: str) -> None:
        if text:
            _emit_working(updater, loop, text, metadata={"hermes/kind": "reasoning"})

    return _cb


def make_tool_progress_cb(
    updater: TaskUpdater,
    loop: asyncio.AbstractEventLoop,
    *,
    tool_io_mode: str = "preview",
) -> Callable[..., None]:
    """Report tool-call starts as working-status updates with tool metadata.

    Matches AIAgent's signature:
    ``tool_progress_callback(event_type, name, preview, args, **kwargs)``.
    """

    def _cb(
        event_type: str | None = None,
        name: str | None = None,
        preview: str | None = None,
        args: Any = None,
        **_kwargs: Any,
    ) -> None:
        if event_type != "tool.started":
            return
        _emit_working(
            updater,
            loop,
            f"⚙ {name}",
            metadata=_tool_call_metadata(name, args, tool_io_mode),
        )

    return _cb


def make_step_cb(
    updater: TaskUpdater,
    loop: asyncio.AbstractEventLoop,
    *,
    tool_io_mode: str = "preview",
) -> Callable[..., None]:
    """Report completed tool calls from AIAgent's ``step_callback`` payload.

    Signature: ``step_callback(api_call_count, prev_tools)`` where ``prev_tools``
    is a list of dicts describing the tools that ran in the previous step.
    """

    def _cb(api_call_count: int | None = None, prev_tools: Any = None) -> None:
        if not isinstance(prev_tools, list):
            return
        for tool in prev_tools:
            if not isinstance(tool, dict):
                continue
            name = tool.get("name") or tool.get("function_name")
            if not name:
                continue
            result = tool.get("result") or tool.get("output")
            _emit_working(
                updater,
                loop,
                f"✓ {name}",
                metadata=_tool_result_metadata(name, result, tool_io_mode),
            )

    return _cb


def _json_safe(value: Any) -> Any:
    """Coerce *value* to something JSON-serializable for event metadata."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    try:
        json.dumps(value)
        return value
    except (TypeError, ValueError):
        return str(value)


def _truncate(value: Any, limit: int = _RESULT_PREVIEW_LIMIT) -> Any:
    if not isinstance(value, str):
        return value
    return value if len(value) <= limit else value[:limit] + "…"


def _bounded(value: Any, limit: int = _RESULT_PREVIEW_LIMIT) -> Any:
    """Bound a value for peer-facing metadata.

    Small structured values pass through unchanged; large strings or large
    structures are rendered to a truncated string preview so a single tool call
    can't blast megabytes (or a wall of secrets) at the peer.
    """
    safe = _json_safe(value)
    if isinstance(safe, str):
        return _truncate(safe, limit)
    try:
        serialized = json.dumps(safe)
    except (TypeError, ValueError):
        return _truncate(str(safe), limit)
    return safe if len(serialized) <= limit else _truncate(serialized, limit)
