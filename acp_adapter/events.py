"""Callback factories for bridging AIAgent events to ACP notifications.

Each factory returns a callable with the signature that AIAgent expects
for its callbacks. Internally, the callbacks push ACP session updates
to the client via ``conn.session_update()`` using
``asyncio.run_coroutine_threadsafe()`` (since AIAgent runs in a worker
thread while the event loop lives on the main thread).
"""

import asyncio
import json
import logging
import re
import threading
from collections import deque
from typing import Any, Callable, Deque, Dict

import acp
from acp.schema import AgentPlanUpdate, PlanEntry

from .tools import (
    build_tool_complete,
    build_tool_start,
    make_tool_call_id,
)

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------
# Live subagent (delegate_task) progress
# ------------------------------------------------------------------
# delegate_task runs SYNCHRONOUSLY, so the parent emits one tool_call (start)
# and one tool_call_update (results) with nothing in between — the VS Code
# Fleet view only learns per-subagent status at the very end. We close that gap
# by polling the thread-safe live registry (tools.delegate_tool.
# list_active_subagents) on a daemon thread while the batch runs and emitting
# intermediate tool_call_update frames the FleetModel parses. No core-loop
# change; reuses the thread-safe _send_update.

_DELEGATE_TOOL_NAMES = {"delegate_task"}
# subagent_id is "sa-<task_index>-<uuid>"; the FleetModel keys children by
# K = task_index + 1, so the index here drives the "Task K" line it parses.
_SA_INDEX_RE = re.compile(r"^sa-(\d+)-")
_POLL_INTERVAL_S = 1.0
_POLL_MAX_S = 1800.0  # backstop: a poll thread can never leak past this


def build_subagent_progress_update(tool_call_id: str, active_subagents: Any) -> Any:
    """Encode a live subagent-registry snapshot into a delegate-batch
    ``tool_call_update`` the VS Code FleetModel parses.

    Pure (no I/O). ``active_subagents`` is the ``list_active_subagents()``
    snapshot. Emits one ``Task K`` line per subagent whose id encodes a task
    index, with its status + tool_count. Returns an ACP ``tool_call_update`` or
    ``None`` when nothing maps (so callers skip emitting an empty frame).
    """
    rows = []
    for sa in active_subagents or []:
        if not isinstance(sa, dict):
            continue
        m = _SA_INDEX_RE.match(sa.get("subagent_id") or "")
        if not m:
            continue
        k = int(m.group(1)) + 1
        status = (sa.get("status") or "running").strip().lower()
        glyph = "✅" if status in {"done", "completed"} else "❌" if status in {"error", "failed"} else "🔄"
        tools = sa.get("tool_count")
        suffix = f" ({tools} tools)" if isinstance(tools, int) else ""
        rows.append((k, f"{glyph} Task {k}: {status}{suffix}"))
    if not rows:
        return None
    rows.sort(key=lambda r: r[0])
    text = "Delegation progress:\n" + "\n".join(line for _, line in rows)
    return acp.update_tool_call(
        tool_call_id,
        kind="execute",
        status="in_progress",
        content=[acp.tool_content(acp.text_block(text))],
    )


def _start_delegate_poll(
    conn: "acp.Client",
    session_id: str,
    loop: asyncio.AbstractEventLoop,
    tool_call_id: str,
    delegate_polls: Dict[str, threading.Event] | None,
) -> None:
    """Start a daemon thread emitting live subagent progress for a running
    delegate_task batch, until stopped (on completion) or a hard backstop."""
    if delegate_polls is None or tool_call_id in delegate_polls:
        return
    stop = threading.Event()

    def _run() -> None:
        try:
            from tools.delegate_tool import list_active_subagents
        except Exception:
            return
        elapsed = 0.0
        last_key = None
        while not stop.wait(_POLL_INTERVAL_S):
            elapsed += _POLL_INTERVAL_S
            if elapsed > _POLL_MAX_S:
                return
            try:
                snapshot = list_active_subagents()
            except Exception:
                continue
            update = build_subagent_progress_update(tool_call_id, snapshot)
            if update is None:
                continue
            key = str(getattr(update, "content", None))
            if key == last_key:  # don't spam identical frames
                continue
            last_key = key
            try:
                _send_update(conn, session_id, loop, update)
            except Exception:
                logger.debug("subagent progress emit failed", exc_info=True)

    delegate_polls[tool_call_id] = stop
    threading.Thread(
        target=_run, name=f"acp-subagent-poll-{tool_call_id[:8]}", daemon=True
    ).start()


def _stop_delegate_poll(
    tool_call_id: str, delegate_polls: Dict[str, threading.Event] | None
) -> None:
    if not delegate_polls:
        return
    stop = delegate_polls.pop(tool_call_id, None)
    if stop is not None:
        stop.set()


def _json_loads_maybe_prefix(value: str) -> Any:
    """Parse a JSON object even when Hermes appended a human hint after it."""
    text = value.strip()
    try:
        return json.loads(text)
    except Exception:
        decoder = json.JSONDecoder()
        data, _ = decoder.raw_decode(text)
        return data


def _build_plan_update_from_todo_result(result: Any) -> AgentPlanUpdate | None:
    """Translate Hermes' todo tool result into ACP's native plan update.

    Zed renders ``sessionUpdate: plan`` as its first-class task/todo panel. The
    Hermes agent already maintains task state through the ``todo`` tool, so the
    ACP adapter should expose that state natively instead of only as a generic
    tool-call transcript block.
    """
    if not isinstance(result, str) or not result.strip():
        return None

    try:
        data = _json_loads_maybe_prefix(result)
    except Exception:
        return None

    if not isinstance(data, dict) or not isinstance(data.get("todos"), list):
        return None

    todos = data["todos"]
    if not todos:
        return AgentPlanUpdate(session_update="plan", entries=[])

    status_map = {
        "pending": "pending",
        "in_progress": "in_progress",
        "completed": "completed",
        # ACP plans only support pending/in_progress/completed. Preserve
        # cancelled tasks as terminal entries instead of dropping them and
        # making the client's full-list replacement lose visible context.
        "cancelled": "completed",
    }
    entries: list[PlanEntry] = []
    for item in todos:
        if not isinstance(item, dict):
            continue
        content = str(item.get("content") or item.get("id") or "").strip()
        if not content:
            continue
        raw_status = str(item.get("status") or "pending").strip()
        status = status_map.get(raw_status, "pending")
        if raw_status == "cancelled":
            content = f"[cancelled] {content}"
        entries.append(PlanEntry(content=content, priority="medium", status=status))

    return AgentPlanUpdate(session_update="plan", entries=entries)


def _send_update(
    conn: acp.Client,
    session_id: str,
    loop: asyncio.AbstractEventLoop,
    update: Any,
) -> None:
    """Fire-and-forget an ACP session update from a worker thread."""
    from agent.async_utils import safe_schedule_threadsafe

    future = safe_schedule_threadsafe(
        conn.session_update(session_id, update),
        loop,
        logger=logger,
        log_message="Failed to send ACP update",
    )
    if future is None:
        return
    try:
        future.result(timeout=5)
    except Exception:
        logger.debug("Failed to send ACP update", exc_info=True)


# ------------------------------------------------------------------
# Tool progress callback
# ------------------------------------------------------------------

def make_tool_progress_cb(
    conn: acp.Client,
    session_id: str,
    loop: asyncio.AbstractEventLoop,
    tool_call_ids: Dict[str, Deque[str]],
    tool_call_meta: Dict[str, Dict[str, Any]],
    edit_approval_policy_getter: Callable[[], tuple[str, str | None]] | None = None,
    delegate_polls: Dict[str, threading.Event] | None = None,
) -> Callable:
    """Create a ``tool_progress_callback`` for AIAgent.

    Signature expected by AIAgent::

        tool_progress_callback(event_type: str, name: str, preview: str, args: dict, **kwargs)

    Emits ``ToolCallStart`` for ``tool.started`` events and tracks IDs in a FIFO
    queue per tool name so duplicate/parallel same-name calls still complete
    against the correct ACP tool call.  Other event types (``tool.completed``,
    ``reasoning.available``) are silently ignored.
    """

    def _tool_progress(event_type: str, name: str = None, preview: str = None, args: Any = None, **kwargs) -> None:
        # Only emit ACP ToolCallStart for tool.started; ignore other event types
        if event_type != "tool.started":
            return
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except (json.JSONDecodeError, TypeError):
                args = {"raw": args}
        if not isinstance(args, dict):
            args = {}

        tc_id = make_tool_call_id()
        queue = tool_call_ids.get(name)
        if queue is None:
            queue = deque()
            tool_call_ids[name] = queue
        elif isinstance(queue, str):
            queue = deque([queue])
            tool_call_ids[name] = queue
        queue.append(tc_id)

        snapshot = None
        if name in {"write_file", "patch", "skill_manage"}:
            try:
                from agent.display import capture_local_edit_snapshot

                snapshot = capture_local_edit_snapshot(name, args)
            except Exception:
                logger.debug("Failed to capture ACP edit snapshot for %s", name, exc_info=True)
        tool_call_meta[tc_id] = {"args": args, "snapshot": snapshot}

        edit_diff = None
        if name in {"write_file", "patch"} and edit_approval_policy_getter is not None:
            try:
                from acp_adapter.edit_approval import build_edit_proposal, should_auto_approve_edit

                proposal = build_edit_proposal(name, args)
                if proposal is not None:
                    policy, cwd = edit_approval_policy_getter()
                    if should_auto_approve_edit(proposal, policy, cwd):
                        edit_diff = proposal
            except Exception:
                logger.debug("Failed to prepare auto-approved ACP edit diff for %s", name, exc_info=True)

        update = build_tool_start(tc_id, name, args, edit_diff=edit_diff)
        _send_update(conn, session_id, loop, update)

        # Live subagent progress: poll the registry until this batch completes.
        if name in _DELEGATE_TOOL_NAMES:
            _start_delegate_poll(conn, session_id, loop, tc_id, delegate_polls)

    return _tool_progress


# ------------------------------------------------------------------
# Thinking callback
# ------------------------------------------------------------------

def make_thinking_cb(
    conn: acp.Client,
    session_id: str,
    loop: asyncio.AbstractEventLoop,
) -> Callable:
    """Create a ``thinking_callback`` for AIAgent."""

    def _thinking(text: str) -> None:
        if not text:
            return
        update = acp.update_agent_thought_text(text)
        _send_update(conn, session_id, loop, update)

    return _thinking


# ------------------------------------------------------------------
# Step callback
# ------------------------------------------------------------------

def make_step_cb(
    conn: acp.Client,
    session_id: str,
    loop: asyncio.AbstractEventLoop,
    tool_call_ids: Dict[str, Deque[str]],
    tool_call_meta: Dict[str, Dict[str, Any]],
    delegate_polls: Dict[str, threading.Event] | None = None,
) -> Callable:
    """Create a ``step_callback`` for AIAgent.

    Signature expected by AIAgent::

        step_callback(api_call_count: int, prev_tools: list)
    """

    def _step(api_call_count: int, prev_tools: Any = None) -> None:
        if prev_tools and isinstance(prev_tools, list):
            for tool_info in prev_tools:
                tool_name = None
                result = None
                function_args = None

                if isinstance(tool_info, dict):
                    tool_name = tool_info.get("name") or tool_info.get("function_name")
                    result = tool_info.get("result") or tool_info.get("output")
                    function_args = tool_info.get("arguments") or tool_info.get("args")
                elif isinstance(tool_info, str):
                    tool_name = tool_info

                queue = tool_call_ids.get(tool_name or "")
                if isinstance(queue, str):
                    queue = deque([queue])
                    tool_call_ids[tool_name] = queue
                if tool_name and queue:
                    tc_id = queue.popleft()
                    meta = tool_call_meta.pop(tc_id, {})
                    update = build_tool_complete(
                        tc_id,
                        tool_name,
                        result=str(result) if result is not None else None,
                        function_args=function_args or meta.get("args"),
                        snapshot=meta.get("snapshot"),
                    )
                    _send_update(conn, session_id, loop, update)
                    if tool_name in _DELEGATE_TOOL_NAMES:
                        _stop_delegate_poll(tc_id, delegate_polls)
                    if tool_name == "todo":
                        plan_update = _build_plan_update_from_todo_result(result)
                        if plan_update is not None:
                            _send_update(conn, session_id, loop, plan_update)
                    if not queue:
                        tool_call_ids.pop(tool_name, None)

    return _step


# ------------------------------------------------------------------
# Agent message callback
# ------------------------------------------------------------------

def make_message_cb(
    conn: acp.Client,
    session_id: str,
    loop: asyncio.AbstractEventLoop,
) -> Callable:
    """Create a callback that streams agent response text to the editor."""

    def _message(text: str) -> None:
        if not text:
            return
        update = acp.update_agent_message_text(text)
        _send_update(conn, session_id, loop, update)

    return _message
