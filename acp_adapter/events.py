"""Callback factories for bridging AIAgent events to ACP notifications.

Each factory returns a callable with the signature AIAgent expects for its
callbacks. AIAgent runs in a worker thread while the event loop lives on the
main thread, so updates are pushed via ``conn.session_update()`` scheduled
thread-safely onto the loop.
"""

import asyncio
import logging
import threading
import uuid
from collections import deque
from contextlib import nullcontext
from typing import Any, Callable, Deque, Dict

import acp
from acp.schema import AgentPlanUpdate, PlanEntry

from .tools import (
    _json_loads_maybe, build_delegation_progress, build_tool_abandoned, build_tool_complete, build_tool_start,
    coerce_tool_args, make_tool_call_id,
)

logger = logging.getLogger(__name__)

# ACP plans only support pending/in_progress/completed. Cancelled tasks are kept
# as terminal entries so the client's full-list replacement doesn't drop them.
_PLAN_STATUS = {"pending": "pending", "in_progress": "in_progress", "completed": "completed", "cancelled": "completed"}


def _build_plan_update_from_todo_result(result: Any) -> AgentPlanUpdate | None:
    """Translate Hermes' todo tool result into ACP's native plan update.

    Zed renders ``sessionUpdate: plan`` as its first-class task panel, so the
    todo state is exposed natively rather than only as a tool-call transcript."""
    if not isinstance(result, str) or not result.strip():
        return None
    data = _json_loads_maybe(result)
    if not isinstance(data, dict) or not isinstance(data.get("todos"), list):
        return None

    entries: list[PlanEntry] = []
    for item in data["todos"]:
        if not isinstance(item, dict):
            continue
        content = str(item.get("content") or item.get("id") or "").strip()
        if not content:
            continue
        raw_status = str(item.get("status") or "pending").strip()
        if raw_status == "cancelled":
            content = f"[cancelled] {content}"
        entries.append(PlanEntry(content=content, priority="medium", status=_PLAN_STATUS.get(raw_status, "pending")))
    return AgentPlanUpdate(session_update="plan", entries=entries)


def _send_update(conn: acp.Client, session_id: str, loop: asyncio.AbstractEventLoop, update: Any) -> None:
    """Fire-and-forget an ACP session update from a worker thread."""
    from agent.async_utils import safe_schedule_threadsafe

    future = safe_schedule_threadsafe(
        conn.session_update(session_id, update), loop, logger=logger, log_message="Failed to send ACP update",
    )
    if future is None:
        return
    try:
        future.result(timeout=5)
    except Exception:
        logger.debug("Failed to send ACP update", exc_info=True)


def _upgrade_queue(tool_call_ids: Dict[str, Deque[str]], name: str) -> Deque[str] | None:
    """Fetch the per-tool FIFO of pending call IDs, upgrading a legacy bare-string entry in place."""
    queue = tool_call_ids.get(name)
    if isinstance(queue, str):
        queue = tool_call_ids[name] = deque([queue])
    return queue


def close_tool_call(
    conn: acp.Client, session_id: str, loop: asyncio.AbstractEventLoop, tool_call_ids: Dict[str, Deque[str]],
    tool_call_meta: Dict[str, Dict[str, Any]], name: str, result: Any = None, is_error: bool | None = None,
) -> str | None:
    """Close the oldest open ACP tool call for ``name``; returns its id, or None when none is open."""
    queue = _upgrade_queue(tool_call_ids, name)
    if not queue:
        return None
    tc_id = queue.popleft()
    meta = tool_call_meta.get(tc_id, {})
    completion_kwargs = {
        "result": str(result) if result is not None else None,
        "function_args": meta.get("args"),
        "snapshot": meta.get("snapshot"),
    }
    if is_error is not None:
        completion_kwargs["is_error"] = is_error
    update = build_tool_complete(tc_id, name, **completion_kwargs)
    # A background delegation's terminal tool result is a dispatch handle: the
    # host's call stays open (raw_output.lifecycle marks it dispatched) and keeps
    # receiving child progress until the background completion closes it.
    lifecycle = update.raw_output if isinstance(update.raw_output, dict) else {}
    background = name == "delegate_task" and lifecycle.get("lifecycle") == {
        "status": "dispatched", "mode": "background"
    }
    if background:
        meta["background_delegation"] = True
        parsed_result = _json_loads_maybe(str(result) if result is not None else None)
        if isinstance(parsed_result, dict) and isinstance(parsed_result.get("delegation_id"), str):
            meta["delegation_id"] = parsed_result["delegation_id"]
        if meta.get("delegation_all_completed"):
            # The last child finished before the dispatch result was projected.
            # Close immediately with the tracked terminal status — no progress
            # update will ever arrive to do it.
            update.status = meta.get("delegation_final_status", "completed")
            tool_call_meta.pop(tc_id, None)
    else:
        tool_call_meta.pop(tc_id, None)
    _send_update(conn, session_id, loop, update)
    if not queue:
        tool_call_ids.pop(name, None)
    return tc_id


def flush_open_tool_calls(
    conn: acp.Client, session_id: str, loop: asyncio.AbstractEventLoop, tool_call_ids: Dict[str, Deque[str]],
    tool_call_meta: Dict[str, Dict[str, Any]], tool_call_lock: Any = None,
) -> int:
    """Close every tool call still open at the end of a turn, and report how many there were.

    A tool blocked by scope, guardrail or an editor permission prompt never
    projects ``tool.completed``, so without this its bubble stays ``in_progress``
    forever and clients read the turn as one that never ran a tool."""
    with tool_call_lock if tool_call_lock is not None else nullcontext():
        open_calls = [(name, list(queue)) for name, queue in list(tool_call_ids.items()) if queue]
        flushed = 0
        for name, ids in open_calls:
            for tc_id in ids:
                tool_call_meta.pop(tc_id, None)
                _send_update(conn, session_id, loop, build_tool_abandoned(tc_id, name))
                flushed += 1
            tool_call_ids.pop(name, None)
    if flushed:
        logger.debug("Flushed %d ACP tool call(s) left open at turn end", flushed)
    return flushed


def _delegation_tool_call_id(
    tool_call_ids: Dict[str, Deque[str]], tool_call_meta: Dict[str, Dict[str, Any]], event: Dict[str, Any]
) -> str | None:
    """Resolve a child event to its open or background parent call without exposing child input.

    ``delegation_id`` pins the parent once known (background dispatch stamps it from
    the tool result). Before that, the (task_index, goal) pair disambiguates parallel
    calls; a single unbound call absorbs unattributable events. A two-way ambiguity
    resolves to None — guessing would misroute progress onto a sibling's row."""
    queue = _upgrade_queue(tool_call_ids, "delegate_task")
    candidates = list(queue or ())
    candidates.extend(
        tc_id
        for tc_id, meta in tool_call_meta.items()
        if meta.get("background_delegation") and tc_id not in candidates
    )
    if not candidates:
        return None

    delegation_id = event.get("delegation_id")
    if isinstance(delegation_id, str) and delegation_id:
        for tc_id in candidates:
            if tool_call_meta.get(tc_id, {}).get("delegation_id") == delegation_id:
                return tc_id

    task_index = event.get("task_index")
    goal = event.get("goal")
    matches: list[str] = []
    if isinstance(task_index, int) and not isinstance(task_index, bool) and task_index >= 0 and isinstance(goal, str):
        for tc_id in candidates:
            meta = tool_call_meta.get(tc_id, {})
            arguments = meta.get("args")
            if not isinstance(arguments, dict):
                continue
            tasks = arguments.get("tasks")
            task_goal = None
            if isinstance(tasks, list) and task_index < len(tasks) and isinstance(tasks[task_index], dict):
                task_goal = tasks[task_index].get("goal")
            elif task_index == 0:
                task_goal = arguments.get("goal")
            if task_goal == goal and not meta.get("delegation_id"):
                matches.append(tc_id)

    if len(matches) == 1:
        tc_id = matches[0]
    else:
        unbound = [tc_id for tc_id in candidates if not tool_call_meta.get(tc_id, {}).get("delegation_id")]
        if len(unbound) != 1:
            return None
        tc_id = unbound[0]

    if isinstance(delegation_id, str) and delegation_id:
        tool_call_meta.setdefault(tc_id, {})["delegation_id"] = delegation_id
    return tc_id


def _delegation_task_count(meta: Dict[str, Any]) -> int:
    """How many children this parent call fan out to (from sanitized arguments)."""
    arguments = meta.get("args")
    if not isinstance(arguments, dict):
        return 0
    tasks = arguments.get("tasks")
    if isinstance(tasks, list):
        return sum(isinstance(task, dict) and isinstance(task.get("goal"), str) for task in tasks)
    return 1 if isinstance(arguments.get("goal"), str) and arguments["goal"] else 0


def make_tool_progress_cb(
    conn: acp.Client, session_id: str, loop: asyncio.AbstractEventLoop, tool_call_ids: Dict[str, Deque[str]],
    tool_call_meta: Dict[str, Dict[str, Any]],
    edit_approval_policy_getter: Callable[[], tuple[str, str | None]] | None = None,
    turn_state: Dict[str, Any] | None = None,
    tool_call_lock: Any = None,
) -> Callable:
    """Create a ``tool_progress_callback`` for AIAgent.

    Signature: ``tool_progress_callback(event_type, name, preview, args, **kwargs)``.
    Emits ``ToolCallStart`` for ``tool.started`` and tracks IDs in a FIFO per tool
    name so parallel same-name calls complete against the right ACP tool call.
    ``tool.completed`` closes that call with its own result — the step callback
    only fires on the *next* step, which leaves a turn's last tools open."""

    # Child events arrive on child worker threads while the parent turn runs on
    # the executor: per-call sequence/terminal bookkeeping must be atomic.
    delegation_lock = tool_call_lock or threading.RLock()

    def _tool_progress(event_type: str, name: str = None, preview: str = None, args: Any = None, **kwargs) -> None:
        if event_type.startswith("subagent."):
            event = {"preview": preview, "tool_name": name, **kwargs}
            with delegation_lock:
                tc_id = _delegation_tool_call_id(tool_call_ids, tool_call_meta, event)
                if tc_id is None:
                    return
                meta = tool_call_meta.setdefault(tc_id, {})
                sequence = int(meta.get("delegation_progress_sequence") or 0) + 1
                terminal_status = None
                if event_type == "subagent.complete":
                    task_index = event.get("task_index")
                    if isinstance(task_index, int) and not isinstance(task_index, bool) and task_index >= 0:
                        completed = meta.setdefault("delegation_completed_tasks", set())
                        completed.add(task_index)
                    raw_status = str(event.get("status") or "completed").strip().lower()
                    if raw_status not in {"completed", "success", "succeeded"}:
                        meta["delegation_had_failure"] = True
                    expected = _delegation_task_count(meta)
                    if expected > 0 and len(meta.get("delegation_completed_tasks", ())) >= expected:
                        # Every child reported: the parent's host call can close.
                        terminal_status = "failed" if meta.get("delegation_had_failure") else "completed"
                        meta["delegation_all_completed"] = True
                        meta["delegation_final_status"] = terminal_status
                update = build_delegation_progress(
                    tc_id, event_type, sequence, terminal_status=terminal_status, **event
                )
                if update is not None:
                    meta["delegation_progress_sequence"] = sequence
                    _send_update(conn, session_id, loop, update)
                    if terminal_status is not None and meta.get("background_delegation"):
                        tool_call_meta.pop(tc_id, None)
            return
        if event_type == "tool.completed" and name:
            if turn_state is not None:
                turn_state["saw_completion"] = True
            # The executor's verdict: a cancelled/errored tool may return plain text the heuristic misses.
            if name == "delegate_task":
                # close_tool_call consults progress bookkeeping (the child-can-win
                # race) — it must not interleave with a subagent.* relay.
                with delegation_lock:
                    close_tool_call(
                        conn, session_id, loop, tool_call_ids, tool_call_meta, name, kwargs.get("result"),
                        is_error=bool(kwargs.get("is_error")),
                    )
            else:
                close_tool_call(
                    conn, session_id, loop, tool_call_ids, tool_call_meta, name, kwargs.get("result"),
                    is_error=bool(kwargs.get("is_error")),
                )
            return
        if event_type != "tool.started":
            return
        args = coerce_tool_args(args)
        with delegation_lock:
            tc_id = make_tool_call_id()
            queue = _upgrade_queue(tool_call_ids, name)
            if queue is None:
                queue = tool_call_ids[name] = deque()
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

            _send_update(conn, session_id, loop, build_tool_start(tc_id, name, args, edit_diff=edit_diff))

    return _tool_progress


# ------------------------------------------------------------------
# Assistant message identity
# ------------------------------------------------------------------


class AssistantMessageIdAllocator:
    """Allocates stable per-message ids for streamed assistant chunks.

    ACP clients group streamed ``agent_message_chunk`` / ``agent_thought_chunk``
    deltas into one assistant reply by ``messageId`` and use a NEW id to start
    the next reply (root-reply replacement semantics). Without ids, a client
    that replaces "the current assistant message" on each chunk collapses
    separate autonomous turns into one bubble.

    One allocator lives per ACP session; a contiguous run of deltas shares
    ``current()`` and ``close()`` marks the message finished so the next delta
    allocates a fresh id. Ids are UUID4 strings because the ACP schema requires
    UUID-format message ids, and a fresh UUID can never collide with an earlier
    turn's id.
    """

    def __init__(self) -> None:
        self._active: str | None = None
        self._last: str | None = None

    def current(self) -> str:
        """Return the active message id, allocating one if none is open."""
        if self._active is None:
            self._active = self._last = str(uuid.uuid4())
        return self._active

    def last(self) -> str | None:
        """Return the most recently allocated id (open or closed)."""
        return self._last

    def close(self) -> None:
        """End the active message; the next chunk starts a new id."""
        self._active = None


def _make_text_cb(
    conn: acp.Client, session_id: str, loop: asyncio.AbstractEventLoop, wrap: Callable[[str], Any],
    message_ids: AssistantMessageIdAllocator | None = None,
) -> Callable:
    # ``None`` is the flush sentinel Hermes core sends between assistant messages
    # (before tool execution / at end of stream): it closes the active messageId so
    # the next delta opens a new bubble instead of merging into the previous one.
    def _cb(text: str | None) -> None:
        if text:
            update = wrap(text)
            if message_ids is not None:
                update.message_id = message_ids.current()
            _send_update(conn, session_id, loop, update)
        elif text is None and message_ids is not None:
            message_ids.close()

    return _cb


def make_thinking_cb(
    conn: acp.Client, session_id: str, loop: asyncio.AbstractEventLoop,
    message_ids: AssistantMessageIdAllocator | None = None,
) -> Callable:
    """Create a ``thinking_callback`` for AIAgent."""
    return _make_text_cb(conn, session_id, loop, acp.update_agent_thought_text, message_ids)


def make_message_cb(
    conn: acp.Client, session_id: str, loop: asyncio.AbstractEventLoop,
    message_ids: AssistantMessageIdAllocator | None = None,
) -> Callable:
    """Create a callback that streams agent response text to the editor."""
    return _make_text_cb(conn, session_id, loop, acp.update_agent_message_text, message_ids)


def make_step_cb(
    conn: acp.Client, session_id: str, loop: asyncio.AbstractEventLoop, tool_call_ids: Dict[str, Deque[str]],
    tool_call_meta: Dict[str, Dict[str, Any]], turn_state: Dict[str, Any] | None = None,
    tool_call_lock: Any = None,
) -> Callable:
    """Create a ``step_callback(api_call_count: int, prev_tools: list)`` for AIAgent."""

    def _step(api_call_count: int, prev_tools: Any = None) -> None:
        if not isinstance(prev_tools, list):
            return
        for tool_info in prev_tools:
            tool_name = result = function_args = None
            if isinstance(tool_info, dict):
                tool_name = tool_info.get("name") or tool_info.get("function_name")
                # Key presence, not truthiness: "", 0 and False are real results (#10845).
                result = tool_info.get("result") if "result" in tool_info else tool_info.get("output")
                function_args = tool_info.get("arguments") or tool_info.get("args")
            elif isinstance(tool_info, str):
                tool_name = tool_info

            if not tool_name:
                continue
            # ``tool.completed`` already closed this call with its own result;
            # this callback is the fallback for runtimes that never project one.
            if not (turn_state or {}).get("saw_completion"):
                with tool_call_lock if tool_call_lock is not None else nullcontext():
                    queue = _upgrade_queue(tool_call_ids, tool_name)
                    if not queue:
                        continue
                    tc_id = queue[0]
                    if function_args:
                        tool_call_meta.setdefault(tc_id, {})["args"] = coerce_tool_args(function_args)
                    # Use the same close path as a native tool.completed callback.
                    # delegate_task dispatch handles remain open and retain their
                    # routing metadata for later child progress.
                    close_tool_call(
                        conn, session_id, loop, tool_call_ids, tool_call_meta, tool_name,
                        result=str(result) if result is not None else None,
                    )
            if tool_name == "todo" and (plan_update := _build_plan_update_from_todo_result(result)) is not None:
                _send_update(conn, session_id, loop, plan_update)

    return _step


# ---- BEGIN PLUGIN-COMPAT (revert-scheduled; see COMPAT_MANIFEST.md) ----
# Names external plugins imported from this module before the Sep 2026 decomposition.
# Internal code MUST NOT use these (scripts/check_compat_pointers.py fails CI if it does).
# The whole block is removed by reverting the commit that added it.
import json  # noqa: F401,E402
# ---- END PLUGIN-COMPAT ----
