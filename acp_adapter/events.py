"""Callback factories for bridging AIAgent events to ACP notifications.

Each factory returns a callable with the signature AIAgent expects for its
callbacks. AIAgent runs in a worker thread while the event loop lives on the
main thread, so updates are pushed via ``conn.session_update()`` scheduled
thread-safely onto the loop.
"""

import asyncio
import logging
from collections import deque
from typing import Any, Callable, Deque, Dict

import acp
from acp.schema import AgentPlanUpdate, PlanEntry

from .tools import _json_loads_maybe, build_tool_complete, build_tool_start, coerce_tool_args, make_tool_call_id

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


def make_tool_progress_cb(
    conn: acp.Client, session_id: str, loop: asyncio.AbstractEventLoop, tool_call_ids: Dict[str, Deque[str]],
    tool_call_meta: Dict[str, Dict[str, Any]],
    edit_approval_policy_getter: Callable[[], tuple[str, str | None]] | None = None,
) -> Callable:
    """Create a ``tool_progress_callback`` for AIAgent.

    Signature: ``tool_progress_callback(event_type, name, preview, args, **kwargs)``.
    Emits ``ToolCallStart`` for ``tool.started`` and tracks IDs in a FIFO per tool
    name so parallel same-name calls complete against the right ACP tool call.
    Other event types (``tool.completed``, ``reasoning.available``) are ignored."""

    def _tool_progress(event_type: str, name: str = None, preview: str = None, args: Any = None, **kwargs) -> None:
        if event_type != "tool.started":
            return
        args = coerce_tool_args(args)
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


def _make_text_cb(conn: acp.Client, session_id: str, loop: asyncio.AbstractEventLoop, wrap: Callable[[str], Any]) -> Callable:
    def _cb(text: str) -> None:
        if text:
            _send_update(conn, session_id, loop, wrap(text))

    return _cb


def make_thinking_cb(conn: acp.Client, session_id: str, loop: asyncio.AbstractEventLoop) -> Callable:
    """Create a ``thinking_callback`` for AIAgent."""
    return _make_text_cb(conn, session_id, loop, acp.update_agent_thought_text)


def make_message_cb(conn: acp.Client, session_id: str, loop: asyncio.AbstractEventLoop) -> Callable:
    """Create a callback that streams agent response text to the editor."""
    return _make_text_cb(conn, session_id, loop, acp.update_agent_message_text)


def make_step_cb(
    conn: acp.Client, session_id: str, loop: asyncio.AbstractEventLoop, tool_call_ids: Dict[str, Deque[str]],
    tool_call_meta: Dict[str, Dict[str, Any]],
) -> Callable:
    """Create a ``step_callback(api_call_count: int, prev_tools: list)`` for AIAgent."""

    def _step(api_call_count: int, prev_tools: Any = None) -> None:
        if not isinstance(prev_tools, list):
            return
        for tool_info in prev_tools:
            tool_name = result = function_args = None
            if isinstance(tool_info, dict):
                tool_name = tool_info.get("name") or tool_info.get("function_name")
                result = tool_info.get("result") or tool_info.get("output")
                function_args = tool_info.get("arguments") or tool_info.get("args")
            elif isinstance(tool_info, str):
                tool_name = tool_info

            if not tool_name:
                continue
            queue = _upgrade_queue(tool_call_ids, tool_name)
            if not queue:
                continue
            tc_id = queue.popleft()
            meta = tool_call_meta.pop(tc_id, {})
            _send_update(conn, session_id, loop, build_tool_complete(
                tc_id, tool_name, result=str(result) if result is not None else None,
                function_args=function_args or meta.get("args"), snapshot=meta.get("snapshot"),
            ))
            if tool_name == "todo" and (plan_update := _build_plan_update_from_todo_result(result)) is not None:
                _send_update(conn, session_id, loop, plan_update)
            if not queue:
                tool_call_ids.pop(tool_name, None)

    return _step


# ---- BEGIN PLUGIN-COMPAT (revert-scheduled; see COMPAT_MANIFEST.md) ----
# Names external plugins imported from this module before the Sep 2026 decomposition.
# Internal code MUST NOT use these (scripts/check_compat_pointers.py fails CI if it does).
# The whole block is removed by reverting the commit that added it.
import json  # noqa: F401,E402
# ---- END PLUGIN-COMPAT ----
