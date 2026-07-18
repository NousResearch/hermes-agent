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
) -> bool:
    """Schedule an ACP update and report whether the loop accepted ownership.

    ``True`` means the coroutine was accepted by the event loop, not that it
    already finished. Accepted futures are observed asynchronously and must
    never be resubmitted: a bounded wait cannot distinguish a slow delivery
    from a lost one and retrying can duplicate completion updates (#33023).
    ``False`` is reserved for scheduler rejection, where the coroutine was
    closed and therefore provably cannot deliver.
    """
    from agent.async_utils import safe_schedule_threadsafe

    future = safe_schedule_threadsafe(
        conn.session_update(session_id, update),
        loop,
        logger=logger,
        log_message="Failed to send ACP update",
    )
    if future is None:
        return False

    def _observe_delivery(done) -> None:
        try:
            done.result()
        except Exception:
            logger.warning(
                "Accepted ACP update later failed "
                "(session=%s, update_type=%s)",
                session_id,
                type(update).__name__,
                exc_info=True,
            )

    future.add_done_callback(_observe_delivery)
    return True


# ------------------------------------------------------------------
# Tool start callbacks
# ------------------------------------------------------------------

def _emit_tool_start(
    conn: acp.Client,
    session_id: str,
    loop: asyncio.AbstractEventLoop,
    tool_call_ids: Dict[str, Deque[str]],
    tool_call_meta: Dict[str, Dict[str, Any]],
    tc_id: str,
    name: str,
    args: Any,
    edit_approval_policy_getter: Callable[[], tuple[str, str | None]] | None,
) -> None:
    """Track and emit one ACP tool start using the supplied canonical ID."""
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except (json.JSONDecodeError, TypeError):
            args = {"raw": args}
    if not isinstance(args, dict):
        args = {}

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


def make_tool_start_cb(
    conn: acp.Client,
    session_id: str,
    loop: asyncio.AbstractEventLoop,
    tool_call_ids: Dict[str, Deque[str]],
    tool_call_meta: Dict[str, Dict[str, Any]],
    edit_approval_policy_getter: Callable[[], tuple[str, str | None]] | None = None,
) -> Callable:
    """Create a canonical-ID ``tool_start_callback`` for AIAgent."""

    def _tool_start(tc_id: str, name: str, args: Any) -> None:
        _emit_tool_start(
            conn,
            session_id,
            loop,
            tool_call_ids,
            tool_call_meta,
            tc_id,
            name,
            args,
            edit_approval_policy_getter,
        )

    return _tool_start

def make_tool_progress_cb(
    conn: acp.Client,
    session_id: str,
    loop: asyncio.AbstractEventLoop,
    tool_call_ids: Dict[str, Deque[str]],
    tool_call_meta: Dict[str, Dict[str, Any]],
    edit_approval_policy_getter: Callable[[], tuple[str, str | None]] | None = None,
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
        _emit_tool_start(
            conn,
            session_id,
            loop,
            tool_call_ids,
            tool_call_meta,
            make_tool_call_id(),
            name,
            args,
            edit_approval_policy_getter,
        )

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
# Canonical tool completion callback
# ------------------------------------------------------------------

def make_tool_complete_cb(
    conn: acp.Client,
    session_id: str,
    loop: asyncio.AbstractEventLoop,
    tool_call_ids: Dict[str, Deque[str]],
    tool_call_meta: Dict[str, Dict[str, Any]],
) -> Callable:
    """Create a real-ID ``tool_complete_callback`` for AIAgent.

    The executor supplies the canonical provider tool-call ID. This avoids the
    FIFO/name correlation used by the legacy step callback and lets concurrent
    same-name tools complete out of order without crossing results.
    """

    def _tool_complete(
        tc_id: str,
        tool_name: str,
        function_args: Any,
        result: Any,
    ) -> None:
        meta = tool_call_meta.get(tc_id, {})
        args = function_args if function_args is not None else meta.get("args")
        update = build_tool_complete(
            tc_id,
            tool_name,
            result=str(result) if result is not None else None,
            function_args=args,
            snapshot=meta.get("snapshot"),
        )

        # A scheduler rejection proves the first coroutine cannot deliver, so
        # one bounded retry is safe. Once accepted, ownership belongs to the
        # loop and _send_update observes the Future without resubmitting it.
        accepted = _send_update(conn, session_id, loop, update)
        if not accepted:
            accepted = _send_update(conn, session_id, loop, update)

        if accepted and tool_name == "todo":
            plan_update = _build_plan_update_from_todo_result(result)
            if plan_update is not None:
                _send_update(conn, session_id, loop, plan_update)
        elif not accepted:
            logger.error(
                "ACP tool completion rejected by scheduler "
                "(session=%s, tool=%s, tc_id=%s)",
                session_id,
                tool_name,
                tc_id,
            )

        queue = tool_call_ids.get(tool_name)
        if isinstance(queue, str):
            if queue == tc_id:
                tool_call_ids.pop(tool_name, None)
        elif queue is not None:
            try:
                queue.remove(tc_id)
            except ValueError:
                pass
            if not queue:
                tool_call_ids.pop(tool_name, None)
        tool_call_meta.pop(tc_id, None)

    return _tool_complete


# ------------------------------------------------------------------
# Legacy step callback
# ------------------------------------------------------------------

def make_step_cb(
    conn: acp.Client,
    session_id: str,
    loop: asyncio.AbstractEventLoop,
    tool_call_ids: Dict[str, Deque[str]],
    tool_call_meta: Dict[str, Dict[str, Any]],
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
                    # Bounded retry of the SAME update (#33023). The update
                    # already carries this call's result, so retrying it cannot
                    # accidentally match a later tool call the way a queue
                    # re-pop would. One extra attempt covers transient drops
                    # (momentary timeout / busy loop); if delivery is still
                    # impossible, surface the loss at ERROR instead of letting
                    # the tool appear "running" in the ACP client forever.
                    delivered = _send_update(conn, session_id, loop, update)
                    if not delivered:
                        delivered = _send_update(conn, session_id, loop, update)
                    if delivered:
                        if tool_name == "todo":
                            plan_update = _build_plan_update_from_todo_result(result)
                            if plan_update is not None:
                                _send_update(conn, session_id, loop, plan_update)
                    else:
                        logger.error(
                            "ACP tool completion permanently undelivered "
                            "(session=%s, tool=%s, tc_id=%s); the ACP client "
                            "may show this tool as still running.",
                            session_id,
                            tool_name,
                            tc_id,
                        )
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
