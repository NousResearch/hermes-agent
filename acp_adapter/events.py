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
    read_snapshots_cache: Dict[str, str | None],
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
        if name in {"read_file", "write_file", "patch", "skill_manage", "terminal"}:
            try:
                from agent.display import capture_local_edit_snapshot

                snapshot = capture_local_edit_snapshot(name, args)
            except Exception:
                logger.debug("Failed to capture ACP edit snapshot for %s", name, exc_info=True)
        meta_entry = {"args": args, "snapshot": snapshot}
        if name == "read_file":
            from pathlib import Path
            rp = args.get("path", "")
            if rp:
                meta_entry["_read_path"] = str(Path(rp).resolve())
        tool_call_meta[tc_id] = meta_entry

        # Cross-turn snapshot cache — so terminal in later API calls can build
        # composite snapshots.  All keys are absolute resolved paths.
        # write_file / patch / skill_manage: save before-state (only if path
        #   not already cached — first writer wins, preserving the original
        #   baseline across multiple edits).
        # read_file: save current disk content (only if path not cached).
        if name in {"write_file", "patch", "skill_manage"}:
            if snapshot is not None and snapshot.before is not None:
                from pathlib import Path as _P
                for rp, before_text in snapshot.before.items():
                    rk = str(_P(rp).resolve())
                    if rk not in read_snapshots_cache:
                        read_snapshots_cache[rk] = before_text
        elif name == "read_file":
            rp = args.get("path", "")
            if rp:
                from pathlib import Path as _P
                rk = str(_P(rp).resolve())
                if rk not in read_snapshots_cache:
                    try:
                        read_snapshots_cache[rk] = _P(rk).read_text(encoding="utf-8")
                    except (FileNotFoundError, OSError):
                        read_snapshots_cache[rk] = None

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
    read_snapshots_cache: Dict[str, str | None],
) -> Callable:
    """Create a ``step_callback`` for AIAgent.

    Signature expected by AIAgent::

        step_callback(api_call_count: int, prev_tools: list)
    """

    def _step(api_call_count: int, prev_tools: Any = None) -> None:

        if prev_tools and isinstance(prev_tools, list):
            from pathlib import Path
            for tool_info in prev_tools:
                tool_name = None
                result = None
                function_args = None

                if isinstance(tool_info, dict):
                    tool_name = tool_info.get("name") or tool_info.get("function_name")
                    result = tool_info.get("result") or tool_info.get("output")
                    function_args = tool_info.get("arguments") or tool_info.get("args")
                    # OpenAI tool_call arguments arrive as a JSON string
                    # (e.g. '{"path":"/foo","offset":1}'), not a dict.
                    # build_tool_complete() → _format_read_file_result()
                    # calls .get("path") on it → AttributeError. Parse here
                    # the same way _tool_progress does on line 138.
                    if isinstance(function_args, str):
                        try:
                            function_args = json.loads(function_args)
                        except (json.JSONDecodeError, TypeError):
                            function_args = {}
                        tool_info["arguments"] = function_args
                elif isinstance(tool_info, str):
                    tool_name = tool_info

                queue = tool_call_ids.get(tool_name or "")
                if isinstance(queue, str):
                    queue = deque([queue])
                    tool_call_ids[tool_name] = queue
                if tool_name and queue:
                    tc_id = queue.popleft()

                    # Build composite snapshot from cross-turn cache for terminal.
                    # The cache persists across API calls — read_file in call #1
                    # fills it, terminal in call #4 reads it.  No ordering bug
                    # because entries are never popped from the cache.
                    if tool_name == "terminal" and read_snapshots_cache:
                        from agent.display import LocalEditSnapshot
                        known = {p: b for p, b in read_snapshots_cache.items() if b is not None}
                        if known:
                            tool_call_meta.setdefault(tc_id, {})["snapshot"] = LocalEditSnapshot(
                                paths=[Path(p) for p in known.keys()],
                                before=known,
                            )
                            logger.debug(
                                "terminal composite snapshot: %d paths from cache",
                                len(known),
                            )

                    meta = tool_call_meta.pop(tc_id, {})
                    update = build_tool_complete(
                        tc_id,
                        tool_name,
                        result=str(result) if result is not None else None,
                        function_args=function_args or meta.get("args"),
                        snapshot=meta.get("snapshot"),
                    )
                    _send_update(conn, session_id, loop, update)
                    if tool_name == "todo":
                        plan_update = _build_plan_update_from_todo_result(result)
                        if plan_update is not None:
                            _send_update(conn, session_id, loop, plan_update)
                    if not queue:
                        tool_call_ids.pop(tool_name, None)

                    # Refresh cross-turn cache from disk after tool completes.
                    # After terminal: update all cached paths so the next
                    # terminal starts from post-change state.
                    # After write_file/patch/skill_manage: update the affected
                    # path so approve/reject reset works correctly.
                    if tool_name == "terminal":
                        for p in list(read_snapshots_cache.keys()):
                            try:
                                disk = Path(p).read_text(encoding="utf-8")
                                read_snapshots_cache[p] = disk
                            except FileNotFoundError:
                                read_snapshots_cache.pop(p, None)
                            except Exception as e:
                                logger.warning(
                                    "Failed to refresh snapshot cache for %s: %s", p, e
                                )
                    elif tool_name in {"write_file", "patch", "skill_manage"}:
                        path_arg = (function_args or {}).get("path", "")
                        if path_arg:
                            rk = str(Path(path_arg).resolve())
                            try:
                                read_snapshots_cache[rk] = Path(rk).read_text(encoding="utf-8")
                            except FileNotFoundError:
                                read_snapshots_cache.pop(rk, None)
                            except Exception as e:
                                logger.warning(
                                    "Failed to refresh snapshot cache for %s after %s: %s",
                                    rk, tool_name, e,
                                )

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
