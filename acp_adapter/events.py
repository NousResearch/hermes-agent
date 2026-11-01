"""Callback factories for bridging AIAgent events to ACP notifications.

Each factory returns a callable with the signature AIAgent expects for its
callbacks. AIAgent runs in a worker thread while the event loop lives on the
main thread, so updates are pushed via ``conn.session_update()`` scheduled
thread-safely onto the loop.
"""

import asyncio
import logging
import uuid
from collections import deque
from pathlib import Path
from typing import Any, Callable, Deque, Dict

import acp
from acp.schema import AgentPlanUpdate, PlanEntry

from .tools import (
    _json_loads_maybe, build_tool_abandoned, build_tool_complete, build_tool_start, coerce_tool_args,
    make_tool_call_id,
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


def _upgrade_queue(tool_call_ids: dict[str, deque[str]], name: str) -> deque[str] | None:
    """Fetch the per-tool FIFO of pending call IDs, upgrading a legacy bare-string entry in place."""
    queue = tool_call_ids.get(name)
    if isinstance(queue, str):
        queue = tool_call_ids[name] = deque([queue])
    return queue


# Tools whose edit is diffed against a before-state snapshot captured at call start.
_SNAPSHOT_WRITE_TOOLS = frozenset({"write_file", "patch", "skill_manage"})
# Tools that may have rewritten files on disk, so cached baselines must be re-read afterwards.
_MUTATING_TOOLS = frozenset({"terminal", "write_file", "patch", "skill_manage"})


def _cache_key(raw_path: Any) -> str | None:
    """Absolute resolved cache key for ``raw_path``, or ``None`` when it is useless.

    A model can emit a non-string path (``{"path": 1}``) or an unresolvable one; both used
    to raise ``TypeError``/``OSError`` inside this (swallowed) callback and silently kill
    the tool-completion path, so they are validated here instead."""
    if not isinstance(raw_path, str) or not raw_path.strip():
        return None
    try:
        return str(Path(raw_path).resolve())
    except (OSError, ValueError, TypeError):
        return None


def _remember_cached_before(cache: dict[str, str | None], snapshot: Any) -> None:
    """Store a write tool's before-state per path. First observation wins, so the baseline a
    later diff is measured from stays the earliest one we saw."""
    before = getattr(snapshot, "before", None)
    if not isinstance(before, dict):
        return
    for raw_path, text in before.items():
        key = _cache_key(raw_path)
        if key is not None and key not in cache:
            cache[key] = text


def _remember_cached_path(cache: dict[str, str | None], raw_path: Any) -> None:
    """Store what ``raw_path`` looks like right now (first observation wins)."""
    key = _cache_key(raw_path)
    if key is None or key in cache:
        return
    try:
        cache[key] = Path(key).read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        # Missing / binary / empty-on-read: an unusable baseline, but keep the path known so
        # a later diff still reports its creation.
        cache[key] = None


def _refresh_cached_paths(cache: dict[str, str | None]) -> None:
    """Re-read every cached baseline from disk after a mutation, so the next diff starts
    from post-mutation state. A path that vanished is dropped from the cache."""
    for key in list(cache):
        try:
            cache[key] = Path(key).read_text(encoding="utf-8")
        except FileNotFoundError:
            cache.pop(key, None)
        except (OSError, UnicodeDecodeError) as exc:
            logger.debug("Failed to refresh ACP snapshot cache for %s: %s", key, exc)


def _composite_snapshot(cache: dict[str, str | None] | None) -> Any:
    """Before-state snapshot spanning every path observed this session, or ``None``.

    Paths whose first observation found no file are skipped: with no baseline there is
    nothing to diff against, and a ``None`` before-text would render as a full addition."""
    if not cache:
        return None
    before: dict[str, str | None] = {path: text for path, text in cache.items() if text is not None}
    if not before:
        return None
    from agent.display import LocalEditSnapshot

    return LocalEditSnapshot(paths=[Path(p) for p in before], before=before)


def close_tool_call(
    conn: acp.Client, session_id: str, loop: asyncio.AbstractEventLoop, tool_call_ids: dict[str, deque[str]],
    tool_call_meta: dict[str, dict[str, Any]], name: str, result: Any = None, is_error: bool = False,
) -> str | None:
    """Close the oldest open ACP tool call for ``name``; returns its id, or None when none is open."""
    queue = _upgrade_queue(tool_call_ids, name)
    if not queue:
        return None
    tc_id = queue.popleft()
    meta = tool_call_meta.pop(tc_id, {})
    _send_update(conn, session_id, loop, build_tool_complete(
        tc_id, name, result=str(result) if result is not None else None,
        function_args=meta.get("args"), snapshot=meta.get("snapshot"), is_error=is_error,
    ))
    if not queue:
        tool_call_ids.pop(name, None)
    return tc_id


def flush_open_tool_calls(
    conn: acp.Client, session_id: str, loop: asyncio.AbstractEventLoop, tool_call_ids: dict[str, deque[str]],
    tool_call_meta: dict[str, dict[str, Any]],
) -> int:
    """Close every tool call still open at the end of a turn, and report how many there were.

    A tool blocked by scope, guardrail or an editor permission prompt never
    projects ``tool.completed``, so without this its bubble stays ``in_progress``
    forever and clients read the turn as one that never ran a tool."""
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


def make_tool_progress_cb(
    conn: acp.Client, session_id: str, loop: asyncio.AbstractEventLoop, tool_call_ids: dict[str, deque[str]],
    tool_call_meta: dict[str, dict[str, Any]],
    read_snapshots_cache: dict[str, str | None] | None = None,
    edit_approval_policy_getter: Callable[[], tuple[str, str | None]] | None = None,
    turn_state: dict[str, Any] | None = None,
) -> Callable:
    """Create a ``tool_progress_callback`` for AIAgent.

    Signature: ``tool_progress_callback(event_type, name, preview, args, **kwargs)``.
    Emits ``ToolCallStart`` for ``tool.started`` and tracks IDs in a FIFO per tool
    name so parallel same-name calls complete against the right ACP tool call.
    ``tool.completed`` closes that call with its own result — the step callback
    only fires on the *next* step, which leaves a turn's last tools open.

    ``read_snapshots_cache`` is session-owned and survives across API rounds, which is
    what lets a ``terminal`` diff files the model read or wrote in an earlier round."""

    def _tool_progress(event_type: str, name: str | None = None, preview: str | None = None, args: Any = None, **kwargs) -> None:
        if event_type == "tool.completed" and name:
            if turn_state is not None:
                turn_state["saw_completion"] = True
            # The executor's verdict: a cancelled/errored tool may return plain text the heuristic misses.
            close_tool_call(
                conn, session_id, loop, tool_call_ids, tool_call_meta, name, kwargs.get("result"),
                is_error=bool(kwargs.get("is_error")),
            )
            # The command may have rewritten files: re-read the cached baselines so the NEXT
            # diff is measured from post-mutation disk state, not the pre-mutation one.
            if read_snapshots_cache is not None and name in _MUTATING_TOOLS:
                _refresh_cached_paths(read_snapshots_cache)
            return
        if event_type != "tool.started":
            return
        args = coerce_tool_args(args)
        tc_id = make_tool_call_id()
        queue = _upgrade_queue(tool_call_ids, name)
        if queue is None:
            queue = tool_call_ids[name] = deque()
        queue.append(tc_id)

        snapshot = None
        if name in _SNAPSHOT_WRITE_TOOLS:
            try:
                from agent.display import capture_local_edit_snapshot

                snapshot = capture_local_edit_snapshot(name, args)
            except Exception:
                logger.debug("Failed to capture ACP edit snapshot for %s", name, exc_info=True)
        elif name == "terminal":
            # A shell command's result carries no structural diff, so seed the before-state
            # from everything observed this session. Copied (not referenced) at start, so the
            # post-command refresh cannot retroactively change the diff being rendered.
            snapshot = _composite_snapshot(read_snapshots_cache)
        tool_call_meta[tc_id] = {"args": args, "snapshot": snapshot}

        # Cross-turn cache (session-owned; keys are absolute resolved paths). First
        # observation wins, so a later diff is measured from the earliest baseline we saw
        # rather than the state after several intermediate hops.
        if read_snapshots_cache is not None:
            if name in _SNAPSHOT_WRITE_TOOLS:
                _remember_cached_before(read_snapshots_cache, snapshot)
            elif name == "read_file" and isinstance(args, dict):
                _remember_cached_path(read_snapshots_cache, args.get("path"))

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
    conn: acp.Client, session_id: str, loop: asyncio.AbstractEventLoop, tool_call_ids: dict[str, deque[str]],
    tool_call_meta: dict[str, dict[str, Any]], turn_state: dict[str, Any] | None = None,
    read_snapshots_cache: dict[str, str | None] | None = None,
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
                queue = _upgrade_queue(tool_call_ids, tool_name)
                if not queue:
                    continue
                tc_id = queue.popleft()
                meta = tool_call_meta.pop(tc_id, {})
                # ``prev_tools`` carries the wire ``arguments`` JSON *string*; the content
                # builders index it as a dict, so an uncoerced string raised inside this
                # (swallowed) callback and the bubble never closed.
                _send_update(conn, session_id, loop, build_tool_complete(
                    tc_id, tool_name, result=str(result) if result is not None else None,
                    function_args=coerce_tool_args(function_args) if function_args else meta.get("args"),
                    snapshot=meta.get("snapshot"),
                ))
                if not queue:
                    tool_call_ids.pop(tool_name, None)
                # Same post-mutation baseline refresh as the ``tool.completed`` path.
                if read_snapshots_cache is not None and tool_name in _MUTATING_TOOLS:
                    _refresh_cached_paths(read_snapshots_cache)
            if tool_name == "todo" and (plan_update := _build_plan_update_from_todo_result(result)) is not None:
                _send_update(conn, session_id, loop, plan_update)

    return _step
