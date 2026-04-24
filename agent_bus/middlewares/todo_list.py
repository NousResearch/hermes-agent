"""TodoListMiddleware — maintains a live per-thread todo list in ThreadState.

Inspired by DeerFlow's TodoListMiddleware (§1.B #10). Exposes a simple
virtual tool `write_todos` that the agent can call; middleware parses the
call payload and updates the todo list in `ctx.metadata['todos']`.

The list is available to the dashboard via a future
`/api/dual-agent/threads/{id}/todos` endpoint (not in this file).

Todo item shape
---------------
    {"content": str, "status": "pending"|"in_progress"|"completed", "activeForm": str}

Env var: HERMES_MW_TODO_LIST (off / core)
"""

from __future__ import annotations

import json
import logging
from typing import Any

from agent_bus.middleware import BaseMiddleware, MiddlewareContext

logger = logging.getLogger(__name__)

VALID_STATUSES = {"pending", "in_progress", "completed"}


def _normalize_todo(item: Any) -> dict[str, Any] | None:
    if not isinstance(item, dict):
        return None
    content = item.get("content")
    if not isinstance(content, str) or not content.strip():
        return None
    status = item.get("status", "pending")
    if status not in VALID_STATUSES:
        status = "pending"
    active = item.get("activeForm") or item.get("active_form") or content
    return {
        "content": content.strip(),
        "status": status,
        "activeForm": str(active).strip(),
    }


def _parse_write_todos_args(args: Any) -> list[dict[str, Any]]:
    """Accept either a dict {todos: [...]}, a list of todos, or JSON str."""
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except json.JSONDecodeError:
            return []
    if isinstance(args, dict):
        raw = args.get("todos") or args.get("items") or []
    elif isinstance(args, list):
        raw = args
    else:
        return []
    return [t for t in (_normalize_todo(x) for x in raw) if t is not None]


class TodoListMiddleware(BaseMiddleware):
    """Intercepts `write_todos` tool calls and maintains ctx.metadata['todos']."""

    name = "todo-list"

    def after_model(self, ctx: MiddlewareContext) -> MiddlewareContext:
        """Scan latest AIMessage for a write_todos call; extract + store."""
        if not ctx.messages:
            return ctx
        # Look at most recent AIMessage only
        latest = None
        for m in reversed(ctx.messages):
            if m.get("role") == "assistant":
                latest = m
                break
        if latest is None:
            return ctx
        tool_calls = latest.get("tool_calls") or []
        for tc in tool_calls:
            if not isinstance(tc, dict):
                continue
            name = tc.get("name") or tc.get("function", {}).get("name")
            if name != "write_todos":
                continue
            args = tc.get("args") or tc.get("arguments") or tc.get("function", {}).get("arguments") or {}
            todos = _parse_write_todos_args(args)
            if not todos:
                continue
            ctx.metadata["todos"] = todos
            # Also mirror into thread-wide state via a record
            ctx.record(
                self.name, "after_model", "updated",
                f"count={len(todos)} in_progress={sum(1 for t in todos if t['status']=='in_progress')}",
            )
            break  # only process first write_todos per turn
        return ctx

    def on_session_end(self, ctx: MiddlewareContext) -> MiddlewareContext:
        todos = ctx.metadata.get("todos") or []
        if not todos:
            return ctx
        incomplete = [t for t in todos if t["status"] != "completed"]
        if incomplete:
            ctx.record(
                self.name, "on_session_end", "incomplete-todos",
                f"remaining={len(incomplete)}",
            )
        return ctx
