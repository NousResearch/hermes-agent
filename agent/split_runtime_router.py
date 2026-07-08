"""Split-runtime routing seam for registry-backed tool calls."""

from __future__ import annotations

import json
import logging
import time
from typing import Any, Optional

from gateway.tool_channel_state import (
    cancel_tool_request,
    get_current_split_runtime,
    has_attached_client,
    submit_tool_request,
)
from tools.execution_target import ExecutionTarget, infer_execution_target
from tools.registry import registry

logger = logging.getLogger(__name__)

_NOT_ROUTED = object()


def _tool_error(message: str, *, code: str, tool_call_id: Optional[str] = None) -> str:
    payload = {"error": message, "code": code}
    if tool_call_id:
        payload["tool_call_id"] = tool_call_id
    return json.dumps(payload, ensure_ascii=False)


def should_route_tool_locally(function_name: str) -> bool:
    """Return True when the active split-runtime context owns this tool.

    Callers use this to distinguish normal server execution from fail-closed
    split-runtime failures. If this returns True, the server must not silently
    fall back to registry.dispatch after a routing error.
    """

    cfg = get_current_split_runtime()
    if not cfg or not cfg.get("enabled"):
        return False
    entry = registry.get_entry(function_name)
    return infer_execution_target(
        entry,
        enabled=True,
        routed_toolsets=cfg.get("routed_toolsets"),
    ) is ExecutionTarget.LOCAL


def route_tool_locally(
    function_name: str,
    function_args: dict[str, Any],
    tool_call_id: Optional[str],
    *,
    task_id: str = "",
    session_id: str = "",
    turn_id: str = "",
    api_request_id: str = "",
) -> Any:
    """Return a local tool result string, or ``_NOT_ROUTED``.

    This is called inside the existing tool-execution middleware, after request
    mutation and pre-tool blocks but before ``registry.dispatch``. It only acts
    when API-server split runtime bound an enabled config in this context.
    """

    cfg = get_current_split_runtime()
    if not cfg or not cfg.get("enabled"):
        return _NOT_ROUTED

    if not should_route_tool_locally(function_name):
        return _NOT_ROUTED

    try:
        from tools.approval import get_current_session_key

        session_key = get_current_session_key(default="")
    except Exception:
        session_key = ""

    call_id = str(tool_call_id or "").strip()
    if not call_id:
        return _tool_error(
            "Split-runtime local execution requires a tool_call_id.",
            code="split_runtime_missing_tool_call_id",
        )

    if not session_key or not has_attached_client(session_key):
        return _tool_error(
            "Split-runtime local execution requested, but no local executor is attached to this run.",
            code="split_runtime_no_executor",
            tool_call_id=call_id,
        )

    request = {
        "v": 1,
        "tool_call_id": call_id,
        "tool_name": function_name,
        "arguments": function_args if isinstance(function_args, dict) else {},
        "session_id": session_id or "",
        "task_id": task_id or "",
        "turn_id": turn_id or "",
        "api_request_id": api_request_id or "",
        "created_at": time.time(),
    }
    pending = submit_tool_request(session_key, request)
    if pending is None:
        return _tool_error(
            "Split-runtime local executor is not available for this run.",
            code="split_runtime_executor_unavailable",
            tool_call_id=call_id,
        )

    try:
        timeout = float(cfg.get("request_timeout_seconds", 300.0))
    except (TypeError, ValueError):
        timeout = 300.0
    timeout = max(0.1, timeout)

    if not pending.event.wait(timeout=timeout):
        cancel_tool_request(session_key, call_id)
        return _tool_error(
            f"Split-runtime local executor did not return a result within {timeout:g}s.",
            code="split_runtime_tool_timeout",
            tool_call_id=call_id,
        )

    if pending.result is None:
        return _tool_error(
            "Split-runtime local executor disconnected before returning a result.",
            code="split_runtime_executor_disconnected",
            tool_call_id=call_id,
        )

    return pending.result
