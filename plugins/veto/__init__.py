"""Bundled Veto plugin for Hermes tool-call policy enforcement."""

from __future__ import annotations

from typing import Any

from agent import veto_guard


def _on_pre_tool_call(
    *,
    tool_name: str = "",
    args: Any = None,
    task_id: str = "",
    session_id: str = "",
    tool_call_id: str = "",
    **_: Any,
) -> dict[str, str] | None:
    result = veto_guard.precheck_tool_call(
        tool_name,
        args if isinstance(args, dict) else {},
        task_id=task_id,
        session_id=session_id,
        tool_call_id=tool_call_id,
    )
    if result.allowed:
        return None
    return {"action": "block", "message": result.message or f"Veto blocked {tool_name}"}


def _status_command(*_: Any, **__: Any) -> str:
    return veto_guard.status_text()


def register(ctx) -> None:
    veto_guard.enable_for_process()
    ctx.register_hook("pre_tool_call", _on_pre_tool_call)
    ctx.register_command(
        "veto-status",
        handler=_status_command,
        description="Show Veto policy guard status for Hermes tool calls.",
    )
