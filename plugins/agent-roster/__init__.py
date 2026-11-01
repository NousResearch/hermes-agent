"""Agent Roster plugin hooks.

Operationalizes profile_role metadata by injecting compact role context into
Kanban worker turns, enforcing forbidden tool actions in warn/block mode, and
auditing tool calls for the dashboard.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any, Optional


def _load_core():
    try:
        from . import roster_core as core  # type: ignore
        return core
    except Exception:
        path = Path(__file__).resolve().with_name("roster_core.py")
        spec = importlib.util.spec_from_file_location("agent_roster_roster_core", path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot load {path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module


def on_pre_llm_call(
    *,
    session_id: str = "",
    user_message: str = "",
    conversation_history: Optional[list] = None,
    is_first_turn: bool = False,
    model: str = "",
    platform: str = "",
    sender_id: str = "",
    **_: Any,
) -> Optional[dict[str, str]]:
    core = _load_core()
    context = core.role_context_for_current_turn()
    if not context:
        return None
    return {"context": context}


def on_pre_tool_call(
    *,
    tool_name: str = "",
    args: Optional[dict[str, Any]] = None,
    task_id: str = "",
    session_id: str = "",
    tool_call_id: str = "",
    **_: Any,
) -> Optional[dict[str, str]]:
    core = _load_core()
    return core.evaluate_tool_policy(
        tool_name,
        args if isinstance(args, dict) else {},
        session_id=session_id,
        task_id=task_id,
        tool_call_id=tool_call_id,
    )


def on_post_tool_call(
    *,
    tool_name: str = "",
    args: Optional[dict[str, Any]] = None,
    result: Any = None,
    task_id: str = "",
    session_id: str = "",
    tool_call_id: str = "",
    duration_ms: int = 0,
    **_: Any,
) -> None:
    core = _load_core()
    core.audit_tool_result(
        tool_name,
        args if isinstance(args, dict) else {},
        result,
        session_id=session_id,
        task_id=task_id,
        tool_call_id=tool_call_id,
        duration_ms=duration_ms,
    )


def register(ctx) -> None:
    ctx.register_hook("pre_llm_call", on_pre_llm_call)
    ctx.register_hook("pre_tool_call", on_pre_tool_call)
    ctx.register_hook("post_tool_call", on_post_tool_call)
