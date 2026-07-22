"""Opt-in persistent role-team plugin."""

from __future__ import annotations

import json
from typing import Any, Dict


INVOKE_ROLE_SCHEMA = {
    "name": "invoke_role",
    "description": (
        "Invoke a canonical role for a durable plan. Persistent mode resumes a "
        "role-specific SessionDB session and returns an authoritative background "
        "delegation handle. delegated_subagent mode uses Hermes' real delegation "
        "rail; it is never simulated. Requires the opt-in role-team plugin/toolset."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "role": {
                "type": "string",
                "description": "Canonical role title, slug, or catalogue alias.",
            },
            "plan_id": {
                "type": "string",
                "pattern": "^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$",
                "description": "Stable safe identifier for the plan bundle.",
            },
            "summary": {
                "type": "string",
                "minLength": 1,
                "description": "Concrete assignment and expected evidence.",
            },
            "execution_mode": {
                "type": "string",
                "enum": ["persistent_role_instance", "delegated_subagent"],
                "default": "persistent_role_instance",
            },
            "workdir": {
                "type": "string",
                "description": (
                    "Optional role workdir inside the current canonical workspace. "
                    "Resolved without changing process cwd."
                ),
            },
        },
        "required": ["role", "plan_id", "summary"],
        "additionalProperties": False,
    },
}

ROLE_TEAM_STATUS_SCHEMA = {
    "name": "role_team_status",
    "description": (
        "Read a role-team plan and report authoritative async delivery state. "
        "A pending result is never reported as delivered."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "plan_id": {
                "type": "string",
                "pattern": "^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$",
            }
        },
        "required": ["plan_id"],
        "additionalProperties": False,
    },
}


def _runtime_available() -> bool:
    try:
        from hermes_state import SessionDB
        from tools.async_delegation import dispatch_async_delegation

        return hasattr(SessionDB, "mutate_session") and callable(dispatch_async_delegation)
    except Exception:
        return False


def _runtime(parent_agent: Any):
    from agent.runtime_cwd import resolve_agent_cwd

    from .runtime import RoleTeamRuntime

    return RoleTeamRuntime(
        parent_agent=parent_agent,
        workspace_root=resolve_agent_cwd(),
    )


def _invoke_handler(args: Dict[str, Any], **kwargs: Any) -> str:
    runtime = _runtime(kwargs.get("parent_agent"))
    result = runtime.invoke(
        role=str(args.get("role") or ""),
        plan_id=str(args.get("plan_id") or ""),
        summary=str(args.get("summary") or ""),
        execution_mode=str(args.get("execution_mode") or "persistent_role_instance"),
        workdir=args.get("workdir"),
    )
    return json.dumps(result, ensure_ascii=False)


def _status_handler(args: Dict[str, Any], **kwargs: Any) -> str:
    runtime = _runtime(kwargs.get("parent_agent"))
    try:
        result = runtime.status(str(args.get("plan_id") or ""))
    except Exception as exc:
        result = {"status": "error", "error": str(exc)}
    return json.dumps(result, ensure_ascii=False)


def register(ctx) -> None:
    ctx.register_tool(
        name="invoke_role",
        toolset="role_team",
        schema=INVOKE_ROLE_SCHEMA,
        handler=_invoke_handler,
        check_fn=_runtime_available,
    )
    ctx.register_tool(
        name="role_team_status",
        toolset="role_team",
        schema=ROLE_TEAM_STATUS_SCHEMA,
        handler=_status_handler,
        check_fn=_runtime_available,
    )
