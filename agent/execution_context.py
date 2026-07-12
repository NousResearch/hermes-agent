"""Context-local identity for agent execution authority.

Kanban workers are detached subprocesses, so ambient gateway/cron environment
markers are not a trustworthy description of who owns a tool call.  The
dispatcher instead supplies the explicit ``HERMES_KANBAN_SESSION=1`` marker.
``AIAgent`` captures that marker at construction and binds the resulting role
only while its conversation is running.

The ContextVar is also important inside a worker process: delegated child
agents share the same process environment as the card owner, but they do not
own the card's approval grant.  ``bind_agent_execution_context`` therefore
downgrades any agent with ``_delegate_depth > 0`` to ``DELEGATE``.
"""

from __future__ import annotations

import json
import os
from contextvars import ContextVar, Token
from enum import Enum
from typing import Any, Mapping, Optional


class ExecutionRole(str, Enum):
    """Authority-bearing role of the agent whose turn is currently running."""

    DIRECT = "direct"
    KANBAN_OWNER = "kanban_owner"
    KANBAN_DELEGATE = "kanban_delegate"
    DELEGATE = "delegate"


_EXECUTION_ROLE: ContextVar[ExecutionRole] = ContextVar(
    "_EXECUTION_ROLE", default=ExecutionRole.DIRECT
)


def execution_role_from_environment(
    environ: Optional[Mapping[str, str]] = None,
) -> ExecutionRole:
    """Resolve the base role from the dispatcher's explicit marker.

    Merely having ``HERMES_KANBAN_TASK`` is deliberately insufficient: that
    value identifies task scope, not approval authority.  Requiring the exact
    marker value also keeps arbitrary truthy strings from accidentally opting
    a process into the privileged card-owner path.
    """

    source = os.environ if environ is None else environ
    if source.get("HERMES_KANBAN_SESSION") == "1":
        return ExecutionRole.KANBAN_OWNER
    return ExecutionRole.DIRECT


def bind_agent_execution_context(agent: Any) -> Token:
    """Bind *agent*'s execution role for one conversation turn."""

    base_role = getattr(agent, "_execution_role", ExecutionRole.DIRECT)
    try:
        base_role = ExecutionRole(base_role)
    except (TypeError, ValueError):
        base_role = ExecutionRole.DIRECT

    if getattr(agent, "_delegate_depth", 0) > 0:
        role = (
            ExecutionRole.KANBAN_DELEGATE
            if base_role is ExecutionRole.KANBAN_OWNER
            else ExecutionRole.DELEGATE
        )
    else:
        role = base_role
    return _EXECUTION_ROLE.set(role)


def reset_agent_execution_context(token: Token) -> None:
    """Restore the execution role that preceded a conversation turn."""

    _EXECUTION_ROLE.reset(token)


def current_execution_role() -> ExecutionRole:
    """Return the role bound to the current context."""

    return _EXECUTION_ROLE.get()


def is_kanban_owner_context() -> bool:
    """Return whether this tool call belongs to the Kanban card owner.

    Approval code should use this helper instead of reading process-global
    environment variables.  ContextVars propagate into Hermes' concurrent
    tool workers, while remaining isolated from unrelated turns and delegates.
    """

    return _EXECUTION_ROLE.get() is ExecutionRole.KANBAN_OWNER


def is_kanban_delegate_context() -> bool:
    """Return whether a delegated child originated from a Kanban card owner.

    Such a child must neither consume the owner's one-use grant nor fall into
    the legacy unattended fail-open path. Approval policy uses this separate
    role to require the delegate's explicit callback or deny the action.
    """

    return _EXECUTION_ROLE.get() is ExecutionRole.KANBAN_DELEGATE


def kanban_approval_pending_metadata(result: Any) -> Optional[dict[str, Any]]:
    """Return a normalized durable-approval pause marker from a tool result.

    The marker intentionally lives at the top level of the JSON result so it
    survives normal tool-result handling and can stop the agent loop before a
    second model call.  Non-string/multimodal results and ordinary errors are
    ignored.
    """

    payload: Any = result
    if isinstance(result, str):
        try:
            payload = json.loads(result)
        except (TypeError, ValueError):
            return None
    if not isinstance(payload, dict):
        return None
    if (
        payload.get("status") != "kanban_approval_pending"
        or payload.get("kanban_approval_pending") is not True
    ):
        return None
    return dict(payload)


__all__ = [
    "ExecutionRole",
    "bind_agent_execution_context",
    "current_execution_role",
    "execution_role_from_environment",
    "is_kanban_delegate_context",
    "is_kanban_owner_context",
    "kanban_approval_pending_metadata",
    "reset_agent_execution_context",
]
