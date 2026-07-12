"""Context-local identity for agent execution authority.

Kanban workers are detached subprocesses, so ambient gateway/cron environment
markers are not a trustworthy description of who owns a tool call. The
dispatcher instead supplies a one-use random bootstrap nonce bound to the
exact active task run. ``AIAgent`` atomically consumes that nonce at
construction and binds the resulting role only while its conversation runs.

The ContextVar is also important inside a worker process: delegated child
agents share the same process environment as the card owner, but they do not
own the card's approval grant.  ``bind_agent_execution_context`` therefore
downgrades any agent with ``_delegate_depth > 0`` to ``DELEGATE``.
"""

from __future__ import annotations

import json
import os
import threading
from collections.abc import MutableMapping
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
_KANBAN_OWNER_MARKER = "HERMES_KANBAN_SESSION"
_KANBAN_OWNER_NONCE = "HERMES_KANBAN_OWNER_BOOTSTRAP_NONCE"
_KANBAN_DELEGATE_MARKER = "HERMES_KANBAN_DELEGATE_SESSION"
_KANBAN_OWNER_CLAIM_LOCK = threading.Lock()


def execution_role_from_environment(
    environ: Optional[Mapping[str, str]] = None,
) -> ExecutionRole:
    """Classify non-owner environment identity without granting authority.

    Owner authority is never derivable from a mapping alone; it requires the
    one-use database consumption performed by :func:`execution_role_for_new_agent`.
    Any apparent owner bootstrap is therefore classified as a delegate here.
    """

    source = os.environ if environ is None else environ
    if source.get(_KANBAN_DELEGATE_MARKER) == "1":
        return ExecutionRole.KANBAN_DELEGATE
    if _KANBAN_OWNER_MARKER in source or _KANBAN_OWNER_NONCE in source:
        # Environment classification alone can never authenticate an owner.
        # Only execution_role_for_new_agent's atomic DB nonce consumption may
        # return KANBAN_OWNER.
        return ExecutionRole.KANBAN_DELEGATE
    return ExecutionRole.DIRECT


def _consume_kanban_owner_bootstrap(source: Mapping[str, str]) -> bool:
    """Consume the dispatcher's exact active-run nonce, failing closed."""
    task_id = str(source.get("HERMES_KANBAN_TASK") or "").strip()
    run_raw = str(source.get("HERMES_KANBAN_RUN_ID") or "").strip()
    profile = str(source.get("HERMES_PROFILE") or "").strip()
    claim_lock = str(source.get("HERMES_KANBAN_CLAIM_LOCK") or "").strip()
    nonce = str(source.get(_KANBAN_OWNER_NONCE) or "").strip()
    if not all((task_id, run_raw, profile, claim_lock, nonce)):
        return False
    try:
        run_id = int(run_raw)
    except ValueError:
        return False
    try:
        from pathlib import Path

        from hermes_cli import kanban_db as kb

        db_path = str(source.get("HERMES_KANBAN_DB") or "").strip()
        conn = kb.connect(Path(db_path).expanduser() if db_path else None)
        try:
            return kb.consume_task_owner_bootstrap(
                conn,
                task_id=task_id,
                run_id=run_id,
                profile=profile,
                claim_lock=claim_lock,
                nonce=nonce,
            )
        finally:
            conn.close()
    except Exception:
        return False


def execution_role_for_new_agent(
    environ: Optional[Mapping[str, str]] = None,
) -> ExecutionRole:
    """Resolve authority for a newly-constructed agent.

    The dispatcher-owned top-level agent is created while the context is
    direct, so it may claim the environment's card-owner capability. Any
    agent constructed while that owner (or one of its descendants) is already
    running is an auxiliary fork and must remain a Kanban delegate even when
    it does not use the public ``delegate_task`` depth counter.
    """

    source = os.environ if environ is None else environ
    with _KANBAN_OWNER_CLAIM_LOCK:
        snapshot = dict(source)
        owner_attempt = (
            _KANBAN_OWNER_MARKER in snapshot
            or _KANBAN_OWNER_NONCE in snapshot
        )
        # The owner bootstrap is process-entry-only. Scrub it before any DB
        # work and replace it with a delegate marker inherited by auxiliary
        # agents and subprocesses. Invalid attempts are deliberately delegates,
        # never DIRECT (which could reach unattended fail-open policy).
        if isinstance(source, MutableMapping):
            source.pop(_KANBAN_OWNER_MARKER, None)
            source.pop(_KANBAN_OWNER_NONCE, None)
            if owner_attempt:
                source[_KANBAN_DELEGATE_MARKER] = "1"

        if not owner_attempt:
            return execution_role_from_environment(snapshot)
        if snapshot.get(_KANBAN_OWNER_MARKER) != "1":
            return ExecutionRole.KANBAN_DELEGATE
        if _EXECUTION_ROLE.get() is not ExecutionRole.DIRECT:
            return ExecutionRole.KANBAN_DELEGATE
        if _consume_kanban_owner_bootstrap(snapshot):
            return ExecutionRole.KANBAN_OWNER
        return ExecutionRole.KANBAN_DELEGATE


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
            if base_role in {
                ExecutionRole.KANBAN_OWNER,
                ExecutionRole.KANBAN_DELEGATE,
            }
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
    "execution_role_for_new_agent",
    "is_kanban_delegate_context",
    "is_kanban_owner_context",
    "kanban_approval_pending_metadata",
    "reset_agent_execution_context",
]
