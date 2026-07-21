"""Safe cancellation gates for preemptive task cancellation.

These gates check the CancellationToken before performing side-effectful
operations (file writes, commits, pushes, PRs, external sends, DB migrations,
deployments). When cancelled, the gate raises OperationCancelled to prevent
the operation from starting.

Usage:
    from agent.cancellation_gates import guard_file_write, OperationCancelled

    # Inside a tool handler:
    guard_file_write(agent, path)
    # ... proceed with file write ...
"""
from __future__ import annotations

import asyncio
import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)


class OperationCancelled(Exception):
    """Raised when a side-effectful operation is cancelled by user request."""
    pass


def _get_token(agent: Any) -> Optional[Any]:
    """Extract CancellationToken from agent, or None if not available."""
    token = getattr(agent, "_cancellation_token", None)
    if token is None:
        return None
    return token if hasattr(token, "is_cancelled") else None


def _is_cancelled(agent: Any) -> bool:
    """Check if the agent's job has been cancelled."""
    token = _get_token(agent)
    if token is None:
        return False
    if token.is_cancelled:
        return True
    # Also check the legacy interrupt flag
    if getattr(agent, "_interrupt_requested", False):
        return True
    return False


def _gate(agent: Any, operation: str, **context) -> None:
    """Common gate logic. Raises OperationCancelled if cancelled."""
    if not _is_cancelled(agent):
        return
    token = _get_token(agent)
    step = token.current_step if token else None
    logger.info(
        "Cancellation gate blocked %s (step=%s, job_id=%s)",
        operation, step, getattr(agent, "_job_id", None),
    )
    raise OperationCancelled(
        f"Operation '{operation}' cancelled by user request"
        + (f" (step: {step})" if step else "")
    )


# ── Individual gates ───────────────────────────────────────────────────

def guard_file_write(agent: Any, path: str) -> None:
    """Block file writes after cancellation is requested."""
    _gate(agent, f"file_write({path})")


def guard_commit(agent: Any) -> None:
    """Block git commits after cancellation is requested."""
    _gate(agent, "git_commit")


def guard_push(agent: Any) -> None:
    """Block git pushes after cancellation is requested."""
    _gate(agent, "git_push")


def guard_pr(agent: Any) -> None:
    """Block PR creation after cancellation is requested."""
    _gate(agent, "create_pr")


def guard_external_send(agent: Any, target: str = "") -> None:
    """Block external message sends after cancellation is requested."""
    _gate(agent, f"external_send({target})" if target else "external_send")


def guard_deploy(agent: Any) -> None:
    """Block deployments after cancellation is requested.

    Deployments are irreversible side-effects that should not start
    after a cancellation request. The gate checks the cancellation state
    before the operation begins. If already in progress, the caller
    should check safe_cancellation_point() instead.
    """
    _gate(agent, "deploy")


def guard_db_migration(agent: Any, migration_name: str = "") -> None:
    """Block DB migrations after cancellation is requested.

    DB migrations require safe cancellation points: the migration must
    either not start, or if already in progress, reach a safe checkpoint
    before stopping. This gate prevents NEW migrations from starting
    after cancellation.
    """
    _gate(agent, f"db_migration({migration_name})" if migration_name else "db_migration")


def guard_deletion(agent: Any, target: str = "") -> None:
    """Block deletion operations after cancellation is requested."""
    _gate(agent, f"deletion({target})" if target else "deletion")


# ── Safe point checker ─────────────────────────────────────────────────

def is_at_safe_cancellation_point(agent: Any) -> bool:
    """Check if the agent is at a safe point for cancellation.

    A safe point means:
    - No file write in progress
    - No commit/push/PR in flight
    - No DB migration mid-execution
    - No external send in progress

    This is used by the cancellation handler to decide whether to
    do a graceful stop (if at a safe point) or wait for the current
    operation to complete.
    """
    # If the agent is executing tools, it's NOT at a safe point
    if getattr(agent, "_executing_tools", False):
        return False
    # If there's an API call in progress, it's NOT at a safe point
    if getattr(agent, "_api_call_count", 0) > 0:
        # But if we're between iterations, we are at a safe point
        # The token check in the loop will catch it
        pass
    return True


def check_cancelled_or_raise(agent: Any) -> None:
    """General-purpose cancellation check. Call at step boundaries.

    Raises OperationCancelled if cancelled, otherwise returns None.
    """
    _gate(agent, "step_boundary")


async def check_cancelled_async(agent: Any) -> None:
    """Async cancellation check for use in async code paths."""
    if _is_cancelled(agent):
        raise asyncio.CancelledError("Job cancelled by user request")
