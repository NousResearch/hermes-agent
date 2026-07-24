"""Context-local state for delegate_task child execution.

The parent Hermes process may itself be a Kanban dispatcher worker with
HERMES_KANBAN_* variables in process env. delegate_task children run inside the
same Python process, but they are not dispatcher-owned Kanban workers. This
module lets code paths that resolve tool schemas or spawn subprocesses fail
closed for delegated children without mutating global os.environ for the parent.
"""
from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from typing import Iterator, Mapping, MutableMapping

_DELEGATED_CHILD_CONTEXT: ContextVar[bool] = ContextVar(
    "hermes_delegated_child_context",
    default=False,
)

_KANBAN_WORKER_OWNER: ContextVar[bool] = ContextVar(
    "hermes_kanban_worker_owner",
    default=False,
)

DELEGATED_CHILD_ENV_MARKER = "HERMES_DELEGATED_CHILD_CONTEXT"

KANBAN_ENV_KEYS: tuple[str, ...] = (
    "HERMES_KANBAN_TASK",
    "HERMES_KANBAN_RUN_ID",
    "HERMES_KANBAN_WORKSPACE",
    "HERMES_KANBAN_WORKSPACES_ROOT",
    "HERMES_KANBAN_CLAIM_LOCK",
    "HERMES_KANBAN_BOARD",
    "HERMES_KANBAN_DB",
)


@contextmanager
def delegated_child_context() -> Iterator[None]:
    """Mark the current execution context as a delegate_task child."""
    token = _DELEGATED_CHILD_CONTEXT.set(True)
    try:
        yield
    finally:
        _DELEGATED_CHILD_CONTEXT.reset(token)


def is_delegated_child_context() -> bool:
    """Return True while code is running for a delegate_task child."""
    return bool(_DELEGATED_CHILD_CONTEXT.get())


def is_delegated_child_process_context() -> bool:
    """Return True in this process or a subprocess spawned by a child."""
    import os

    return bool(_DELEGATED_CHILD_CONTEXT.get()) or bool(
        os.environ.get(DELEGATED_CHILD_ENV_MARKER)
    )


def set_kanban_worker_owner() -> None:
    """Mark the current execution context as a verified Kanban worker owner.

    Called once at the CLI boundary after verifying that the dispatcher
    query marker matches ``HERMES_KANBAN_TASK``.  Lifecycle code consumes
    :func:`is_kanban_worker_owner` instead of re-reading ``os.environ``,
    so a nested ``hermes chat`` subprocess that inherited
    ``HERMES_KANBAN_*`` env vars is never treated as the parent owner.
    """
    _KANBAN_WORKER_OWNER.set(True)


def is_kanban_worker_owner() -> bool:
    """Return True only when this process has verified dispatcher ownership.

    Unlike ``os.environ.get("HERMES_KANBAN_TASK")``, this ContextVar
    cannot be inherited by a child subprocess.  A nested ``hermes chat``
    that inherited ``HERMES_KANBAN_*`` env vars will get ``False`` here
    because the verification step in the CLI entry point won't match the
    arbitrary query.
    """
    return bool(_KANBAN_WORKER_OWNER.get())


def scrub_kanban_env(env: Mapping[str, str] | MutableMapping[str, str]) -> dict[str, str]:
    """Return *env* with dispatcher-only Kanban variables removed."""
    cleaned = dict(env)
    for key in KANBAN_ENV_KEYS:
        cleaned.pop(key, None)
    cleaned[DELEGATED_CHILD_ENV_MARKER] = "1"
    return cleaned


def delegated_child_subprocess_env(
    env: Mapping[str, str] | MutableMapping[str, str] | None = None,
) -> dict[str, str] | None:
    """Return an env override only when delegated-child lineage must cross fork.

    Most subprocess call sites historically used ``env=None`` to inherit the
    process environment.  In a ``delegate_task`` child, inheriting as-is leaks
    parent dispatcher ``HERMES_KANBAN_*`` vars while losing the ContextVar in
    the new process.  This helper preserves normal ``env=None`` semantics for
    non-delegated calls, and only materializes a scrubbed env when the lineage
    marker must be propagated across a child-process boundary.
    """
    if not is_delegated_child_process_context():
        return None if env is None else dict(env)

    if env is None:
        import os

        env = os.environ
    return scrub_kanban_env(env)
