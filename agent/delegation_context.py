"""Context-local state for delegate_task child execution.

A Hermes process may itself be a Kanban dispatcher worker with HERMES_KANBAN_* in
os.environ. In-process delegate_task children and cron jobs fired via
``cronjob(action="run")`` are NOT dispatcher-owned, so identity gates must fail
closed for them without mutating the process-global environment.
"""
from __future__ import annotations

import os
from contextlib import contextmanager
from contextvars import ContextVar, Token
from typing import Iterator, Mapping, MutableMapping

_DELEGATED_CHILD_CONTEXT: ContextVar[bool] = ContextVar("hermes_delegated_child_context", default=False)
# Any in-process execution that is NOT the dispatcher-owned worker (cron jobs). Kept separate
# so delegate_task-specific behaviour (subprocess env scrubbing, its error strings) is unchanged.
_NON_DISPATCHER_OWNED_CONTEXT: ContextVar[bool] = ContextVar("hermes_non_dispatcher_owned_context", default=False)
# Set only by the process bootstrap when a concrete worker identity was inherited
# by a new Hermes process.  It survives the environment scrub so CLI mutation
# guards can still reject the parent task without retaining board authority.
_INHERITED_KANBAN_TASK: ContextVar[str | None] = ContextVar(
    "hermes_inherited_kanban_task", default=None
)
# Explicitly supervised runtimes (currently the Hermes-tools MCP endpoint inside
# Codex) are not the worker entry PID, but are allowed to use its task identity
# after their own runtime bootstrap.
_SUPERVISED_KANBAN_RUNTIME: ContextVar[bool] = ContextVar(
    "hermes_supervised_kanban_runtime", default=False
)

DELEGATED_CHILD_ENV_MARKER = "HERMES_DELEGATED_CHILD_CONTEXT"
KANBAN_OWNER_PID_ENV = "HERMES_KANBAN_OWNER_PID"
KANBAN_OWNER_PID_PENDING = "pending"
KANBAN_CLAIM_TOKEN_ENV = "HERMES_KANBAN_CLAIM_TOKEN"
KANBAN_RUNTIME_ENV = "HERMES_KANBAN_RUNTIME"
KANBAN_CODEX_MCP_RUNTIME = "codex-mcp"

# Workspace paths are useful to an ordinary child, but they do not grant board
# mutation authority.  Every other HERMES_KANBAN_* name is treated as
# dispatcher identity so future authority keys fail closed automatically.
KANBAN_WORKSPACE_ENV_KEYS = frozenset({
    "HERMES_KANBAN_WORKSPACE", "HERMES_KANBAN_WORKSPACES_ROOT",
})

KANBAN_ENV_KEYS: tuple[str, ...] = (
    "HERMES_KANBAN_TASK", "HERMES_KANBAN_RUN_ID", "HERMES_KANBAN_WORKSPACE",
    "HERMES_KANBAN_WORKSPACES_ROOT", "HERMES_KANBAN_CLAIM_LOCK",
    "HERMES_KANBAN_BOARD", "HERMES_KANBAN_DB", KANBAN_OWNER_PID_ENV,
    KANBAN_CLAIM_TOKEN_ENV,
)


@contextmanager
def delegated_child_context(session_id: str | None = None) -> Iterator[None]:
    """Mark child execution and isolate its task-local session identity. Even a context
    entered without an id must restore the parent's session ContextVar (child
    construction calls ``set_current_session_id``)."""
    token = _DELEGATED_CHILD_CONTEXT.set(True)
    try:
        from gateway.session_context import scoped_current_session_id  # lazy: it calls is_delegated_child_context()

        with scoped_current_session_id(session_id):
            yield
    finally:
        _DELEGATED_CHILD_CONTEXT.reset(token)


def is_delegated_child_context() -> bool:
    """Return True while code is running for a delegate_task child."""
    return bool(_DELEGATED_CHILD_CONTEXT.get()) or bool(
        os.environ.get(DELEGATED_CHILD_ENV_MARKER)
    )


def enter_non_dispatcher_owned_context() -> Token[bool]:
    """Token form of :func:`non_dispatcher_owned_context` for long try/finally scopes."""
    return _NON_DISPATCHER_OWNED_CONTEXT.set(True)


def exit_non_dispatcher_owned_context(token: Token[bool]) -> None:
    """Restore the flag saved by :func:`enter_non_dispatcher_owned_context`."""
    _NON_DISPATCHER_OWNED_CONTEXT.reset(token)


@contextmanager
def non_dispatcher_owned_context() -> Iterator[None]:
    """Mark in-process execution that does NOT own the dispatcher's Kanban task; without it
    a cron agent run inside a worker is misread as that worker (kanban toolset force-added,
    ``kanban_complete`` defaulting to its task). ContextVar-scoped rather than clearing
    os.environ, which the worker's claim heartbeat and concurrent readers share."""
    token = enter_non_dispatcher_owned_context()
    try:
        yield
    finally:
        exit_non_dispatcher_owned_context(token)


def is_dispatcher_owned_worker_context() -> bool:
    """Return whether this process owns the dispatcher-provided task identity.

    A task marker without a matching owner PID is an inherited or malformed
    claim, not a worker.  Processes without a task marker remain neutral so
    explicitly configured Kanban orchestrators keep their existing behavior.
    """
    if (
        _DELEGATED_CHILD_CONTEXT.get()
        or _NON_DISPATCHER_OWNED_CONTEXT.get()
        or os.environ.get(DELEGATED_CHILD_ENV_MARKER)
    ):
        return False
    task = (os.environ.get("HERMES_KANBAN_TASK") or "").strip()
    if _INHERITED_KANBAN_TASK.get():
        return False
    if not task:
        return True
    if _SUPERVISED_KANBAN_RUNTIME.get():
        return True
    try:
        return int(os.environ.get(KANBAN_OWNER_PID_ENV, "")) == os.getpid()
    except (TypeError, ValueError):
        return False


def inherited_kanban_task_id() -> str | None:
    """Return the task identity rejected by this process bootstrap, if any.

    This is a local guard only; it is deliberately not copied into child
    environments and never grants access to the Kanban database.
    """
    return _INHERITED_KANBAN_TASK.get()


def dispatcher_owned_kanban_task_id() -> str | None:
    """Return the task id only when this execution owns its Kanban identity."""
    task = (os.environ.get("HERMES_KANBAN_TASK") or "").strip()
    if not task or not is_dispatcher_owned_worker_context():
        return None
    return task


def bind_kanban_worker_identity(
    env: MutableMapping[str, str] | None = None,
) -> bool:
    """Bind a dispatcher-spawned process to its own PID.

    ``Popen`` cannot know the child's PID while constructing its environment,
    so the dispatcher sends ``KANBAN_OWNER_PID_PENDING``.  The worker replaces
    that one-shot bootstrap value with ``os.getpid()`` before tool discovery;
    an inherited concrete PID never binds again in a descendant.
    """
    target = os.environ if env is None else env
    if not (target.get("HERMES_KANBAN_TASK") or "").strip():
        return False
    owner = (target.get(KANBAN_OWNER_PID_ENV) or "").strip()
    if owner == KANBAN_OWNER_PID_PENDING:
        target[KANBAN_OWNER_PID_ENV] = str(os.getpid())
        return True
    try:
        return int(owner) == os.getpid()
    except (TypeError, ValueError):
        return False


def strip_kanban_env(
    env: Mapping[str, str] | MutableMapping[str, str],
) -> dict[str, str]:
    """Remove Kanban authority from an ordinary child, retaining workspace paths.

    The prefix rule covers dispatcher keys added in the future without a
    second hand-maintained deny list.  ``scrub_kanban_env`` below remains the
    stronger delegate-child variant and also adds its lineage marker.
    """
    return {
        key: value
        for key, value in env.items()
        if not (key.startswith("HERMES_KANBAN_") and key not in KANBAN_WORKSPACE_ENV_KEYS)
    }


def initialize_kanban_worker_process(
    env: MutableMapping[str, str] | None = None,
) -> bool:
    """Bind the real worker or drop inherited Kanban identity from this process.

    Returns ``True`` only when the current process owns the dispatcher claim.
    Non-worker processes keep workspace convenience variables but lose task,
    run, board, lock, and future Kanban authority variables.
    """
    target = os.environ if env is None else env
    is_process_environment = env is None
    if is_process_environment:
        _INHERITED_KANBAN_TASK.set(None)
        _SUPERVISED_KANBAN_RUNTIME.set(False)
    if bind_kanban_worker_identity(target):
        return True
    task = (target.get("HERMES_KANBAN_TASK") or "").strip()
    if not task:
        # A stale owner marker without a task cannot identify a worker.
        target.pop(KANBAN_OWNER_PID_ENV, None)
        return False
    if is_process_environment:
        _INHERITED_KANBAN_TASK.set(task)
    cleaned = strip_kanban_env(target)
    target.clear()
    target.update(cleaned)
    return False


def activate_supervised_kanban_runtime(runtime: str | None = None) -> bool:
    """Authorize a named internal runtime after it receives worker context.

    The generic Hermes CLI never calls this function.  It is reserved for a
    supervised endpoint such as ``hermes-tools`` MCP, whose process is not the
    worker entry PID but is intentionally launched by that worker's runtime.
    """
    if (runtime or os.environ.get(KANBAN_RUNTIME_ENV) or "").strip() != KANBAN_CODEX_MCP_RUNTIME:
        return False
    if not (os.environ.get("HERMES_KANBAN_TASK") or "").strip():
        return False
    try:
        int(os.environ.get(KANBAN_OWNER_PID_ENV, ""))
    except (TypeError, ValueError):
        return False
    _SUPERVISED_KANBAN_RUNTIME.set(True)
    return True


def is_delegated_child_process_context() -> bool:
    """Return True in this process or a subprocess spawned by a child."""
    return bool(_DELEGATED_CHILD_CONTEXT.get()) or bool(os.environ.get(DELEGATED_CHILD_ENV_MARKER))


def scrub_kanban_env(env: Mapping[str, str] | MutableMapping[str, str]) -> dict[str, str]:
    """Remove all Kanban variables and mark the child as delegated."""
    cleaned = {k: v for k, v in env.items() if not k.startswith("HERMES_KANBAN_")}
    cleaned[DELEGATED_CHILD_ENV_MARKER] = "1"
    return cleaned


def delegated_child_subprocess_env(
    env: Mapping[str, str] | MutableMapping[str, str] | None = None,
) -> dict[str, str] | None:
    """Return a child environment with Kanban authority removed.

    Delegate children additionally receive the existing lineage marker.  A
    concrete dict is returned even for ordinary calls so a helper that used to
    rely on ``env=None`` cannot accidentally inherit a worker's authority.
    """
    source = os.environ if env is None else env
    if is_delegated_child_process_context():
        return scrub_kanban_env(source)
    return strip_kanban_env(source)
