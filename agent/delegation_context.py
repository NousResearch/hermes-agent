"""Context-local state for delegate_task child execution.

The parent Hermes process may itself be a Kanban dispatcher worker with
HERMES_KANBAN_* variables in process env. delegate_task children run inside the
same Python process, but they are not dispatcher-owned Kanban workers. This
module lets code paths that resolve tool schemas or spawn subprocesses fail
closed for delegated children without mutating global os.environ for the parent.

Cron jobs need the same treatment for the same reason: ``cronjob(action="run")``
executes ``run_job()`` in-process, so a cron agent fired from inside a Kanban
worker would otherwise inherit that worker's dispatcher identity.
``non_dispatcher_owned_context()`` covers both cases.

A third leak is a nested ``hermes chat`` subprocess launched from a worker
terminal (for example ``hermes chat -Q … --source tool`` for Browser Use).
Those children inherit ``os.environ`` and are a new process, so ContextVars
cannot help. ``scrub_kanban_lifecycle_ownership`` / the cmd_chat startup
drop close that path unless the process was explicitly launched as the
board worker.
"""
from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar, Token
from typing import Iterator, Mapping, MutableMapping

_DELEGATED_CHILD_CONTEXT: ContextVar[bool] = ContextVar(
    "hermes_delegated_child_context",
    default=False,
)

# Set for any in-process execution that is NOT the dispatcher-owned worker even
# though the worker's HERMES_KANBAN_* vars are legitimately in os.environ (cron
# jobs fired via the `cronjob` tool).  Kept separate from
# _DELEGATED_CHILD_CONTEXT so the delegate_task-specific behaviour attached to
# that flag (subprocess env scrubbing, its own error strings) is unchanged.
_NON_DISPATCHER_OWNED_CONTEXT: ContextVar[bool] = ContextVar(
    "hermes_non_dispatcher_owned_context",
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

# Lifecycle ownership only. Board routing pins (BOARD / DB) stay on ordinary
# terminal children so ``hermes kanban`` shell-outs remain on the same board
# (#20074). A child ``hermes chat`` must not inherit the run identity that
# makes ``kanban_complete`` default to the parent card.
KANBAN_LIFECYCLE_OWNERSHIP_KEYS: tuple[str, ...] = (
    "HERMES_KANBAN_TASK",
    "HERMES_KANBAN_RUN_ID",
    "HERMES_KANBAN_WORKSPACE",
    "HERMES_KANBAN_WORKSPACES_ROOT",
    "HERMES_KANBAN_CLAIM_LOCK",
)

_BOARD_WORKER_QUERY_PREFIX = "work kanban task "


@contextmanager
def delegated_child_context(session_id: str | None = None) -> Iterator[None]:
    """Mark child execution and isolate its task-local session identity.

    Child construction calls ``set_current_session_id`` internally, so even a
    context entered without an id must restore the parent's ContextVar.  Child
    execution passes its explicit id and receives it only for this scope.
    """
    token = _DELEGATED_CHILD_CONTEXT.set(True)
    try:
        # Import lazily: session_context calls is_delegated_child_context() when
        # deciding whether the compatibility os.environ mirror is safe.
        from gateway.session_context import scoped_current_session_id

        with scoped_current_session_id(session_id):
            yield
    finally:
        _DELEGATED_CHILD_CONTEXT.reset(token)


def is_delegated_child_context() -> bool:
    """Return True while code is running for a delegate_task child."""
    return bool(_DELEGATED_CHILD_CONTEXT.get())


@contextmanager
def non_dispatcher_owned_context() -> Iterator[None]:
    """Mark in-process execution that does NOT own the dispatcher's Kanban task.

    A Kanban worker is a normal CLI agent whose default toolset includes
    ``cronjob``; ``cronjob(action="run")`` runs ``run_job()`` inside the worker's
    own process, where ``HERMES_KANBAN_TASK`` is legitimately set.  Without this
    marker the cron agent is misread as that worker: the kanban toolset is
    force-added, the worker protocol is injected into its system prompt, and
    ``kanban_complete`` defaults ``task_id`` to ``$HERMES_KANBAN_TASK`` — letting
    an unrelated cron job close the worker's task and overwrite real results.

    Scoped via ContextVar rather than by clearing ``os.environ``: the env is
    process-global and shared with the worker's own claim heartbeat, the
    gateway's Kanban watchers, and concurrent cron jobs on the parallel pool, so
    mutating it would starve the worker's claim and race those readers.
    """
    token = _NON_DISPATCHER_OWNED_CONTEXT.set(True)
    try:
        yield
    finally:
        _NON_DISPATCHER_OWNED_CONTEXT.reset(token)


def is_dispatcher_owned_worker_context() -> bool:
    """Return True only when this execution owns the dispatcher's Kanban task.

    The single predicate every ``HERMES_KANBAN_*`` identity gate should use
    before trusting those vars.  False for delegate_task children and for cron
    jobs fired in-process from a worker.
    """
    if _DELEGATED_CHILD_CONTEXT.get():
        return False
    return not _NON_DISPATCHER_OWNED_CONTEXT.get()


def enter_non_dispatcher_owned_context() -> Token[bool]:
    """Token-based form of :func:`non_dispatcher_owned_context`.

    For callers whose scope is a long ``try`` with a matching ``finally`` rather
    than a ``with`` block (``cron.scheduler.run_job``).  Pair with
    :func:`exit_non_dispatcher_owned_context`.
    """
    return _NON_DISPATCHER_OWNED_CONTEXT.set(True)


def exit_non_dispatcher_owned_context(token: Token[bool]) -> None:
    """Restore the flag saved by :func:`enter_non_dispatcher_owned_context`."""
    _NON_DISPATCHER_OWNED_CONTEXT.reset(token)


def is_delegated_child_process_context() -> bool:
    """Return True in this process or a subprocess spawned by a child."""
    import os

    return bool(_DELEGATED_CHILD_CONTEXT.get()) or bool(
        os.environ.get(DELEGATED_CHILD_ENV_MARKER)
    )


def scrub_kanban_env(env: Mapping[str, str] | MutableMapping[str, str]) -> dict[str, str]:
    """Return *env* with dispatcher-only Kanban variables removed."""
    cleaned = dict(env)
    for key in KANBAN_ENV_KEYS:
        cleaned.pop(key, None)
    cleaned[DELEGATED_CHILD_ENV_MARKER] = "1"
    return cleaned


def scrub_kanban_lifecycle_ownership(
    env: Mapping[str, str] | MutableMapping[str, str],
) -> dict[str, str]:
    """Strip run/workspace ownership from a child-process environment.

    Unlike :func:`scrub_kanban_env`, this does **not** set the delegated-child
    marker and does **not** remove board routing pins. Use it on every
    subprocess spawn from a Kanban worker so a nested ``hermes chat`` cannot
    inherit ``HERMES_KANBAN_TASK``.
    """
    cleaned = dict(env)
    for key in KANBAN_LIFECYCLE_OWNERSHIP_KEYS:
        cleaned.pop(key, None)
    return cleaned


def is_explicit_board_worker_launch(
    *,
    query: str | None,
    source: str | None,
    task_id: str | None,
) -> bool:
    """True only for the dispatcher worker argv/env contract.

    ``_default_spawn`` launches ``hermes chat -q "work kanban task <id>"``
    with ``HERMES_SESSION_SOURCE=kanban``. Any other child chat — including
    ``hermes chat --source tool`` used by Browser Use benchmarks — is not
    the board worker, even if it inherited the parent's env.
    """
    tid = (task_id or "").strip()
    if not tid:
        return False
    if (source or "").strip() != "kanban":
        return False
    return (query or "").strip() == f"{_BOARD_WORKER_QUERY_PREFIX}{tid}"


def drop_inherited_kanban_lifecycle_if_not_board_worker(
    *,
    query: str | None = None,
    source: str | None = None,
    environ: MutableMapping[str, str] | None = None,
) -> bool:
    """Drop inherited lifecycle ownership unless this process is the worker.

    Returns True when vars were removed. Mutates *environ* (default
    ``os.environ``) so a child ``hermes chat`` that slipped past spawn-time
    scrubbing still cannot ``kanban_complete`` the parent card.
    """
    import os

    env = os.environ if environ is None else environ
    task = (env.get("HERMES_KANBAN_TASK") or "").strip()
    if not task:
        return False
    resolved_source = source if source is not None else env.get("HERMES_SESSION_SOURCE")
    if is_explicit_board_worker_launch(
        query=query, source=resolved_source, task_id=task
    ):
        return False
    for key in KANBAN_LIFECYCLE_OWNERSHIP_KEYS:
        env.pop(key, None)
    return True


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
