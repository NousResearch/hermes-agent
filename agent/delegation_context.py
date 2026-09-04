"""Context-local provenance for delegated and dispatcher-owned execution.

A Hermes process may itself be a Kanban dispatcher worker with
``HERMES_KANBAN_*`` values in ``os.environ``.  Those process-global values are
not sufficient authority for every in-process execution: cron jobs and
unrelated delegated children can run in the same interpreter.  This module
keeps lineage and authority separate with ContextVars, and scrubs dispatcher
identity when delegated lineage crosses a subprocess boundary.
"""
from __future__ import annotations

import hashlib
import hmac
import os
from contextlib import contextmanager
from contextvars import ContextVar, Token
from typing import Iterator, Mapping, MutableMapping

__all__ = [
    "DELEGATED_CHILD_ENV_MARKER",
    "DISPATCHER_OWNERSHIP_BOOTSTRAP_ENV",
    "DispatcherAuthorityError",
    "bootstrap_dispatcher_authority",
    "delegated_child_context",
    "delegated_child_inherits_authority",
    "delegated_child_subprocess_env",
    "dispatcher_ownership_marker",
    "enter_non_dispatcher_owned_context",
    "exit_dispatcher_authority",
    "exit_non_dispatcher_owned_context",
    "has_dispatcher_owned_authority",
    "is_delegated_child_context",
    "is_delegated_child_process_context",
    "is_dispatcher_owned_worker_context",
    "non_dispatcher_authority_veto",
    "non_dispatcher_owned_context",
    "scrub_kanban_env",
]

_DELEGATED_CHILD_CONTEXT: ContextVar[bool] = ContextVar(
    "hermes_delegated_child_context", default=False
)
_NON_DISPATCHER_OWNED_CONTEXT: ContextVar[bool] = ContextVar(
    "hermes_non_dispatcher_owned_context", default=False
)
_DISPATCHER_AUTHORITY: ContextVar[bool] = ContextVar(
    "hermes_dispatcher_owned_authority", default=False
)
_NON_DISPATCHER_VETO: ContextVar[bool] = ContextVar(
    "hermes_non_dispatcher_veto", default=False
)

DELEGATED_CHILD_ENV_MARKER = "HERMES_DELEGATED_CHILD_CONTEXT"
DISPATCHER_OWNERSHIP_BOOTSTRAP_ENV = "HERMES_KANBAN_WORKER_OWNERSHIP"

KANBAN_ENV_KEYS: tuple[str, ...] = (
    "HERMES_KANBAN_TASK",
    "HERMES_KANBAN_RUN_ID",
    "HERMES_KANBAN_WORKSPACE",
    "HERMES_KANBAN_TERMINAL_RUNTIME",
    "HERMES_KANBAN_WORKSPACES_ROOT",
    "HERMES_KANBAN_CLAIM_LOCK",
    "HERMES_KANBAN_BOARD",
    "HERMES_KANBAN_DB",
    DISPATCHER_OWNERSHIP_BOOTSTRAP_ENV,
)


class DispatcherAuthorityError(RuntimeError):
    """Dispatcher authority is absent, stale, mismatched, or unprovable."""


def _dispatcher_ownership_proof(task_id: str) -> tuple[str, str]:
    """Return the deterministic one-shot marker components for *task_id*.

    This marker fences accidental authority inheritance across the worker
    process boundary; it is not intended to defend against a hostile local
    process that can freely forge this user's environment.  The marker is
    consumed during CLI bootstrap, while the resulting authority lives only in
    process-local ContextVar state.
    """
    raw = str(task_id or "").strip()
    if not raw:
        raise DispatcherAuthorityError("dispatcher ownership proof needs a task id")
    digest = hashlib.sha256(
        f"hermes-kanban-worker-ownership:{raw}".encode("utf-8")
    ).hexdigest()
    return digest[:32], digest[32:64]


def dispatcher_ownership_marker(task_id: str) -> str:
    """Return the one-shot worker bootstrap marker for *task_id*."""
    proof, nonce = _dispatcher_ownership_proof(task_id)
    return f"{proof}.{nonce}"


@contextmanager
def delegated_child_context(session_id: str | None = None) -> Iterator[None]:
    """Mark delegated execution and isolate its task-local session identity.

    ContextVars naturally preserve an already-established dispatcher authority
    value.  The child may therefore use the parent's task runtime, but it still
    remains delegated lineage for Kanban-mutation/toolset gates.
    """
    token = _DELEGATED_CHILD_CONTEXT.set(True)
    try:
        from gateway.session_context import scoped_current_session_id

        with scoped_current_session_id(session_id):
            yield
    finally:
        _DELEGATED_CHILD_CONTEXT.reset(token)


def is_delegated_child_context() -> bool:
    return bool(_DELEGATED_CHILD_CONTEXT.get())


def has_dispatcher_owned_authority() -> bool:
    """Whether the current execution may consume the worker runtime.

    Positive authority is established only by a successful one-shot bootstrap.
    A cron/non-dispatcher scope or explicit veto dominates at every delegation
    depth.
    """
    if _NON_DISPATCHER_OWNED_CONTEXT.get() or _NON_DISPATCHER_VETO.get():
        return False
    return bool(_DISPATCHER_AUTHORITY.get())


def is_dispatcher_owned_worker_context() -> bool:
    """Whether this is the root worker, not merely an authorized delegate."""
    return bool(
        has_dispatcher_owned_authority()
        and not _DELEGATED_CHILD_CONTEXT.get()
    )


def delegated_child_inherits_authority() -> None:
    """Assert that delegated execution inherited positive parent authority."""
    if not has_dispatcher_owned_authority():
        raise DispatcherAuthorityError(
            "delegate cannot manufacture dispatcher authority: parent has none"
        )


def bootstrap_dispatcher_authority(
    *,
    task_id: str,
    workspace: str | None = None,
    environ: MutableMapping[str, str] | None = None,
) -> Token[bool]:
    """Consume the one-shot worker marker and establish process-local authority."""
    env = environ if environ is not None else os.environ
    marker_raw = str(env.get(DISPATCHER_OWNERSHIP_BOOTSTRAP_ENV, "")).strip()
    kanban_task = str(env.get("HERMES_KANBAN_TASK", "")).strip()
    source = str(env.get("HERMES_SESSION_SOURCE", "")).strip().lower()

    def deny(reason: str) -> None:
        env.pop(DISPATCHER_OWNERSHIP_BOOTSTRAP_ENV, None)
        _DISPATCHER_AUTHORITY.set(False)
        raise DispatcherAuthorityError(
            f"dispatcher ownership bootstrap denied: {reason}"
        )

    if _NON_DISPATCHER_OWNED_CONTEXT.get() or _NON_DISPATCHER_VETO.get():
        deny("non-dispatcher scope dominates bootstrap")
    if not marker_raw:
        deny("no unconsumed dispatcher ownership marker present")
    if source != "kanban" or not kanban_task:
        deny("runtime identity missing HERMES_SESSION_SOURCE=kanban or task id")
    expected_task = str(task_id or "").strip()
    if kanban_task != expected_task:
        deny(f"task mismatch: env={kanban_task!r} expected={expected_task!r}")
    if workspace is not None:
        env_workspace = str(env.get("HERMES_KANBAN_WORKSPACE", "")).strip()
        if env_workspace != str(workspace).strip():
            deny(
                f"workspace mismatch: env={env_workspace!r} "
                f"expected={str(workspace).strip()!r}"
            )
    try:
        expected_marker = dispatcher_ownership_marker(expected_task)
    except Exception as exc:
        deny(f"ownership proof computation failed: {exc}")
    if not hmac.compare_digest(marker_raw, expected_marker):
        deny("ownership marker does not match this task runtime")

    env.pop(DISPATCHER_OWNERSHIP_BOOTSTRAP_ENV, None)
    return _DISPATCHER_AUTHORITY.set(True)


def exit_dispatcher_authority(token: Token[bool]) -> None:
    _DISPATCHER_AUTHORITY.reset(token)


def enter_non_dispatcher_owned_context() -> Token[bool]:
    """Enter a dominating non-dispatcher scope; pair with the exit helper."""
    return _NON_DISPATCHER_OWNED_CONTEXT.set(True)


def exit_non_dispatcher_owned_context(token: Token[bool]) -> None:
    _NON_DISPATCHER_OWNED_CONTEXT.reset(token)


@contextmanager
def non_dispatcher_owned_context() -> Iterator[None]:
    token = enter_non_dispatcher_owned_context()
    try:
        yield
    finally:
        exit_non_dispatcher_owned_context(token)


@contextmanager
def non_dispatcher_authority_veto() -> Iterator[None]:
    """Explicit dominating veto for nested non-dispatcher execution."""
    token = _NON_DISPATCHER_VETO.set(True)
    try:
        yield
    finally:
        _NON_DISPATCHER_VETO.reset(token)


def is_delegated_child_process_context() -> bool:
    return bool(_DELEGATED_CHILD_CONTEXT.get()) or bool(
        os.environ.get(DELEGATED_CHILD_ENV_MARKER)
    )


def scrub_kanban_env(
    env: Mapping[str, str] | MutableMapping[str, str],
) -> dict[str, str]:
    """Remove dispatcher-only identity and propagate delegated lineage."""
    cleaned = {key: value for key, value in env.items() if key not in KANBAN_ENV_KEYS}
    cleaned[DELEGATED_CHILD_ENV_MARKER] = "1"
    return cleaned


def delegated_child_subprocess_env(
    env: Mapping[str, str] | MutableMapping[str, str] | None = None,
) -> dict[str, str] | None:
    """Return a scrubbed env only when delegated lineage crosses a process."""
    if not is_delegated_child_process_context():
        return None if env is None else dict(env)
    return scrub_kanban_env(os.environ if env is None else env)
