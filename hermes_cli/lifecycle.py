"""Hermes lifecycle dispatch for first-party observers and plugins."""

from __future__ import annotations

import logging
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, List

logger = logging.getLogger(__name__)

_hooks_suppressed: ContextVar[bool] = ContextVar(
    "hermes_lifecycle_hooks_suppressed", default=False
)


@contextmanager
def suppress_lifecycle_hooks():
    """Task-local fail-closed boundary for internal unattended execution."""
    token = _hooks_suppressed.set(True)
    try:
        yield
    finally:
        _hooks_suppressed.reset(token)


def invoke_hook(hook_name: str, **kwargs: Any) -> List[Any]:
    """Notify first-party observers, then invoke compatibility plugin hooks."""
    if _hooks_suppressed.get():
        return []
    try:
        from hermes_cli.observability import observe_lifecycle

        observe_lifecycle(hook_name, **kwargs)
    except Exception:
        logger.warning("Built-in observability hook failed", exc_info=True)

    from hermes_cli import plugins

    return plugins.invoke_hook(hook_name, **kwargs)


def has_hook(hook_name: str) -> bool:
    """Return whether a first-party observer or plugin consumes a hook."""
    if _hooks_suppressed.get():
        return False
    try:
        from hermes_cli.observability import handles_hook

        if handles_hook(hook_name):
            return True
    except Exception:
        logger.warning("Unable to inspect built-in observability hooks", exc_info=True)

    from hermes_cli import plugins

    return plugins.has_hook(hook_name)


def finalize_session(**kwargs: Any) -> List[Any]:
    """Notify observers and hard-close one core-owned Relay conversation."""
    if _hooks_suppressed.get():
        return []
    try:
        from hermes_cli.observability import observe_lifecycle

        observe_lifecycle("on_session_finalize", **kwargs)
    except Exception:
        logger.warning("Built-in observability hook failed", exc_info=True)

    session_id = str(kwargs.get("session_id") or "")
    if session_id:
        try:
            from agent import relay_runtime

            relay_runtime.SESSION_COORDINATOR.finalize_conversation(
                profile_key=relay_runtime.current_profile_key(),
                session_id=session_id,
            )
        except Exception:
            logger.warning("Core Relay session finalization failed", exc_info=True)

    from hermes_cli import plugins

    return plugins.invoke_hook("on_session_finalize", **kwargs)
