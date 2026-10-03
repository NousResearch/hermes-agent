"""Hermes lifecycle dispatch for first-party observers and plugins."""

from __future__ import annotations

import logging
from typing import Any, List

logger = logging.getLogger(__name__)


def _observe(hook_name: str, **kwargs: Any) -> None:
    try:
        from hermes_cli.observability import observe_lifecycle

        observe_lifecycle(hook_name, **kwargs)
    except Exception:
        logger.warning("Built-in observability hook failed", exc_info=True)


def _plugin_hooks(hook_name: str, **kwargs: Any) -> List[Any]:
    from hermes_cli import plugins

    return plugins.invoke_hook(hook_name, **kwargs)


def invoke_hook(hook_name: str, **kwargs: Any) -> List[Any]:
    """Notify first-party observers, then invoke compatibility plugin hooks."""
    _observe(hook_name, **kwargs)
    return _plugin_hooks(hook_name, **kwargs)


async def ainvoke_hook(hook_name: str, **kwargs: Any) -> List[Any]:
    """:func:`invoke_hook` for callers on an event loop: same observers-then-plugins
    composition, with ``async def`` plugin callbacks awaited on that loop."""
    _observe(hook_name, **kwargs)
    from hermes_cli import plugins

    return await plugins.ainvoke_hook(hook_name, **kwargs)


def has_hook(hook_name: str) -> bool:
    """Return whether a first-party observer or plugin consumes a hook."""
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
    _observe("on_session_finalize", **kwargs)

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

    return _plugin_hooks("on_session_finalize", **kwargs)


def notify_session_deleted(session_id: str, **kwargs: Any) -> List[Any]:
    """Fire the ``on_session_delete`` post-commit observer (#124511).

    Call only after the delete transaction commits and only when a row was
    actually removed. Observer-only: returns ignored, failures fail open
    (logged, never raised) so a slow/broken plugin cannot break deletion.
    Unbounded/caller-thread like ``on_session_finalize`` — cleanup work
    ("close a handle, unlink a directory") must not be abandoned mid-way.
    # ponytail: single process-local fan-out; per-surface reason/platform
    # enrichment stays with callers, add a queue here only if a callback
    # measurably blocks a delete path.
    """
    try:
        return invoke_hook("on_session_delete", session_id=session_id, **kwargs)
    except Exception:
        logger.warning("on_session_delete dispatch failed", exc_info=True)
        return []
