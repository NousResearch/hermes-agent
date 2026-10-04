"""Hermes lifecycle dispatch for first-party observers and plugins."""

from __future__ import annotations

import logging
from typing import Any, List, Optional

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


def archive_session(db: Any, session_id: str, archived: bool, *, surface: str,
                    profile: Optional[str] = None) -> bool:
    """Deliberately (un)archive *session_id* via ``set_session_archived``; returns its result.

    Fires ``on_session_archived`` once the flag has committed, and only on a real
    unarchived -> archived transition, so repeated archive requests and un-archives stay
    silent. The idle sweep and profile adoption bypass this helper on purpose: they are
    housekeeping, not a user's "done" signal. A failing observer never fails the archive.
    """
    was_archived = None
    if archived:
        try:
            row = db.get_session(session_id)
            was_archived = bool(row.get("archived")) if row else None
        except Exception:
            logger.debug("could not read archive state before archiving %s", session_id, exc_info=True)
    changed = db.set_session_archived(session_id, archived)
    if archived and changed and was_archived is False:
        notify_session_archived(session_id, surface=surface, profile=profile)
    return changed


def notify_session_archived(session_id: str, *, surface: str, profile: Optional[str] = None) -> None:
    """Fire ``on_session_archived`` for a committed deliberate archive. Never raises."""
    try:
        if has_hook("on_session_archived"):
            invoke_hook("on_session_archived", session_id=session_id, surface=surface, profile=profile)
    except Exception:
        logger.warning("on_session_archived dispatch failed", exc_info=True)
