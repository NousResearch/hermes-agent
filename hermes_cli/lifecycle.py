"""Hermes lifecycle dispatch for first-party observers and plugins."""

from __future__ import annotations

import logging
from typing import Any, List

logger = logging.getLogger(__name__)


def invoke_hook(hook_name: str, **kwargs: Any) -> List[Any]:
    """Notify first-party observers, then invoke compatibility plugin hooks."""
    try:
        from hermes_cli.observability import observe_lifecycle

        observe_lifecycle(hook_name, **kwargs)
    except Exception:
        logger.warning("Built-in observability hook failed", exc_info=True)

    from hermes_cli import plugins

    return plugins.invoke_hook(hook_name, **kwargs)


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


def route_pre_user_input(
    *,
    surface: str,
    text: Any,
    session_key: str,
    platform: str,
    goal_active: bool,
    has_attachments: bool,
) -> tuple[Any, str | None]:
    """Apply the first valid ``pre_user_input_route`` rewrite, fail-open."""
    if (
        goal_active
        or has_attachments
        or not isinstance(text, str)
        or not text.strip()
        or text.lstrip().startswith("/")
    ):
        return text, None
    try:
        results = invoke_hook(
            "pre_user_input_route",
            surface=surface,
            text=text,
            session_key=session_key,
            platform=platform,
            goal_active=goal_active,
            has_attachments=has_attachments,
        )
    except Exception:
        logger.warning("pre_user_input_route invocation failed", exc_info=True)
        return text, None

    if not isinstance(results, list):
        return text, None

    for result in results:
        if not isinstance(result, dict) or result.get("action") != "rewrite":
            continue
        rewritten = result.get("text")
        if not isinstance(rewritten, str) or not rewritten.strip():
            continue
        notice = result.get("notice")
        return rewritten, notice if isinstance(notice, str) else None
    return text, None


def finalize_session(**kwargs: Any) -> List[Any]:
    """Notify observers and hard-close one core-owned Relay conversation."""
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
