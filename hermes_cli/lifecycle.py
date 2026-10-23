"""Hermes lifecycle dispatch for first-party observers and plugins."""

from __future__ import annotations

import logging
from typing import Any, List

logger = logging.getLogger(__name__)


def _observe(hook_name: str, **kwargs: Any) -> None:
    from agent.safe_worker_policy import safe_worker_enabled

    if safe_worker_enabled():
        return
    try:
        from hermes_cli.observability import observe_lifecycle

        observe_lifecycle(hook_name, **kwargs)
    except Exception:
        logger.warning("Built-in observability hook failed", exc_info=True)


def _plugin_hooks(hook_name: str, **kwargs: Any) -> list[Any]:
    from hermes_cli import plugins

    return plugins.invoke_hook(hook_name, **kwargs)


def collect_render_blocks(section: str, **kwargs: Any) -> str:
    """Run the ``transform_render`` plugin hook for *section* and join the blocks.

    One render-extension surface for every place Hermes appends plugin text to
    something it already rendered: ``section="footer"`` (the final gateway reply)
    and ``section="usage"`` (the ``/usage`` command output) today. Callbacks get
    ``section`` plus that surface's payload, and return a string block; ``None``
    or whitespace-only return values contribute nothing, while non-empty blocks
    are joined with one blank line in registration order.

    Returns ``""`` when no plugin handles the hook, so the no-plugin case costs a
    single lookup. Fail-open: dispatching itself never raises into the caller
    (each callback is already isolated by the hook dispatcher), so a broken plugin
    cannot abort a reply or a command render.
    """
    try:
        if not has_hook("transform_render"):
            return ""
        blocks = invoke_hook("transform_render", section=section, **kwargs)
    except Exception:  # pragma: no cover — defensive, dispatch failure must not surface
        logger.debug("transform_render(%s) dispatch failed", section, exc_info=True)
        return ""
    return "\n\n".join(
        block for block in (str(raw).strip() for raw in blocks if raw is not None) if block
    )


def invoke_hook(hook_name: str, **kwargs: Any) -> list[Any]:
    """Notify first-party observers, then invoke compatibility plugin hooks."""
    from agent.safe_worker_policy import safe_worker_enabled

    if safe_worker_enabled():
        return []
    _observe(hook_name, **kwargs)
    return _plugin_hooks(hook_name, **kwargs)


async def ainvoke_hook(hook_name: str, **kwargs: Any) -> list[Any]:
    """:func:`invoke_hook` for callers on an event loop: same observers-then-plugins
    composition, with ``async def`` plugin callbacks awaited on that loop."""
    _observe(hook_name, **kwargs)
    from hermes_cli import plugins

    return await plugins.ainvoke_hook(hook_name, **kwargs)


def has_hook(hook_name: str) -> bool:
    """Return whether a first-party observer or plugin consumes a hook."""
    from agent.safe_worker_policy import safe_worker_enabled

    if safe_worker_enabled():
        return False
    try:
        from hermes_cli.observability import handles_hook

        if handles_hook(hook_name):
            return True
    except Exception:
        logger.warning("Unable to inspect built-in observability hooks", exc_info=True)

    from hermes_cli import plugins

    return plugins.has_hook(hook_name)


def session_end_messages(results: Any) -> list[str]:
    """User-facing text from ``on_session_finalize`` results: a non-empty ``str`` or ``{"message": str}``.

    Surfaces show these out-of-band (CLI print, TUI system line, gateway chat send) — never as a model
    turn. ``on_session_end`` is per-turn and stays observer-only."""
    messages: list[str] = []
    for result in results or ():
        text = result.get("message") if isinstance(result, dict) else result
        if isinstance(text, str) and text.strip():
            messages.append(text.strip())
    return messages


def finalize_session(**kwargs: Any) -> list[Any]:
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
