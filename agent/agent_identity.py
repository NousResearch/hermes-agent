"""Who an agent serves, built once for both plugin slots: memory providers receive it in
``initialize()``, context engines in the ``on_session_start()`` that follows agent construction."""

from __future__ import annotations

import inspect
from contextlib import suppress
from typing import Any, Callable, Optional

from agent.agent_init_params import _GATEWAY_IDENTITY_PARAMS


def agent_identity_kwargs(agent: Any, platform: Optional[str]) -> dict[str, Any]:
    """Agent context, gateway user/chat identity (set values only) and profile identity."""
    # platform="cron" (scheduler) / "subagent" (delegate_task) → providers skip writes (MemoryProvider.initialize).
    kwargs: dict[str, Any] = {"agent_context": platform if platform in ("cron", "subagent") else "primary"}
    # Gateway user/chat identity for per-user scoping (gateway_session_key: stable per-chat
    # Honcho session isolation).
    for name in _GATEWAY_IDENTITY_PARAMS:
        if value := getattr(agent, f"_{name}", None):
            kwargs[name] = value
    # Profile identity for per-profile scoping
    with suppress(Exception):
        from hermes_cli.profiles import get_active_profile_name
        kwargs["agent_identity"] = get_active_profile_name()
        kwargs["agent_workspace"] = "hermes"
    return kwargs


def context_engine_identity_kwargs(agent: Any, on_session_start: Callable[..., Any]) -> dict[str, Any]:
    """``agent_identity_kwargs`` plus ``parent_session_id`` (delegated subagents, background-review
    forks), limited to the names an older explicit ``on_session_start`` signature declares."""
    kwargs = agent_identity_kwargs(agent, getattr(agent, "platform", None))
    if parent_session_id := getattr(agent, "_parent_session_id", None):
        kwargs["parent_session_id"] = parent_session_id
    try:
        params = inspect.signature(on_session_start).parameters
    except (TypeError, ValueError):
        return {}
    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()):
        return kwargs
    return {name: value for name, value in kwargs.items() if name in params}
