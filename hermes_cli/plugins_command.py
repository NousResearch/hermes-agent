"""Signature-compatible invocation of native plugin slash commands."""

from __future__ import annotations

import inspect
from typing import Any, Callable

from hermes_constants import get_hermes_home


def plugin_command_context(
    *, session_id: str | None = None, task_id: str | None = None,
    runtime_session_id: str | None = None, stored_session_id: str | None = None,
    surface: str | None = None,
) -> dict[str, Any]:
    """Snapshot dispatch-owned identifiers under the caller's profile scope."""
    from hermes_cli.profiles import get_active_profile_name

    return dict(session_id=session_id, task_id=task_id,
                runtime_session_id=runtime_session_id, stored_session_id=stored_session_id,
                profile=get_active_profile_name(), hermes_home=get_hermes_home(), surface=surface)


def invoke_plugin_command(handler: Callable, raw_args: str, **context: Any) -> Any:
    """Call once, forwarding only opted-in context; leave awaitables to the caller.

    Uninspectable extension callables retain the original ``handler(raw_args)``
    contract. Never retry a handler's TypeError: it may have already mutated state.
    """
    try:
        signature = inspect.signature(handler)
    except (TypeError, ValueError):
        return handler(raw_args)
    parameters = signature.parameters
    bound = signature.bind_partial(raw_args).arguments
    accepts_kwargs = any(p.kind == p.VAR_KEYWORD for p in parameters.values())
    kwargs = {
        key: value for key, value in context.items()
        if (key not in bound or parameters[key].kind == inspect.Parameter.POSITIONAL_ONLY)
        and (accepts_kwargs or (
            key in parameters and parameters[key].kind in (
                inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)))
    }
    return handler(raw_args, **kwargs)
