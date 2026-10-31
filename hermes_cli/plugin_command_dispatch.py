"""Compatibility-aware call boundary for plugin slash-command handlers."""

from __future__ import annotations

import inspect
from typing import Any, Callable

_KEYWORD_KINDS = (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)


def plugin_command_context(
    *, session_id: str | None = None, session_key: str | None = None, platform: str | None = None,
    source: Any = None, task_id: str | None = None, runtime_session_id: str | None = None,
    stored_session_id: str | None = None, surface: str | None = None,
) -> dict[str, Any]:
    """The calling-session keywords every surface offers a handler, as one snapshot.

    ``source``/``session_id``/``session_key``/``platform`` name the calling chat. The execution
    identities (``task_id``, ``runtime_session_id``, ``stored_session_id``) and the owning profile
    (``profile``, ``hermes_home``, read under the caller's profile scope) let a plugin bind
    session-owned resources without guessing from a process-global "last session".
    """
    from hermes_cli.profiles import get_active_profile_name
    from hermes_constants import get_hermes_home

    return dict(source=source, session_id=session_id or "", session_key=session_key or "",
                platform=platform or "", task_id=task_id or None,
                runtime_session_id=runtime_session_id or None,
                stored_session_id=stored_session_id or None, profile=get_active_profile_name(),
                hermes_home=get_hermes_home(), surface=surface or None)


def invoke_plugin_command(handler: Callable, raw_args: str, **context: Any) -> Any:
    """Call *handler* once with *raw_args* plus the *context* keywords it declares.

    Every surface passes the same context (``plugin_command_context``); a handler opts in by naming any of them as keyword-capable parameters, or
    takes them all with ``**kwargs``. Legacy ``fn(raw_args)`` handlers and opaque signatures keep
    the one-argument call. Handler exceptions propagate; they are never retried as a legacy call.
    """
    try:
        parameters = list(inspect.signature(handler).parameters.values())
    except (TypeError, ValueError):
        return handler(raw_args)
    # The first positional parameter receives raw_args whatever its name (``def h(source, **kw)``
    # must not get ``source`` twice).
    if parameters and parameters[0].kind is not inspect.Parameter.KEYWORD_ONLY:
        if parameters[0].kind is inspect.Parameter.POSITIONAL_OR_KEYWORD:
            context = {k: v for k, v in context.items() if k != parameters[0].name}
        parameters = parameters[1:]
    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in parameters):
        return handler(raw_args, **context)
    accepted = {p.name for p in parameters if p.kind in _KEYWORD_KINDS}
    return handler(raw_args, **{k: v for k, v in context.items() if k in accepted})
