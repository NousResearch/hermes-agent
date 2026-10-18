"""Compatibility-aware call boundary for plugin slash-command handlers."""

from __future__ import annotations

import inspect
from typing import Any, Callable

_KEYWORD_KINDS = (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)


def invoke_plugin_command(handler: Callable, raw_args: str, **context: Any) -> Any:
    """Call *handler* once with *raw_args* plus the *context* keywords it declares.

    Every surface passes the same context (``source``, ``session_id``, ``session_key``,
    ``platform``); a handler opts in by naming any of them as keyword-capable parameters, or
    takes them all with ``**kwargs``. Legacy ``fn(raw_args)`` handlers and opaque signatures keep
    the one-argument call. Handler exceptions propagate; they are never retried as a legacy call.
    """
    try:
        parameters = list(inspect.signature(handler).parameters.values())
    except (TypeError, ValueError):
        return handler(raw_args)
    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in parameters):
        return handler(raw_args, **context)
    # The first positional parameter receives raw_args whatever its name.
    if parameters and parameters[0].kind is not inspect.Parameter.KEYWORD_ONLY:
        parameters = parameters[1:]
    accepted = {p.name for p in parameters if p.kind in _KEYWORD_KINDS}
    return handler(raw_args, **{k: v for k, v in context.items() if k in accepted})
