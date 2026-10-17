"""Compatibility-aware call boundary for plugin slash-command handlers."""

from __future__ import annotations

import inspect
from typing import Any, Callable


def invoke_plugin_command(handler: Callable, raw_args: str, *, source: Any = None) -> Any:
    """Call a handler once, passing gateway ``source`` only when its signature supports the keyword.

    Opaque callables retain the legacy one-argument contract. Exceptions raised by the handler
    propagate; they must not be mistaken for signature incompatibility and retried.
    """
    try:
        parameters = inspect.signature(handler).parameters.values()
    except (TypeError, ValueError):
        parameters = ()
    accepts_source = any(
        (parameter.name == "source" and parameter.kind in {
            inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY,
        }) or parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in parameters
    )
    if accepts_source:
        return handler(raw_args, source=source)
    return handler(raw_args)
