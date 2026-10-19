"""Compatibility helper for explicit agent stop producers."""

from __future__ import annotations

import inspect
from typing import Any


def _accepts_keyword(callable_obj: Any, name: str) -> bool:
    """Return whether a callable explicitly supports a keyword argument."""
    try:
        parameters = inspect.signature(callable_obj).parameters.values()
    except (TypeError, ValueError):
        return False
    return any(
        p.kind is inspect.Parameter.VAR_KEYWORD
        or (p.name == name and p.kind is not inspect.Parameter.POSITIONAL_ONLY)
        for p in parameters
    )


def request_hard_interrupt(
    agent: Any,
    message: str | None = None,
    *,
    tool_reason: str | None = None,
) -> bool:
    """Request an explicit stop, falling back to the legacy interrupt ABI.

    New agents expose ``hard_interrupt(message=None)``; third-party agents and old test
    doubles may only expose ``interrupt(message=None)`` and must not receive keyword
    arguments they do not know. ``tool_reason`` is a trusted, fixed category that may be
    exposed in model-visible tool cancellation output, forwarded only when the callable
    explicitly supports it. Returns ``False`` only when neither callable is available.
    """
    # Static lookup first: a dynamic ``__getattr__`` proxy (unspecced MagicMock, RPC
    # facade) must not be treated as genuinely implementing the new ABI.
    try:
        inspect.getattr_static(agent, "hard_interrupt")
    except AttributeError:
        interrupt = None
    else:
        interrupt = getattr(agent, "hard_interrupt", None)
    if not callable(interrupt):
        interrupt = getattr(agent, "interrupt", None)
    if not callable(interrupt):
        return False
    kwargs = {}
    if tool_reason is not None and _accepts_keyword(interrupt, "tool_reason"):
        kwargs["tool_reason"] = tool_reason
    if message is None:
        interrupt(**kwargs)
    else:
        interrupt(message, **kwargs)
    return True
