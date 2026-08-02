"""Compatibility helper for explicit agent stop producers."""

from __future__ import annotations

import inspect
from typing import Any


def request_hard_interrupt(agent: Any, message: str | None = None) -> bool:
    """Request an explicit stop, falling back to the legacy interrupt ABI.

    New agents expose ``hard_interrupt(message=None)``. Third-party agents and
    old test doubles may only expose ``interrupt(message=None)``; keep those
    usable without sending the newer ``hard_cancel=`` keyword they do not know.
    Returns ``False`` only when neither callable is available.
    """
    # Avoid treating a dynamic ``__getattr__`` proxy (notably an unspecced
    # ``MagicMock`` or a third-party RPC facade) as if it genuinely implements
    # the new ABI. Static lookup proves the attribute exists on the instance or
    # its type before normal descriptor binding retrieves the callable.
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
    if message is None:
        interrupt()
    else:
        interrupt(message)
    return True
