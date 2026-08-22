"""Compatibility helper for explicit agent stop producers."""

from __future__ import annotations

import inspect
from typing import Any


def request_hard_interrupt(agent: Any, message: str | None = None, *, stop_kind: str | None = None) -> bool:
    """Request an explicit stop, falling back to the legacy interrupt ABI.

    New agents expose ``hard_interrupt(message=None)``. Third-party agents and
    old test doubles may only expose ``interrupt(message=None)``; keep those
    usable without sending the newer ``hard_cancel=`` keyword they do not know.
    ``stop_kind`` (``"user_stop"``/``"client_disconnect"``) is forwarded only
    to callables that accept it. Returns ``False`` only when neither callable
    is available.
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
    # Stamp structured provenance when the resolved callable can't carry it
    # itself (#84207).  Static lookup avoids treating an unspecced
    # MagicMock's auto-attribute as genuine.  The stamp is applied AFTER the
    # interrupt call below: AIAgent.interrupt() rewrites _interrupt_stop_kind
    # from its own parameter, so stamping before the call would be lost.
    _can_stamp = False
    if stop_kind is not None:
        try:
            inspect.getattr_static(agent, "_interrupt_stop_kind")
        except (AttributeError, TypeError):
            pass
        else:
            _can_stamp = True
    if message is not None and stop_kind is not None and not _can_stamp:
        try:
            interrupt(message, stop_kind=stop_kind)
            return True
        except TypeError:
            # Legacy ABI without stop_kind support.
            pass
    if message is None:
        interrupt()
    else:
        interrupt(message)
    if _can_stamp:
        agent._interrupt_stop_kind = stop_kind
    return True
