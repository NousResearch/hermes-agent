"""Internal invocation contract for generation-bound post-delivery callbacks."""

from __future__ import annotations

import inspect
import logging
from typing import Any, Awaitable, Callable

logger = logging.getLogger(__name__)


def callback_accepts_delivery_result(callback: Callable) -> bool:
    """Return whether ``callback`` declares a required positional result argument.

    Optional positional parameters are treated as legacy closure captures.  This preserves
    callbacks such as ``lambda label=label: ...`` while allowing new callbacks to opt into the
    delivery result with ``def callback(result): ...``.
    """
    if getattr(callback, "_hermes_accepts_delivery_result", False):
        return True
    try:
        parameters = inspect.signature(callback).parameters.values()
    except (TypeError, ValueError):
        return False
    return any(
        parameter.kind in (parameter.POSITIONAL_ONLY, parameter.POSITIONAL_OR_KEYWORD)
        and parameter.default is parameter.empty
        for parameter in parameters
    )


def invoke_post_delivery_callback(callback: Callable, delivery_result: Any) -> Any:
    """Invoke a result-aware callback, retaining legacy zero-argument compatibility."""
    if callback_accepts_delivery_result(callback):
        return callback(delivery_result)
    return callback()


def chain_post_delivery_callbacks(*callbacks: Callable) -> Callable[[Any], Awaitable[None]]:
    """Run callbacks in registration order with per-callback failure isolation."""
    async def _chained(delivery_result: Any = None) -> None:
        for callback in callbacks:
            try:
                result = invoke_post_delivery_callback(callback, delivery_result)
                if inspect.isawaitable(result):
                    await result
            except Exception:
                logger.debug("Post-delivery callback failed", exc_info=True)

    _chained._hermes_accepts_delivery_result = True
    return _chained
