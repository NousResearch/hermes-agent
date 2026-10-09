"""Post-approval execution middleware for trusted synchronous tool consumers.

Unlike the older tool_execution chain, an error before execution never falls
through. Callbacks cannot rewrite approved arguments or retain a continuation.
"""

from __future__ import annotations

from contextvars import ContextVar
from contextlib import contextmanager
from copy import deepcopy
import logging
import threading
from typing import Any, Callable

from hermes_cli.middleware import AUTHORIZED_TOOL_EXECUTION_MIDDLEWARE, middleware_payload
_ACTIVE: ContextVar[bool] = ContextVar("hermes_authorized_tool_execution", default=False)
logger = logging.getLogger(__name__)


def authorized_tool_execution_active() -> bool:
    """A protected foreground command must not silently detach from its lease."""
    return _ACTIVE.get()


@contextmanager
def hold_foreground_execution():
    """Keep an admitted foreground command owned until its execution returns.

    Consumers opt in only for calls whose synchronous lifetime they protect.
    Merely registering middleware leaves normal yielding behavior unchanged.
    Explicit or automatically promoted background calls must be rejected by
    that consumer before entering this scope.
    """
    token = _ACTIVE.set(True)
    try:
        yield
    finally:
        _ACTIVE.reset(token)


def run_authorized_tool_execution_middleware(
    tool_name: str, args: dict[str, Any], next_call: Callable[[], Any], *,
    env_type: str = "", clear_interrupt: bool = False,
) -> Any:
    """Run the already-approved call once, in the owning thread.

    No callbacks means the existing tool path is untouched. A callback should
    acquire a bounded resource in a try/finally around its zero-argument
    continuation. Ordinary exceptions before execution block; cleanup errors
    after successful execution preserve the actual result instead of inviting
    replay. Background processes require their own lifetime-aware consumer.
    """
    from agent.tool_execution_context import current_tool_execution_context
    from hermes_cli.plugins import _delivery_manager
    from tools.interrupt import clear_current_thread_interrupt, is_interrupted

    callbacks = list(_delivery_manager()._middleware.get(AUTHORIZED_TOOL_EXECUTION_MIDDLEWARE, []))
    if not callbacks:
        return next_call()
    if clear_interrupt:
        clear_current_thread_interrupt()
    context = dict(current_tool_execution_context())
    approved_args = deepcopy(args)
    owner_thread = threading.get_ident()

    def call_at(index: int) -> Any:
        if is_interrupted():
            raise InterruptedError("Authorized tool execution cancelled before dispatch")
        if index == len(callbacks):
            return next_call()
        callback = callbacks[index]
        live, called, succeeded, result = True, False, False, None

        def continuation() -> Any:
            nonlocal called, succeeded, result
            if not live or threading.get_ident() != owner_thread or called:
                raise RuntimeError("Authorized tool continuation is synchronous and single-use")
            called = True
            result = call_at(index + 1)
            succeeded = True
            return result

        try:
            returned = callback(**middleware_payload(
                tool_name=tool_name, args=deepcopy(approved_args), env_type=env_type,
                next_call=continuation, **context,
            ))
            return result if succeeded else returned
        except Exception:
            if succeeded:
                # Do not log callback exception text: it may include credentials
                # or a command. The real tool result is already authoritative.
                logger.warning("authorized_tool_execution cleanup failed after completed dispatch")
                return result
            raise
        finally:
            live = False

    return call_at(0)
