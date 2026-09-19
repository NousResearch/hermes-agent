#!/usr/bin/env python3
"""
gateway_hook.py - Hermes gateway integration for safety_redline

Drop-in module that wraps Hermes gateway calls with safety_redline checks.
When SAFETY_REDLINE_ENABLED=true, every gateway method checks pause state first
and records success/failure after.

Usage in gateway/run.py:
    from hermes.safety_redline.gateway_hook import guarded_call
    result = guarded_call("some_method", lambda: do_actual_call())
"""
import functools
import logging
from typing import Any, Callable

from . import (
    SafetyRedlineState,
    SAFETY_REDLINE_ENABLED,
    check_safety_before_api_call,
    record_api_result,
)

logger = logging.getLogger("hermes.safety_redline.gateway_hook")


def guarded_call(method_name: str, fn: Callable[[], Any]) -> Any:
    """Wrap a callable with safety_redline gates.

    Args:
        method_name: name for logging
        fn: the actual method to call

    Returns:
        The result of fn() if not paused.

    Raises:
        SafetyRedlinePausedError: if currently in pause state.
    """
    if not SAFETY_REDLINE_ENABLED:
        return fn()

    check = check_safety_before_api_call()
    if check.get("paused"):
        state = check.get("state", {})
        raise SafetyRedlinePausedError(
            f"safety_redline: {method_name} blocked (paused until "
            f"{state.get('pause_until')}, hard_paused={state.get('hard_paused')})"
        )

    try:
        result = fn()
        record_api_result(success=True)
        return result
    except Exception as e:
        record_api_result(success=False)
        raise


class SafetyRedlinePausedError(RuntimeError):
    """Raised when a call is blocked by active safety_redline pause."""


# Convenience decorator
def with_safety_redline(method_name: str | None = None):
    """Decorator: wrap a method with safety_redline gates."""

    def decorator(fn):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            name = method_name or fn.__name__
            return guarded_call(name, lambda: fn(*args, **kwargs))

        return wrapper

    return decorator
