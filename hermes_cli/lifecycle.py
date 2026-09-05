"""Hermes lifecycle dispatch for first-party observers and plugins."""

from __future__ import annotations

import logging
import threading
import time
from collections import OrderedDict
from typing import Any, List

logger = logging.getLogger(__name__)

MAX_TURN_CORRELATIONS = 2048
_TURN_CORRELATION_TTL_S = 3600.0
_TURN_CORRELATION_LOCK = threading.Lock()
_TURN_CORRELATIONS: OrderedDict[str, tuple[float, dict[str, str]]] = OrderedDict()


def _bounded_correlation_id(value: Any) -> str | None:
    """Accept opaque identifiers only; never retain arbitrary content blobs."""
    if not isinstance(value, str) or not 0 < len(value) <= 256:
        return None
    if any(ord(char) < 32 or ord(char) == 127 for char in value):
        return None
    return value


def publish_turn_correlation(
    *, turn_id: str, trace_id: str, observation_id: str | None = None, **kwargs: Any
) -> bool:
    """Publish a bounded, content-free turn correlation envelope."""
    bounded_turn_id = _bounded_correlation_id(turn_id)
    bounded_trace_id = _bounded_correlation_id(trace_id)
    bounded_observation_id = (
        _bounded_correlation_id(observation_id) if observation_id is not None else None
    )
    if (
        kwargs
        or bounded_turn_id is None
        or bounded_trace_id is None
        or (observation_id is not None and bounded_observation_id is None)
    ):
        return False
    envelope = {"turn_id": bounded_turn_id, "trace_id": bounded_trace_id}
    if bounded_observation_id:
        envelope["observation_id"] = bounded_observation_id
    now = time.monotonic()
    with _TURN_CORRELATION_LOCK:
        _TURN_CORRELATIONS[bounded_turn_id] = (now, envelope)
        _TURN_CORRELATIONS.move_to_end(bounded_turn_id)
        cutoff = now - _TURN_CORRELATION_TTL_S
        while _TURN_CORRELATIONS:
            first_key, (created_at, _) = next(iter(_TURN_CORRELATIONS.items()))
            if len(_TURN_CORRELATIONS) <= MAX_TURN_CORRELATIONS and created_at >= cutoff:
                break
            _TURN_CORRELATIONS.pop(first_key, None)
    return True


def take_turn_correlation(turn_id: str) -> dict[str, str] | None:
    """Consume one correlation envelope after the originating turn returns."""
    bounded_turn_id = _bounded_correlation_id(turn_id)
    if bounded_turn_id is None:
        return None
    with _TURN_CORRELATION_LOCK:
        item = _TURN_CORRELATIONS.pop(bounded_turn_id, None)
    if item is None or item[0] < time.monotonic() - _TURN_CORRELATION_TTL_S:
        return None
    return dict(item[1])


def clear_turn_correlations() -> None:
    """Clear transient state (primarily for process teardown and tests)."""
    with _TURN_CORRELATION_LOCK:
        _TURN_CORRELATIONS.clear()


def _observe(hook_name: str, **kwargs: Any) -> None:
    try:
        from hermes_cli.observability import observe_lifecycle

        observe_lifecycle(hook_name, **kwargs)
    except Exception:
        logger.warning("Built-in observability hook failed", exc_info=True)


def _plugin_hooks(hook_name: str, **kwargs: Any) -> List[Any]:
    from hermes_cli import plugins

    return plugins.invoke_hook(hook_name, **kwargs)


def invoke_hook(hook_name: str, **kwargs: Any) -> List[Any]:
    """Notify first-party observers, then invoke compatibility plugin hooks."""
    _observe(hook_name, **kwargs)
    return _plugin_hooks(hook_name, **kwargs)


def has_hook(hook_name: str) -> bool:
    """Return whether a first-party observer or plugin consumes a hook."""
    try:
        from hermes_cli.observability import handles_hook

        if handles_hook(hook_name):
            return True
    except Exception:
        logger.warning("Unable to inspect built-in observability hooks", exc_info=True)

    from hermes_cli import plugins

    return plugins.has_hook(hook_name)


def finalize_session(**kwargs: Any) -> List[Any]:
    """Notify observers and hard-close one core-owned Relay conversation."""
    _observe("on_session_finalize", **kwargs)

    session_id = str(kwargs.get("session_id") or "")
    if session_id:
        try:
            from agent import relay_runtime

            relay_runtime.SESSION_COORDINATOR.finalize_conversation(
                profile_key=relay_runtime.current_profile_key(),
                session_id=session_id,
            )
        except Exception:
            logger.warning("Core Relay session finalization failed", exc_info=True)

    return _plugin_hooks("on_session_finalize", **kwargs)
