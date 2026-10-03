"""Preserve explicit gateway reasoning overrides across runtime route changes.

Channel- and session-scoped reasoning are more specific than per-model/global
configuration. Provider fallback and same-turn model switching both call the
shared ``resolve_reasoning_config`` chokepoint, so a scoped value must survive
those re-resolutions for the rest of the turn.

This module keeps the scope marker on the reasoning-config object itself. That
avoids a second mutable flag on cached agents: when the gateway refreshes an
agent for an unscoped turn it assigns a normal ``dict`` again, automatically
restoring the existing per-model fallback behaviour.
"""

from __future__ import annotations

import inspect
from typing import Any

import hermes_constants

from .session_state import ConversationState


class ScopedReasoningConfig(dict):
    """Reasoning config that originated from a gateway session/channel scope."""


_INSTALLED = False
_ORIGINAL_PARSE_REASONING_EFFORT = hermes_constants.parse_reasoning_effort
_ORIGINAL_RESOLVE_REASONING_CONFIG = hermes_constants.resolve_reasoning_config
_ORIGINAL_CONVERSATION_SETATTR = ConversationState.__setattr__


def _parse_reasoning_effort(effort: Any) -> dict | None:
    """Mark channel-level parses while leaving global/model parses untouched."""
    result = _ORIGINAL_PARSE_REASONING_EFFORT(effort)
    if result is None:
        return None

    frame = inspect.currentframe()
    try:
        caller = frame.f_back if frame is not None else None
        if (
            caller is not None
            and caller.f_code.co_name == "_resolve_session_reasoning_config"
            and caller.f_globals.get("__name__") == "gateway.run_config_loaders"
        ):
            return ScopedReasoningConfig(result)
    finally:
        del frame

    return result


def _resolve_reasoning_config(cfg: dict | None, model: str = "") -> dict | None:
    """Keep a scoped value when runtime code re-resolves for another route.

    The two route-changing call sites (provider fallback and same-turn model
    switching) call the shared resolver with their ``agent`` argument in local
    scope. If that agent currently carries a scoped config, returning it here
    preserves the documented session/channel > per-model > global precedence.
    Unscoped agents fall straight through to the original resolver.
    """
    frame = inspect.currentframe()
    try:
        caller = frame.f_back if frame is not None else None
        agent = caller.f_locals.get("agent") if caller is not None else None
        current = getattr(agent, "reasoning_config", None)
        if isinstance(current, ScopedReasoningConfig):
            return current
    finally:
        del frame

    return _ORIGINAL_RESOLVE_REASONING_CONFIG(cfg, model)


def _conversation_state_setattr(self: ConversationState, name: str, value: Any) -> None:
    """Mark explicit ``/reasoning`` session overrides at their storage boundary."""
    if (
        name == "reasoning_override"
        and isinstance(value, dict)
        and not isinstance(value, ScopedReasoningConfig)
    ):
        value = ScopedReasoningConfig(value)
    _ORIGINAL_CONVERSATION_SETATTR(self, name, value)


def install_scoped_reasoning_runtime() -> None:
    """Install the gateway-only scope marker plumbing once per process."""
    global _INSTALLED
    if _INSTALLED:
        return

    hermes_constants.parse_reasoning_effort = _parse_reasoning_effort
    hermes_constants.resolve_reasoning_config = _resolve_reasoning_config
    ConversationState.__setattr__ = _conversation_state_setattr
    _INSTALLED = True
