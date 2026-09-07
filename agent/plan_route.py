"""Turn-scoped routing for the built-in ``/plan`` prompt."""

from __future__ import annotations

from typing import Any

from agent.plan_prompt import PLAN_PROMPT_MARKER


def planning_route_for_message(message: Any, config: dict | None) -> dict | None:
    """Return the configured planning route only for a built-in /plan turn.

    The marker is emitted by :func:`agent.plan_prompt.build_plan_prompt`; keeping the
    decision at the turn boundary means the route is never persisted as a session override.
    """
    if not isinstance(message, str):
        return None
    routed_message = message
    if routed_message.startswith("[") and "] " in routed_message:
        routed_message = routed_message.split("] ", 1)[1]
    if not routed_message.startswith(PLAN_PROMPT_MARKER):
        return None
    planning = config.get("planning") if isinstance(config, dict) else None
    if not isinstance(planning, dict):
        return None
    model = planning.get("model")
    provider = planning.get("provider")
    if not str(model or "").strip() and not str(provider or "").strip():
        return None
    return {
        "model": str(model or "").strip(),
        "provider": str(provider or "").strip(),
        "reasoning_effort": planning.get("reasoning_effort", ""),
    }
