"""A normalized reasoning choice scoped to one gateway conversation turn."""
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Literal

from agent.reasoning_effort import clamp_effort, route_supported_efforts

TURN_REASONING_HOOK = "select_turn_reasoning"
TurnReasoningEffort = Literal["none", "minimal", "low", "medium", "high", "xhigh", "max", "ultra"]
_turn_effort: ContextVar[tuple[int, str | None] | None] = ContextVar("turn_reasoning_effort", default=None)


def _config_for_route(effort, provider, model, api_mode):
    fitted = clamp_effort(effort, route_supported_efforts(provider, model, api_mode))
    return {"enabled": False} if fitted == "none" else {"enabled": True, "effort": fitted}


def _active_config(agent):
    selection = _turn_effort.get()
    if selection is None or selection[0] != id(agent) or selection[1] is None:
        return None
    return _config_for_route(selection[1], agent.provider, agent.model, agent.api_mode)


@contextmanager
def _reasoning_scope(agent, effort, baseline):
    token = _turn_effort.set((id(agent), effort))
    try:
        yield
    finally:
        _turn_effort.reset(token)
        if effort is not None:
            agent.reasoning_config = baseline
            snapshot = getattr(agent, "_primary_runtime", None)
            if snapshot is not None and "reasoning_config" in snapshot:
                snapshot["reasoning_config"] = dict(baseline) if baseline is not None else None
