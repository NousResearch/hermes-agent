"""Detached workflow state for context engines that explicitly opt in."""

from __future__ import annotations

import copy
import inspect
import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)


def _goal_snapshot(agent: Any) -> dict | None:
    session_id = getattr(agent, "session_id", None)
    if not session_id:
        return None
    try:
        from hermes_cli.goals import GoalState, _meta_key, load_goal

        database = getattr(agent, "_session_db", None)
        if database is not None:
            # The agent's DB is authoritative even when the process launched under another profile.
            raw = database.get_meta(_meta_key(session_id))
            state = GoalState.from_json(raw) if raw else None
        else:
            state = load_goal(session_id)
        if state is None or state.status not in {"active", "paused"}:
            return None
        return {"text": state.goal, "status": state.status,
                "contract": state.contract.to_dict(), "subgoals": state.subgoals}
    except Exception as exc:
        logger.debug("Context engine host-state goal unavailable (%s)", type(exc).__name__, exc_info=True)
        return None


def build_host_state(agent: Any) -> dict:
    """Return a fresh snapshot; plugin mutations cannot write back into the host."""
    store = getattr(agent, "_todo_store", None)
    return copy.deepcopy({"todos": store.read() if store is not None else [],
                          "goal": _goal_snapshot(agent), "plan_path": None})


def _names_host_state(parameters: dict) -> bool:
    parameter = parameters.get("host_state")
    return parameter is not None and parameter.kind in {
        inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY}


def host_state_kwargs(hook: Any, agent: Any) -> dict:
    """A generic **kwargs wrapper must keep its existing forwarding contract."""
    try:
        parameters = inspect.signature(hook).parameters
    except (TypeError, ValueError):
        return {}
    return {"host_state": build_host_state(agent)} if _names_host_state(parameters) else {}


def supported_compression_kwargs(
    compress_fn: Any, *, current_tokens: Optional[int], focus_topic: Optional[str], force: bool,
    memory_context: str, bypass_cooldown: bool = False, agent: Any = None,
) -> dict:
    """Return only compression kwargs accepted by an engine callable.
    Inspecting first keeps older plugin signatures compatible without catching ``TypeError`` and running a
    stateful compressor twice."""
    candidates = {"current_tokens": current_tokens, "focus_topic": focus_topic, "force": force}
    if bypass_cooldown:
        candidates["bypass_cooldown"] = True
    if memory_context:
        candidates["memory_context"] = memory_context
    try:
        parameters = inspect.signature(compress_fn).parameters
    except (TypeError, ValueError):
        # current_tokens has always been in the ContextEngine ABC; use the oldest call
        # shape when the callable has no inspectable signature.
        return {"current_tokens": current_tokens}
    if not any(parameter.kind is inspect.Parameter.VAR_KEYWORD for parameter in parameters.values()):
        candidates = {name: value for name, value in candidates.items() if name in parameters}
    if _names_host_state(parameters):
        candidates["host_state"] = build_host_state(agent)
    return candidates
