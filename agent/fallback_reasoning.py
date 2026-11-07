"""Reasoning effort across a fallback swap and the primary restore.

A swap carries the effort the session was running (a ``/reasoning`` pin, a delegated child's
effort, the global default) to the fallback, clamped to the levels that route accepts. A
``fallback_providers`` entry's own ``reasoning_effort`` and ``agent.reasoning_overrides.<model>``
outrank the carried value. The primary's effort is recorded as the route is left so the restore
puts it back, unless the user picked a new effort while the fallback was serving.

Request parameters only: nothing here touches the prompt or the cached prefix, and retries reuse
whatever the agent holds verbatim.
"""

from __future__ import annotations

import copy
import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)

_UNSET = object()


def record_primary_reasoning(agent: Any) -> None:
    """Snapshot the live effort into ``_primary_runtime`` as the session route is left: the startup
    snapshot predates any ``/reasoning`` change made since."""
    rt = getattr(agent, "_primary_runtime", None)
    if isinstance(rt, dict):
        rt["reasoning_config"] = copy.deepcopy(getattr(agent, "reasoning_config", None))


def _agent_cfg(cfg: dict[str, Any]) -> dict[str, Any]:
    section = cfg.get("agent")
    return section if isinstance(section, dict) else {}


def _entry_effort(entry: dict[str, Any]) -> Any:
    """The entry's own ``reasoning_effort`` parsed, or ``_UNSET`` when absent or unrecognized."""
    raw = entry.get("reasoning_effort")
    if raw is None or (isinstance(raw, str) and not raw.strip()):
        return _UNSET
    from hermes_constants import parse_reasoning_effort
    parsed = parse_reasoning_effort(raw)
    if parsed is None:
        logger.warning("Ignoring unknown reasoning_effort %r on fallback %s", raw, entry.get("model"))
        return _UNSET
    return parsed


def _carried_effort(agent: Any, cfg: dict[str, Any]) -> Optional[dict[str, Any]]:
    """The primary's effort to carry over. A value that only exists because of the PRIMARY model's
    own ``reasoning_overrides`` entry (e.g. ``none`` for a local model) is not the session's choice:
    the fallback gets the global default instead (#21256)."""
    from hermes_constants import resolve_per_model_reasoning_effort, resolve_reasoning_config
    rt = getattr(agent, "_primary_runtime", None) or {}
    carried = rt["reasoning_config"] if "reasoning_config" in rt else getattr(agent, "reasoning_config", None)
    agent_cfg = _agent_cfg(cfg)
    primary_override = resolve_per_model_reasoning_effort(
        str(rt.get("model") or ""), agent_cfg.get("reasoning_overrides") or {})
    if primary_override is not None and carried == primary_override:
        return resolve_reasoning_config({"agent": {**agent_cfg, "reasoning_overrides": {}}}, agent.model)
    return copy.deepcopy(carried)


def _clamp_to_route(agent: Any, reasoning: Optional[dict[str, Any]]) -> Optional[dict[str, Any]]:
    """Clamp an effort onto the fallback route's declared levels (provider profile first, else the
    route's entry vocabulary); nearest weaker level, never an escalation."""
    from agent.reasoning_effort import clamp_reasoning_config, route_supported_efforts
    from agent.transports.codex import _profile_declared_efforts
    if not isinstance(reasoning, dict) or not reasoning.get("effort"):
        return reasoning
    supported = _profile_declared_efforts(agent.provider, agent.model, agent.base_url)
    if supported == ():
        return reasoning  # the route takes no reasoning field; the transport omits it
    return clamp_reasoning_config(reasoning, supported or route_supported_efforts(agent.provider, agent.model, agent.api_mode))


def resolve_fallback_reasoning(agent: Any, entry: dict[str, Any], cfg: dict[str, Any]) -> Optional[dict[str, Any]]:
    """Effort for the route ``agent`` was just bound to: entry ``reasoning_effort`` >
    ``agent.reasoning_overrides.<fallback model>`` > the carried primary effort, then clamped."""
    from hermes_constants import resolve_per_model_reasoning_effort
    chosen = _entry_effort(entry)
    if chosen is _UNSET:
        chosen = resolve_per_model_reasoning_effort(agent.model, _agent_cfg(cfg).get("reasoning_overrides") or {})
    if chosen is None:
        chosen = _carried_effort(agent, cfg)
    return _clamp_to_route(agent, chosen)


def rebind_fallback_reasoning(agent: Any, entry: dict[str, Any]) -> None:
    """Set the bound route's effort; a config load failure keeps the current reasoning_config
    rather than killing the swap."""
    try:
        from hermes_cli.config import load_config
        cfg = load_config() or {}
    except Exception as err:  # health: allow BLE001 -- a config read failure must not abort the fallback swap
        logger.debug("Failed to load config for fallback %s reasoning; keeping current: %s", agent.model, err)
        return
    agent.reasoning_config = resolve_fallback_reasoning(agent, entry, cfg)
    agent._fallback_bound_reasoning = copy.deepcopy(agent.reasoning_config)
    logger.info("Fallback %s: reasoning_config resolved: %s", agent.model, agent.reasoning_config)


def restored_reasoning(agent: Any, rt: dict[str, Any]) -> Any:
    """The effort the restored route runs with, or ``_UNSET`` to keep the current one: an older
    snapshot without the key, or an effort the user changed while the fallback was serving."""
    bound = vars(agent).get("_fallback_bound_reasoning", _UNSET)  # only a value a swap really recorded
    agent._fallback_bound_reasoning = _UNSET
    if "reasoning_config" not in rt:
        return _UNSET
    if bound is not _UNSET and getattr(agent, "reasoning_config", None) != bound:
        return _UNSET
    saved = rt["reasoning_config"]
    return dict(saved) if saved else None


def effort_label(reasoning: Optional[dict[str, Any]]) -> str:
    if isinstance(reasoning, dict) and reasoning.get("enabled") is False:
        return "off"
    return str((reasoning or {}).get("effort") or "default")


def effort_change_note(before: Optional[dict[str, Any]], after: Optional[dict[str, Any]]) -> str:
    """`` Reasoning effort: high → medium.`` for the fallback notice, or "" when unchanged."""
    old, new = effort_label(before), effort_label(after)
    return "" if old == new else f" Reasoning effort: {old} → {new}."
