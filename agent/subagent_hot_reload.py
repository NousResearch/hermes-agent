"""Per-request hot reload of a running subagent's delegation route.

``delegation.hot_reload_model`` (default false) keeps the historical behavior: a child
freezes its provider:model at spawn. When true, a RUNNING child re-reads
``delegation.provider`` and ``delegation.model`` from config before every provider API
request and rebinds its live route in place, so a config change lands on the next request
without respawning the child.

Consulted from :func:`agent.turn_api_request.build_api_request` (once per API attempt).
"""

from __future__ import annotations

import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)


def hot_reload_enabled(agent: Any) -> bool:
    """True when this agent was spawned with ``delegation.hot_reload_model`` enabled.

    The flag is frozen at spawn (``tools.delegate_tool._build_child_agent``); only the route
    values are re-read per request."""
    return bool(getattr(agent, "_delegation_hot_reload_model", False))


def resolve_hot_reload_route(agent: Any) -> Optional[dict]:
    """Re-resolve the delegation credential bundle, or None when the child is pure-inherit.

    An unpinned child (no ``delegation.provider`` / ``delegation.model``) has no
    spawn-independent route to re-read, so its spawn-time binding stands."""
    from tools.delegate_tool_config import _load_config, _resolve_delegation_credentials

    cfg = _load_config()
    if not str(cfg.get("provider") or "").strip() and not str(cfg.get("model") or "").strip():
        return None
    return _resolve_delegation_credentials(cfg, agent)


def refresh_subagent_model(agent: Any) -> None:
    """Re-read the delegation pin and rebind the live route when it changed.

    No-op unless ``hot_reload_enabled`` and the configured route differs from the live
    binding. Re-resolution failure keeps the spawn-time binding and logs a warning —
    hot reload must never kill a running child."""
    if not hot_reload_enabled(agent):
        return
    try:
        target = resolve_hot_reload_route(agent)
    except Exception as exc:  # a bad pin must not abort the child's in-flight request
        logger.warning("delegation hot reload: could not resolve the configured route: %s", exc)
        return
    if target is None:
        return
    new_provider = target.get("provider") or agent.provider or ""
    new_model = target.get("model") or agent.model or ""
    if new_provider == (agent.provider or "") and new_model == (agent.model or ""):
        return  # already on the configured route — do not churn the client every request
    from agent.agent_runtime_helpers import switch_model

    switch_model(
        agent, new_model, new_provider,
        api_key=target.get("api_key") or getattr(agent, "api_key", "") or "",
        base_url=target.get("base_url") or "",
        api_mode=target.get("api_mode") or "",
    )
