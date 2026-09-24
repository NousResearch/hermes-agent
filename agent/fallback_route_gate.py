"""Deterministic route gate for side-effecting tools during automatic provider fallback.

When the primary provider fails mid-turn, ``try_activate_fallback`` swaps the
model/provider in place and the tool loop continues on the new backend without any
route check (issue #117495). Tools with irreversible external effects (git push,
service restarts, DB writes, sent messages) can execute on a route the caller never
selected — and no post-hoc result invalidation can undo them.

``halt_on_side_effecting_tools`` (config key under the ``fallback:`` block, default
``False``) closes this:
while an automatic provider fallback is the acting route AND that key is set, tools
that may have side effects are refused *before* dispatch; read-only tools keep
working. The fallback model can keep reading, summarizing, and answering; it just
cannot mutate external state until the primary recovers or you switch deliberately
via ``/model``.

The gate is armed by ``_provider_fallback_active``, which is set ONLY by the automatic
fallback activation path and cleared on deliberate switches (``/model``) and on
primary restore — so a route the user explicitly selected is never restricted.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

# clarify is a prompt for the USER, not an external write; blocking it while a
# fallback route is acting would make the fallback useless for its legitimate work.
_GATE_ALLOWED_TOOLS = frozenset({"clarify"})

_BLOCK_MESSAGE = (
    "Blocked: tool '{tool}' can have external side effects and the current route is an "
    "automatic provider fallback ({model} via {provider}), not the route you selected. "
    "Set fallback.halt_on_side_effecting_tools: false in config.yaml to disable this gate, "
    "switch deliberately with /model to a trusted route, or retry after the primary recovers."
)


def fallback_route_block_reason(agent: Any, tool_name: str, provider: Any, model: Any) -> str | None:
    """Return a deterministic refusal reason, or ``None`` when execution may proceed.

    Called at the tool-dispatch chokepoints right before execution. ``provider``/``model``
    are the route the agent is actually serving on at dispatch time. Opt-in via the
    ``fallback.halt_on_side_effecting_tools`` config key (default ``False`` keeps the
    legacy unrestricted behavior).
    """
    from agent.tool_result_classification import tool_may_have_side_effect

    if not getattr(agent, "_provider_fallback_active", False):
        return None
    if not _gate_enabled():
        return None
    if tool_name in _GATE_ALLOWED_TOOLS:
        return None
    if not tool_may_have_side_effect(tool_name):
        return None
    return _BLOCK_MESSAGE.format(
        tool=tool_name,
        model=str(model or "unknown"),
        provider=str(provider or "unknown"),
    )


def _gate_enabled() -> bool:
    """Resolve ``fallback.halt_on_side_effecting_tools`` from config (default False)."""
    try:
        from hermes_cli.config import load_config_readonly

        fallback_cfg = load_config_readonly().get("fallback", {})
        return bool(isinstance(fallback_cfg, dict) and fallback_cfg.get("halt_on_side_effecting_tools", False))
    except Exception as _cfg_err:
        # Fail-open on a broken config read (legacy behavior): log so the failure is observable.
        logger.debug("fallback gate config read failed; gate disabled: %s", _cfg_err)
        return False
