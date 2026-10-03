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

# Machine-readable refusal identity, matching the {"error": code, "message": prose}
# convention _blocked_tool_result already accepts — so tests and log consumers key on the
# code instead of grepping the wording below.
_FALLBACK_ROUTE_BLOCK_ERROR = "fallback_route_block"

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
    # Cheapest predicates first, config read last: the gate is off for most agents, and this
    # function sits on the per-tool-call dispatch path.
    if not getattr(agent, "_provider_fallback_active", False):
        return None
    if tool_name in _GATE_ALLOWED_TOOLS:
        return None

    from agent.tool_result_classification import tool_may_have_side_effect

    if not tool_may_have_side_effect(tool_name):
        return None
    if not _gate_enabled():
        if _gate_config_unreadable():
            logger.warning("fallback route gate config unreadable; gate off for this call")
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

        cfg = load_config_readonly()
        fallback_cfg = cfg.get("fallback", {})
        return bool(isinstance(fallback_cfg, dict) and fallback_cfg.get("halt_on_side_effecting_tools", False))
    except Exception as _cfg_err:
        # Fail-open: the gate exists only because #117495's failure mode is unrecoverable, but a
        # transient read error must not freeze every tool. WARNING, not debug — the root logger
        # defaults to INFO and errors.log to WARNING, so a debug record here is dropped and the
        # gate silently disappears for a user who explicitly opted in.
        logger.warning("fallback gate config read failed; gate disabled for this call: %s", _cfg_err)
        return False


def _gate_config_unreadable() -> bool:
    """True when the config file exists but could not be read.

    ``load_config_readonly``/``read_raw_config_readonly`` do NOT raise in that case — they return
    a mapping carrying ``read_error``, so the key silently resolves to its default and the gate
    turns itself off for a user who opted in. This is the only way to notice.
    """
    try:
        from hermes_cli.config import load_config_readonly

        return getattr(load_config_readonly(), "read_error", None) is not None
    except Exception:
        return False

def arm_route_gate(agent, model: str, provider: str) -> None:
    """Record that an automatic provider fallback is the acting route (#117495).

    Call this BEFORE the route fields are swapped. A raise between the swap and this write is
    caught upstream and continues to the next chain entry, leaving the turn serving the fallback
    with the gate off — the exact failure this gate exists to prevent. Arming early can only
    refuse one tool a moment early, which is recoverable.
    """
    agent._provider_fallback_active = True
    agent._provider_fallback_route = (str(model), str(provider))


def clear_route_gate(agent) -> None:
    """Record that the acting route is the user's own again, not an automatic fallback.

    Call this BEFORE any fallible work on the restore or switch path, for the mirror reason:
    a raise afterwards would leave the gate armed on the primary, freezing every side-effecting
    tool on a route the user picked.
    """
    agent._provider_fallback_active = False
    agent._provider_fallback_route = None
