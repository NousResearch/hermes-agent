"""Session-scoped Tool Search bridge resolution and resolved-target policy."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Optional

from tools.registry import tool_error

if TYPE_CHECKING:
    from model_tools import _CallIds

logger = logging.getLogger(__name__)


def dispatch_bridge_tool(function_name: str, function_args: dict[str, Any],
                          enabled_toolsets: Optional[list[str]], disabled_toolsets: Optional[list[str]],
                          ids: Optional[_CallIds] = None):
    """Handle a Tool Search bridge call (tool_search / tool_describe / tool_call).

    None when *function_name* is not a bridge tool; ``(result, None)`` for a
    finished catalog read or error; ``(None, (name, args))`` when a validated
    tool_call should be re-dispatched as the real tool.
    """
    from model_tools import get_tool_definitions

    try:
        from tools import tool_search as ts
    except Exception:
        logger.debug("Tool Search bridge unavailable", exc_info=True)
        return None
    if not ts.is_bridge_tool(function_name):
        return None
    # Un-collapsed catalog scoped to the session's toolsets, so a restricted
    # session (subagent, kanban worker) can't reach the whole registry via the bridge.
    try:
        current_defs = get_tool_definitions(enabled_toolsets=enabled_toolsets, disabled_toolsets=disabled_toolsets,
                                            quiet_mode=True, skip_tool_search_assembly=True) or []
    except Exception:
        logger.debug("Tool Search catalog resolution failed", exc_info=True)
        current_defs = []
    args = function_args or {}
    from tools.managed_gateway_identity import CallRef
    call = CallRef(turn_id=ids.turn_id, tool_call_id=ids.tool_call_id) if ids else None
    if function_name == ts.TOOL_SEARCH_NAME:
        return ts.dispatch_tool_search(args, current_tool_defs=current_defs, call=call), None
    if function_name == ts.TOOL_DESCRIBE_NAME:
        return ts.dispatch_tool_describe(args, current_tool_defs=current_defs, call=call), None
    # Session-scoped names for the error path below: a core tool stripped from this
    # session (clarify in a subagent) must not be told to "call it directly".
    underlying_name, underlying_args, err = ts.resolve_underlying_call(args, ts.session_tool_names(current_defs))
    if err or not underlying_name:
        return tool_error(err or "tool_call could not be resolved"), None
    if underlying_name == ts.CONNECTOR_BATCH_SENTINEL:
        if not ts.connections_in_scope(current_defs):
            return tool_error("Connectors are not available in this session."), None
        return None, (underlying_name, underlying_args)
    # Defense in depth: resolve_underlying_call only checks the global
    # registry; also require membership in the session-scoped catalog.
    # Session-gated GUI tools fail fast with their real reason (#120413):
    # tool_search can never surface them in this session.
    if underlying_name not in ts.scoped_deferrable_names(current_defs):
        return tool_error(ts.scope_block_reason(underlying_name)), None
    # Validate against the deferred tool's concrete schema — the generic
    # ``arguments: object`` bridge schema can't enforce it.
    probe_err = ts.validate_deferred_call_args(underlying_name, underlying_args)
    if probe_err is not None:
        return probe_err, None
    # The executor may already have fired pre_tool_call for the bridge. Recheck
    # authority without replaying hooks (modify/block/approve must stay single-fire).
    from hermes_cli.plugins_tool_policy import tool_whitelist_denial
    denial = tool_whitelist_denial(underlying_name)
    if denial is not None:
        return tool_error(denial), None
    return None, (underlying_name, underlying_args)
