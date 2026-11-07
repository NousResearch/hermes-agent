"""Recovery rung: drop the one MCP tool whose schema the provider rejected, then retry.

Anthropic validates tool schemas more strictly than JSON Schema's meta-schema does, so a schema from an MCP server
can pass every local check and still fail the whole request with ``tools.N.custom.input_schema: JSON schema is
invalid``. The rejection is the only thing that can name the tool, and one bad tool otherwise stops every turn.

Only MCP tools are dropped: their schemas are a third party's, and the rest of the agent works without them. A
built-in tool failing here is Hermes's own bug, and losing one could leave the agent quietly unable to do its job, so
that case keeps failing the request.

The drop lasts as long as this agent, through ``agent_tool_drops`` so an MCP refresh cannot bring the tool back. A new
agent for the session (a restart, ``--resume``) pays for one rejection and drops it again.
"""
from __future__ import annotations

import json
import logging
from typing import Any, Optional

from agent.error_classifier import _REJECTED_TOOL_SCHEMA_RE, FailoverReason
from agent.turn_retry_state import TurnRetryState

logger = logging.getLogger(__name__)

# A safeguard, not a tuning knob: every prune shrinks ``agent.tools``, so the retries already terminate.
_MAX_PRUNES_PER_CALL = 5
_SCHEMA_LOG_LIMIT = 2000


def _sent_tool_name(api_kwargs: Any, index: int) -> Optional[str]:
    """The name at ``index`` in the tools array AS SENT: Anthropic-shaped, or OpenAI-shaped through a relay. The wire
    array is what the provider's index counts, and it need not match ``agent.tools`` (dedup, ordering)."""
    sent = api_kwargs.get("tools") if isinstance(api_kwargs, dict) else None
    if not isinstance(sent, list) or not 0 <= index < len(sent) or not isinstance(sent[index], dict):
        return None
    entry = sent[index]
    function = entry.get("function")
    name = function.get("name") if isinstance(function, dict) else entry.get("name")
    return name if isinstance(name, str) and name else None


def _agent_tool_matching(agent: Any, wire_name: str) -> Optional[dict[str, Any]]:
    """``agent.tools`` entry for a wire name, undoing Anthropic OAuth's ``mcp_x`` -> ``mcp__x`` rewrite."""
    bare = wire_name.removeprefix("mcp__")
    candidates = (wire_name, f"mcp_{bare}", bare)
    by_name = {tool["function"]["name"]: tool for tool in agent.tools or []}
    return next((by_name[name] for name in candidates if name in by_name), None)


def recover_rejected_tool_schema(
    agent: Any, api_error: Exception, classified: Any, _retry: TurnRetryState, api_kwargs: Any,
) -> bool:
    """Remove the rejected MCP tool from this agent's tools. True means retry; False leaves everything as it was."""
    if classified.reason != FailoverReason.tool_schema_rejected:
        return False
    match = _REJECTED_TOOL_SCHEMA_RE.search(f"{api_error} {classified.message}")
    wire_name = _sent_tool_name(api_kwargs, int(match.group(1))) if match else None
    tool = _agent_tool_matching(agent, wire_name) if wire_name else None
    if tool is None:
        logger.warning("%sProvider rejected a tool schema that could not be mapped to a tool: %s",
                       agent.log_prefix, api_error)
        return False

    name = tool["function"]["name"]
    from tools.registry import registry
    toolset = registry.get_toolset_for_tool(name) or ""
    if not toolset.startswith("mcp-"):
        logger.error("%sProvider rejected the schema of non-MCP tool '%s'; not dropping it: %s",
                     agent.log_prefix, name, api_error)
        agent._emit_warning(f"The provider rejected the schema of Hermes tool '{name}', which is not an MCP tool, so "
                            f"it was kept and the request failed. Please report this as a Hermes bug.")
        return False
    if _retry.tool_schema_prunes >= _MAX_PRUNES_PER_CALL:
        logger.error("%sProvider rejected the schema of '%s' after %d tools were already dropped from this call; "
                     "giving up: %s", agent.log_prefix, name, _retry.tool_schema_prunes, api_error)
        return False

    _retry.tool_schema_prunes += 1
    _retry.tools_rewritten = True
    from tools.mcp_tool_agent import _agent_tools_lock
    with _agent_tools_lock:
        agent._rejected_schema_tools = {*getattr(agent, "_rejected_schema_tools", ()), name}
        # New containers on purpose: the tool dicts are shared with the process-wide definitions cache.
        agent.tools = [t for t in agent.tools if t is not tool]
        agent.valid_tool_names = set(agent.valid_tool_names) - {name}

    schema = json.dumps(tool["function"].get("parameters"), default=str)[:_SCHEMA_LOG_LIMIT]
    logger.error("%sProvider rejected the schema of MCP tool '%s' (toolset %s); dropped it. "
                 "Schema: %s. Provider error: %s", agent.log_prefix, name, toolset, schema, api_error)
    agent._emit_warning(f"The provider rejected the schema of MCP tool '{name}', so it was removed and the request "
                        f"retried. Report the schema to that server's maintainer; details are in the log.")
    return True
