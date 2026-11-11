"""Session-scoped recovery text for calls to tools reachable only through the Tool Search bridge.

Nothing here touches the system prompt or tool schemas: the bridge's own ``tool_search``
description already lists deferred tools and points at ``tool_describe``/``tool_call``.
"""

import json

from tools.tool_search_catalog import BRIDGE_TOOL_NAMES


def session_deferred_tools(agent) -> frozenset[str]:
    """Use the same session scope as bridge dispatch; an absent bridge has no catalog."""
    if not BRIDGE_TOOL_NAMES.issubset(agent.valid_tool_names):
        return frozenset()
    from agent.tool_executor import _tool_search_scoped_names

    return _tool_search_scoped_names(agent)


def deferred_tool_hint(name: str) -> str:
    quoted = json.dumps(name)
    return (
        f"{quoted} is available through the discovery bridge. "
        f"Call tool_describe(names=[{quoted}]), then tool_call(calls=[{{\"name\": {quoted}, \"arguments\": {{...}}}}]) "
        "using the returned parameter schema. Use tool_call even after describing the tool."
    )


def invalid_tool_name_error_content(agent, name: str) -> str:
    """Error for an unknown tool name: a name this session reaches only through the bridge
    gets the bridge recipe; every other name keeps main's text from
    ``_invalid_tool_name_error_content`` (blank-name anti-priming included, #47967)."""
    if (name or "").strip() and name in session_deferred_tools(agent):
        return deferred_tool_hint(name) + " This attempted call did not execute."
    from agent.conversation_loop import _invalid_tool_name_error_content

    return _invalid_tool_name_error_content(name, agent.valid_tool_names)
