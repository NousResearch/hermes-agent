"""Keep MCP servers away from secondary profiles under multiplex (4c).

MCP server processes run once per gateway: ``tools/mcp_tool.py`` keeps them
in one process-global registry keyed by server name, started from the
**default** profile's ``mcp_servers`` with the default's credentials
(Webflow, Stripe, GitHub, …). Discovery is idempotent by name, so even a
secondary profile that declares a server of its own would dispatch to the
default's process when the names match. Until per-profile MCP processes
exist (a non-goal of the design), a secondary profile gets **no** MCP
toolset at all.

Applied twice: when the turn's toolsets are resolved (the model is never
offered the tools, gateway turns and cron) and at dispatch (the
``model_tools`` backstop). Inert with multiplex off and for the default
profile. With multiplex on, a guard that cannot evaluate fails closed.

See docs/superpowers/specs/2026-11-16-multiplex-fork-hardening-design.md.
"""
from __future__ import annotations

import logging
from typing import Iterable, List, Optional

logger = logging.getLogger(__name__)

MCP_TOOLSET_PREFIX = "mcp-"
MCP_TOOL_PREFIX = "mcp__"


def _multiplex_on() -> bool:
    try:
        from agent.secret_scope import is_multiplex_active

        return is_multiplex_active()
    except Exception:
        return False


def is_secondary_profile_context() -> bool:
    """True when multiplexing is on and the current home is not the process's."""
    if not _multiplex_on():
        return False
    from hermes_constants import get_process_hermes_home, hermes_home_key

    return hermes_home_key() != hermes_home_key(get_process_hermes_home())


def _secondary_or_unknown() -> bool:
    """``is_secondary_profile_context``, failing closed under multiplex."""
    try:
        return is_secondary_profile_context()
    except Exception:
        multiplex = _multiplex_on()
        if multiplex:
            logger.warning(
                "multiplex MCP guard could not resolve the profile context; "
                "treating it as a secondary profile",
                exc_info=True,
            )
        return multiplex


def mcp_server_for_toolset(toolset: str) -> Optional[str]:
    """Server name behind an MCP toolset (``mcp-<x>`` or its alias), else None."""
    name = str(toolset or "")
    if name.startswith(MCP_TOOLSET_PREFIX):
        return name[len(MCP_TOOLSET_PREFIX):]
    try:
        from tools.registry import registry

        target = registry.get_toolset_alias_target(name)
    except Exception:
        target = None
    if target and str(target).startswith(MCP_TOOLSET_PREFIX):
        return str(target)[len(MCP_TOOLSET_PREFIX):]
    return None


def filter_foreign_mcp_toolsets(toolsets: Iterable[str]) -> List[str]:
    """Drop every MCP toolset in a secondary profile's context.

    Returns the input unchanged (as a list) with multiplex off or in the
    default profile's context.
    """
    items = list(toolsets)
    if not _secondary_or_unknown():
        return items
    kept: List[str] = []
    for ts in items:
        if mcp_server_for_toolset(ts) is not None:
            logger.info(
                "multiplex: dropping MCP toolset %r from a secondary profile's "
                "run (MCP servers are process-wide, owned by the default profile)",
                ts,
            )
            continue
        kept.append(ts)
    return kept


def foreign_mcp_denial(tool_name: str) -> Optional[str]:
    """Dispatch backstop: deny any MCP tool in a secondary profile's context.

    Fails closed: when the profile context or the tool's toolset cannot be
    resolved under multiplex, an ``mcp__`` tool is denied.
    """
    if not _secondary_or_unknown():
        return None
    try:
        from tools.registry import registry

        toolset = registry.get_toolset_for_tool(tool_name)
        is_mcp = bool(toolset and mcp_server_for_toolset(toolset) is not None)
    except Exception:
        is_mcp = False
    if not is_mcp and not str(tool_name).startswith(MCP_TOOL_PREFIX):
        return None
    return (
        f"⛔ '{tool_name}' is an MCP tool. MCP servers run once per gateway with "
        "the default profile's credentials, so they are not available here."
    )
