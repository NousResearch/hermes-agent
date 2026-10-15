"""The terminal's view of the skills MCP servers publish.

`hermes skills list` and `hermes skills config` both show remote skills beside local
ones, so the row shape and the connect-on-demand live here once rather than in each
surface. A config command opens no MCP connection of its own, so the catalog is
fetched on the way past — and only from the servers that opted in to publishing
skills. Anything that goes wrong costs these rows alone: the on-disk skills still
list, which is what the operator came for.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def remote_skill_rows() -> list[dict]:
    """Every published remote skill this profile may load, marked ``remote``.

    Includes the switched-off ones: both surfaces that call this are places where a
    disabled skill has to be visible, or it could never be switched back on.
    """
    try:
        from tools.skills_tool_mcp import known_server_labels, list_rows
    except ImportError:  # `mcp` is an optional extra; the CLI works without it
        return []
    try:
        servers = sorted(known_server_labels())
        if not servers:
            return []
        if not list_rows(skip_disabled=True):
            _connect_servers(servers)
        return [dict(row, remote=True) for row in list_rows(skip_disabled=True)]
    except Exception:
        logger.warning("Remote skills could not be listed", exc_info=True)
        return []


def _connect_servers(servers: list[str]) -> None:
    """Connect just *servers*, without letting OAuth read from this command's stdin."""
    try:
        from tools.mcp_oauth import suppress_interactive_oauth
    except ImportError:
        from contextlib import nullcontext as suppress_interactive_oauth
    from tools.mcp_tool_discovery import discover_mcp_tools

    with suppress_interactive_oauth():
        discover_mcp_tools(allowed_mcp_names=servers)
