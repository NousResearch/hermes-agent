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

    Answered from the catalog already held — the screen this feeds has to render without
    waiting on a server. A listing the user reads as *current* wants
    :func:`verified_remote_skill_rows` instead.
    """
    try:
        from tools.skills_tool_mcp import _connect_servers, known_server_labels, list_rows
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


def verified_remote_skill_rows() -> tuple[list[dict], list[dict]]:
    """Rows for a listing the user asked for now, plus one problem per unreadable server.

    The counterpart of :func:`remote_skill_rows`, for the surfaces where a person reads the
    result as today's answer: the servers are re-read first, and one that cannot be read
    contributes a problem instead of its cached rows. Those rows would advertise skills this
    host may no longer be able to load — the failure this distinction exists to remove.
    """
    try:
        from tools.skills_tool_mcp import known_server_labels, list_rows, refresh_catalogs
    except ImportError:  # `mcp` is an optional extra; the CLI works without it
        return [], []
    try:
        servers = sorted(known_server_labels())
    except Exception:
        logger.warning("MCP server labels unavailable", exc_info=True)
        return [], []
    if not servers:
        return [], []
    try:
        verified, problems = refresh_catalogs()
        answering = set(verified)
        rows = [dict(row, remote=True) for row in list_rows(skip_disabled=True)
                if row.get("server") in answering]
        return rows, problems
    except Exception:
        logger.warning("Remote skills could not be listed", exc_info=True)
        return [], [{"server": name, "error": "the remote listing failed"} for name in servers]
