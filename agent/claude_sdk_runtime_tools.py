"""Hybrid MCP bridge enablement and the registry snapshot handed to the session.
Extracted from ``claude_sdk_runtime.py``.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

# Same logger name as the origin module so log records / caplog filters are unchanged.
logger = logging.getLogger("agent.claude_sdk_runtime")


def _hybrid_bridge_enabled() -> bool:
    """agent.claude_agent_sdk.hybrid_mcp_bridge from config.yaml.

    Off by default — the wide bridge exposes agent-level tools whose
    enablement is a security choice. Operators opt in via config.
    """
    from agent.transports.claude_agent_sdk_session import _provider_flag

    return _provider_flag("hybrid_mcp_bridge", default=False)


def _snapshot_agent_tools_with_mcp_refresh(agent) -> Optional[List[Dict[str, Any]]]:
    """Return ``agent.tools`` after making sure late-registered MCP tools are
    included. See ``tools/mcp_tool.py::refresh_agent_mcp_tools`` for context:
    the agent snapshots its tool list once at build time and never re-reads
    the registry, so MCP servers whose initial connect finishes AFTER that
    snapshot (slow HTTP handshake, OAuth-gated servers, ``/reload-mcp``) are
    invisible until the snapshot is rebuilt. ``turn_context.py`` already does
    this between turns; we do it at session-creation time too so the hybrid
    MCP bridge (which is built ONCE per session) doesn't freeze a stale
    snapshot into the SDK's ``mcp_servers`` for the entire session.

    Never raises; a refresh failure falls back to the raw snapshot.
    """
    try:
        # Same import-cost gate as turn_context.py's between-turns refresh:
        # ``tools.mcp_tool`` is heavy (~0.4s) and only worth importing when
        # something else already did (which means MCPs may be registered).
        import sys as _sys
        if "tools.mcp_tool" in _sys.modules:
            from tools.mcp_tool_agent import refresh_agent_mcp_tools
            from tools.mcp_tool_discovery import has_registered_mcp_tools
            if has_registered_mcp_tools() and not getattr(
                agent, "_skip_mcp_refresh", False
            ):
                refresh_agent_mcp_tools(agent, quiet_mode=True)
    except Exception:
        logger.debug("MCP refresh before hybrid build failed", exc_info=True)
    return getattr(agent, "tools", None)
