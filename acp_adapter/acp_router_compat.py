"""Compatibility import for ACP's agent router."""

from __future__ import annotations

try:
    from acp.agent.router import build_agent_router  # type: ignore[import-not-found]
except ImportError:
    from _acp_fallback.agent.router import build_agent_router
