"""Backwards-compatible alias for the Copilot ACP client.

The generic ACP client now lives in ``agent.acp_client``.  This module
re-exports ``CopilotACPClient`` so existing imports continue to work.
"""

from __future__ import annotations

from typing import Any

from agent.acp_client import ACPClient

# Re-export the marker so existing checks (``base_url.startswith(...)``) work.
ACP_MARKER_BASE_URL = "acp://copilot"


class CopilotACPClient(ACPClient):
    """Copilot-specific ACP client — thin wrapper around the generic ACPClient."""

    def __init__(self, **kwargs: Any):
        kwargs.setdefault("agent_name", "copilot")
        super().__init__(**kwargs)
