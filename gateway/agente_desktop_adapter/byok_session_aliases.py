"""
byok_session_aliases.py

BYOK / session aliasing and credential seam for Desktop.

Houses any future dedicated BYOK token exchange, session_key alias
resolution, and companion-app credential injection that is not already
covered by jsonrpc_compat or the web_server session_token file logic
(which is kept in core but with Agente mentions stripped).
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


def register(app: Any, adapter: Any = None) -> None:
    """Apply BYOK/desktop session alias seams if needed at startup."""
    if adapter is None:
        adapter = getattr(app, "get", lambda k: None)("api_server_adapter") if hasattr(app, "get") else None
        if adapter is None and hasattr(app, "__getitem__"):
            try:
                adapter = app["api_server_adapter"]
            except Exception:
                adapter = None
    # Currently most aliasing lives in jsonrpc_compat (which this module is
    # still called alongside). Placeholder for dedicated BYOK flows or
    # session token file integration if split further.
    logger.debug("agente_desktop_adapter.byok_session_aliases: register called (mostly delegated to jsonrpc_compat)")
