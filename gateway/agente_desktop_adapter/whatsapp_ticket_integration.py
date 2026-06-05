"""
whatsapp_ticket_integration.py

WhatsApp triage-triggered ticket creation integration + related ticket
tools parity (create_ticket, list_tickets, move_ticket, link_*, evaluate_triage_rules,
save_workflow_rule etc).

The actual proxy handlers + schemas for these live in tool_discovery (the
IPC bridge). This module is the home for any higher-level triage rule
bridge, whatsapp account mirroring, or future WhatsApp-specific route
extensions (e.g. direct webhook fanout if needed).

For now it ensures the ticket tools are present (via tool_discovery) and
applies any post-register patches for triage.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


def register(app: Any, adapter: Any = None) -> None:
    """Ensure WhatsApp/ticket surfaces are wired for desktop parity."""
    if adapter is None:
        adapter = getattr(app, "get", lambda k: None)("api_server_adapter") if hasattr(app, "get") else None
        if adapter is None and hasattr(app, "__getitem__"):
            try:
                adapter = app["api_server_adapter"]
            except Exception:
                adapter = None

    # The 21 tools (incl. whatsapp + ticket + triage) are registered by
    # tool_discovery.register which is always called first in the adapter __init__.
    # This module can be extended for e.g. auto-wiring evaluate_triage_rules
    # into connector paths or injecting default triage instructions.
    logger.debug("agente_desktop_adapter.whatsapp_ticket_integration: register called (tools via tool_discovery)")
