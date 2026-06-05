"""
agente_desktop_adapter

Stable shim isolating all Agente Desktop / API compatibility logic.

Per hermes-upstream-compatibility-plan.md:
- All Agente-specific Desktop/API compatibility lives here.
- The only integration point in upstream is the single hook:
  register_agente_desktop_routes(app, adapter)
  inside gateway/platforms/api_server.py .
- Upstream internals carry zero Agente edits.

This enables daily rebase from NousResearch/hermes-agent without recurring
merge conflicts on Desktop surfaces (tools, workflows, routines, BYOK, WhatsApp triage, JSON-RPC, etc.).
"""
from . import tool_discovery, jsonrpc_compat, workflow_routine_bridge, byok_session_aliases, whatsapp_ticket_integration

def register_agente_desktop_routes(app, adapter=None):
    """
    Register Desktop-specific routes and patches on the API app.

    Called from api_server.py after base routes.
    """
    tool_discovery.register(app, adapter)
    jsonrpc_compat.register(app, adapter)
    workflow_routine_bridge.register(app, adapter)
    byok_session_aliases.register(app, adapter)
    whatsapp_ticket_integration.register(app, adapter)
