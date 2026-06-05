"""
tool_discovery.py

Handles GET /api/tools (discovery payload shape) and POST /api/tools/{tool_name}
invocation round-trips for the agente-desktop toolset.

See coverage table in hermes-upstream-compatibility-plan.md §5.
"""
def register(app, adapter=None):
    # TODO (thin adapter impl): register /api/tools and dispatch for Desktop tools
    # (list_whatsapp_accounts, create_ticket, etc. via IPC back to Electron)
    # Wire the schemas, check_fns for env availability (AGENTE_TOOL_PORT/SECRET).
    pass
