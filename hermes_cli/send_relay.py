"""Relay Matrix CLI sends to the gateway owning its crypto state."""


def relay_matrix_send(target, tool_args):
    """None means no relay; an uncertain delivery is an error, never a retry."""
    from gateway.control_socket import query_gateway_control
    from hermes_constants import get_hermes_home

    if not target.lower().startswith("matrix:"):
        return None
    home = get_hermes_home()
    live = query_gateway_control(home, "send-capabilities")
    if not live or "send-message" not in live.get("supported_verbs", []):
        return None
    result = query_gateway_control(home, "send-message", params={**tool_args, "home": str(home)}, timeout=60)
    return result if result is not None else {
        "error": "Gateway delivery outcome unknown; not retrying to avoid duplicates"}
