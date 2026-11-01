"""Out-of-process Matrix delivery through the crypto-owning gateway."""


def relay_matrix_delivery(chat_id, message, *, media_files=None, thread_id=None, force_document=False):
    from gateway.control_socket import query_gateway_control
    from hermes_constants import get_hermes_home

    home = get_hermes_home()
    live = query_gateway_control(home, "send-capabilities")
    if not live:
        return None
    if "matrix-send" not in live.get("supported_verbs", []):
        return {"error": "Running gateway lacks Matrix media relay support; upgrade the gateway"}
    result = query_gateway_control(home, "matrix-send", params={
        "home": str(home), "chat_id": chat_id, "message": message,
        "media_files": media_files or [], "thread_id": thread_id, "force_document": force_document,
    }, timeout=60)
    return result if result is not None else {
        "error": "Gateway delivery outcome unknown; not retrying to avoid duplicates"}
