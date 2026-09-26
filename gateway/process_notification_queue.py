"""Retain process identity across the gateway FIFO, then discard superseded wakes."""


def notification_metadata(text, event) -> dict:
    """Only process completions/heartbeats have a registry-backed stale verdict."""
    if event.get("type") not in {"completion", "heartbeat"}:
        return {}
    from gateway.run import _redact_gateway_user_facing_secrets

    entries = event.get("_process_notification_entries") or [(text, event)]
    # Keep the renderer's fields, not routing/credentials or an entire producer payload.
    fields = ("type", "session_id", "exit_code", "completion_reason", "output", "started_at")
    return {"process_notifications": [
        {"text": _redact_gateway_user_facing_secrets(text),
         "event": {key: (_redact_gateway_user_facing_secrets(str(raw[key]))
                         if key == "output" else raw[key]) for key in fields if key in raw}}
        for text, raw in entries
    ]}


def refresh_process_notification(event, format_batch) -> bool:
    """Discard obsolete internal wakes; retain unread completion results.

    Admission is not execution: the foreground agent can read a completion, or a process can
    exit, while the synthetic event waits in the adapter slot or runner overflow FIFO.
    """
    if not event.internal:
        return True
    entries = (event.metadata or {}).get("process_notifications")
    if not entries:
        return True
    from tools.process_registry import process_registry

    live = []
    for entry in entries:
        raw = entry["event"]
        process_id = raw.get("session_id", "")
        if raw.get("type") == "completion" and process_registry.is_completion_consumed(process_id):
            continue
        if raw.get("type") == "heartbeat":
            process = process_registry.get(process_id)
            expected_start = raw.get("started_at")
            if (process is None or process.id != process_id or process.exited
                    or (expected_start is not None and process.started_at != expected_start)):
                continue
        live.append(entry)
    if not live:
        return False
    if len(live) != len(entries):
        event.metadata["process_notifications"] = live
        event.text = (live[0]["text"] if len(live) == 1 else
                      format_batch([(item["text"], item["event"], None) for item in live]))
    return True
