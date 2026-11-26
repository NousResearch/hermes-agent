"""Bounded, explicitly selected conversation provenance stored on Kanban tasks."""
from urllib.parse import urlencode

from agent.redact import redact_sensitive_text


_ORIGIN_COORDINATES = (
    "profile", "session_id", "platform", "chat_id", "thread_id", "message_id",
    "scope_id", "parent_chat_id", "transport_profile",
)


def sanitize_origin(origin: dict | None) -> dict | None:
    if origin is None:
        return None
    clean = {}
    for key in (*_ORIGIN_COORDINATES, "prompt_excerpt"):
        value = origin.get(key)
        if isinstance(value, str) and value:
            if key != "prompt_excerpt" and len(value) > 256:
                continue  # Never invent a different identifier by truncation.
            clean[key] = redact_sensitive_text(value, force=True)[:2000]
    row_id = origin.get("message_row_id")
    if type(row_id) is int and row_id > 0:
        clean["message_row_id"] = row_id
    if "session_link" in origin and clean.get("session_id"):
        clean["session_link"] = "/chat?" + urlencode({
            "resume": clean["session_id"], "profile": clean.get("profile", "default")})
    return clean
