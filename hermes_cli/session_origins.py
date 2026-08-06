"""Privacy-safe gateway origin metadata for session list responses."""

import json
from typing import Any, Dict, List, Optional


def _session_origin_text(value: Any, *, max_chars: int = 240) -> str:
    """Normalize untrusted gateway labels for a compact JSON UI payload."""
    if value is None:
        return ""
    text = str(value).replace("\r\n", "\n").replace("\r", "\n").strip()
    text = "".join(ch if ch >= " " else " " for ch in text)
    if len(text) > max_chars:
        text = text[: max_chars - 3] + "..."
    return text


def session_origin_metadata(row: Dict[str, Any]) -> Optional[Dict[str, str]]:
    """Project a state.db gateway row into non-secret, UI-ready metadata."""
    raw = row.get("origin_json")
    if not raw:
        return None
    try:
        origin = json.loads(raw) if isinstance(raw, str) else raw
    except (TypeError, ValueError):
        return None
    if not isinstance(origin, dict):
        return None

    platform = _session_origin_text(origin.get("platform") or row.get("source"))
    if not platform:
        return None
    chat_type = _session_origin_text(origin.get("chat_type") or row.get("chat_type")) or "chat"
    is_dm = chat_type.casefold() == "dm"
    fallback_kind = "DM" if is_dm else chat_type
    display_label = f"{platform.title()} {fallback_kind}"

    if not is_dm:
        identifier_fields = (
            "chat_id",
            "chat_id_alt",
            "guild_id",
            "message_id",
            "parent_chat_id",
            "routing_key",
            "scope_id",
            "session_key",
            "thread_id",
            "user_id",
            "user_id_alt",
            "user_name",
        )
        identifiers = set()
        for key in identifier_fields:
            for source in (row, origin):
                value = _session_origin_text(source.get(key))
                if value:
                    identifiers.add(value.casefold())
        for candidate in (row.get("display_name"), origin.get("chat_name")):
            label = _session_origin_text(candidate)
            if label and label.casefold() not in identifiers:
                display_label = label
                break

    return {"platform": platform, "chat_type": chat_type, "display_label": display_label}


def enrich_sessions_with_origins(sessions: List[Dict[str, Any]], db: Any) -> None:
    """Attach gateway origins from the canonical state.db routing index."""
    if not sessions:
        return
    wanted = [str(session.get("id") or "") for session in sessions]
    origins: Dict[str, Dict[str, str]] = {}
    for row in db.get_gateway_session_metadata(wanted).values():
        session_id = str(row.get("id") or "")
        metadata = session_origin_metadata(row)
        if metadata:
            origins[session_id] = metadata
    for session in sessions:
        origin = origins.get(str(session.get("id") or ""))
        if origin:
            session["origin"] = origin
