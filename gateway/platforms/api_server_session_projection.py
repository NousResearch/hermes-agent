"""Client-safe projection of a ``sessions`` row for the API server's session resources."""

from __future__ import annotations

import json
from typing import Any


def session_response(session: dict[str, Any]) -> dict[str, Any]:
    """Return a stable, client-safe session representation."""
    safe_keys = (
        "id", "source", "user_id", "model", "title", "started_at", "ended_at", "end_reason",
        "message_count", "tool_call_count", "input_tokens", "output_tokens",
        "context_tokens", "context_limit",
        "cache_read_tokens", "cache_write_tokens", "reasoning_tokens", "estimated_cost_usd",
        "actual_cost_usd", "api_call_count", "parent_session_id", "last_active", "preview",
        "_lineage_root_id", "pinned", "archived", "hidden")
    payload = {key: session.get(key) for key in safe_keys if key in session}
    # SQLite stores the flags as 0/1.
    payload.update(
        {f: bool(payload[f]) for f in ("pinned", "archived", "hidden") if f in payload})
    # Full system prompts / model_config never cross the client API; only their presence.
    payload["has_system_prompt"] = bool(session.get("system_prompt"))
    payload["has_model_config"] = bool(session.get("model_config"))
    raw_model_config = session.get("model_config")
    try:
        model_config = (
            json.loads(raw_model_config)
            if isinstance(raw_model_config, str)
            else raw_model_config
        )
    except (TypeError, json.JSONDecodeError):
        model_config = None
    # Exact-id consumers may inspect/resume delegate children even though
    # list endpoints intentionally omit them. Project only the provenance
    # bit the client needs so it cannot accidentally promote such a row
    # into an ordinary session list; never expose the model snapshot.
    payload["is_internal_child"] = bool(
        isinstance(model_config, dict)
        and model_config.get("_delegate_from") is not None
    )
    return payload
