"""Session resource projection and client metadata for the API server's ``/api/sessions`` routes.

``session_response`` is the client-safe row shape every session endpoint returns.
``metadata_patch`` / ``metadata_filter`` turn a PATCH body / list query into the validated inputs
``hermes_state_session_metadata`` stores and filters by; a violation is a 400 naming the field.
"""

from __future__ import annotations

import asyncio
import json
from typing import TYPE_CHECKING, Any

from hermes_state_session_metadata import (
    SessionMetadataError, decode_metadata, merge_session_metadata, parse_metadata_filter,
    parse_metadata_patch,
)

if TYPE_CHECKING:  # aiohttp is optional; api_server.py guards its own import of it.
    from aiohttp import web

_SAFE_KEYS = (
    "id", "source", "user_id", "model", "title", "started_at", "ended_at", "end_reason",
    "message_count", "tool_call_count", "input_tokens", "output_tokens",
    "cache_read_tokens", "cache_write_tokens", "reasoning_tokens", "estimated_cost_usd",
    "actual_cost_usd", "api_call_count", "parent_session_id", "last_active", "preview",
    "_lineage_root_id", "pinned", "archived", "hidden")
_FILTER_PREFIX = "metadata."


def session_response(session: dict[str, Any]) -> dict[str, Any]:
    """Return a stable, client-safe session representation."""
    payload = {key: session.get(key) for key in _SAFE_KEYS if key in session}
    # SQLite stores the flags as 0/1.
    payload.update({f: bool(payload[f]) for f in ("pinned", "archived", "hidden") if f in payload})
    payload["metadata"] = decode_metadata(session.get("metadata"))
    # Full system prompts / model_config never cross the client API; only their presence.
    payload["has_system_prompt"] = bool(session.get("system_prompt"))
    payload["has_model_config"] = bool(session.get("model_config"))
    raw_model_config = session.get("model_config")
    try:
        model_config = json.loads(raw_model_config) if isinstance(raw_model_config, str) else raw_model_config
    except (TypeError, json.JSONDecodeError):
        model_config = None
    # Exact-id consumers may inspect/resume delegate children even though list endpoints
    # intentionally omit them. Project only the provenance bit the client needs so it cannot
    # accidentally promote such a row into an ordinary session list; never expose the model snapshot.
    payload["is_internal_child"] = bool(
        isinstance(model_config, dict) and model_config.get("_delegate_from") is not None)
    return payload


def _invalid(exc: SessionMetadataError) -> web.Response:
    # Late: api_server imports this module at load time.
    from gateway.platforms.api_server import _error_response
    return _error_response(str(exc), 400, param=exc.field, code="invalid_session_metadata")


def metadata_patch(body: dict[str, Any]) -> tuple[dict[str, str | None] | None, web.Response | None]:
    """``(patch, None)`` for a body carrying ``metadata``, ``(None, None)`` without one, ``(None, 400)``."""
    if "metadata" not in body:
        return None, None
    try:
        return parse_metadata_patch(body["metadata"]), None
    except SessionMetadataError as exc:
        return None, _invalid(exc)


async def apply_metadata_patch(db, session_id: str, patch: dict[str, str | None] | None) -> web.Response | None:
    """Merge *patch* off the event loop; a 400 when the merged object would exceed the key cap."""
    if patch is None:
        return None
    try:
        await asyncio.to_thread(merge_session_metadata, db, session_id, patch)
    except SessionMetadataError as exc:
        return _invalid(exc)
    return None


def metadata_filter(query) -> tuple[dict[str, str] | None, web.Response | None]:
    """Repeated ``metadata.<key>=<value>`` query params, ANDed. One key with two different values is
    a 400: no row can match both, and keeping either one would silently widen the filter."""
    pairs = [(name[len(_FILTER_PREFIX):], value) for name, value in query.items()
             if name.startswith(_FILTER_PREFIX)]
    if not pairs:
        return None, None
    try:
        parsed = parse_metadata_filter(pairs)
    except SessionMetadataError as exc:
        return None, _invalid(exc)
    if conflict := next((k for k, v in pairs if parsed[k] != v), None):
        return None, _invalid(SessionMetadataError(
            f"metadata.{conflict}", f"'metadata.{conflict}' was given more than one value"))
    return parsed, None
