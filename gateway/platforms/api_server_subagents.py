"""Session-owned delegate children over HTTP: list and cooperative stop.

``GET /api/sessions/{session_id}/subagents`` lists the live ``delegate_task`` children this
session's conversation (compression lineage) owns in this profile's SessionDB, projected like the
TUI ``subagent.list`` roster. ``POST .../subagents/{subagent_id}/stop`` asks one of them to stop.
Both refuse without a configured API key: a keyless listener cannot tell owners apart.
"""
from __future__ import annotations

import asyncio
from typing import Any, Callable

from aiohttp import web

from tools.delegate_tool_registry import list_session_subagents, request_session_subagent_stop

_API_SESSION_SOURCE = "api_server"
# Same projection as the TUI roster (tui_gateway/methods_subagents.py), plus the child's own session.
_SNAPSHOT_FIELDS = (
    "subagent_id", "parent_id", "depth", "goal", "delegation_id", "model",
    "started_at", "status", "tool_count", "last_tool", "accepting_steer",
)


def _http_routes(adapter, *, error_response: Callable[..., web.Response]) -> list[tuple[str, str, Any]]:
    async def list_children(request: web.Request) -> web.Response:
        return await list_session_subagents_route(adapter, request, error_response=error_response)

    async def stop_child(request: web.Request) -> web.Response:
        return await stop_session_subagent(adapter, request, error_response=error_response)

    return [("GET", "/api/sessions/{session_id}/subagents", list_children),
            ("POST", "/api/sessions/{session_id}/subagents/{subagent_id}/stop", stop_child)]


async def _owned_session(adapter, request: web.Request, error_response) -> web.Response | tuple[str, Any]:
    """``(session_id, db)`` for an authenticated API-server session, else the error response."""
    auth_err = adapter._check_auth(request)
    if auth_err:
        return auth_err
    if not adapter._expected_api_key():
        return error_response("Subagent control requires API key authentication", 403,
                              code="subagent_auth_required")
    session_id = request.match_info["session_id"]
    session, error = await adapter._get_existing_session_or_404(session_id)
    if error is not None:
        return error
    if session.get("source") != _API_SESSION_SOURCE:
        return error_response("Session not found", 404, code="session_not_found")
    return session_id, await adapter._ensure_session_db_async()


def _snapshot(record: dict[str, Any]) -> dict[str, Any]:
    child_sid = getattr(record.get("agent"), "session_id", None)
    return {**{key: record.get(key) for key in _SNAPSHOT_FIELDS},
            "child_session_id": child_sid if isinstance(child_sid, str) and child_sid else None}


async def list_session_subagents_route(adapter, request: web.Request, *, error_response) -> web.Response:
    owned = await _owned_session(adapter, request, error_response)
    if isinstance(owned, web.Response):
        return owned
    session_id, db = owned
    # Ownership resolves compression lineage through the DB: off the event loop.
    records = await asyncio.to_thread(list_session_subagents, session_id, db)
    return web.json_response({"object": "list", "session_id": session_id,
                              "data": [_snapshot(r) for r in records]})


async def stop_session_subagent(adapter, request: web.Request, *, error_response) -> web.Response:
    owned = await _owned_session(adapter, request, error_response)
    if isinstance(owned, web.Response):
        return owned
    session_id, db = owned
    outcome = await asyncio.to_thread(
        request_session_subagent_stop, request.match_info["subagent_id"], session_id, db)
    if outcome is None:
        return error_response("Subagent not found", 404, code="subagent_not_found")
    if outcome != "stopping":
        return error_response("Subagent is no longer stoppable", 409, code="subagent_not_active")
    return web.json_response({"status": "stopping"}, status=202)
