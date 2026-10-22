"""HTTP adapter for the existing cooperative, conversation-owned child stop."""
from __future__ import annotations

from aiohttp import web


async def stop_session_subagent(adapter, request: web.Request) -> web.Response:
    from gateway.platforms.api_server import _error_response
    from tools.delegate_tool_registry import request_session_subagent_stop

    if not adapter._expected_api_key():
        return _error_response("Subagent stop requires API key authentication", 403, code="subagent_auth_required")
    session_id = request.match_info["session_id"]
    session, error = await adapter._get_existing_session_or_404(session_id)
    if error is not None:
        return error
    if session.get("source") != "api_server":
        return _error_response("Session not found", 404, code="session_not_found")
    db = await adapter._ensure_session_db_async()
    outcome = request_session_subagent_stop(request.match_info["subagent_id"], session_id, db)
    if outcome is None:
        return _error_response("Subagent not found", 404, code="subagent_not_found")
    if outcome != "stopping":
        return _error_response("Subagent is no longer stoppable", 409, code="subagent_not_active")
    return web.json_response({"status": "stopping"}, status=202)
