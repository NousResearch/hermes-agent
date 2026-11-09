"""``POST /api/relay``: start a turn in a Discord thread (``gateway/session_relay.py``).

A topical sibling of ``platforms/api_server.py`` (root AGENTS.md: the facade stays under its
frozen line cap). Served on the owner listener only: a profile-scoped key reached through
``/p/<profile>/`` must not open the launch profile's allowlisted channels.
"""
from __future__ import annotations

from typing import Any


async def handle_relay(adapter: Any, request: Any) -> Any:
    from aiohttp import web

    from gateway.platforms.api_server import _api_request_profile, _error_response
    from gateway.session_relay import RelayError, relay

    auth_err = adapter._check_auth(request)
    if auth_err:
        return auth_err
    profile = _api_request_profile.get()
    if profile and profile != "default":
        return _error_response(
            "session relay is served on the owner listener only", 403, code="relay_forbidden")
    body, err = await adapter._read_json_body(request)
    if err:
        return err
    runner = adapter.gateway_runner or request.app.get("gateway_runner")
    if runner is None:
        return _error_response("gateway runner unavailable", 503, code="relay_unavailable")
    try:
        result = await relay(runner, body)
    except RelayError as exc:
        return _error_response(str(exc), exc.status, code=exc.code)
    return web.json_response(result)


def http_routes(adapter: Any) -> list[tuple[str, str, Any]]:
    """Route rows for ``APIServerAdapter._http_route_table``."""

    async def _handler(request: Any) -> Any:
        return await handle_relay(adapter, request)

    return [("POST", "/api/relay", _handler)]
