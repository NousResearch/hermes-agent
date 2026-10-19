"""aiohttp middleware answering a config-backend outage instead of letting it end the gateway.

``ConfigBackendUnavailable`` is a ``SystemExit`` on purpose (config readers must not serve
defaults instead of the selected backend's config). Escaping an aiohttp handler task, a
``SystemExit`` stops the event loop, so a remote-config plane blip during one request took the
whole gateway down. This middleware turns it into that request's 503.
"""

from __future__ import annotations

from aiohttp import web

from hermes_cli.config_backend import ConfigBackendUnavailable


@web.middleware
async def config_unavailable_middleware(request: web.Request, handler):
    try:
        return await handler(request)
    except ConfigBackendUnavailable as exc:
        return web.json_response(
            {"error": {"message": str(exc.code), "type": "server_error", "code": "config_backend_unavailable"}},
            status=503)
