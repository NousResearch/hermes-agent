"""api_server's aiohttp middleware stack — FILE_LINES sibling of the api_server facade.

``build_middlewares`` imports aiohttp at call time, not import time, so it can be re-run after
the ``messaging`` extra's lazy installer (#136990) heals an environment the facade was imported
into bare: ``ensure_api_server_requirements`` rebinds ``web`` and re-runs this builder, and the
adapter comes up with the full middleware stack instead of the None placeholders the import-time
path leaves behind.
"""

_SECURITY_HEADERS = {
    "Content-Security-Policy": "default-src 'none'; frame-ancestors 'none'",
    "Permissions-Policy": "camera=(), microphone=(), geolocation=()",
    "Strict-Transport-Security": "max-age=31536000; includeSubDomains",
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "DENY",
    "X-XSS-Protection": "0",
    "Referrer-Policy": "no-referrer"}


def build_middlewares(error_response, max_request_bytes: int):
    """Return ``(cors, body_limit, security_headers)`` aiohttp middlewares, all ``None`` when
    aiohttp is not importable. ``error_response`` is injected so this sibling never imports the
    facade (no import cycle) while the wire format stays defined in exactly one place."""
    try:
        from aiohttp import web
    except ImportError:
        return None, None, None

    @web.middleware
    async def cors_middleware(request, handler):
        """Add CORS headers for explicitly allowed origins; handle OPTIONS preflight."""
        adapter = request.app.get("api_server_adapter")
        origin = request.headers.get("Origin", "")
        cors_headers = None
        if adapter is not None:
            if not adapter._origin_allowed(origin):
                return web.Response(status=403)
            cors_headers = adapter._cors_headers_for_origin(origin)
        if request.method == "OPTIONS":
            if cors_headers is None:
                return web.Response(status=403)
            return web.Response(status=200, headers=cors_headers)
        response = await handler(request)
        if cors_headers is not None:
            response.headers.update(cors_headers)
        return response

    @web.middleware
    async def body_limit_middleware(request, handler):
        """Reject overly large request bodies early based on Content-Length."""
        if request.method in {"POST", "PUT", "PATCH"}:
            cl = request.headers.get("Content-Length")
            if cl is not None:
                try:
                    if int(cl) > max_request_bytes:
                        return error_response("Request body too large.", 413, code="body_too_large")
                except ValueError:
                    return error_response("Invalid Content-Length header.", 400, code="invalid_content_length")
        try:
            return await handler(request)
        except web.HTTPRequestEntityTooLarge:
            # client_max_size tripped mid-read (chunked bodies carry no Content-Length): a
            # proper 413, not the handler's 400 "Invalid JSON".
            return error_response("Request body too large.", 413, code="body_too_large")

    @web.middleware
    async def security_headers_middleware(request, handler):
        """Add security headers to all responses (including errors)."""
        response = await handler(request)
        for k, v in _SECURITY_HEADERS.items():
            response.headers.setdefault(k, v)
        return response

    return cors_middleware, body_limit_middleware, security_headers_middleware
