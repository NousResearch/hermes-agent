"""
DEXAI Proxy - Forward /dex and /DEXAI requests to the DEXAI Node.js server on port 3847.

Usage in server.py:
    from Cosmos.dex.dex_proxy import register_dex_routes
    register_dex_routes(app)
"""

import logging
import httpx

logger = logging.getLogger("cosmos.dex.proxy")

DEXAI_URL = "http://localhost:3847"


def register_dex_routes(app):
    """Register DEX proxy routes on the FastAPI app."""
    from fastapi import Request
    from fastapi.responses import Response, JSONResponse, RedirectResponse

    async def _proxy(path: str, request: Request):
        """Proxy a request to the DEXAI Node.js server."""
        async with httpx.AsyncClient(timeout=30.0) as client:
            url = f"{DEXAI_URL}/{path}"
            if request.query_params:
                url += f"?{request.query_params}"
            try:
                if request.method == "GET":
                    resp = await client.get(url)
                else:
                    body = await request.body()
                    resp = await client.request(
                        method=request.method,
                        url=url,
                        content=body,
                        headers={"Content-Type": request.headers.get("content-type", "application/json")}
                    )
                return Response(
                    content=resp.content,
                    status_code=resp.status_code,
                    media_type=resp.headers.get("content-type", "text/html"),
                )
            except Exception as e:
                logger.error(f"DEXAI proxy error: {e}")
                return JSONResponse({"error": "DEXAI unavailable"}, status_code=502)

    async def _proxy_home():
        """Proxy the DEXAI home page."""
        async with httpx.AsyncClient(timeout=30.0) as client:
            try:
                resp = await client.get(f"{DEXAI_URL}/")
                return Response(
                    content=resp.content,
                    status_code=resp.status_code,
                    media_type=resp.headers.get("content-type", "text/html"),
                )
            except Exception as e:
                logger.error(f"DEXAI proxy error: {e}")
                return JSONResponse({"error": "DEXAI unavailable"}, status_code=502)

    # Register routes for both /dex and /DEXAI paths
    # Redirect bare /dex to /dex/ so relative asset paths (styles.css, app.js) resolve correctly
    @app.get("/dex")
    async def dex_redirect():
        return RedirectResponse(url="/dex/", status_code=301)

    @app.get("/DEXAI")
    async def dexai_redirect():
        return RedirectResponse(url="/DEXAI/", status_code=301)

    @app.get("/dex/")
    async def dex_home():
        return await _proxy_home()

    @app.get("/DEXAI/")
    async def dexai_home():
        return await _proxy_home()

    @app.api_route("/dex/{path:path}", methods=["GET", "POST", "PUT", "DELETE"])
    async def dex_proxy(path: str, request: Request):
        return await _proxy(path, request)

    @app.api_route("/DEXAI/{path:path}", methods=["GET", "POST", "PUT", "DELETE"])
    async def dexai_proxy(path: str, request: Request):
        return await _proxy(path, request)

    logger.info("DEXAI proxy registered at /dex and /DEXAI")
