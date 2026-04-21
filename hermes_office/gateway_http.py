"""Mount Digital Office on the aiohttp Gateway API server (user-facing HTTP).

Registers on the same host/port as the OpenAI-compatible API (default 8642):

- ``GET /api/health`` — ``{"status": "ok"}`` (probe for bundled UIs)
- ``/api/office/...`` — sub-app forwarding to the FastAPI office ASGI app
- ``GET /ws/office`` — WebSocket activity stream
- ``/office/`` — built Vite SPA
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from aiohttp import web

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def get_office_app():
    from hermes_office.server import build_app

    return build_app()


def office_dist_dir() -> Path | None:
    here = Path(__file__).resolve().parent
    dist = here / "frontend" / "dist"
    if (dist / "index.html").exists():
        return dist
    return None


async def handle_classic_health(request: "web.Request") -> "web.StreamResponse":
    from aiohttp import web

    return web.json_response({"status": "ok"})


async def _office_sub_forward(request: "web.Request") -> "web.StreamResponse":
    """Strip ``/api/office`` prefix (aiohttp subapps still expose the full URL path)."""
    from aiohttp import web

    try:
        import httpx
    except ImportError:
        return web.Response(status=501, text="httpx required for Digital Office API bridge")

    prefix = "/api/office"
    path = request.path
    if not path.startswith(prefix):
        return web.Response(status=404, text="Not found")
    tail = path[len(prefix) :] or "/"
    if not tail.startswith("/"):
        tail = "/" + tail
    inner = "/api" + tail if tail != "/" else "/api"
    if inner == "/api" and request.method == "GET":
        return web.Response(status=404, text="Not found")

    body = await request.read()
    hdrs = {
        k: v
        for k, v in request.headers.items()
        if k.lower()
        not in (
            "host",
            "content-length",
            "connection",
            "transfer-encoding",
        )
    }
    transport = httpx.ASGITransport(app=get_office_app(), raise_app_exceptions=False)
    try:
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://office.internal",
            timeout=120.0,
        ) as client:
            resp = await client.request(
                request.method,
                inner,
                content=body if body else None,
                headers=hdrs,
                params=request.rel_url.query,
            )
    except Exception as exc:  # noqa: BLE001
        logger.exception("Office ASGI bridge failed: %s", exc)
        return web.Response(status=502, text="Digital Office bridge error")

    skip = {"transfer-encoding", "connection"}
    headers = {k: v for k, v in resp.headers.items() if k.lower() not in skip}
    return web.Response(status=resp.status_code, body=resp.content, headers=headers)


async def handle_office_ws(request: "web.Request") -> "web.WebSocketResponse":
    from aiohttp import web

    ws = web.WebSocketResponse()
    await ws.prepare(request)

    origin = request.headers.get("Origin", "")
    if origin and not origin.startswith(("http://127.0.0.1", "http://localhost")):
        await ws.close(code=1008, message=b"origin not allowed")
        return ws

    app = get_office_app()
    bus = app.state.bus
    import hermes_office as _ho

    version = getattr(_ho, "__version__", "0.0.0")

    async def _drain_client() -> None:
        try:
            async for msg in ws:
                if msg.type == web.WSMsgType.CLOSE:
                    break
        except Exception:
            pass

    drain = asyncio.create_task(_drain_client())
    q = await bus.subscribe()
    try:
        await ws.send_str(json.dumps({"kind": "hello", "version": version}))
        while True:
            evt = await q.get()
            await ws.send_str(evt.model_dump_json())
    except (ConnectionResetError, asyncio.CancelledError):
        pass
    except Exception as exc:  # noqa: BLE001
        logger.debug("office ws closed: %s", exc)
    finally:
        drain.cancel()
        with contextlib.suppress(Exception):
            await drain
        await bus.unsubscribe(q)
        with contextlib.suppress(Exception):
            await ws.close()
    return ws


async def handle_office_root_redirect(request: "web.Request") -> "web.StreamResponse":
    from aiohttp import web

    raise web.HTTPFound("/office/")


async def handle_office_spa(request: "web.Request") -> "web.StreamResponse":
    from aiohttp import web

    dist = office_dist_dir()
    if dist is None:
        return web.Response(
            status=503,
            text="Digital Office frontend not built (cd hermes_office/frontend && npm run build)",
        )
    rel = request.match_info.get("path", "").lstrip("/")
    if rel.startswith("assets/"):
        f = dist / rel
        try:
            if f.resolve().is_relative_to(dist.resolve()) and f.is_file():
                return web.FileResponse(f)
        except (OSError, ValueError):
            pass
    return web.FileResponse(dist / "index.html")


def register_digital_office_routes(app: "web.Application") -> bool:
    """Register routes. Returns False if office cannot start."""
    try:
        from aiohttp import web
    except ImportError:
        return False

    try:
        get_office_app()
    except Exception as exc:  # noqa: BLE001
        logger.warning("Digital Office disabled (init): %s", exc)
        return False

    office_api = web.Application()
    office_api.router.add_route("*", "/{path:.*}", _office_sub_forward)
    app.add_subapp("/api/office", office_api)

    app.router.add_get("/api/health", handle_classic_health)
    app.router.add_get("/ws/office", handle_office_ws)
    app.router.add_get("/office", handle_office_root_redirect)
    app.router.add_get("/office/", handle_office_spa)
    app.router.add_route("*", "/office/{path:.*}", handle_office_spa)
    logger.info("Digital Office routes registered (/office/, /api/office/, /ws/office)")
    return True
