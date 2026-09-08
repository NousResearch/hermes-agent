"""Opt-in public dashboard routes for the gateway's A2A and API listeners.

Hosted deployments terminate TLS at the dashboard hostname while the gateway's
protocol listeners remain on loopback.  These routes bridge that boundary without
sharing dashboard cookies: each upstream listener keeps ownership of its bearer
authentication.
"""
from __future__ import annotations

import os
import urllib.parse
from dataclasses import dataclass
from typing import AsyncIterator

import httpx
from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, StreamingResponse

from hermes_cli.config import load_config
from hermes_cli.dashboard_auth.prefix import prefix_from_request, resolve_public_url

router = APIRouter()

_A2A_PREFIX = "/a2a"
_API_PREFIX = "/hermes-api"
_A2A_MAX_BODY_BYTES = 1024 * 1024
_ALLOWED_METHODS = ["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS", "HEAD"]
_HOP_BY_HOP_HEADERS = frozenset({
    "connection", "keep-alive", "proxy-authenticate", "proxy-authorization",
    "te", "trailer", "transfer-encoding", "upgrade",
})
_PRIVATE_REQUEST_HEADERS = frozenset({
    "cookie", "host", "content-length", "x-hermes-session-token",
    "forwarded", "x-forwarded-for", "x-forwarded-host", "x-forwarded-prefix",
    "x-forwarded-proto",
})
_PRIVATE_RESPONSE_HEADERS = _HOP_BY_HOP_HEADERS | frozenset({"content-length", "set-cookie"})


@dataclass(frozen=True)
class _ServiceRoute:
    name: str
    prefix: str
    port: int


def _merge_platform_block(config: dict, name: str) -> dict:
    """Mirror gateway platform-block precedence for the two fields used here."""
    gateway = config.get("gateway") if isinstance(config.get("gateway"), dict) else {}
    merged: dict = {}
    merged_extra: dict = {}
    sources = (
        (gateway.get("platforms") or {}).get(name) if isinstance(gateway.get("platforms"), dict) else None,
        (config.get("platforms") or {}).get(name) if isinstance(config.get("platforms"), dict) else None,
        gateway.get(name),
    )
    for block in sources:
        if not isinstance(block, dict):
            continue
        merged.update(block)
        if isinstance(block.get("extra"), dict):
            merged_extra.update(block["extra"])
    if merged_extra:
        merged["extra"] = merged_extra
    return merged


def _service_route(name: str) -> _ServiceRoute | None:
    specs = {
        "a2a": (_A2A_PREFIX, "A2A_PORT", 9900),
        "api_server": (_API_PREFIX, "API_SERVER_PORT", 8642),
    }
    prefix, port_env, default_port = specs[name]
    config = load_config() or {}
    block = _merge_platform_block(config, name)
    extra = block.get("extra") if isinstance(block.get("extra"), dict) else {}
    if block.get("enabled") is not True or extra.get("public_route") is not True:
        return None
    if name == "a2a" and not (
        os.environ.get("A2A_BEARER_TOKEN", "").strip()
        or os.environ.get("A2A_PEER_TOKENS", "").strip()
    ):
        return None
    raw_port = os.environ.get(port_env, "").strip() or extra.get("port", default_port)
    try:
        port = int(raw_port)
    except (TypeError, ValueError):
        return None
    if not 1 <= port <= 65535:
        return None
    return _ServiceRoute(name=name, prefix=prefix, port=port)


def _forwarded_headers(request: Request, route: _ServiceRoute) -> dict[str, str]:
    headers = {
        key: value for key, value in request.headers.items()
        if key.lower() not in _HOP_BY_HOP_HEADERS | _PRIVATE_REQUEST_HEADERS
    }
    public_url = resolve_public_url()
    parsed = urllib.parse.urlparse(public_url) if public_url else None
    public_host = parsed.netloc if parsed and parsed.netloc else request.headers.get("host", "")
    public_scheme = parsed.scheme if parsed and parsed.scheme else request.url.scheme
    base_prefix = parsed.path.rstrip("/") if parsed and parsed.path else prefix_from_request(request)
    headers["host"] = f"127.0.0.1:{route.port}"
    headers["x-forwarded-host"] = public_host
    headers["x-forwarded-proto"] = public_scheme
    headers["x-forwarded-prefix"] = f"{base_prefix}{route.prefix}"
    return headers


async def _stream_upstream(
    response: httpx.Response, client: httpx.AsyncClient
) -> AsyncIterator[bytes]:
    try:
        async for chunk in response.aiter_raw():
            yield chunk
    finally:
        await response.aclose()
        await client.aclose()


async def _proxy(request: Request, service: str, path: str) -> StreamingResponse | JSONResponse:
    route = _service_route(service)
    if route is None:
        return JSONResponse({"error": "service route is not enabled"}, status_code=404)

    suffix = f"/{path}" if path else "/"
    query = request.url.query
    upstream_url = f"http://127.0.0.1:{route.port}{suffix}" + (f"?{query}" if query else "")
    client = httpx.AsyncClient(
        trust_env=False,
        timeout=httpx.Timeout(connect=5.0, read=None, write=30.0, pool=5.0),
    )
    try:
        if service == "a2a":
            body = await request.body()
            if len(body) > _A2A_MAX_BODY_BYTES:
                await client.aclose()
                return JSONResponse({"error": "request body too large"}, status_code=413)
            content: bytes | AsyncIterator[bytes] = body
        elif request.method in {"GET", "HEAD", "OPTIONS"}:
            content = b""
        else:
            content = request.stream()
        upstream_request = client.build_request(
            request.method,
            upstream_url,
            headers=_forwarded_headers(request, route),
            content=content,
        )
        upstream = await client.send(upstream_request, stream=True)
    except (httpx.HTTPError, OSError):
        await client.aclose()
        return JSONResponse({"error": f"{service} listener is unavailable"}, status_code=503)

    response_headers = {
        key: value for key, value in upstream.headers.items()
        if key.lower() not in _PRIVATE_RESPONSE_HEADERS
    }
    return StreamingResponse(
        _stream_upstream(upstream, client),
        status_code=upstream.status_code,
        headers=response_headers,
    )


@router.api_route(_A2A_PREFIX, methods=_ALLOWED_METHODS)
@router.api_route(f"{_A2A_PREFIX}/{{path:path}}", methods=_ALLOWED_METHODS)
async def proxy_a2a(request: Request, path: str = ""):
    return await _proxy(request, "a2a", path)


@router.api_route(_API_PREFIX, methods=_ALLOWED_METHODS)
@router.api_route(f"{_API_PREFIX}/{{path:path}}", methods=_ALLOWED_METHODS)
async def proxy_api_server(request: Request, path: str = ""):
    return await _proxy(request, "api_server", path)
