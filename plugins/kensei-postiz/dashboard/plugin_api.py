"""Backend proxy for the kensei-postiz dashboard plugin.

Forwards requests from the dashboard origin (:9119) to the local Postiz
instance on :4007, attaching the cached auth cookie. Single-user assumption:
one Postiz session is held in module-level state.

Mounted at /api/plugins/kensei-postiz/ when the dashboard starts. Plugin API
routes bypass session-token auth (per the upstream plugin contract), so this
file should only be deployed on a localhost-only dashboard.
"""
from __future__ import annotations

from typing import Any, Optional

import httpx
from fastapi import APIRouter, HTTPException, Request, Response
from pydantic import BaseModel

router = APIRouter()

POSTIZ_BASE = "http://127.0.0.1:4007"
_TIMEOUT = 30.0

# Module-level session cache. The Postiz cookie is captured on login and
# reattached to every subsequent proxied request.
_session: dict[str, Optional[str]] = {"cookie": None}


def _build_cookie_header() -> dict[str, str]:
    if _session["cookie"]:
        return {"Cookie": _session["cookie"]}
    return {}


class LoginBody(BaseModel):
    email: str
    password: str
    provider: str = "LOCAL"


@router.post("/login")
async def login(body: LoginBody) -> dict[str, Any]:
    """Authenticate against Postiz and cache the resulting cookie."""
    payload = body.model_dump()
    try:
        async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
            r = await client.post(f"{POSTIZ_BASE}/api/auth/login", json=payload)
    except httpx.RequestError as exc:
        raise HTTPException(502, f"postiz unreachable: {exc!s}")

    if r.status_code >= 500:
        raise HTTPException(502, f"postiz {r.status_code}: {r.text[:300]}")
    if r.status_code >= 400:
        # Postiz returns the validation reason as plaintext or JSON in the body.
        detail = r.text.strip() or f"postiz {r.status_code}"
        raise HTTPException(401, detail[:500])

    # Postiz scopes its auth cookie to FRONTEND_URL's host (e.g. the Tailscale
    # IP), so httpx.r.cookies.items() drops it when we hit 127.0.0.1. Parse the
    # raw Set-Cookie headers so we keep the JWT regardless of domain scope.
    cookie_parts: list[str] = []
    for raw in r.headers.get_list("set-cookie"):
        kv = raw.split(";", 1)[0].strip()
        if "=" in kv:
            cookie_parts.append(kv)
    if cookie_parts:
        _session["cookie"] = "; ".join(cookie_parts)

    try:
        data = r.json()
    except Exception:
        data = {"ok": True}
    return {"ok": True, "data": data}


@router.get("/probe")
async def probe() -> dict[str, Any]:
    """Quick health check — does Postiz answer + is a session cached?"""
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            r = await client.get(f"{POSTIZ_BASE}/")
        reachable = r.status_code < 500
    except httpx.RequestError:
        reachable = False
    return {"ok": reachable, "authenticated": _session["cookie"] is not None}


async def _proxy(method: str, path: str, request: Request) -> Response:
    """Generic proxy: forward method + body to Postiz with cached cookie."""
    headers = _build_cookie_header()
    headers["Accept"] = request.headers.get("accept", "application/json")
    body_bytes = await request.body()
    if body_bytes:
        headers["Content-Type"] = request.headers.get("content-type", "application/json")

    target = f"{POSTIZ_BASE}/api/{path.lstrip('/')}"
    if request.url.query:
        target = f"{target}?{request.url.query}"

    try:
        async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
            r = await client.request(method, target, headers=headers, content=body_bytes)
    except httpx.RequestError as exc:
        raise HTTPException(502, f"postiz unreachable: {exc!s}")

    media_type = r.headers.get("content-type", "application/json")
    return Response(content=r.content, status_code=r.status_code, media_type=media_type)


@router.get("/posts")
async def list_posts(request: Request) -> Response:
    # Postiz `/api/posts` is the calendar endpoint and requires startDate +
    # endDate query params. The list panel just wants recent posts, so route
    # through `/api/posts/list` (paginated) instead.
    return await _proxy("GET", "posts/list", request)


@router.get("/posts/{post_id}")
async def get_post(post_id: str, request: Request) -> Response:
    return await _proxy("GET", f"posts/{post_id}", request)


@router.post("/posts")
async def create_post(request: Request) -> Response:
    # The Postiz /api/posts POST endpoint is known to reject every payload
    # shape we've tried with 'All posts must have an integration id'. We pass
    # the request through anyway so the user can see the upstream response;
    # the original adapter has a DB-fallback path for create that we're not
    # replicating here.
    return await _proxy("POST", "posts", request)


@router.delete("/posts/{post_id}")
async def delete_post(post_id: str, request: Request) -> Response:
    return await _proxy("DELETE", f"posts/{post_id}", request)


@router.get("/integrations")
async def list_integrations(request: Request) -> Response:
    return await _proxy("GET", "integrations/list", request)


@router.get("/analytics/platforms")
async def platform_analytics(request: Request) -> Response:
    return await _proxy("GET", "analytics/platforms", request)


@router.get("/analytics/posts/{post_id}")
async def post_analytics(post_id: str, request: Request) -> Response:
    return await _proxy("GET", f"analytics/posts/{post_id}", request)
