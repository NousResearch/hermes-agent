"""Cloud peer access: ``/api/peer/...`` carries another Hermes agent's API calls to this agent's
gateway api_server on loopback, so ``hermes peer`` can reach a hosted agent without its key.

The bearer is a short-lived pass minted for this one agent, verified before this route runs
(``hermes_cli.dashboard_auth.peer_pass``). Each path in :data:`_ROUTES` needs one permission; a path
not listed is never forwarded, which keeps machine plumbing (``/api/cron/fire``, platform event
intake, RoomLink grants, ``/health/detailed``) off this route. The request goes on with the
API_SERVER_KEY the listener checks, which never leaves the machine, and with the pass holder's id
in ``X-Hermes-Peer-Principal`` so jobs they create stay attributable. Control actions are written to
the dashboard-auth audit log. ``/api/peer/p/<profile>/...`` reaches one profile.
"""
from __future__ import annotations

import asyncio
import http.cookiejar
import logging
import urllib.parse
from typing import Optional

import httpx
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from starlette.background import BackgroundTask

from gateway.platforms.api_server_profile_prefix import DEFAULT_PROFILE_HANDLE, UNKNOWN_PROFILE_ERROR
from hermes_cli.dashboard_auth.audit import AuditEvent, audit_log
from hermes_cli.dashboard_auth.peer_pass import PEER_PREFIX
from hermes_cli.dashboard_auth.request_utils import client_ip

_log = logging.getLogger(__name__)
router = APIRouter()

#: Messaging: everything ``hermes peer`` sends today, plus run events.
MESSAGE = "agents:message"
#: Control: what a user controls through the agent's API. Includes messaging.
CONTROL = "agents:control"

# (method, path with ``*`` for one id segment) -> permission.
_ROUTES: dict[tuple[str, str], str] = {
    **{route: MESSAGE for route in (
        ("GET", "health"), ("GET", "v1/health"), ("GET", "v1/capabilities"),
        ("GET", "api/sessions"), ("POST", "api/sessions"), ("GET", "api/sessions/*"),
        ("POST", "api/sessions/*/chat"),
        ("POST", "v1/runs"), ("GET", "v1/runs/*"), ("GET", "v1/runs/*/events"), ("POST", "v1/runs/*/stop"))},
    **{route: CONTROL for route in (
        ("POST", "v1/runs/*/steer"), ("POST", "v1/runs/*/approval"), ("POST", "v1/runs/*/clarify"),
        ("POST", "v1/runs/*/resolve-unknown"),
        ("PATCH", "api/sessions/*"), ("DELETE", "api/sessions/*"), ("GET", "api/sessions/*/messages"),
        ("POST", "api/sessions/*/fork"), ("POST", "api/sessions/*/chat/stream"), ("POST", "api/sessions/*/model"),
        ("GET", "api/jobs"), ("POST", "api/jobs"), ("GET", "api/jobs/*"), ("PATCH", "api/jobs/*"),
        ("DELETE", "api/jobs/*"), ("POST", "api/jobs/*/pause"), ("POST", "api/jobs/*/resume"),
        ("POST", "api/jobs/*/run"),
        ("GET", "api/model/options"), ("GET", "v1/models"), ("GET", "v1/skills"), ("GET", "v1/toolsets"),
        ("POST", "v1/chat/completions"), ("POST", "v1/responses"), ("GET", "v1/responses/*"),
        ("DELETE", "v1/responses/*"))},
}
_METHODS = sorted({method for method, _ in _ROUTES})

_HOP_BY_HOP = frozenset({"connection", "keep-alive", "proxy-authenticate", "proxy-authorization", "te",
                         "trailer", "transfer-encoding", "upgrade"})
#: Request headers that never reach the gateway: the caller's credentials and provenance (the pass,
#: cookies, browser origin). The agent's key and the verified pass holder replace them.
_DROP_REQUEST = _HOP_BY_HOP | frozenset({
    "host", "cookie", "authorization", "origin", "referer", "forwarded", "x-real-ip",
    "x-hermes-session-token", "x-hermes-gateway-ticket", "x-hermes-window-ticket"})
#: Picks the api_server's memory scope, which it keeps for key holders: only a control pass, which
#: has key-holder parity, may set it.
_CONTROL_ONLY_REQUEST = frozenset({"x-hermes-session-key"})
#: Response headers that come back: what describes the body, plus the api_server's own signals
#: (session, replayed admission, partial or failed turn).
_KEEP_RESPONSE = frozenset({"content-type", "content-length", "content-encoding", "cache-control",
                            "x-accel-buffering", "retry-after", "x-hermes-session-id", "x-hermes-session-key",
                            "idempotency-replayed", "x-hermes-completed", "x-hermes-partial", "x-hermes-error"})
#: Failures before the request reached the gateway (asleep, restarting, off): safe to retry.
_NOT_SENT = (httpx.ConnectError, httpx.ConnectTimeout, httpx.PoolTimeout)


def _permission(method: str, segments: list[str]) -> Optional[str]:
    for (route_method, template), permission in _ROUTES.items():
        parts = template.split("/")
        if route_method == method and len(parts) == len(segments) and all(
                part in ("*", segment) for part, segment in zip(parts, segments)):
            return permission
    return None


def _parse(scope) -> Optional[tuple[Optional[str], list[str]]]:
    """``(profile or None, decoded path segments)`` after the prefix, or None for a path this route
    never carries: an empty, ``.`` or ``..`` segment (however encoded) or a control character."""
    raw = (scope.get("raw_path") or scope.get("path", "").encode("utf-8")).decode("latin-1")
    index = raw.find(PEER_PREFIX)
    if index < 0:
        return None
    segments = [urllib.parse.unquote(part) for part in raw[index + len(PEER_PREFIX):].split("/")]
    if any(not s or s in (".", "..") or any(ord(c) < 0x20 or ord(c) == 0x7F for c in s) for s in segments):
        return None
    if len(segments) > 2 and segments[0] == "p":
        return segments[1], segments[2:]
    return None, segments


def _upstream(profile: Optional[str]) -> Optional[tuple[str, str]]:
    """``(loopback base URL, API_SERVER_KEY)`` of the gateway api_server serving *profile* (None: the
    profile this dashboard serves), or None when this host has no such profile. The prefix reads as
    the api_server reads it: the default profile's handle ``hermes`` is the default profile unless a
    profile has that name. Reads config and .env files: call it off the event loop."""
    from hermes_cli import profiles
    from hermes_cli import web_server_cron as cron
    if profile is None:
        name, home = cron._cron_profile_home(None)
        return cron._gateway_api_target(name, home)
    name = profiles.normalize_profile_name(profile)
    if name == DEFAULT_PROFILE_HANDLE and not profiles.profile_exists(name):
        name = "default"
    if not profiles.profile_exists(name):
        return None
    return cron._gateway_api_target(name, profiles.get_profile_dir(name))


def _gateway_client(state) -> httpx.AsyncClient:
    """One client per event loop: no redirects, no proxy settings from the environment, no read
    deadline (run events and chat streams last as long as the turn), and no cookie jar, since every
    pass holder's requests share it."""
    loop = asyncio.get_running_loop()
    held = getattr(state, "peer_gateway_client", None)
    if held is not None and held[0] is loop:
        return held[1]
    no_cookies = http.cookiejar.CookieJar(policy=http.cookiejar.DefaultCookiePolicy(allowed_domains=[]))
    client = httpx.AsyncClient(follow_redirects=False, trust_env=False, cookies=no_cookies,
                               timeout=httpx.Timeout(connect=10.0, read=None, write=60.0, pool=10.0))
    state.peer_gateway_client = (loop, client)
    return client


def _refusal(error: str, detail: str, status: int, **headers: str) -> JSONResponse:
    return JSONResponse({"error": error, "detail": detail}, status_code=status, headers=headers or None)


@router.api_route(PEER_PREFIX + "{path:path}", methods=_METHODS, include_in_schema=False)
async def peer_forward(request: Request, path: str):
    principal = getattr(request.state, "peer_principal", None)
    if principal is None:
        raise HTTPException(401, "A peer pass for this agent is required.")
    parsed = _parse(request.scope)
    needed = _permission(request.method, parsed[1]) if parsed else None
    if parsed is None or needed is None:
        return _refusal("not_found", "This agent's peer route does not carry that path.", 404)
    profile, segments = parsed
    if needed not in principal.scopes and CONTROL not in principal.scopes:
        audit_log(AuditEvent.TOKEN_AUTH_FAILURE, reason="insufficient_scope", principal=principal.principal,
                  method=request.method, path=request.url.path, ip=client_ip(request))
        return _refusal("insufficient_scope", f"This pass does not grant {needed}.", 403)

    upstream_target = await asyncio.to_thread(_upstream, profile)
    if upstream_target is None:
        return JSONResponse({"error": UNKNOWN_PROFILE_ERROR}, status_code=404)
    base, key = upstream_target
    if not key:
        return _refusal("peer_gateway_unavailable", "This agent's API server has no key configured.", 503)
    target = "/" + "/".join(urllib.parse.quote(segment, safe="") for segment in segments)
    query = (request.scope.get("query_string") or b"").decode("latin-1")
    headers = {name: value for name, value in request.headers.items()
               if name.lower() not in _DROP_REQUEST
               and not name.lower().startswith(("x-forwarded-", "x-hermes-peer-"))
               and (name.lower() not in _CONTROL_ONLY_REQUEST or CONTROL in principal.scopes)}
    headers.update({"Authorization": f"Bearer {key}", "X-Hermes-Peer-Principal": principal.principal})
    if client_ip(request):
        headers["X-Forwarded-For"] = client_ip(request)
    method = request.method
    has_body = method not in {"GET", "HEAD"} or request.headers.get("content-length") not in (None, "0")
    client = _gateway_client(request.app.state)
    upstream = client.build_request(method, f"{base}{target}" + (f"?{query}" if query else ""),
                                    headers=headers, content=request.stream() if has_body else None)
    response: Optional[httpx.Response] = None
    try:
        response = await client.send(upstream, stream=True)
    except _NOT_SENT as exc:
        _log.info("peer route: gateway api_server at %s unreachable: %s", base, exc)
        return _refusal("peer_gateway_unavailable", "This agent's API server is not reachable yet.", 503,
                        **{"Retry-After": "5"})
    except httpx.HTTPError as exc:  # sent, then the connection broke: the action may have run
        _log.warning("peer route: gateway api_server at %s dropped %s %s: %s", base, method, target, exc)
    if response is not None and response.status_code == 401:
        await response.aclose()
        return _refusal("peer_gateway_auth", "This agent's API server refused the agent's own key.", 502)
    if needed == CONTROL:
        audit_log(AuditEvent.PEER_CONTROL, principal=principal.principal, provider=principal.provider,
                  method=method, path=target, profile=profile or "",
                  status=response.status_code if response is not None else 502, ip=client_ip(request))
    if response is None:
        return _refusal("peer_gateway_dropped",
                        "This agent's API server dropped the request before answering; it may have run.", 502)
    out = {name: value for name, value in response.headers.items() if name.lower() in _KEEP_RESPONSE}
    return StreamingResponse(response.aiter_raw(), status_code=response.status_code, headers=out,
                             background=BackgroundTask(response.aclose))
