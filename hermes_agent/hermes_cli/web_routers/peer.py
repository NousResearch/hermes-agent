"""Cloud peer access: ``/api/peer/...`` carries another Hermes agent's API calls to this agent's
gateway api_server on loopback, so ``hermes peer`` can reach a hosted agent without its key.

The bearer is a short-lived pass minted for this one agent, verified before this route runs
(``hermes_cli.dashboard_auth.peer_pass``). Each path in :data:`_ROUTES` needs one permission; a path
not listed is never forwarded, which keeps machine plumbing (``/api/cron/fire``, platform event
intake, RoomLink grants, ``/health/detailed``) off this route. The request goes on with the
API_SERVER_KEY the listener checks, which never leaves the machine, and with the pass holder's id
in ``X-Hermes-Peer-Principal`` so jobs they create stay attributable. Every turn is credited to the
pass holder. A messaging pass reaches only the agent's canonical Bot Chat, the one session every
peer message lands in; a control pass has key-holder parity. Control actions are written to the
dashboard-auth audit log. ``/api/peer/p/<profile>/...`` reaches one profile.
"""
from __future__ import annotations

import asyncio
import http.cookiejar
import json
import logging
import urllib.parse
from typing import Optional

import httpx
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from starlette.background import BackgroundTask

from hermes_agent.gateway.platforms.api_server_profile_prefix import DEFAULT_PROFILE_HANDLE, UNKNOWN_PROFILE_ERROR
from hermes_cli.dashboard_auth.audit import AuditEvent, audit_log
from hermes_cli.dashboard_auth.peer_pass import PEER_PREFIX
from hermes_cli.dashboard_auth.request_utils import client_ip

_log = logging.getLogger(__name__)
router = APIRouter()

#: Messaging: everything ``hermes peer`` sends today, plus run events and the streaming chat turn,
#: in the agent's Bot Chat only. Each turn runs with the agent's tools, terminal included.
MESSAGE = "agents:message"
#: Control: what a user controls through the agent's API, in any session. Includes messaging.
CONTROL = "agents:control"

# (method, path with ``*`` for one id segment) -> permission.
_ROUTES: dict[tuple[str, str], str] = {
    **{route: MESSAGE for route in (
        ("GET", "health"), ("GET", "v1/health"), ("GET", "v1/capabilities"),
        ("GET", "api/sessions"), ("POST", "api/sessions"), ("GET", "api/sessions/*"),
        ("POST", "api/sessions/*/chat"), ("POST", "api/sessions/*/chat/stream"),
        ("POST", "v1/runs"), ("GET", "v1/runs/*"), ("GET", "v1/runs/*/events"), ("POST", "v1/runs/*/stop"))},
    **{route: CONTROL for route in (
        ("POST", "v1/runs/*/steer"), ("POST", "v1/runs/*/approval"), ("POST", "v1/runs/*/clarify"),
        ("POST", "v1/runs/*/resolve-unknown"),
        ("PATCH", "api/sessions/*"), ("DELETE", "api/sessions/*"), ("GET", "api/sessions/*/messages"),
        ("POST", "api/sessions/*/fork"), ("POST", "api/sessions/*/model"),
        ("GET", "api/jobs"), ("POST", "api/jobs"), ("GET", "api/jobs/*"), ("PATCH", "api/jobs/*"),
        ("DELETE", "api/jobs/*"), ("POST", "api/jobs/*/pause"), ("POST", "api/jobs/*/resume"),
        ("POST", "api/jobs/*/run"),
        ("GET", "api/model/options"), ("GET", "v1/models"), ("GET", "v1/skills"), ("GET", "v1/toolsets"),
        ("POST", "v1/chat/completions"), ("POST", "v1/responses"), ("GET", "v1/responses/*"),
        ("DELETE", "v1/responses/*"))},
}
_METHODS = sorted({method for method, _ in _ROUTES})
#: The routes whose turns the api_server credits to the body's ``author`` (memory attribution).
_TURNS = frozenset({("POST", "api/sessions/*/chat"), ("POST", "api/sessions/*/chat/stream"), ("POST", "v1/runs")})
#: The body fields a messaging pass may send where this route reads the body: what ``hermes peer``
#: sends (the api_server's own Bot Chat create also takes only ``title`` and ``source``). The
#: api_server takes more there, which change more than the turn's own message: a model, kept for
#: every later caller with ``require_model_lock``; a system prompt; a history; a stored response
#: chain; the new session's settings. Those need ``agents:control``.
_MESSAGING_FIELDS: dict[tuple[str, str], frozenset[str]] = {
    ("POST", "api/sessions"): frozenset({"title", "source"}),
    ("POST", "api/sessions/*/chat"): frozenset({"message", "author"}),
    ("POST", "api/sessions/*/chat/stream"): frozenset({"message", "author"}),
    ("POST", "v1/runs"): frozenset({"input", "session_id", "author"}),
}
#: The api_server's request size limit (``api_server.MAX_REQUEST_BYTES``).
_MAX_BODY_BYTES = 10_000_000

_HOP_BY_HOP = frozenset({"connection", "keep-alive", "proxy-authenticate", "proxy-authorization", "te",
                         "trailer", "transfer-encoding", "upgrade"})
#: Request headers that never reach the gateway: the caller's credentials and provenance (the pass,
#: cookies, browser origin). The agent's key and the verified pass holder replace them.
_DROP_REQUEST = _HOP_BY_HOP | frozenset({
    "host", "cookie", "authorization", "origin", "referer", "forwarded", "x-real-ip",
    "x-hermes-session-token", "x-hermes-gateway-ticket", "x-hermes-window-ticket"})
#: The api_server's memory scope, which it keeps for key holders: only a control pass, which has
#: key-holder parity, may set it or see it.
_CONTROL_ONLY = frozenset({"x-hermes-session-key"})
#: Response headers that come back: what describes the body, plus the api_server's own signals
#: (session, replayed admission, partial or failed turn); the memory scope only to a control pass.
_KEEP_RESPONSE = frozenset({"content-type", "content-length", "content-encoding", "cache-control",
                            "x-accel-buffering", "retry-after", "x-hermes-session-id", "x-hermes-session-key",
                            "idempotency-replayed", "x-hermes-completed", "x-hermes-partial", "x-hermes-error"})
#: Request headers replaced when this route sends a body it read.
_BODY_HEADERS = frozenset({"content-length", "content-type", "content-encoding"})
#: Failures before the request reached the gateway (asleep, restarting, off): safe to retry.
_NOT_SENT = (httpx.ConnectError, httpx.ConnectTimeout, httpx.PoolTimeout)


def _route(method: str, segments: list[str]) -> Optional[tuple[str, str]]:
    """``(route template, permission)`` for *method* and *segments*, or None when not in the table."""
    for (route_method, template), permission in _ROUTES.items():
        parts = template.split("/")
        if route_method == method and len(parts) == len(segments) and all(
                part in ("*", segment) for part, segment in zip(parts, segments)):
            return template, permission
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


def _unreachable(base: str, exc: Exception) -> JSONResponse:
    _log.info("peer route: gateway api_server at %s unreachable: %s", base, exc)
    return _refusal("peer_gateway_unavailable", "This agent's API server is not reachable yet.", 503,
                    **{"Retry-After": "5"})


def _gateway_auth_refused() -> JSONResponse:
    return _refusal("peer_gateway_auth", "This agent's API server refused the agent's own key.", 502)


async def _json_object(request: Request) -> tuple[Optional[dict], Optional[JSONResponse]]:
    """The request body as a JSON object, read once with the api_server's size limit."""
    chunks, size = [], 0
    async for chunk in request.stream():
        size += len(chunk)
        if size > _MAX_BODY_BYTES:
            return None, _refusal("request_too_large", "The request body is too large.", 413)
        chunks.append(chunk)
    try:
        body = json.loads(b"".join(chunks).decode("utf-8"))
    except (UnicodeDecodeError, ValueError):
        body = None
    if not isinstance(body, dict):
        return None, _refusal("invalid_json", "The request body must be a JSON object.", 400)
    return body, None


def _control_only_parts(method: str, template: str, body: dict) -> list[str]:
    """What in a body only a control pass may send: fields ``hermes peer`` never sends, and a turn
    whose message is not plain text (a list ``input`` stands in for the history, and list content
    skips the hand-off to a Desktop holding the Bot Chat)."""
    parts = sorted(set(body) - _MESSAGING_FIELDS[(method, template)])
    text = "input" if template == "v1/runs" else "message"
    if (method, template) in _TURNS and text in body and not isinstance(body[text], str):
        parts.append(f"a non-text {text}")
    return parts


def _holder_author(claimed, principal) -> dict:
    """The turn's author with the verified pass holder as its id, which is what memory attributes
    the turn to, so a pass holder cannot credit a turn to anyone else. The caller's display name,
    bot flag and hop count stay: the bot-loop guard counts hops, and only ever upward."""
    author = {key: claimed[key] for key in ("name", "is_bot", "hops") if isinstance(claimed, dict) and key in claimed}
    return {"id": f"{principal.provider}:{principal.principal}", **author}


def _bot_chat_only() -> JSONResponse:
    return _refusal("bot_chat_only", "A messaging pass reaches only this agent's Bot Chat.", 403)


async def _run_session(client: httpx.AsyncClient, base: str, key: str,
                       run_id: str) -> tuple[str, Optional[JSONResponse]]:
    """``(session id, None)`` of an existing run, from its status on the gateway, or ``("", refusal)``.
    A run with no session (or none the gateway knows) is outside the Bot Chat."""
    try:
        found = await client.get(f"{base}/v1/runs/{urllib.parse.quote(run_id, safe='')}",
                                 headers={"Authorization": f"Bearer {key}"})
    except httpx.HTTPError as exc:  # a read: safe to retry whatever broke
        return "", _unreachable(base, exc)
    if found.status_code == 401:
        return "", _gateway_auth_refused()
    if found.status_code == 404:
        return "", _bot_chat_only()
    if found.status_code != 200:
        return "", _unreachable(base, RuntimeError(f"run lookup answered {found.status_code}"))
    try:
        session_id = found.json().get("session_id")
    except (ValueError, AttributeError):
        session_id = None
    if not isinstance(session_id, str) or not session_id:
        return "", _bot_chat_only()
    return session_id, None


async def _outside_the_bot_chat(client: httpx.AsyncClient, base: str, key: str, template: str,
                                segments: list[str], query: list[tuple[str, str]],
                                body: Optional[dict]) -> Optional[JSONResponse]:
    """The refusal for a messaging pass's request that would reach beyond the agent's canonical Bot
    Chat (other chats' titles and previews, turns in them, a session of its own), or None. A
    session id is checked by its title on the gateway itself, which keeps that title unique."""
    from hermes_agent.tools.bot_mode_probe import BOT_CHAT_TITLE
    if template == "api/sessions" and body is None:  # the listing: only the exact-title lookup
        titles = [value for name, value in query if name == "title"]
        return None if titles and all(title.strip() == BOT_CHAT_TITLE for title in titles) else _bot_chat_only()
    if template == "api/sessions":
        return None if str(body.get("title") or "").strip() == BOT_CHAT_TITLE else _bot_chat_only()
    if template == "v1/runs":
        session_id = body.get("session_id")
        if not isinstance(session_id, str) or not session_id:
            return _bot_chat_only()
    elif template.startswith("api/sessions/"):
        session_id = segments[2]
    elif template.startswith("v1/runs/"):  # an existing run: status, events, stop
        session_id, refused = await _run_session(client, base, key, segments[2])
        if refused is not None:
            return refused
    else:
        return None
    try:
        found = await client.get(f"{base}/api/sessions/{urllib.parse.quote(session_id, safe='')}",
                                 headers={"Authorization": f"Bearer {key}"})
    except httpx.HTTPError as exc:  # a read: safe to retry whatever broke
        return _unreachable(base, exc)
    if found.status_code == 401:
        return _gateway_auth_refused()
    if found.status_code == 404:
        return _bot_chat_only()
    if found.status_code != 200:
        return _unreachable(base, RuntimeError(f"session lookup answered {found.status_code}"))
    try:
        title = (found.json().get("session") or {}).get("title")
    except (ValueError, AttributeError):
        title = None
    return None if str(title or "").strip() == BOT_CHAT_TITLE else _bot_chat_only()


@router.api_route(PEER_PREFIX + "{path:path}", methods=_METHODS, include_in_schema=False)
async def peer_forward(request: Request, path: str):
    principal = getattr(request.state, "peer_principal", None)
    if principal is None:
        raise HTTPException(401, "A peer pass for this agent is required.")
    parsed = _parse(request.scope)
    matched = _route(request.method, parsed[1]) if parsed else None
    if parsed is None or matched is None:
        return _refusal("not_found", "This agent's peer route does not carry that path.", 404)
    profile, segments = parsed
    template, needed = matched
    control = CONTROL in principal.scopes
    if needed not in principal.scopes and not control:
        audit_log(AuditEvent.TOKEN_AUTH_FAILURE, reason="insufficient_scope", principal=principal.principal,
                  method=request.method, path=request.url.path, ip=client_ip(request))
        return _refusal("insufficient_scope", f"This pass does not grant {needed}.", 403)
    method = request.method
    body: Optional[dict] = None
    if (method, template) in _MESSAGING_FIELDS:
        body, invalid = await _json_object(request)
        if invalid is not None:
            return invalid
        parts = [] if control else _control_only_parts(method, template, body)
        if parts:
            audit_log(AuditEvent.TOKEN_AUTH_FAILURE, reason="insufficient_scope", principal=principal.principal,
                      method=method, path=request.url.path, ip=client_ip(request))
            named = ", ".join(part[:64] for part in parts[:5])
            return _refusal("insufficient_scope", f"Sending {named} here needs {CONTROL}.", 403)
        if (method, template) in _TURNS:
            body["author"] = _holder_author(body.get("author"), principal)

    upstream_target = await asyncio.to_thread(_upstream, profile)
    if upstream_target is None:
        return JSONResponse({"error": UNKNOWN_PROFILE_ERROR}, status_code=404)
    base, key = upstream_target
    if not key:
        return _refusal("peer_gateway_unavailable", "This agent's API server has no key configured.", 503)
    # The query goes on re-encoded from the pairs checked here: raw, a ``#`` in it would end the URL
    # built below, and the gateway would read a different query than the one checked.
    pairs = urllib.parse.parse_qsl((request.scope.get("query_string") or b"").decode("latin-1"), keep_blank_values=True)
    query = urllib.parse.urlencode(pairs)
    client = _gateway_client(request.app.state)
    if not control:
        refused = await _outside_the_bot_chat(client, base, key, template, segments, pairs, body)
        if refused is not None:
            return refused
    target = "/" + "/".join(urllib.parse.quote(segment, safe="") for segment in segments)
    headers = {name: value for name, value in request.headers.items()
               if name.lower() not in _DROP_REQUEST
               and not name.lower().startswith(("x-forwarded-", "x-hermes-peer-"))
               and (name.lower() not in _CONTROL_ONLY or control)
               and (body is None or name.lower() not in _BODY_HEADERS)}
    headers.update({"Authorization": f"Bearer {key}", "X-Hermes-Peer-Principal": principal.principal})
    if client_ip(request):
        headers["X-Forwarded-For"] = client_ip(request)
    if body is not None:
        headers["Content-Type"] = "application/json"
        content = json.dumps(body).encode("utf-8")
    else:
        has_body = method not in {"GET", "HEAD"} or request.headers.get("content-length") not in (None, "0")
        content = request.stream() if has_body else None
    upstream = client.build_request(method, f"{base}{target}" + (f"?{query}" if query else ""),
                                    headers=headers, content=content)
    response: Optional[httpx.Response] = None
    try:
        response = await client.send(upstream, stream=True)
    except _NOT_SENT as exc:
        return _unreachable(base, exc)
    except httpx.HTTPError as exc:  # sent, then the connection broke: the action may have run
        _log.warning("peer route: gateway api_server at %s dropped %s %s: %s", base, method, target, exc)
    if response is not None and response.status_code == 401:
        await response.aclose()
        return _gateway_auth_refused()
    if needed == CONTROL:
        audit_log(AuditEvent.PEER_CONTROL, principal=principal.principal, provider=principal.provider,
                  method=method, path=target, profile=profile or "",
                  status=response.status_code if response is not None else 502, ip=client_ip(request))
    if response is None:
        return _refusal("peer_gateway_dropped",
                        "This agent's API server dropped the request before answering; it may have run.", 502)
    out = {name: value for name, value in response.headers.items() if name.lower() in _KEEP_RESPONSE
           and (name.lower() not in _CONTROL_ONLY or control)}
    return StreamingResponse(response.aiter_raw(), status_code=response.status_code, headers=out,
                             background=BackgroundTask(response.aclose))
