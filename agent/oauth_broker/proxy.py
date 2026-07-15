"""Loopback proxy application for the OAuth broker.

Exactly two client-facing routes exist (docs/design/oauth-broker.md §五):

    POST /accounts/{alias}/backend-api/codex/responses
    GET  /accounts/{alias}/backend-api/wham/usage

Everything else is 404. Requests must present the local broker bearer
(constant-time compared). The broker strips inbound credentials, injects the
selected account's Authorization and ChatGPT-Account-Id, forwards the body
byte-for-byte to the pinned upstream origin, and streams the response back
without buffering or rewriting. On an upstream 401 it forces one slot
refresh and replays the buffered request once; a 429 is propagated verbatim
and never triggers account switching here — failover stays in the Hermes
credential pool.

Log lines carry request id, alias, route kind, status, and duration only.
Request/response bodies and header values are never logged.
"""

from __future__ import annotations

import asyncio
import hmac
import logging
import time
import uuid
from typing import Callable, Dict, Optional
from urllib.parse import urlsplit

import aiohttp
from aiohttp import web

from agent.oauth_broker.account_slot import AccountSlot, SlotRefreshError
from agent.oauth_broker.models import ACCOUNT_ALIASES

logger = logging.getLogger(__name__)

DEFAULT_UPSTREAM_ORIGIN = "https://chatgpt.com"
RESPONSES_UPSTREAM_PATH = "/backend-api/codex/responses"
USAGE_UPSTREAM_PATH = "/backend-api/wham/usage"

# Mirrors hermes_cli/proxy/server.py's request cap.
DEFAULT_MAX_REQUEST_BYTES = 10_000_000
DEFAULT_SOCK_CONNECT_TIMEOUT = 15.0
DEFAULT_SOCK_READ_TIMEOUT = 300.0

# Minimal forwarded-header allowlist. Only what the Codex upstream actually
# needs crosses the broker; everything else (cookies, tracing baggage,
# stainless telemetry, forwarded-for, connection headers, client-supplied
# credentials or account routing) is dropped. Proven against the existing
# transport call sites:
#   accept, user-agent      — agent/account_usage.py, agent/auxiliary_client.py
#   content-type            — Responses POST body
#   originator              — Cloudflare first-party gate (auxiliary_client.py)
#   openai-beta             — OpenAI SDK feature negotiation
#   session_id,
#   x-client-request-id     — agent/transports/codex.py cache-scope routing
# Authorization and ChatGPT-Account-Id are always broker-inserted.
FORWARD_REQUEST_HEADER_ALLOWLIST = frozenset(
    {
        "accept",
        "content-type",
        "user-agent",
        "originator",
        "openai-beta",
        "session_id",
        "x-client-request-id",
    }
)

# Upstream response headers forwarded to the client. Everything else —
# Set-Cookie, CDN internals, connection headers — is dropped.
_RESPONSE_HEADER_ALLOW_EXACT = frozenset(
    {
        "content-type",
        "content-length",
        "content-encoding",
        "retry-after",
        "x-request-id",
    }
)
_RESPONSE_HEADER_ALLOW_PREFIXES = ("x-ratelimit-", "x-codex-", "openai-")

SessionFactory = Callable[[], aiohttp.ClientSession]

_SLOTS_KEY = web.AppKey("oauth_broker_slots", dict)
_KEY_BYTES_KEY = web.AppKey("oauth_broker_local_key", bytes)
_ORIGIN_KEY = web.AppKey("oauth_broker_upstream_origin", str)
_SESSION_KEY = web.AppKey("oauth_broker_client_session", aiohttp.ClientSession)
_SESSION_FACTORY_KEY: web.AppKey[SessionFactory] = web.AppKey(
    "oauth_broker_session_factory"
)


def validate_upstream_origin(
    origin: str, *, allow_test_upstream: bool = False
) -> str:
    """Allow only the pinned production origin by default.

    ``http://127.0.0.1:<port>`` exists solely for injected test servers and is
    accepted only with the explicit in-process ``allow_test_upstream`` gate.
    No CLI or launchd path exposes that gate.
    """
    if not isinstance(origin, str) or not origin.strip():
        raise ValueError("upstream origin must be a non-empty string")
    if origin == DEFAULT_UPSTREAM_ORIGIN:
        return DEFAULT_UPSTREAM_ORIGIN
    parsed = urlsplit(origin)
    if parsed.username is not None or parsed.password is not None:
        raise ValueError("upstream origin must not carry userinfo")
    if parsed.path not in ("", "/") or parsed.query or parsed.fragment:
        raise ValueError("upstream origin must not carry a path, query, or fragment")
    port = parsed.port  # raises ValueError for malformed or out-of-range ports
    if parsed.scheme == "http" and parsed.hostname == "127.0.0.1":
        if not allow_test_upstream:
            raise ValueError("loopback upstream is disabled outside explicit tests")
        if port is None or not 1 <= port <= 65535:
            raise ValueError("test loopback upstream requires a port in 1..65535")
        return origin.rstrip("/")
    raise ValueError(
        "upstream origin must be the pinned https://chatgpt.com origin"
    )


def _json_error(status: int, code: str, message: str) -> web.Response:
    return web.json_response(
        {"error": {"code": code, "message": message, "type": code}},
        status=status,
    )


def _authorized(request: web.Request, key_bytes: bytes) -> bool:
    header = request.headers.get("Authorization", "")
    if not header.startswith("Bearer "):
        return False
    presented = header[len("Bearer "):].strip().encode("utf-8")
    return hmac.compare_digest(presented, key_bytes)


def _build_upstream_headers(
    request: web.Request, token: str, account_id: Optional[str]
) -> Dict[str, str]:
    headers: Dict[str, str] = {}
    for key, value in request.headers.items():
        if key.lower() in FORWARD_REQUEST_HEADER_ALLOWLIST:
            headers[key] = value
    headers["Authorization"] = f"Bearer {token}"
    if account_id:
        headers["ChatGPT-Account-Id"] = account_id
    return headers


def _filter_response_headers(headers) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for key, value in headers.items():
        lowered = key.lower()
        if lowered == "transfer-encoding":
            continue  # hop-by-hop; recomputed by the streaming writer
        if lowered in _RESPONSE_HEADER_ALLOW_EXACT or lowered.startswith(
            _RESPONSE_HEADER_ALLOW_PREFIXES
        ):
            out[key] = value
    return out


def create_proxy_app(
    *,
    slots: Dict[str, AccountSlot],
    local_key: str,
    upstream_origin: str = DEFAULT_UPSTREAM_ORIGIN,
    request_body_limit: int = DEFAULT_MAX_REQUEST_BYTES,
    client_session_factory: Optional[SessionFactory] = None,
    allow_test_upstream: bool = False,
) -> web.Application:
    if not isinstance(local_key, str) or not local_key.strip():
        raise ValueError("local_key must be a non-empty string")
    if not slots:
        raise ValueError("at least one account slot is required")
    for alias, slot in slots.items():
        if alias not in ACCOUNT_ALIASES or slot is None:
            raise ValueError(f"invalid slot alias {alias!r}")
    origin = validate_upstream_origin(
        upstream_origin, allow_test_upstream=allow_test_upstream
    )

    app = web.Application(client_max_size=request_body_limit)
    app[_SLOTS_KEY] = dict(slots)
    app[_KEY_BYTES_KEY] = local_key.encode("utf-8")
    app[_ORIGIN_KEY] = origin
    app[_SESSION_FACTORY_KEY] = client_session_factory or (
        lambda: aiohttp.ClientSession(
            cookie_jar=aiohttp.DummyCookieJar(),
            auto_decompress=False,
            timeout=aiohttp.ClientTimeout(
                total=None,
                sock_connect=DEFAULT_SOCK_CONNECT_TIMEOUT,
                sock_read=DEFAULT_SOCK_READ_TIMEOUT,
            )
        )
    )

    async def _session_ctx(app: web.Application):
        session = app[_SESSION_FACTORY_KEY]()
        if not isinstance(session.cookie_jar, aiohttp.DummyCookieJar) or (
            session.auto_decompress is not False
        ):
            await session.close()
            raise RuntimeError(
                "upstream session must use DummyCookieJar and disable auto decompression"
            )
        app[_SESSION_KEY] = session
        yield
        await app[_SESSION_KEY].close()

    app.cleanup_ctx.append(_session_ctx)

    async def handle_responses(request: web.Request) -> web.StreamResponse:
        return await _handle(request, kind="responses")

    async def handle_usage(request: web.Request) -> web.StreamResponse:
        return await _handle(request, kind="usage")

    async def _handle(request: web.Request, *, kind: str) -> web.StreamResponse:
        rid = uuid.uuid4().hex[:8]
        raw_alias = request.match_info.get("alias", "")
        alias = raw_alias if raw_alias in ACCOUNT_ALIASES else "unknown"
        started = time.monotonic()
        status = 500
        try:
            result = await _handle_inner(request, kind=kind)
            status = result.status
            return result
        except web.HTTPException as exc:
            status = exc.status
            raise
        finally:
            logger.info(
                "oauth broker proxy: rid=%s alias=%s kind=%s status=%s dur_ms=%.0f",
                rid,
                alias,
                kind,
                status,
                (time.monotonic() - started) * 1000,
            )

    async def _handle_inner(
        request: web.Request, *, kind: str
    ) -> web.StreamResponse:
        alias = request.match_info["alias"]
        if not _authorized(request, request.app[_KEY_BYTES_KEY]):
            return _json_error(401, "unauthorized", "missing or invalid broker key")
        slot = request.app[_SLOTS_KEY].get(alias)
        if slot is None:
            return _json_error(404, "unknown_account", "no such account route")

        body = await request.read()

        try:
            token = await slot.get_access_token()
        except SlotRefreshError as exc:
            return _json_error(
                503,
                f"broker_slot_{exc.category}",
                "account slot cannot serve credentials",
            )

        if kind == "responses":
            method = "POST"
            upstream_path = RESPONSES_UPSTREAM_PATH
        else:
            method = "GET"
            upstream_path = USAGE_UPSTREAM_PATH
        url = request.app[_ORIGIN_KEY] + upstream_path
        if request.query_string:
            url = f"{url}?{request.query_string}"
        session = request.app[_SESSION_KEY]

        async def _send(active_token: str):
            return await session.request(
                method,
                url,
                data=body if body else None,
                headers=_build_upstream_headers(
                    request, active_token, slot.account_id()
                ),
                allow_redirects=False,
            )

        try:
            upstream_resp = await _send(token)
        except (asyncio.TimeoutError, TimeoutError):
            return _json_error(504, "upstream_timeout", "upstream request timed out")
        except aiohttp.ClientError:
            return _json_error(502, "upstream_unreachable", "upstream connection failed")

        if upstream_resp.status == 401:
            # Single forced refresh, single replay of the buffered request.
            # A second 401 flows back unchanged — no refresh loops (§八.7).
            upstream_resp.close()
            try:
                token = await slot.refresh_after_unauthorized(token)
            except SlotRefreshError as exc:
                return _json_error(
                    503,
                    f"broker_slot_{exc.category}",
                    "account slot cannot serve credentials",
                )
            try:
                upstream_resp = await _send(token)
            except (asyncio.TimeoutError, TimeoutError):
                return _json_error(504, "upstream_timeout", "upstream request timed out")
            except aiohttp.ClientError:
                return _json_error(502, "upstream_unreachable", "upstream connection failed")

        out = web.StreamResponse(
            status=upstream_resp.status,
            headers=_filter_response_headers(upstream_resp.headers),
        )
        try:
            await out.prepare(request)
            async for chunk in upstream_resp.content.iter_any():
                if chunk:
                    await out.write(chunk)
            await out.write_eof()
            return out
        except asyncio.CancelledError:
            raise
        except (aiohttp.ClientError, ConnectionResetError) as exc:
            logger.debug(
                "oauth broker proxy: stream interrupted (%s)",
                type(exc).__name__,
            )
            transport = request.transport
            if transport is not None and not transport.is_closing():
                transport.abort()
            raise
        finally:
            upstream_resp.close()

    app.router.add_post(
        "/accounts/{alias}/backend-api/codex/responses", handle_responses
    )
    app.router.add_get(
        "/accounts/{alias}/backend-api/wham/usage",
        handle_usage,
        allow_head=False,
    )
    return app


__all__ = [
    "DEFAULT_MAX_REQUEST_BYTES",
    "DEFAULT_UPSTREAM_ORIGIN",
    "FORWARD_REQUEST_HEADER_ALLOWLIST",
    "RESPONSES_UPSTREAM_PATH",
    "USAGE_UPSTREAM_PATH",
    "create_proxy_app",
    "validate_upstream_origin",
]
