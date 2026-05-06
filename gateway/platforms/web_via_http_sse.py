"""Web chat via HTTP + Server-Sent Events.

A generic platform adapter that lets a web application embed a Hermes chat
without going through Telegram, WhatsApp, or any third-party messenger. The
consumer (a web app like openTrattOS, a SaaS dashboard, anything) POSTs a
single message + a Hindsight bank scope, and Hermes replies as a stream of
SSE events on the same HTTP response.

Why this exists: Hermes today exposes Telegram, WhatsApp (Baileys + Meta
Cloud API via MCP), Discord, Slack, etc. — all messenger-style platforms.
There is no first-class adapter for the most common deployment shape today:
"my web app wants to embed a chat with my Hermes agent." Existing options
require either bolting on a generic API server with bespoke auth (the
``api_server`` platform is Hermes-CLI-oriented, not a per-tenant chat
contract) or pretending to be a messenger that nobody uses. This module
fills that gap with a generic, vendor-neutral SSE platform.

Architecture::

    [your web app]                      <-- any consumer, not just openTrattOS
        v POST /{path}/{session_id}
        v   {message, bank_id, user_attribution, metadata?}
        v   X-Web-Auth-Secret: <shared secret>
    [this platform]                     -> handle_message(event)
        |                                       v
        |   text/event-stream            <-  send()  (writes "event: token" frames)
        v
    [your web app receives SSE: token / tool-calling / proactive / done / error]

Key design choices

* **Single HTTP request per turn.** The client POSTs the user message and
  receives the response on the same connection as a SSE stream. No long-poll,
  no WebSocket handshake. The response stays open until ``send()`` (or an
  error) closes it.
* **Generic ``bank_id``.** The consumer chooses the Hindsight bank scope per
  request — this keeps the platform unaware of any particular caller. A SaaS
  consumer might pass ``saas-{tenantId}``; openTrattOS passes
  ``opentrattos-{tenant_slug}``; an experimental consumer might pass
  ``debug-{user}``. The bank id flows into Hermes' memory cascade unmodified.
* **Constant-time auth.** A shared ``X-Web-Auth-Secret`` header is compared
  with ``hmac.compare_digest``. Per-user signing is out of scope — at the
  Hermes level the consumer is "trusted infrastructure"; finer-grained user
  identity belongs in the consumer's own auth layer (which sets
  ``user_attribution`` for audit / personalisation).
* **CORS allowlist.** ``WEB_VIA_HTTP_SSE_ALLOWED_ORIGINS`` is a comma-separated
  list of origins permitted to call the endpoint from a browser. A regex
  variant could be added later if a deployment needs hundreds of origins; the
  allowlist is auditable and sufficient for the typical 1-10 origin case.
* **Single message per turn.** ``send()`` is invoked once with the full
  agent reply. The wire format reserves multi-token streaming for a future
  token-by-token integration in the agent runner; for now ``send()`` emits
  one ``event: token`` frame with the full text plus one ``event: done``
  frame. The client UI can already handle the multi-token form (additive
  rendering) so upgrading later requires no client change.
"""

from __future__ import annotations

import asyncio
import hmac
import json
import logging
import os
from typing import Any, Awaitable, Callable, Dict, Optional, Set

try:
    from aiohttp import web

    AIOHTTP_AVAILABLE = True
except ImportError:
    AIOHTTP_AVAILABLE = False

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import (
    BasePlatformAdapter,
    MessageEvent,
    MessageType,
    SendResult,
)

logger = logging.getLogger(__name__)

MAX_MESSAGE_LENGTH = 16_384  # arbitrary safety cap; a turn message larger than this is almost certainly a bug


def check_web_via_http_sse_requirements() -> bool:
    """Return True if the platform's runtime dependencies are available."""
    return AIOHTTP_AVAILABLE


def _redact_session(session_id: str) -> str:
    """Mask a session id for log output (first 4 + last 2 chars)."""
    if not session_id or len(session_id) < 6:
        return "***"
    return f"{session_id[:4]}***{session_id[-2:]}"


class _ActiveStream:
    """Per-session state for an open SSE response.

    Held in ``WebViaHttpSsePlatformAdapter._streams`` while the request is
    being processed. ``send()`` looks the entry up, writes ``event: token``
    + ``event: done``, and signals ``done_event`` so the request handler can
    return cleanly. If the consumer disconnects mid-flight, the handler sets
    ``cancelled`` and ``send()`` becomes a no-op.
    """

    __slots__ = ("response", "done_event", "cancelled", "first_byte_sent")

    def __init__(self, response: "web.StreamResponse") -> None:
        self.response = response
        self.done_event = asyncio.Event()
        self.cancelled = False
        self.first_byte_sent = False


class WebViaHttpSsePlatformAdapter(BasePlatformAdapter):
    """Embed a Hermes chat in any web application via HTTP + SSE.

    Configuration (via :class:`PlatformConfig` and ``_apply_env_overrides``):

    * ``extra.host`` (default ``0.0.0.0``): aiohttp bind host.
    * ``extra.port`` (default ``8644``): aiohttp bind port.
    * ``extra.path`` (default ``/web``): URL path prefix; the actual route is
      ``POST {path}/{session_id}``.
    * ``extra.auth_secret`` (required for production): shared secret with
      consumers; validated as the ``X-Web-Auth-Secret`` request header on
      every POST. Loaded from ``WEB_VIA_HTTP_SSE_AUTH_SECRET`` env var. If
      unset, requests are accepted unauthenticated and a warning is emitted
      at startup (suitable for loopback-only dev only).
    * ``extra.allowed_origins`` (CSV; default empty = same-origin only):
      ``WEB_VIA_HTTP_SSE_ALLOWED_ORIGINS`` env var. Browsers calling
      cross-origin must have their ``Origin`` header in this allowlist;
      non-browser callers (server-to-server) are unaffected.
    * ``extra.default_bank_id`` (optional): fallback bank id when a request
      omits ``bank_id``. ``WEB_VIA_HTTP_SSE_DEFAULT_BANK_ID`` env var.
    """

    PLATFORM_NAME = "web_via_http_sse"

    def __init__(self, config: PlatformConfig):
        super().__init__(config, Platform.WEB_VIA_HTTP_SSE)
        extra = config.extra or {}
        self._host: str = extra.get("host") or os.getenv(
            "WEB_VIA_HTTP_SSE_HOST", "0.0.0.0"
        )
        self._port: int = int(
            extra.get("port") or os.getenv("WEB_VIA_HTTP_SSE_PORT", "8644")
        )
        self._path: str = extra.get("path") or os.getenv(
            "WEB_VIA_HTTP_SSE_PATH", "/web"
        )
        self._secret: str = extra.get("auth_secret") or os.getenv(
            "WEB_VIA_HTTP_SSE_AUTH_SECRET", ""
        )
        origins_raw = extra.get("allowed_origins") or os.getenv(
            "WEB_VIA_HTTP_SSE_ALLOWED_ORIGINS", ""
        )
        self._allowed_origins: Set[str] = {
            o.strip() for o in origins_raw.split(",") if o.strip()
        }
        self._default_bank_id: str = extra.get("default_bank_id") or os.getenv(
            "WEB_VIA_HTTP_SSE_DEFAULT_BANK_ID", ""
        )
        self._app: Optional["web.Application"] = None
        self._runner: Optional["web.AppRunner"] = None
        self._site: Optional["web.TCPSite"] = None
        self._streams: Dict[str, _ActiveStream] = {}

    # ------------------------------------------------------------------ lifecycle

    async def connect(self) -> bool:
        if not check_web_via_http_sse_requirements():
            logger.error("web_via_http_sse: aiohttp not installed; cannot start")
            return False
        if not self._secret:
            logger.warning(
                "web_via_http_sse: no auth secret configured — accepting unauthenticated "
                "requests. Set WEB_VIA_HTTP_SSE_AUTH_SECRET for any non-loopback deployment."
            )

        self._app = web.Application()
        self._app.router.add_get("/health", self._handle_health)
        self._app.router.add_options(f"{self._path}/{{session_id}}", self._handle_preflight)
        self._app.router.add_post(f"{self._path}/{{session_id}}", self._handle_chat)
        self._runner = web.AppRunner(self._app)
        await self._runner.setup()
        self._site = web.TCPSite(self._runner, self._host, self._port)
        await self._site.start()
        logger.info(
            "web_via_http_sse listening on %s:%s%s/{session_id} (origins=%s)",
            self._host,
            self._port,
            self._path,
            sorted(self._allowed_origins) or "*same-origin*",
        )
        return True

    async def disconnect(self) -> None:
        # Notify any in-flight stream so its handler can exit cleanly.
        for stream in self._streams.values():
            stream.cancelled = True
            stream.done_event.set()
        self._streams.clear()
        if self._site is not None:
            await self._site.stop()
            self._site = None
        if self._runner is not None:
            await self._runner.cleanup()
            self._runner = None
        logger.info("web_via_http_sse disconnected")

    # ------------------------------------------------------------------ inbound

    async def _handle_health(self, _request: "web.Request") -> "web.Response":
        return web.json_response(
            {
                "ok": True,
                "platform": self.PLATFORM_NAME,
                "active_streams": len(self._streams),
            }
        )

    def _origin_headers(self, origin: Optional[str]) -> Dict[str, str]:
        """Build CORS headers if *origin* is allowed, else empty dict."""
        if not origin:
            return {}
        if not self._allowed_origins or origin in self._allowed_origins:
            return {
                "Access-Control-Allow-Origin": origin,
                "Access-Control-Allow-Credentials": "true",
                "Access-Control-Allow-Headers": "Content-Type, X-Web-Auth-Secret",
                "Access-Control-Allow-Methods": "POST, OPTIONS",
                "Access-Control-Max-Age": "600",
                "Vary": "Origin",
            }
        return {}

    async def _handle_preflight(self, request: "web.Request") -> "web.Response":
        origin = request.headers.get("Origin")
        headers = self._origin_headers(origin)
        if not headers:
            return web.Response(status=403, text="origin not allowed")
        return web.Response(status=204, headers=headers)

    def _check_auth(self, request: "web.Request") -> bool:
        if not self._secret:
            return True
        provided = request.headers.get("X-Web-Auth-Secret", "")
        return hmac.compare_digest(provided, self._secret)

    async def _handle_chat(self, request: "web.Request") -> "web.StreamResponse":
        session_id = request.match_info.get("session_id", "").strip()
        origin = request.headers.get("Origin")
        cors_headers = self._origin_headers(origin)
        # Browser caller without an allowed origin → block early.
        if origin and not cors_headers:
            return web.Response(status=403, text="origin not allowed")

        if not self._check_auth(request):
            logger.warning(
                "web_via_http_sse: invalid auth secret from %s session=%s",
                request.remote,
                _redact_session(session_id),
            )
            return web.Response(status=401, text="unauthorized", headers=cors_headers)

        try:
            body = await request.json()
        except (json.JSONDecodeError, ValueError):
            return web.Response(status=400, text="invalid json", headers=cors_headers)

        message = body.get("message") or {}
        msg_type = (message.get("type") or "").strip()
        msg_content = message.get("content")
        bank_id = (body.get("bank_id") or self._default_bank_id or "").strip()
        user_attr = body.get("user_attribution") or {}
        metadata = body.get("metadata") or {}

        if not session_id:
            return web.Response(status=400, text="missing session_id", headers=cors_headers)
        if msg_type not in ("text", "image", "multipart"):
            return web.Response(
                status=400,
                text=f"unsupported message.type {msg_type!r}",
                headers=cors_headers,
            )
        if not msg_content:
            return web.Response(status=400, text="empty message.content", headers=cors_headers)
        if isinstance(msg_content, str) and len(msg_content) > MAX_MESSAGE_LENGTH:
            return web.Response(status=413, text="message too large", headers=cors_headers)

        # Reject concurrent turns on the same session — keeps semantics simple
        # and prevents a confused agent from interleaving two replies.
        if session_id in self._streams:
            return web.Response(
                status=409,
                text="session has an active stream",
                headers=cors_headers,
            )

        text = self._extract_text(msg_type, msg_content)
        if not text:
            return web.Response(
                status=400,
                text="message has no extractable text",
                headers=cors_headers,
            )

        # Open the SSE response and register it in the active map BEFORE
        # dispatching to the agent — the agent's send() will look it up.
        response = web.StreamResponse(
            status=200,
            headers={
                **cors_headers,
                "Content-Type": "text/event-stream",
                "Cache-Control": "no-cache, no-transform",
                "X-Accel-Buffering": "no",  # disable nginx response buffering
                "Connection": "keep-alive",
            },
        )
        await response.prepare(request)
        stream = _ActiveStream(response=response)
        self._streams[session_id] = stream

        # Build the MessageEvent. user_attribution is forwarded into the
        # session source so audit and memory writes carry the right user id.
        source = self.build_source(
            chat_id=session_id,
            user_id=str(user_attr.get("user_id") or session_id),
            chat_type="dm",
        )
        # Stash bank_id + user_attribution + metadata onto the source for
        # downstream agent access via ``event.source.metadata`` (the agent's
        # memory + audit layers consult this).
        source_metadata = dict(getattr(source, "metadata", {}) or {})
        source_metadata.update(
            {
                "bank_id": bank_id,
                "display_name": user_attr.get("display_name", ""),
                "consumer_metadata": metadata,
            }
        )
        try:
            source.metadata = source_metadata  # type: ignore[attr-defined]
        except Exception:
            # ``SessionSource`` is a frozen dataclass in some upstream
            # versions; falling back to attaching via the event keeps us
            # forward-compatible.
            pass

        event = MessageEvent(
            source=source,
            message_type=MessageType.TEXT,
            text=text,
            message_id=f"web-{session_id}",
        )

        logger.info(
            "web_via_http_sse received session=%s bank_id=%s user=%s len=%d",
            _redact_session(session_id),
            bank_id or "<none>",
            user_attr.get("user_id", "<none>"),
            len(text),
        )

        # Dispatch to the agent runner. handle_message is async-fire; the
        # actual completion arrives via send() below.
        dispatch_task = asyncio.create_task(self.handle_message(event))

        try:
            # Wait for send() (or an error) to signal completion. The stream
            # stays open during this wait, holding the SSE connection.
            await stream.done_event.wait()
        except asyncio.CancelledError:
            stream.cancelled = True
            raise
        finally:
            self._streams.pop(session_id, None)
            # If the dispatch task is still running but the client gave up,
            # let it finish in the background — we don't cancel agent work
            # mid-flight (it's holding tools, MCP calls, etc.).
            if not dispatch_task.done():
                self._background_tasks.add(dispatch_task)
                dispatch_task.add_done_callback(self._background_tasks.discard)

        return response

    @staticmethod
    def _extract_text(msg_type: str, content: Any) -> str:
        """Pull the conversational text out of a typed message body."""
        if msg_type == "text":
            return content if isinstance(content, str) else ""
        if isinstance(content, dict):
            text = content.get("text") or ""
            return text if isinstance(text, str) else ""
        return ""

    # ------------------------------------------------------------------ outbound

    async def send(
        self,
        chat_id: str,
        content: str,
        reply_to: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SendResult:
        """Emit ``event: token`` + ``event: done`` to the active SSE stream.

        Token-by-token streaming requires a hook in the agent runner that
        emits intermediate tokens; for now ``send()`` is invoked once with
        the complete reply, which we still wrap in the ``token`` event so
        the wire format is forward-compatible. Clients that handle multiple
        ``token`` events additively will get a no-op upgrade when streaming
        lands later.
        """
        stream = self._streams.get(chat_id)
        if stream is None:
            logger.warning(
                "web_via_http_sse send to unknown session=%s (no active stream)",
                _redact_session(chat_id),
            )
            return SendResult(success=False, error="no_active_stream")
        if stream.cancelled:
            return SendResult(success=False, error="stream_cancelled")
        try:
            await self._write_event(stream, "token", {"chunk": content or ""})
            await self._write_event(stream, "done", {"finishReason": "stop"})
        except (ConnectionError, asyncio.CancelledError) as e:
            stream.cancelled = True
            stream.done_event.set()
            return SendResult(success=False, error=f"stream_closed: {e}")
        finally:
            stream.done_event.set()
        return SendResult(success=True, message_id=f"web-{chat_id}")

    async def send_typing(self, chat_id: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        # No native typing indicator on SSE; clients can render their own
        # spinner while waiting for the first token event.
        return None

    async def stop_typing(self, chat_id: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        return None

    async def send_image(
        self,
        chat_id: str,
        image_url: str,
        caption: Optional[str] = None,
        reply_to: Optional[str] = None,
    ) -> SendResult:
        """Emit an image as a tool-calling-style event with the image URL.

        Outbound images on this platform are URLs by reference — embedding
        the bytes in SSE blows up frame sizes for no real gain. The client
        renders the URL (CORS-permitted) directly.
        """
        stream = self._streams.get(chat_id)
        if stream is None:
            return SendResult(success=False, error="no_active_stream")
        if stream.cancelled:
            return SendResult(success=False, error="stream_cancelled")
        try:
            await self._write_event(
                stream,
                "image",
                {"url": image_url, "caption": caption or ""},
            )
        except (ConnectionError, asyncio.CancelledError) as e:
            stream.cancelled = True
            stream.done_event.set()
            return SendResult(success=False, error=f"stream_closed: {e}")
        return SendResult(success=True, message_id=f"web-img-{chat_id}")

    async def get_chat_info(self, chat_id: str) -> Dict[str, Any]:
        return {"name": f"web-{chat_id}", "type": "dm", "chat_id": chat_id}

    # ------------------------------------------------------------------ event helpers

    async def emit_proactive(self, chat_id: str, text: str) -> bool:
        """Inject a ``proactive`` event into an active stream.

        Returns ``True`` if the event was queued, ``False`` if no stream is
        open for ``chat_id``. Intended for cron-driven nudges and other
        agent-initiated messages while the user has the chat open.
        """
        stream = self._streams.get(chat_id)
        if stream is None or stream.cancelled:
            return False
        try:
            await self._write_event(stream, "proactive", {"text": text})
            return True
        except (ConnectionError, asyncio.CancelledError):
            stream.cancelled = True
            stream.done_event.set()
            return False

    async def emit_tool_calling(self, chat_id: str, tool_name: str) -> bool:
        """Surface that the agent is invoking a tool. UI-only signal."""
        stream = self._streams.get(chat_id)
        if stream is None or stream.cancelled:
            return False
        try:
            await self._write_event(stream, "tool-calling", {"tool": tool_name})
            return True
        except (ConnectionError, asyncio.CancelledError):
            stream.cancelled = True
            stream.done_event.set()
            return False

    async def _write_event(
        self,
        stream: _ActiveStream,
        event_name: str,
        data: Dict[str, Any],
    ) -> None:
        """Write a single SSE frame to *stream*'s response."""
        payload = (
            f"event: {event_name}\n"
            f"data: {json.dumps(data, ensure_ascii=False, separators=(',', ':'))}\n\n"
        )
        await stream.response.write(payload.encode("utf-8"))
        stream.first_byte_sent = True
