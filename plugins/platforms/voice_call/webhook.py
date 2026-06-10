"""Carrier webhook server for the voice_call platform.

One aiohttp application serves three surfaces:

- ``POST <serve.path>`` — carrier webhooks, behind the layered security
  pipeline below
- ``GET <serve.stream_path>/{token}`` — media-stream WebSocket upgrades
  (wired by the realtime phase; rejected until then)
- ``POST /voice/admin`` — localhost-only control endpoint for the CLI,
  authenticated by a pre-shared token stored next to the call log

Security pipeline for carrier webhooks, in order (ported from OpenClaw's
webhook.ts):

1. path match (aiohttp router)        → 404
2. method gate (router, POST only)    → 405
3. pre-auth header gate               → 401 before body read
4. per-IP in-flight limiter           → 429
5. body size limit / read timeout     → 413 / 408
6. provider signature verification    → 403
7. replay cache (TTL)                 → cached 200, not reprocessed
8. provider parse → normalized events
9. inbound policy (new inbound calls) → reject + provider hangup response
10. manager.process_event() per event → provider-expected response
"""

import asyncio
import hashlib
import json
import logging
import secrets
import time
from collections import OrderedDict
from typing import Awaitable, Callable, Dict, Optional

from .config import VoiceCallConfig, normalize_e164
from .events import EventType, NormalizedEvent
from .providers.base import VoiceCallProvider, WebhookContext

logger = logging.getLogger(__name__)

ADMIN_PATH = "/voice/admin"

# Providers whose webhooks carry a signature header; requests without it are
# rejected before the body is read (flood mitigation). The mock provider has
# no signature scheme.
_PREAUTH_HEADERS = {
    "telnyx": ("telnyx-signature-ed25519", "x-telnyx-signature-v02"),
    "twilio": ("x-twilio-signature",),
    "plivo": ("x-plivo-signature-v3", "x-plivo-signature-v2"),
}

AdminHandler = Callable[[dict], Awaitable[dict]]


class _ReplayCache:
    """Remembers recently processed requests: key → response body."""

    def __init__(self, ttl_s: float, cap: int = 4096):
        self.ttl_s = ttl_s
        self.cap = cap
        self._entries: "OrderedDict[str, tuple]" = OrderedDict()  # key → (ts, body, ct)

    def _evict(self) -> None:
        now = time.time()
        while self._entries:
            key, (ts, _, _) = next(iter(self._entries.items()))
            if ts < now - self.ttl_s or len(self._entries) > self.cap:
                self._entries.popitem(last=False)
            else:
                break

    def get(self, key: str):
        self._evict()
        entry = self._entries.get(key)
        return (entry[1], entry[2]) if entry else None

    def put(self, key: str, body: str, content_type: str) -> None:
        self._evict()
        self._entries[key] = (time.time(), body, content_type)


class VoiceCallWebhookServer:
    def __init__(
        self,
        config: VoiceCallConfig,
        provider: VoiceCallProvider,
        process_event: Callable[[NormalizedEvent], Awaitable[None]],
        admin_handler: Optional[AdminHandler] = None,
        admin_token: Optional[str] = None,
    ):
        self.config = config
        self.provider = provider
        self.process_event = process_event
        self.admin_handler = admin_handler
        self.admin_token = admin_token or secrets.token_urlsafe(32)
        self._replay = _ReplayCache(ttl_s=config.security.replay_ttl_s)
        self._inflight: Dict[str, int] = {}
        self._app = None
        self._runner = None
        self._site = None
        # The realtime phase installs a websocket upgrade handler here.
        self.stream_handler = None

    # -- lifecycle ----------------------------------------------------------

    async def start(self) -> None:
        """Bind and serve. Raises OSError when the port is taken."""
        from aiohttp import web

        self._app = web.Application(
            client_max_size=self.config.security.max_body_bytes
        )
        self._app.router.add_post(self.config.serve.path, self._handle_webhook)
        self._app.router.add_get(
            self.config.serve.stream_path + "/{token}", self._handle_stream
        )
        self._app.router.add_post(ADMIN_PATH, self._handle_admin)
        self._runner = web.AppRunner(self._app, access_log=None)
        await self._runner.setup()
        try:
            self._site = web.TCPSite(
                self._runner, self.config.serve.bind, self.config.serve.port
            )
            await self._site.start()
        except OSError:
            await self._runner.cleanup()
            self._runner = None
            self._site = None
            raise
        logger.info(
            "voice_call webhook server listening on %s:%s%s",
            self.config.serve.bind, self.bound_port, self.config.serve.path,
        )

    @property
    def bound_port(self) -> Optional[int]:
        """The actually bound port (differs from config with ``port: 0``)."""
        try:
            server = self._site._server  # noqa: SLF001 — aiohttp exposes no API
            return server.sockets[0].getsockname()[1]
        except (AttributeError, IndexError, TypeError):
            return self.config.serve.port if self._site else None

    async def stop(self) -> None:
        """Stop serving and release the port. Idempotent."""
        if self._site is not None:
            try:
                await self._site.stop()
            except Exception:  # noqa: BLE001
                logger.debug("voice_call webhook site stop failed", exc_info=True)
            self._site = None
        if self._runner is not None:
            try:
                await self._runner.cleanup()
            except Exception:  # noqa: BLE001
                logger.debug("voice_call webhook runner cleanup failed", exc_info=True)
            self._runner = None
        self._app = None

    # -- carrier webhooks -----------------------------------------------------

    def _preauth_ok(self, request) -> bool:
        required = _PREAUTH_HEADERS.get(self.provider.name)
        if not required or self.config.security.skip_signature_verification:
            return True
        return any(request.headers.get(h) for h in required)

    async def _handle_webhook(self, request):
        from aiohttp import web

        # 3. Pre-auth header gate — before the body is read.
        if not self._preauth_ok(request):
            return web.json_response({"error": "missing signature"}, status=401)

        # 4. Per-IP in-flight limiter.
        remote_ip = request.remote or "?"
        if self._inflight.get(remote_ip, 0) >= self.config.security.max_inflight_per_ip:
            return web.json_response({"error": "too many requests"}, status=429)
        self._inflight[remote_ip] = self._inflight.get(remote_ip, 0) + 1
        try:
            return await self._process_webhook(request, remote_ip)
        finally:
            count = self._inflight.get(remote_ip, 1) - 1
            if count <= 0:
                self._inflight.pop(remote_ip, None)
            else:
                self._inflight[remote_ip] = count

    async def _process_webhook(self, request, remote_ip: str):
        from aiohttp import web

        # 5. Body read with size limit (client_max_size) and timeout.
        try:
            body = await asyncio.wait_for(
                request.read(), timeout=self.config.security.body_read_timeout_s
            )
        except asyncio.TimeoutError:
            return web.json_response({"error": "body read timeout"}, status=408)
        except web.HTTPRequestEntityTooLarge:
            return web.json_response({"error": "body too large"}, status=413)

        public_base = (self.provider.public_url or "").rstrip("/")
        ctx = WebhookContext(
            method=request.method,
            path=request.path,
            body=body,
            headers=dict(request.headers),
            query=dict(request.query),
            remote_ip=remote_ip,
            url=f"{public_base}{request.path_qs}" if public_base else str(request.url),
        )

        # 6. Signature verification.
        if not self.config.security.skip_signature_verification:
            verification = self.provider.verify_webhook(ctx)
            if not verification.ok:
                logger.warning(
                    "voice_call webhook: signature verification failed (%s): %s",
                    self.provider.name, verification.error,
                )
                return web.json_response({"error": "invalid signature"}, status=403)
            dedupe_key = verification.dedupe_key
        else:
            dedupe_key = None
        if not dedupe_key:
            dedupe_key = hashlib.sha256(body).hexdigest()

        # 7. Replay gate — return the cached response without reprocessing.
        cached = self._replay.get(dedupe_key)
        if cached is not None:
            body_text, content_type = cached
            return web.Response(
                text=body_text, content_type=content_type, status=200
            )

        # 8. Parse into normalized events + provider-expected response.
        result = self.provider.parse_webhook(ctx)
        if result.response_status >= 400:
            return web.Response(
                text=result.response_body,
                content_type=result.response_content_type,
                status=result.response_status,
            )

        # 9/10. Inbound policy on call-opening inbound events (carriers
        # differ in which event type announces a new inbound call), then
        # process.
        _call_opening = (
            EventType.CALL_INITIATED,
            EventType.CALL_RINGING,
            EventType.CALL_ANSWERED,
        )
        for event in result.events:
            if (
                event.type in _call_opening
                and event.direction == "inbound"
                and not self._inbound_allowed(event.from_number)
            ):
                logger.warning(
                    "voice_call webhook: rejected inbound call from %s (policy=%s)",
                    event.from_number, self.config.inbound_policy,
                )
                continue
            await self.process_event(event)

        # Let the provider rewrite the response based on state created while
        # the events were processed (e.g. Twilio realtime <Connect><Stream>).
        try:
            result = self.provider.finalize_response(ctx, result)
        except Exception:  # noqa: BLE001 — fall back to the parse-time response
            logger.exception("voice_call webhook: finalize_response failed")

        self._replay.put(dedupe_key, result.response_body, result.response_content_type)
        return web.Response(
            text=result.response_body,
            content_type=result.response_content_type,
            status=result.response_status,
        )

    def _inbound_allowed(self, from_number: Optional[str]) -> bool:
        policy = self.config.inbound_policy
        if policy == "open":
            return True
        if policy == "disabled":
            return False
        caller = normalize_e164(from_number or "")
        return bool(caller) and caller in self.config.allow_from

    # -- media stream upgrades (realtime phase) ----------------------------------

    async def _handle_stream(self, request):
        from aiohttp import web

        if self.stream_handler is None:
            return web.json_response({"error": "streaming not enabled"}, status=404)
        return await self.stream_handler(request)

    # -- CLI admin endpoint --------------------------------------------------------

    async def _handle_admin(self, request):
        from aiohttp import web

        token = request.headers.get("x-voice-call-admin-token", "")
        if not secrets.compare_digest(token, self.admin_token):
            return web.json_response({"error": "unauthorized"}, status=401)
        if self.admin_handler is None:
            return web.json_response({"error": "admin handler not wired"}, status=503)
        try:
            payload = json.loads(await request.read() or b"{}")
            if not isinstance(payload, dict):
                raise ValueError
        except (json.JSONDecodeError, ValueError, UnicodeDecodeError):
            return web.json_response({"error": "invalid json"}, status=400)
        try:
            result = await self.admin_handler(payload)
        except Exception as e:  # noqa: BLE001 — CLI gets the error as JSON
            logger.exception("voice_call admin command failed")
            return web.json_response({"success": False, "error": str(e)}, status=500)
        return web.json_response(result)
