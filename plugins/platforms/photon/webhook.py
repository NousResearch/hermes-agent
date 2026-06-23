"""Photon iMessage inbound via webhook (push) instead of the gRPC/WS stream.

Photon Spectrum can deliver inbound iMessage events as signed HTTPS POSTs
(see https://photon.codes/docs/webhooks/events). Receiving inbound this way
avoids the long-lived ``catchUpEvents`` stream entirely — no 16-stream
concurrency limit, no zombie/"Live stream ended" death spirals, no shared-line
inbound starvation. Outbound still goes through the sidecar's ``spectrum-ts``
SDK, because Photon exposes no public HTTP send endpoint.

This server:
  * verifies the ``X-Spectrum-Signature`` HMAC-SHA256 over ``v0:{ts}:{body}``,
  * rejects stale deliveries (clock skew guard) and unsigned/forged ones,
  * maps the webhook body to the SAME normalized event shape the sidecar's
    NDJSON stream emits, then hands it to ``PhotonAdapter._dispatch_inbound``
    so dedup and all downstream handling stay shared with the stream path.

Funnel/ingress (public HTTPS → 127.0.0.1:port) is configured outside Hermes
(e.g. Tailscale Funnel); this server only binds loopback.
"""
from __future__ import annotations

import hashlib
import hmac
import json
import logging
import time
from typing import TYPE_CHECKING, Any, Dict, Optional

from aiohttp import web

if TYPE_CHECKING:
    from .adapter import PhotonAdapter

logger = logging.getLogger(__name__)

# Reject deliveries whose signed timestamp is older/newer than this (seconds).
# Photon signs with the delivery time; their docs reject >5 min on their side.
_MAX_TIMESTAMP_SKEW_SECONDS = 300
# Defence-in-depth: cap the request body so a peer can't OOM us.
_MAX_BODY_BYTES = 5 * 1024 * 1024


class PhotonWebhookServer:
    """Loopback aiohttp server that turns Photon webhooks into MessageEvents."""

    def __init__(
        self,
        adapter: "PhotonAdapter",
        *,
        host: str,
        port: int,
        path: str,
        secret: str,
    ) -> None:
        self._adapter = adapter
        self._host = host
        self._port = port
        self._path = path if path.startswith("/") else f"/{path}"
        self._secret = secret or ""
        self._runner: Optional[web.AppRunner] = None

    async def start(self) -> None:
        app = web.Application(client_max_size=_MAX_BODY_BYTES)
        app.router.add_post(self._path, self._handle)
        app.router.add_get(self._path, self._handle_health)
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, self._host, self._port)
        await site.start()
        self._runner = runner
        logger.info(
            "[photon] webhook inbound listening on http://%s:%d%s",
            self._host, self._port, self._path,
        )

    async def stop(self) -> None:
        if self._runner is not None:
            try:
                await self._runner.cleanup()
            finally:
                self._runner = None

    # -- handlers ---------------------------------------------------------

    async def _handle_health(self, request: web.Request) -> web.Response:
        # A bare GET lets you (and Funnel) sanity-check reachability without
        # a valid signature. Never reveals payloads.
        return web.json_response({"ok": True, "service": "photon-webhook"})

    async def _handle(self, request: web.Request) -> web.Response:
        raw = await request.read()

        if not self._verify(request, raw):
            # 401 (not 5xx) so Photon does NOT retry a forged/misconfigured
            # delivery forever.
            return web.json_response({"ok": False, "error": "bad signature"}, status=401)

        try:
            body = json.loads(raw)
        except (ValueError, UnicodeDecodeError):
            return web.json_response({"ok": False, "error": "bad json"}, status=400)

        # Acknowledge anything we recognize as a valid signed delivery with 2xx
        # so Photon stops retrying; only signature/parse failures get non-2xx.
        try:
            await self._process(body)
        except Exception:
            logger.exception("[photon] webhook processing failed (acked to avoid retry storm)")
        return web.json_response({"ok": True})

    # -- verification -----------------------------------------------------

    def _verify(self, request: web.Request, raw: bytes) -> bool:
        if not self._secret:
            logger.error("[photon] webhook secret not configured — rejecting delivery")
            return False
        ts = request.headers.get("X-Spectrum-Timestamp", "")
        sig = request.headers.get("X-Spectrum-Signature", "")
        if not ts or not sig:
            logger.warning("[photon] webhook missing signature/timestamp headers")
            return False

        # Clock-skew / replay guard.
        try:
            skew = abs(time.time() - int(ts))
        except ValueError:
            logger.warning("[photon] webhook timestamp not an integer: %r", ts)
            return False
        if skew > _MAX_TIMESTAMP_SKEW_SECONDS:
            logger.warning("[photon] webhook timestamp too skewed (%.0fs) — rejecting", skew)
            return False

        # HMAC-SHA256 of "v0:{timestamp}:{rawBody}" keyed by the signing secret.
        base = b"v0:" + ts.encode("utf-8") + b":" + raw
        expected = "v0=" + hmac.new(
            self._secret.encode("utf-8"), base, hashlib.sha256
        ).hexdigest()
        if not hmac.compare_digest(expected, sig):
            logger.warning("[photon] webhook signature mismatch")
            return False
        return True

    # -- payload mapping --------------------------------------------------

    async def _process(self, body: Dict[str, Any]) -> None:
        if body.get("event") != "messages":
            logger.debug("[photon] webhook ignoring event=%r", body.get("event"))
            return
        message = body.get("message") or {}
        if message.get("direction") and message.get("direction") != "inbound":
            return  # our own outbound echo

        msg_id = message.get("id")
        if msg_id and self._adapter._is_duplicate(msg_id):
            return  # at-least-once delivery replay

        space = message.get("space") or body.get("space") or {}
        sender = message.get("sender") or {}

        # Normalize into the exact shape PhotonAdapter._dispatch_inbound expects
        # (matching sidecar/index.mjs), so all downstream handling is shared.
        event = {
            "messageId": msg_id,
            "platform": message.get("platform") or "iMessage",
            "space": {
                "id": space.get("id"),
                "type": space.get("type"),
                "phone": space.get("phone"),
            },
            "sender": {"id": sender.get("id")},
            "content": message.get("content") or {},
            "timestamp": message.get("timestamp") or "",
        }
        if not event["space"]["id"]:
            logger.warning("[photon] webhook delivery missing space.id — dropping")
            return
        await self._adapter._dispatch_inbound(event)
