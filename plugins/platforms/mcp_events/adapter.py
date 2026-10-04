"""MCP Events inbound adapter: a stdlib ``http.server`` (daemon thread) that receives
signed event deliveries from subscribed MCP event emitters and routes them into
the LIVE gateway session — the agent that wakes is the same one serving the user,
with full memory and context, not a throwaway clone.

Deliveries are answered 200 as soon as they are verified and dispatched; the
adapter never blocks on the agent's turn (emitters retry with backoff, so a slow
200 would multiply deliveries). ``interactive_resume = False``: this is a
webhook-style platform — nobody is on the other end to answer a restore prompt,
so auto-resume finishes interrupted work instead of asking.

No token configured => binds 127.0.0.1 only (same posture as the A2A adapter)."""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import threading
import time
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Callable, Dict, Optional

from gateway.platforms.base import BasePlatformAdapter, SendResult
from gateway.platforms.event import MessageEvent, MessageType

from . import protocol, security

logger = logging.getLogger(__name__)

_WEBHOOK_PREFIX = "/mcp/events/webhook"
_MAX_BODY = protocol.MAX_EVENT_BYTES + 4096  # delivery ceiling + header slack


def _daemon_thread(target, name: str) -> threading.Thread:
    t = threading.Thread(target=target, name=name, daemon=True)
    t.start()
    return t


class MCPEventsRequestHandler(BaseHTTPRequestHandler):
    """HTTP handler for the webhook surface; all state lives on ``self.server.adapter``."""

    @property
    def adapter(self) -> "MCPEventsAdapter":
        return self.server.adapter  # type: ignore[attr-defined]

    def log_message(self, format, *args):  # noqa: A002,N802
        logger.debug("MCP Events http: " + format, *args)  # silence the default stderr access log

    def _json(self, code: int, payload: dict):
        body = json.dumps(payload).encode("utf-8")
        self.send_response(code)
        for k, v in (("Content-Type", "application/json"), ("Content-Length", str(len(body)))):
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(body)

    def _read_body(self) -> Optional[bytes]:
        try:
            length = int(self.headers.get("Content-Length") or 0)
        except (ValueError, TypeError):
            return None
        if length <= 0 or length > _MAX_BODY:
            return None
        return self.rfile.read(length)

    def do_GET(self):  # noqa: N802
        # Table-driven routing: no if/elif ladder on the path.
        routes: dict[str, Callable[[], None]] = {
            "/": self._status,
            "/health": self._status,
            "/mcp/events/subscriptions": self._subscriptions,
        }
        path = urllib.parse.urlparse(self.path).path.rstrip("/") or "/"
        handler = routes.get(path)
        if handler is None:
            return self._json(404, {"error": "not found"})
        handler()

    def do_POST(self):  # noqa: N802
        path = urllib.parse.urlparse(self.path).path
        if path == _WEBHOOK_PREFIX or path.startswith(_WEBHOOK_PREFIX + "/"):
            return self._ingest(path)
        return self._json(404, {"error": "not found"})

    def _status(self):
        adapter = self.adapter
        self._json(200, {"status": "ok", "platform": "mcp_events",
                         "subscriptions": len(adapter.store.list()),
                         "mode": "localhost-only" if adapter._sec.localhost_only() else "remote"})

    def _subscriptions(self):
        # Local introspection only — subscription records name emitters, never secrets.
        adapter = self.adapter
        if not adapter._sec.is_loopback_bind():
            auth = self.headers.get("Authorization", "")
            if auth != f"Bearer {adapter._sec.webhook_secret}":
                return self._json(401, {"error": "unauthorized"})
        subs = [{"id": s["id"], "emitter_url": s["emitter_url"], "event": s["event"],
                 "created_at": s.get("created_at"), "expires_at": s.get("expires_at")}
                for s in adapter.store.list()]
        self._json(200, {"subscriptions": subs})

    def _ingest(self, path: str):
        """Verify a signed event delivery and dispatch it into the live session.

        Every rejection answers a bare status (never the reason — the reason goes
        to the audit log); every acceptance answers 200 immediately, before the
        agent's turn runs.
        """
        adapter = self.adapter
        sec = adapter._sec

        def drop(http_code: int, reason: str, **audit_kw):
            security.audit("drop", audit_kw.get("emitter", "?"), audit_kw.get("ref", "-"), reason)
            adapter.metrics["dropped"] += 1
            logger.warning("MCP Events: dropped delivery (%s)", reason)
            self._json(http_code, {"ok": False})

        body = self._read_body()
        if body is None:
            return drop(413, "body missing or over limit")
        webhook_id = self.headers.get("webhook-id", "")
        timestamp = self.headers.get("webhook-timestamp", "")
        signature = self.headers.get("webhook-signature", "")
        sub_id = path[len(_WEBHOOK_PREFIX):].strip("/")
        # The callback URL embeds our local id (the emitter's id is only known
        # after subscribe returns); resolve() accepts either.
        sub = adapter.store.resolve(sub_id) if sub_id else None
        if sub is None:
            return drop(404, "unknown subscription", ref=sub_id)
        if not protocol.verify_delivery(sec.webhook_secret, webhook_id, timestamp, signature, body,
                                         skew_seconds=sec.timestamp_skew):
            return drop(401, "signature/timestamp verification failed", emitter=sub["emitter_url"], ref=sub_id)
        if adapter._seen.seen(webhook_id):
            # Retried delivery — already dispatched; ack without waking the agent twice.
            return self._json(200, {"ok": True, "duplicate": True})
        emitter_host = (urllib.parse.urlparse(sub["emitter_url"]).hostname or "")
        if not adapter._rate.check(f"emitter:{emitter_host}"):
            return drop(429, "per-emitter rate limit exceeded", emitter=sub["emitter_url"], ref=sub_id)
        if not adapter._storm.check(f"sub:{sub_id}"):
            adapter.metrics["storm_drops"] += 1
            return drop(429, "subscription storm guard tripped", emitter=sub["emitter_url"], ref=sub_id)
        try:
            payload = json.loads(body.decode("utf-8"))
        except Exception:
            return drop(400, "delivery body is not JSON", emitter=sub["emitter_url"], ref=sub_id)

        event_name = str(payload.get("event") or sub.get("event") or "unknown")
        text = security.wrap_event(sub["emitter_url"], event_name, sub_id,
                                   security.render_event_payload(payload))
        chat_id = f"mcp-events:{sub_id}"  # one conversation per subscription, like A2A's per-context routing
        dispatched = adapter._dispatch_to_session(chat_id, sub["emitter_url"], event_name, text)
        if not dispatched:
            return drop(503, "agent gateway not ready", emitter=sub["emitter_url"], ref=sub_id)
        security.audit("inbound", sub["emitter_url"], sub_id, f"event {event_name!r} dispatched")
        adapter.metrics["accepted"] += 1
        self._json(200, {"ok": True})


class MCPEventsAdapter(BasePlatformAdapter):
    """Platform adapter for the MCP Events receiver."""

    # Nobody is on the other end of a webhook: auto-resume finishes interrupted
    # work instead of asking a question no one will answer (#57056 pattern).
    interactive_resume = False

    def __init__(self, config, **kwargs):
        super().__init__(config, **kwargs)
        self._sec = security.MCPEventsSecurityContext.capture()
        self.host = self._sec.resolve_bind_host()
        self.port = self._sec.port
        self.store = protocol.SubscriptionStore()
        self._rate = security.SlidingWindow(self._sec.rate_limit_per_min, 60)
        self._storm = security.SlidingWindow(self._sec.storm_max_per_min, 60)
        self._seen = security.IdempotencySet()
        self.metrics = {"accepted": 0, "dropped": 0, "storm_drops": 0}
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._httpd: Optional[ThreadingHTTPServer] = None
        self._server_thread: Optional[threading.Thread] = None

    def _callback_base(self) -> str:
        """Public base URL handed to emitters at subscribe time. Remote mode needs
        ``mcp_events.public_base_url`` in config.yaml; localhost mode uses the bind."""
        if self._sec.public_base_url:
            return self._sec.public_base_url
        return f"http://{self.host}:{self.port}"

    def callback_url(self, subscription_id: str) -> str:
        return f"{self._callback_base()}{_WEBHOOK_PREFIX}/{subscription_id}"

    def _dispatch_to_session(self, chat_id: str, emitter_url: str, event_name: str, text: str) -> bool:
        """Fire-and-forget into the live gateway session (HTTP worker thread)."""
        if self._loop is None:
            return False
        event = MessageEvent(
            text=text, message_type=MessageType.TEXT, message_id=f"evt-{int(time.time() * 1000)}",
            source=self.build_source(chat_id=chat_id, chat_name=f"mcp-events:{event_name}",
                                     chat_type="dm", user_id=emitter_url, user_name=emitter_url))
        try:
            asyncio.run_coroutine_threadsafe(self.handle_message(event), self._loop)
        except Exception:
            logger.warning("MCP Events: dispatch to session failed", exc_info=True)
            return False
        return True

    def _refresh_subscriptions(self) -> None:
        """Re-subscribe anything past ``expires_at`` so restarts don't silently go deaf."""
        for sub in self.store.expired():
            try:
                fresh = protocol.subscribe(sub["emitter_url"], sub["event"],
                                           self.callback_url(sub["id"]), self._sec.webhook_secret,
                                           filter_args=sub.get("filter") or None)
            except Exception as e:
                logger.warning("MCP Events: refresh of subscription %s failed: %s", sub["id"], e)
                security.audit("drop", sub.get("emitter_url", "?"), sub["id"], f"refresh failed: {e}")
                continue
            if fresh["id"] != sub["id"]:
                # Emitter rotated the id — retire the old record, keep the new one.
                self.store.remove(sub["id"])
            fresh["filter"] = sub.get("filter") or {}
            self.store.add(fresh)
            security.audit("subscribe", sub.get("emitter_url", "?"), fresh["id"], "refreshed after expiry")

    async def connect(self, **_kwargs) -> bool:
        # Capture the gateway loop so the HTTP thread can marshal events via run_coroutine_threadsafe.
        self._loop = asyncio.get_running_loop()
        try:
            self._httpd = ThreadingHTTPServer((self.host, self.port), MCPEventsRequestHandler)
        except OSError as e:
            logger.error("MCP Events: could not bind %s:%s — %s", self.host, self.port, e)
            self._set_fatal_error("bind_failed", f"MCP Events bind failed: {e}", retryable=True)
            return False
        self._httpd.daemon_threads = True
        self._httpd.adapter = self  # type: ignore[attr-defined]
        self._server_thread = _daemon_thread(self._httpd.serve_forever, "mcp-events-http")
        self._mark_connected()
        with contextlib.suppress(Exception):
            self._refresh_subscriptions()
        logger.info("MCP Events: webhook receiver on http://%s:%s (%s); %d subscription(s) loaded",
                    self.host, self.port,
                    "localhost-only" if self._sec.localhost_only() else "REMOTE (secret <redacted>)",
                    len(self.store.list()))
        if not self._sec.localhost_only() and not self._sec.public_base_url:
            logger.warning("MCP Events: remote bind with no mcp_events.public_base_url — emitters "
                           "cannot be told where to deliver. Set it in config.yaml.")
        return True

    async def disconnect(self) -> None:
        if self._httpd is not None:
            with contextlib.suppress(Exception):
                self._httpd.shutdown()
            self._httpd = None
        self._loop = None

    async def send(self, chat_id: str, content: str, reply_to: Optional[str] = None,
                   metadata: Optional[Dict[str, Any]] = None):
        """One-way channel: event deliveries have no reply path to the emitter, so the
        agent's turn simply ends in its session. Reported as success so the gateway
        doesn't treat event turns as failed sends."""
        logger.debug("MCP Events: send() for %s absorbed (one-way channel)", chat_id)
        return SendResult(success=True, message_id=str(int(time.time() * 1000)))

    async def send_typing(self, chat_id: str, metadata=None) -> None:
        return None

    async def get_chat_info(self, chat_id: str) -> Dict[str, Any]:
        return {"name": chat_id, "type": "dm"}
