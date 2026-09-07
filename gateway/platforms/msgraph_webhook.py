"""Microsoft Graph webhook adapter for change-notification ingress."""

from __future__ import annotations

import asyncio
import hmac
import ipaddress
import json
import logging
import re
from collections import deque
from hashlib import sha1
from typing import Any, Awaitable, Callable, Dict, Optional

try:
    from aiohttp import web

    AIOHTTP_AVAILABLE = True
except ImportError:
    AIOHTTP_AVAILABLE = False
    web = None  # type: ignore[assignment]

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import (
    BasePlatformAdapter, MessageEvent, MessageType, SendResult, is_network_accessible)

logger = logging.getLogger(__name__)

# ``None`` → aiohttp binds one socket per address family (IPv4 + IPv6); the old "0.0.0.0" default was
# unreachable over IPv6-only private networks. Pin a host via extra.host. The all-interfaces default
# still requires extra.allowed_source_cidrs (see _source_allowlist_required_but_missing).
DEFAULT_HOST = None
DEFAULT_PORT = 8646
DEFAULT_WEBHOOK_PATH = "/msgraph/webhook"
DEFAULT_MAX_SEEN_RECEIPTS = 5000
DEFAULT_MAX_BODY_BYTES = 1_048_576
NotificationScheduler = Callable[[Dict[str, Any], MessageEvent], Awaitable[None] | None]
_TEMPLATE_KEY_RE = re.compile(r"\{([a-zA-Z0-9_.]+)\}")


def check_msgraph_webhook_requirements() -> bool:
    """Return whether required webhook dependencies are available."""
    return AIOHTTP_AVAILABLE


def _string_or_none(value: Any) -> Optional[str]:
    return None if value is None else (str(value).strip() or None)


def _normalize_path(path: Any) -> str:
    raw = str(path or "").strip() or "/"
    return raw if raw.startswith("/") else f"/{raw}"


def _parse_allowed_source_cidrs(raw: Any) -> list[ipaddress._BaseNetwork]:
    """Parse the optional CIDR allowlist; empty/missing means "allow everything". When populated, source
    IPs outside every listed CIDR get 403 before the body is parsed (restrict to Microsoft Graph's
    published webhook source ranges in production)."""
    if isinstance(raw, str):
        candidates = raw.split(",")
    elif isinstance(raw, (list, tuple, set)):
        candidates = [str(chunk) for chunk in raw]
    else:
        return []
    networks: list[ipaddress._BaseNetwork] = []
    for chunk in (c.strip() for c in candidates):
        if not chunk:
            continue
        try:
            networks.append(ipaddress.ip_network(chunk, strict=False))
        except ValueError:
            logger.warning("[msgraph_webhook] Ignoring invalid allowed_source_cidrs entry: %r", chunk)
    return networks


def _prefix_match(resource: str, prefix: str) -> bool:
    return resource == prefix or resource.startswith(f"{prefix}/")


def _render_template(template: str, payload: Dict[str, Any]) -> str:
    """Substitute ``{dotted.key}`` placeholders from *payload*; unknown keys stay literal."""

    def _resolve(match: re.Match[str]) -> str:
        key = match.group(1)
        value: Any = payload
        for part in key.split("."):
            if not isinstance(value, dict):
                return f"{{{key}}}"
            value = value.get(part, f"{{{key}}}")
        if isinstance(value, (dict, list)):
            return json.dumps(value, sort_keys=True)[:2000]
        return str(value)

    return _TEMPLATE_KEY_RE.sub(_resolve, template)


class MSGraphWebhookAdapter(BasePlatformAdapter):
    """Receive Microsoft Graph change notifications and surface them internally."""

    def __init__(self, config: PlatformConfig):
        super().__init__(config, Platform.MSGRAPH_WEBHOOK)
        extra = config.extra or {}
        # Falsy host (None/"") collapses to the dual-stack default.
        _raw_host = extra.get("host", DEFAULT_HOST) or DEFAULT_HOST
        self._host: Optional[str] = str(_raw_host) if _raw_host else None
        self._port: int = int(extra.get("port", DEFAULT_PORT))
        self._webhook_path: str = _normalize_path(extra.get("webhook_path", DEFAULT_WEBHOOK_PATH))
        self._health_path: str = _normalize_path(extra.get("health_path", "/health"))
        self._accepted_resources: list[str] = [
            str(value).strip() for value in (extra.get("accepted_resources") or []) if str(value).strip()]
        self._client_state: Optional[str] = _string_or_none(extra.get("client_state"))
        self._max_seen_receipts = max(1, int(extra.get("max_seen_receipts", DEFAULT_MAX_SEEN_RECEIPTS)))
        self._max_body_bytes = max(1, int(extra.get("max_body_bytes", DEFAULT_MAX_BODY_BYTES)))
        self._allowed_source_networks = _parse_allowed_source_cidrs(extra.get("allowed_source_cidrs"))
        self._runner = None
        self._notification_scheduler: Optional[NotificationScheduler] = None
        self._seen_receipts: set[str] = set()
        self._seen_receipt_order: deque[str] = deque()
        self._accepted_count = self._duplicate_count = 0

    def set_notification_scheduler(self, scheduler: Optional[NotificationScheduler]) -> None:
        self._notification_scheduler = scheduler

    def _source_allowlist_required_but_missing(self) -> bool:
        # host=None binds all interfaces (both families) — network-accessible.
        host_is_public = self._host is None or is_network_accessible(self._host)
        return host_is_public and not self._allowed_source_networks

    async def connect(self, *, is_reconnect: bool = False) -> bool:
        if self._client_state is None:
            logger.error("[msgraph_webhook] Refusing to start without extra.client_state configured")
            return False
        if self._source_allowlist_required_but_missing():
            logger.error("[msgraph_webhook] Refusing to start: binding to %s requires extra.allowed_source_cidrs. "
                         "Configure the Microsoft Graph source CIDRs or bind to loopback (127.0.0.1/::1) behind a "
                         "tunnel or reverse proxy.", self._host)
            return False
        app = web.Application(client_max_size=self._max_body_bytes)
        app.router.add_get(self._health_path, self._handle_health)
        app.router.add_get(self._webhook_path, self._handle_validation)
        app.router.add_post(self._webhook_path, self._handle_notification)
        # Plugin-registered native routes; wired before AppRunner.setup() freezes the router.
        self._wire_plugin_handlers(app)
        self._runner = web.AppRunner(app)
        await self._runner.setup()
        site = web.TCPSite(self._runner, self._host, self._port)
        await site.start()
        self._mark_connected()
        logger.info("[msgraph_webhook] Listening on %s:%d%s", self._host, self._port, self._webhook_path)
        return True

    async def disconnect(self) -> None:
        if self._runner is not None:
            await self._runner.cleanup()
            self._runner = None
        self._mark_disconnected()

    async def send(self, chat_id: str, content: str, reply_to: Optional[str] = None,
                   metadata: Optional[Dict[str, Any]] = None) -> SendResult:
        logger.info("[msgraph_webhook] Response for %s: %s", chat_id, content[:200])
        return SendResult(success=True)

    async def get_chat_info(self, chat_id: str) -> Dict[str, Any]:
        return {"name": chat_id, "type": "webhook"}

    async def _handle_health(self, request: "web.Request") -> "web.Response":
        if not self._source_ip_allowed(request):
            return web.Response(status=403)
        return web.json_response({
            "status": "ok", "platform": self.platform.value, "webhook_path": self._webhook_path,
            "accepted": self._accepted_count, "duplicates": self._duplicate_count})

    async def _handle_validation(self, request: "web.Request") -> "web.Response":
        """Graph subscription validation handshake: echo ``validationToken`` verbatim as text/plain. Bare GETs
        are rejected so the endpoint can't be enumerated."""
        if not self._source_ip_allowed(request):
            return web.Response(status=403)
        if not (validation_token := request.query.get("validationToken", "")):
            return web.Response(status=400)
        return web.Response(text=validation_token, content_type="text/plain")

    def _ingest_notification(self, raw_notification: Any) -> str:
        """Classify + schedule one notification: 'accepted' | 'duplicate' | 'auth' | 'other'."""
        if not isinstance(raw_notification, dict):
            return "other"
        notification = dict(raw_notification)
        if not self._resource_accepted(str(notification.get("resource") or "")):
            return "other"
        if not self._verify_client_state(notification):
            # Bad clientState is an auth failure: a fully forged batch gets 403 so the sender stops
            # retrying; legitimate Graph retries carry a valid clientState → accepted/duplicate paths.
            return "auth"
        receipt_key = f"id:{explicit_id}" if (explicit_id := str(notification.get("id") or "").strip()) else None
        if receipt_key is not None:
            if receipt_key in self._seen_receipts:
                return "duplicate"
            self._remember_receipt(receipt_key)
        self._accepted_count += 1
        self._schedule_notification(notification, self._build_message_event(notification, receipt_key))
        return "accepted"

    async def _handle_notification(self, request: "web.Request") -> "web.Response":
        if not self._source_ip_allowed(request):
            return web.Response(status=403)
        # Graph never sends validationToken on POST, but tolerate clients replaying it in-band.
        if validation_token := request.query.get("validationToken", ""):
            return web.Response(text=validation_token, content_type="text/plain")
        status, notifications = await self._read_notifications(request)
        if status:
            return web.Response(status=status)
        counts = {"accepted": 0, "duplicate": 0, "auth": 0, "other": 0}
        for raw_notification in notifications:
            counts[self._ingest_notification(raw_notification)] += 1
        self._duplicate_count += counts["duplicate"]
        # Anything ingested OR deduped → 202 with empty body (Graph acks; no counter leak). Every item
        # failed auth → 403 so forged POSTs get a clear reject. Otherwise (malformed / not accepted) → 400.
        if counts["accepted"] or counts["duplicate"]:
            return web.Response(status=202)
        if counts["auth"] and not counts["other"]:
            return web.Response(status=403)
        return web.Response(status=400)

    async def _read_notifications(self, request: "web.Request") -> tuple[int, list]:
        """Read and validate the POST body; returns (error_status, []) or (0, notifications)."""
        try:
            content_length = request.content_length
        except Exception:
            content_length = None
        if content_length is not None and content_length > self._max_body_bytes:
            return 413, []
        try:
            raw_body = await request.read()
        except Exception:
            return 400, []
        if len(raw_body) > self._max_body_bytes:
            return 413, []
        try:
            body = json.loads(raw_body.decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            return 400, []
        notifications = body.get("value") if isinstance(body, dict) else None
        return (0, notifications) if isinstance(notifications, list) else (400, [])

    def _source_ip_allowed(self, request: "web.Request") -> bool:
        """Loopback-only binds may omit ``allowed_source_cidrs`` (local proxies, dev tunnels);
        network-accessible binds fail closed without one."""
        if self._source_allowlist_required_but_missing():
            return False
        if not self._allowed_source_networks:
            return True
        try:
            peer_addr = ipaddress.ip_address(request.remote or "")
        except ValueError:
            return False
        return any(peer_addr in network for network in self._allowed_source_networks)

    def _resource_accepted(self, resource: str) -> bool:
        if not self._accepted_resources:
            return True
        resource = resource.strip().strip("/")
        for pattern in (p.strip().strip("/") for p in self._accepted_resources):
            if pattern.endswith("*"):
                pattern = pattern[:-1].rstrip("/")
            if pattern and _prefix_match(resource, pattern):
                return True
        return False

    def _verify_client_state(self, notification: Dict[str, Any]) -> bool:
        """Timing-safe compare of the Graph-supplied clientState against the configured shared secret
        (``openssl rand -hex 32`` in the setup guide)."""
        expected = self._client_state
        provided = _string_or_none(notification.get("clientState"))
        if expected is None or provided is None:
            return False
        # Compare as bytes: compare_digest raises TypeError on non-ASCII str (clientState is request-controlled).
        return hmac.compare_digest(provided.encode(), expected.encode())

    def _remember_receipt(self, receipt_key: str) -> None:
        self._seen_receipts.add(receipt_key)
        self._seen_receipt_order.append(receipt_key)
        while len(self._seen_receipt_order) > self._max_seen_receipts:
            self._seen_receipts.discard(self._seen_receipt_order.popleft())

    def _build_message_event(self, notification: Dict[str, Any], receipt_key: Optional[str]) -> MessageEvent:
        message_id = receipt_key or f"sha1:{sha1(json.dumps(notification, sort_keys=True).encode('utf-8')).hexdigest()}"
        source = self.build_source(
            chat_id=f"msgraph:{notification.get('subscriptionId', 'unknown')}", chat_name="msgraph/webhook",
            chat_type="webhook", user_id="msgraph", user_name="Microsoft Graph")
        return MessageEvent(
            text=self._render_prompt(notification), message_type=MessageType.TEXT, source=source,
            raw_message=notification, message_id=message_id, internal=True)

    def _render_prompt(self, notification: Dict[str, Any]) -> str:
        template = self.config.extra.get("prompt", "")
        if template:
            return _render_template(template, {
                "notification": notification, "resource": notification.get("resource", ""),
                "change_type": notification.get("changeType", ""),
                "subscription_id": notification.get("subscriptionId", "")})
        rendered = json.dumps(notification, indent=2, sort_keys=True)[:4000]
        return f"Microsoft Graph change notification:\n\n```json\n{rendered}\n```"

    def _schedule_notification(self, notification: Dict[str, Any], event: MessageEvent) -> None:
        scheduler = self._notification_scheduler
        if scheduler is None:
            coro = self.handle_message(event)
        else:
            coro = scheduler(notification, event)
            if not asyncio.iscoroutine(coro):
                return
        task = asyncio.create_task(coro)
        self._background_tasks.add(task)
        task.add_done_callback(self._background_tasks.discard)
