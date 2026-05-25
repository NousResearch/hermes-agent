"""Microsoft Graph webhook adapter for change-notification ingress."""

from __future__ import annotations

import asyncio
import hmac
import ipaddress
import json
import logging
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
    BasePlatformAdapter,
    MessageEvent,
    MessageType,
    SendResult,
)

logger = logging.getLogger(__name__)

DEFAULT_HOST = "0.0.0.0"
DEFAULT_PORT = 8646
DEFAULT_WEBHOOK_PATH = "/msgraph/webhook"
DEFAULT_MAX_SEEN_RECEIPTS = 5000
NotificationScheduler = Callable[[Dict[str, Any], MessageEvent], Awaitable[None] | None]


def check_msgraph_webhook_requirements() -> bool:
    """Return whether required webhook dependencies are available."""
    return AIOHTTP_AVAILABLE


class MSGraphWebhookAdapter(BasePlatformAdapter):
    """Receive Microsoft Graph change notifications and surface them internally."""

    def __init__(self, config: PlatformConfig):
        super().__init__(config, Platform.MSGRAPH_WEBHOOK)
        extra = config.extra or {}
        self._host: str = str(extra.get("host", DEFAULT_HOST))
        self._port: int = int(extra.get("port", DEFAULT_PORT))
        self._webhook_path: str = self._normalize_path(
            extra.get("webhook_path", DEFAULT_WEBHOOK_PATH)
        )
        self._health_path: str = self._normalize_path(extra.get("health_path", "/health"))
        self._accepted_resources: list[str] = [
            str(value).strip()
            for value in (extra.get("accepted_resources") or [])
            if str(value).strip()
        ]
        self._client_state: Optional[str] = self._string_or_none(extra.get("client_state"))
        self._max_seen_receipts = max(
            1, int(extra.get("max_seen_receipts", DEFAULT_MAX_SEEN_RECEIPTS))
        )
        self._allowed_source_networks: list[ipaddress._BaseNetwork] = (
            self._parse_allowed_source_cidrs(extra.get("allowed_source_cidrs"))
        )
        self._runner = None
        self._notification_scheduler: Optional[NotificationScheduler] = None
        self._seen_receipts: set[str] = set()
        self._seen_receipt_order: deque[str] = deque()
        self._accepted_count = 0
        self._duplicate_count = 0

    @staticmethod
    def _string_or_none(value: Any) -> Optional[str]:
        if value is None:
            return None
        text = str(value).strip()
        return text or None

    @staticmethod
    def _normalize_path(path: Any) -> str:
        raw = str(path or "").strip() or "/"
        return raw if raw.startswith("/") else f"/{raw}"

    @staticmethod
    def _build_receipt_key(notification: Dict[str, Any]) -> Optional[str]:
        explicit_id = str(notification.get("id") or "").strip()
        if explicit_id:
            return f"id:{explicit_id}"
        return None

    @staticmethod
    def _normalize_resource_value(resource: str) -> str:
        return str(resource or "").strip().strip("/")

    @staticmethod
    def _parse_allowed_source_cidrs(
        raw: Any,
    ) -> list[ipaddress._BaseNetwork]:
        """Parse an optional list of CIDR ranges allowed to POST to the webhook.

        An empty or missing value means "allow everything" (same behavior as
        before this field existed). When populated, requests from source IPs
        outside every listed CIDR are rejected with 403 before the body is
        parsed. Use this to restrict the endpoint to Microsoft Graph's
        published webhook source ranges in production deployments.
        """
        if raw is None:
            return []
        if isinstance(raw, str):
            candidates = [chunk.strip() for chunk in raw.split(",")]
        elif isinstance(raw, (list, tuple, set)):
            candidates = [str(chunk).strip() for chunk in raw]
        else:
            return []

        networks: list[ipaddress._BaseNetwork] = []
        for chunk in candidates:
            if not chunk:
                continue
            try:
                networks.append(ipaddress.ip_network(chunk, strict=False))
            except ValueError:
                logger.warning(
                    "[msgraph_webhook] Ignoring invalid allowed_source_cidrs entry: %r",
                    chunk,
                )
        return networks

    def set_notification_scheduler(self, scheduler: Optional[NotificationScheduler]) -> None:
        self._notification_scheduler = scheduler

    async def connect(self) -> bool:
        app = web.Application()
        app.router.add_get(self._health_path, self._handle_health)
        app.router.add_get(self._webhook_path, self._handle_validation)
        app.router.add_post(self._webhook_path, self._handle_notification)

        self._runner = web.AppRunner(app)
        await self._runner.setup()
        site = web.TCPSite(self._runner, self._host, self._port)
        await site.start()
        self._mark_connected()
        logger.info(
            "[msgraph_webhook] Listening on %s:%d%s",
            self._host,
            self._port,
            self._webhook_path,
        )
        return True

    async def disconnect(self) -> None:
        if self._runner is not None:
            await self._runner.cleanup()
            self._runner = None
        self._mark_disconnected()

    async def send(
        self,
        chat_id: str,
        content: str,
        reply_to: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SendResult:
        logger.info("[msgraph_webhook] Response for %s: %s", chat_id, content[:200])
        return SendResult(success=True)

    async def get_chat_info(self, chat_id: str) -> Dict[str, Any]:
        return {"name": chat_id, "type": "webhook"}

    async def _handle_health(self, request: "web.Request") -> "web.Response":
        return web.json_response(
            {
                "status": "ok",
                "platform": self.platform.value,
                "webhook_path": self._webhook_path,
                "accepted": self._accepted_count,
                "duplicates": self._duplicate_count,
            }
        )

    async def _handle_validation(self, request: "web.Request") -> "web.Response":
        """Handle Microsoft Graph subscription validation handshake.

        Graph validates a subscription endpoint by sending a GET with
        ``validationToken`` in the query string; the service must echo the
        token verbatim as ``text/plain`` within 10 seconds. Anything else
        (bare GET, GET without the token) is rejected so the endpoint can't
        be enumerated or mistakenly used for data exfiltration.
        """
        if not self._source_ip_allowed(request):
            return web.Response(status=403)
        validation_token = request.query.get("validationToken", "")
        if not validation_token:
            return web.Response(status=400)
        return web.Response(text=validation_token, content_type="text/plain")

    async def _handle_notification(self, request: "web.Request") -> "web.Response":
        if not self._source_ip_allowed(request):
            return web.Response(status=403)

        # Graph never sends validationToken on POST, but tolerate it for
        # defensive clients that replay the handshake in-band.
        validation_token = request.query.get("validationToken", "")
        if validation_token:
            return web.Response(text=validation_token, content_type="text/plain")

        try:
            body = await request.json()
        except Exception:
            return web.Response(status=400)

        notifications = body.get("value")
        if not isinstance(notifications, list):
            return web.Response(status=400)

        accepted = 0
        duplicates = 0
        auth_rejected = 0
        other_rejected = 0

        for raw_notification in notifications:
            if not isinstance(raw_notification, dict):
                other_rejected += 1
                continue
            notification = dict(raw_notification)
            if not self._resource_accepted(str(notification.get("resource") or "")):
                other_rejected += 1
                continue
            if not self._verify_client_state(notification):
                # Treat bad clientState as an auth failure: if the whole
                # batch is forged, we want to signal 403 so the sender
                # stops retrying. Legitimate Graph retries have valid
                # clientState and hit the accepted/duplicate paths.
                auth_rejected += 1
                continue

            receipt_key = self._build_receipt_key(notification)
            if receipt_key is not None:
                if self._has_seen_receipt(receipt_key):
                    duplicates += 1
                    continue
                self._remember_receipt(receipt_key)

            accepted += 1
            self._accepted_count += 1
            event = self._build_message_event(notification, receipt_key)
            self._schedule_notification(notification, event)

        self._duplicate_count += duplicates
        # If anything ingested OR deduped, return 202 with empty body so
        # Graph acks successfully and we don't leak internal counters. If
        # every item failed auth, return 403 so an attacker POSTing fake
        # notifications gets a clear reject. Other failures (malformed,
        # resource-not-accepted) are the sender's configuration problem,
        # so 400.
        if accepted or duplicates:
            return web.Response(status=202)
        if auth_rejected and not other_rejected:
            return web.Response(status=403)
        return web.Response(status=400)

    def _source_ip_allowed(self, request: "web.Request") -> bool:
        """Return True if the request's source IP is in the configured allowlist.

        When ``allowed_source_cidrs`` is empty (the default), everything is
        allowed — preserves behavior for dev tunnels / localhost setups.
        """
        if not self._allowed_source_networks:
            return True
        peer = request.remote or ""
        if not peer:
            return False
        try:
            peer_addr = ipaddress.ip_address(peer)
        except ValueError:
            return False
        return any(peer_addr in network for network in self._allowed_source_networks)

    def _resource_accepted(self, resource: str) -> bool:
        if not self._accepted_resources:
            return True
        normalized_resource = self._normalize_resource_value(resource)
        for pattern in self._accepted_resources:
            normalized_pattern = self._normalize_resource_value(pattern)
            if not normalized_pattern:
                continue
            if normalized_pattern.endswith("*"):
                prefix = normalized_pattern[:-1].rstrip("/")
                if normalized_resource == prefix or normalized_resource.startswith(f"{prefix}/"):
                    return True
                continue
            if (
                normalized_resource == normalized_pattern
                or normalized_resource.startswith(f"{normalized_pattern}/")
            ):
                return True
        return False

    def _verify_client_state(self, notification: Dict[str, Any]) -> bool:
        """Verify the Graph-supplied clientState matches the configured secret.

        Uses ``hmac.compare_digest`` instead of ``==`` so that a mismatch
        doesn't leak how many leading characters matched via string-compare
        timing. The configured client_state is a shared secret (documented in
        the setup guide as "generate with ``openssl rand -hex 32``"), so a
        timing-safe compare is the right primitive.
        """
        expected = self._client_state
        if expected is None:
            return True
        provided = self._string_or_none(notification.get("clientState"))
        if provided is None:
            return False
        return hmac.compare_digest(provided, expected)

    def _has_seen_receipt(self, receipt_key: str) -> bool:
        return receipt_key in self._seen_receipts

    def _remember_receipt(self, receipt_key: str) -> None:
        self._seen_receipts.add(receipt_key)
        self._seen_receipt_order.append(receipt_key)
        while len(self._seen_receipt_order) > self._max_seen_receipts:
            oldest = self._seen_receipt_order.popleft()
            self._seen_receipts.discard(oldest)

    def _build_message_event(
        self,
        notification: Dict[str, Any],
        receipt_key: Optional[str],
    ) -> MessageEvent:
        message_id = receipt_key or f"sha1:{sha1(json.dumps(notification, sort_keys=True).encode('utf-8')).hexdigest()}"
        source = self.build_source(
            chat_id=f"msgraph:{notification.get('subscriptionId', 'unknown')}",
            chat_name="msgraph/webhook",
            chat_type="webhook",
            user_id="msgraph",
            user_name="Microsoft Graph",
        )
        return MessageEvent(
            text=self._render_prompt(notification),
            message_type=MessageType.TEXT,
            source=source,
            raw_message=notification,
            message_id=message_id,
            internal=True,
        )

    def _render_prompt(self, notification: Dict[str, Any]) -> str:
        template = self.config.extra.get("prompt", "")
        if template:
            payload = {
                "notification": notification,
                "resource": notification.get("resource", ""),
                "change_type": notification.get("changeType", ""),
                "subscription_id": notification.get("subscriptionId", ""),
            }
            return self._render_template(template, payload)
        rendered = json.dumps(notification, indent=2, sort_keys=True)[:4000]
        return f"Microsoft Graph change notification:\n\n```json\n{rendered}\n```"

    def _render_template(self, template: str, payload: Dict[str, Any]) -> str:
        import re

        def _resolve(match: "re.Match[str]") -> str:
            key = match.group(1)
            value: Any = payload
            for part in key.split("."):
                if isinstance(value, dict):
                    value = value.get(part, f"{{{key}}}")
                else:
                    return f"{{{key}}}"
            if isinstance(value, (dict, list)):
                return json.dumps(value, sort_keys=True)[:2000]
            return str(value)

        return re.sub(r"\{([a-zA-Z0-9_.]+)\}", _resolve, template)

    def _schedule_notification(
        self,
        notification: Dict[str, Any],
        event: MessageEvent,
    ) -> None:
        scheduler = self._notification_scheduler
        if scheduler is not None:
            result = scheduler(notification, event)
            if asyncio.iscoroutine(result):
                task = asyncio.create_task(result)
                self._background_tasks.add(task)
                task.add_done_callback(self._background_tasks.discard)
            return

        task = asyncio.create_task(self.handle_message(event))
        self._background_tasks.add(task)
        task.add_done_callback(self._background_tasks.discard)
