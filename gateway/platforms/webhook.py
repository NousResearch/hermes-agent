"""Generic webhook platform adapter.

Runs an aiohttp HTTP server that receives webhook POSTs from external
services (GitHub, GitLab, JIRA, Stripe, etc.), validates HMAC signatures,
transforms payloads into agent prompts, and routes responses back to the
source or to another configured platform.

Configuration lives in config.yaml under platforms.webhook.extra.routes.
Each route defines:
  - events: which event types to accept (header-based filtering)
  - secret: HMAC secret for signature validation (REQUIRED)
  - prompt: template string formatted with the webhook payload
  - skills: optional list of skills to load for the agent
  - deliver: where to send the response (github_comment, telegram, etc.)
  - deliver_extra: additional delivery config (repo, pr_number, chat_id)
  - deliver_only: if true, skip the agent — the rendered prompt IS the
    message that gets delivered.  Use for external push notifications
    (Supabase, monitoring alerts, inter-agent pings) where zero LLM cost
    and sub-second delivery matter more than agent reasoning.

Security:
  - HMAC secret is required per route (validated at startup)
  - Rate limiting per route (fixed-window, configurable)
  - Idempotency cache prevents duplicate agent runs on webhook retries
  - Body size limits checked before reading payload
  - Set secret to "INSECURE_NO_AUTH" to skip validation (testing only)
"""

import asyncio
import base64
import binascii
import hashlib
import hmac
import json
import logging
import os
import re
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

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
    ProcessingOutcome,
    SendResult,
)

logger = logging.getLogger(__name__)

_BUILTIN_DELIVER_PLATFORMS = {
    "telegram", "discord", "slack", "signal", "sms", "whatsapp",
    "matrix", "mattermost", "homeassistant", "email", "dingtalk",
    "feishu", "wecom", "wecom_callback", "weixin", "bluebubbles",
    "qqbot", "yuanbao",
}

DEFAULT_HOST = "0.0.0.0"
DEFAULT_PORT = 8644
_INSECURE_NO_AUTH = "INSECURE_NO_AUTH"
_DYNAMIC_ROUTES_FILENAME = "webhook_subscriptions.json"

# Hostnames/IP literals that only serve connections originating on the same
# machine. Anything else is treated as a public bind for safety-rail purposes.
_LOOPBACK_HOSTS = frozenset({
    "127.0.0.1",
    "localhost",
    "::1",
    "ip6-localhost",
    "ip6-loopback",
})


def _is_loopback_host(host: str) -> bool:
    """True when `host` binds only to the local machine.

    Covers IPv4 loopback, the standard `localhost` alias, IPv6 loopback in
    both bracketed and bare form, and the common Debian-style aliases. Any
    falsy value (empty string, None) is conservatively treated as non-loopback
    because an unset host usually means the platform-default public bind.
    """
    if not host:
        return False
    return host.strip().lower() in _LOOPBACK_HOSTS


def check_webhook_requirements() -> bool:
    """Check if webhook adapter dependencies are available."""
    return AIOHTTP_AVAILABLE


class WebhookAdapter(BasePlatformAdapter):
    """Generic webhook receiver that triggers agent runs from HTTP POSTs."""

    def __init__(self, config: PlatformConfig):
        super().__init__(config, Platform.WEBHOOK)
        self._host: str = config.extra.get("host", DEFAULT_HOST)
        self._port: int = int(config.extra.get("port", DEFAULT_PORT))
        self._global_secret: str = config.extra.get("secret", "")
        self._static_routes: Dict[str, dict] = config.extra.get("routes", {})
        self._dynamic_routes: Dict[str, dict] = {}
        self._dynamic_routes_mtime: float = 0.0
        self._routes: Dict[str, dict] = dict(self._static_routes)
        self._runner = None

        # Fizzy/card-scoped memory state keyed by synthetic webhook chat_id.
        self._card_memory_info: Dict[str, dict] = {}

        # Delivery info keyed by session chat_id.
        #
        # Read by every send() invocation for the chat_id (status messages
        # AND the final response).  Cleaned up via TTL on each POST so the
        # dict stays bounded — see _prune_delivery_info().  Do NOT pop on
        # send(), or interim status messages (e.g. fallback notifications,
        # context-pressure warnings) will consume the entry before the
        # final response arrives, causing the response to silently fall
        # back to the "log" deliver type.
        self._delivery_info: Dict[str, dict] = {}
        self._delivery_info_created: Dict[str, float] = {}

        # Reference to gateway runner for cross-platform delivery (set externally)
        self.gateway_runner = None

        # Idempotency: TTL cache of recently processed delivery IDs.
        # Prevents duplicate agent runs when webhook providers retry.
        self._seen_deliveries: Dict[str, float] = {}
        self._idempotency_ttl: int = 3600  # 1 hour
        self._queue_record_by_chat_id: Dict[str, str] = {}
        self._queue_retry_delay_seconds: float = float(
            config.extra.get("queue_retry_delay_seconds", 60.0)
        )

        # Rate limiting: per-route timestamps in a fixed window.
        self._rate_counts: Dict[str, List[float]] = {}
        self._rate_limit: int = int(config.extra.get("rate_limit", 30))  # per minute

        # Body size limit (auth-before-body pattern)
        self._max_body_bytes: int = int(
            config.extra.get("max_body_bytes", 1_048_576)
        )  # 1MB

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def connect(self) -> bool:
        # Load agent-created subscriptions before validating
        self._reload_dynamic_routes()

        # Validate routes at startup — secret is required per route
        for name, route in self._routes.items():
            secret = route.get("secret", self._global_secret)
            if not secret:
                raise ValueError(
                    f"[webhook] Route '{name}' has no HMAC secret. "
                    f"Set 'secret' on the route or globally. "
                    f"For testing without auth, set secret to '{_INSECURE_NO_AUTH}'."
                )

            # Safety rail: refuse to start if INSECURE_NO_AUTH is combined with a
            # non-loopback bind. The escape hatch is for local testing only;
            # serving an unauthenticated route on a public interface is a
            # deployment-grade footgun we'd rather crash early than ship.
            if secret == _INSECURE_NO_AUTH and not _is_loopback_host(self._host):
                raise ValueError(
                    f"[webhook] Route '{name}' uses INSECURE_NO_AUTH secret "
                    f"but is bound to non-loopback host '{self._host}'. "
                    f"INSECURE_NO_AUTH is for local testing only. "
                    f"Refusing to start to prevent accidental exposure."
                )
            # deliver_only routes bypass the agent — the POST body becomes a
            # direct push notification via the configured delivery target.
            # Validate up-front so misconfiguration surfaces at startup rather
            # than on the first webhook POST.
            if route.get("deliver_only"):
                deliver = route.get("deliver", "log")
                if not deliver or deliver == "log":
                    raise ValueError(
                        f"[webhook] Route '{name}' has deliver_only=true but "
                        f"deliver is '{deliver}'. Direct delivery requires a "
                        f"real target (telegram, discord, slack, github_comment, etc.)."
                    )

        app = web.Application()
        app.router.add_get("/health", self._handle_health)
        app.router.add_post("/webhooks/{route_name}", self._handle_webhook)

        # Port conflict detection — fail fast if port is already in use
        import socket as _socket
        try:
            with _socket.socket(_socket.AF_INET, _socket.SOCK_STREAM) as _s:
                _s.settimeout(1)
                _s.connect(('127.0.0.1', self._port))
            logger.error('[webhook] Port %d already in use. Set a different port in config.yaml: platforms.webhook.port', self._port)
            return False
        except (ConnectionRefusedError, OSError):
            pass  # port is free

        self._runner = web.AppRunner(app)
        await self._runner.setup()
        site = web.TCPSite(self._runner, self._host, self._port)
        await site.start()
        self._mark_connected()

        route_names = ", ".join(self._routes.keys()) or "(none configured)"
        logger.info(
            "[webhook] Listening on %s:%d — routes: %s",
            self._host,
            self._port,
            route_names,
        )
        return True

    async def disconnect(self) -> None:
        if self._runner:
            await self._runner.cleanup()
            self._runner = None
        self._mark_disconnected()
        logger.info("[webhook] Disconnected")

    def _classify_protocol_failure(
        self, content: Optional[str]
    ) -> tuple[bool, Optional[str], str]:
        """Classify agent-produced protocol/API errors for webhook retry handling."""
        text = str(content or "").strip()
        if not text:
            return False, None, "success"
        lowered = text.lower()
        failure_markers = (
            "notion comment handling failed",
            "webhook handling failed",
            "webhook protocol failure",
            "validation_error",
            "permission_denied",
            "unauthorized",
            "forbidden",
            "invalid_request",
            "apiresponseerror",
            "http 4",
            "http 5",
            "service_unavailable",
            "timeout",
            "timed out",
            "connection error",
        )
        if not any(marker in lowered for marker in failure_markers):
            return False, None, "success"
        retryable_markers = (
            "408",
            "429",
            "500",
            "502",
            "503",
            "504",
            "service_unavailable",
            "timeout",
            "timed out",
            "temporar",
            "rate_limited",
            "rate limit",
            "connection error",
        )
        classification = (
            "retryable"
            if any(marker in lowered for marker in retryable_markers)
            else "terminal"
        )
        return True, text[:4000], classification

    async def on_processing_complete(
        self, event: MessageEvent, outcome: ProcessingOutcome
    ) -> None:
        """Update durable webhook queue state after gateway processing finishes."""
        chat_id = getattr(getattr(event, "source", None), "chat_id", "") or ""
        record_id = self._queue_record_by_chat_id.get(chat_id)
        if not record_id:
            return

        from gateway import webhook_queue

        delivery_error = getattr(event, "_hermes_delivery_error", None)
        if not delivery_error:
            delivery_error = self._delivery_info.get(chat_id, {}).get("last_protocol_error")
        classification = "retryable"
        if delivery_error and ":" in str(delivery_error):
            maybe_classification, _, rest = str(delivery_error).partition(":")
            if maybe_classification in {"terminal", "retryable"}:
                classification = maybe_classification
                delivery_error = rest.strip()
        failed, protocol_error, detected_classification = self._classify_protocol_failure(
            delivery_error
        )
        if failed:
            classification = detected_classification

        if outcome == ProcessingOutcome.SUCCESS and not failed:
            webhook_queue.mark_done(record_id)
            self._queue_record_by_chat_id.pop(chat_id, None)
            return

        error = (
            protocol_error
            or str(delivery_error or "")
            or f"webhook processing {outcome.value}"
        )
        if classification == "terminal":
            webhook_queue.mark_dead_letter(
                record_id,
                error,
                classification=classification,
                details={"chat_id": chat_id, "outcome": outcome.value},
            )
            self._queue_record_by_chat_id.pop(chat_id, None)
            return

        webhook_queue.mark_retry(
            record_id,
            error,
            retry_delay_seconds=self._queue_retry_delay_seconds,
        )

    async def send(
        self,
        chat_id: str,
        content: str,
        reply_to: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SendResult:
        """Deliver the agent's response to the configured destination.

        chat_id is ``webhook:{route}:{delivery_id}``.  The delivery info
        stored during webhook receipt is read with ``.get()`` (not popped)
        so that interim status messages emitted before the final response
        — fallback-model notifications, context-pressure warnings, etc. —
        do not consume the entry and silently downgrade the final response
        to the ``log`` deliver type.  TTL cleanup happens on POST.
        """
        self._persist_card_memory_update(chat_id, content)

        delivery = self._delivery_info.get(chat_id, {})
        deliver_type = delivery.get("deliver", "log")

        if deliver_type == "log":
            logger.info("[webhook] Response for %s: %s", chat_id, content[:200])
            failed, error, _classification = self._classify_protocol_failure(content)
            if failed:
                delivery["last_protocol_error"] = error or content[:4000]
            return SendResult(success=not failed, error=error if failed else None)

        if deliver_type == "github_comment":
            return await self._deliver_github_comment(content, delivery)

        # Cross-platform delivery — any platform with a gateway adapter.
        # Check both built-in names and plugin-registered platforms.
        _is_known_platform = deliver_type in _BUILTIN_DELIVER_PLATFORMS
        if not _is_known_platform:
            try:
                from gateway.platform_registry import platform_registry
                _is_known_platform = platform_registry.is_registered(deliver_type)
            except Exception:
                pass
        if self.gateway_runner and _is_known_platform:
            return await self._deliver_cross_platform(
                deliver_type, content, delivery
            )

        logger.warning("[webhook] Unknown deliver type: %s", deliver_type)
        return SendResult(
            success=False, error=f"Unknown deliver type: {deliver_type}"
        )

    def _prune_delivery_info(self, now: float) -> None:
        """Drop delivery_info entries older than the idempotency TTL.

        Mirrors the cleanup pattern used for ``_seen_deliveries``.  Called
        on each POST so the dict size is bounded by ``rate_limit * TTL``
        even if many webhooks fire and never receive a final response.
        """
        cutoff = now - self._idempotency_ttl
        stale = [
            k
            for k, t in self._delivery_info_created.items()
            if t < cutoff
        ]
        for k in stale:
            self._delivery_info.pop(k, None)
            self._delivery_info_created.pop(k, None)

    async def get_chat_info(self, chat_id: str) -> Dict[str, Any]:
        return {"name": chat_id, "type": "webhook"}

    # ------------------------------------------------------------------
    # HTTP handlers
    # ------------------------------------------------------------------

    async def _handle_health(self, request: "web.Request") -> "web.Response":
        """GET /health — simple health check."""
        return web.json_response({"status": "ok", "platform": "webhook"})

    def _handle_notion_route_event(
        self,
        *,
        route_name: str,
        route_config: dict,
        payload: dict,
        event_type: str,
        delivery_id: str,
        prompt: str,
    ) -> Optional["web.Response"]:
        """Apply Notion-specific webhook routing before generic agent dispatch."""
        if event_type in {"comment.created", "comment.updated"}:
            return None
        if event_type == "comment.deleted":
            return web.json_response(
                {
                    "status": "ignored",
                    "route": route_name,
                    "event": event_type,
                    "delivery_id": delivery_id,
                },
                status=200,
            )

        record = self._make_notion_sync_record(
            route_name=route_name,
            payload=payload,
            event_type=event_type,
            delivery_id=delivery_id,
            prompt=prompt,
            deliver_config={
                "deliver": route_config.get("deliver", "log"),
                "deliver_extra": self._render_delivery_extra(
                    route_config.get("deliver_extra", {}), payload
                ),
                "payload": payload,
            },
        )
        queue_path = self._notion_sync_queue_path()
        queue_path.parent.mkdir(parents=True, exist_ok=True)
        with queue_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
        return web.json_response(
            {
                "status": "queued",
                "route": route_name,
                "event": event_type,
                "delivery_id": delivery_id,
            },
            status=202,
        )

    def _notion_sync_queue_path(self) -> Path:
        from hermes_constants import get_hermes_home

        return get_hermes_home() / "data" / "notion-sync" / "notion-sync-queue.jsonl"

    def _make_notion_sync_record(
        self,
        *,
        route_name: str,
        payload: dict,
        event_type: str,
        delivery_id: str,
        prompt: str,
        deliver_config: dict,
    ) -> Dict[str, Any]:
        entity = payload.get("entity") if isinstance(payload.get("entity"), dict) else {}
        data = payload.get("data") if isinstance(payload.get("data"), dict) else {}
        parent = data.get("parent") if isinstance(data.get("parent"), dict) else {}
        entity_id = entity.get("id") or payload.get("entity_id") or payload.get("page_id") or payload.get("id")
        entity_type = entity.get("type") or payload.get("entity_type")
        page_id = entity_id if entity_type == "page" else payload.get("page_id")
        return {
            "route_name": route_name,
            "delivery_id": delivery_id,
            "webhook_id": payload.get("id") or delivery_id,
            "type": event_type,
            "timestamp": payload.get("timestamp"),
            "workspace_id": payload.get("workspace_id"),
            "subscription_id": payload.get("subscription_id"),
            "attempt_number": payload.get("attempt_number"),
            "entity_id": entity_id,
            "entity_type": entity_type,
            "page_id": page_id,
            "parent_id": parent.get("id"),
            "parent_type": parent.get("type"),
            "data_source_id": parent.get("data_source_id"),
            "updated_blocks": data.get("updated_blocks"),
            "payload": payload,
            "prompt": prompt,
            "deliver_config": deliver_config,
            "queued_at": time.time(),
            "reason": "non_comment_event_context_sync_only",
        }

    def _reload_dynamic_routes(self) -> None:
        """Reload agent-created subscriptions from disk if the file changed."""
        from hermes_constants import get_hermes_home
        hermes_home = get_hermes_home()
        subs_path = hermes_home / _DYNAMIC_ROUTES_FILENAME
        if not subs_path.exists():
            if self._dynamic_routes:
                self._dynamic_routes = {}
                self._routes = dict(self._static_routes)
                logger.debug("[webhook] Dynamic subscriptions file removed, cleared dynamic routes")
            return
        try:
            mtime = subs_path.stat().st_mtime
            if mtime <= self._dynamic_routes_mtime:
                return  # No change
            data = json.loads(subs_path.read_text(encoding="utf-8"))
            if not isinstance(data, dict):
                return
            # Merge: static routes take precedence over dynamic ones.
            # Reject any dynamic route whose effective secret is empty —
            # an empty secret would cause _handle_webhook to skip HMAC
            # validation entirely, letting unauthenticated callers in.
            new_dynamic: Dict[str, dict] = {}
            for k, v in data.items():
                if k in self._static_routes:
                    continue
                effective_secret = v.get("secret", self._global_secret)
                if not effective_secret:
                    logger.warning(
                        "[webhook] Dynamic route '%s' skipped: 'secret' is "
                        "missing or empty. Set a valid HMAC secret, or use "
                        "'%s' to explicitly disable auth (testing only).",
                        k,
                        _INSECURE_NO_AUTH,
                    )
                    continue
                if (
                    effective_secret == _INSECURE_NO_AUTH
                    and not _is_loopback_host(self._host)
                ):
                    logger.warning(
                        "[webhook] Dynamic route '%s' skipped: INSECURE_NO_AUTH "
                        "is only allowed on loopback hosts. Current host: '%s'.",
                        k,
                        self._host,
                    )
                    continue
                new_dynamic[k] = v
            self._dynamic_routes = new_dynamic
            self._routes = {**self._dynamic_routes, **self._static_routes}
            self._dynamic_routes_mtime = mtime
            logger.info(
                "[webhook] Reloaded %d dynamic route(s): %s",
                len(self._dynamic_routes),
                ", ".join(self._dynamic_routes.keys()) or "(none)",
            )
        except Exception as e:
            logger.error("[webhook] Failed to reload dynamic routes: %s", e)

    async def _handle_webhook(self, request: "web.Request") -> "web.Response":
        """POST /webhooks/{route_name} — receive and process a webhook event."""
        # Hot-reload dynamic subscriptions on each request (mtime-gated, cheap)
        self._reload_dynamic_routes()

        route_name = request.match_info.get("route_name", "")
        route_config = self._routes.get(route_name)

        if not route_config:
            return web.json_response(
                {"error": f"Unknown route: {route_name}"}, status=404
            )

        # ── Auth-before-body ─────────────────────────────────────
        # Check Content-Length before reading the full payload.
        content_length = request.content_length or 0
        if content_length > self._max_body_bytes:
            return web.json_response(
                {"error": "Payload too large"}, status=413
            )

        # Read body (must be done before any validation)
        try:
            raw_body = await request.read()
        except Exception as e:
            logger.error("[webhook] Failed to read body: %s", e)
            return web.json_response({"error": "Bad request"}, status=400)

        # Validate HMAC signature FIRST (skip only for the explicit local-test
        # INSECURE_NO_AUTH mode). Missing/empty secrets must fail closed here,
        # not only during connect(), so direct handler reuse cannot turn a
        # network webhook route into an unauthenticated agent-dispatch surface.
        secret = route_config.get("secret", self._global_secret)
        if not secret:
            logger.error(
                "[webhook] Route %s has no HMAC secret; refusing request",
                route_name,
            )
            return web.json_response(
                {"error": "Webhook route is missing an HMAC secret"},
                status=403,
            )
        if secret != _INSECURE_NO_AUTH:
            if not self._validate_signature(request, raw_body, secret):
                logger.warning(
                    "[webhook] Invalid signature for route %s", route_name
                )
                return web.json_response(
                    {"error": "Invalid signature"}, status=401
                )

        # ── Rate limiting (after auth) ───────────────────────────
        now = time.time()
        window = self._rate_counts.setdefault(route_name, [])
        window[:] = [t for t in window if now - t < 60]
        if len(window) >= self._rate_limit:
            return web.json_response(
                {"error": "Rate limit exceeded"}, status=429
            )
        window.append(now)

        # Parse payload
        try:
            payload = json.loads(raw_body)
        except json.JSONDecodeError:
            # Try form-encoded as fallback
            try:
                import urllib.parse

                payload = dict(
                    urllib.parse.parse_qsl(raw_body.decode("utf-8"))
                )
            except Exception:
                return web.json_response(
                    {"error": "Cannot parse body"}, status=400
                )

        # Check event type filter
        event_type = (
            request.headers.get("X-GitHub-Event", "")
            or request.headers.get("X-GitLab-Event", "")
            or payload.get("event_type", "")
            or payload.get("type", "")
            or "unknown"
        )
        allowed_events = route_config.get("events", [])
        if allowed_events and event_type not in allowed_events:
            logger.debug(
                "[webhook] Ignoring event %s for route %s (allowed: %s)",
                event_type,
                route_name,
                allowed_events,
            )
            return web.json_response(
                {"status": "ignored", "event": event_type}
            )

        # Format prompt from template
        prompt_template = route_config.get("prompt", "")
        prompt = self._render_prompt(
            prompt_template, payload, event_type, route_name
        )

        # Inject skill content if configured.
        # We call build_skill_invocation_message() directly rather than
        # using /skill-name slash commands — the gateway's command parser
        # would intercept those and break the flow.
        skills = route_config.get("skills", [])
        if skills:
            try:
                from agent.skill_commands import (
                    build_skill_invocation_message,
                    get_skill_commands,
                )

                skill_cmds = get_skill_commands()
                for skill_name in skills:
                    cmd_key = f"/{skill_name}"
                    if cmd_key in skill_cmds:
                        skill_content = build_skill_invocation_message(
                            cmd_key, user_instruction=prompt
                        )
                        if skill_content:
                            prompt = skill_content
                            break  # Load the first matching skill
                    else:
                        logger.warning(
                            "[webhook] Skill '%s' not found", skill_name
                        )
            except Exception as e:
                logger.warning("[webhook] Skill loading failed: %s", e)

        # Build a unique delivery ID
        delivery_id = request.headers.get(
            "X-GitHub-Delivery",
            request.headers.get(
                "svix-id",
                request.headers.get("X-Request-ID", str(int(time.time() * 1000))),
            ),
        )

        # ── Idempotency ─────────────────────────────────────────
        # Skip duplicate deliveries (webhook retries).
        now = time.time()
        # Prune expired entries
        self._seen_deliveries = {
            k: v
            for k, v in self._seen_deliveries.items()
            if now - v < self._idempotency_ttl
        }
        if delivery_id in self._seen_deliveries:
            logger.info(
                "[webhook] Skipping duplicate delivery %s", delivery_id
            )
            return web.json_response(
                {"status": "duplicate", "delivery_id": delivery_id},
                status=200,
            )
        self._seen_deliveries[delivery_id] = now

        # ── Notion route policy ───────────────────────────────────
        # Our notion-sync route is intentionally asymmetric:
        # comment.created/comment.updated wake the agent immediately;
        # comment.deleted is ignored; non-comment Notion events are queued for
        # hourly context sync so page/database churn does not spawn agents.
        if route_config.get("notion_routing"):
            notion_result = self._handle_notion_route_event(
                route_name=route_name,
                route_config=route_config,
                payload=payload,
                event_type=event_type,
                delivery_id=delivery_id,
                prompt=prompt,
            )
            if notion_result is not None:
                return notion_result

        # ── Direct delivery mode (deliver_only) ─────────────────
        # Skip the agent entirely — the rendered prompt IS the message we
        # deliver.  Use case: external services (Supabase, monitoring,
        # cron jobs, other agents) that need to push a plain notification
        # to a user's chat with zero LLM cost.  Reuses the same HMAC auth,
        # rate limiting, idempotency, and template rendering as agent mode.
        if route_config.get("deliver_only"):
            delivery = {
                "deliver": route_config.get("deliver", "log"),
                "deliver_extra": self._render_delivery_extra(
                    route_config.get("deliver_extra", {}), payload
                ),
                "payload": payload,
            }
            logger.info(
                "[webhook] direct-deliver event=%s route=%s target=%s msg_len=%d delivery=%s",
                event_type,
                route_name,
                delivery["deliver"],
                len(prompt),
                delivery_id,
            )
            try:
                result = await self._direct_deliver(prompt, delivery)
            except Exception:
                logger.exception(
                    "[webhook] direct-deliver failed route=%s delivery=%s",
                    route_name,
                    delivery_id,
                )
                return web.json_response(
                    {"status": "error", "error": "Delivery failed", "delivery_id": delivery_id},
                    status=502,
                )

            if result.success:
                return web.json_response(
                    {
                        "status": "delivered",
                        "route": route_name,
                        "target": delivery["deliver"],
                        "delivery_id": delivery_id,
                    },
                    status=200,
                )
            # Delivery attempted but target rejected it — surface as 502
            # with a generic error (don't leak adapter-level detail).
            logger.warning(
                "[webhook] direct-deliver target rejected route=%s target=%s error=%s",
                route_name,
                delivery["deliver"],
                result.error,
            )
            return web.json_response(
                {"status": "error", "error": "Delivery failed", "delivery_id": delivery_id},
                status=502,
            )

        # Use delivery_id in session key so concurrent webhooks on the
        # same route get independent agent runs (not queued/interrupted).
        session_chat_id = f"webhook:{route_name}:{delivery_id}"

        # Store delivery info for send().  Read by every send() invocation
        # for this chat_id (interim status messages and the final response),
        # so we do NOT pop on send.  TTL-based cleanup keeps the dict bounded.
        deliver_config = {
            "deliver": route_config.get("deliver", "log"),
            "deliver_extra": self._render_delivery_extra(
                route_config.get("deliver_extra", {}), payload
            ),
            "payload": payload,
        }

        from gateway import webhook_queue

        queue_record = webhook_queue.make_record(
            route_name=route_name,
            delivery_id=delivery_id,
            event_type=event_type,
            payload=payload,
            prompt=prompt,
            deliver_config=deliver_config,
            message_id=delivery_id,
        )
        queue_id = webhook_queue.enqueue(queue_record)
        webhook_queue.mark_inflight(queue_id, now=now)
        self._queue_record_by_chat_id[session_chat_id] = queue_id

        self._delivery_info[session_chat_id] = deliver_config
        self._delivery_info_created[session_chat_id] = now
        self._prune_delivery_info(now)

        card_memory_context = self._load_card_memory_context(route_config, payload, route_name)
        if card_memory_context:
            prompt = f"{card_memory_context}\n\n{prompt}"
            self._card_memory_info[session_chat_id] = {
                "route": route_name,
                "payload": payload,
                "card_memory": route_config.get("card_memory"),
            }

        # Build source and event
        source = self.build_source(
            chat_id=session_chat_id,
            chat_name=f"webhook/{route_name}",
            chat_type="webhook",
            user_id=f"webhook:{route_name}",
            user_name=route_name,
        )
        event = MessageEvent(
            text=prompt,
            message_type=MessageType.TEXT,
            source=source,
            raw_message=payload,
            message_id=delivery_id,
        )

        logger.info(
            "[webhook] %s event=%s route=%s prompt_len=%d delivery=%s",
            request.method,
            event_type,
            route_name,
            len(prompt),
            delivery_id,
        )

        # Non-blocking — return 202 Accepted immediately
        task = asyncio.create_task(self.handle_message(event))
        self._background_tasks.add(task)
        task.add_done_callback(self._background_tasks.discard)

        return web.json_response(
            {
                "status": "accepted",
                "route": route_name,
                "event": event_type,
                "delivery_id": delivery_id,
            },
            status=202,
        )

    # ------------------------------------------------------------------
    # Signature validation
    # ------------------------------------------------------------------

    def _validate_signature(
        self, request: "web.Request", body: bytes, secret: str
    ) -> bool:
        """Validate webhook signature (GitHub, GitLab, Svix, generic HMAC-SHA256)."""
        def _header(name: str) -> str:
            return (
                request.headers.get(name, "")
                or request.headers.get(name.lower(), "")
                or request.headers.get(name.upper(), "")
            )

        # Svix / AgentMail:
        #   svix-id: msg_...
        #   svix-timestamp: unix seconds
        #   svix-signature: v1,<base64-hmac> [v1,<base64-hmac> ...]
        # Signed content is: "{id}.{timestamp}.{raw_body}".  Svix secrets
        # usually start with "whsec_" and the remainder is base64-encoded.
        svix_id = _header("svix-id")
        svix_timestamp = _header("svix-timestamp")
        svix_signature = _header("svix-signature")
        if svix_id or svix_timestamp or svix_signature:
            return self._validate_svix_signature(
                body=body,
                secret=secret,
                msg_id=svix_id,
                timestamp=svix_timestamp,
                signature_header=svix_signature,
            )

        # GitHub: X-Hub-Signature-256 = sha256=<hex>
        gh_sig = request.headers.get("X-Hub-Signature-256", "")
        if gh_sig:
            expected = "sha256=" + hmac.new(
                secret.encode(), body, hashlib.sha256
            ).hexdigest()
            return hmac.compare_digest(gh_sig, expected)

        # GitLab: X-Gitlab-Token = <plain secret>
        gl_token = request.headers.get("X-Gitlab-Token", "")
        if gl_token:
            return hmac.compare_digest(gl_token, secret)

        # Notion: X-Notion-Signature = sha256=<hex HMAC-SHA256(secret, raw_body)>
        notion_sig = _header("X-Notion-Signature")
        if notion_sig:
            expected = "sha256=" + hmac.new(
                secret.encode(), body, hashlib.sha256
            ).hexdigest()
            return hmac.compare_digest(notion_sig, expected)

        # Generic: X-Webhook-Signature = <hex HMAC-SHA256>
        generic_sig = request.headers.get("X-Webhook-Signature", "")
        if generic_sig:
            expected = hmac.new(
                secret.encode(), body, hashlib.sha256
            ).hexdigest()
            return hmac.compare_digest(generic_sig, expected)

        # No recognised signature header but secret is configured → reject
        logger.debug(
            "[webhook] Secret configured but no signature header found"
        )
        return False

    def _validate_svix_signature(
        self,
        body: bytes,
        secret: str,
        msg_id: str,
        timestamp: str,
        signature_header: str,
        tolerance_seconds: int = 300,
    ) -> bool:
        """Validate Svix-compatible signatures used by AgentMail webhooks."""
        if not (msg_id and timestamp and signature_header and secret):
            return False

        try:
            ts = int(timestamp)
        except (TypeError, ValueError):
            return False
        if abs(int(time.time()) - ts) > tolerance_seconds:
            logger.warning("[webhook] Svix signature timestamp outside replay window")
            return False

        if secret.startswith("whsec_"):
            encoded_secret = secret.removeprefix("whsec_")
            try:
                key = base64.b64decode(encoded_secret, validate=True)
            except (binascii.Error, ValueError):
                logger.debug("[webhook] Invalid whsec_ Svix signing secret")
                return False
        else:
            # Be permissive for providers that document Svix-style headers but
            # hand out raw shared secrets rather than whsec_ base64 secrets.
            logger.debug("[webhook] Validating Svix-style signature with raw secret")
            key = secret.encode()

        signed_content = msg_id.encode() + b"." + timestamp.encode() + b"." + body
        expected = base64.b64encode(
            hmac.new(key, signed_content, hashlib.sha256).digest()
        ).decode()

        # Svix can send multiple signatures separated by spaces during secret
        # rotation. Each entry is formatted as "vN,<base64>".
        for part in signature_header.split():
            try:
                version, signature = part.split(",", 1)
            except ValueError:
                continue
            if version == "v1" and hmac.compare_digest(signature, expected):
                return True
        return False

    # ------------------------------------------------------------------
    # Prompt rendering
    # ------------------------------------------------------------------

    def _render_prompt(
        self,
        template: str,
        payload: dict,
        event_type: str,
        route_name: str,
    ) -> str:
        """Render a prompt template with the webhook payload.

        Supports dot-notation access into nested dicts:
        ``{pull_request.title}`` → ``payload["pull_request"]["title"]``

        Special token ``{__raw__}`` dumps the entire payload as indented
        JSON (truncated to 4000 chars).  Useful for monitoring alerts or
        any webhook where the agent needs to see the full payload.
        """
        if not template:
            truncated = json.dumps(payload, indent=2)[:4000]
            return (
                f"Webhook event '{event_type}' on route "
                f"'{route_name}':\n\n```json\n{truncated}\n```"
            )

        def _resolve(match: re.Match) -> str:
            key = match.group(1)
            # Special token: dump the entire payload as JSON
            if key == "__raw__":
                return json.dumps(payload, indent=2)[:4000]
            value: Any = payload
            for part in key.split("."):
                if isinstance(value, dict):
                    value = value.get(part, f"{{{key}}}")
                else:
                    return f"{{{key}}}"
            if isinstance(value, (dict, list)):
                return json.dumps(value, indent=2)[:2000]
            return str(value)

        return re.sub(r"\{([a-zA-Z0-9_.]+)\}", _resolve, template)


    # ------------------------------------------------------------------
    # Card-scoped memory for Fizzy-style board/card webhooks
    # ------------------------------------------------------------------

    def _card_memory_config(self, route_config: dict) -> dict:
        cfg = route_config.get("card_memory")
        if isinstance(cfg, dict):
            enabled = cfg.get("enabled", True)
            if enabled is False:
                return {}
            return cfg
        if cfg is True:
            return {"enabled": True, "dir": "~/.hermes/data/card-memory"}
        # Fizzy routes should have durable card memory by default.  This makes
        # future Fizzy boards safe even if the route creator forgets to add the
        # explicit card_memory stanza.
        if route_config.get("fizzy") or str(route_config.get("source", "")).lower() == "fizzy":
            return {"enabled": True, "dir": "~/.hermes/data/card-memory"}
        return {}

    def _extract_card_memory_key(self, payload: dict, route_name: str) -> str:
        def _get(obj: Any, path: tuple[str, ...]) -> Any:
            cur = obj
            for part in path:
                if not isinstance(cur, dict):
                    return None
                cur = cur.get(part)
            return cur

        board_id = (
            _get(payload, ("board", "id"))
            or _get(payload, ("data", "board", "id"))
            or payload.get("board_id")
            or payload.get("boardId")
            or route_name
        )
        card_id = (
            _get(payload, ("card", "id"))
            or _get(payload, ("data", "card", "id"))
            or payload.get("card_id")
            or payload.get("cardId")
            or payload.get("id")
        )
        if not card_id:
            return ""

        def _safe(value: Any) -> str:
            return re.sub(r"[^a-zA-Z0-9_.-]+", "-", str(value)).strip(".-")[:120]

        return f"{_safe(board_id)}__{_safe(card_id)}"

    def _card_memory_dir(self, cfg: dict) -> Path:
        raw = cfg.get("dir") or "~/.hermes/data/card-memory"
        return Path(os.path.expanduser(str(raw))).resolve()

    def _load_card_memory_context(self, route_config: dict, payload: dict, route_name: str) -> str:
        cfg = self._card_memory_config(route_config)
        if not cfg:
            return ""
        key = self._extract_card_memory_key(payload, route_name)
        if not key:
            return ""
        path = self._card_memory_dir(cfg) / f"{key}.md"
        content = ""
        try:
            if path.exists():
                content = path.read_text(encoding="utf-8")[-12000:]
        except Exception:
            logger.exception("[webhook] failed to read card memory %s", path)
            content = ""
        return (
            "Persistent card memory is enabled for this board/card. "
            "Use it as the continuity log for prior work on this card. "
            "At the end of your final response, include an HTML comment exactly like "
            "<!-- CARD_MEMORY_UPDATE: ... --> with concise durable progress/state for future instances.\n\n"
            f"Card memory file: {path}\n"
            f"Existing memory:\n{content if content else '(none yet)'}"
        )

    def _persist_card_memory_update(self, chat_id: str, content: str) -> None:
        info = self._card_memory_info.get(chat_id)
        if not info or not content:
            return
        match = re.search(r"<!--\s*CARD_MEMORY_UPDATE:\s*(.*?)\s*-->", content, re.DOTALL | re.IGNORECASE)
        if not match:
            return
        update = match.group(1).strip()
        if not update:
            return
        raw_cfg = info.get("card_memory")
        if raw_cfg is True:
            cfg = {"enabled": True, "dir": "~/.hermes/data/card-memory"}
        elif isinstance(raw_cfg, dict):
            cfg = raw_cfg if raw_cfg.get("enabled", True) is not False else {}
        else:
            cfg = {"enabled": True, "dir": "~/.hermes/data/card-memory"}
        if not cfg:
            return
        key = self._extract_card_memory_key(info.get("payload") or {}, info.get("route") or "webhook")
        if not key:
            return
        path = self._card_memory_dir(cfg) / f"{key}.md"
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            ts = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            prior = path.read_text(encoding="utf-8") if path.exists() else ""
            entry = f"\n\n## {ts}\n{update}\n"
            path.write_text((prior + entry).strip() + "\n", encoding="utf-8")
            logger.info("[webhook] persisted card memory update %s", path)
        except Exception:
            logger.exception("[webhook] failed to persist card memory %s", path)

    def _render_delivery_extra(
        self, extra: dict, payload: dict
    ) -> dict:
        """Render delivery_extra template values with payload data."""
        rendered: Dict[str, Any] = {}
        for key, value in extra.items():
            if isinstance(value, str):
                rendered[key] = self._render_prompt(value, payload, "", "")
            else:
                rendered[key] = value
        return rendered

    # ------------------------------------------------------------------
    # Response delivery
    # ------------------------------------------------------------------

    async def _direct_deliver(
        self, content: str, delivery: dict
    ) -> SendResult:
        """Deliver *content* directly without invoking the agent.

        Used by ``deliver_only`` routes: the rendered template becomes the
        literal message body, and we dispatch to the same delivery helpers
        that the agent-mode ``send()`` flow uses.  All target types that
        work in agent mode work here — Telegram, Discord, Slack, GitHub
        PR comments, etc.
        """
        deliver_type = delivery.get("deliver", "log")

        if deliver_type == "log":
            # Shouldn't reach here — startup validation rejects deliver_only
            # with deliver=log — but guard defensively.
            logger.info("[webhook] direct-deliver log-only: %s", content[:200])
            return SendResult(success=True)

        if deliver_type == "github_comment":
            return await self._deliver_github_comment(content, delivery)

        # Fall through to the cross-platform dispatcher, which validates the
        # target name and routes via the gateway runner.
        return await self._deliver_cross_platform(
            deliver_type, content, delivery
        )

    async def _deliver_github_comment(
        self, content: str, delivery: dict
    ) -> SendResult:
        """Post agent response as a GitHub PR/issue comment via ``gh`` CLI."""
        extra = delivery.get("deliver_extra", {})
        repo = extra.get("repo", "")
        pr_number = extra.get("pr_number", "")

        if not repo or not pr_number:
            logger.error(
                "[webhook] github_comment delivery missing repo or pr_number"
            )
            return SendResult(
                success=False, error="Missing repo or pr_number"
            )

        try:
            result = subprocess.run(
                [
                    "gh",
                    "pr",
                    "comment",
                    str(pr_number),
                    "--repo",
                    repo,
                    "--body",
                    content,
                ],
                capture_output=True,
                text=True,
                timeout=30,
            )
            if result.returncode == 0:
                logger.info(
                    "[webhook] Posted comment on %s#%s", repo, pr_number
                )
                return SendResult(success=True)
            else:
                logger.error(
                    "[webhook] gh pr comment failed: %s", result.stderr
                )
                return SendResult(success=False, error=result.stderr)
        except FileNotFoundError:
            logger.error(
                "[webhook] 'gh' CLI not found — install GitHub CLI for "
                "github_comment delivery"
            )
            return SendResult(
                success=False, error="gh CLI not installed"
            )
        except Exception as e:
            logger.error("[webhook] github_comment delivery error: %s", e)
            return SendResult(success=False, error=str(e))

    async def _deliver_cross_platform(
        self, platform_name: str, content: str, delivery: dict
    ) -> SendResult:
        """Route response to another platform (telegram, discord, etc.)."""
        if not self.gateway_runner:
            return SendResult(
                success=False,
                error="No gateway runner for cross-platform delivery",
            )

        try:
            target_platform = Platform(platform_name)
        except ValueError:
            return SendResult(
                success=False, error=f"Unknown platform: {platform_name}"
            )

        adapter = self.gateway_runner.adapters.get(target_platform)
        if not adapter:
            return SendResult(
                success=False,
                error=f"Platform {platform_name} not connected",
            )

        # Use home channel if no specific chat_id in deliver_extra
        extra = delivery.get("deliver_extra", {})
        chat_id = extra.get("chat_id", "")
        if not chat_id:
            home = self.gateway_runner.config.get_home_channel(target_platform)
            if home:
                chat_id = home.chat_id
            else:
                return SendResult(
                    success=False,
                    error=f"No chat_id or home channel for {platform_name}",
                )

        # Pass thread_id from deliver_extra so Telegram forum topics work
        metadata = None
        thread_id = extra.get("message_thread_id") or extra.get("thread_id")
        if thread_id:
            metadata = {"thread_id": thread_id}

        return await adapter.send(chat_id, content, metadata=metadata)
