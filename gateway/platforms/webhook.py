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
import hashlib
import hmac
import json
import logging
import re
import subprocess
import time
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
    SendResult,
)

logger = logging.getLogger(__name__)

DEFAULT_HOST = "0.0.0.0"
DEFAULT_PORT = 8644
_INSECURE_NO_AUTH = "INSECURE_NO_AUTH"
_DYNAMIC_ROUTES_FILENAME = "webhook_subscriptions.json"


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

        # Delivery info keyed by session chat_id.
        #
        # Read by every send() invocation for the chat_id.  For
        # rose_callback deliveries, send() stores the latest content in
        # _pending_callback instead of firing immediately — the actual
        # callback is fired once from the _process_message_background
        # override's finally block.  This ensures interim messages
        # (fallback notifications, context-pressure warnings) don't each
        # produce a callback — only the terminal message does.
        #
        # Cleaned up via TTL on each POST — see _prune_delivery_info().
        self._delivery_info: Dict[str, dict] = {}
        self._delivery_info_created: Dict[str, float] = {}

        # Deferred rose_callback content: chat_id -> (content, metadata).
        # Populated by send(), consumed by _fire_pending_callback().
        self._pending_callback: Dict[str, tuple] = {}

        # Reference to gateway runner for cross-platform delivery (set externally)
        self.gateway_runner = None

        # Idempotency: TTL cache of recently processed delivery IDs.
        # Prevents duplicate agent runs when webhook providers retry.
        self._seen_deliveries: Dict[str, float] = {}
        self._idempotency_ttl: int = 3600  # 1 hour

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
        delivery = self._delivery_info.get(chat_id, {})
        deliver_type = delivery.get("deliver", "log")

        if deliver_type == "log":
            logger.info("[webhook] Response for %s: %s", chat_id, content[:200])
            return SendResult(success=True)

        if deliver_type == "github_comment":
            return await self._deliver_github_comment(content, delivery)

        # Rose Command Centre callback — deferred to task completion.
        # Store the latest content so _fire_pending_callback() (called
        # from _process_message_background override's finally block)
        # sends exactly one callback per job with the terminal content.
        if deliver_type == "rose_callback":
            self._pending_callback[chat_id] = (content, metadata)
            logger.debug(
                "[webhook] rose_callback content stored for %s (%d chars, deferred)",
                chat_id,
                len(content or ""),
            )
            return SendResult(success=True)

        # Cross-platform delivery — any platform with a gateway adapter
        if self.gateway_runner and deliver_type in (
            "telegram",
            "discord",
            "slack",
            "signal",
            "sms",
            "whatsapp",
            "matrix",
            "mattermost",
            "homeassistant",
            "email",
            "dingtalk",
            "feishu",
            "wecom",
            "wecom_callback",
            "weixin",
            "bluebubbles",
            "qqbot",
        ):
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
            self._pending_callback.pop(k, None)

    async def get_chat_info(self, chat_id: str) -> Dict[str, Any]:
        return {"name": chat_id, "type": "webhook"}

    # ------------------------------------------------------------------
    # HTTP handlers
    # ------------------------------------------------------------------

    async def _handle_health(self, request: "web.Request") -> "web.Response":
        """GET /health — simple health check."""
        return web.json_response({"status": "ok", "platform": "webhook"})

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
            # Merge: static routes take precedence over dynamic ones
            self._dynamic_routes = {
                k: v for k, v in data.items()
                if k not in self._static_routes
            }
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

        # Validate HMAC signature FIRST (skip for INSECURE_NO_AUTH testing mode)
        secret = route_config.get("secret", self._global_secret)
        if secret and secret != _INSECURE_NO_AUTH:
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
            request.headers.get("X-Request-ID", str(int(time.time() * 1000))),
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
            # ── rose_callback support (Rose Command Centre integration) ──
            # When deliver == "rose_callback" the agent's response is POSTed
            # back to Rose's callback endpoint with the same HMAC secret used
            # for inbound dispatch.  We capture the inbound request's secret,
            # X-Rose-Request-Id, AND callback_url at receive time so send()
            # can sign and route the callback without spelunking through the
            # nested payload structure later (Rose's dispatch envelope is
            # {job_type, payload: {callback_url, ...}, context} and we don't
            # want to bake that shape into the delivery path).
            "secret": secret,
            "rose_request_id": request.headers.get("X-Rose-Request-Id", ""),
            "delivery_id": delivery_id,
            "started_at": now,
            "callback_url": (
                payload.get("payload", {}).get("callback_url", "")
                if isinstance(payload, dict)
                else ""
            ),
        }
        self._delivery_info[session_chat_id] = deliver_config
        self._delivery_info_created[session_chat_id] = now
        self._prune_delivery_info(now)

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
    # Rose callback: override background processing to fire callback
    # after the terminal send, not after handle_message() returns.
    # handle_message() spawns _process_message_background() as a
    # fire-and-forget task and returns immediately, so a done_callback
    # on handle_message()'s task would fire before the terminal send.
    # ------------------------------------------------------------------

    async def _process_message_background(self, event, session_key: str) -> None:
        chat_id = event.source.chat_id
        delivery = self._delivery_info.get(chat_id, {})
        is_rose_callback = delivery.get("deliver") == "rose_callback"

        if not is_rose_callback:
            return await super()._process_message_background(event, session_key)

        task_error = None
        try:
            await super()._process_message_background(event, session_key)
        except asyncio.CancelledError:
            task_error = "Agent task was cancelled"
            raise
        except Exception as exc:
            task_error = str(exc) or repr(exc)
            raise
        finally:
            try:
                await self._fire_pending_callback(chat_id, task_error=task_error)
            except asyncio.CancelledError:
                logger.warning(
                    "[webhook] Callback delivery interrupted by cancellation for %s; "
                    "pending content retained for TTL cleanup",
                    chat_id,
                )
                raise
            except Exception:
                logger.exception(
                    "[webhook] Failed to fire deferred callback in finally for %s",
                    chat_id,
                )

    # ------------------------------------------------------------------
    # Signature validation
    # ------------------------------------------------------------------

    def _validate_signature(
        self, request: "web.Request", body: bytes, secret: str
    ) -> bool:
        """Validate webhook signature (GitHub, GitLab, generic HMAC-SHA256)."""
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

    async def _fire_pending_callback(
        self, chat_id: str, *, task_error: Optional[str] = None
    ) -> None:
        """Fire the deferred rose_callback after the agent task completes.

        Called from the ``_process_message_background`` override's
        ``finally`` block, so it runs after the terminal send completes
        (not after ``handle_message()`` returns).

        ``task_error`` is set when the processing coroutine raised or
        was cancelled.  If no ``send()`` was ever called and there is
        no error, sends ``status: "failed"`` so Rose doesn't wait.
        """
        delivery = self._delivery_info.get(chat_id, {})
        if not delivery or delivery.get("deliver") != "rose_callback":
            self._pending_callback.pop(chat_id, None)
            return

        if task_error:
            content = task_error
            metadata = {"agent_status": "failed", "error": task_error}
        else:
            pending = self._pending_callback.get(chat_id)
            if pending:
                content, metadata = pending
            else:
                content = "Agent produced no response"
                metadata = {
                    "agent_status": "failed",
                    "error": "No response from agent",
                }

        self._pending_callback[chat_id] = (content, metadata)

        max_attempts = 2
        for attempt in range(max_attempts):
            result = None
            try:
                result = await self._deliver_rose_callback(
                    content, delivery, metadata
                )
            except Exception:
                logger.exception(
                    "[webhook] Callback delivery attempt %d/%d failed for %s",
                    attempt + 1,
                    max_attempts,
                    chat_id,
                )
            if result and result.success:
                self._pending_callback.pop(chat_id, None)
                return
            if attempt < max_attempts - 1:
                try:
                    await asyncio.sleep(1)
                except asyncio.CancelledError:
                    logger.warning(
                        "[webhook] Retry backoff cancelled for %s; "
                        "attempting final delivery before propagating",
                        chat_id,
                    )
                    try:
                        delivery_task = asyncio.create_task(
                            self._deliver_rose_callback(
                                content, delivery, metadata
                            )
                        )
                        self._background_tasks.add(delivery_task)

                        def _on_delivery_done(
                            t: asyncio.Task, _cid: str = chat_id
                        ) -> None:
                            self._background_tasks.discard(t)
                            if not t.cancelled() and not t.exception():
                                r = t.result()
                                if r and r.success:
                                    self._pending_callback.pop(_cid, None)

                        delivery_task.add_done_callback(_on_delivery_done)

                        result = await asyncio.shield(delivery_task)
                        if result and result.success:
                            self._pending_callback.pop(chat_id, None)
                    except asyncio.CancelledError:
                        pass
                    except Exception:
                        logger.exception(
                            "[webhook] Final delivery attempt after cancellation failed for %s",
                            chat_id,
                        )
                    raise

        logger.warning(
            "[webhook] Callback delivery exhausted %d attempts for %s; "
            "pending content retained for TTL cleanup",
            max_attempts,
            chat_id,
        )

    async def _deliver_rose_callback(
        self,
        content: str,
        delivery: dict,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SendResult:
        """POST agent response back to Rose's /api/hermes-callback endpoint.

        Used by Rose Command Centre dispatches (``deliver: rose_callback``).
        Signs the body with HMAC-SHA256 using the same per-route secret that
        verified the inbound dispatch, and includes the Rose-side request ID
        so Rose's callback handler can match the result to its hermes_jobs
        row.

        Body format matches Rose's HermesCallbackPayload type
        (server/hermesClient.ts):
            {
              "job_id": str,
              "rose_request_id": str,
              "status": "completed" | "failed",
              "summary": str,
              "artifacts": list[{type, path?, value?, url?}],
              "duration_ms": int,
              "token_cost_usd": float | null,
              "error": str | null
            }

        Headers:
            X-Hermes-Signature: t=<unix>,sha256=<HMAC of timestamp.body>
            Content-Type:        application/json

        Status detection — the load-bearing change in this method:
            Rose's polling loop in subTaskExecutor.executeHermesTask treats
            ``status == "completed"`` as terminal-success and feeds ``summary``
            straight into packageAssembler.  Reporting ``completed`` for a
            failed agent run causes Rose to surface Hermes errors as if they
            were research output (the "⏳ Retrying in 2.6s..." regression on
            30 Apr 2026).  We must split success vs failure correctly here.

            Detection order, most-trusted first:
              1. metadata["agent_status"] == "failed" or metadata["error"] set
                 → failure (explicit signal from agent runtime, when available)
              2. content head matches a known Hermes failure marker
                 → failure (heuristic, bounded to first 1000 chars so a
                  legitimate research summary that mentions "error" deep
                  inside isn't misclassified)
              3. otherwise → success
        """
        # Direct field reads — populated at receive time in _handle_webhook.
        callback_url = delivery.get("callback_url", "") or delivery.get(
            "deliver_extra", {}
        ).get("callback_url", "")
        secret = delivery.get("secret", "")
        rose_request_id = delivery.get("rose_request_id", "")
        delivery_id = delivery.get("delivery_id", "")
        started_at = delivery.get("started_at")

        if not callback_url:
            logger.error(
                "[webhook] rose_callback delivery missing callback_url"
            )
            return SendResult(
                success=False, error="Missing callback_url"
            )
        if not secret:
            logger.error("[webhook] rose_callback delivery missing secret")
            return SendResult(success=False, error="Missing secret")
        if not rose_request_id:
            logger.error(
                "[webhook] rose_callback delivery missing X-Rose-Request-Id"
            )
            return SendResult(
                success=False, error="Missing rose_request_id"
            )

        # ── Detect agent success vs failure ──
        metadata = metadata or {}
        agent_status = str(metadata.get("agent_status") or "").lower()
        metadata_error = metadata.get("error")

        is_failure = False
        error_message: Optional[str] = None

        if agent_status == "failed" or metadata_error:
            is_failure = True
            error_message = (
                str(metadata_error)
                if metadata_error
                else "Agent reported failure via metadata"
            )
        else:
            # Heuristic content sniff. Bound to first 1000 chars so a
            # legitimate research summary that happens to mention "error"
            # later isn't misclassified.
            content_head = (content or "")[:1000]
            FAILURE_MARKERS = (
                "API call failed",
                "Final error:",
                "Max retries (3) exhausted",
                "Max retries exhausted",
                "⏳ Retrying in",
                "Retrying in ",  # without emoji prefix, defensive
                "❌",
                "☠",
                "Traceback (most recent call last):",
                # HTTP error codes from provider SDK formatting
                "Error code: 4",  # 4xx client errors
                "Error code: 5",  # 5xx server errors
                "⚠️ Error code:",  # with emoji prefix
                # Anthropic/OpenAI-style error type dicts
                "'type': 'error'",
            )
            matched_marker: Optional[str] = None
            for marker in FAILURE_MARKERS:
                if marker in content_head:
                    matched_marker = marker
                    break
            if matched_marker:
                is_failure = True
                # Pull the most informative line for the error field —
                # prefer the "Final error:" line if present, else the line
                # containing the matched marker, else the marker itself.
                lines = [
                    ln.strip()
                    for ln in (content or "").splitlines()
                    if ln.strip()
                ]
                error_message = next(
                    (ln for ln in lines if "Final error:" in ln),
                    next(
                        (ln for ln in lines if matched_marker in ln),
                        matched_marker,
                    ),
                )[:500]

        duration_ms = (
            int((time.time() - started_at) * 1000) if started_at else None
        )

        if is_failure:
            body_obj = {
                "job_id": delivery_id or rose_request_id,
                "rose_request_id": rose_request_id,
                "status": "failed",
                # Keep a truncated summary for debugging visibility — Rose
                # logs this when surfacing the failed sub-task to the user.
                "summary": (content or "")[:2000],
                "artifacts": [],
                "duration_ms": duration_ms,
                "token_cost_usd": None,
                "error": error_message
                or "Agent failed without an error message",
            }
            logger.warning(
                "[webhook] rose_callback marking job %s FAILED: %s",
                rose_request_id,
                (error_message or "no detail")[:200],
            )
        else:
            body_obj = {
                "job_id": delivery_id or rose_request_id,
                "rose_request_id": rose_request_id,
                "status": "completed",
                "summary": content,
                "artifacts": [],
                "duration_ms": duration_ms,
                "token_cost_usd": None,
                "error": None,
            }

        body_bytes = json.dumps(body_obj, ensure_ascii=False).encode("utf-8")
        timestamp = str(int(time.time()))
        signature_input = f"{timestamp}.".encode("utf-8") + body_bytes
        sig_hex = hmac.new(
            secret.encode(), signature_input, hashlib.sha256
        ).hexdigest()
        signature = f"t={timestamp},sha256={sig_hex}"

        headers = {
            "Content-Type": "application/json",
            "X-Hermes-Signature": signature,
        }

        try:
            import aiohttp as _aiohttp  # local import to keep top of file clean
            timeout = _aiohttp.ClientTimeout(total=30)
            async with _aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(
                    callback_url, data=body_bytes, headers=headers
                ) as resp:
                    response_text = await resp.text()
                    if resp.status >= 400:
                        logger.warning(
                            "[webhook] rose_callback target returned HTTP %d: %s",
                            resp.status,
                            response_text[:200],
                        )
                        return SendResult(
                            success=False,
                            error=f"HTTP {resp.status}: {response_text[:200]}",
                        )
                    logger.info(
                        "[webhook] rose_callback delivered to %s (HTTP %d, status=%s)",
                        callback_url,
                        resp.status,
                        body_obj["status"],
                    )
                    return SendResult(success=True)
        except Exception as e:
            logger.exception("[webhook] rose_callback delivery failed")
            return SendResult(success=False, error=str(e))

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
