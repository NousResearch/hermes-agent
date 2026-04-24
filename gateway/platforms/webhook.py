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

        # ── Rate limiting ────────────────────────────────────────
        now = time.time()
        window = self._rate_counts.setdefault(route_name, [])
        window[:] = [t for t in window if now - t < 60]
        if len(window) >= self._rate_limit:
            return web.json_response(
                {"error": "Rate limit exceeded"}, status=429
            )
        window.append(now)

        # Read body
        try:
            raw_body = await request.read()
        except Exception as e:
            logger.error("[webhook] Failed to read body: %s", e)
            return web.json_response({"error": "Bad request"}, status=400)

        # Validate HMAC signature (skip for INSECURE_NO_AUTH testing mode)
        secret = route_config.get("secret", self._global_secret)
        if secret and secret != _INSECURE_NO_AUTH:
            if not self._validate_signature(request, raw_body, secret):
                logger.warning(
                    "[webhook] Invalid signature for route %s", route_name
                )
                return web.json_response(
                    {"error": "Invalid signature"}, status=401
                )

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

        # ── Built-in route: code-crab-completion ────────────────
        # Structured handoff callback from Code Crab — handled inline
        # (does NOT go through the normal agent processing flow).
        if route_name == "code-crab-completion":
            return await self._handle_code_crab_completion(payload)

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
    # Built-in handler: Code Crab completion callback
    # ------------------------------------------------------------------

    async def _handle_code_crab_completion(
        self, payload: dict
    ) -> "web.Response":
        """Handle a structured handoff completion callback from Code Crab.

        Validates the payload, updates the handoff record, posts a concise
        return message to the origin thread/channel, and optionally triggers
        auto-resume for safe queued actions.

        This bypasses the normal agent processing flow entirely — no prompt
        rendering, no skill loading, no agent run.
        """
        from gateway.handoffs import (
            get_handoff,
            render_origin_message,
            should_auto_resume,
            update_handoff_status,
            validate_status_transition,
        )

        handoff_id = payload.get("handoff_id")
        status = payload.get("status")

        # Validate required fields
        if not handoff_id or not isinstance(handoff_id, str):
            return web.json_response(
                {"error": "Missing or invalid handoff_id"}, status=400
            )
        if status not in ("done", "blocked", "failed"):
            return web.json_response(
                {"error": f"Invalid status: {status}"}, status=400
            )

        # Look up handoff record
        record = get_handoff(handoff_id)
        if record is None:
            logger.warning(
                "[webhook] Unknown handoff_id in completion callback: %s",
                handoff_id,
            )
            return web.json_response(
                {"error": f"Unknown handoff_id: {handoff_id}"}, status=404
            )

        # Validate status transition
        current_status = record.get("status", "requested")
        if not validate_status_transition(current_status, status):
            return web.json_response(
                {
                    "error": f"Invalid transition: {current_status} -> {status}",
                    "current_status": current_status,
                },
                status=409,
            )

        # Update handoff record
        try:
            record = update_handoff_status(
                handoff_id, status, callback_payload=payload
            )
        except ValueError as e:
            return web.json_response({"error": str(e)}, status=409)

        # Post return message to origin thread/channel
        return_message_id = None
        origin = record.get("origin", {})
        origin_platform = origin.get("platform", "")
        origin_channel_id = origin.get("channel_id", "")

        if origin_platform and origin_channel_id and self.gateway_runner:
            message = render_origin_message(record)
            try:
                target_platform = Platform(origin_platform)
                adapter = self.gateway_runner.adapters.get(target_platform)
                if adapter:
                    metadata = None
                    if origin.get("thread_id"):
                        metadata = {"thread_id": origin["thread_id"]}
                    result = await adapter.send(
                        origin_channel_id, message, metadata=metadata
                    )
                    if result.success:
                        return_message_id = getattr(
                            result, "message_id", None
                        )
                        logger.info(
                            "[webhook] Posted handoff return message to %s/%s",
                            origin_platform,
                            origin_channel_id,
                        )
                    else:
                        logger.error(
                            "[webhook] Failed to post return message: %s",
                            result.error,
                        )
                else:
                    logger.warning(
                        "[webhook] Origin platform %s not connected",
                        origin_platform,
                    )
            except Exception as e:
                logger.error(
                    "[webhook] Error posting return message: %s", e
                )

        # Store return message ID
        if return_message_id:
            try:
                update_handoff_status(
                    handoff_id,
                    status,
                    return_message_id=str(return_message_id),
                )
            except ValueError:
                pass  # Already at terminal status, ignore

        # Check auto-resume
        auto_resume = should_auto_resume(record)
        if auto_resume:
            logger.info(
                "[webhook] Auto-resume eligible for %s: %s",
                handoff_id,
                record.get("next_action", ""),
            )
            # Auto-resume is logged but not executed inline — the gateway
            # supervisor or a separate scheduler should pick this up.
            # For now we include it in the response so callers know.

        return web.json_response(
            {
                "status": "accepted",
                "handoff_id": handoff_id,
                "new_status": status,
                "return_message_posted": return_message_id is not None,
                "return_message_id": return_message_id,
                "auto_resume_eligible": auto_resume,
            }
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
