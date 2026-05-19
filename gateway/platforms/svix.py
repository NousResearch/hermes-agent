"""Svix polling platform adapter.

Consumes webhook events through Svix's polling-endpoint API rather than
hosting an HTTP server. Useful when the machine running Hermes can't
expose a public ingress (laptops behind NAT, dev boxes, restricted
networks): Svix receives the upstream webhook, the adapter polls Svix
for new messages, and the events flow through the same route → prompt
→ delivery pipeline as the generic ``webhook`` adapter.

Configuration lives in config.yaml under platforms.svix.extra.routes.
Each route declares the polling endpoint to consume from plus the
prompt/skills/delivery contract used by the agent:

    platforms:
      svix:
        enabled: true
        extra:
          poll_interval: 5    # seconds to wait after Svix reports caught-up
          poll_limit: 50      # max messages fetched per poll request
          routes:
            github_events:
              url: https://api.svix.com/api/v1/app/app_xxx/poller/poll_yyy/
              # Per-route auth: either inline, or pulled from an env var.
              # Required when the polling endpoint uses an endpoint-scoped
              # token (``sk_endp_*``) instead of an account-wide token.
              auth_token_env: SVIX_GITHUB_INGEST_TOKEN
              events: ["issues.opened"]   # optional eventType filter
              prompt: "GitHub issue opened: {issue.title}\\n{__raw__}"
              skills: []
              deliver: telegram
              deliver_extra:
                chat_id: "-1001234567890"

Auth resolution order (per route):
  1. ``route.auth_token`` literal in config.yaml
  2. ``route.auth_token_env`` env var lookup

The Svix CLI (`brew install svix/svix/svix-cli`, `svix login`) is the
recommended way to explore Ingest sources and obtain endpoint-scoped
tokens (``sk_endp_*``): ``svix ingest source list`` shows all sources;
``svix ingest source get <source_id>`` shows the polling URL and token.

Cursor tracking (see https://docs.svix.com/advanced-endpoints/polling-endpoints):
  - Each route polls with a stable **consumer ID** (``hermes-<route>``).
    Svix tracks that consumer's position server-side, so an interrupted
    process resumes where it left off even if local state is lost.
  - The iterator returned by each poll is kept **in memory only** for the
    life of the process; it is not persisted to disk. On (re)start the
    first request omits the iterator so the server resumes from the
    consumer's tracked position — this avoids ever re-seeding from a stale
    on-disk cursor that could be out-of-sequence (e.g. after a ``seek``)
    and wedge the route in a permanent error-retry loop. ``done == False``
    means "more pages, keep polling now" (an empty ``data`` page is normal
    during catch-up); ``done == True`` means caught up — wait
    ``poll_interval`` before the next request.
"""

import asyncio
import logging
import os
import re
import time
from typing import Any, Dict, Optional
from urllib.parse import urlparse

try:
    from svix.api import (
        MessagePollerConsumerPollOptions,
        SvixAsync,
        SvixOptions,
    )
    SVIX_AVAILABLE = True
except ImportError:
    SVIX_AVAILABLE = False
    SvixAsync = None  # type: ignore[assignment]
    SvixOptions = None  # type: ignore[assignment]
    MessagePollerConsumerPollOptions = None  # type: ignore[assignment]


def _try_import_svix() -> bool:
    """Re-attempt the svix import and rebind module-level aliases on success.

    Called from ``check_svix_requirements`` after a lazy-install pass so the
    adapter can be instantiated without restarting the gateway.
    """
    global SVIX_AVAILABLE, SvixAsync, SvixOptions, MessagePollerConsumerPollOptions
    if SVIX_AVAILABLE:
        return True
    try:
        from svix.api import (
            MessagePollerConsumerPollOptions as _MPCPO,
            SvixAsync as _SA,
            SvixOptions as _SO,
        )
    except ImportError:
        return False
    SvixAsync = _SA
    SvixOptions = _SO
    MessagePollerConsumerPollOptions = _MPCPO
    SVIX_AVAILABLE = True
    return True

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import (
    BasePlatformAdapter,
    MessageEvent,
    MessageType,
)
from gateway.platforms.helpers import WebhookDeliveryMixin

logger = logging.getLogger(__name__)

_DEFAULT_POLL_INTERVAL = 5.0
_DEFAULT_POLL_LIMIT = 50
# Fallback sleep used only when a route's poll_interval is configured to 0
# (or otherwise falsy); the normal caught-up wait is self._poll_interval.
_CAUGHT_UP_SLEEP = 5.0
# Polling-endpoint URL shape: /api/v1/app/{app_id}/poller/{sink_id}/
_URL_PATH_RE = re.compile(
    r"/api/v\d+/app/(?P<app_id>[^/]+)/poller/(?P<sink_id>[^/]+)/?",
)


def check_svix_requirements() -> bool:
    """Return True when the Svix SDK is importable.

    Auth-token presence isn't checked here: each route supplies its own
    endpoint-scoped token via ``auth_token`` / ``auth_token_env``.
    The adapter raises a clear error at ``connect()`` time if a route has
    no resolvable token.

    If the Svix SDK is missing, attempts to lazy-install it via
    ``tools.lazy_deps.ensure("platform.svix")`` (mirrors the Telegram /
    Discord / Slack pattern). After a successful install the module-level
    SDK aliases are rebound so the adapter can be constructed without
    restarting the gateway.
    """
    if not SVIX_AVAILABLE:
        try:
            from tools.lazy_deps import ensure as _lazy_ensure
            _lazy_ensure("platform.svix", prompt=False)
        except Exception:
            return False
        if not _try_import_svix():
            return False
    return True


def _parse_polling_url(url: str) -> tuple[str, str, Optional[str]]:
    """Split a polling-endpoint URL into (app_id, sink_id, server_url).

    Accepts both ``https://api.svix.com/api/v1/app/app_xxx/poller/poll_yyy/``
    and ``https://api.eu.svix.com/...`` so self-hosted / region-specific
    deployments work without an extra config knob. The server_url is the
    scheme+host (no path); None when the URL lacks one (relative form).
    """
    parsed = urlparse(url)
    match = _URL_PATH_RE.search(parsed.path or "")
    if not match:
        raise ValueError(
            f"Invalid Svix polling URL: {url!r}. Expected "
            f"https://<svix-host>/api/v1/app/<app_id>/poller/<sink_id>/"
        )
    server_url = None
    if parsed.scheme and parsed.netloc:
        server_url = f"{parsed.scheme}://{parsed.netloc}"
    return match.group("app_id"), match.group("sink_id"), server_url


class SvixAdapter(WebhookDeliveryMixin, BasePlatformAdapter):
    """Polls Svix polling endpoints and dispatches events to the gateway."""

    # Identifiers used by WebhookDeliveryMixin for logs / fallback rendering.
    _log_tag = "[svix]"
    _event_noun = "Svix"

    def __init__(self, config: PlatformConfig):
        super().__init__(config, Platform.SVIX)
        self._poll_interval: float = float(
            config.extra.get("poll_interval", _DEFAULT_POLL_INTERVAL)
        )
        self._poll_limit: int = int(
            config.extra.get("poll_limit", _DEFAULT_POLL_LIMIT)
        )
        self._routes: Dict[str, dict] = config.extra.get("routes", {}) or {}

        self._iterators: Dict[str, str] = {}

        # Polling task per route.
        self._poll_tasks: Dict[str, asyncio.Task] = {}

        # Per-route Svix clients (one per server_url, but cheap to keep per route).
        self._clients: Dict[str, Any] = {}
        self._client_endpoints: Dict[str, tuple[str, str]] = {}  # route → (app_id, sink_id)
        # Server-tracked consumer ID per route. Lets the Svix server remember
        # our cursor position so an interrupted process resumes where it left
        # off even if the local iterator file is lost.
        self._consumer_ids: Dict[str, str] = {}

        # Cross-platform delivery (set by GatewayRunner._create_adapter).
        self.gateway_runner = None

        # Delivery info keyed by session chat_id, same lifecycle as webhook.py:
        # populated when an event fires, read by every send() invocation,
        # TTL-pruned to keep the dict bounded.
        self._delivery_info: Dict[str, dict] = {}
        self._delivery_info_created: Dict[str, float] = {}
        self._delivery_info_ttl: int = 3600  # 1 hour

        # In-process dedup of message IDs (defense-in-depth against
        # accidental cursor non-advancement; the iterator already handles
        # the normal case).
        self._seen_message_ids: Dict[str, float] = {}

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def _resolve_route_token(self, name: str, route: dict) -> str:
        """Pick the auth token for a route.

        Precedence: route-literal ``auth_token`` > route ``auth_token_env``.
        Returns an empty string and logs a warning when neither field is set;
        ``connect()`` will then surface this as a fatal startup error.
        """
        literal = route.get("auth_token") or ""
        if literal:
            return str(literal)
        env_name = route.get("auth_token_env")
        if env_name:
            val = os.getenv(str(env_name), "")
            if not val:
                raise ValueError(
                    f"[svix] Route '{name}' sets auth_token_env={env_name!r} but "
                    f"that environment variable is empty. Export it or set a "
                    f"literal 'auth_token' on the route."
                )
            return val
        logger.warning(
            "[svix] Route '%s' has no 'auth_token' or 'auth_token_env'. "
            "Set one of these on the route (endpoint-scoped 'sk_endp_*' token "
            "from the Svix dashboard).",
            name,
        )
        return ""

    async def connect(self) -> bool:
        """Validate routes, wire per-route clients, and start polling.

        Returns ``False`` on permanent misconfiguration (missing SDK, bad
        polling URL, unresolvable auth token, ``deliver_only`` with no real
        target) after marking a *non-retryable* fatal error. These won't fix
        themselves on a 30s reconnect, so they must not be reported as
        transient — that is the same contract the SDK-missing path uses, and
        keeps ``connect()`` returning a bool rather than raising for the
        caller in ``_connect_adapter_with_timeout`` to interpret.
        """
        if not SVIX_AVAILABLE:
            msg = "[svix] svix SDK not installed. Run: pip install svix"
            logger.error(msg)
            self._set_fatal_error(
                "svix_missing_dependency", msg, retryable=False
            )
            return False
        if not self._routes:
            logger.warning(
                "[svix] No routes configured under platforms.svix.extra.routes — "
                "adapter will be idle."
            )

        # Validate every route up front and prepare per-route clients so
        # misconfiguration surfaces at startup rather than on the first poll.
        try:
            for name, route in self._routes.items():
                url = route.get("url")
                if not url:
                    raise ValueError(
                        f"[svix] Route '{name}' has no 'url'. Each route must "
                        f"declare its Svix polling endpoint URL."
                    )
                try:
                    app_id, sink_id, server_url = _parse_polling_url(url)
                except ValueError as exc:
                    raise ValueError(f"[svix] Route '{name}': {exc}") from exc

                token = self._resolve_route_token(name, route)
                if not token:
                    raise ValueError(
                        f"[svix] Route '{name}' has no auth token. Set "
                        f"'auth_token' or 'auth_token_env' on the route. "
                        f"Polling endpoints need an endpoint-scoped 'sk_endp_*' "
                        f"token found in the Svix dashboard."
                    )

                options = SvixOptions(server_url=server_url) if server_url else SvixOptions()
                self._clients[name] = SvixAsync(token, options)
                self._client_endpoints[name] = (app_id, sink_id)
                # Stable per-route consumer ID so the Svix server tracks our
                # position. Configurable, but the default is deterministic so
                # restarts reuse the same consumer (and thus the same cursor).
                self._consumer_ids[name] = f"hermes-{name}"

                if route.get("deliver_only"):
                    deliver = route.get("deliver", "log")
                    if not deliver or deliver == "log":
                        raise ValueError(
                            f"[svix] Route '{name}' has deliver_only=true but "
                            f"deliver is '{deliver}'. Direct delivery requires a "
                            f"real target (telegram, discord, slack, github_comment, etc.)."
                        )
        except ValueError as exc:
            logger.error("%s", exc)
            self._set_fatal_error("svix_invalid_config", str(exc), retryable=False)
            return False

        for name in self._routes:
            task = asyncio.create_task(
                self._poll_route(name), name=f"svix-poll-{name}"
            )
            self._poll_tasks[name] = task

        self._mark_connected()
        logger.info(
            "[svix] Polling %d route(s): %s",
            len(self._routes),
            ", ".join(self._routes.keys()) or "(none)",
        )
        return True

    async def disconnect(self) -> None:
        for task in self._poll_tasks.values():
            task.cancel()
        if self._poll_tasks:
            await asyncio.gather(
                *self._poll_tasks.values(), return_exceptions=True
            )
        self._poll_tasks.clear()
        self._mark_disconnected()
        logger.info("[svix] Disconnected")

    # ------------------------------------------------------------------
    # Polling loop
    # ------------------------------------------------------------------

    async def _poll_route(self, route_name: str) -> None:
        """Long-running poll loop for a single route.

        Uses Svix's server-tracked **consumer** iterator: the first call
        after (re)start omits the iterator so the server resumes from the
        position it last saw for our consumer ID, and every later call
        passes the iterator returned by the previous response to move
        forward. The cursor lives only in memory for the life of this loop;
        it is never persisted, so a restart always falls back to the
        server-tracked consumer position rather than risking a stale local
        cursor.

        Per the Svix docs, ``done == False`` means "keep polling now"
        (the response may legitimately carry an empty ``data`` page during
        catch-up), while ``done == True`` means we're caught up and should
        wait ``poll_interval`` before the next request. Failures back off
        exponentially rather than tight-looping.
        """
        client = self._clients[route_name]
        app_id, sink_id = self._client_endpoints[route_name]
        consumer_id = self._consumer_ids[route_name]
        backoff = 1.0
        max_backoff = 60.0
        iterator: Optional[str] = self._iterators.get(route_name)

        while True:
            try:
                options = MessagePollerConsumerPollOptions(
                    limit=self._poll_limit,
                    iterator=iterator,
                )
                result = await client.message.poller.consumer_poll(
                    app_id, sink_id, consumer_id, options
                )
                backoff = 1.0  # reset after a successful call

                for message in result.data:
                    try:
                        await self._process_message(route_name, message)
                    except Exception:
                        logger.exception(
                            "[svix] Error processing message %s on route %s",
                            getattr(message, "id", "?"),
                            route_name,
                        )

                # Advance our in-memory cursor. Not persisted — the consumer
                # ID handles durable resume across restarts.
                if result.iterator and result.iterator != iterator:
                    iterator = result.iterator
                    self._iterators[route_name] = result.iterator

                if result.done:
                    await asyncio.sleep(self._poll_interval or _CAUGHT_UP_SLEEP)
                # else: loop immediately to drain the next page
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.warning(
                    "[svix] Poll error on route %s: %s (retrying in %.1fs)",
                    route_name,
                    exc,
                    backoff,
                )
                try:
                    await asyncio.sleep(backoff)
                except asyncio.CancelledError:
                    raise
                backoff = min(backoff * 2, max_backoff)

    # ------------------------------------------------------------------
    # Event processing
    # ------------------------------------------------------------------

    async def _process_message(self, route_name: str, message: Any) -> None:
        """Render the prompt, build the event, and dispatch it.

        Mirrors the second half of ``WebhookAdapter._handle_webhook`` —
        event-type filtering, prompt rendering, skill injection, optional
        direct-deliver bypass, and finally ``self.handle_message()`` to
        run the agent.
        """
        route_config = self._routes.get(route_name)
        if not route_config:
            return

        message_id = str(getattr(message, "id", ""))
        event_type = str(getattr(message, "event_type", "") or "unknown")
        payload = getattr(message, "payload", None) or {}

        # Drop already-seen IDs (in-process dedup; iterator advancement
        # is the primary mechanism).
        now = time.time()
        self._seen_message_ids = {
            k: t for k, t in self._seen_message_ids.items()
            if now - t < self._delivery_info_ttl
        }
        if message_id and message_id in self._seen_message_ids:
            logger.debug(
                "[svix] Skipping already-seen message %s on route %s",
                message_id,
                route_name,
            )
            return
        if message_id:
            self._seen_message_ids[message_id] = now

        allowed_events = route_config.get("events") or []
        if allowed_events and event_type not in allowed_events:
            logger.debug(
                "[svix] Ignoring event %s on route %s (allowed: %s)",
                event_type,
                route_name,
                allowed_events,
            )
            return

        prompt = self._render_prompt(
            route_config.get("prompt", ""), payload, event_type, route_name
        )
        prompt = self._inject_skill(prompt, route_config.get("skills", []))

        delivery_id = message_id or str(int(time.time() * 1000))
        deliver_extra = self._render_delivery_extra(
            route_config.get("deliver_extra", {}), payload
        )

        if route_config.get("deliver_only"):
            delivery = {
                "deliver": route_config.get("deliver", "log"),
                "deliver_extra": deliver_extra,
                "payload": payload,
            }
            logger.info(
                "[svix] direct-deliver event=%s route=%s target=%s msg_id=%s",
                event_type,
                route_name,
                delivery["deliver"],
                delivery_id,
            )
            await self._direct_deliver(prompt, delivery)
            return

        session_chat_id = f"svix:{route_name}:{delivery_id}"
        self._delivery_info[session_chat_id] = {
            "deliver": route_config.get("deliver", "log"),
            "deliver_extra": deliver_extra,
            "payload": payload,
        }
        self._delivery_info_created[session_chat_id] = now
        self._prune_delivery_info(now)

        source = self.build_source(
            chat_id=session_chat_id,
            chat_name=f"svix/{route_name}",
            chat_type="webhook",
            user_id=f"svix:{route_name}",
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
            "[svix] event=%s route=%s prompt_len=%d msg_id=%s",
            event_type,
            route_name,
            len(prompt),
            delivery_id,
        )

        task = asyncio.create_task(self.handle_message(event))
        self._background_tasks.add(task)
        task.add_done_callback(self._background_tasks.discard)

    # ------------------------------------------------------------------
    # Send / delivery
    # ------------------------------------------------------------------
    # The render → deliver pipeline (send, _direct_deliver,
    # _deliver_github_comment, _deliver_cross_platform, _render_prompt,
    # _render_delivery_extra, _inject_skill, _prune_delivery_info) is
    # provided by WebhookDeliveryMixin and shared with the webhook adapter.

    async def get_chat_info(self, chat_id: str) -> Dict[str, Any]:
        return {"name": chat_id, "type": "svix"}
