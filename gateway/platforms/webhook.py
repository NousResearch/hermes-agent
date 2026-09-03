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
  - Generic HMAC supports a V2 signature (X-Webhook-Signature-V2) that
    binds a timestamp into the signed data for replay protection; the
    legacy body-only V1 (X-Webhook-Signature) is deprecated but still
    accepted with a warning, since it has no replay protection
  - A route may instead declare a `signature` block describing the header
    layout its provider uses (see "Route-configurable signature schemes"
    below).  That is exclusive: built-in header probing, including the
    legacy V1 fallback, is skipped for that route
  - Set secret to "INSECURE_NO_AUTH" to skip validation (testing only)
"""

import asyncio
import base64
import binascii
import hashlib
import hmac
import json
import logging
import re
import subprocess
import sys
import time
from collections import deque
from contextlib import nullcontext
from typing import Any, Deque, Dict, List, Optional

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
from gateway.platforms.webhook_filters import (
    DEFAULT_SCRIPT_TIMEOUT_SECONDS,
    WebhookRouteProcessor,
)
from gateway.response_filters import is_autonomous_silence_response

logger = logging.getLogger(__name__)


def _is_webhook_silence_response(content: Any) -> bool:
    """Whether an agent response means "deliberately say nothing".

    Webhook routes are autonomous background lanes: a subscription prompt tells
    the agent to answer with ``[SILENT]`` when a tick produced nothing worth a
    human's attention (a duplicate inbound, a stand-down because a sibling lane
    already replied, a routine close).  Nobody is waiting on the other end, so
    there is no reader for whom a "nothing happened" message is useful.

    The reason this is the loose autonomous rule rather than the live gateway's
    is what the two lanes optimise for.  In an interactive chat, swallowing a
    real answer because it happens to open with a marker is much worse than
    showing a stray marker, so ``is_intentional_silence_response`` demands the
    response be EXACTLY a marker.  A webhook run has the opposite payoff: the
    cost of a leaked non-story is a pointless notification on every tick, and
    models reliably add a sentence explaining why they stayed quiet — which
    under the strict rule flips the whole thing back to "deliver".  That is not
    a hypothetical: it is why a Helper support lane kept messaging its owner to
    report that it had nothing to report.

    So use the shared autonomous-lane matcher (also used by cron), which treats
    a marker on its own first or last line as silence while still delivering
    prose that merely mentions one mid-sentence.  Sharing the function keeps
    the two autonomous lanes from drifting apart, and keeps the interactive
    path untouched.
    """
    return is_autonomous_silence_response(content)

# Sentinel returned by _resolve_request_profile when a /p/<profile>/ prefix
# names a profile this gateway does not serve (→ 404). Distinct from None
# (no prefix / multiplexing off → handle as the default profile).
_PROFILE_REJECTED = object()

_BUILTIN_DELIVER_PLATFORMS = {
    "telegram", "discord", "slack", "signal", "sms", "whatsapp",
    "matrix", "mattermost", "homeassistant", "email", "dingtalk",
    "feishu", "wecom", "wecom_callback", "weixin", "bluebubbles",
    "qqbot", "yuanbao",
}

# Default bind host. ``None`` tells aiohttp/asyncio's ``create_server`` to bind
# BOTH address families (IPv4 + IPv6) — the portable dual-stack default.
#
# Why not "0.0.0.0" (the old default) or "::"?
#   - "0.0.0.0" binds IPv4 ONLY. On IPv6-only private networks — notably Fly.io
#     6PN, where an agent's ``<app>.internal`` name resolves to an ``fdaa:…``
#     IPv6 address — an IPv4-only listener is unreachable. That is exactly why
#     hosted-agent webhook routes were publicly unreachable: the edge router
#     reverse-proxies to ``<app>.internal:8644`` over 6PN (IPv6) but the adapter
#     was listening on 0.0.0.0 (v4 only) → connection refused.
#   - "::" is NOT a safe fix: on hosts where the kernel sets IPV6_V6ONLY=1
#     (verified on Fly machines), binding "::" yields an IPv6-ONLY socket, which
#     then breaks the IPv4 loopback health check (``curl 127.0.0.1:8644/health``)
#     and the AF_INET port-conflict probe in connect().
#   - ``None`` asks the event loop to create a listening socket per resolved
#     family, so both 127.0.0.1 (v4) and the 6PN fdaa (v6) are served regardless
#     of the bindv6only sysctl. Users can still pin a specific host via
#     ``platforms.webhook.extra.host``.
DEFAULT_HOST = None
DEFAULT_PORT = 8644
_INSECURE_NO_AUTH = "INSECURE_NO_AUTH"
_DYNAMIC_ROUTES_FILENAME = "webhook_subscriptions.json"
_RATE_WINDOW_SECONDS = 60.0
# Hostnames/IP literals that only serve connections originating on the same
# machine. Anything else is treated as a public bind for safety-rail purposes.
_LOOPBACK_HOSTS = frozenset({
    "127.0.0.1",
    "localhost",
    "::1",
    "ip6-localhost",
    "ip6-loopback",
})


def _is_loopback_host(host: Optional[str]) -> bool:
    """True when `host` binds only to the local machine.

    Covers IPv4 loopback, the standard `localhost` alias, IPv6 loopback in
    both bracketed and bare form, and the common Debian-style aliases. Any
    falsy value (empty string, None) is conservatively treated as non-loopback
    because an unset host usually means the platform-default public bind.
    """
    if not host:
        return False
    return host.strip().lower() in _LOOPBACK_HOSTS


def _hmac_str_equal(provided: str, expected: str) -> bool:
    """Timing-safe equality for two ``str`` values, tolerant of non-ASCII input.

    ``hmac.compare_digest`` raises ``TypeError`` when given a ``str`` that
    contains non-ASCII characters. The ``provided`` value here is an
    attacker-controlled signature/token header on a public, unauthenticated
    webhook endpoint, so a single non-ASCII byte would otherwise raise out of
    the request handler and return a 500 instead of rejecting the request.
    Comparing as UTF-8 bytes keeps the constant-time guarantee while making a
    hostile header fail closed with a clean rejection.
    """
    return hmac.compare_digest(provided.encode(), expected.encode())


# ---------------------------------------------------------------------------
# Route-configurable signature schemes
# ---------------------------------------------------------------------------
#
# A large family of providers authenticates webhooks in exactly the same way —
# an HMAC over a timestamp-bound message — and differs only in how the pieces
# are packaged into headers:
#
#   ElevenLabs  ElevenLabs-Signature: t=<unix>,v0=<hex>    over "<t>.<body>"
#   Stripe      Stripe-Signature:     t=<unix>,v1=<hex>    over "<t>.<body>"
#   Slack       X-Slack-Signature:    v0=<hex>             over "v0:<ts>:<body>"
#               X-Slack-Request-Timestamp: <unix>
#
# The cryptography there is byte-identical to the generic V2 scheme this
# adapter already implements; only the packaging differs.  Growing one
# hard-coded branch per vendor does not scale — each new provider needs a code
# change and a release — so a route can instead describe the packaging in
# config and reuse the same verified primitive:
#
#   routes:
#     my-provider:
#       secret: "..."
#       signature:
#         header: "ElevenLabs-Signature"
#         signature_part: "v0"      # label inside a "k=v,k=v" header
#         timestamp_part: "t"       # or timestamp_header: "X-Some-Timestamp"
#         template: "{timestamp}.{body}"
#         algorithm: "sha256"       # sha1 | sha256 | sha512
#         encoding: "hex"           # hex | base64
#         tolerance_seconds: 1800
#
# Configuring `signature` is EXCLUSIVE: the built-in GitHub/GitLab/Svix/
# Linear/generic probing is skipped entirely for that route.  A route pinned to
# one provider therefore cannot be authenticated through a different — possibly
# weaker — scheme, which also closes the legacy V1 downgrade path for it.

_SIGNATURE_ALGORITHMS = {
    # HMAC remains a sound authenticator over SHA-1, and some providers still
    # offer nothing stronger; the docs steer new integrations to sha256.
    "sha1": hashlib.sha1,
    "sha256": hashlib.sha256,
    "sha512": hashlib.sha512,
}
_SIGNATURE_ENCODINGS = ("hex", "base64")
_SIGNATURE_TEMPLATE_TOKENS = ("body", "timestamp")
_DEFAULT_SIGNATURE_TEMPLATE = "{timestamp}.{body}"
_DEFAULT_SIGNATURE_TOLERANCE_SECONDS = 300
_TEMPLATE_TOKEN_RE = re.compile(r"\{([^{}]*)\}")


def _parse_signature_spec(route_name: str, raw: Any) -> Dict[str, Any]:
    """Normalise a route's ``signature`` block, or raise ``ValueError``.

    Every rejection here is a configuration bug the operator can fix, so the
    message names the route and the offending key. ``connect()`` lets the error
    propagate (a typo should stop the gateway, not silently 401 every delivery
    until someone reads the logs); the request path catches it and fails
    closed, because dynamically-registered routes never pass through
    ``connect()``.
    """
    prefix = f"[webhook] Route '{route_name}' signature config"
    if not isinstance(raw, dict):
        raise ValueError(f"{prefix} must be a mapping, got {type(raw).__name__}")

    def _opt_str(key: str) -> str:
        value = raw.get(key, "")
        if value in (None, ""):
            return ""
        if not isinstance(value, str):
            raise ValueError(f"{prefix} key '{key}' must be a string")
        return value.strip()

    header = _opt_str("header")
    if not header:
        raise ValueError(f"{prefix} requires a non-empty 'header'")

    algorithm_name = (_opt_str("algorithm") or "sha256").lower()
    if algorithm_name not in _SIGNATURE_ALGORITHMS:
        raise ValueError(
            f"{prefix} has unknown algorithm '{algorithm_name}' "
            f"(expected one of {', '.join(sorted(_SIGNATURE_ALGORITHMS))})"
        )

    encoding = (_opt_str("encoding") or "hex").lower()
    if encoding not in _SIGNATURE_ENCODINGS:
        raise ValueError(
            f"{prefix} has unknown encoding '{encoding}' "
            f"(expected one of {', '.join(_SIGNATURE_ENCODINGS)})"
        )

    template = raw.get("template", _DEFAULT_SIGNATURE_TEMPLATE)
    if not isinstance(template, str) or not template:
        raise ValueError(f"{prefix} key 'template' must be a non-empty string")
    tokens = set(_TEMPLATE_TOKEN_RE.findall(template))
    unknown = sorted(tokens - set(_SIGNATURE_TEMPLATE_TOKENS))
    if unknown:
        raise ValueError(
            f"{prefix} template uses unknown placeholder(s) "
            f"{', '.join('{%s}' % t for t in unknown)} "
            f"(expected only {', '.join('{%s}' % t for t in _SIGNATURE_TEMPLATE_TOKENS)})"
        )
    # Exactly one, not merely at least one: the renderer splices the raw body
    # at a single point, so a repeated marker would leave a literal "{body}" in
    # the signed message. Rejecting is the fail-closed reading, and it is the
    # direction that stays compatible — a later release can accept more
    # templates without breaking anyone, whereas tightening this later would
    # break configs that had silently "worked".
    body_markers = template.count("{body}")
    if body_markers == 0:
        raise ValueError(
            f"{prefix} template must contain '{{body}}' — a signature that "
            f"does not cover the payload authenticates nothing about it"
        )
    if body_markers > 1:
        raise ValueError(
            f"{prefix} template contains '{{body}}' {body_markers} times; "
            f"exactly one is required so the signed message is unambiguous"
        )
    uses_timestamp = "timestamp" in tokens

    timestamp_part = _opt_str("timestamp_part")
    timestamp_header = _opt_str("timestamp_header")
    if timestamp_part and timestamp_header:
        raise ValueError(
            f"{prefix} sets both 'timestamp_part' and 'timestamp_header'; "
            f"the timestamp comes from exactly one place"
        )
    if uses_timestamp and not (timestamp_part or timestamp_header):
        raise ValueError(
            f"{prefix} template uses '{{timestamp}}' but neither "
            f"'timestamp_part' nor 'timestamp_header' says where to read it"
        )
    if (timestamp_part or timestamp_header) and not uses_timestamp:
        raise ValueError(
            f"{prefix} reads a timestamp but the template never uses "
            f"'{{timestamp}}', so it would not be authenticated"
        )

    tolerance = raw.get("tolerance_seconds", _DEFAULT_SIGNATURE_TOLERANCE_SECONDS)
    if isinstance(tolerance, bool) or not isinstance(tolerance, int):
        raise ValueError(f"{prefix} key 'tolerance_seconds' must be an integer")
    if tolerance <= 0:
        raise ValueError(f"{prefix} key 'tolerance_seconds' must be positive")

    return {
        "header": header,
        "signature_part": _opt_str("signature_part"),
        "signature_prefix": _opt_str("signature_prefix"),
        "timestamp_part": timestamp_part,
        "timestamp_header": timestamp_header,
        "template": template,
        "uses_timestamp": uses_timestamp,
        "algorithm": _SIGNATURE_ALGORITHMS[algorithm_name],
        "algorithm_name": algorithm_name,
        "encoding": encoding,
        "tolerance_seconds": tolerance,
    }


def _split_signature_header(value: str) -> Dict[str, List[str]]:
    """Parse a ``k=v,k=v`` signature header into label → list of values.

    Values collect into a list because providers emit a label more than once
    during secret rotation (Stripe sends several ``v1=`` entries while the old
    and new secrets are both live), and accepting any of them is what makes
    rotation non-breaking. Chunks without ``=`` are ignored rather than fatal:
    a hostile sender can pad the header with junk, and the parts that matter
    are looked up by name, never by position.
    """
    parsed: Dict[str, List[str]] = {}
    for chunk in value.split(","):
        label, sep, part = chunk.partition("=")
        if not sep:
            continue
        parsed.setdefault(label.strip(), []).append(part.strip())
    return parsed


def _render_signed_message(template: str, timestamp: str, body: bytes) -> bytes:
    """Build the exact byte string the sender signed.

    The body is spliced in as raw bytes rather than decoded to ``str``: a
    signature covers the bytes on the wire, and round-tripping a payload that
    is not valid UTF-8 (or that re-encodes differently) would turn a genuine
    delivery into a mismatch.

    ``{body}`` is located in the *template* before the timestamp is
    substituted, so an attacker-supplied timestamp cannot introduce a second
    ``{body}`` marker and move where the payload is spliced. The template is
    validated to hold exactly one ``{body}``; ``{timestamp}`` may repeat, and
    every occurrence is substituted.
    """
    head, _, tail = template.partition("{body}")
    return (
        head.replace("{timestamp}", timestamp).encode()
        + body
        + tail.replace("{timestamp}", timestamp).encode()
    )


def check_webhook_requirements() -> bool:
    """Check if webhook adapter dependencies are available."""
    return AIOHTTP_AVAILABLE


class WebhookAdapter(BasePlatformAdapter):
    """Generic webhook receiver that triggers agent runs from HTTP POSTs."""

    # No human is present to answer a "session restored — what next?" prompt:
    # webhook runs are event-triggered.  The startup auto-resume turn must
    # instruct the model to FINISH the interrupted work instead of emitting an
    # interactive acknowledgement that abandons the task (#57056).
    interactive_resume: bool = False

    def __init__(self, config: PlatformConfig):
        super().__init__(config, Platform.WEBHOOK)
        # ``host`` may be None (dual-stack default) or a user-pinned string.
        # A config value of empty string / null is normalised to None so it
        # also means "bind all families" rather than an invalid "" host.
        _cfg_host = config.extra.get("host", DEFAULT_HOST)
        self._host: Optional[str] = _cfg_host or None
        self._port: int = int(config.extra.get("port", DEFAULT_PORT))
        self._global_secret: str = config.extra.get("secret", "")
        self._static_routes: Dict[str, dict] = config.extra.get("routes", {})
        self._dynamic_routes: Dict[str, dict] = {}
        self._dynamic_routes_mtime: float = 0.0
        self._routes: Dict[str, dict] = dict(self._static_routes)
        self._runner = None
        # Routes already warned about legacy V1 body-only signatures
        # (once-per-route so a busy sender doesn't spam the log).
        self._v1_signature_warned: set[str] = set()
        # Routes whose configured signature template binds no timestamp
        self._unbound_signature_warned: set[str] = set()

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
        self._delivery_info_order: Deque[tuple[float, str]] = deque()

        # Reference to gateway runner for cross-platform delivery (set externally)
        self.gateway_runner = None

        # Idempotency: TTL cache of recently processed delivery IDs.
        # Prevents duplicate agent runs when webhook providers retry.
        self._seen_deliveries: Dict[str, float] = {}
        self._idempotency_ttl: int = 3600  # 1 hour
        self._seen_deliveries_next_prune_at: float = 0.0

        # Rate limiting: per-route timestamps in a fixed window.
        self._rate_counts: Dict[str, Deque[float]] = {}
        self._rate_limit: int = int(config.extra.get("rate_limit", 30))  # per minute

        # Body size limit (auth-before-body pattern)
        self._max_body_bytes: int = int(
            config.extra.get("max_body_bytes", 1_048_576)
        )  # 1MB
        self._script_timeout_seconds: int = int(
            config.extra.get(
                "script_timeout_seconds",
                DEFAULT_SCRIPT_TIMEOUT_SECONDS,
            )
        )
        self._route_processor = WebhookRouteProcessor(
            script_timeout_seconds=self._script_timeout_seconds
        )

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def connect(self, *, is_reconnect: bool = False) -> bool:
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

            # Surface a malformed `signature` block at startup rather than as
            # a 401 on every delivery that the operator has to guess at.
            if route.get("signature") is not None:
                _parse_signature_spec(name, route["signature"])
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

        # client_max_size makes aiohttp enforce the cap on every read path,
        # including Transfer-Encoding: chunked bodies that carry no
        # Content-Length and would otherwise bypass the header check below.
        app = web.Application(client_max_size=self._max_body_bytes)
        app.router.add_get("/health", self._handle_health)
        app.router.add_post("/webhooks/{route_name}", self._handle_webhook)
        # Multi-profile multiplexing: a /p/<profile>/webhooks/<route> prefix
        # routes the inbound event to that profile. Same handler; the profile is
        # captured from the path and stamped onto the SessionSource so the agent
        # turn resolves that profile's config/skills/credentials. Only honored
        # when gateway.multiplex_profiles is on (the handler validates).
        app.router.add_post(
            "/p/{profile}/webhooks/{route_name}", self._handle_webhook
        )

        self._runner = web.AppRunner(app)
        await self._runner.setup()
        # Do not probe only one address family before binding. With the
        # dual-stack default, an IPv6-only listener can already own this port
        # while 127.0.0.1 still looks free.
        #
        # SO_REUSEADDR is platform-dependent:
        #   - macOS (BSD semantics): two wildcard/specific sockets with
        #     SO_REUSEADDR can silently split traffic while both servers
        #     report success — so disable it there.
        #   - Linux: SO_REUSEADDR only permits rebinding past TIME_WAIT
        #     (a second live listener needs SO_REUSEPORT, which we never
        #     set). Disabling it would make a quick gateway restart fail
        #     to bind for up to ~60s — so keep the default (enabled).
        site = web.TCPSite(
            self._runner,
            self._host,
            self._port,
            reuse_address=False if sys.platform == "darwin" else None,
        )
        try:
            await site.start()
        except OSError as exc:
            await self._runner.cleanup()
            self._runner = None
            logger.error(
                "[webhook] Could not bind %s:%d: %s. "
                "Set a different host or port in config.yaml under "
                "platforms.webhook.extra.",
                self._host or "all IPv4+IPv6 interfaces",
                self._port,
                exc,
            )
            return False
        self._mark_connected()

        route_names = ", ".join(self._routes.keys()) or "(none configured)"
        logger.info(
            "[webhook] Listening on %s:%d — routes: %s",
            self._host or "* (all interfaces, IPv4+IPv6)",
            self._port,
            route_names,
        )
        # Plugin-registered native handlers (ctx.register_platform_handler).
        self._wire_plugin_handlers(None)
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
        if _is_webhook_silence_response(content):
            logger.info(
                "[webhook] Response for %s is a silence marker — not delivering", chat_id
            )
            return SendResult(success=True)

        delivery = self._delivery_info.get(chat_id, {})
        deliver_type = delivery.get("deliver", "log")

        if deliver_type == "log":
            logger.info("[webhook] Response for %s: %s", chat_id, content[:200])
            return SendResult(success=True)

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
        if len(self._delivery_info_order) < len(self._delivery_info_created):
            self._delivery_info_order = deque(
                (created_at, key)
                for key, created_at in sorted(
                    self._delivery_info_created.items(), key=lambda item: item[1]
                )
            )
        cutoff = now - self._idempotency_ttl
        while self._delivery_info_order and self._delivery_info_order[0][0] < cutoff:
            created_at, key = self._delivery_info_order.popleft()
            if self._delivery_info_created.get(key) != created_at:
                continue
            self._delivery_info.pop(key, None)
            self._delivery_info_created.pop(key, None)

    def _prune_seen_deliveries(self, now: float) -> None:
        """Occasionally prune expired delivery IDs without scanning every POST."""
        if now < self._seen_deliveries_next_prune_at:
            return
        cutoff = now - self._idempotency_ttl
        stale = [k for k, t in self._seen_deliveries.items() if t < cutoff]
        for k in stale:
            self._seen_deliveries.pop(k, None)
        self._seen_deliveries_next_prune_at = now + min(60.0, max(1.0, self._idempotency_ttl / 10))

    def _record_rate_limit_hit(self, route_name: str, now: float) -> bool:
        """Return True if route is still within limit after recording this hit."""
        window = self._rate_counts.get(route_name)
        if not isinstance(window, deque):
            new_window: Deque[float] = deque(window or ())
            self._rate_counts[route_name] = new_window
            window = new_window
        cutoff = now - _RATE_WINDOW_SECONDS
        while window and window[0] < cutoff:
            window.popleft()
        if len(window) >= self._rate_limit:
            return False
        window.append(now)
        return True

    def _record_delivery_id(self, delivery_id: str, now: float) -> bool:
        """Return True when this delivery should be processed."""
        seen_at = self._seen_deliveries.get(delivery_id)
        if seen_at is not None and now - seen_at < self._idempotency_ttl:
            return False
        if seen_at is not None:
            self._seen_deliveries.pop(delivery_id, None)
        self._seen_deliveries[delivery_id] = now
        if len(self._seen_deliveries) > max(self._rate_limit * 2, 128):
            self._prune_seen_deliveries(now)
        return True

    async def get_chat_info(self, chat_id: str) -> Dict[str, Any]:
        return {"name": chat_id, "type": "webhook"}

    def toolsets_for_source(self, source) -> Optional[List[str]]:
        """Per-route toolset override.

        Webhook session chat_ids are ``webhook:{route}:{delivery_id}``.
        When the matching route config carries a ``toolsets`` list, that list
        replaces the platform-level ``platform_toolsets.webhook`` resolution
        for this run only. Routes without the key keep the platform default
        (the intentionally constrained webhook-safe toolset), so a single
        trusted route (e.g. a localhost monitoring push) can be granted
        ``terminal`` without widening every other webhook route.

        Set via ``platforms.webhook.extra.routes.<name>.toolsets`` in
        config.yaml or a ``toolsets`` key on a subscription in
        ``webhook_subscriptions.json`` (manual edit — deliberately NOT
        exposed through `hermes webhook subscribe`, so an agent-created
        subscription cannot self-grant elevated tools).
        """
        chat_id = str(getattr(source, "chat_id", "") or "")
        parts = chat_id.split(":", 2)
        if len(parts) < 2 or parts[0] != "webhook":
            return None
        route_config = self._routes.get(parts[1])
        if not isinstance(route_config, dict):
            return None
        toolsets = route_config.get("toolsets")
        if not isinstance(toolsets, list) or not toolsets:
            return None
        cleaned = [str(t).strip() for t in toolsets if str(t).strip()]
        return cleaned or None

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

    def _resolve_request_profile(self, request: "web.Request"):
        """Resolve + validate the /p/<profile>/ URL prefix on a webhook request.

        Returns:
          - ``None`` when no profile prefix is present, or when multiplexing
            is off and the prefix names this gateway's own profile (the
            request is handled as the serving profile).
          - the profile name (str) when present, multiplexing is on, and the
            profile is one this gateway serves.
          - ``_PROFILE_REJECTED`` when a prefix is present but the profile is
            unknown/unconfigured, or names a profile this single-profile
            gateway does not serve (handler returns 404).
        """
        profile = (request.match_info.get("profile") or "").strip()
        if not profile:
            return None
        runner = self.gateway_runner
        cfg = getattr(runner, "config", None)
        if not getattr(cfg, "multiplex_profiles", False):
            # Prefix supplied but multiplexing is off. Only a self-referential
            # prefix (naming this gateway's own profile) may fall through to
            # the bare route; anything else fails closed — silently ignoring
            # the prefix served the gateway owner's routes/config under
            # another profile's URL (#91583 defect 2).
            try:
                from hermes_cli.profiles import profile_matches_home

                if profile_matches_home(profile):
                    return None
            except Exception:
                pass
            return _PROFILE_REJECTED
        try:
            from hermes_cli.profiles import profiles_to_serve
            served = {
                name
                for name, _ in profiles_to_serve(
                    multiplex=True,
                    profile_allowlist=getattr(
                        cfg, "multiplex_profile_allowlist", None
                    ),
                )
            }
        except Exception:
            return _PROFILE_REJECTED
        if profile not in served:
            return _PROFILE_REJECTED
        return profile

    @staticmethod
    def _route_allows_profile(
        route_config: dict,
        request_profile: Optional[str],
    ) -> bool:
        """Return whether a route is bound to the URL-selected profile.

        Omitting ``profile`` keeps a route on the default profile. An explicit
        null, blank, or non-string value is malformed and fails closed.
        """
        if "profile" not in route_config:
            configured_profile = "default"
        else:
            configured_profile = route_config.get("profile")
        if not isinstance(configured_profile, str):
            return False
        configured_profile = configured_profile.strip()
        if not configured_profile:
            return False
        effective_profile = request_profile or "default"
        return configured_profile == effective_profile

    @staticmethod
    def _profile_scope(profile: Optional[str]):
        """Enter the URL-resolved profile's runtime scope, or a no-op.

        Only a resolved ``/p/<profile>/`` prefix enters a scope (same helper
        the runner wraps ``handle_message`` in); bare routes keep serving the
        launch profile exactly as before.
        """
        if not profile or not isinstance(profile, str):
            return nullcontext()
        from gateway.run import _profile_runtime_scope
        from hermes_cli.profiles import get_profile_dir

        return _profile_runtime_scope(get_profile_dir(profile))

    async def _handle_webhook(self, request: "web.Request") -> "web.Response":
        """POST /webhooks/{route_name} — receive and process a webhook event."""
        # Hot-reload dynamic subscriptions on each request (mtime-gated, cheap)
        self._reload_dynamic_routes()

        route_name = request.match_info.get("route_name", "")
        route_config = self._routes.get(route_name)

        # Multi-profile: resolve + validate the /p/<profile>/ prefix if present.
        profile = self._resolve_request_profile(request)
        if profile is _PROFILE_REJECTED:
            return web.json_response(
                {"error": "Unknown or unconfigured profile"}, status=404
            )

        if not route_config:
            return web.json_response(
                {"error": f"Unknown route: {route_name}"}, status=404
            )

        if not self._route_allows_profile(route_config, profile):
            effective_profile = profile or "default"
            logger.warning(
                "[webhook] Route %s is not authorized for profile %r",
                route_name,
                effective_profile,
            )
            # Match the unknown-route response so callers cannot use profile
            # mismatches to enumerate route bindings.
            return web.json_response(
                {"error": f"Unknown route: {route_name}"}, status=404
            )

        # Disabled routes are kept in the subscriptions file (so the dashboard
        # can re-enable them) but reject incoming events.  Default-enabled:
        # only an explicit ``enabled: false`` turns a route off, matching the
        # mcp_servers ``enabled`` semantics.
        if route_config.get("enabled", True) is False:
            return web.json_response(
                {"error": f"Route disabled: {route_name}"}, status=403
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
        except web.HTTPRequestEntityTooLarge:
            # aiohttp's client_max_size tripped — chunked or lying
            # Content-Length. Same 413 as the header check above.
            return web.json_response(
                {"error": "Payload too large"}, status=413
            )
        except Exception as e:
            logger.error("[webhook] Failed to read body: %s", e)
            return web.json_response({"error": "Bad request"}, status=400)
        if len(raw_body) > self._max_body_bytes:
            # Defense in depth: enforce the cap on the actual bytes read even
            # if the server-level limit was bypassed or misconfigured.
            return web.json_response(
                {"error": "Payload too large"}, status=413
            )

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
            if not self._validate_signature(
                request, raw_body, secret, route_config=route_config
            ):
                logger.warning(
                    "[webhook] Invalid signature for route %s", route_name
                )
                return web.json_response(
                    {"error": "Invalid signature"}, status=401
                )

        # ── Rate limiting (after auth) ───────────────────────────
        now = time.time()
        if not self._record_rate_limit_hit(route_name, now):
            return web.json_response(
                {"error": "Rate limit exceeded"}, status=429
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

        if not self._route_processor.route_filters_match(
            route_config, payload, event_type, request.headers
        ):
            logger.info(
                "[webhook] filtered event=%s route=%s",
                event_type,
                route_name,
            )
            return web.json_response(
                {
                    "status": "ignored",
                    "reason": "filter",
                    "route": route_name,
                }
            )

        # The route script, prompt render and skill lookup below read the
        # profile's home (skills/, config). The runner only enters the routed
        # profile's scope later, around handle_message, so without this they
        # ran against the launch (default) profile (#67277). Only a resolved
        # /p/<profile>/ enters a scope; bare routes are unchanged.
        with self._profile_scope(profile):
            if route_config.get("script"):
                # run_route_script shells out (subprocess.run, up to its
                # timeout); run it in a worker thread so it can't block the
                # gateway event loop. to_thread copies the contextvars, so
                # the profile scope follows it.
                keep, transformed_payload = await asyncio.to_thread(
                    self._route_processor.run_route_script,
                    route_config.get("script"),
                    payload,
                )
                if not keep:
                    logger.info(
                        "[webhook] script ignored event=%s route=%s",
                        event_type,
                        route_name,
                    )
                    return web.json_response(
                        {
                            "status": "ignored",
                            "reason": "script",
                            "route": route_name,
                        }
                    )
                payload = transformed_payload or payload

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
        if not self._record_delivery_id(delivery_id, now):
            logger.info(
                "[webhook] Skipping duplicate delivery %s", delivery_id
            )
            return web.json_response(
                {"status": "duplicate", "delivery_id": delivery_id},
                status=200,
            )

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
        }
        self._delivery_info[session_chat_id] = deliver_config
        self._delivery_info_created[session_chat_id] = now
        self._delivery_info_order.append((now, session_chat_id))
        self._prune_delivery_info(now)

        # Build source and event
        source = self.build_source(
            chat_id=session_chat_id,
            chat_name=f"webhook/{route_name}",
            chat_type="webhook",
            user_id=f"webhook:{route_name}",
            user_name=route_name,
        )
        if profile and isinstance(profile, str):
            source.profile = profile
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

        # Non-blocking — return 202 Accepted immediately.  The per-delivery
        # session is closed by the ``on_processing_complete`` override below
        # once the agent run actually finishes (``handle_message`` itself is
        # fire-and-forget: it spawns ``_process_message_background`` and
        # returns before the run starts, so nothing can be closed here).
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

    async def on_processing_complete(
        self, event: "MessageEvent", outcome: Any
    ) -> None:
        """Close the per-delivery webhook session once its run finishes.

        A webhook delivery is one-shot: the ``delivery_id`` is baked into the
        session key, so the session will never receive a second turn.  Mirror
        the cron completion path (``cron/scheduler.py`` →
        ``end_session(..., "cron_complete")``) by marking the session ended
        when the run completes.  Without this, webhook sessions keep
        ``ended_at`` NULL forever; ``SessionDB.prune_sessions`` only reaps
        rows with ``ended_at`` set, so unclosed webhook sessions accumulate
        unbounded and drive state.db bloat (the ghost-session leak).

        This hook is the one seam that runs at the TRUE end of the run:
        ``BasePlatformAdapter._process_message_background`` fires it after the
        message handler returns, on the success, failure, and cancellation
        paths alike — so error runs are reaped too.  (``handle_message`` is
        fire-and-forget; wrapping IT closes before the run even starts.)
        ``end_session()`` is first-reason-wins and no-ops on an already-ended
        row, so this never clobbers a ``compression``/``agent_close`` reason.
        """
        await self._end_webhook_session(event, event.source.chat_id)

    async def _end_webhook_session(
        self, event: "MessageEvent", session_chat_id: str
    ) -> None:
        """Mark the per-delivery webhook session ended in state.db.

        Resolves the persisted ``session_id`` from the gateway session store
        using the SAME source the run was keyed on (so profile multiplexing
        and key construction match exactly), then closes it via the existing
        ``SessionDB.end_session`` API — never a hand-written UPDATE.
        """
        runner = self.gateway_runner
        if runner is None:
            return
        session_db = getattr(runner, "_session_db", None)
        store = getattr(runner, "session_store", None)
        if session_db is None or store is None:
            return
        try:
            key_fn = getattr(runner, "_session_key_for_source", None)
            if key_fn is None:
                return
            session_key = key_fn(event.source)
            # Resolve the persisted session_id via the store's public,
            # lock-held accessor (peek_session_id) rather than reaching into
            # the private _entries dict without the store lock. Fall back to
            # the private path only for older stores / test doubles that
            # predate the accessor.
            peek = getattr(store, "peek_session_id", None)
            if callable(peek):
                session_id = peek(session_key)
            else:
                if hasattr(store, "_ensure_loaded"):
                    try:
                        store._ensure_loaded()
                    except Exception:
                        pass
                entries = getattr(store, "_entries", {}) or {}
                entry = entries.get(session_key)
                session_id = getattr(entry, "session_id", None) if entry else None
            if not session_id:
                logger.debug(
                    "[webhook] No session_id to close for %s (key=%s)",
                    session_chat_id,
                    session_key,
                )
                return
            # AsyncSessionDB forwards end_session via asyncio.to_thread; a
            # plain SessionDB exposes it synchronously.  Handle both.
            _end = session_db.end_session
            result = _end(session_id, "webhook_complete")
            if asyncio.iscoroutine(result):
                await result
            logger.debug(
                "[webhook] Closed session %s for delivery %s",
                session_id,
                session_chat_id,
            )
        except Exception as e:
            logger.debug(
                "[webhook] Failed to close session for %s: %s",
                session_chat_id,
                e,
            )

    # ------------------------------------------------------------------
    # Signature validation
    # ------------------------------------------------------------------

    def _validate_signature(
        self,
        request: "web.Request",
        body: bytes,
        secret: str,
        route_config: Optional[dict] = None,
    ) -> bool:
        """Validate a webhook signature.

        A route carrying a ``signature`` block is validated against exactly
        that scheme; otherwise the built-in providers are probed in turn
        (GitHub, GitLab, Svix, Linear, generic HMAC-SHA256).
        """
        def _header(name: str) -> str:
            return (
                request.headers.get(name, "")
                or request.headers.get(name.lower(), "")
                or request.headers.get(name.upper(), "")
            )

        route_name = request.match_info.get("route_name", "")
        if route_config is None:
            # Direct callers (and the test suite) may not pass the route in.
            # Resolving it here keeps a configured scheme authoritative on
            # every path instead of silently reverting to header probing.
            route_config = self._routes.get(route_name) or {}

        # Route-configured scheme: exclusive and fail-closed. Built-in probing
        # below is unreachable for such a route, so a sender cannot downgrade
        # it to a weaker scheme it happens to also have headers for.
        configured = route_config.get("signature")
        if configured is not None:
            try:
                spec = _parse_signature_spec(route_name, configured)
            except ValueError as exc:
                # Reachable for dynamically-registered routes, which never go
                # through connect()'s startup validation.
                logger.error("%s", exc)
                return False
            return self._validate_configured_signature(
                route_name=route_name,
                spec=spec,
                header_value=_header(spec["header"]),
                timestamp_value=(
                    _header(spec["timestamp_header"])
                    if spec["timestamp_header"]
                    else ""
                ),
                body=body,
                secret=secret,
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

        # Linear: linear-signature = <hex HMAC-SHA256 of the raw body, keyed
        # by the webhook signing key>. Linear's documented scheme signs the
        # body only (no timestamp binding), so this mirrors it exactly;
        # without this branch every Linear delivery to a secret-configured
        # route was rejected as unrecognized (#87348).
        linear_sig = _header("linear-signature")
        if linear_sig:
            expected_linear = hmac.new(
                secret.encode(), body, hashlib.sha256
            ).hexdigest()
            return _hmac_str_equal(linear_sig, expected_linear)

        # GitHub: X-Hub-Signature-256 = sha256=<hex>
        gh_sig = request.headers.get("X-Hub-Signature-256", "")
        if gh_sig:
            expected = "sha256=" + hmac.new(
                secret.encode(), body, hashlib.sha256
            ).hexdigest()
            return _hmac_str_equal(gh_sig, expected)

        # GitLab: X-Gitlab-Token = <plain secret>
        gl_token = request.headers.get("X-Gitlab-Token", "")
        if gl_token:
            return _hmac_str_equal(gl_token, secret)

        # Generic V2: X-Webhook-Signature-V2 = <hex HMAC-SHA256 of "<timestamp>.<body>">
        #             X-Webhook-Timestamp = <unix seconds> (required for V2)
        # Checked independently of (and before) legacy V1 below — a sender
        # that only ever sends V2 headers must still validate here; nesting
        # this inside `if generic_sig:` would silently skip V2-only senders.
        #
        # The presence of X-Webhook-Signature-V2 alone selects V2 mode and
        # commits to it — it must NOT fall through to the V1 branch just
        # because the timestamp is missing/malformed/expired. A sender
        # migrating to V2 typically sends both V1 and V2 headers together
        # for compatibility; if incomplete V2 fell through to V1, an
        # attacker who captured one such mixed request could strip the
        # X-Webhook-Timestamp header from a replay and have it validate
        # against the still-present, still-unprotected V1 signature instead
        # — silently downgrading a V2-protected request back to the replay
        # hole V2 exists to close.
        v2_sig = request.headers.get("X-Webhook-Signature-V2", "")
        if v2_sig:
            v2_timestamp = request.headers.get("X-Webhook-Timestamp", "")
            if not v2_timestamp:
                logger.warning(
                    "[webhook] Route '%s' sent X-Webhook-Signature-V2 with "
                    "no X-Webhook-Timestamp — rejecting rather than "
                    "falling back to legacy V1",
                    request.match_info.get("route_name", ""),
                )
                return False
            try:
                ts = int(v2_timestamp)
            except (TypeError, ValueError):
                return False
            if abs(int(time.time()) - ts) > 300:
                logger.warning(
                    "[webhook] Route '%s' generic HMAC V2 timestamp outside replay window",
                    request.match_info.get("route_name", ""),
                )
                return False
            signed_content = v2_timestamp.encode() + b"." + body
            expected_v2 = hmac.new(
                secret.encode(), signed_content, hashlib.sha256
            ).hexdigest()
            return _hmac_str_equal(v2_sig, expected_v2)

        # Generic V1 (legacy): X-Webhook-Signature = <hex HMAC-SHA256 of body>
        # (deprecated — no replay protection, since the signature only
        # covers the body: a captured (body, signature) pair replays
        # indefinitely with no timestamp binding it to a specific delivery.)
        # Only reachable when X-Webhook-Signature-V2 was not sent at all —
        # see the guard above.
        generic_sig = request.headers.get("X-Webhook-Signature", "")
        if generic_sig:
            expected = hmac.new(
                secret.encode(), body, hashlib.sha256
            ).hexdigest()
            if route_name not in self._v1_signature_warned:
                self._v1_signature_warned.add(route_name)
                logger.warning(
                    "[webhook] Route '%s' uses legacy body-only HMAC (no "
                    "timestamp), which is vulnerable to replay attacks. Add "
                    "an 'X-Webhook-Timestamp' header and switch to "
                    "'X-Webhook-Signature-V2' (HMAC-SHA256 of "
                    "'<timestamp>.<body>').",
                    route_name,
                )
            return _hmac_str_equal(generic_sig, expected)

        # No recognised signature header but secret is configured → reject
        logger.debug(
            "[webhook] Secret configured but no signature header found"
        )
        return False

    def _validate_configured_signature(
        self,
        *,
        route_name: str,
        spec: Dict[str, Any],
        header_value: str,
        timestamp_value: str,
        body: bytes,
        secret: str,
    ) -> bool:
        """Validate a signature described by a route's ``signature`` block.

        Every path out of here is either an explicit ``True`` on a verified
        HMAC or a logged ``False`` — there is no fall-through to another
        scheme, because the block exists precisely to pin a route to one.
        """
        if not header_value:
            logger.warning(
                "[webhook] Route '%s' expects signature header '%s' but the "
                "request did not send it",
                route_name,
                spec["header"],
            )
            return False

        needs_parts = bool(spec["signature_part"] or spec["timestamp_part"])
        parts = _split_signature_header(header_value) if needs_parts else {}

        if spec["signature_part"]:
            candidates = parts.get(spec["signature_part"], [])
            if not candidates:
                logger.warning(
                    "[webhook] Route '%s' header '%s' has no '%s=' part",
                    route_name,
                    spec["header"],
                    spec["signature_part"],
                )
                return False
        else:
            candidates = [header_value.strip()]

        signature_prefix = spec["signature_prefix"]
        if signature_prefix:
            candidates = [
                candidate[len(signature_prefix):]
                for candidate in candidates
                if candidate.startswith(signature_prefix)
            ]
            if not candidates:
                logger.warning(
                    "[webhook] Route '%s' header '%s' is missing the required "
                    "'%s' prefix",
                    route_name,
                    spec["header"],
                    signature_prefix,
                )
                return False

        candidates = [candidate for candidate in candidates if candidate]
        if not candidates:
            logger.warning(
                "[webhook] Route '%s' header '%s' carried an empty signature",
                route_name,
                spec["header"],
            )
            return False

        timestamp = ""
        if spec["uses_timestamp"]:
            if spec["timestamp_part"]:
                # A hostile sender can repeat "t=" to pair one timestamp with
                # another's signature. Only the first occurrence is honoured,
                # and a conflicting second one is refused outright rather than
                # letting the caller pick whichever value validates.
                found = parts.get(spec["timestamp_part"], [])
                if len(set(found)) > 1:
                    logger.warning(
                        "[webhook] Route '%s' header '%s' carried conflicting "
                        "'%s=' values",
                        route_name,
                        spec["header"],
                        spec["timestamp_part"],
                    )
                    return False
                timestamp = found[0] if found else ""
            else:
                timestamp = timestamp_value.strip()

            if not timestamp:
                logger.warning(
                    "[webhook] Route '%s' signature requires a timestamp but "
                    "none was sent",
                    route_name,
                )
                return False
            try:
                ts = int(timestamp)
            except (TypeError, ValueError):
                logger.warning(
                    "[webhook] Route '%s' sent a non-integer signature "
                    "timestamp",
                    route_name,
                )
                return False
            # Symmetric window: a timestamp far in the future is as much a
            # sign of a forged or misconfigured sender as a stale one, and
            # accepting it would hand an attacker an unbounded replay ticket.
            if abs(int(time.time()) - ts) > spec["tolerance_seconds"]:
                logger.warning(
                    "[webhook] Route '%s' signature timestamp outside the "
                    "%ds replay window",
                    route_name,
                    spec["tolerance_seconds"],
                )
                return False
        elif route_name not in self._unbound_signature_warned:
            self._unbound_signature_warned.add(route_name)
            logger.warning(
                "[webhook] Route '%s' signs the body only (no '{timestamp}' "
                "in the template), so a captured request replays "
                "indefinitely. Bind a timestamp if the provider offers one.",
                route_name,
            )

        message = _render_signed_message(spec["template"], timestamp, body)
        digest = hmac.new(secret.encode(), message, spec["algorithm"]).digest()
        if spec["encoding"] == "hex":
            expected = digest.hex()
            # Hex case carries no information, so normalising it before the
            # compare keeps the timing guarantee while accepting providers
            # that emit uppercase.
            candidates = [candidate.lower() for candidate in candidates]
        else:
            expected = base64.b64encode(digest).decode()

        for candidate in candidates:
            if _hmac_str_equal(candidate, expected):
                return True

        logger.warning(
            "[webhook] Route '%s' signature mismatch on header '%s'",
            route_name,
            spec["header"],
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
            if version == "v1" and _hmac_str_equal(signature, expected):
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
            if key == "event_type":
                return event_type
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

        # --- Input validation (prevent CLI argument injection) ---
        # pr_number must be a positive integer.
        try:
            pr_int = int(pr_number)
            if pr_int <= 0:
                raise ValueError("non-positive")
        except (ValueError, TypeError):
            logger.error(
                "[webhook] invalid pr_number: %r", pr_number
            )
            return SendResult(
                success=False, error="Invalid pr_number"
            )

        # repo must match owner/name (alphanumeric, hyphens, underscores, dots).
        if not re.fullmatch(r"[A-Za-z0-9._-]+/[A-Za-z0-9._-]+", repo):
            logger.error("[webhook] invalid repo format: %r", repo)
            return SendResult(
                success=False, error="Invalid repo format"
            )

        try:
            # Off-loop: `gh` does network I/O and can take its full 30s
            # timeout. Running it inline froze every adapter and timer on
            # the gateway event loop for the duration (Pattern A, #91912
            # class). asyncio.to_thread keeps the loop serving while the
            # subprocess runs; the worker thread is bounded by the
            # subprocess timeout below.
            result = await asyncio.to_thread(
                subprocess.run,
                [
                    "gh",
                    "pr",
                    "comment",
                    str(pr_int),
                    "--repo",
                    repo,
                    "--body",
                    content,
                ],
                capture_output=True,
                text=True, encoding='utf-8', errors='replace',
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

        # Default adapters first; multiplex may park Slack/etc. only on a
        # secondary profile (self._profile_adapters). Fall back so webhook
        # deliver:slack still works when default has slack disabled.
        adapter = self.gateway_runner.adapters.get(target_platform)
        if not adapter:
            for _prof, amap in (getattr(self.gateway_runner, "_profile_adapters", None) or {}).items():
                if not isinstance(amap, dict):
                    continue
                cand = amap.get(target_platform)
                if cand is not None:
                    adapter = cand
                    break
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
