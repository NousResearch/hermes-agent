"""RelayAdapter — one generic gateway adapter fronted by the connector. EXPERIMENTAL.

A single ``BasePlatformAdapter`` subclass that, at handshake, receives a
``CapabilityDescriptor`` from the connector telling it which platform it is
fronting and which capabilities to advertise to the ``GatewayStreamConsumer``.
It implements the four abstract methods (``connect`` / ``disconnect`` / ``send``
/ ``get_chat_info``) plus the capability surface (``MAX_MESSAGE_LENGTH``,
``message_len_fn``, ``supports_draft_streaming``) by delegating wire I/O to an
injected transport and reading capabilities off the descriptor.

There is NO per-platform gateway code: the connector is the only side that knows
"this chat_id maps to a Discord channel, send it via the Discord websocket."
The gateway sees an ordinary ``MessageEvent`` in and calls ``adapter.send`` out.

EXPERIMENTAL: the transport protocol and descriptor schema may change without a
deprecation cycle until >=2 Class-1 platforms validate them.
"""

from __future__ import annotations

import asyncio
import logging
import secrets
import time
from collections import OrderedDict
from typing import Any, Callable, Dict, Optional, Tuple, cast

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import BasePlatformAdapter, MessageEvent, SendResult
from gateway.relay.descriptor import CapabilityDescriptor
from gateway.relay.media import RelayMediaClient
from gateway.relay.transport import RelayTransport
from gateway.session import SessionSource

logger = logging.getLogger(__name__)

# Keep the drain-path going-idle ACK budget strictly under the runner's default
# adapter disconnect timeout (5s). If go_idle consumes the whole outer budget,
# cancellation can fire before transport.disconnect() and leave the websocket
# open. Paired with transport teardown budgets of 1s each for supervisor,
# reader, and ws.close (~3s), the full drain path stays inside 5s.
_RELAY_GO_IDLE_ON_DISCONNECT_TIMEOUT_S = 2.0
_RELAY_REVOCATION_MONITOR_TEARDOWN_TIMEOUT_S = 1.0

# How many already-answered prompt ids to remember, so a duplicate answer for
# one of them (a double tap, or a connector redelivery of the same forward) is
# recognized as a repeat rather than mistaken for a stale prompt. One short
# string per entry; sized well above the number of prompts a single gateway can
# have in flight, and far below anything that matters for memory.
_RESOLVED_PROMPT_MEMORY = 256


def _utf16_len(text: str) -> int:
    """Count UTF-16 code units (Telegram's length unit)."""
    return len(text.encode("utf-16-le")) // 2


# Table-driven length-unit selection from the descriptor's ``len_unit``.
_LEN_FNS: Dict[str, Callable[[str], int]] = {
    "chars": len,
    "utf16": _utf16_len,
}


class RelayAdapter(BasePlatformAdapter):
    """Generic relay adapter advertising a connector-negotiated capability profile."""

    def __init__(
        self,
        config: PlatformConfig,
        descriptor: CapabilityDescriptor,
        transport: Optional[RelayTransport] = None,
    ) -> None:
        # The relay adapter fronts many platforms but presents as a single
        # logical platform to the runner; Platform.RELAY identifies it.
        super().__init__(config, Platform.RELAY)
        self.descriptor = descriptor
        self._transport = transport
        # Capability surface read by stream_consumer (getattr(..., 4096)).
        self.MAX_MESSAGE_LENGTH = descriptor.max_message_length
        # chat_id -> scope_id (server/workspace scope), learned from inbound
        # events. The connector's egress guard resolves the owning tenant from
        # the OUTBOUND action's metadata.scope_id; the gateway's generic delivery
        # path (run.py _thread_metadata_for_source) only carries thread_id, so we
        # re-attach the scope here from what we saw inbound. Keyed by chat_id
        # (channel) since that's what send() receives. See routedEgressGuard.ts.
        self._scope_by_chat: Dict[str, str] = {}
        # chat_id -> author user_id for DM chats (no scope). A DM reply has
        # no scope discriminator, so the connector resolves its tenant from the
        # recipient's author binding; we re-attach this user_id as
        # metadata.user_id on the outbound action so it can. See _capture_scope.
        self._dm_user_by_chat: Dict[str, str] = {}
        # chat_id -> (thread_id, initial_name) of the auto-thread the CONNECTOR
        # created for our most recent send into that chat (auto-thread routing
        # feedback off SendResult — see send()). Consumed by the gateway's
        # semantic thread-rename lane; bounded like the sibling caches.
        self._auto_thread_by_chat: Dict[str, Tuple[str, str]] = {}
        # Bounded FIFO seen-set for inbound replay dedupe (finding #3);
        # dict preserves insertion order, giving cheap oldest-first eviction.
        self._seen_inbound: Dict[str, None] = {}
        # chat_id -> event fired when the entry above lands, so a consumer that
        # arrives before the send can wait for it instead of polling. See
        # wait_for_auto_thread_info.
        self._auto_thread_waiters: Dict[str, asyncio.Event] = {}
        # chat_id -> chat_type (e.g. "dm", "channel", "group") learned from the
        # inbound event. Used to reproduce native Slack's synthetic-DM-thread
        # suppression on the relay lane: a DM streaming reply carries
        # reply_to=<triggering message ts> as its edit anchor, but the connector
        # maps a raw reply_to to a Slack thread_ts — so a plain DM reply would be
        # threaded UNDER the user's message (and lose progressive edit streaming)
        # instead of posting flat at the DM root. Native SlackAdapter drops that
        # synthetic reply_to in _resolve_thread_ts; the relay lane needs the same
        # disambiguation, and it needs the chat_type to know a chat is a DM.
        self._chat_type_by_chat: Dict[str, str] = {}
        # chat_id -> last triggering message ts (Slack). The typing/status
        # lane's synthetic thread anchor in thread-per-message mode;
        # see _capture_scope and send_typing.
        self._last_inbound_ts_by_chat: Dict[str, str] = {}
        # chat_id -> the UNDERLYING platform (e.g. "discord", "telegram") this
        # chat belongs to (Phase 1.5 multi-platform-per-agent). One relay adapter
        # fronts N platforms on one WS; an outbound reply must egress through the
        # platform the inbound came from. We remember it per chat_id from the
        # inbound event's source.platform and stamp it on the OutboundFrame so the
        # connector dispatches to the right sender. Empty for a single-platform
        # gateway (the connector falls back to its session default). See
        # _capture_scope / send.
        self._platform_by_chat: Dict[str, str] = {}
        self.supports_code_blocks = descriptor.markdown_dialect not in ("", "plain")
        # Phase 7 Unit 7d-B: watches the transport for a terminal auth revocation
        # (a 4401 close after a successful handshake = the operator opted this
        # instance out of the relay). On revocation we surface a clean,
        # non-retryable "relay disabled" fatal so the dashboard stops showing a
        # red "retrying" spin against a dead credential.
        self._revocation_monitor: Optional[asyncio.Task[None]] = None
        # Phase 2 media: the authenticated client for the connector's
        # /relay/media routes (upload for send_media source_url; download for
        # inbound re-hosted attachments → local paths the vision/file tools
        # consume). Built lazily from the relay dial URL + per-gateway creds;
        # None when either is absent (media lanes then degrade to the
        # pre-media text fallbacks).
        self._media_client: Optional["RelayMediaClient"] = None
        # Phase 3 interactive: prompt_id -> pending-prompt state. A `prompt`
        # op renders native options; the user's pick comes back inbound as a
        # prompt_response naming this id. The registry maps it back to the
        # waiting primitive (approval / slash-confirm / clarify) so the click
        # resolves EXACTLY like the native adapters' button callbacks.
        # Entries expire lazily (see _pop_prompt) so an unanswered prompt
        # never leaks. Keyed by our own minted ids (see _prompt_owner_nonce).
        self._pending_prompts: Dict[str, Dict[str, Any]] = {}
        # Per-process marker prefixed onto every prompt id this adapter mints,
        # so an answer can be recognized as OURS before it is acted on.
        #
        # WHY: a button press arrives on the passthrough plane, which the
        # connector fans out to EVERY live gateway session of the tenant
        # (relayServer.routeBusMessage delivers `passthrough` via
        # sessionsByTenant, unlike `message`, which narrows to the admitted
        # instance set). The prompt itself went out from exactly one instance,
        # and _pending_prompts is process-local — so every OTHER instance sees
        # an answer for a prompt it never minted. Without this marker those
        # instances cannot tell "a sibling owns this" from "my own prompt
        # expired", and the id-shaped text ("/c1") falls through to chat
        # dispatch, where run.py answers "Unknown command `/c1`" — one copy per
        # sibling gateway. Siblings are the common case in a DM: the connector
        # resolves the invoker's bindings to DISTINCT TENANTS, so several
        # instances of one tenant all receive the forward.
        self._prompt_owner_nonce: str = secrets.token_hex(3)
        # Prompt ids this process has already resolved, newest last. A repeat
        # answer for the same id (double tap, or a connector redelivery) is
        # then consumed silently instead of being treated as a stale prompt.
        self._resolved_prompts: "OrderedDict[str, float]" = OrderedDict()

    # ── capability surface (from descriptor) ─────────────────────────────
    @property
    def authorization_is_upstream(self) -> bool:
        """Relay authorization is enforced by the connector, not locally.

        The connector authenticates this gateway's WS (per-instance secret) and
        performs owner-only author-binding resolution before delivering, so any
        inbound relay event was already authorized as THIS instance's bound user
        (``user_instance_binding``, keyed on the connector-observed author id).
        The instance therefore must not default-deny relay users for lack of a
        local ``RELAY_ALLOWED_USERS`` env allowlist. See
        ``BasePlatformAdapter.authorization_is_upstream``.
        """
        return True

    @property
    def message_len_fn(self) -> Callable[[str], int]:
        return _LEN_FNS.get(self.descriptor.len_unit, len)

    @property
    def supports_status_text(self) -> bool:  # type: ignore[override]
        """Whether the fronted platform renders a TEXT status line.

        Native parity (rich status text): Slack's typing surface is the
        assistant status line ("Finding answers…" next to the bot name), a
        text-rendering indicator. When the relay fronts Slack, advertise it so
        run.py's live-status lane feeds per-tool phrases via
        ``set_status_text()`` — exactly the wiring the native SlackAdapter
        gets (``supports_status_text = True``). Other fronted platforms keep
        textless typing bubbles and must NOT receive phrase traffic.

        Property (not class attr) because ONE RelayAdapter class fronts many
        platforms; the answer depends on the handshaked descriptor. On a
        multi-platform relay this scalar reflects the PRIMARY identity's
        platform (same convention as the scalar ``descriptor``); per-chat
        egress paths that need chat-accurate capabilities use
        ``_descriptor_for_chat`` below.
        """
        return self.descriptor.platform == Platform.SLACK.value

    # ── per-chat capability resolution (Phase 1.5 multi-platform) ─────────
    def _descriptor_for_chat(self, chat_id: str) -> CapabilityDescriptor:
        """The capability descriptor governing a specific chat.

        A multi-platform gateway fronts N platforms on ONE adapter, but the
        scalar `descriptor`/`MAX_MESSAGE_LENGTH` surface can only carry one
        platform's profile (the primary identity's). Platform caps genuinely
        differ — Discord 2000 / Telegram 4096 / Slack 39000 — so applying the
        primary's cap to every chat either fragments needlessly (small primary)
        or over-sends into a platform 400 (large primary; the live bug: 2,543
        and 2,641-char sends rejected by Discord). Resolve the chat's platform
        from what we saw inbound (`_platform_by_chat`, the same map per-frame
        egress uses) and look up that platform's negotiated descriptor on the
        transport. Falls back to the scalar descriptor when the chat's platform
        is unknown (never saw inbound) or the transport predates the map.
        """
        platform = self._platform_by_chat.get(str(chat_id))
        if platform and self._transport is not None:
            resolve = getattr(self._transport, "descriptor_for_platform", None)
            if callable(resolve):
                try:
                    per_platform = cast(
                        Optional[CapabilityDescriptor], resolve(platform)
                    )
                except Exception:  # noqa: BLE001 - capability lookup must never break a send
                    per_platform = None
                if per_platform is not None:
                    return per_platform
        return self.descriptor

    def max_message_length_for_chat(self, chat_id: str) -> int:
        return self._descriptor_for_chat(chat_id).max_message_length

    def message_len_fn_for_chat(self, chat_id: str) -> Callable[[str], int]:
        return _LEN_FNS.get(self._descriptor_for_chat(chat_id).len_unit, len)

    def supports_draft_streaming(
        self,
        chat_type: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        return self.descriptor.supports_draft_streaming

    # ── abstract methods (delegated to the transport) ────────────────────
    async def connect(self, *, is_reconnect: bool = False) -> bool:
        # ``is_reconnect`` is part of the BasePlatformAdapter.connect contract:
        # the gateway's reconnect watcher (gateway/run.py) re-establishes a
        # platform after a fatal adapter error by building a fresh adapter and
        # calling ``connect(is_reconnect=True)``. Relay MUST accept the kwarg or
        # that recovery path raises TypeError and the relay platform can never
        # come back through the watcher.
        #
        # Relay deliberately IGNORES the flag. The flag exists so adapters with a
        # server-side update queue (e.g. Telegram's Bot API) preserve that queue
        # across an outage instead of dropping it (#46621). Relay has no such
        # gateway-side queue: messages buffered during a gap live in the
        # CONNECTOR's durable buffer and are replayed when the transport
        # re-handshakes. Routine WS drops are handled entirely by the transport's
        # own reconnect supervisor (WebSocketRelayTransport, reconnect=True);
        # a watcher-driven reconnect builds a fresh transport from scratch (the
        # fatal-error handler disconnect()s the old adapter first, cancelling its
        # supervisor), so there is nothing at the adapter layer to preserve.
        if self._transport is None:
            raise RuntimeError("RelayAdapter has no transport configured")
        self._transport.set_inbound_handler(self._on_inbound)
        # Inbound interrupts (connector -> owning gateway) arrive as
        # interrupt_inbound frames over the SAME outbound WS; bridge them to the
        # adapter's interrupt path. WS-only: there is no inbound HTTP receiver.
        set_interrupt = getattr(self._transport, "set_interrupt_inbound_handler", None)
        if callable(set_interrupt):
            set_interrupt(self.on_interrupt)
        # Passthrough-plane forwards (Discord interactions, Twilio, …) also ride
        # the SAME outbound WS (Phase 5 §5.1) — the connector edge-ACKed and
        # forwards the real request here, so a hosted gateway needs no public
        # inbound port. Bridge them to the adapter's passthrough handler.
        set_passthrough = getattr(self._transport, "set_passthrough_handler", None)
        if callable(set_passthrough):
            set_passthrough(self._on_passthrough)
        ok = await self._transport.connect()
        if not ok:
            return False
        # Negotiate the real capability descriptor from the connector and adopt
        # it — the placeholder passed at construction is replaced by what the
        # connector advertises for the platform this gateway actually fronts.
        try:
            descriptor = await self._transport.handshake()
        except Exception as exc:  # noqa: BLE001 - a failed handshake = a failed connect
            logger.warning("relay handshake failed: %s", exc)
            return False
        self._apply_descriptor(descriptor)
        # Inbound (messages + interrupts) is delivered over the outbound WS via
        # the connector's relay bus — there is NO inbound HTTP endpoint (hosted
        # gateways have no public IP). The transport's reader already dispatches
        # `inbound` / `interrupt_inbound` frames to the handlers wired above.
        # Phase 7 Unit 7d-B: start watching for a terminal auth revocation
        # (opt-out). Only meaningful when the transport exposes `auth_revoked`
        # (the production WebSocket transport); the test/stub transports don't.
        if hasattr(self._transport, "auth_revoked"):
            self._start_revocation_monitor()
        return True

    def _start_revocation_monitor(self) -> None:
        """Spawn (once) the task that turns a transport auth-revocation into a
        clean non-retryable 'relay disabled' fatal. Idempotent."""
        if self._revocation_monitor is not None and not self._revocation_monitor.done():
            return
        try:
            self._revocation_monitor = asyncio.create_task(
                self._watch_for_revocation(), name="relay-revocation-monitor"
            )
        except RuntimeError:
            # No running loop (e.g. a unit test calling connect() synchronously
            # via a stub) — nothing to monitor.
            self._revocation_monitor = None

    async def _watch_for_revocation(self, poll_interval_s: float = 1.0) -> None:
        """Poll the transport for a terminal 4401 revocation (opt-out). On
        revocation, surface a non-retryable `relay_disabled` fatal so the
        dashboard renders a clean 'Relay disabled' state instead of a red
        'retrying' spin, and notify the gateway's fatal-error handler so the
        adapter is cleanly removed (it is NOT queued for reconnection, because
        the credential is dead until the instance is recreated)."""
        transport = self._transport
        try:
            while True:
                if transport is None or getattr(transport, "auth_revoked", False):
                    break
                await asyncio.sleep(poll_interval_s)
        except asyncio.CancelledError:
            raise
        if transport is None or not getattr(transport, "auth_revoked", False):
            return
        logger.warning(
            "relay credential revoked (opt-out) — marking the relay adapter disabled"
        )
        # Non-retryable: a revoked secret never comes back without a recreate, so
        # _handle_adapter_fatal_error must NOT queue it for reconnection.
        self._set_fatal_error(
            "relay_disabled",
            "Relay disabled (opted out — recreate the instance to re-enable)",
            retryable=False,
        )
        try:
            await self._notify_fatal_error()
        except Exception:  # noqa: BLE001 - notification is best-effort
            logger.debug("relay revocation fatal-error notify failed", exc_info=True)

    def _apply_descriptor(self, descriptor: CapabilityDescriptor) -> None:
        """Adopt a (re)negotiated descriptor into the live capability surface."""
        self.descriptor = descriptor
        self.MAX_MESSAGE_LENGTH = descriptor.max_message_length
        self.supports_code_blocks = descriptor.markdown_dialect not in ("", "plain")

    async def _on_inbound(self, event) -> None:
        """Bridge a connector-delivered MessageEvent into the normal adapter path."""
        # Inbound replay dedupe (live-canary finding #3, Alice staging): the
        # relay leg is at-least-once — on WS re-handshake the connector
        # replays its durable per-instance buffer, and a long multi-tool turn
        # (60-100s) straddling a quiet socket drop gets its ORIGINAL inbound
        # replayed after the turn completes, re-running the whole turn (user
        # saw the final answer 2-5x). Platform message identity (chat_id +
        # message_id/ts) is stable across replays, so a bounded seen-set
        # drops them. Consumer-side idempotency; no wire change.
        dedupe_key = self._inbound_dedupe_key(event)
        if dedupe_key is not None:
            if dedupe_key in self._seen_inbound:
                logger.info(
                    "relay inbound dropped as replay (dedupe key=%s)", dedupe_key
                )
                return
            self._seen_inbound[dedupe_key] = None
            while len(self._seen_inbound) > self._SEEN_INBOUND_MAX:
                self._seen_inbound.pop(next(iter(self._seen_inbound)))
        self._capture_scope(event)
        self._stamp_slack_session_thread(event)
        # Phase 3: a structured prompt answer resolves its waiting primitive
        # (approval/confirm/clarify) and is CONSUMED — it must not also
        # dispatch as a chat message. Unknown/expired prompt ids fall through
        # (the command-shaped text then behaves like a typed reply).
        if await self._consume_prompt_response(event):
            return
        await self._localize_inbound_media(event)
        await self.handle_message(event)

    _SEEN_INBOUND_MAX = 512

    def _inbound_dedupe_key(self, event) -> Optional[str]:
        """Stable replay identity: (platform, chat, platform message id).

        Chat identity lives on ``event.source`` (MessageEvent has no top-level
        ``chat_id``), and this adapter can front SEVERAL platforms over one
        relay socket (Phase 1.5 multiplex), so the underlying platform joins
        the key — two platforms' numeric chat/message ids must never collide
        into one identity.

        Returns None when the event carries no platform message id (synthetic
        events, some prompt responses) — those never dedupe, fail-open by
        design: dropping a real user message is strictly worse than rerunning
        one, so only dedupe when identity is certain.
        """
        source = getattr(event, "source", None)
        message_id = getattr(event, "message_id", None)
        chat_id = getattr(source, "chat_id", None)
        if not message_id or not chat_id:
            return None
        platform = getattr(getattr(source, "platform", None), "value", "")
        return f"{platform}:{chat_id}:{message_id}"

    def _relay_slack_extra(self) -> Dict[str, Any]:
        """The Slack-behavior subset of the RELAY platform config.

        Enterprise knob shape (Hermes-config directed, relay-namespaced):

            platforms:
              relay:
                extra:
                  slack:              # supported subset of native Slack fields
                    reply_in_thread: true

        The native ``platforms.slack`` block keeps meaning "native adapter
        settings"; relay-fronted Slack reads its subset here. Legacy fallback:
        a flat key on the relay extra (``extra.reply_in_thread``) still wins
        when no ``slack`` object exists, preserving current staging configs.
        """
        extra = getattr(self.config, "extra", None) or {}
        sub = extra.get("slack")
        return sub if isinstance(sub, dict) else extra

    @staticmethod
    def _coerce_flag(raw: Any, default: bool) -> bool:
        """Coerce an operator-supplied boolean exactly as native Slack does.

        Native SlackAdapter reads its behavior flags with
        ``str(raw).strip().lower() in {"1","true","yes","on"}``, so a
        YAML-quoted ``"false"`` — a shape operators write routinely — turns
        the flag OFF. A bare ``bool()`` would read that same string as True
        (non-empty string), silently ignoring the off switch. These knobs are
        documented as native-parity mirrors, so they must coerce identically
        or the parity claim only holds for unquoted YAML booleans.
        """
        if raw is None:
            return default
        if isinstance(raw, bool):
            return raw
        return str(raw).strip().lower() in {"1", "true", "yes", "on"}

    def _effective_reply_in_thread(self) -> bool:
        """Resolve the thread-per-message vs flat-DM mode for fronted Slack."""
        try:
            return self._coerce_flag(
                self._relay_slack_extra().get("reply_in_thread"), True
            )
        except Exception:  # noqa: BLE001 - config shape is operator-owned
            return True

    def _dm_top_level_threads_as_sessions(self) -> bool:
        """Native-parity escape hatch: per-message DM sessions on/off.

        Mirrors native SlackAdapter._dm_top_level_threads_as_sessions
        (platforms.slack.extra.dm_top_level_threads_as_sessions). Default
        True: in thread-per-message mode each top-level DM message keys its
        own session (parallel turns). Set
        platforms.relay.extra.slack.dm_top_level_threads_as_sessions: false
        to keep threaded reply PLACEMENT but ONE rolling DM session — the
        legacy steer/queue posture, decoupled from reply_in_thread.
        """
        try:
            return self._coerce_flag(
                self._relay_slack_extra().get("dm_top_level_threads_as_sessions"),
                True,
            )
        except Exception:  # noqa: BLE001 - config shape is operator-owned
            return True

    def _stamp_slack_session_thread(self, event) -> None:
        """Native session-keying parity for fronted Slack DMs.

        Native SlackAdapter's inbound handler stamps ``thread_ts =
        event.thread_ts or ts`` — every TOP-LEVEL message carries its own ts
        as ``source.thread_id``, so build_session_key appends it and each
        top-level message gets a FRESH session (per-message threads ⇒
        per-message sessions; a 2nd message runs parallel instead of steering
        the in-flight turn). The connector normalizes a top-level message
        with thread_id=null, so without this stamp every top-level DM
        collapses into ONE session key and message 2 pre-empts message 1
        ("Redirected current run", 2026-07-27 report).

        Only in thread-per-message mode: flat mode keeps the shared rolling
        DM session on purpose (steer/queue there is the intended UX). Never
        overwrites a real thread_id (an in-thread reply must keep resolving
        to its thread's session).
        """
        try:
            src = getattr(event, "source", None)
            if not src:
                return
            platform = getattr(src, "platform", None)
            if getattr(platform, "value", platform) != Platform.SLACK.value:
                return
            if getattr(src, "thread_id", None):
                return  # real thread — its session key is already correct
            message_id = getattr(event, "message_id", None) or getattr(
                src, "message_id", None
            )
            if not message_id:
                return
            if not self._effective_reply_in_thread():
                return
            if not self._dm_top_level_threads_as_sessions():
                return  # opt-out: threaded replies, one rolling session
            src.thread_id = str(message_id)
        except Exception:  # noqa: BLE001 - session stamping must never break inbound
            logger.debug("slack session-thread stamp failed", exc_info=True)

    async def _localize_inbound_media(self, event) -> None:
        """Download connector re-hosted attachments to local temp paths.

        The wire's ``media_urls`` name connector re-hosts
        (``{connector}/relay/media/{id}``, per-gateway-bearer-authenticated) or
        public platform CDN URLs (Discord pass-through). Every NATIVE adapter
        presents inbound media to the agent as LOCAL FILE PATHS (the vision /
        file tools consume paths, and an authenticated URL would be useless in
        the agent's context anyway) — so mirror that here: fetch each URL and
        swap the list entries for temp paths. Best-effort per entry: a failed
        download drops that entry (never the message); no client ⇒ only
        re-host URLs are dropped (they'd 401 for every consumer downstream),
        public URLs stay.
        """
        try:
            urls = list(getattr(event, "media_urls", None) or [])
            if not urls:
                return
            client = self._get_media_client()
            localized: list[str] = []
            for url in urls:
                if not isinstance(url, str) or not url:
                    continue
                if client is None:
                    # No authenticated client: keep public URLs, drop re-hosts.
                    if "/relay/media/" not in url:
                        localized.append(url)
                    continue
                path = await client.download(url)
                if path:
                    localized.append(path)
                elif "/relay/media/" not in url:
                    # A public URL that failed to download still has value as
                    # a URL (native adapters pass URLs to vision in some
                    # lanes); a dead re-host reference does not.
                    localized.append(url)
            event.media_urls = localized
        except Exception:  # noqa: BLE001 - media localization must never break inbound
            logger.debug("relay inbound media localization failed", exc_info=True)

    def prime_routing_cache(self, event) -> None:
        """Warm the per-chat egress routing caches from a SYNTHETIC event.

        The caches (_scope_by_chat/_dm_user_by_chat/...) are normally warmed
        only by the inbound path (_on_inbound -> _capture_scope). A synthetic
        completion turn injected right after a restart (durable
        async-delegation replay) reaches handle_message with the caches COLD,
        so every reply it produces egresses without metadata.scope_id /
        metadata.user_id and is declined by the connector's fail-closed
        tenant guard ("target not routed to an onboarded tenant" — staging
        2026-08-09, defect #4). The synthetic event's session-store origin
        already carries the discriminators; feed it through the same capture
        used for real inbound. Never raises.
        """
        if event is None or getattr(event, "source", None) is None:
            return
        self._capture_scope(event)

    def _capture_scope(self, event) -> None:
        """Remember a chat_id's egress discriminator from an inbound event so our
        outbound (the agent's reply) can re-assert it for the connector's egress
        tenant resolution. Never raises — scope tracking must not break inbound.

        Two discriminators, captured independently (a scoped message has BOTH):
          - scope_id: for a scoped (guild/channel) message. The connector's
            primary path resolves the tenant from metadata.scope_id (routing
            table).
          - user_id: the authentic author id, captured for EVERY message (DM
            and scoped alike). The connector resolves the tenant from the
            recipient's author binding (resolveByUser) when a route lookup
            misses. This is the sole discriminator for a DM (no scope), AND the
            author-first FALLBACK for a scoped reply whose guild has no route
            row — a managed agent joins guilds dynamically, so a provision-time
            guild route is not guaranteed. Re-attaching user_id on scoped
            replies too lets the connector's guild-route-miss fallback resolve
            the tenant the same way inbound already does (SharedSocketRouter
            targets()). Without a resolvable discriminator the connector's
            egress guard declines the reply as 'target not routed to an
            onboarded tenant'. See gateway-gateway routedEgressGuard.ts /
            discordTenant.ts (makeDiscordTenantOf).
        """
        try:
            src = getattr(event, "source", None)
            if not src:
                return
            chat = getattr(src, "chat_id", None)
            if not chat:
                return
            # Phase 1.5: remember the underlying platform for this chat so the
            # reply egresses through the right sender (one relay adapter fronts N
            # platforms). source.platform is a Platform enum (e.g. Platform.DISCORD,
            # mapped from the connector's "discord" by ws_transport _frame_to_event);
            # record its string VALUE, skipping the generic RELAY fallback (a
            # single-platform connector that didn't tag a concrete platform — the
            # connector's session default handles egress then).
            platform = getattr(src, "platform", None)
            platform_value = getattr(platform, "value", platform)
            if platform_value and platform_value != "relay":
                self._platform_by_chat[str(chat)] = str(platform_value)
            # Author id for outbound author-binding resolution. Captured for BOTH
            # DM and scoped messages: it's the sole discriminator for a DM and
            # the guild-route-miss fallback for a scoped reply. (Formerly captured
            # for DMs only, which left managed-agent guild replies with no
            # resolvable tenant when the guild had no route row.)
            user_id = getattr(src, "user_id", None)
            if user_id:
                self._dm_user_by_chat[str(chat)] = str(user_id)
            scope = getattr(src, "scope_id", None)
            if scope:
                self._scope_by_chat[str(chat)] = str(scope)
            # Remember the chat_type so send() can suppress the synthetic-DM
            # thread anchor on Slack (native _resolve_thread_ts parity). send()
            # only receives a chat_id, so it needs this per-chat cache to know a
            # chat is a DM.
            chat_type = getattr(src, "chat_type", None)
            if chat_type:
                self._chat_type_by_chat[str(chat)] = str(chat_type)
            # Triggering message ts: the typing/status lane's metadata
            # (base.py _thread_metadata_for_source) carries NO thread anchor
            # for a top-level DM, but in thread-per-message mode the status
            # must target the per-message thread (its root = this ts). Cache
            # it per chat so send_typing can synthesize the anchor, mirroring
            # native send_typing's _resolve_thread_ts(metadata.message_id).
            # NOTE: message_id lives on the EVENT (MessageEvent), not the
            # source — fall back to source for defensive coverage.
            message_id = getattr(event, "message_id", None) or getattr(
                src, "message_id", None
            )
            if message_id:
                self._last_inbound_ts_by_chat[str(chat)] = str(message_id)
        except Exception:  # noqa: BLE001 - scope tracking must never break inbound
            pass

    def _with_scope(
        self, chat_id: str, metadata: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Ensure the outbound metadata carries the discriminator(s) the connector's
        egress guard needs to resolve the owning tenant.

          - scope_id: re-attached for a scoped reply (guild/channel) →
            routing-table resolution (the primary path).
          - user_id: the authentic author id we saw inbound, re-attached for
            EVERY reply we know it for. It is the sole discriminator for a DM
            (no scope), AND the author-first FALLBACK the connector uses when a
            scoped reply's guild has no route row (a managed agent joins guilds
            dynamically — the guild route may not be provisioned). Carrying both
            on a scoped reply is harmless: the connector tries scope_id first and
            only falls back to user_id on a route miss. Without a resolvable
            discriminator egress is declined as 'target not routed to an
            onboarded tenant'. See gateway-gateway routedEgressGuard.ts /
            discordTenant.ts.

        No-op when the relevant value is already present or unknown for this chat.
        """
        meta: Dict[str, Any] = dict(metadata or {})
        if not meta.get("scope_id"):
            scope = self._scope_by_chat.get(str(chat_id))
            if scope:
                meta["scope_id"] = scope
        # Author-binding discriminator. Attached whenever we know the author for
        # this chat and it isn't already set — for DMs (the sole discriminator)
        # AND scoped replies (the connector's guild-route-miss fallback). It is
        # only consulted by the connector when the scope/route lookup misses, so
        # carrying it alongside scope_id never overrides routing-table resolution.
        if not meta.get("user_id"):
            author = self._dm_user_by_chat.get(str(chat_id))
            if author:
                meta["user_id"] = author
        return meta

    def fronts_platform(self, platform: Any) -> bool:
        """Whether the authenticated relay transport advertises ``platform``.

        This is the restart-safe delivery ownership signal: it comes from the
        configured identity set sent during handshake, not from an inbound
        chat cache learned only after a user sends another message.
        """
        platform_value = getattr(platform, "value", platform)
        if not platform_value:
            return False
        ids = getattr(self._transport, "_identities", None)
        if not ids:
            return False
        return any(p == str(platform_value) for p, _ in ids)

    def _platform_is_fronted(self, platform: str) -> bool:
        """Backward-compatible internal alias for follow-up routing."""
        return self.fronts_platform(platform)

    async def on_interrupt(self, session_key: str, chat_id: str) -> None:
        """Bridge a connector-delivered /stop into the adapter's interrupt path.

        The connector forwards a mid-turn interrupt down the socket owned by
        the gateway instance running ``session_key``; this routes it to the
        existing per-session interrupt mechanism (sets the
        ``_active_sessions[session_key]`` Event and clears typing), cancelling
        the right turn without touching sibling sessions.
        """
        await self.interrupt_session_activity(session_key, chat_id)

    async def _on_passthrough(self, forward, buffer_id: Optional[str] = None) -> None:
        """Handle a connector-forwarded passthrough request (Phase 5 §5.1).

        The passthrough plane (Discord interactions, Twilio webhooks, …) answers
        the provider's latency-critical ACK at the connector EDGE, then forwards
        the real, ALREADY-SANITIZED request to this gateway over the outbound WS.
        The connector is the trust boundary: it verified the provider signature
        at the edge and stripped any shared-identity credential (e.g. a Discord
        interaction follow-up token) into its vault — so this body carries no
        token, and the agent later acts on it via the token-less ``follow_up``
        path (``send_follow_up``), never holding the credential.

        For a Discord interaction we decode the (JSON) body and convert it to a
        normalized ``MessageEvent`` so it flows through the SAME agent path as a
        chat message (``handle_message``); the agent's reply egresses over the
        normal outbound/follow_up path. Non-JSON or non-interaction forwards are
        logged and dropped for now (Twilio/SMS over the relay is a later unit).

        NEVER raises: a malformed forward must not kill the read loop.

        Interaction -> MessageEvent command mapping (formerly flagged here as an
        open sub-design, now implemented): an APPLICATION_COMMAND interaction is
        normalized to a leading-slash COMMAND event ("/name arg…", mirroring the
        connector's Slack slash-command lane, normalizeSlackCommand), so the
        dispatcher routes it as a command instead of plain chat. Component
        interactions (custom_id) still surface as best-effort TEXT; the
        deferred-vs-immediate response UX remains connector-side.
        """
        try:
            platform = getattr(forward, "platform", "") or ""
            if platform == "discord":
                event = self._discord_interaction_to_event(forward)
                if event is not None:
                    self._capture_scope(event)
                    # Phase 3: a component press carrying a Hermes prompt token
                    # resolves its waiting primitive and is consumed (same
                    # gate as _on_inbound's prompt_response arm).
                    if await self._consume_prompt_response(event):
                        return
                    await self.handle_message(event)
                    return
            logger.info(
                "relay passthrough_forward dropped (no handler): platform=%s method=%s path=%s",
                platform,
                getattr(forward, "method", "?"),
                getattr(forward, "path", "?"),
            )
        except Exception:  # noqa: BLE001 - a bad forward must never break the reader
            logger.warning("relay passthrough_forward handling failed", exc_info=True)

    def _discord_interaction_to_event(self, forward):
        """Convert a forwarded Discord interaction body to a MessageEvent, or None.

        Builds the session source the same way the connector does for an
        interaction (``interactionSessionSource`` on the connector side), so the
        agent's session key matches the one the connector bound the follow-up
        capability under. Returns None when the body isn't a usable interaction
        (e.g. a PING, which the connector already answers at the edge and never
        forwards).
        """
        import json

        from gateway.platforms.base import MessageType

        try:
            payload = json.loads(bytes(getattr(forward, "body", b"")).decode("utf-8"))
        except Exception:  # noqa: BLE001
            return None
        if not isinstance(payload, dict):
            return None
        # type 1 = PING (answered at the edge, never forwarded); 2 = APPLICATION_COMMAND;
        # 3 = MESSAGE_COMPONENT; 5 = MODAL_SUBMIT. Surface a best-effort text.
        itype = payload.get("type")
        data = payload.get("data") or {}
        message_type = MessageType.TEXT
        if itype == 2:
            # Normalize a real slash-command interaction to a leading-slash
            # command string — the shape the dispatcher (MessageEvent.is_command:
            # text.startswith("/")) and the native Discord adapter's
            # _run_simple_slash lane (f"/model {name}".strip()) both expect.
            # Options render space-separated: scalar options contribute their
            # value; SUB_COMMAND/SUB_COMMAND_GROUP (types 1/2) contribute their
            # name then their nested options. Mirrors the connector's Slack
            # slash lane (normalizeSlackCommand: `${command} ${args}`.trim()).
            text = ("/" + str(data.get("name") or "")).rstrip("/") or ""
            if text:
                parts = [text] + self._render_interaction_options(data.get("options"))
                text = " ".join(parts).strip()
                message_type = MessageType.COMMAND
        elif itype == 3:
            text = str(data.get("custom_id") or "")
        else:
            text = ""
        member = payload.get("member") or {}
        user = (
            (member.get("user") if isinstance(member, dict) else None)
            or payload.get("user")
            or {}
        )
        channel_id = str(payload.get("channel_id") or "")
        guild_id = payload.get("guild_id")  # real Discord interaction wire field
        source = SessionSource(
            # The LOGICAL platform, not Platform.RELAY. This lane parses a
            # Discord interaction wire payload, so the underlying platform is
            # known statically — and it must be stamped for three consumers:
            #   1. Session keys: the connector binds the interaction's
            #      follow-up capability under buildSessionKey with
            #      platform="discord" (interactionSessionSource); the relay
            #      TEXT lane (ws_transport._event_from_wire) also maps to the
            #      logical platform. RELAY here forked interaction sessions
            #      away from both.
            #   2. /sethome: with platform=RELAY the handler filed the home
            #      channel under platforms.relay.home_channel (invisible to
            #      cron delivery, which looks up the logical platform) and
            #      mirrored it into the dead RELAY_HOME_CHANNEL env var.
            #   3. Egress: _capture_scope records _platform_by_chat from this
            #      value and deliberately skips the generic "relay".
            platform=Platform.DISCORD,
            chat_id=channel_id,
            # "group", not "channel": the session key embeds chat_type, and
            # BOTH the connector's capability binding (interactionSessionSource
            # → buildSessionKey, chat_type "group") and the native Discord
            # adapter's channel events key guild channels as "group". A
            # "channel" slot here forked the interaction session from the
            # chat's message session AND from the vault key the connector
            # bound the follow-up capability under.
            chat_type="group" if guild_id else "dm",
            user_id=str(user.get("id"))
            if isinstance(user, dict) and user.get("id")
            else None,
            user_name=str(user.get("username"))
            if isinstance(user, dict) and user.get("username")
            else None,
            scope_id=str(guild_id)
            if guild_id
            else None,  # Discord guild → generic scope slot
            message_id=str(payload.get("id")) if payload.get("id") else None,
            # Same upstream-trust marker the relay text lane stamps
            # (ws_transport._event_from_wire): this interaction arrived over
            # the per-instance-authenticated relay WS after the connector
            # verified Discord's edge signature and resolved the tenant.
            # Without it, authz treated the event as unauthenticated relay
            # traffic — and /sethome's via_relay guard never engaged, which is
            # how platform=RELAY home channels slipped through in the first
            # place. Set locally, never read off the wire.
            delivered_via_upstream_relay=True,
        )
        event = MessageEvent(text=text, message_type=message_type, source=source)
        if itype == 3:
            # Phase 3: a component press whose custom_id is a Hermes prompt
            # token (hp1:<prompt_id>:<option_id>) becomes a STRUCTURED prompt
            # answer — _on_inbound's _consume_prompt_response then resolves
            # the waiting approval/confirm/clarify, replacing the bare-
            # custom_id-as-text stub. Foreign custom_ids keep the legacy
            # best-effort TEXT shape.
            decoded = self._decode_prompt_token(text)
            if decoded:
                prompt_id, option_id = decoded
                msg = payload.get("message") or {}
                prompt_message_id = (
                    str(msg.get("id"))
                    if isinstance(msg, dict) and msg.get("id")
                    else None
                )
                event.prompt_response = {
                    "prompt_id": prompt_id,
                    "option_id": option_id,
                    "prompt_message_id": prompt_message_id,
                }
                event.text = f"/{option_id}"
                event.message_type = MessageType.COMMAND
        return event

    @staticmethod
    def _decode_prompt_token(token: str):
        """Decode an hp1:<prompt_id>:<option_id> callback token, or None.

        Mirrors the connector's promptCodec.decodePromptCallback (the token
        alphabet is [A-Za-z0-9_.-], ≤32 per id) so both ends agree on what is
        — and is not — a Hermes prompt answer.
        """
        import re

        if not token:
            return None
        parts = token.split(":")
        if len(parts) != 3 or parts[0] != "hp1":
            return None
        id_re = re.compile(r"^[A-Za-z0-9_.\-]{1,32}$")
        if not id_re.match(parts[1]) or not id_re.match(parts[2]):
            return None
        return parts[1], parts[2]

    @staticmethod
    def _render_interaction_options(options) -> list:
        """Render Discord interaction options to space-separated text parts.

        Discord's `data.options` is a list of {name, value, type}. Scalar
        options (STRING/INTEGER/BOOLEAN/…) contribute just their value —
        matching the native adapter's `f"/model {name}".strip()` shape, where
        only the value follows the command. SUB_COMMAND (1) and
        SUB_COMMAND_GROUP (2) contribute their *name* then recurse into their
        nested `options` list (one level of nesting per Discord's schema:
        group -> subcommand -> scalars).
        """
        parts: list = []
        if not isinstance(options, list):
            return parts
        for opt in options:
            if not isinstance(opt, dict):
                continue
            if opt.get("type") in (1, 2):  # SUB_COMMAND / SUB_COMMAND_GROUP
                sub_name = str(opt.get("name") or "").strip()
                if sub_name:
                    parts.append(sub_name)
                parts.extend(
                    RelayAdapter._render_interaction_options(opt.get("options"))
                )
            else:
                value = opt.get("value")
                if value is not None and str(value).strip():
                    parts.append(str(value).strip())
        return parts

    async def disconnect(self) -> None:
        # Budget accounting: the runner wraps this whole call in
        # asyncio.wait_for(_adapter_disconnect_timeout_secs()). Everything we
        # spend on monitor teardown and go_idle below eats into what the
        # transport can spend on its outbound drain, so measure from the top
        # and thread the REMAINDER down — otherwise worst-case
        # monitor(1s) + go_idle(2s) + drain + 3×teardown(1s) can blow the 5s
        # budget, cancelling teardown mid-drain and skipping the transport's
        # fail-pending loop (callers then block on _OUTBOUND_TIMEOUT_S).
        from gateway.relay.ws_transport import _env_disconnect_budget_s
        _started = time.monotonic()
        _budget = _env_disconnect_budget_s()
        # Phase 7 Unit 7d-B: stop the revocation monitor first so it can't fire a
        # spurious fatal during/after a deliberate teardown.
        if self._revocation_monitor is not None:
            self._revocation_monitor.cancel()
            try:
                await asyncio.wait_for(
                    self._revocation_monitor,
                    timeout=_RELAY_REVOCATION_MONITOR_TEARDOWN_TIMEOUT_S,
                )
            except (asyncio.TimeoutError, asyncio.CancelledError, Exception):  # noqa: BLE001 - best-effort teardown
                pass
            self._revocation_monitor = None
        if self._transport is not None:
            # Phase 5 §5.3: emit going_idle as part of the gateway's EXISTING
            # drain/shutdown transition (the runner calls adapter.disconnect()
            # when the gateway enters `draining`). Asking the connector to flip
            # this instance to buffered-only BEFORE we tear down the socket means
            # inbound that arrives while we're asleep buffers durably and replays
            # on reconnect, instead of being pushed at a closing socket. The
            # connector is authoritative (it acks the flip); we stay serving until
            # the ack (Q-5.3c). Best-effort + guarded: a transport without go_idle
            # (the stub) or a failed/timed-out ack must not block shutdown — we
            # proceed to disconnect exactly as before, no regression.
            #
            # transport.disconnect() runs in finally so an outer cancellation
            # during go_idle (runner default adapter budget is 5s) still closes
            # the socket/supervisor instead of leaking them. shield() keeps the
            # teardown await itself from being cancelled mid-flight.
            try:
                go_idle = getattr(self._transport, "go_idle", None)
                if callable(go_idle):
                    try:
                        result: Any = go_idle(
                            timeout_s=_RELAY_GO_IDLE_ON_DISCONNECT_TIMEOUT_S
                        )
                        if asyncio.iscoroutine(result):
                            await result
                    except Exception:  # noqa: BLE001 - going-idle is an optimization, never blocks drain
                        logger.debug(
                            "relay going_idle failed during drain", exc_info=True
                        )
            finally:
                try:
                    _remaining = max(0.0, _budget - (time.monotonic() - _started))
                    try:
                        _td = cast(Any, self._transport).disconnect(
                            budget_s=_remaining
                        )
                    except TypeError:
                        # Transports without the budget_s keyword (stubs,
                        # older implementations) keep the legacy signature.
                        _td = self._transport.disconnect()
                    await asyncio.shield(_td)
                except Exception:  # noqa: BLE001 - teardown must not block outer cancel propagation
                    logger.debug(
                        "relay transport disconnect failed during drain",
                        exc_info=True,
                    )

    async def go_dormant(self) -> bool:
        """Quiesce the relay for a scale-to-zero suspend (D12 / Phase 0).

        Unlike ``disconnect()`` (terminal teardown for shutdown/restart), this
        keeps the adapter's reconnect path armed so the gateway re-dials and
        drains its buffered backlog when the machine wakes. Delegates to the
        transport's ``go_dormant()`` when available; a transport without it (the
        stub) is a no-op that returns False, so callers degrade safely.

        NOTE: deliberately does NOT stop the revocation monitor — going dormant
        is not a teardown; the monitor stays live so a real opt-out/revocation
        during dormancy is still surfaced on wake.
        """
        if self._transport is None:
            return False
        go_dormant = getattr(self._transport, "go_dormant", None)
        if not callable(go_dormant):
            return False
        try:
            result: Any = go_dormant()
            if asyncio.iscoroutine(result):
                return bool(await result)
            return bool(result)
        except Exception:  # noqa: BLE001 - dormancy is best-effort, never blocks the idle path
            logger.debug("relay go_dormant failed", exc_info=True)
            return False

    async def send_for_platform(
        self,
        logical_platform: Any,
        chat_id: str,
        content: str,
        reply_to: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SendResult:
        """Send to an explicitly advertised logical platform over Relay.

        Scheduled and persisted-home deliveries have no fresh inbound event to
        populate ``_platform_by_chat``. The shared delivery resolver calls this
        method only after ``fronts_platform`` succeeds, and this method repeats
        that check fail-closed before stamping the outbound frame.
        """
        platform_value = getattr(logical_platform, "value", logical_platform)
        if not self.fronts_platform(platform_value):
            return SendResult(
                success=False,
                error=f"relay does not front platform {platform_value}",
            )
        if self._transport is None:
            return SendResult(success=False, error="no transport")
        result = await self._transport.send_outbound(
            {
                "op": "send",
                "chat_id": chat_id,
                "content": content,
                "reply_to": reply_to,
                "metadata": self._with_scope(chat_id, metadata),
            },
            platform=str(platform_value),
        )
        return SendResult(
            success=bool(result.get("success")),
            message_id=result.get("message_id"),
            error=result.get("error"),
            raw_response=result,
        )

    async def send(
        self,
        chat_id: str,
        content: str,
        reply_to: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SendResult:
        send_metadata = dict(metadata or {})
        explicit_platform = send_metadata.pop("_relay_logical_platform", None)
        if explicit_platform:
            return await self.send_for_platform(
                explicit_platform,
                chat_id,
                content,
                reply_to=reply_to,
                metadata=send_metadata or None,
            )
        if self._transport is None:
            return SendResult(success=False, error="no transport")
        # Native _resolve_thread_ts parity: a Slack DM reply must post flat at
        # the DM root, not threaded under the triggering message. One shared
        # helper resolves the anchor for EVERY egress lane (see
        # _apply_slack_thread_anchor) so the text and media lanes cannot drift.
        effective_reply_to = self._apply_slack_thread_anchor(
            chat_id, reply_to, send_metadata
        )
        result = await self._transport.send_outbound(
            {
                "op": "send",
                "chat_id": chat_id,
                "content": content,
                "reply_to": effective_reply_to,
                "metadata": self._with_scope(chat_id, send_metadata),
            },
            platform=self._platform_by_chat.get(str(chat_id)),
        )
        # Auto-thread routing feedback (contract §SendResult): when the
        # connector's auto-thread egress policy routed this send into a
        # thread it just created, the result carries thread_id (+ the
        # initial name). The conversation was keyed on the PARENT channel —
        # the thread didn't exist at ingest — so this is the only place the
        # gateway learns where the reply landed. The semantic-rename lane
        # (auto session title) reads it via auto_thread_info_for_chat.
        try:
            _at_thread = result.get("thread_id")
            _at_name = result.get("auto_thread_name")
            if _at_thread and _at_name:
                self._auto_thread_by_chat[str(chat_id)] = (
                    str(_at_thread),
                    str(_at_name),
                )
                if len(self._auto_thread_by_chat) > 256:
                    self._auto_thread_by_chat.pop(
                        next(iter(self._auto_thread_by_chat)), None
                    )
        except Exception:  # noqa: BLE001 - feedback capture must never break send
            pass
        # Wake the rename lane on EVERY send into this chat, not only the ones
        # that auto-threaded. It is waiting to learn where this turn's reply
        # landed, and "nowhere new" is an answer — one it should get now rather
        # than by outlasting a timeout.
        waiter = self._auto_thread_waiters.get(str(chat_id))
        if waiter is not None:
            waiter.set()
        return SendResult(
            success=bool(result.get("success")),
            message_id=result.get("message_id"),
            error=result.get("error"),
        )

    def auto_thread_info_for_chat(
        self, chat_id: str
    ) -> Optional[Tuple[str, str]]:
        """(thread_id, initial_name) of the auto-thread the connector created
        for the most recent send into *chat_id*, if any. Consumed by the
        gateway's semantic thread-rename lane (auto session title)."""
        return self._auto_thread_by_chat.get(str(chat_id))

    async def wait_for_auto_thread_info(
        self, chat_id: str, timeout: float
    ) -> Optional[Tuple[str, str]]:
        """``auto_thread_info_for_chat``, but willing to wait for the send.

        The rename lane asks where the reply landed as soon as the session is
        titled, and the session is titled from the user's opening message —
        before the model has answered, let alone before we've sent anything. So
        the question arrives a whole turn early, and a turn is a one-liner or
        twenty minutes of tool calls.

        Waits for the next send into this chat and then answers, so a reply the
        connector didn't auto-thread reports its miss as soon as it's sent
        instead of holding until *timeout* — which is only a backstop for a turn
        that never sends at all.
        """
        info = self.auto_thread_info_for_chat(chat_id)
        if info is not None:
            return info
        key = str(chat_id)
        waiter = self._auto_thread_waiters.get(key)
        if waiter is None:
            waiter = asyncio.Event()
            self._auto_thread_waiters[key] = waiter
        try:
            await asyncio.wait_for(waiter.wait(), timeout)
        except asyncio.TimeoutError:
            return None
        finally:
            # Only the waiter we may have installed, and only if no later call
            # replaced it; a fired event must not be left behind to make the
            # next turn's wait return instantly on stale feedback.
            if self._auto_thread_waiters.get(key) is waiter:
                self._auto_thread_waiters.pop(key, None)
        return self.auto_thread_info_for_chat(chat_id)

    def _resolve_reply_to_for_send(
        self,
        chat_id: str,
        reply_to: Optional[str],
        metadata: Optional[Dict[str, Any]],
    ) -> Optional[str]:
        """Suppress the synthetic-DM thread anchor for a Slack DM reply.

        A DM turn's streaming reply is sent with ``reply_to`` = the triggering
        message's ts (the stream consumer's ``initial_reply_to_id``, used as the
        edit anchor and, on threading platforms, the reply target). The
        connector's slackRestSender maps a raw ``reply_to`` to a Slack
        ``thread_ts``, so a plain DM reply would be posted THREADED under the
        user's message instead of flat at the DM root — and a threaded first
        send loses the progressive edit-streaming the user sees in a real
        thread (the reported symptom: DM/home replies arrive flat, no
        progressive edits).

        Native Slack Hermes already suppresses this synthetic DM thread anchor:
        ``SlackAdapter._resolve_thread_ts`` returns ``None`` for a top-level /
        DM message when ``reply_in_thread`` is off. The relay lane has no such
        disambiguation, so we reproduce it here. run.py already encodes the
        real-thread decision in ``metadata["thread_id"]`` (it is set only when
        progress threading is active — a real thread, or channel autoThread);
        for a DM with no real thread that key is absent. So the rule is:

          Slack DM + no real ``thread_id`` in metadata  ⇒  drop ``reply_to``.

        This posts the reply flat at the DM root and lets the consumer edit its
        own first-send ts — streaming works exactly as in a thread. It does NOT:
          * reintroduce a synthetic DM thread_id (#18859 / the /sethome
            landmine) — it removes an anchor, never adds one;
          * regress real-thread streaming — a real thread carries a distinct
            ``thread_id`` in metadata, so the guard leaves ``reply_to`` alone;
          * regress channel autoThread — a channel/group top-level reply carries
            ``thread_id`` (the message's own ts) in metadata when threading is
            on, so it is left alone; and a non-DM chat is never matched here.
        """
        if reply_to is None:
            return None
        if self._platform_by_chat.get(str(chat_id)) != Platform.SLACK.value:
            return reply_to
        if self._chat_type_by_chat.get(str(chat_id)) != "dm":
            return reply_to
        md = metadata or {}
        if md.get("thread_id") or md.get("thread_ts"):
            # A real thread was resolved by run.py — honour it.
            return reply_to
        # Mode gate (native _resolve_thread_ts parity). The final-reply lane
        # (gateway/platforms/base.py) builds metadata from source.thread_id
        # ONLY — for a top-level DM that is None, so in thread-per-message
        # mode the triggering-ts reply_to here is the final reply's ONLY
        # threading signal (run.py's synthetic root feeds just the
        # progress/status lane). Dropping it unconditionally exiled the final
        # message to the DM root while progress stayed threaded (2026-07-27
        # report, same class as the prompt-placement bug). Native SlackAdapter only
        # suppresses the anchor when reply_in_thread=false; mirror that.
        reply_in_thread = self._effective_reply_in_thread()
        if reply_in_thread:
            # Thread-per-message: the triggering ts is the thread anchor.
            return reply_to
        # Flat mode: synthetic DM self-anchor — post flat at the DM root.
        return None

    def _apply_slack_thread_anchor(
        self,
        chat_id: str,
        reply_to: Optional[str],
        metadata: Dict[str, Any],
        *,
        mirror_key: str = "reply_to_message_id",
    ) -> Optional[str]:
        """Resolve the outbound Slack thread anchor for ONE egress frame.

        The single choke point every send lane goes through — text (``send``)
        and media (``send_media``) alike. It does three things that must always
        happen together, and previously only happened on the text lane:

          1. Mode gate: ``_resolve_reply_to_for_send`` drops the synthetic DM
             self-anchor in flat mode, keeps it in thread-per-message mode.
          2. Mirror strip: when the anchor is dropped, remove the mirrored
             ``metadata.reply_to_message_id`` too, so the connector cannot
             thread on the copy we forgot about.
          3. Anchor promotion: the connector's Slack sender THREADS ON METADATA
             ONLY — ``threadTs()`` reads ``metadata.thread_id``/``thread_ts``
             and never looks at the frame's ``reply_to``. A surviving anchor is
             promoted into ``metadata.thread_id`` or the message silently lands
             in the home channel instead of the per-message thread.

        ``metadata`` is mutated in place; the effective ``reply_to`` is
        returned. Non-Slack and non-DM chats are untouched by (1), and (3) is
        Slack-only, so other fronted platforms keep their existing behaviour.
        """
        effective_reply_to = self._resolve_reply_to_for_send(
            chat_id, reply_to, metadata
        )
        if effective_reply_to is None and reply_to is not None:
            metadata.pop(mirror_key, None)
        if (
            effective_reply_to is not None
            and self._platform_by_chat.get(str(chat_id)) == Platform.SLACK.value
            and not (metadata.get("thread_id") or metadata.get("thread_ts"))
        ):
            metadata["thread_id"] = str(effective_reply_to)
        return effective_reply_to

    def _with_status_thread_anchor(
        self, chat_id: str, metadata: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Copy ``metadata`` with the typing/status thread anchor applied.

        Slack's status line is THREAD-scoped: the connector's typing case
        no-ops without a thread anchor, and the typing lane's metadata
        (base.py ``_thread_metadata_for_source``) carries none for a top-level
        DM (``source.thread_id`` is None). Synthesize it from the per-chat
        inbound-ts cache, exactly as native ``send_typing`` resolves
        ``thread_ts`` from ``metadata.message_id``.

        Unconditional across both modes: in flat mode the send lane strips its
        own anchors (see ``_apply_slack_thread_anchor``), so the status anchor
        cannot leak into reply placement, and ``setStatus`` clears without
        leaving a message artifact.

        Shared by ``send_typing`` and ``stop_typing`` — the clear MUST target
        the same thread the heartbeat set, or the status line sticks until
        Slack's own timeout. Keeping one implementation is what guarantees it.
        """
        md = dict(metadata or {})
        if (
            not (md.get("thread_id") or md.get("thread_ts"))
            and self._platform_by_chat.get(str(chat_id)) == Platform.SLACK.value
            and self._chat_type_by_chat.get(str(chat_id)) == "dm"
        ):
            anchor = self._last_inbound_ts_by_chat.get(str(chat_id))
            if anchor:
                md["thread_id"] = anchor
        return md

    async def edit_message(
        self,
        chat_id: str,
        message_id: str,
        content: str,
        *,
        finalize: bool = False,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SendResult:
        """Edit a relayed message through the connector-owned platform API."""
        if self._transport is None:
            return SendResult(success=False, error="no transport")
        result = await self._transport.send_outbound(
            {
                "op": "edit",
                "chat_id": chat_id,
                "message_id": message_id,
                "content": content,
                "metadata": self._with_scope(chat_id, metadata),
            },
            platform=self._platform_by_chat.get(str(chat_id)),
        )
        return SendResult(
            success=bool(result.get("success")),
            message_id=result.get("message_id") or message_id,
            error=result.get("error"),
        )

    async def send_typing(self, chat_id: str, metadata=None) -> None:
        """Egress a typing indicator through the connector.

        The base class spawns ``_keep_typing`` for every adapter (a 2s refresh
        loop for the life of the turn), but the relay adapter inherited the
        base no-op ``send_typing`` — so hosted/relay chats never showed
        "is typing…" even though the wire contract (``OutboundOp "typing"``)
        and every connector-side sender (Discord ``POST /channels/{id}/typing``,
        Telegram ``sendChatAction``, Signal ``sendTyping``, Slack assistant
        status) already implement it. This bridges the loop's tick onto the
        existing outbound frame.

        Two details are load-bearing, mirroring ``send()``:
          - ``_with_scope``: the connector's egress guard wraps ALL ops
            (routedEgressGuard), so a typing frame without a resolvable tenant
            discriminator (metadata.scope_id, or user_id for DMs) is declined
            exactly like a bare send would be.
          - the per-frame ``platform`` tag (Phase 1.5): a multi-platform
            gateway must egress typing through the platform the chat lives on.

        Best-effort: failures are swallowed (``_keep_typing`` already treats
        send_typing errors as non-fatal, and an older connector that rejects
        the op just returns an unsuccessful result we ignore). Each call is
        one-shot — Discord/Telegram indicators self-expire and need no cleanup;
        Slack Assistant status persists, so ``stop_typing`` below sends an
        explicit clear for Slack only.
        """
        if self._transport is None:
            return
        # Thread anchor for the status surface. Slack's status line
        # ("is thinking…" in the thread's replies footer — works with plain
        # chat:write, confirmed on native no-assistant bots) is THREAD-only:
        # the connector's typing case no-ops without a thread_ts. But the
        # typing lane's metadata (base.py _thread_metadata_for_source) has no
        # anchor for a top-level DM — source.thread_id is None — so every
        # heartbeat was silently dropped. In thread-per-message mode the
        # turn's thread root IS the triggering message ts (run.py's synthetic
        # root); synthesize it here from the per-chat inbound cache, exactly
        # like native send_typing resolves thread_ts from metadata.message_id.
        # Flat mode (reply_in_thread=false) keeps the no-anchor no-op: there
        # is no thread and must not be one (#18859).
        md = self._with_status_thread_anchor(chat_id, metadata)
        # Rich status parity: run.py's live-status lane stashes the
        # current per-tool phrase via set_status_text() (base class store).
        # Carry it as the typing frame's content so the connector's Slack
        # sender renders it on assistant.threads.setStatus — the same phrase
        # the native adapter shows ("is running pytest…", "Finding answers…").
        # Absent (None/empty) => omit content; the connector falls back to its
        # default "is typing…" heartbeat, preserving pre-phrase behaviour on
        # every platform. Never send empty-string content here: on Slack that
        # is the explicit CLEAR request reserved for stop_typing.
        frame: Dict[str, Any] = {
            "op": "typing",
            "chat_id": chat_id,
            "metadata": self._with_scope(chat_id, md),
        }
        phrase = getattr(self, "_status_text", {}).get(str(chat_id))
        if phrase:
            frame["content"] = str(phrase)
        try:
            await self._transport.send_outbound(
                frame,
                platform=self._platform_by_chat.get(str(chat_id)),
            )
        except Exception:  # noqa: BLE001 - typing is cosmetic, never breaks a turn
            logger.debug("relay send_typing failed for %s", chat_id, exc_info=True)

    async def stop_typing(
        self,
        chat_id: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Forward an explicit typing/status clear to the connector.

        Slack Assistant status persists until explicitly cleared (empty
        ``content`` on the ``typing`` op). Other relay senders expose only
        one-shot typing heartbeats; sending an empty heartbeat there would
        incorrectly re-trigger typing at completion, so this is Slack-gated.

        NOTE (deploy order): a connector older than gateway-gateway #154
        hardcodes ``status: "is typing…"`` for the typing op, so an empty
        clear frame would SET the status instead of clearing it. Deploy the
        connector first. Best-effort like ``send_typing``: status clearing is
        cosmetic and must never break turn completion.
        """
        if self._transport is None:
            return
        platform = self._platform_by_chat.get(str(chat_id))
        if platform != Platform.SLACK.value:
            return
        # Clear must target the SAME thread the heartbeat set, or the clear
        # frame no-ops threadless and the status line sticks until Slack's own
        # timeout. Shared helper with send_typing so the two guards cannot drift.
        md = self._with_status_thread_anchor(chat_id, metadata)
        try:
            await self._transport.send_outbound(
                {
                    "op": "typing",
                    "chat_id": chat_id,
                    "content": "",
                    "metadata": self._with_scope(chat_id, md),
                },
                platform=platform,
            )
        except Exception:  # noqa: BLE001 - status clear is cosmetic, never breaks a turn
            logger.debug("relay stop_typing failed for %s", chat_id, exc_info=True)

    async def get_chat_info(self, chat_id: str) -> Dict[str, Any]:
        # Proxied to the connector (it owns the platform connection / cache).
        # Gated on op-level capability discovery: a connector that doesn't
        # advertise get_chat_info in supported_ops (including every legacy
        # connector, where supported_ops is empty and the LEGACY_OPS set
        # applies) would only return "unsupported op", so skip the round trip
        # and answer with the same local fallback the error path produced.
        if self._transport is None or not self.descriptor.supports_op("get_chat_info"):
            return {"name": chat_id, "type": "dm"}
        return await self._transport.get_chat_info(chat_id)

    async def send_follow_up(
        self,
        session_key: str,
        kind: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SendResult:
        """Send via a shared-identity capability bound to a session (A2 outbound).

        The gateway never holds the credential: it names the session it is
        already in plus the capability ``kind``, and the connector resolves the
        real value from its vault and egresses (enforcing the tenant match). Used
        e.g. to post a Discord interaction follow-up as the shared bot without
        the token ever reaching the gateway. See RelayTransport.send_follow_up.
        """
        if self._transport is None:
            return SendResult(success=False, error="no transport")
        # Phase 1.5: the capability `kind` is platform-prefixed (e.g.
        # "discord.interaction_token"), so derive the egress platform from it when
        # it names one we front — that tags the OutboundFrame so a multi-platform
        # gateway routes the follow-up through the right sender. Falls back to the
        # session default (connector-side) when the prefix isn't a fronted platform.
        follow_up_platform = None
        if kind and "." in kind:
            prefix = kind.split(".", 1)[0]
            if self._platform_is_fronted(prefix):
                follow_up_platform = prefix
        result = await self._transport.send_follow_up(
            {
                "op": "follow_up",
                "session_key": session_key,
                "kind": kind,
                "content": content,
                "metadata": metadata or {},
            },
            platform=follow_up_platform,
        )
        return SendResult(
            success=bool(result.get("success")),
            message_id=result.get("message_id"),
            error=result.get("error"),
        )

    # ── Phase 2 media ─────────────────────────────────────────────────────

    def _get_media_client(self) -> Optional[RelayMediaClient]:
        """Lazily build the authenticated /relay/media client.

        Uses the SAME connector base URL the WS dials and the SAME per-gateway
        (id, secret) the upgrade authenticates with — no new configuration.
        None when either is unavailable (unenrolled/dev), and every media lane
        then degrades to its pre-media fallback.
        """
        if self._media_client is not None:
            return self._media_client
        try:
            from gateway.relay import relay_connection_auth, relay_url
            from gateway.relay.media import media_base_url

            url = relay_url()
            gateway_id, secret = relay_connection_auth()
            if not url:
                return None
            client = RelayMediaClient(media_base_url(url), gateway_id, secret)
            if not client.enabled:
                return None
            self._media_client = client
            return client
        except Exception:  # noqa: BLE001 - media plumbing must never break the adapter
            logger.debug("relay media client init failed", exc_info=True)
            return None

    async def _send_media(
        self,
        chat_id: str,
        *,
        media_kind: str,
        source: str,
        source_is_path: bool,
        caption: Optional[str] = None,
        filename: Optional[str] = None,
        reply_to: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[SendResult]:
        """Egress one media object via the connector's ``send_media`` op.

        ``source`` is either a LOCAL file path (uploaded to the connector's
        /relay/media first — the connector cannot reach our filesystem) or an
        already-public URL (fal.media output etc. — passed through; the
        connector downloads it directly).

        Returns None when the lane is unavailable (op not advertised, no
        transport, upload failed) so each caller can fall back to its
        pre-media behaviour — media delivery is progressive enhancement,
        never a regression when the connector predates the op.
        """
        if self._transport is None or not self.descriptor.supports_op("send_media"):
            return None
        source_url = source
        if source_is_path:
            client = self._get_media_client()
            if client is None:
                return None
            uploaded = await client.upload(source, filename=filename)
            if not uploaded:
                return None
            source_url = uploaded
        # Same Slack thread-anchor contract as the text lane (send). Media
        # frames egress through the connector's Slack sender too, so an
        # unresolved anchor threads an image under the user's DM message in
        # flat mode, and loses the per-message thread entirely in thread mode
        # (threadTs() reads metadata only). Route through the shared helper.
        media_metadata: Dict[str, Any] = dict(metadata or {})
        effective_reply_to = self._apply_slack_thread_anchor(
            chat_id, reply_to, media_metadata
        )
        action: Dict[str, Any] = {
            "op": "send_media",
            "chat_id": chat_id,
            "media_kind": media_kind,
            "source_url": source_url,
            "content": caption or "",
            "reply_to": effective_reply_to,
            "metadata": self._with_scope(chat_id, media_metadata),
        }
        if filename:
            action["filename"] = filename
        try:
            result = await self._transport.send_outbound(
                action,
                platform=self._platform_by_chat.get(str(chat_id)),
            )
        except Exception:  # noqa: BLE001 - transport failure degrades to the caller's fallback
            logger.debug("relay send_media transport failure", exc_info=True)
            return None
        if not result.get("success"):
            # A structured connector decline (size cap, platform rejection).
            # Surface it as a failed lane so the caller's fallback still
            # delivers SOMETHING (the caption/notice), mirroring native
            # adapters' upload-failure paths.
            logger.warning(
                "relay send_media declined for %s: %s",
                chat_id,
                result.get("error"),
            )
            return None
        return SendResult(
            success=True,
            message_id=result.get("message_id"),
            raw_response=result,
        )

    async def send_image(
        self,
        chat_id: str,
        image_url: str,
        caption: Optional[str] = None,
        reply_to: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SendResult:
        """Send an image (public URL) as a native attachment via the connector."""
        result = await self._send_media(
            chat_id,
            media_kind="image",
            source=image_url,
            source_is_path=False,
            caption=caption,
            reply_to=reply_to,
            metadata=metadata,
        )
        if result is not None:
            return result
        return await super().send_image(
            chat_id, image_url, caption=caption, reply_to=reply_to, metadata=metadata
        )

    async def send_image_file(
        self,
        chat_id: str,
        image_path: str,
        caption: Optional[str] = None,
        reply_to: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> SendResult:
        """Send a local image file natively (upload → send_media)."""
        result = await self._send_media(
            chat_id,
            media_kind="image",
            source=image_path,
            source_is_path=True,
            caption=caption,
            reply_to=reply_to,
            metadata=metadata,
        )
        if result is not None:
            return result
        return await super().send_image_file(
            chat_id,
            image_path,
            caption=caption,
            reply_to=reply_to,
            metadata=metadata,
            **kwargs,
        )

    async def send_voice(
        self,
        chat_id: str,
        audio_path: str,
        caption: Optional[str] = None,
        reply_to: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> SendResult:
        """Send a local audio file as a native voice message (upload → send_media)."""
        result = await self._send_media(
            chat_id,
            media_kind="voice",
            source=audio_path,
            source_is_path=True,
            caption=caption,
            reply_to=reply_to,
            metadata=metadata,
        )
        if result is not None:
            return result
        return await super().send_voice(
            chat_id,
            audio_path,
            caption=caption,
            reply_to=reply_to,
            metadata=metadata,
            **kwargs,
        )

    async def send_video(
        self,
        chat_id: str,
        video_path: str,
        caption: Optional[str] = None,
        reply_to: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> SendResult:
        """Send a local video file natively (upload → send_media)."""
        result = await self._send_media(
            chat_id,
            media_kind="video",
            source=video_path,
            source_is_path=True,
            caption=caption,
            reply_to=reply_to,
            metadata=metadata,
        )
        if result is not None:
            return result
        return await super().send_video(
            chat_id,
            video_path,
            caption=caption,
            reply_to=reply_to,
            metadata=metadata,
            **kwargs,
        )

    async def send_document(
        self,
        chat_id: str,
        file_path: str,
        caption: Optional[str] = None,
        file_name: Optional[str] = None,
        reply_to: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> SendResult:
        """Send a local file as a downloadable attachment (upload → send_media)."""
        result = await self._send_media(
            chat_id,
            media_kind="document",
            source=file_path,
            source_is_path=True,
            caption=caption,
            filename=file_name,
            reply_to=reply_to,
            metadata=metadata,
        )
        if result is not None:
            return result
        return await super().send_document(
            chat_id,
            file_path,
            caption=caption,
            file_name=file_name,
            reply_to=reply_to,
            metadata=metadata,
            **kwargs,
        )

    # ── Phase 3 interactive: prompt + react ──────────────────────────────

    def _mint_prompt(
        self, kind: str, state: Dict[str, Any], timeout_s: float = 3600.0
    ) -> str:
        """Register a pending prompt and return its id.

        ``state`` carries what the resolver needs when the answer comes back
        (session_key, resolver kind, per-kind extras). Expiry is enforced
        gateway-side on consumption (_pop_prompt) — the wire's timeout_s is
        advisory only.

        The id is ``<owner nonce>.<8 hex>``: the nonce marks the minting
        process so a sibling gateway that receives the same fanned-out answer
        can tell it is not the owner and stay quiet (see
        ``_prompt_owner_nonce``). Both segments use the callback alphabet the
        connector's prompt codec accepts ([A-Za-z0-9_.-], <=32 chars).
        """
        import time

        prompt_id = f"{self._prompt_owner_nonce}.{secrets.token_hex(4)}"
        self._pending_prompts[prompt_id] = {
            **state,
            "kind": kind,
            "expires_at": time.time() + timeout_s,
        }
        # Opportunistic sweep so abandoned prompts can't accumulate: drop
        # anything already expired (cheap — dict is small by construction).
        now = time.time()
        for stale in [
            k for k, v in self._pending_prompts.items() if v.get("expires_at", 0) < now
        ]:
            self._pending_prompts.pop(stale, None)
        return prompt_id

    def _minted_here(self, prompt_id: str) -> bool:
        """True when this process minted ``prompt_id``.

        Ids minted before the owner nonce existed (a prompt still pending
        across an in-place upgrade) carry no ``.`` segment; treat them as ours
        so an in-flight prompt from the previous build still resolves.
        """
        head, sep, _ = str(prompt_id).partition(".")
        return head == self._prompt_owner_nonce if sep else True

    def _pop_prompt(self, prompt_id: str) -> Optional[Dict[str, Any]]:
        """Consume a pending prompt: one answer wins, expired entries miss."""
        import time

        state = self._pending_prompts.pop(str(prompt_id), None)
        if not state:
            return None
        if state.get("expires_at", 0) < time.time():
            return None
        return state

    def _note_prompt_resolved(self, prompt_id: str) -> None:
        """Remember that this process already answered ``prompt_id``.

        Bounded FIFO: a repeat answer is only interesting for as long as a
        redelivery or a double tap can plausibly arrive, so old ids are
        dropped rather than retained for the process lifetime.
        """
        self._resolved_prompts[str(prompt_id)] = time.time()
        while len(self._resolved_prompts) > _RESOLVED_PROMPT_MEMORY:
            self._resolved_prompts.popitem(last=False)

    async def _send_prompt(
        self,
        chat_id: str,
        *,
        prompt_kind: str,
        text: str,
        prompt_id: str,
        options: list,
        reply_to: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        timeout_s: Optional[int] = None,
    ) -> Optional[SendResult]:
        """Egress one `prompt` op; None when the lane is unavailable.

        Mirrors _send_media's progressive-enhancement posture: op-gated on the
        descriptor advertising `prompt`, and every failure returns None so the
        caller falls back to its numbered-text base behaviour (which is the
        pre-Phase-3 UX, still fully functional via typed replies).
        """
        if self._transport is None or not self.descriptor.supports_op("prompt"):
            return None
        # An interactive prompt (approval / clarify / slash-confirm) is emitted
        # mid-turn in reply to the triggering inbound event, so `metadata` carries
        # that event's thread context (run.py _thread_metadata_for_source stamps
        # metadata.thread_id — for a Slack DM the triggering message's own ts,
        # used only as a session-keying fallback). Forwarding it makes the
        # connector thread the prompt card UNDER the triggering message instead
        # of posting it flat at the DM root (the reported bug). Native Slack
        # Hermes suppresses this synthetic DM thread anchor; drop it here for the
        # same Slack-DM-with-no-real-thread case, matching _resolve_reply_to_for_send.
        # Prompt metadata is forwarded VERBATIM. The threading mode is decided
        # in exactly one place — run.py's _resolve_progress_thread_id (flat mode
        # suppresses the synthetic self-anchor there; thread mode stamps the
        # turn's thread). Boundary pinned by test_run_py_suppresses_self_anchor*.
        prompt_metadata = metadata
        action: Dict[str, Any] = {
            "op": "prompt",
            "chat_id": chat_id,
            "content": text,
            "prompt_kind": prompt_kind,
            "prompt_id": prompt_id,
            "options": options,
            "reply_to": self._resolve_reply_to_for_send(
                chat_id, reply_to, prompt_metadata
            ),
            "metadata": self._with_scope(chat_id, prompt_metadata),
        }
        if timeout_s is not None:
            action["timeout_s"] = int(timeout_s)
        try:
            result = await self._transport.send_outbound(
                action,
                platform=self._platform_by_chat.get(str(chat_id)),
            )
        except Exception:  # noqa: BLE001 - transport failure degrades to fallback
            logger.debug("relay prompt transport failure", exc_info=True)
            return None
        if not result.get("success"):
            logger.warning(
                "relay prompt declined for %s: %s", chat_id, result.get("error")
            )
            return None
        return SendResult(
            success=True,
            message_id=result.get("message_id"),
            raw_response=result,
        )

    async def send_exec_approval(
        self,
        chat_id: str,
        command: str,
        session_key: str,
        description: str = "dangerous command",
        metadata: Optional[Dict[str, Any]] = None,
        allow_permanent: bool = True,
        allow_session: bool = True,
        smart_denied: bool = False,
    ) -> SendResult:
        """Native-button exec approval over the relay (Phase 3).

        Renders the same choice set as the native adapters (Allow Once /
        Session / Always / Deny, gated by the same flags) through the
        connector's `prompt` op. The user's press comes back as a
        prompt_response and resolves via tools.approval.resolve_gateway_approval
        — the exact mechanism the native button handlers use. When the lane is
        unavailable the send FAILS (success=False) so gateway/run.py's existing
        button→text fallback takes over (same contract as a native adapter's
        failed button send).
        """
        options: list = [{"id": "once", "label": "Allow Once", "style": "primary"}]
        if not smart_denied and allow_session:
            options.append({"id": "session", "label": "Allow Session"})
            if allow_permanent:
                options.append({
                    "id": "always",
                    "label": "Always Allow",
                })
        options.append({"id": "deny", "label": "Deny", "style": "danger"})

        cmd_preview = command if len(command) <= 1500 else command[:1500] + "..."
        text = (
            "⚠️ **Command Approval Required**\n\n"
            f"```\n{cmd_preview}\n```\n"
            f"Reason: {description}"
        )
        if smart_denied:
            text += (
                "\n\n**Smart DENY:** owner override applies to this one operation only."
            )

        prompt_id = self._mint_prompt(
            "exec_approval",
            {"session_key": session_key, "chat_id": str(chat_id)},
        )
        result = await self._send_prompt(
            chat_id,
            prompt_kind="approval",
            text=text,
            prompt_id=prompt_id,
            options=options,
            metadata=metadata,
        )
        if result is not None:
            return result
        # Lane unavailable: unregister and let run.py's text fallback run.
        self._pending_prompts.pop(prompt_id, None)
        return SendResult(success=False, error="relay prompt op unavailable")

    async def send_slash_confirm(
        self,
        chat_id: str,
        title: str,
        message: str,
        session_key: str,
        confirm_id: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SendResult:
        """Three-button slash-command confirmation over the relay (Phase 3).

        Resolves via tools.slash_confirm.resolve() — the same primitive the
        native button handlers call. Falls back (success=False) to the
        gateway's text-intercept flow when the prompt lane is unavailable.
        """
        options = [
            {"id": "once", "label": "Approve Once", "style": "primary"},
            {"id": "always", "label": "Always Approve"},
            {"id": "cancel", "label": "Cancel", "style": "danger"},
        ]
        text = f"**{title}**\n\n{message}" if title else message
        prompt_id = self._mint_prompt(
            "slash_confirm",
            {
                "session_key": session_key,
                "confirm_id": confirm_id,
                "chat_id": str(chat_id),
            },
        )
        result = await self._send_prompt(
            chat_id,
            prompt_kind="approval",
            text=text,
            prompt_id=prompt_id,
            options=options,
            metadata=metadata,
        )
        if result is not None:
            return result
        self._pending_prompts.pop(prompt_id, None)
        return SendResult(success=False, error="relay prompt op unavailable")

    async def send_clarify(
        self,
        chat_id: str,
        question: str,
        choices: Optional[list],
        clarify_id: str,
        session_key: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SendResult:
        """Native-button clarify over the relay (Phase 3).

        Multiple-choice clarifies render as a `prompt` op (one button per
        choice + Other). A press resolves via
        tools.clarify_gateway.resolve_gateway_clarify with the CHOICE TEXT
        (never the option id); "Other" flips the clarify to text-capture
        exactly like the native adapters. Open-ended clarifies and
        unavailable lanes fall back to the base numbered-text behaviour.

        Option ids are positional (c0..cN / other) — choice text can be
        arbitrary UTF-8 and would blow the 64-byte callback budget; the
        registry maps ids back to the real strings on the answer.
        """
        if choices and self.descriptor.supports_op("prompt"):
            options = [
                {"id": f"c{i}", "label": str(choice)[:75]}
                for i, choice in enumerate(choices)
            ]
            options.append({"id": "other", "label": "✏️ Other (type your answer)"})
            prompt_id = self._mint_prompt(
                "clarify",
                {
                    "session_key": session_key,
                    "clarify_id": clarify_id,
                    "choices": [str(c) for c in choices],
                    "chat_id": str(chat_id),
                },
            )
            result = await self._send_prompt(
                chat_id,
                prompt_kind="clarify",
                text=f"❓ {question}",
                prompt_id=prompt_id,
                options=options,
                metadata=metadata,
            )
            if result is not None:
                return result
            self._pending_prompts.pop(prompt_id, None)
        return await super().send_clarify(
            chat_id, question, choices, clarify_id, session_key, metadata=metadata
        )

    async def _consume_prompt_response(self, event) -> bool:
        """Route an inbound prompt_response to its waiting primitive.

        Returns True when the event was a prompt answer (consumed — do NOT
        dispatch it as a chat message), False otherwise.

        A prompt answer is ALWAYS consumed, whoever it belongs to. The three
        non-resolving cases each stay silent rather than falling through to
        chat dispatch:

        * **A sibling's prompt** (``_minted_here`` false). The connector fans a
          passthrough forward out to every live session of the tenant, so one
          button press reaches every gateway of that tenant while only the
          minting one can resolve it. Falling through here is what produced a
          wall of ``Unknown command `/c1``` — one per sibling — under the
          single ``✅`` from the real owner.
        * **A repeat answer** for an id this process already resolved (a double
          tap, or a redelivered forward). The first answer won; a second must
          not re-run the resolver or re-ack.
        * **Our own expired/unknown prompt.** Answer with a short expiry notice
          instead of a command-not-found error: the option ids a prompt renders
          ("c1", "once", "other") are not commands, so the text lane cannot do
          anything useful with them.
        """
        pr = getattr(event, "prompt_response", None)
        if not isinstance(pr, dict):
            return False
        prompt_id = str(pr.get("prompt_id") or "")
        option_id = str(pr.get("option_id") or "")
        if not prompt_id or not option_id:
            return False
        if not self._minted_here(prompt_id):
            # A sibling gateway of this tenant owns this prompt and is
            # resolving it right now. Consume silently — two gateways must
            # never both answer one press.
            logger.debug(
                "relay prompt_response %s (option=%s) belongs to another "
                "gateway instance — ignoring",
                prompt_id,
                option_id,
            )
            return True
        if prompt_id in self._resolved_prompts:
            logger.debug(
                "relay prompt_response %s (option=%s) already resolved — ignoring "
                "repeat",
                prompt_id,
                option_id,
            )
            return True
        state = self._pop_prompt(prompt_id)
        if state is None:
            logger.info(
                "relay prompt_response for unknown/expired prompt %s (option=%s)",
                prompt_id,
                option_id,
            )
            await self._notify_prompt_expired(event)
            return True
        self._note_prompt_resolved(prompt_id)

        kind = state.get("kind")
        chat_id = str(state.get("chat_id") or getattr(event.source, "chat_id", ""))
        session_key = str(state.get("session_key") or "")
        try:
            if kind == "exec_approval":
                from tools.approval import resolve_gateway_approval

                choice = (
                    option_id
                    if option_id in {"once", "session", "always", "deny"}
                    else "deny"
                )
                count = resolve_gateway_approval(session_key, choice)
                label = {
                    "once": "✅ Approved once",
                    "session": "✅ Approved for session",
                    "always": "✅ Approved permanently",
                    "deny": "❌ Denied",
                }.get(choice, "Resolved")
                if not count:
                    label = "⌛ Approval expired — no command was waiting."
                # Acknowledge in-channel (the connector's prompt message can't
                # be edited cross-platform yet — edit support varies; a short
                # confirmation preserves the audit trail the native edit gives).
                await self.send(
                    chat_id, label, metadata=self._prompt_reply_metadata(event)
                )
                if count:
                    self.resume_typing_for_chat(chat_id)
            elif kind == "slash_confirm":
                from tools import slash_confirm as slash_confirm_mod

                choice = (
                    option_id if option_id in {"once", "always", "cancel"} else "cancel"
                )
                result_text = await slash_confirm_mod.resolve(
                    session_key, str(state.get("confirm_id") or ""), choice
                )
                label = {
                    "once": "✅ Approved once",
                    "always": "🔒 Always approve",
                    "cancel": "❌ Cancelled",
                }.get(choice, "Resolved")
                await self.send(
                    chat_id, label, metadata=self._prompt_reply_metadata(event)
                )
                if result_text:
                    await self.send(
                        chat_id,
                        str(result_text),
                        metadata=self._prompt_reply_metadata(event),
                    )
            elif kind == "clarify":
                from tools.clarify_gateway import (
                    mark_awaiting_text,
                    resolve_gateway_clarify,
                )

                clarify_id = str(state.get("clarify_id") or "")
                if option_id == "other":
                    mark_awaiting_text(clarify_id)
                    await self.send(
                        chat_id,
                        "✏️ Type your answer:",
                        metadata=self._prompt_reply_metadata(event),
                    )
                else:
                    choices = state.get("choices") or []
                    try:
                        idx = int(option_id[1:]) if option_id.startswith("c") else -1
                    except ValueError:
                        idx = -1
                    if 0 <= idx < len(choices):
                        resolve_gateway_clarify(clarify_id, str(choices[idx]))
                        await self.send(
                            chat_id,
                            f"✅ {choices[idx]}",
                            metadata=self._prompt_reply_metadata(event),
                        )
                    else:
                        # Unmappable option: flip to text capture so the user
                        # can answer by typing (never dead-end a clarify).
                        mark_awaiting_text(clarify_id)
            else:
                logger.warning("relay prompt_response with unknown kind %r", kind)
        except Exception:  # noqa: BLE001 - a resolver failure must not kill the reader
            logger.warning("relay prompt_response resolution failed", exc_info=True)
        return True

    async def _notify_prompt_expired(self, event) -> None:
        """Tell the presser their prompt is no longer waiting.

        Only the OWNING gateway reaches this (siblings return earlier), so the
        user sees exactly one notice. Best-effort: a send failure here must
        not break the reader.
        """
        chat_id = str(getattr(event.source, "chat_id", "") or "")
        if not chat_id:
            return
        try:
            await self.send(
                chat_id,
                "⌛ That prompt is no longer waiting for an answer. "
                "Send your reply as a normal message.",
                metadata=self._prompt_reply_metadata(event),
            )
        except Exception:  # noqa: BLE001 - notification is best-effort
            logger.debug("relay expired-prompt notice failed", exc_info=True)

    def _prompt_reply_metadata(self, event) -> Dict[str, Any]:
        """Thread/topic metadata so prompt acks land where the prompt lives."""
        meta: Dict[str, Any] = {}
        thread_id = getattr(event.source, "thread_id", None)
        if thread_id:
            meta["thread_id"] = str(thread_id)
        return meta

    # ── Phase 3 ack lifecycle (👀 → ✅/❌) ────────────────────────────────

    async def _react(
        self,
        chat_id: str,
        message_id: str,
        emoji: str,
        *,
        remove: bool = False,
    ) -> bool:
        """Egress one `react` op; best-effort (False on any failure).

        Reactions are cosmetic by contract: a failure is logged at debug and
        never surfaces to the caller's flow (mirrors the native Discord
        adapter's _add_reaction posture).
        """
        if self._transport is None or not self.descriptor.supports_op("react"):
            return False
        if not chat_id or not message_id:
            return False
        try:
            result = await self._transport.send_outbound(
                {
                    "op": "react",
                    "chat_id": chat_id,
                    "message_id": message_id,
                    "emoji": emoji,
                    "remove": remove,
                    "metadata": self._with_scope(chat_id, None),
                },
                platform=self._platform_by_chat.get(str(chat_id)),
            )
            return bool(result.get("success"))
        except Exception:  # noqa: BLE001 - reactions are cosmetic
            logger.debug("relay react failed", exc_info=True)
            return False

    async def on_processing_start(self, event) -> None:
        """Add the 👀 in-progress reaction (op-gated; silent no-op otherwise)."""
        message_id = getattr(event, "message_id", None) or getattr(
            event.source, "message_id", None
        )
        chat_id = getattr(event.source, "chat_id", None)
        if message_id and chat_id:
            await self._react(str(chat_id), str(message_id), "👀")

    async def on_processing_complete(self, event, outcome) -> None:
        """Swap 👀 for ✅/❌ per outcome (op-gated; silent no-op otherwise)."""
        from gateway.platforms.base import ProcessingOutcome

        message_id = getattr(event, "message_id", None) or getattr(
            event.source, "message_id", None
        )
        chat_id = getattr(event.source, "chat_id", None)
        if not (message_id and chat_id):
            return
        await self._react(str(chat_id), str(message_id), "👀", remove=True)
        if outcome == ProcessingOutcome.SUCCESS:
            await self._react(str(chat_id), str(message_id), "✅")
        elif outcome == ProcessingOutcome.FAILURE:
            await self._react(str(chat_id), str(message_id), "❌")

    # ── Phase 4 thread lifecycle ──────────────────────────────────────────

    async def create_handoff_thread(
        self,
        parent_chat_id: str,
        name: str,
    ) -> Optional[str]:
        """Create a thread/topic under ``parent_chat_id`` via the connector.

        One `thread_create` op covers Discord (channel thread), Telegram
        (forum topic), and Slack (named seed root message — threads there are
        message-anchored). Op-gated on the descriptor advertising
        `thread_create`; None on any failure/unavailability so the handoff
        watcher falls back to the parent channel — the same contract as the
        native adapters' create_handoff_thread.
        """
        if self._transport is None or not self.descriptor.supports_op("thread_create"):
            return None
        thread_name = (str(name or "").strip() or "handoff")[:100]
        try:
            result = await self._transport.send_outbound(
                {
                    "op": "thread_create",
                    "chat_id": str(parent_chat_id),
                    "thread_name": thread_name,
                    "metadata": self._with_scope(str(parent_chat_id), None),
                },
                platform=self._platform_by_chat.get(str(parent_chat_id)),
            )
        except Exception:  # noqa: BLE001 - handoff falls back to the parent channel
            logger.debug("relay thread_create transport failure", exc_info=True)
            return None
        if not result.get("success"):
            logger.info(
                "relay thread_create declined for %s: %s",
                parent_chat_id,
                result.get("error"),
            )
            return None
        thread_id = result.get("thread_id") or result.get("message_id")
        return str(thread_id) if thread_id else None

    async def rename_thread(
        self,
        thread_id: str,
        name: str,
        *,
        only_if_current_name: Optional[str] = None,
        prefer_connector_created: bool = False,
        parent_chat_id: Optional[str] = None,
    ) -> bool:
        """Best-effort thread rename via the connector's `thread_rename` op.

        The relay sibling of the native Discord adapter's rename_thread —
        called by the SAME semantic-rename lane (run.py
        _rename_discord_auto_thread_for_session_title), which fires only for
        sources carrying the connector-stamped auto-thread markers.

        No-clobber guard: prefer ``prefer_connector_created=True``, which asks
        the CONNECTOR to enforce the guard from ITS OWN created-name memory
        (only_if_connector_created) — the gateway no longer has to reproduce
        the thread's initial name byte-for-byte, which drifted on any
        normalization difference and silently declined every relay rename.
        ``only_if_current_name`` is the legacy string guard, kept for the
        native-marker lane and older connectors. ``parent_chat_id`` is the
        containing chat where the caller knows it (Telegram needs it);
        defaults to the thread id itself (Discord ignores chat_id).
        """
        if self._transport is None or not self.descriptor.supports_op("thread_rename"):
            return False
        cleaned = " ".join(str(name or "").split()).strip()
        if not cleaned or not thread_id:
            return False
        chat_id = str(parent_chat_id or thread_id)
        action: Dict[str, Any] = {
            "op": "thread_rename",
            "chat_id": chat_id,
            "message_id": str(thread_id),
            "thread_name": cleaned[:100],
            "metadata": self._with_scope(chat_id, None),
        }
        if prefer_connector_created:
            action["only_if_connector_created"] = True
        elif only_if_current_name is not None:
            action["only_if_current_name"] = str(only_if_current_name)
        try:
            result = await self._transport.send_outbound(
                action,
                platform=self._platform_by_chat.get(chat_id)
                or self._platform_by_chat.get(str(thread_id)),
            )
        except Exception:  # noqa: BLE001 - renames are cosmetic
            logger.debug("relay thread_rename transport failure", exc_info=True)
            return False
        if not result.get("success"):
            logger.info(
                "relay thread_rename declined for %s: %s",
                thread_id,
                result.get("error"),
            )
            return False
        return True
