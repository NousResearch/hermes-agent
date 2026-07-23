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
from typing import Any, Callable, Dict, Optional

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import BasePlatformAdapter, MessageEvent, SendResult
from gateway.relay.descriptor import CapabilityDescriptor
from gateway.relay.transport import RelayTransport
from gateway.session import SessionSource

logger = logging.getLogger(__name__)


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
        self._capture_scope(event)
        await self.handle_message(event)

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
        except Exception:  # noqa: BLE001 - scope tracking must never break inbound
            pass

    def _with_scope(self, chat_id: str, metadata: Optional[Dict[str, Any]]) -> Dict[str, Any]:
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

        NOTE (open semantic sub-design, flagged for review): the interaction ->
        MessageEvent mapping below is the v1 default. The exact agent UX for a
        slash-command / button interaction (vs. a plain message) — command name
        surfacing, option rendering, deferred-vs-immediate response — is the open
        piece tracked in the spec; the TRANSPORT + receive mechanism (this whole
        path) is settled.
        """
        try:
            platform = getattr(forward, "platform", "") or ""
            if platform == "discord":
                event = self._discord_interaction_to_event(forward)
                if event is not None:
                    self._capture_scope(event)
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
        if itype == 2:
            text = str(data.get("name") or "")
        elif itype == 3:
            text = str(data.get("custom_id") or "")
        else:
            text = ""
        member = payload.get("member") or {}
        user = (member.get("user") if isinstance(member, dict) else None) or payload.get("user") or {}
        channel_id = str(payload.get("channel_id") or "")
        guild_id = payload.get("guild_id")  # real Discord interaction wire field
        source = SessionSource(
            platform=Platform.RELAY,
            chat_id=channel_id,
            chat_type="channel" if guild_id else "dm",
            user_id=str(user.get("id")) if isinstance(user, dict) and user.get("id") else None,
            user_name=str(user.get("username")) if isinstance(user, dict) and user.get("username") else None,
            scope_id=str(guild_id) if guild_id else None,  # Discord guild → generic scope slot
            message_id=str(payload.get("id")) if payload.get("id") else None,
        )
        return MessageEvent(text=text, message_type=MessageType.TEXT, source=source)

    async def disconnect(self) -> None:
        # Phase 7 Unit 7d-B: stop the revocation monitor first so it can't fire a
        # spurious fatal during/after a deliberate teardown.
        if self._revocation_monitor is not None:
            self._revocation_monitor.cancel()
            try:
                await self._revocation_monitor
            except (asyncio.CancelledError, Exception):  # noqa: BLE001 - best-effort teardown
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
            go_idle = getattr(self._transport, "go_idle", None)
            if callable(go_idle):
                try:
                    result: Any = go_idle()
                    if asyncio.iscoroutine(result):
                        await result
                except Exception:  # noqa: BLE001 - going-idle is an optimization, never blocks drain
                    logger.debug("relay going_idle failed during drain", exc_info=True)
            await self._transport.disconnect()

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
        result = await self._transport.send_outbound(
            {
                "op": "send",
                "chat_id": chat_id,
                "content": content,
                "reply_to": reply_to,
                "metadata": self._with_scope(chat_id, send_metadata),
            },
            platform=self._platform_by_chat.get(str(chat_id)),
        )
        return SendResult(
            success=bool(result.get("success")),
            message_id=result.get("message_id"),
            error=result.get("error"),
        )

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
        try:
            await self._transport.send_outbound(
                {
                    "op": "typing",
                    "chat_id": chat_id,
                    "metadata": self._with_scope(chat_id, metadata),
                },
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
        try:
            await self._transport.send_outbound(
                {
                    "op": "typing",
                    "chat_id": chat_id,
                    "content": "",
                    "metadata": self._with_scope(chat_id, metadata),
                },
                platform=platform,
            )
        except Exception:  # noqa: BLE001 - status clear is cosmetic, never breaks a turn
            logger.debug("relay stop_typing failed for %s", chat_id, exc_info=True)

    async def get_chat_info(self, chat_id: str) -> Dict[str, Any]:
        # Proxied to the connector (it owns the platform connection / cache).
        if self._transport is None:
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
