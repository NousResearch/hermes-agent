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
        # chat_id -> guild_id (Discord) / workspace scope, learned from inbound
        # events. The connector's egress guard resolves the owning tenant from
        # the OUTBOUND action's metadata.guild_id; the gateway's generic delivery
        # path (run.py _thread_metadata_for_source) only carries thread_id, so we
        # re-attach the scope here from what we saw inbound. Keyed by chat_id
        # (channel) since that's what send() receives. See routedEgressGuard.ts.
        self._scope_by_chat: Dict[str, str] = {}
        self.supports_code_blocks = descriptor.markdown_dialect not in ("", "plain")
        # Phase 7 Unit 7d-B: watches the transport for a terminal auth revocation
        # (a 4401 close after a successful handshake = the operator opted this
        # instance out of the relay). On revocation we surface a clean,
        # non-retryable "relay disabled" fatal so the dashboard stops showing a
        # red "retrying" spin against a dead credential.
        self._revocation_monitor: Optional[asyncio.Task[None]] = None

    # ── capability surface (from descriptor) ─────────────────────────────
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
    async def connect(self) -> bool:
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
        """Remember chat_id -> guild scope from an inbound event so our outbound
        (the agent's reply) can re-assert it for the connector's egress tenant
        resolution. Never raises — scope tracking must not break inbound."""
        try:
            src = getattr(event, "source", None)
            scope = getattr(src, "guild_id", None) if src else None
            chat = getattr(src, "chat_id", None) if src else None
            if scope and chat:
                self._scope_by_chat[str(chat)] = str(scope)
        except Exception:  # noqa: BLE001 - scope tracking must never break inbound
            pass

    def _with_scope(self, chat_id: str, metadata: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Ensure the outbound metadata carries guild_id for the connector's
        egress tenant resolution. The connector resolves the owning tenant from
        metadata.guild_id (Discord); without it egress is declined as
        'target not routed to an onboarded tenant'. No-op when we have no scope
        for this chat (e.g. DMs) or it's already present."""
        meta: Dict[str, Any] = dict(metadata or {})
        if not meta.get("guild_id"):
            scope = self._scope_by_chat.get(str(chat_id))
            if scope:
                meta["guild_id"] = scope
        return meta

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
        guild_id = payload.get("guild_id")
        source = SessionSource(
            platform=Platform.RELAY,
            chat_id=channel_id,
            chat_type="channel" if guild_id else "dm",
            user_id=str(user.get("id")) if isinstance(user, dict) and user.get("id") else None,
            user_name=str(user.get("username")) if isinstance(user, dict) and user.get("username") else None,
            guild_id=str(guild_id) if guild_id else None,
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

    async def send(
        self,
        chat_id: str,
        content: str,
        reply_to: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SendResult:
        if self._transport is None:
            return SendResult(success=False, error="no transport")
        result = await self._transport.send_outbound(
            {
                "op": "send",
                "chat_id": chat_id,
                "content": content,
                "reply_to": reply_to,
                "metadata": self._with_scope(chat_id, metadata),
            }
        )
        return SendResult(
            success=bool(result.get("success")),
            message_id=result.get("message_id"),
            error=result.get("error"),
        )

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
        result = await self._transport.send_follow_up(
            {
                "op": "follow_up",
                "session_key": session_key,
                "kind": kind,
                "content": content,
                "metadata": metadata or {},
            }
        )
        return SendResult(
            success=bool(result.get("success")),
            message_id=result.get("message_id"),
            error=result.get("error"),
        )
