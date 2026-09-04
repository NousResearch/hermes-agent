"""Production WebSocket RelayTransport — the gateway's live link to the connector.

The gateway dials OUT to the connector's relay endpoint and speaks the
newline-delimited JSON frame protocol of ``docs/relay-connector-contract.md``:
gateway -> connector: hello, outbound, interrupt, going_idle, inbound_ack;
connector -> gateway: descriptor, inbound, outbound_result, interrupt_inbound,
going_idle_ack, passthrough_forward. Outbound calls block on a per-request future
keyed by ``requestId`` until the matching ``outbound_result``; a background reader
pumps inbound frames to the registered handler. EXPERIMENTAL schema.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
import uuid
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Dict, List, Optional

from gateway.platforms.base import MessageEvent, MessageType
from gateway.session import SessionSource
from gateway.relay.descriptor import CapabilityDescriptor
from gateway.relay.transport import InboundHandler

logger = logging.getLogger(__name__)

try:  # lazy/optional dep — mirrors gateway/platforms/feishu.py
    import websockets
except ImportError:  # pragma: no cover - exercised only when the extra is absent
    websockets = None  # type: ignore[assignment]

WEBSOCKETS_AVAILABLE = websockets is not None

_HANDSHAKE_TIMEOUT_S = 30.0
_OUTBOUND_TIMEOUT_S = 30.0
# Bound on each of the three sequential teardown awaits (supervisor, reader,
# ws.close) so a wedged peer cannot stall adapter.disconnect past the runner's
# default 5s budget.
_TEARDOWN_AWAIT_TIMEOUT_S = 1.0
# Max drain for in-flight outbound frames at disconnect: long enough for a
# platform edit round-trip through the connector, short enough that shutdown
# stays snappy when the connector is gone. Clamped at disconnect time so drain +
# teardown stay inside the runner's adapter-disconnect budget: blowing it cancels
# teardown mid-drain and leaves callers blocked on _OUTBOUND_TIMEOUT_S.
_DISCONNECT_DRAIN_GRACE_S = 5.0
# Private-use close code the connector sends when it rejects/revokes a gateway's
# WS upgrade auth. Received AFTER a successful handshake it means the per-gateway
# secret was revoked (opt-out / deprovision) — terminal, no reconnect.
_RELAY_UNAUTHORIZED_CLOSE_CODE = 4401


def _disconnect_drain_grace_s(budget_s: Optional[float] = None) -> float:
    """Effective drain grace: clamped to the caller's REMAINING disconnect budget
    (None mirrors the runner's env default so the transport imports without the
    runner), reserving the three sequential teardown awaits plus a small margin."""
    budget = _env_disconnect_budget_s() if budget_s is None else max(0.0, budget_s)
    reserved = 3 * _TEARDOWN_AWAIT_TIMEOUT_S + 0.5
    return max(0.0, min(_DISCONNECT_DRAIN_GRACE_S, budget - reserved))


def _env_disconnect_budget_s() -> float:
    """The runner's adapter-disconnect budget (same env var + default as
    gateway/run.py:_adapter_disconnect_timeout_secs), apportioned by callers
    across go_idle / monitor teardown / drain."""
    budget = 5.0
    raw = os.getenv("HERMES_GATEWAY_ADAPTER_DISCONNECT_TIMEOUT", "").strip()
    if raw:
        with contextlib.suppress(ValueError):
            budget = max(0.0, float(raw))
    return budget


def _ws_dial_url(url: str) -> str:
    """Normalize the configured connector BASE URL to the ``ws(s)://…/relay`` dial target.

    ``https -> wss`` / ``http -> ws`` (websockets.connect rejects http schemes) and
    the path must end in ``/relay`` (the connector 400s any other upgrade path).
    Idempotent on an already-normalized URL.
    """
    raw = (url or "").strip()
    if raw.startswith("https://"):
        raw = "wss://" + raw[len("https://"):]
    elif raw.startswith("http://"):
        raw = "ws://" + raw[len("http://"):]
    raw = raw.rstrip("/")
    if not raw.endswith("/relay"):
        raw = f"{raw}/relay"
    return raw


def _render_relay_context(context: Any) -> Optional[str]:
    """Flatten the connector's read-only ``context`` list (oldest→newest) into the
    ``MessageEvent.channel_context`` string history-backfill already uses.

    Reference only — never triggers the agent. None when there is no usable context
    so ``channel_context`` stays unset. Never raises: a malformed payload must not
    break delivery of the already-admitted turn.
    """
    if not context or not isinstance(context, list):
        return None
    lines: List[str] = []
    for item in context:
        if not isinstance(item, dict):
            continue
        text = item.get("text")
        if not text:
            continue
        src = item.get("source") or {}
        author = (src.get("user_name") or src.get("user_id") or "") if isinstance(src, dict) else ""
        lines.append(f"{author}: {text}" if author else str(text))
    if not lines:
        return None
    return "[Recent channel messages]\n" + "\n".join(lines)


def _normalize_slack_parent_command(text: str, message_type: MessageType) -> tuple[str, MessageType]:
    """Mirror native Slack ``/hermes`` routing for authenticated relay text."""
    parent_parts = text.strip().split(maxsplit=1)
    if not parent_parts or parent_parts[0] != "/hermes":
        return text, message_type

    from hermes_cli.commands_platforms import slack_subcommand_map

    payload = parent_parts[1].strip() if len(parent_parts) > 1 else ""
    subcommand_map = slack_subcommand_map()
    subcommand_map["compact"] = "/compress"
    first_word = payload.split()[0] if payload else ""

    if first_word in subcommand_map:
        rest = payload[len(first_word) :].strip()
        normalized = f"{subcommand_map[first_word]} {rest}".strip()
    else:
        normalized = payload or "/help"

    normalized_type = MessageType.COMMAND if normalized.startswith("/") else MessageType.TEXT
    return normalized, normalized_type


def _media_types_from_wire(raw: Dict[str, Any]) -> list[str]:
    """Per-attachment MIME types, parallel to ``media_urls``.

    INVARIANT: always the same length as ``media_urls`` (padded with ``""``), or
    empty when there are no urls — consumers index the two lists pairwise, so a
    short list would shift later entries onto the wrong url. Resolved BY URL LOOKUP
    into ``media[]``, never by position: the two wire fields are independent and may
    disagree in order. A url with no match keeps ``""`` (message-level classification).
    """
    urls = raw.get("media_urls")
    if not isinstance(urls, list) or not urls:
        return []
    media = raw.get("media")
    mime_by_url: dict[str, str] = {
        m["url"]: m.get("mime") or ""
        for m in (media if isinstance(media, list) else [])
        if isinstance(m, dict) and isinstance(m.get("url"), str) and m["url"]
    }
    types = [mime_by_url.get(u, "") if isinstance(u, str) else "" for u in urls]
    missing = sum(1 for t in types if not t)
    if missing and mime_by_url:
        logger.debug(
            "relay inbound: %d/%d media_urls had no matching media[] mime", missing, len(types)
        )
    return types


def _event_from_wire(raw: Dict[str, Any]) -> MessageEvent:
    """Rebuild a MessageEvent from the connector's normalized inbound payload (§3).
    Unknown platforms fall back to RELAY, unknown message types to TEXT."""
    src = raw.get("source", {}) or {}
    from gateway.config import Platform

    try:
        platform_enum = Platform(src.get("platform", "relay"))
    except ValueError:
        platform_enum = Platform.RELAY

    source = SessionSource(
        platform=platform_enum,
        chat_id=src.get("chat_id", ""),
        chat_type=src.get("chat_type", "dm"),
        chat_name=src.get("chat_name"),
        user_id=src.get("user_id"),
        # Native adapters surface the DISPLAY name, so prefer it over the raw
        # username. Session keys derive from user_id, so this is presentation-only.
        user_name=(src.get("user_display_name") or src.get("user_name") or src.get("user_handle")),
        thread_id=src.get("thread_id"),
        chat_topic=src.get("chat_topic"),
        user_id_alt=src.get("user_id_alt"),
        chat_id_alt=src.get("chat_id_alt"),
        scope_id=src.get("scope_id"),
        parent_chat_id=src.get("parent_chat_id"),
        message_id=src.get("message_id"),
        # Multiplex mode: the connector stamps the target Hermes profile; None on
        # a single-profile gateway keeps the legacy ``agent:main`` namespace.
        profile=src.get("profile"),
        # Connector-stamped auto-thread markers light the same semantic-rename
        # lane native Discord uses.
        auto_thread_created=bool(src.get("auto_thread_created", False)),
        auto_thread_initial_name=src.get("auto_thread_initial_name"),
        # Thread id this channel message's reply WILL be auto-threaded into, so the
        # initiating message and its in-thread follow-ups share ONE session.
        prospective_thread_id=src.get("prospective_thread_id"),
        # Stamped here, never read off the wire: this event arrived over the
        # authenticated relay WS. Authz keys upstream trust off THIS flag, not
        # ``platform`` (which is the UNDERLYING platform, not ``relay``).
        delivered_via_upstream_relay=True,
    )
    try:
        msg_type = MessageType(raw.get("message_type", "text"))
    except ValueError:
        msg_type = MessageType.TEXT

    text = raw.get("text", "")
    if platform_enum == Platform.SLACK:
        # Slack slash text arrives over the relay bypassing the native command
        # callback; normalize at the wire boundary so adapter gates see the real
        # gateway command rather than the legacy `hermes` parent name.
        text, msg_type = _normalize_slack_parent_command(text, msg_type)

    reply_to = raw.get("reply_to") or {}
    prompt_response = raw.get("prompt_response")
    return MessageEvent(
        text=text,
        message_type=msg_type,
        source=source,
        message_id=raw.get("message_id"),
        reply_to_message_id=raw.get("reply_to_message_id"),
        reply_to_text=reply_to.get("text"),
        reply_to_author_name=reply_to.get("author"),
        reply_to_is_own_message=bool(reply_to.get("is_own", False)),
        media_urls=raw.get("media_urls") or [],
        # Parallel to media_urls; run.py's per-attachment classifiers consult
        # media_types[i] FIRST (routes a relayed image/document/voice like native).
        media_types=_media_types_from_wire(raw),
        channel_context=_render_relay_context(raw.get("context")),
        # Structured interactive-prompt reply, verbatim off the wire; the adapter
        # consumes it to resolve pending approvals/confirms/clarifies.
        prompt_response=dict(prompt_response) if isinstance(prompt_response, dict) else None,
    )


@dataclass
class PassthroughForward:
    """A connector-forwarded passthrough-plane request (§5.1): the connector answered
    the provider's latency-critical ACK at its edge, then forwarded the sanitized
    request. ``body`` is the exact decoded bytes; ``headers`` preserve arrival order."""

    platform: str
    bot_id: str
    method: str
    path: str
    headers: list[tuple[str, str]]
    body: bytes
    # Multiplex-mode target profile, mirroring the inbound frame's SessionSource;
    # None keeps the legacy ``agent:main`` namespace. Without it a relayed Discord
    # slash-command/button/modal fell back to agent:main even when the equivalent
    # plain message routed to the right profile.
    # Mirrors the ``profile`` field _event_from_wire already carries on the ``inbound`` frame's
    # SessionSource (#60586) — the connector stamps it when NAS resolves the target profile for a
    # Team-Gateway interaction; absent for a single-profile gateway, where it stays None and session keys
    # keep the legacy ``agent:main`` namespace.
    profile: Optional[str] = None


def _passthrough_from_wire(raw: Dict[str, Any]) -> PassthroughForward:
    """Rebuild a PassthroughForward from the wire frame (body base64-decoded). No
    verification here: the connector is the trust boundary and verified at the edge."""
    import base64

    try:
        body = base64.b64decode(raw.get("bodyB64", "") or "")
    except Exception:  # noqa: BLE001 - a malformed body must not crash the reader
        body = b""
    headers = [
        (str(pair[0]), str(pair[1]))
        for pair in (raw.get("headers", []) or [])
        if isinstance(pair, (list, tuple)) and len(pair) == 2
    ]
    return PassthroughForward(
        platform=str(raw.get("platform", "")), bot_id=str(raw.get("botId", "")),
        method=str(raw.get("method", "")), path=str(raw.get("path", "")), headers=headers,
        body=body, profile=raw.get("profile"),
    )


async def _await_bounded(aw: Awaitable[Any]) -> None:
    """Best-effort teardown await: bounded, swallows timeout/cancel/errors."""
    try:
        await asyncio.wait_for(aw, timeout=_TEARDOWN_AWAIT_TIMEOUT_S)
    except (asyncio.TimeoutError, asyncio.CancelledError, Exception):  # noqa: BLE001
        pass


class WebSocketRelayTransport:
    """RelayTransport over a WebSocket connection the gateway dials to the connector."""

    def __init__(
        self,
        url: str,
        platform: str,
        bot_id: str,
        *,
        identities: Optional[list[tuple[str, str]]] = None,
        connect_timeout_s: float = _HANDSHAKE_TIMEOUT_S,
        outbound_timeout_s: float = _OUTBOUND_TIMEOUT_S,
        gateway_id: Optional[str] = None,
        upgrade_secret: Optional[str] = None,
        reconnect: bool = False,
        reconnect_backoff_s: float = 1.0,
        reconnect_max_backoff_s: float = 30.0,
    ) -> None:
        if not WEBSOCKETS_AVAILABLE:
            raise RuntimeError(
                "WebSocketRelayTransport requires the 'websockets' package "
                "(install the messaging extra)."
            )
        self._url = _ws_dial_url(url)
        self._platform = platform
        self._bot_id = bot_id
        # Every (platform, bot_id) this gateway fronts on this one WS: one `hello`
        # per identity; the first is the default an untagged outbound falls back to.
        self._identities = list(identities) if identities else [(platform, bot_id)]
        self._connect_timeout_s = connect_timeout_s
        self._outbound_timeout_s = outbound_timeout_s
        # Upgrade auth: with a per-gateway secret the gateway presents an HMAC
        # bearer keyed by gateway_id; absent -> unauthenticated upgrade.
        self._gateway_id = gateway_id
        self._upgrade_secret = upgrade_secret

        # Reconnect supervisor: re-dial + re-handshake after an UNEXPECTED close
        # (not disconnect()), which makes the connector drain this instance's
        # buffered backlog on the new handshake. Off by default (stub/tests).
        self._reconnect = reconnect
        self._reconnect_backoff_s = reconnect_backoff_s
        self._reconnect_max_backoff_s = reconnect_max_backoff_s
        self._supervisor: Optional[asyncio.Task[None]] = None
        # Dormant close (go_dormant) is distinct from disconnect() (terminal) and
        # an unexpected close (fast re-dial): the socket closes WITHOUT _closing,
        # so the reader still arms the supervisor, but it polls on the dormant
        # cadence so it does not fight the platform's suspend window. Cleared on
        # a successful re-dial. A suspended machine's event loop is frozen, so the
        # timer only advances once awake; it just needs to re-dial promptly then.
        self._dormant = False
        self._dormant_redial_s = 1.0

        self._ws: Any = None
        self._reader: Optional[asyncio.Task[None]] = None
        self._inbound: Optional[InboundHandler] = None
        self._interrupt_inbound_handler: Any = None
        self._passthrough_handler: Any = None
        # `_descriptor` is the FIRST (primary-identity) descriptor; the map holds
        # one per hello'd identity, keyed by platform (descriptor_for_platform).
        self._descriptor: Optional[CapabilityDescriptor] = None
        self._descriptors_by_platform: Dict[str, CapabilityDescriptor] = {}
        self._descriptor_ready: asyncio.Future[CapabilityDescriptor] | None = None
        self._pending: Dict[str, asyncio.Future[Dict[str, Any]]] = {}
        self._going_idle_ack: asyncio.Future[None] | None = None
        self._closing = False
        # A 4401 close AFTER at least one successful handshake means the connector
        # REVOKED this gateway's secret (opt-out): terminal, stop reconnecting.
        # A 4401 BEFORE any handshake is a cold-start race and stays retryable.
        self._handshake_succeeded = False
        self._auth_revoked = False

    # ── lifecycle ────────────────────────────────────────────────────────
    async def connect(self) -> bool:
        await self._dial_and_start()
        return True

    async def _dial_and_start(self) -> None:
        """Open the socket, start the reader, send hello(s). Used by connect() and
        by the reconnect supervisor on a re-dial."""
        self._descriptor_ready = asyncio.get_running_loop().create_future()
        # Fresh handshake generation: a reconnected connector re-sends one
        # descriptor per hello, so stale descriptors must not survive.
        self._descriptor = None
        self._descriptors_by_platform = {}
        # A successful (re-)dial ends any dormant state.
        self._dormant = False
        # WAN-friendly keepalive: the library default (20s pong deadline) produces
        # spurious `1011 keepalive ping timeout` closes under transient latency /
        # event-loop stalls; 60s tolerates them while detecting a dead link ~90s.
        kwargs: Dict[str, Any] = {"ping_interval": 30, "ping_timeout": 60}
        headers = self._upgrade_headers()
        if headers:
            kwargs["additional_headers"] = headers
        self._ws = await websockets.connect(self._url, **kwargs)  # type: ignore[union-attr]
        self._reader = asyncio.create_task(self._read_loop(), name="relay-ws-reader")
        # One hello PER fronted identity; the connector accumulates them (first
        # sets the session default). The FIRST descriptor resolves handshake().
        for platform, bot_id in self._identities:
            hello: Dict[str, Any] = {"type": "hello", "platform": platform, "botId": bot_id}
            # Declare the slash-command set on the Discord hello so the connector
            # (which holds the bot token) reconciles Discord's registration.
            # Enrichment only — never blocks the handshake; a connector predating the
            # field ignores it. Only Discord has an app-command registry.
            if platform == "discord":
                try:
                    from gateway.relay.command_manifest import build_relay_command_manifest

                    hello["command_manifest"] = build_relay_command_manifest()
                except Exception:  # noqa: BLE001
                    logger.debug("relay command manifest build failed", exc_info=True)
            await self._send(hello)

    def _upgrade_headers(self) -> Dict[str, str]:
        """``Authorization: Bearer <signed token>`` for the WS upgrade, or {} when
        no secret is configured (the connector closes 4401 on a bad/missing one)."""
        if not (self._upgrade_secret and self._gateway_id):
            return {}
        from gateway.relay.auth import make_upgrade_token

        return {"Authorization": f"Bearer {make_upgrade_token(self._gateway_id, self._upgrade_secret)}"}

    async def disconnect(self, *, budget_s: Optional[float] = None) -> None:
        """Tear down the socket, draining in-flight outbound frames first.
        ``budget_s`` is the REMAINING wall-clock budget the caller can spend here;
        None applies the env-mirrored runner default."""
        self._closing = True
        try:
            # A trailing outbound frame (typically the turn's finalize edit) may
            # still await its outbound_result; failing it immediately loses a
            # message the connector was about to ack. asyncio.wait (not
            # wait_for+gather): on timeout it must NOT cancel the futures — the
            # finally below owns their terminal state.
            pending = [f for f in self._pending.values() if not f.done()]
            if pending:
                grace = _disconnect_drain_grace_s(budget_s)
                if grace > 0:
                    with contextlib.suppress(Exception):  # grace is best-effort
                        await asyncio.wait(pending, timeout=grace)
            for attr in ("_supervisor", "_reader"):
                task = getattr(self, attr)
                if task is not None:
                    task.cancel()
                    await _await_bounded(task)
                    setattr(self, attr, None)
            if self._ws is not None:
                await _await_bounded(self._ws.close())
                self._ws = None
        finally:
            # Fail in-flight waiters so callers don't hang. In a finally so a
            # cancellation landing anywhere above (runner budget, outer cleanup
            # deadline) can NEVER leave a registered future unresolved for the
            # full _OUTBOUND_TIMEOUT_S. Idempotent: done futures are skipped.
            closed = RuntimeError("relay transport closed")
            self._fail_pending(lambda fut: fut.set_exception(closed))
            if self._going_idle_ack is not None and not self._going_idle_ack.done():
                self._going_idle_ack.set_exception(closed)

    def _fail_pending(self, settle: Callable[[asyncio.Future[Dict[str, Any]]], None]) -> None:
        """Settle every unresolved pending future via ``settle`` and clear the map.
        list() snapshot: settling wakes waiters whose finally-pop mutates the dict."""
        for fut in list(self._pending.values()):
            if not fut.done():
                settle(fut)
        self._pending.clear()

    async def handshake(self) -> CapabilityDescriptor:
        if self._descriptor is not None:
            return self._descriptor
        if self._descriptor_ready is None:
            raise RuntimeError("handshake() called before connect()")
        return await asyncio.wait_for(self._descriptor_ready, timeout=self._connect_timeout_s)

    def descriptor_for_platform(self, platform: str) -> Optional[CapabilityDescriptor]:
        """The negotiated descriptor for one fronted platform (per-chat caps), or None."""
        return self._descriptors_by_platform.get(platform)

    @property
    def auth_revoked(self) -> bool:
        """True once the connector closed 4401 AFTER a successful handshake — the
        per-gateway secret was revoked. Terminal: no reconnect."""
        return self._auth_revoked

    def set_inbound_handler(self, handler: InboundHandler) -> None:
        self._inbound = handler

    def set_interrupt_inbound_handler(self, handler: Any) -> None:
        """Register the callback for connector->gateway interrupt_inbound frames."""
        self._interrupt_inbound_handler = handler

    def set_passthrough_handler(self, handler: Any) -> None:
        """Register ``handler(forward, buffer_id)`` for passthrough_forward frames (§5.1)."""
        self._passthrough_handler = handler

    # ── outbound ─────────────────────────────────────────────────────────
    async def send_outbound(
        self, action: Dict[str, Any], *, platform: Optional[str] = None
    ) -> Dict[str, Any]:
        return await self._request_response(action, platform=platform)

    async def send_follow_up(
        self, action: Dict[str, Any], *, platform: Optional[str] = None
    ) -> Dict[str, Any]:
        # Same outbound frame; the connector dispatches by action.op. Kept as a
        # distinct method to satisfy the transport Protocol.
        return await self._request_response(action, platform=platform)

    def _bot_id_for(self, platform: Optional[str]) -> Optional[str]:
        """The bot_id hello'd for ``platform``, or None when we don't front it.

        A per-frame ``platform`` must ride with its MATCHING botId: the connector
        validates against its accumulated ``platform:botId`` set, and the session
        default botId belongs to the first identity only. For a platform we don't
        front the connector rejects the frame with a structured failure — never a
        wrong-credential send.
        """
        if not platform:
            return None
        return next((b for p, b in self._identities if p == platform), None)

    async def get_chat_info(self, chat_id: str) -> Dict[str, Any]:
        result = await self._request_response({"op": "get_chat_info", "chat_id": chat_id})
        # The connector answers chat-info inside the outbound_result envelope.
        info = result.get("chat_info") or result
        return {"name": info.get("name", chat_id), "type": info.get("type", "dm")}

    async def send_interrupt(self, session_key: str, reason: Optional[str] = None) -> None:
        await self._send({"type": "interrupt", "session_key": session_key, "reason": reason})

    # ── going-idle / buffered-flip (§5.3) ────────────────────────────────
    async def go_idle(self, timeout_s: float = 10.0) -> bool:
        """Ask the connector to flip this instance to buffered-only.

        Awaits the connector-AUTHORITATIVE ``going_idle_ack``. False on timeout /
        not-connected (the caller closes anyway). The read loop keeps serving until
        the ack, so an event landing in the flip window is delivered live, not lost.
        """
        if self._ws is None:
            return False
        self._going_idle_ack = asyncio.get_running_loop().create_future()
        try:
            await self._send({"type": "going_idle"})
            await asyncio.wait_for(self._going_idle_ack, timeout=timeout_s)
            return True
        except Exception:  # noqa: BLE001 - ack is best-effort
            return False
        finally:
            self._going_idle_ack = None

    async def go_dormant(self, timeout_s: float = 10.0) -> bool:
        """Quiesce for a scale-to-zero suspend: go_idle, then close the socket
        WITHOUT setting ``_closing``.

        disconnect() cancels the supervisor (never re-dials on wake, stranding the
        backlog); an unexpected close re-dials immediately (the platform proxy never
        sees load drop, never suspends). Here the reader's fall-through still arms
        the supervisor on the dormant cadence; on resume the re-dial makes the
        connector drain the buffered backlog. Returns the go_idle ack result; the
        close happens regardless. No-op (False) when never connected.
        """
        if self._ws is None:
            return False
        acked = await self.go_idle(timeout_s=timeout_s)
        # Mark dormant BEFORE closing so the supervisor takes the dormant cadence.
        self._dormant = True
        try:
            await asyncio.wait_for(self._ws.close(), timeout=_TEARDOWN_AWAIT_TIMEOUT_S)
        except Exception:  # noqa: BLE001 - best-effort; the reader still ends + arms reconnect
            logger.debug("relay go_dormant: ws.close() raised or timed out", exc_info=True)
        return acked

    async def _send_inbound_ack(self, buffer_id: str) -> None:
        """Ack durable receipt of a replayed buffered inbound; the connector only
        advances its buffer cursor after this (drain-without-dup)."""
        try:
            await self._send({"type": "inbound_ack", "bufferId": buffer_id})
        except Exception:  # noqa: BLE001 - a failed ack just redelivers the entry next time
            logger.debug("relay: inbound_ack send failed for %s", buffer_id)

    async def _request_response(
        self, action: Dict[str, Any], *, platform: Optional[str] = None
    ) -> Dict[str, Any]:
        # Fail fast during teardown: the disconnect() fail-pending loop may already
        # have run, so a future registered now would never be settled.
        if self._closing:
            return {"success": False, "error": "relay transport closed"}
        if self._ws is None:
            return {"success": False, "error": "relay transport not connected"}
        request_id = uuid.uuid4().hex
        fut: asyncio.Future[Dict[str, Any]] = asyncio.get_running_loop().create_future()
        self._pending[request_id] = fut
        frame: Dict[str, Any] = {"type": "outbound", "requestId": request_id, "action": action}
        # Tag the egress platform with its MATCHING advertised botId only when a
        # concrete platform was resolved, so a single-platform gateway emits the
        # exact frame shape as before (connector falls back to session default).
        if platform:
            frame["platform"] = platform
            bot_id = self._bot_id_for(platform)
            if bot_id:
                frame["botId"] = bot_id
        frame_sent = False
        try:
            await self._send(frame)
            frame_sent = True
            return await asyncio.wait_for(fut, timeout=self._outbound_timeout_s)
        except asyncio.TimeoutError:
            # AMBIGUOUS by contract: the frame reached the wire and only the ack is
            # missing — the connector may have applied it. The fail-fast paths
            # above never sent anything (definite non-delivery) and stay unmarked.
            return {"success": False, "error": "relay outbound timed out", "ambiguous": True}
        except Exception as exc:  # noqa: BLE001 - a dead socket is a failed send, not a raise
            # The socket can die between the liveness guard and the write, so _send
            # may raise into callers whose contract is a result dict. A raise from
            # the WRITE = frame never sent (no flag); a failure surfaced by the
            # FUTURE (disconnect failing pending mid-flight) = frame sent, outcome
            # unknown -> ambiguous. CancelledError still propagates (BaseException).
            logger.debug("relay outbound send failed", exc_info=True)
            result: Dict[str, Any] = {"success": False, "error": f"relay send failed: {exc}"}
            if frame_sent:
                result["ambiguous"] = True
            return result
        finally:
            self._pending.pop(request_id, None)

    # ── wire I/O ─────────────────────────────────────────────────────────
    async def _send(self, frame: Dict[str, Any]) -> None:
        if self._ws is None:
            raise RuntimeError("relay transport not connected")
        await self._ws.send(json.dumps(frame) + "\n")

    async def _read_loop(self) -> None:
        # Bind the socket this reader serves: the finally must only clear _ws if
        # it still points at THIS socket (a re-dial may have installed a fresh one).
        ws = self._ws
        buf = ""
        try:
            if ws is None:
                # Lifecycle bug, not a normal path. Fall through to the finally so
                # pending waiters are still settled (an assert here stranded them).
                logger.error("relay ws read loop started with no socket")
                return
            try:
                async for chunk in self._ws:
                    buf += chunk if isinstance(chunk, str) else chunk.decode("utf-8")
                    # Newline-delimited frames; keep any trailing partial line.
                    *lines, buf = buf.split("\n")
                    for line in lines:
                        if line.strip():
                            await self._handle_frame(line)
            except Exception as exc:  # noqa: BLE001 - log + let the task end; reconnection handled below
                if self._close_code_of(exc) == _RELAY_UNAUTHORIZED_CLOSE_CODE and self._handshake_succeeded:
                    self._auth_revoked = True
                    if not self._closing:
                        logger.warning(
                            "relay ws closed 4401 (unauthorized) after a successful handshake — "
                            "treating as a revoked relay credential (opt-out); not reconnecting"
                        )
                elif not self._closing:
                    logger.warning("relay ws read loop ended: %s", exc)
            # Socket closed. Unless this was a deliberate disconnect() or a terminal
            # revocation (re-dialing a dead credential just spins), arm the
            # supervisor: it re-dials and starts a fresh reader.
            if (
                self._reconnect
                and not self._closing
                and not self._auth_revoked
                and (self._supervisor is None or self._supervisor.done())
            ):
                self._supervisor = asyncio.create_task(self._reconnect_loop(), name="relay-ws-reconnect")
        finally:
            # Drop the dead handle (identity-guarded) so every `_ws is None`
            # liveness check reports "not connected" for the whole outage — on
            # exits that arm NO supervisor (terminal 4401, reconnect=False) a send
            # would otherwise register a future nothing can resolve. disconnect()
            # owns the handle during deliberate teardown.
            if self._ws is ws and not self._closing:
                self._ws = None
            # The reader is the ONLY thing that resolves pending futures; once it
            # exits every waiter would block the full outbound timeout. Fail them
            # NOW with the dict shape callers expect (never an exception).
            self._fail_pending(
                lambda fut: fut.set_result({"success": False, "error": "relay transport connection lost"})
            )

    @staticmethod
    def _close_code_of(exc: BaseException) -> Optional[int]:
        """WebSocket close code from a raised exception, or None. websockets'
        ConnectionClosed* expose the Close frame via `.rcvd`/`.sent` (`.code` is
        deprecated in websockets 13+)."""
        for attr in ("rcvd", "sent"):
            fcode = getattr(getattr(exc, attr, None), "code", None)
            if isinstance(fcode, int):
                return fcode
        code = getattr(exc, "code", None)
        return code if isinstance(code, int) else None

    async def _reconnect_loop(self) -> None:
        """Re-dial with capped exponential backoff until a dial succeeds (its reader
        takes over) or disconnect(). Never raises out. After go_dormant() start from
        the dormant cadence; a successful dial clears _dormant so any LATER
        unexpected drop uses the fast backoff."""
        backoff = self._dormant_redial_s if self._dormant else self._reconnect_backoff_s
        while not self._closing:
            await asyncio.sleep(backoff)
            if self._closing:
                return
            try:
                await self._dial_and_start()
                logger.info("relay ws reconnected")
                return
            except Exception as exc:  # noqa: BLE001 - keep retrying on dial failure
                logger.warning("relay ws reconnect failed: %s", exc)
                backoff = min(backoff * 2, self._reconnect_max_backoff_s)

    # ── inbound frame dispatch ───────────────────────────────────────────
    async def _handle_frame(self, line: str) -> None:
        try:
            frame = json.loads(line)
        except json.JSONDecodeError:
            logger.warning("relay: skipping malformed frame")
            return
        # hello/outbound/interrupt are gateway->connector; ignored if echoed.
        handler = self._FRAME_HANDLERS.get(frame.get("type"))
        if handler is not None:
            await handler(self, frame)

    async def _on_descriptor(self, frame: Dict[str, Any]) -> None:
        descriptor = CapabilityDescriptor.from_json(json.dumps(frame.get("descriptor", {})))
        # One descriptor per hello'd identity, keyed by platform for per-chat caps.
        if descriptor.platform:
            self._descriptors_by_platform[descriptor.platform] = descriptor
        # The FIRST descriptor of this generation is the session default; later
        # arrivals must NOT overwrite it (else last-writer-wins across platforms).
        if self._descriptor is None:
            self._descriptor = descriptor
        # Upgrade auth passed at least once: a LATER 4401 is a revocation.
        self._handshake_succeeded = True
        if self._descriptor_ready is not None and not self._descriptor_ready.done():
            self._descriptor_ready.set_result(descriptor)

    async def _on_inbound(self, frame: Dict[str, Any]) -> None:
        if self._inbound is None:
            return
        await self._inbound(_event_from_wire(frame.get("event", {})))
        # A replayed buffered delivery carries a bufferId; ack AFTER the handler
        # has taken it so the connector advances its cursor (no dup).
        buffer_id = frame.get("bufferId")
        if buffer_id:
            await self._send_inbound_ack(str(buffer_id))

    async def _on_going_idle_ack(self, frame: Dict[str, Any]) -> None:
        if self._going_idle_ack is not None and not self._going_idle_ack.done():
            self._going_idle_ack.set_result(None)

    async def _on_outbound_result(self, frame: Dict[str, Any]) -> None:
        fut = self._pending.get(frame.get("requestId", ""))
        if fut is not None and not fut.done():
            fut.set_result(frame.get("result", {}))

    async def _on_interrupt_inbound(self, frame: Dict[str, Any]) -> None:
        if self._interrupt_inbound_handler is not None:
            await self._interrupt_inbound_handler(frame.get("session_key", ""), frame.get("chat_id", ""))

    async def _on_passthrough_forward(self, frame: Dict[str, Any]) -> None:
        # Edge-ACKed passthrough request riding the same WS (no public inbound
        # port needed); bufferId (§5.3) is passed through for ack.
        if self._passthrough_handler is not None:
            fwd = _passthrough_from_wire(frame.get("forward", {}))
            await self._passthrough_handler(fwd, frame.get("bufferId"))

    _FRAME_HANDLERS = {
        "descriptor": _on_descriptor,
        "inbound": _on_inbound,
        "going_idle_ack": _on_going_idle_ack,
        "outbound_result": _on_outbound_result,
        "interrupt_inbound": _on_interrupt_inbound,
        "passthrough_forward": _on_passthrough_forward,
    }
