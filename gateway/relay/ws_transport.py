"""Production WebSocket RelayTransport — the gateway's live link to the connector.

The gateway dials OUT to the connector's relay endpoint over a WebSocket and
speaks the newline-delimited JSON frame protocol defined in the connector repo
(``gateway-gateway`` ``src/relay/protocol.ts``) and mirrored in
``docs/relay-connector-contract.md``:

  gateway -> connector : hello, outbound, interrupt
  connector -> gateway : descriptor, inbound, outbound_result, interrupt_inbound

Frames:
  hello            {type, platform, botId}
  descriptor       {type, descriptor}                       (handshake reply)
  inbound          {type, event, bufferId?}                 (a normalized MessageEvent)
  outbound         {type, requestId, action}                (send/edit/typing/follow_up)
  outbound_result  {type, requestId, result}
  interrupt        {type, session_key, reason?}             (gateway egresses /stop)
  interrupt_inbound{type, session_key, chat_id}             (connector -> owning gateway)

This is the concrete transport behind the ``RelayTransport`` Protocol; the
``RelayAdapter`` delegates all wire I/O to it. Outbound calls block on a
per-request future keyed by ``requestId`` until the matching ``outbound_result``
arrives. A background reader task pumps inbound frames to the registered handler
and resolves pending outbound futures.

EXPERIMENTAL: the frame schema may change without a deprecation cycle until at
least two Class-1 platforms validate it.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import uuid
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

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

# How long to wait for the handshake descriptor and for each outbound result.
_HANDSHAKE_TIMEOUT_S = 30.0
_OUTBOUND_TIMEOUT_S = 30.0
# Bound supervisor/reader/ws.close awaits so a wedged peer cannot stall
# adapter.disconnect. Three sequential awaits at 1.0s stay under the runner's
# default 5s adapter disconnect budget (plus the 2s go_idle ACK budget).
_TEARDOWN_AWAIT_TIMEOUT_S = 1.0
# Bounded drain for in-flight outbound frames at disconnect: long enough for a
# platform edit round-trip through the connector, short enough that shutdown
# stays snappy when the connector is gone. The EFFECTIVE grace is clamped at
# disconnect time so drain + the three sequential teardown awaits stay inside
# the runner's adapter-disconnect budget (gateway/run.py wraps disconnect() in
# asyncio.wait_for; blowing that budget cancels teardown mid-drain, skips the
# fail-pending loop, and leaves callers blocked on _OUTBOUND_TIMEOUT_S).
_DISCONNECT_DRAIN_GRACE_S = 5.0


def _disconnect_drain_grace_s(budget_s: Optional[float] = None) -> float:
    """Effective drain grace: clamped to the caller's disconnect budget.

    ``budget_s`` is the REMAINING budget threaded down by the caller
    (RelayAdapter.disconnect measures what go_idle and monitor teardown
    already consumed). When None, mirrors
    gateway/run.py:_adapter_disconnect_timeout_secs (env override with
    the same variable, same default) rather than importing it — the
    transport must stay importable without the gateway runner. Reserves
    the three sequential teardown awaits plus a small margin.
    """
    budget = _env_disconnect_budget_s() if budget_s is None else max(0.0, budget_s)
    reserved = 3 * _TEARDOWN_AWAIT_TIMEOUT_S + 0.5
    return max(0.0, min(_DISCONNECT_DRAIN_GRACE_S, budget - reserved))


def _env_disconnect_budget_s() -> float:
    """The runner's adapter-disconnect budget, read the same way
    gateway/run.py:_adapter_disconnect_timeout_secs reads it (same env
    variable, same default). Callers above the transport use this to
    apportion the budget across go_idle / monitor teardown / drain."""
    budget = 5.0  # _ADAPTER_DISCONNECT_TIMEOUT_SECS_DEFAULT in gateway/run.py
    raw = os.getenv("HERMES_GATEWAY_ADAPTER_DISCONNECT_TIMEOUT", "").strip()
    if raw:
        try:
            budget = max(0.0, float(raw))
        except ValueError:
            pass
    return budget

# Phase 7 Unit 7d-B: the application close code the connector sends when it
# rejects/revokes a gateway's WS upgrade auth (mirrors the connector's
# `4401` "unauthorized" close — a private-use code, not a standard WS code).
# A 4401 received AFTER a successful handshake means the per-gateway secret was
# revoked (opt-out / deprovision), which the transport treats as terminal.
_RELAY_UNAUTHORIZED_CLOSE_CODE = 4401


def _ws_dial_url(url: str) -> str:
    """Normalize a connector URL to the ``ws(s)://…/relay`` dial target.

    The relay URL is configured once (``GATEWAY_RELAY_URL`` / ``gateway.relay_url``)
    as the connector's BASE URL (e.g. ``https://connector.example``) and shared by
    both the provision POST (which needs ``http(s)://…/relay/provision`` — see
    ``_provision_url``) and the WS dial (which needs ``ws(s)://…/relay``, the path
    the connector mounts its ``WebSocketServer`` on). Two normalizations, both
    load-bearing:

      - scheme: ``https -> wss``, ``http -> ws`` (``websockets.connect`` raises
        "scheme isn't ws or wss" on an http(s) URL).
      - path: ensure it ends in ``/relay`` (the connector returns HTTP 400 on an
        upgrade to any other path, since the WS server is mounted at ``/relay``).

    Idempotent: an already-``ws(s)://…/relay`` URL is returned unchanged, so a URL
    configured WITH the scheme and/or ``/relay`` still works.
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
    """Render the connector's read-only surrounding-context array into the string
    ``MessageEvent.channel_context`` field.

    The connector attaches ``context`` as a list of normalized message objects
    (oldest→newest, same channel) for an addressed turn on a context-capable
    platform (design relay-channel-context). We flatten each to a
    ``<author>: <text>`` line so it rides the SAME read-only injection path that
    history-backfill already uses (run.py prepends ``channel_context`` ahead of
    the trigger message). This is REFERENCE context only — it never triggers the
    agent; the trigger decision was already made connector-side on the addressed
    event alone.

    Returns None when there is no usable context (absent/empty list, or a
    connector that doesn't send the field), so ``channel_context`` stays unset
    and behaviour is byte-identical to today. Never raises — a malformed context
    payload must not break inbound delivery of the (already-admitted) turn.
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
        author = ""
        if isinstance(src, dict):
            author = src.get("user_name") or src.get("user_id") or ""
        lines.append(f"{author}: {text}" if author else str(text))
    if not lines:
        return None
    body = "\n".join(lines)
    return f"[Recent channel messages]\n{body}"


def _normalize_slack_parent_command(
    text: str,
    message_type: MessageType,
) -> tuple[str, MessageType]:
    """Mirror native Slack ``/hermes`` routing for authenticated relay text."""
    stripped = text.strip()
    parent_parts = stripped.split(maxsplit=1)
    if not parent_parts or parent_parts[0] != "/hermes":
        return text, message_type

    from hermes_cli.commands import slack_subcommand_map

    payload = parent_parts[1].strip() if len(parent_parts) > 1 else ""
    subcommand_map = slack_subcommand_map()
    subcommand_map["compact"] = "/compress"
    payload_parts = payload.split() if payload else []
    first_word = payload_parts[0] if payload_parts else ""

    if first_word in subcommand_map:
        rest = payload[len(first_word) :].strip()
        normalized = (
            f"{subcommand_map[first_word]} {rest}".strip()
            if rest
            else subcommand_map[first_word]
        )
    elif payload:
        normalized = payload
    else:
        normalized = "/help"

    normalized_type = (
        MessageType.COMMAND if normalized.startswith("/") else MessageType.TEXT
    )
    return normalized, normalized_type


def _event_from_wire(raw: Dict[str, Any]) -> MessageEvent:
    """Rebuild a MessageEvent from the connector's normalized inbound payload.

    The connector emits SessionSource as the snake_case wire form (§3); map it
    back onto the gateway dataclasses. Unknown message types fall back to TEXT.
    """
    src = raw.get("source", {}) or {}
    from gateway.config import Platform

    platform = src.get("platform", "relay")
    try:
        platform_enum = Platform(platform)
    except ValueError:
        platform_enum = Platform.RELAY

    source = SessionSource(
        platform=platform_enum,
        chat_id=src.get("chat_id", ""),
        chat_type=src.get("chat_type", "dm"),
        chat_name=src.get("chat_name"),
        user_id=src.get("user_id"),
        # Native adapters surface the human-facing DISPLAY name as user_name
        # (e.g. Discord `message.author.display_name`); the connector sends the
        # raw platform username as user_name plus optional user_display_name /
        # user_handle enrichments (contract §3). Prefer the display name for
        # parity with native lanes — session keys derive from user_id, never
        # user_name, so this is presentation-only and key-stable.
        user_name=(
            src.get("user_display_name")
            or src.get("user_name")
            or src.get("user_handle")
        ),
        thread_id=src.get("thread_id"),
        chat_topic=src.get("chat_topic"),
        user_id_alt=src.get("user_id_alt"),
        chat_id_alt=src.get("chat_id_alt"),
        scope_id=src.get("scope_id"),
        parent_chat_id=src.get("parent_chat_id"),
        message_id=src.get("message_id"),
        # The HERMES profile this event is routed to (multiplex mode). The
        # connector stamps it on the wire source when NAS resolves the target
        # profile for a Team-Gateway message; absent for a single-profile
        # gateway, where it stays None and session keys keep the legacy
        # ``agent:main`` namespace (SessionStore._resolve_profile_for_key).
        # Consumed by build_session_key's profile namespacing + the per-turn
        # config/credential scope — the same field the /p/<profile>/ HTTP
        # prefix and per-credential polling adapters already set.
        profile=src.get("profile"),
        # Auto-thread markers (Phase 4): stamped by the CONNECTOR when this
        # event's thread was auto-created by its auto-thread egress policy.
        # Lights the SAME semantic-rename lane native Discord uses
        # (_is_discord_auto_thread_lane's relay-aware sibling reads these).
        auto_thread_created=bool(src.get("auto_thread_created", False)),
        auto_thread_initial_name=src.get("auto_thread_initial_name"),
        # Discord auto-thread session continuity: the connector stamps the
        # thread id this channel message's reply WILL be auto-threaded into
        # (== the message id) so the gateway keys the initiating channel message
        # and its later in-thread follow-ups to ONE session. See
        # build_session_key / SessionSource.prospective_thread_id.
        prospective_thread_id=src.get("prospective_thread_id"),
        # Authentic upstream-trust signal: this event arrived over the
        # per-instance-authenticated relay WS, so the connector already resolved
        # it to this instance's owner-bound author. ``platform`` is the
        # UNDERLYING platform (e.g. discord), not ``relay`` — authz keys the
        # upstream-trust decision off THIS flag, not off ``platform`` (which
        # would miss because the relay adapter is registered under
        # ``Platform.RELAY``). Stamped here, never read off the wire.
        delivered_via_upstream_relay=True,
    )
    try:
        msg_type = MessageType(raw.get("message_type", "text"))
    except ValueError:
        msg_type = MessageType.TEXT

    text = raw.get("text", "")
    if platform_enum == Platform.SLACK:
        # Team Gateway carries Slack slash text over the authenticated message
        # relay, bypassing Hermes' native Slack command callback. Normalize at
        # the wire boundary so adapter-level active-session gates see the real
        # gateway command rather than the legacy `hermes` parent name.
        text, msg_type = _normalize_slack_parent_command(text, msg_type)

    return MessageEvent(
        text=text,
        message_type=msg_type,
        source=source,
        message_id=raw.get("message_id"),
        reply_to_message_id=raw.get("reply_to_message_id"),
        # Richer quoted-reply context (Phase 4): what the user replied TO,
        # when the connector had it in hand (Discord referenced_message,
        # Telegram reply_to_message, WhatsApp context + text cache). Maps to
        # the SAME MessageEvent fields native adapters populate, so run.py's
        # reply-context injection works identically over the relay.
        reply_to_text=(raw.get("reply_to") or {}).get("text"),
        reply_to_author_name=(raw.get("reply_to") or {}).get("author"),
        reply_to_is_own_message=bool((raw.get("reply_to") or {}).get("is_own", False)),
        media_urls=raw.get("media_urls") or [],
        # Surrounding channel/group CONTEXT the connector attached for this
        # addressed turn (design relay-channel-context): a read-only, oldest→
        # newest list of nearby non-addressed messages (Model A pull / Model B
        # buffer). Rendered into the existing ``channel_context`` field — the
        # same read-only injection path history-backfill already uses
        # (run.py prepends it ahead of the trigger message). Absent / empty on a
        # connector that doesn't send it, a dm, or a no-context platform, so
        # this is purely additive and byte-identical to today when unset.
        channel_context=_render_relay_context(raw.get("context")),
        # Structured interactive-prompt reply (Phase 3): carried verbatim off
        # the wire when present ({prompt_id, option_id, label?,
        # prompt_message_id?}). The RelayAdapter's inbound bridge consumes it
        # to resolve pending approvals/confirms/clarifies; a gateway that
        # predates the resolvers just sees the command-shaped text.
        prompt_response=(
            dict(raw["prompt_response"])
            if isinstance(raw.get("prompt_response"), dict)
            else None
        ),
    )


@dataclass
class PassthroughForward:
    """A connector-forwarded passthrough-plane request (Phase 5 §5.1).

    The connector answered the provider's latency-critical ACK at its edge, then
    forwarded the real (already-sanitized) request to this gateway over the WS.
    ``body`` is the exact decoded bytes the connector forwarded (the wire carries
    it base64-encoded for byte parity). ``headers`` preserve arrival order.
    """

    platform: str
    bot_id: str
    method: str
    path: str
    headers: list[tuple[str, str]]
    body: bytes


def _passthrough_from_wire(raw: Dict[str, Any]) -> PassthroughForward:
    """Rebuild a PassthroughForward from the connector's wire frame.

    Mirrors the connector's ``PassthroughForward`` (relay/protocol.ts): the body
    is base64-decoded back to the exact bytes the connector forwarded, so the
    gateway re-processes byte-identical content (the connector is the trust
    boundary; it already verified at the edge).
    """
    import base64

    body_b64 = raw.get("bodyB64", "") or ""
    try:
        body = base64.b64decode(body_b64)
    except Exception:  # noqa: BLE001 - a malformed body must not crash the reader
        body = b""
    headers_raw = raw.get("headers", []) or []
    headers: list[tuple[str, str]] = []
    for pair in headers_raw:
        if isinstance(pair, (list, tuple)) and len(pair) == 2:
            headers.append((str(pair[0]), str(pair[1])))
    return PassthroughForward(
        platform=str(raw.get("platform", "")),
        bot_id=str(raw.get("botId", "")),
        method=str(raw.get("method", "")),
        path=str(raw.get("path", "")),
        headers=headers,
        body=body,
    )


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
        # Phase 1.5 (Shape A): the full SET of (platform, bot_id) this gateway
        # fronts on this one WS. The handshake sends one `hello` per identity so
        # the connector accumulates them into its advertised set (gateway-gateway
        # D-Q1.5b.1); the first identity (platform/bot_id above) is the default an
        # untagged outbound falls back to. Defaults to the single (platform, bot_id)
        # so existing single-platform callers are unchanged.
        self._identities = list(identities) if identities else [(platform, bot_id)]
        self._connect_timeout_s = connect_timeout_s
        self._outbound_timeout_s = outbound_timeout_s
        # Connection auth (Phase 2): when a per-gateway secret is configured the
        # gateway presents an HMAC bearer on the WS upgrade so the connector can
        # authenticate it (reject 4401 otherwise). gateway_id identifies the
        # enrolled instance — the connector peeks it to index its secret verify
        # list, then verifies the signature. Absent -> unauthenticated upgrade
        # (dev/test, or a connector that doesn't enforce auth).
        self._gateway_id = gateway_id
        self._upgrade_secret = upgrade_secret

        # Phase 5 §5.3: a NET-NEW reconnect supervisor. The base transport's
        # _read_loop just ends on socket close ("reconnection is caller policy");
        # with reconnect=True the transport re-dials + re-handshakes after an
        # UNEXPECTED close (not a deliberate disconnect()), so a gateway that went
        # idle/suspended re-establishes its socket — which makes the connector
        # drain that instance's buffered-only delivery-leg backlog (onResume) on
        # the new handshake. Off by default so existing tests + the stub are
        # unaffected; register_relay_adapter turns it on in production.
        self._reconnect = reconnect
        self._reconnect_backoff_s = reconnect_backoff_s
        self._reconnect_max_backoff_s = reconnect_max_backoff_s
        self._supervisor: Optional[asyncio.Task[None]] = None
        # scale-to-zero §Phase 0 (D12/F14): a DORMANT close is distinct from both
        # disconnect() (terminal: cancels the supervisor) and an unexpected close
        # (re-dials immediately). go_dormant() sets this True, then closes the
        # socket WITHOUT setting _closing — so _read_loop's fall-through still
        # kicks the reconnect supervisor (the wake path stays armed), but the
        # supervisor waits on the longer dormant cadence instead of the fast
        # reconnect backoff, so it does not fight the platform's suspend window.
        # On resume (process unfrozen) the pending wait completes, the re-dial
        # succeeds, and the connector drains this instance's buffered backlog on
        # the new handshake. Cleared on a successful re-dial (_dial_and_start).
        self._dormant = False
        # The re-dial poll cadence while dormant. A suspended machine's event
        # loop is frozen, so this timer only advances once the machine is awake;
        # it just needs to be short enough that a freshly-woken machine re-dials
        # promptly (the connector's wake poke is what triggers the platform
        # autostart in the first place — §3.4(5)).
        self._dormant_redial_s = 1.0

        self._ws: Any = None
        self._reader: Optional[asyncio.Task[None]] = None
        self._inbound: Optional[InboundHandler] = None
        self._descriptor: Optional[CapabilityDescriptor] = None
        # Phase 1.5 multi-platform: descriptors keyed by the underlying platform
        # (one per hello'd identity). `_descriptor` above stays the FIRST
        # (primary-identity) descriptor for back-compat; this map is the
        # per-platform capability surface read via `descriptor_for_platform`.
        self._descriptors_by_platform: Dict[str, CapabilityDescriptor] = {}
        self._descriptor_ready: asyncio.Future[CapabilityDescriptor] | None = None
        # requestId -> future awaiting the matching outbound_result.
        self._pending: Dict[str, asyncio.Future[Dict[str, Any]]] = {}
        # Phase 5 §5.3: future awaiting the connector's going_idle_ack.
        self._going_idle_ack: asyncio.Future[None] | None = None
        self._closing = False
        # Phase 7 Unit 7d-B: a 4401 (unauthorized) close AFTER we have already
        # handshaked successfully at least once means the connector REVOKED this
        # gateway's per-gateway secret — i.e. the operator opted this instance
        # OUT of the relay (Unit 7b deprovision). That is TERMINAL: the secret is
        # gone, so re-dialing just spins against a dead credential forever
        # (the "retrying 4401" the dashboard showed). We stop reconnecting and
        # surface it as a clean, non-retryable "disabled" state. A 4401 BEFORE
        # any successful handshake stays retryable — that's a cold-start /
        # not-yet-provisioned race, not a revocation.
        self._handshake_succeeded = False
        self._auth_revoked = False

    # ── lifecycle ────────────────────────────────────────────────────────
    async def connect(self) -> bool:
        await self._dial_and_start()
        return True

    async def _dial_and_start(self) -> None:
        """Open the socket, start the reader, send hello. Used by connect() and
        by the reconnect supervisor on a re-dial."""
        loop = asyncio.get_running_loop()
        self._descriptor_ready = loop.create_future()
        # A fresh handshake is coming; clear any stale descriptor so handshake()
        # awaits the new one (matters on a re-dial). The per-platform map resets
        # with it — a reconnected connector re-sends one descriptor per hello.
        self._descriptor = None
        self._descriptors_by_platform = {}
        # scale-to-zero (D12): a successful (re-)dial ends any dormant state — we
        # are live again, so a subsequent UNEXPECTED close should reconnect on the
        # normal fast backoff, not the dormant cadence.
        self._dormant = False
        headers = self._upgrade_headers()
        # WAN-friendly keepalive: customer gateways cross WAN paths to the
        # connector; the websockets library default (ping_interval=20,
        # ping_timeout=20) gives the peer only a 20s pong deadline, which
        # produces spurious `1011 keepalive ping timeout` closes under
        # transient latency / event-loop stalls (Coatue incident 2026-08-18).
        # ping_timeout=60 tolerates such stalls while still detecting a dead
        # link within ~90s worst case (30s interval + 60s pong deadline).
        if headers:
            self._ws = await websockets.connect(  # type: ignore[union-attr]
                self._url,
                additional_headers=headers,
                ping_interval=30,
                ping_timeout=60,
            )
        else:
            self._ws = await websockets.connect(  # type: ignore[union-attr]
                self._url,
                ping_interval=30,
                ping_timeout=60,
            )
        self._reader = asyncio.create_task(self._read_loop(), name="relay-ws-reader")
        # Send one hello PER fronted identity (Phase 1.5 Shape A). The connector
        # accumulates them into its advertised set (the first sets the session
        # default; each adds to the egress-allowed set). A single-platform gateway
        # sends exactly one hello — byte-identical to before. The descriptor for
        # the FIRST identity resolves handshake(); later descriptors are absorbed.
        for platform, bot_id in self._identities:
            hello: Dict[str, Any] = {"type": "hello", "platform": platform, "botId": bot_id}
            # Phase 4: declare the gateway's slash-command set on the Discord
            # hello. The connector (which holds the bot token) reconciles
            # Discord's global registration against it — idempotent, detached,
            # best-effort on its side; a connector predating the field ignores
            # it (additive). Only Discord has an app-command registry.
            if platform == "discord":
                try:
                    from gateway.relay.command_manifest import build_relay_command_manifest

                    hello["command_manifest"] = build_relay_command_manifest()
                except Exception:  # noqa: BLE001 - manifest is enrichment, never blocks the handshake
                    logger.debug("relay command manifest build failed", exc_info=True)
            await self._send(hello)

    def _upgrade_headers(self) -> Dict[str, str]:
        """Auth headers for the WS upgrade, or {} when no secret is configured.

        Presents ``Authorization: Bearer *** where the token is a signed
        bearer built with the per-gateway secret (``gateway/relay/auth.py``
        ``make_upgrade_token``), keyed by ``gateway_id`` so the connector can
        index its verify list. The connector rejects the upgrade (close 4401)
        when this is missing/invalid/revoked; an unauthenticated connector
        ignores it.
        """
        if not (self._upgrade_secret and self._gateway_id):
            return {}
        from gateway.relay.auth import make_upgrade_token

        token = make_upgrade_token(self._gateway_id, self._upgrade_secret)
        return {"Authorization": f"Bearer {token}"}

    async def disconnect(self, *, budget_s: Optional[float] = None) -> None:
        """Tear down the socket, draining in-flight outbound frames first.

        ``budget_s`` is the REMAINING wall-clock budget the caller can spend
        here (RelayAdapter.disconnect threads it down after go_idle / monitor
        teardown). When None, the env-mirrored runner default applies.
        """
        self._closing = True
        try:
            # Drain grace: a trailing outbound frame (typically the turn's
            # finalize edit) may still be awaiting its outbound_result. Failing
            # it immediately loses a message the connector was about to ack —
            # staging incident 2026-08-09 froze a Slack reply at its preview
            # snapshot exactly this way. Give in-flight requests a short bounded
            # window to resolve before tearing the socket down.
            pending = [f for f in self._pending.values() if not f.done()]
            if pending:
                _grace = _disconnect_drain_grace_s(budget_s)
                if _grace > 0:
                    try:
                        # asyncio.wait (not wait_for+gather): on timeout it must NOT
                        # cancel the futures — the fail-any-remaining loop below owns
                        # their terminal state.
                        await asyncio.wait(pending, timeout=_grace)
                    except Exception:  # noqa: BLE001 - grace is best-effort
                        pass
            if self._supervisor is not None:
                self._supervisor.cancel()
                try:
                    await asyncio.wait_for(
                        self._supervisor, timeout=_TEARDOWN_AWAIT_TIMEOUT_S
                    )
                except (asyncio.TimeoutError, asyncio.CancelledError, Exception):  # noqa: BLE001 - best-effort teardown
                    pass
                self._supervisor = None
            if self._reader is not None:
                self._reader.cancel()
                try:
                    await asyncio.wait_for(self._reader, timeout=_TEARDOWN_AWAIT_TIMEOUT_S)
                except (asyncio.TimeoutError, asyncio.CancelledError, Exception):  # noqa: BLE001 - best-effort teardown
                    pass
                self._reader = None
            if self._ws is not None:
                try:
                    await asyncio.wait_for(self._ws.close(), timeout=_TEARDOWN_AWAIT_TIMEOUT_S)
                except (asyncio.TimeoutError, asyncio.CancelledError, Exception):  # noqa: BLE001
                    pass
                finally:
                    self._ws = None
        finally:
            # Fail any in-flight outbound waiters so callers don't hang.
            # Runs in a finally so a cancellation landing anywhere in the
            # drain/teardown above (the runner's wait_for budget, an outer
            # cleanup deadline) can NEVER leave a registered future
            # unresolved — a stranded waiter would otherwise block until
            # _OUTBOUND_TIMEOUT_S (30s). Idempotent: done futures are
            # skipped, so a second disconnect() pass is safe.
            for fut in self._pending.values():
                if not fut.done():
                    fut.set_exception(RuntimeError("relay transport closed"))
            self._pending.clear()
            if self._going_idle_ack is not None and not self._going_idle_ack.done():
                self._going_idle_ack.set_exception(RuntimeError("relay transport closed"))

    async def handshake(self) -> CapabilityDescriptor:
        if self._descriptor is not None:
            return self._descriptor
        if self._descriptor_ready is None:
            raise RuntimeError("handshake() called before connect()")
        return await asyncio.wait_for(self._descriptor_ready, timeout=self._connect_timeout_s)

    def descriptor_for_platform(self, platform: str) -> Optional[CapabilityDescriptor]:
        """The negotiated descriptor for one fronted platform, or None.

        Phase 1.5 multi-platform: the connector replies one descriptor per
        hello'd identity; they accumulate here keyed by the descriptor's own
        ``platform`` field. Callers (RelayAdapter) use this to resolve PER-CHAT
        capabilities — e.g. Discord's 2000-char max_message_length vs
        Telegram's 4096 — instead of applying the primary identity's scalar
        descriptor to every platform this gateway fronts.
        """
        return self._descriptors_by_platform.get(platform)

    @property
    def auth_revoked(self) -> bool:
        """True once the connector closed the socket with 4401 AFTER a prior
        successful handshake — i.e. the per-gateway secret was revoked (the
        operator opted this instance out of the relay). Terminal: the transport
        stops reconnecting, and the adapter surfaces a clean "disabled" state."""
        return self._auth_revoked

    def set_inbound_handler(self, handler: InboundHandler) -> None:
        self._inbound = handler

    # ── outbound ─────────────────────────────────────────────────────────
    async def send_outbound(
        self, action: Dict[str, Any], *, platform: Optional[str] = None
    ) -> Dict[str, Any]:
        return await self._request_response(action, platform=platform)

    async def send_follow_up(
        self, action: Dict[str, Any], *, platform: Optional[str] = None
    ) -> Dict[str, Any]:
        # follow_up rides the same outbound frame; the connector dispatches by
        # action.op. Kept as a distinct method to satisfy the transport Protocol
        # and to make the A2 call site explicit.
        return await self._request_response(action, platform=platform)

    def _bot_id_for(self, platform: Optional[str]) -> Optional[str]:
        """The bot_id this transport advertised at hello for ``platform`` (Phase 1.5).

        The connector validates a per-frame egress target against the SET of
        ``platform:botId`` pairs it accumulated from the N hellos, so a per-frame
        ``platform`` must ride with its MATCHING ``botId`` (the session default
        botId belongs to the first identity and would mis-key for a second
        platform). Resolved from the identity set this transport was built with.
        None when the platform isn't one we front (the connector then rejects it
        with a structured failure — never a wrong-credential send)."""
        if not platform:
            return None
        for p, b in self._identities:
            if p == platform:
                return b
        return None

    async def get_chat_info(self, chat_id: str) -> Dict[str, Any]:
        result = await self._request_response(
            {"op": "get_chat_info", "chat_id": chat_id}, frame_type="outbound"
        )
        # The connector answers chat-info inside the outbound_result envelope.
        info = result.get("chat_info") or result
        return {"name": info.get("name", chat_id), "type": info.get("type", "dm")}

    async def send_interrupt(self, session_key: str, reason: Optional[str] = None) -> None:
        await self._send({"type": "interrupt", "session_key": session_key, "reason": reason})

    # ── going-idle / buffered-flip (Phase 5 §5.3) ────────────────────────
    async def go_idle(self, timeout_s: float = 10.0) -> bool:
        """Ask the connector to flip this instance's destination to buffered-only.

        Sends ``going_idle`` and awaits the connector's ``going_idle_ack`` — the
        connector-AUTHORITATIVE confirmation that live delivery has stopped and
        subsequent inbound buffers durably (Q-5.3c). Returns True on ack, False on
        timeout / not-connected (the caller proceeds to close anyway — at worst a
        live event races a closing socket exactly as before §5.3, no regression).

        The gateway stays serving (the read loop keeps handling inbound) until the
        ack, so an event landing in the flip window is delivered live, not lost.
        """
        if self._ws is None:
            return False
        loop = asyncio.get_running_loop()
        self._going_idle_ack = loop.create_future()
        try:
            await self._send({"type": "going_idle"})
            await asyncio.wait_for(self._going_idle_ack, timeout=timeout_s)
            return True
        except (asyncio.TimeoutError, Exception):  # noqa: BLE001 - ack is best-effort
            return False
        finally:
            self._going_idle_ack = None

    async def go_dormant(self, timeout_s: float = 10.0) -> bool:
        """Quiesce this transport for a scale-to-zero suspend (D12 / Phase 0).

        Distinct from BOTH ``disconnect()`` and an unexpected close (F14):
          - ``disconnect()`` sets ``_closing=True`` and CANCELS the reconnect
            supervisor — terminal, "shutting down for good." A machine suspended
            after that never re-dials on wake, so its buffered backlog strands.
          - An unexpected close re-dials IMMEDIATELY (fast backoff) — the socket
            never stays down, so the platform proxy never sees the connection go
            away and never suspends the machine.

        ``go_dormant()`` is the third mode the suspend behaviour needs:
          1. ``go_idle()`` → the connector flips this instance to buffered-only
             and acks (so inbound that arrives while we sleep buffers durably and
             replays on the next handshake).
          2. Close the socket so the platform proxy sees load drop to zero (the
             precondition for Fly ``autostop:"suspend"``) — but WITHOUT setting
             ``_closing``. The reader's normal end-of-socket fall-through still
             arms the reconnect supervisor, so the wake path stays live; the
             ``_dormant`` flag just makes that supervisor poll on the dormant
             cadence rather than fight the suspend window.

        On resume (process unfrozen) the supervisor's pending wait completes, the
        re-dial succeeds, and the connector drains the buffered backlog on the new
        handshake. Returns the ``go_idle`` ack result (True on ack); the dormancy
        close happens regardless (a missed ack at worst races one live event onto
        a closing socket, exactly as §5.3 already tolerates).

        No-op-safe: a transport that never connected (``_ws is None``) just
        returns False without closing.
        """
        if self._ws is None:
            return False
        acked = await self.go_idle(timeout_s=timeout_s)
        # Mark dormant BEFORE closing so the supervisor (armed by the reader's
        # fall-through) takes the dormant cadence, and a racing live event can't
        # flip us back to a fast reconnect.
        self._dormant = True
        try:
            await asyncio.wait_for(
                self._ws.close(), timeout=_TEARDOWN_AWAIT_TIMEOUT_S
            )
        except (asyncio.TimeoutError, Exception):  # noqa: BLE001 - best-effort; the reader still ends + arms reconnect
            logger.debug("relay go_dormant: ws.close() raised or timed out", exc_info=True)
        return acked

    async def _send_inbound_ack(self, buffer_id: str) -> None:
        """Acknowledge durable receipt of a buffered inbound delivery (§5.3).

        Sent after the adapter has durably taken a buffered inbound event the
        connector replayed on reconnect; the connector acks the buffer entry only
        after this, giving drain-without-dup on the delivery leg.
        """
        try:
            await self._send({"type": "inbound_ack", "bufferId": buffer_id})
        except Exception:  # noqa: BLE001 - a failed ack just redelivers the entry next time
            logger.debug("relay: inbound_ack send failed for %s", buffer_id)

    async def _request_response(
        self,
        action: Dict[str, Any],
        frame_type: str = "outbound",
        *,
        platform: Optional[str] = None,
    ) -> Dict[str, Any]:
        if self._closing:
            # Teardown in progress: the disconnect() fail-pending loop may
            # already have run, so a future registered now would never be
            # resolved or failed — the caller would block the full
            # _OUTBOUND_TIMEOUT_S for a socket that is going away. Fail fast.
            return {"success": False, "error": "relay transport closed"}
        if self._ws is None:
            return {"success": False, "error": "relay transport not connected"}
        request_id = uuid.uuid4().hex
        loop = asyncio.get_running_loop()
        fut: asyncio.Future[Dict[str, Any]] = loop.create_future()
        self._pending[request_id] = fut
        frame: Dict[str, Any] = {"type": frame_type, "requestId": request_id, "action": action}
        # Phase 1.5: tag the per-frame egress platform on the OutboundFrame
        # envelope (gateway-gateway D-Q1.5b.1), with its MATCHING advertised botId
        # so the connector's `${platform}:${botId}` advertised-set check passes.
        # Only set when a concrete platform was resolved for this chat so a
        # single-platform gateway emits the exact frame shape as before (the
        # connector falls back to the session's default platform when absent).
        if platform:
            frame["platform"] = platform
            bot_id = self._bot_id_for(platform)
            if bot_id:
                frame["botId"] = bot_id
        try:
            await self._send(frame)
            return await asyncio.wait_for(fut, timeout=self._outbound_timeout_s)
        except asyncio.TimeoutError:
            return {"success": False, "error": "relay outbound timed out"}
        except Exception as exc:  # noqa: BLE001 - a dead socket is a failed send, not a raise
            # No `is None` check can close the window where the socket dies
            # BETWEEN the liveness guard above and the actual write — the
            # reader's finally hasn't cleared _ws yet, so _send raises
            # ConnectionClosed straight into callers whose contract is a
            # result dict (RelayAdapter.send consumes it with no try).
            # Report it like every other failed send. CancelledError is a
            # BaseException, so cancellation still propagates.
            logger.debug("relay %s send failed", frame_type, exc_info=True)
            return {"success": False, "error": f"relay send failed: {exc}"}
        finally:
            self._pending.pop(request_id, None)

    # ── wire I/O ─────────────────────────────────────────────────────────
    async def _send(self, frame: Dict[str, Any]) -> None:
        if self._ws is None:
            raise RuntimeError("relay transport not connected")
        await self._ws.send(json.dumps(frame) + "\n")

    async def _read_loop(self) -> None:
        # Bind the socket this reader serves: the finally below must only
        # clear _ws if it still points at THIS socket (a supervisor re-dial
        # may have already installed a fresh one by the time we unwind).
        ws = self._ws
        buf = ""
        try:
            if ws is None:
                # Scheduled without a socket (a lifecycle bug, not a normal
                # path). The old `assert` here escaped BEFORE the finally
                # existed to fail pending futures — the one exit that could
                # still strand waiters for the full outbound timeout. Fall
                # through to the finally instead; it settles them all.
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
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001 - log + let the task end; reconnection handled below
                # Phase 7 Unit 7d-B: detect a 4401 (unauthorized) close. After a prior
                # successful handshake this is a REVOCATION (opt-out / deprovision) —
                # the per-gateway secret is gone, so reconnecting is futile. Latch a
                # terminal "auth revoked" state and DON'T re-dial. Before any
                # successful handshake a 4401 stays retryable (cold-start race).
                if self._close_code_of(exc) == _RELAY_UNAUTHORIZED_CLOSE_CODE and self._handshake_succeeded:
                    self._auth_revoked = True
                    if not self._closing:
                        logger.warning(
                            "relay ws closed 4401 (unauthorized) after a successful handshake — "
                            "treating as a revoked relay credential (opt-out); not reconnecting"
                        )
                elif not self._closing:
                    logger.warning("relay ws read loop ended: %s", exc)
            # Phase 5 §5.3: the socket closed. If reconnect is enabled and this was
            # NOT a deliberate disconnect(), kick the reconnect supervisor so the
            # gateway re-dials + re-handshakes (which triggers the connector's
            # buffered-flip drain on the new handshake). Self-scheduling: the reader
            # ends here, the supervisor re-dials and starts a fresh reader.
            # Phase 7 Unit 7d-B: a revoked credential (terminal 4401) is the one case
            # we deliberately do NOT reconnect — the secret is dead until the
            # instance is recreated, so spinning would just reproduce the failure.
            if (
                self._reconnect
                and not self._closing
                and not self._auth_revoked
                and (self._supervisor is None or self._supervisor.done())
            ):
                self._supervisor = asyncio.create_task(
                    self._reconnect_loop(), name="relay-ws-reconnect"
                )
        finally:
            # The socket this reader served is dead. Drop the handle (identity-
            # guarded: a re-dial that already installed a FRESH socket must not
            # be clobbered) so every `self._ws is None` liveness check — send,
            # _request_response, go_idle, go_dormant — reports "not connected"
            # for the whole outage. Without this, _ws kept pointing at the dead
            # socket on every reader exit that arms NO supervisor (terminal
            # 4401 revocation, reconnect=False transports), and a send there
            # registered a future nothing could resolve: a full
            # _outbound_timeout_s (~30s) wedge — including the revocation
            # path's own fatal-error notification. disconnect() owns the
            # handle during deliberate teardown, so leave it alone then.
            if self._ws is ws and not self._closing:
                self._ws = None
            # The reader is the ONLY thing that can resolve a pending
            # outbound_result future — once it exits (socket dropped, error,
            # or cancellation cleanup) every in-flight _request_response waiter
            # is unresolvable and would otherwise block the full
            # _outbound_timeout_s (~30s) on a dead socket (Coatue incident
            # 2026-08-18: stuck sends after a 1011 keepalive close). Fail them
            # NOW with the dict shape callers expect (never an exception on
            # the outbound path). list() snapshot: set_result wakes waiters
            # whose finally-pop would otherwise mutate the dict mid-iteration.
            for _rid, fut in list(self._pending.items()):
                if not fut.done():
                    fut.set_result(
                        {"success": False, "error": "relay transport connection lost"}
                    )
            self._pending.clear()

    @staticmethod
    def _close_code_of(exc: BaseException) -> Optional[int]:
        """Best-effort extraction of a WebSocket close code from a raised
        exception. websockets' ConnectionClosed* expose the peer's Close frame
        via `.rcvd`/`.sent` (preferred; `.code` is deprecated in websockets 13+).
        Returns None when unknown."""
        for attr in ("rcvd", "sent"):
            frame = getattr(exc, attr, None)
            fcode = getattr(frame, "code", None)
            if isinstance(fcode, int):
                return fcode
        code = getattr(exc, "code", None)
        return code if isinstance(code, int) else None

    async def _reconnect_loop(self) -> None:
        """Re-dial the connector with capped exponential backoff until reconnected
        or disconnect() is called. NET-NEW for §5.3: a re-established socket makes
        the connector replay this instance's buffered-only backlog on the new
        handshake (the delivery-leg onResume). Never raises out (a re-dial failure
        just retries); ends when a dial succeeds (its reader takes over) or closing.

        scale-to-zero (D12): when the close was a deliberate go_dormant() rather
        than an unexpected drop, start from the dormant poll cadence. On a
        suspended machine the event loop is frozen, so this sleep only advances
        once the machine is awake — it just needs to be short enough that a
        freshly-woken machine re-dials promptly. A successful _dial_and_start()
        clears _dormant, so any LATER unexpected drop reconnects on the normal
        fast backoff."""
        backoff = self._dormant_redial_s if self._dormant else self._reconnect_backoff_s
        while not self._closing:
            try:
                await asyncio.sleep(backoff)
            except asyncio.CancelledError:
                raise
            if self._closing:
                return
            try:
                await self._dial_and_start()
                logger.info("relay ws reconnected")
                return  # the fresh reader is running; supervisor's job is done
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001 - keep retrying on dial failure
                logger.warning("relay ws reconnect failed: %s", exc)
                backoff = min(backoff * 2, self._reconnect_max_backoff_s)

    async def _handle_frame(self, line: str) -> None:
        try:
            frame = json.loads(line)
        except json.JSONDecodeError:
            logger.warning("relay: skipping malformed frame")
            return
        ftype = frame.get("type")
        if ftype == "descriptor":
            descriptor = CapabilityDescriptor.from_json(json.dumps(frame.get("descriptor", {})))
            # Phase 1.5 multi-platform: one descriptor frame arrives per hello'd
            # identity. Accumulate them keyed by the descriptor's own platform so
            # the adapter can resolve PER-CHAT capabilities (e.g. Discord's 2000
            # vs Telegram's 4096 max_message_length) instead of collapsing N
            # platforms onto whichever descriptor arrived last.
            if descriptor.platform:
                self._descriptors_by_platform[descriptor.platform] = descriptor
            # The FIRST descriptor of this connection generation is the session
            # default (the primary identity's) — later arrivals must NOT
            # overwrite it, or the scalar capability surface silently becomes
            # last-writer-wins across platforms.
            if self._descriptor is None:
                self._descriptor = descriptor
            # Phase 7 Unit 7d-B: a received descriptor means the WS upgrade auth
            # passed and the connector accepted us — record that we've handshaked
            # at least once, so a LATER 4401 close is read as a revocation
            # (opt-out), not a cold-start race.
            self._handshake_succeeded = True
            if self._descriptor_ready is not None and not self._descriptor_ready.done():
                self._descriptor_ready.set_result(descriptor)
        elif ftype == "inbound":
            if self._inbound is not None:
                event = _event_from_wire(frame.get("event", {}))
                await self._inbound(event)
                # Phase 5 §5.3: a buffered delivery (replayed on reconnect) carries
                # a bufferId; ack it after the handler has durably taken it so the
                # connector advances its delivery-leg buffer cursor (no dup). A live
                # delivery has no bufferId — nothing to ack.
                buffer_id = frame.get("bufferId")
                if buffer_id:
                    await self._send_inbound_ack(str(buffer_id))
        elif ftype == "going_idle_ack":
            # Phase 5 §5.3: the connector confirmed our destination is now
            # buffered-only; resolve the waiter go_idle() is blocked on.
            if self._going_idle_ack is not None and not self._going_idle_ack.done():
                self._going_idle_ack.set_result(None)
        elif ftype == "outbound_result":
            fut = self._pending.get(frame.get("requestId", ""))
            if fut is not None and not fut.done():
                fut.set_result(frame.get("result", {}))
        elif ftype == "interrupt_inbound":
            # Bridged into the adapter's interrupt path by the runner wiring.
            handler = getattr(self, "_interrupt_inbound_handler", None)
            if handler is not None:
                await handler(frame.get("session_key", ""), frame.get("chat_id", ""))
        elif ftype == "passthrough_forward":
            # Phase 5 §5.1: a forwarded passthrough-plane request (Discord
            # interaction, Twilio, …) the connector already edge-ACKed. It rides
            # the SAME outbound WS as inbound messages so a hosted gateway needs
            # no public inbound port. Dispatch to the adapter's handler; the
            # bufferId (when present, §5.3 buffered flip) is passed for ack.
            handler = getattr(self, "_passthrough_handler", None)
            if handler is not None:
                fwd = _passthrough_from_wire(frame.get("forward", {}))
                await handler(fwd, frame.get("bufferId"))
        else:
            # hello/outbound/interrupt are gateway->connector; ignore if echoed.
            pass

    def set_interrupt_inbound_handler(self, handler: Any) -> None:
        """Register the callback for connector->gateway interrupt_inbound frames."""
        self._interrupt_inbound_handler = handler

    def set_passthrough_handler(self, handler: Any) -> None:
        """Register the callback for connector->gateway passthrough_forward frames.

        Mirrors set_interrupt_inbound_handler: the runner/adapter wires this so a
        forwarded passthrough request (Phase 5 §5.1) reaches the adapter over the
        same outbound WS the gateway already holds. ``handler(forward, buffer_id)``.
        """
        self._passthrough_handler = handler
