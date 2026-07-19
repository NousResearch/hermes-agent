"""
Home Assistant platform adapter.

Connects to the HA WebSocket API for real-time event monitoring.
State-change events are converted to MessageEvent objects and forwarded
to the agent for processing.  Outbound messages are delivered as HA
persistent notifications.

Requires:
- aiohttp (already in messaging extras)
- HASS_TOKEN env var (Long-Lived Access Token)
- HASS_URL env var (default: http://homeassistant.local:8123)
"""

import asyncio
import json
import logging
import os
import time
import uuid
from datetime import datetime
from typing import Any, Dict, Optional, Set

try:
    import aiohttp
    AIOHTTP_AVAILABLE = True
except ImportError:
    AIOHTTP_AVAILABLE = False
    aiohttp = None  # type: ignore[assignment]

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import (
    BasePlatformAdapter,
    MessageEvent,
    MessageType,
    SendResult,
)

logger = logging.getLogger(__name__)

# Bounds every WS/session teardown await (ws.close()/session.close()) so a
# wedged CLOSE-WAIT socket can't block the reconnect ladder or disconnect()
# forever. Refs: NousResearch/hermes-agent#67470
_DRAIN_TIMEOUT = 5.0
# Bounds each receive_json() in the auth handshake ladder so a server that
# accepts the socket but never responds can't freeze _ws_connect() forever.
# Refs: NousResearch/hermes-agent#67470
_HANDSHAKE_TIMEOUT = 30.0
# Cause-agnostic watchdog (#67470, mirrors the Telegram adapter's wedged-
# recovery watchdog, commit c2cb37532): if _listen_loop stops making progress
# -- wedged on an await no local bound covers -- for this long while the
# adapter is still "running", nothing else notices and the gateway goes
# silently deaf. The watchdog force-cancels and respawns it.
_LISTEN_STUCK_TIMEOUT = 300.0
# How often the watchdog checks _last_progress against _LISTEN_STUCK_TIMEOUT.
_WATCHDOG_INTERVAL = 60.0
# After the watchdog's HA-protocol ping, how long to wait for the pong to
# surface as reader progress before declaring the listener wedged. The pong
# arrives through _read_events' async-for (single-reader invariant), so the
# watchdog observes it indirectly via _last_progress.
_PING_GRACE = 10.0


def check_ha_requirements() -> bool:
    """Check if Home Assistant runtime dependencies are available."""
    return AIOHTTP_AVAILABLE


def validate_ha_config(config: PlatformConfig) -> bool:
    """Return True when Home Assistant has enough credential config to connect."""
    token = (getattr(config, "token", None) or os.getenv("HASS_TOKEN", "")).strip()
    return bool(token)


class HomeAssistantAdapter(BasePlatformAdapter):
    """
    Home Assistant WebSocket adapter.

    Subscribes to ``state_changed`` events and forwards them as
    MessageEvent objects.  Supports domain/entity filtering and
    per-entity cooldowns to avoid event floods.
    """

    MAX_MESSAGE_LENGTH = 4096

    # Reconnection backoff schedule (seconds)
    _BACKOFF_STEPS = [5, 10, 30, 60]

    def __init__(self, config: PlatformConfig):
        super().__init__(config, Platform.HOMEASSISTANT)

        # Connection state
        self._session: Optional["aiohttp.ClientSession"] = None
        self._ws: Optional["aiohttp.ClientWebSocketResponse"] = None
        self._rest_session: Optional["aiohttp.ClientSession"] = None
        self._listen_task: Optional[asyncio.Task] = None
        self._watchdog_task: Optional[asyncio.Task] = None
        self._msg_id: int = 0
        # Monotonic timestamp bumped by _listen_loop/_read_events on every
        # iteration or received event; the watchdog compares against this to
        # detect a wedged listener (#67470).
        self._last_progress: float = time.monotonic()

        # Configuration from extra
        extra = config.extra or {}
        token = config.token or os.getenv("HASS_TOKEN", "")
        url = extra.get("url") or os.getenv("HASS_URL", "http://homeassistant.local:8123")
        self._hass_url: str = url.rstrip("/")
        self._hass_token: str = token

        # Event filtering
        self._watch_domains: Set[str] = set(extra.get("watch_domains", []))
        self._watch_entities: Set[str] = set(extra.get("watch_entities", []))
        self._ignore_entities: Set[str] = set(extra.get("ignore_entities", []))
        self._watch_all: bool = bool(extra.get("watch_all", False))
        self._cooldown_seconds: int = int(extra.get("cooldown_seconds", 30))

        # Cooldown tracking: entity_id -> last_event_timestamp
        self._last_event_time: Dict[str, float] = {}

    def _next_id(self) -> int:
        """Return the next WebSocket message ID."""
        self._msg_id += 1
        return self._msg_id

    # ------------------------------------------------------------------
    # Connection lifecycle
    # ------------------------------------------------------------------

    async def connect(self, *, is_reconnect: bool = False) -> bool:
        """Connect to HA WebSocket API and subscribe to events."""
        if not AIOHTTP_AVAILABLE:
            logger.warning("[%s] aiohttp not installed. Run: pip install aiohttp", self.name)
            return False

        if not self._hass_token:
            logger.warning("[%s] No HASS_TOKEN configured", self.name)
            return False

        try:
            success = await self._ws_connect()
            if not success:
                return False

            # Dedicated REST session for send() calls
            self._rest_session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=30)
            )

            # Warn if no event filters are configured
            if not self._watch_domains and not self._watch_entities and not self._watch_all:
                logger.warning(
                    "[%s] No watch_domains, watch_entities, or watch_all configured. "
                    "All state_changed events will be dropped. Configure filters in "
                    "your HA platform config to receive events.",
                    self.name,
                )

            # Start background listener + its cause-agnostic watchdog (#67470)
            self._last_progress = time.monotonic()
            self._listen_task = asyncio.create_task(self._listen_loop())
            self._watchdog_task = asyncio.create_task(self._watchdog_loop())
            self._running = True
            logger.info("[%s] Connected to %s", self.name, self._hass_url)
            return True

        except Exception as e:
            logger.error("[%s] Failed to connect: %s", self.name, e)
            return False

    async def _ws_connect(self) -> bool:
        """Establish WebSocket connection and authenticate."""
        ws_url = self._hass_url.replace("https://", "wss://").replace("http://", "ws://")
        ws_url = f"{ws_url}/api/websocket"

        # Build into a local first (#67470). The previous code assigned
        # self._session before attempting ws_connect(); if ws_connect()
        # raised, that session was left dangling — referenced by self._session
        # but never connected — until the next reconnect loop's cleanup
        # happened to close it. Only wire self._session/self._ws up once the
        # socket is actually usable, and close the local on failure.
        session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=30))
        try:
            ws = await session.ws_connect(ws_url, heartbeat=30, timeout=30)
        except Exception:
            await self._bounded_close(session, "WS session")
            raise

        self._session = session
        self._ws = ws

        try:
            # Step 1: Receive auth_required
            msg = await asyncio.wait_for(self._ws.receive_json(), timeout=_HANDSHAKE_TIMEOUT)
            if msg.get("type") != "auth_required":
                logger.error("[%s] Expected auth_required, got: %s", self.name, msg.get("type"))
                await self._cleanup_ws()
                return False

            # Step 2: Send auth
            await asyncio.wait_for(
                self._ws.send_json({
                    "type": "auth",
                    "access_token": self._hass_token,
                }),
                timeout=_HANDSHAKE_TIMEOUT,
            )

            # Step 3: Wait for auth_ok
            msg = await asyncio.wait_for(self._ws.receive_json(), timeout=_HANDSHAKE_TIMEOUT)
            if msg.get("type") != "auth_ok":
                logger.error("[%s] Auth failed: %s", self.name, msg)
                await self._cleanup_ws()
                return False

            # Step 4: Subscribe to state_changed events
            sub_id = self._next_id()
            await asyncio.wait_for(
                self._ws.send_json({
                    "id": sub_id,
                    "type": "subscribe_events",
                    "event_type": "state_changed",
                }),
                timeout=_HANDSHAKE_TIMEOUT,
            )

            # Verify subscription acknowledgement
            msg = await asyncio.wait_for(self._ws.receive_json(), timeout=_HANDSHAKE_TIMEOUT)
            if not msg.get("success"):
                logger.error("[%s] Failed to subscribe to events: %s", self.name, msg)
                await self._cleanup_ws()
                return False
        except asyncio.TimeoutError:
            # A server that accepts the socket but never responds must not
            # freeze the handshake ladder forever (#67470).
            logger.error(
                "[%s] HA WebSocket auth handshake timed out after %.0fs",
                self.name, _HANDSHAKE_TIMEOUT,
            )
            await self._cleanup_ws()
            return False
        except asyncio.CancelledError:
            # Cancelled mid-handshake (disconnect / watchdog respawn): don't
            # leave the half-authenticated connection dangling.
            await self._cleanup_ws()
            raise
        except Exception as e:
            # Any other handshake failure (send/receive raising a client
            # error, malformed frame, ...) must also tear the connection down
            # here rather than leaking it to a later loop pass (#67470).
            logger.error("[%s] HA WebSocket handshake failed: %s", self.name, e)
            await self._cleanup_ws()
            return False

        return True

    async def _bounded_close(self, closeable: Any, label: str) -> None:
        """Await ``closeable.close()`` bounded by ``_DRAIN_TIMEOUT``.

        A wedged CLOSE-WAIT socket can make ``close()`` hang forever, which
        would otherwise stall the reconnect ladder or ``disconnect()``
        indefinitely. Timeout and any other close-time error are swallowed —
        teardown is best-effort by design. Refs: NousResearch/hermes-agent#67470
        """
        try:
            await asyncio.wait_for(closeable.close(), timeout=_DRAIN_TIMEOUT)
        except asyncio.TimeoutError:
            logger.warning(
                "[%s] %s close timed out after %.0fs; abandoning it",
                self.name, label, _DRAIN_TIMEOUT,
            )
        except Exception as e:
            logger.debug("[%s] %s close failed (non-fatal): %s", self.name, label, e)

    async def _cancel_task_bounded(self, task: Optional["asyncio.Task"], label: str) -> None:
        """Cancel *task* and await it, bounded by ``_DRAIN_TIMEOUT``.

        A truly wedged task can ignore cancellation (blocked in an
        uncancellable await); an unbounded ``await task`` there would hang
        the watchdog or ``disconnect()`` — the very stall this fix removes.
        On timeout the zombie is logged and abandoned: staying deaf is worse
        than leaking one stuck task (#67470).
        """
        if task is None:
            return
        task.cancel()
        # asyncio.wait (not wait_for): wait_for's timeout path cancels the
        # future and then AWAITS that cancellation completing, so a task that
        # swallows CancelledError would hang it — the very stall being fixed.
        # asyncio.wait just observes with a deadline and never raises.
        done, pending = await asyncio.wait({task}, timeout=_DRAIN_TIMEOUT)
        if pending:
            logger.error(
                "[%s] %s did not exit within %.0fs of cancellation; "
                "abandoning it",
                self.name, label, _DRAIN_TIMEOUT,
            )
            return
        for finished in done:
            try:
                finished.result()
            except asyncio.CancelledError:
                pass
            except Exception as e:
                logger.debug(
                    "[%s] %s raised on cancel (non-fatal): %s",
                    self.name, label, e,
                )

    async def _cleanup_ws(self) -> None:
        """Close WebSocket and session, each bounded by ``_DRAIN_TIMEOUT`` so
        one wedged close can't skip the other resource's teardown (#67470)."""
        ws, self._ws = self._ws, None
        if ws is not None and not ws.closed:
            await self._bounded_close(ws, "WebSocket")
        session, self._session = self._session, None
        if session is not None and not session.closed:
            await self._bounded_close(session, "WS session")

    async def disconnect(self) -> None:
        """Disconnect from Home Assistant."""
        self._running = False
        # Watchdog first so it can't respawn the listener mid-teardown; both
        # awaits are bounded so a wedged task can't hang shutdown (#67470).
        await self._cancel_task_bounded(self._watchdog_task, "watchdog task")
        self._watchdog_task = None
        await self._cancel_task_bounded(self._listen_task, "listen task")
        self._listen_task = None

        await self._cleanup_ws()
        if self._rest_session and not self._rest_session.closed:
            await self._bounded_close(self._rest_session, "REST session")
        self._rest_session = None
        logger.info("[%s] Disconnected", self.name)

    # ------------------------------------------------------------------
    # Event listener
    # ------------------------------------------------------------------

    async def _listen_loop(self) -> None:
        """Main event loop with automatic reconnection."""
        backoff_idx = 0

        while self._running:
            # Progress heartbeat for the watchdog (#67470): each pass through
            # the outer loop counts as forward motion even before any event
            # arrives, so a connect that never yields a message still shows
            # up as "alive" rather than immediately tripping the watchdog.
            self._last_progress = time.monotonic()
            try:
                await self._read_events()
            except asyncio.CancelledError:
                return
            except Exception as e:
                logger.warning("[%s] WebSocket error: %s", self.name, e)

            if not self._running:
                return

            # Reconnect with backoff
            delay = self._BACKOFF_STEPS[min(backoff_idx, len(self._BACKOFF_STEPS) - 1)]
            logger.info("[%s] Reconnecting in %ds...", self.name, delay)
            await asyncio.sleep(delay)
            backoff_idx += 1

            try:
                await self._cleanup_ws()
                success = await self._ws_connect()
                if success:
                    backoff_idx = 0  # Reset on successful reconnect
                    logger.info("[%s] Reconnected", self.name)
            except Exception as e:
                logger.warning("[%s] Reconnection failed: %s", self.name, e)

    async def _watchdog_loop(self) -> None:
        """Cause-agnostic watchdog over ``_listen_loop`` (#67470).

        ``_listen_loop`` can wedge on an await with no local bound (e.g. a
        hung aiohttp internals call) and never re-enter its own
        except/reconnect branch. Nothing else observes that stall — the
        process stays alive but the gateway goes silently deaf. Mirrors the
        Telegram adapter's wedged-recovery watchdog: an independent task
        periodically checks ``_last_progress`` and force-recovers when it
        goes stale.
        """
        while self._running:
            try:
                await asyncio.sleep(_WATCHDOG_INTERVAL)
            except asyncio.CancelledError:
                return

            if not self._running:
                return

            stalled_for = time.monotonic() - self._last_progress
            if stalled_for <= _LISTEN_STUCK_TIMEOUT:
                continue

            # Quiet ≠ wedged: aiohttp answers heartbeat PINGs internally, so
            # a healthy HA with no state changes produces no frames for
            # _read_events and looks stalled by progress alone. Probe at the
            # HA protocol layer: send a `ping`; the `pong` comes back as a
            # normal frame, so the (healthy) reader bumps _last_progress and
            # we skip the respawn. A wedged socket/reader can't answer.
            if await self._listener_alive_after_ping():
                continue

            logger.error(
                "[%s] Listen loop wedged for %.0fs with no progress "
                "(HA ping probe unanswered); cancelling and respawning it",
                self.name, stalled_for,
            )

            await self._cancel_task_bounded(self._listen_task, "wedged listen task")

            await self._cleanup_ws()

            if not self._running:
                return

            self._last_progress = time.monotonic()
            self._listen_task = asyncio.create_task(self._listen_loop())

    async def _listener_alive_after_ping(self) -> bool:
        """Send an HA-protocol ping and report whether the reader saw a reply.

        Keeps the single-reader invariant: this never reads the socket — the
        pong arrives through ``_read_events``'s ``async for``, which bumps
        ``_last_progress``. Returns True when progress advanced within
        ``_PING_GRACE`` (listener demonstrably alive), False otherwise
        (#67470 review follow-up).
        """
        ws = self._ws
        if ws is None or ws.closed:
            return False
        probe_start = time.monotonic()
        try:
            await asyncio.wait_for(
                ws.send_json({"id": self._next_id(), "type": "ping"}),
                timeout=_DRAIN_TIMEOUT,
            )
        except asyncio.CancelledError:
            raise
        except Exception:
            return False  # can't even send — treat as wedged
        try:
            await asyncio.sleep(_PING_GRACE)
        except asyncio.CancelledError:
            raise
        return self._last_progress >= probe_start

    async def _read_events(self) -> None:
        """Read events from WebSocket until disconnected."""
        if self._ws is None or self._ws.closed:
            return
        async for ws_msg in self._ws:
            # Any received frame is progress for the watchdog (#67470), not
            # just ones that parse into a state_changed event.
            self._last_progress = time.monotonic()
            if ws_msg.type == aiohttp.WSMsgType.TEXT:
                try:
                    data = json.loads(ws_msg.data)
                    if data.get("type") == "event":
                        await self._handle_ha_event(data.get("event", {}))
                except json.JSONDecodeError:
                    logger.debug("Invalid JSON from HA WS: %s", ws_msg.data[:200])
            elif ws_msg.type in {aiohttp.WSMsgType.CLOSED, aiohttp.WSMsgType.ERROR}:
                break

    async def _handle_ha_event(self, event: Dict[str, Any]) -> None:
        """Process a state_changed event from Home Assistant."""
        event_data = event.get("data", {})
        entity_id: str = event_data.get("entity_id", "")

        if not entity_id:
            return

        # Apply ignore filter
        if entity_id in self._ignore_entities:
            return

        # Apply domain/entity watch filters (closed by default — require
        # explicit watch_domains, watch_entities, or watch_all to forward)
        domain = entity_id.split(".")[0] if "." in entity_id else ""
        if self._watch_domains or self._watch_entities:
            domain_match = domain in self._watch_domains if self._watch_domains else False
            entity_match = entity_id in self._watch_entities if self._watch_entities else False
            if not domain_match and not entity_match:
                return
        elif not self._watch_all:
            # No filters configured and watch_all is off — drop the event
            return

        # Apply cooldown
        now = time.time()
        last = self._last_event_time.get(entity_id, 0)
        if (now - last) < self._cooldown_seconds:
            return
        self._last_event_time[entity_id] = now

        # Build human-readable message
        old_state = event_data.get("old_state", {})
        new_state = event_data.get("new_state", {})
        message = self._format_state_change(entity_id, old_state, new_state)

        if not message:
            return

        # Build MessageEvent and forward to handler
        source = self.build_source(
            chat_id="ha_events",
            chat_name="Home Assistant Events",
            chat_type="channel",
            user_id="homeassistant",
            user_name="Home Assistant",
        )

        msg_event = MessageEvent(
            text=message,
            message_type=MessageType.TEXT,
            source=source,
            message_id=f"ha_{entity_id}_{int(now)}",
            timestamp=datetime.now(),
        )

        await self.handle_message(msg_event)

    @staticmethod
    def _format_state_change(
        entity_id: str,
        old_state: Dict[str, Any],
        new_state: Dict[str, Any],
    ) -> Optional[str]:
        """Convert a state_changed event into a human-readable description."""
        if not new_state:
            return None

        old_val = old_state.get("state", "unknown") if old_state else "unknown"
        new_val = new_state.get("state", "unknown")

        # Skip if state didn't actually change
        if old_val == new_val:
            return None

        friendly_name = new_state.get("attributes", {}).get("friendly_name", entity_id)
        domain = entity_id.split(".")[0] if "." in entity_id else ""

        # Domain-specific formatting
        if domain == "climate":
            attrs = new_state.get("attributes", {})
            temp = attrs.get("current_temperature", "?")
            target = attrs.get("temperature", "?")
            return (
                f"[Home Assistant] {friendly_name}: HVAC mode changed from "
                f"'{old_val}' to '{new_val}' (current: {temp}, target: {target})"
            )

        if domain == "sensor":
            unit = new_state.get("attributes", {}).get("unit_of_measurement", "")
            return (
                f"[Home Assistant] {friendly_name}: changed from "
                f"{old_val}{unit} to {new_val}{unit}"
            )

        if domain == "binary_sensor":
            return (
                f"[Home Assistant] {friendly_name}: "
                f"{'triggered' if new_val == 'on' else 'cleared'} "
                f"(was {'triggered' if old_val == 'on' else 'cleared'})"
            )

        if domain in {"light", "switch", "fan"}:
            return (
                f"[Home Assistant] {friendly_name}: turned "
                f"{'on' if new_val == 'on' else 'off'}"
            )

        if domain == "alarm_control_panel":
            return (
                f"[Home Assistant] {friendly_name}: alarm state changed from "
                f"'{old_val}' to '{new_val}'"
            )

        # Generic fallback
        return (
            f"[Home Assistant] {friendly_name} ({entity_id}): "
            f"changed from '{old_val}' to '{new_val}'"
        )

    # ------------------------------------------------------------------
    # Outbound messaging
    # ------------------------------------------------------------------

    async def send(
        self,
        chat_id: str,
        content: str,
        reply_to: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SendResult:
        """Send a notification via HA REST API (persistent_notification.create).

        Uses the REST API instead of WebSocket to avoid a race condition
        with the event listener loop that reads from the same WS connection.
        """
        url = f"{self._hass_url}/api/services/persistent_notification/create"
        headers = {
            "Authorization": f"Bearer {self._hass_token}",
            "Content-Type": "application/json",
        }
        payload = {
            "title": "Hermes Agent",
            "message": content[:self.MAX_MESSAGE_LENGTH],
        }

        try:
            if self._rest_session:
                async with self._rest_session.post(
                    url,
                    headers=headers,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=10),
                ) as resp:
                    if resp.status < 300:
                        return SendResult(success=True, message_id=uuid.uuid4().hex[:12])
                    else:
                        body = await resp.text()
                        return SendResult(success=False, error=f"HTTP {resp.status}: {body}")
            else:
                async with aiohttp.ClientSession() as session:
                    async with session.post(
                        url,
                        headers=headers,
                        json=payload,
                        timeout=aiohttp.ClientTimeout(total=10),
                    ) as resp:
                        if resp.status < 300:
                            return SendResult(success=True, message_id=uuid.uuid4().hex[:12])
                        else:
                            body = await resp.text()
                            return SendResult(success=False, error=f"HTTP {resp.status}: {body}")

        except asyncio.TimeoutError:
            return SendResult(success=False, error="Timeout sending notification to HA")
        except Exception as e:
            return SendResult(success=False, error=str(e))

    async def send_typing(self, chat_id: str, metadata=None) -> None:
        """No typing indicator for Home Assistant."""

    async def get_chat_info(self, chat_id: str) -> Dict[str, Any]:
        """Return basic info about the HA event channel."""
        return {
            "name": "Home Assistant Events",
            "type": "channel",
            "url": self._hass_url,
        }


# ---------------------------------------------------------------------------
# Standalone (out-of-process) sender — used by cron deliver=homeassistant
# ---------------------------------------------------------------------------


async def _standalone_send(
    pconfig,
    chat_id: str,
    message: str,
    *,
    thread_id: Optional[str] = None,
    media_files: Optional[list] = None,
    force_document: bool = False,
) -> Dict[str, Any]:
    """Send a notification via the HA ``notify.notify`` service without a
    live gateway adapter.

    Used by ``tools/send_message_tool._send_via_adapter`` when the gateway
    runner is not in this process (typical for cron jobs running
    out-of-process).  The HTTP path is the same one the legacy
    ``_send_homeassistant`` helper used in ``tools/send_message_tool.py``
    before this migration.

    Reads ``HASS_TOKEN`` from ``pconfig.token`` (set by the gateway config
    loader from env) and falls back to the ``HASS_TOKEN`` env var.  Server
    URL comes from ``pconfig.extra["url"]`` (seeded by the env loader in
    ``gateway/config.py``) or the ``HASS_URL`` env var.

    ``thread_id``, ``media_files`` and ``force_document`` are accepted for
    signature parity with other standalone senders.  HA notifications have
    no native threading or attachment model — these arguments are ignored.
    """
    if not AIOHTTP_AVAILABLE:
        return {"error": "aiohttp not installed. Run: pip install aiohttp"}

    extra = getattr(pconfig, "extra", {}) or {}
    hass_url = (extra.get("url") or os.getenv("HASS_URL", "")).rstrip("/")
    token = (getattr(pconfig, "token", None) or os.getenv("HASS_TOKEN", "")).strip()
    if not hass_url or not token:
        return {
            "error": (
                "Home Assistant standalone send: HASS_URL and HASS_TOKEN "
                "must both be set"
            )
        }

    url = f"{hass_url}/api/services/notify/notify"
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }
    payload = {"message": message, "target": chat_id}

    try:
        async with aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=30)
        ) as session:
            async with session.post(url, headers=headers, json=payload) as resp:
                if resp.status not in {200, 201}:
                    body = await resp.text()
                    return {
                        "error": (
                            f"Home Assistant API error ({resp.status}): {body}"
                        )
                    }
        return {
            "success": True,
            "platform": "homeassistant",
            "chat_id": chat_id,
        }
    except asyncio.TimeoutError:
        return {"error": "Timeout sending notification to Home Assistant"}
    except Exception as e:
        return {"error": f"Home Assistant send failed: {e}"}


# ---------------------------------------------------------------------------
# is_connected probe
# ---------------------------------------------------------------------------


def _is_connected(config) -> bool:
    """Home Assistant is considered connected when ``HASS_TOKEN`` is set.

    Looks up via ``hermes_cli.gateway.get_env_value`` at call time (not via
    the plugin's own bound import) so tests that patch
    ``gateway_mod.get_env_value`` can suppress ambient ``HASS_TOKEN`` env
    vars.  Matches what the legacy connected-platforms check did before
    this migration.
    """
    import hermes_cli.gateway as gateway_mod
    return bool((gateway_mod.get_env_value("HASS_TOKEN") or "").strip())


# ---------------------------------------------------------------------------
# Plugin registration entry point
# ---------------------------------------------------------------------------


def _build_adapter(config):
    """Factory wrapper that constructs HomeAssistantAdapter from a PlatformConfig."""
    return HomeAssistantAdapter(config)


def register(ctx) -> None:
    """Plugin entry point — called by the Hermes plugin system."""
    ctx.register_platform(
        name="homeassistant",
        label="Home Assistant",
        adapter_factory=_build_adapter,
        check_fn=check_ha_requirements,
        validate_config=validate_ha_config,
        is_connected=_is_connected,
        required_env=["HASS_TOKEN"],
        install_hint="pip install aiohttp",
        # Out-of-process cron delivery via the HA ``notify.notify`` service.
        # Without this hook, ``deliver=homeassistant`` cron jobs would fail
        # with "No live adapter" when cron runs separately from the gateway.
        # Mirrors the Discord / Teams / Mattermost pattern.
        standalone_sender_fn=_standalone_send,
        # HA notification message cap — matches MAX_MESSAGE_LENGTH on the
        # adapter class above.
        max_message_length=HomeAssistantAdapter.MAX_MESSAGE_LENGTH,
        # Display
        emoji="🏠",
        allow_update_command=True,
    )
