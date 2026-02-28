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
from datetime import datetime
from typing import Any, Dict, List, Optional, Set

try:
    import aiohttp
    AIOHTTP_AVAILABLE = True
except ImportError:
    AIOHTTP_AVAILABLE = False
    aiohttp = None  # type: ignore[assignment]

import sys
from pathlib import Path as _Path
sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import (
    BasePlatformAdapter,
    MessageEvent,
    MessageType,
    SendResult,
)

logger = logging.getLogger(__name__)


def check_ha_requirements() -> bool:
    """Check if Home Assistant dependencies are available and configured."""
    if not AIOHTTP_AVAILABLE:
        return False
    if not os.getenv("HASS_TOKEN"):
        return False
    return True


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
        self._listen_task: Optional[asyncio.Task] = None
        self._msg_id: int = 0

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

    async def connect(self) -> bool:
        """Connect to HA WebSocket API and subscribe to events."""
        if not AIOHTTP_AVAILABLE:
            print(f"[{self.name}] aiohttp not installed. Run: pip install aiohttp")
            return False

        if not self._hass_token:
            print(f"[{self.name}] No HASS_TOKEN configured")
            return False

        try:
            success = await self._ws_connect()
            if not success:
                return False

            # Start background listener
            self._listen_task = asyncio.create_task(self._listen_loop())
            self._running = True
            print(f"[{self.name}] Connected to {self._hass_url}")
            return True

        except Exception as e:
            print(f"[{self.name}] Failed to connect: {e}")
            return False

    async def _ws_connect(self) -> bool:
        """Establish WebSocket connection and authenticate."""
        ws_url = self._hass_url.replace("http://", "ws://").replace("https://", "wss://")
        ws_url = f"{ws_url}/api/websocket"

        self._session = aiohttp.ClientSession()
        self._ws = await self._session.ws_connect(ws_url, heartbeat=30)

        # Step 1: Receive auth_required
        msg = await self._ws.receive_json()
        if msg.get("type") != "auth_required":
            logger.error("Expected auth_required, got: %s", msg.get("type"))
            await self._cleanup_ws()
            return False

        # Step 2: Send auth
        await self._ws.send_json({
            "type": "auth",
            "access_token": self._hass_token,
        })

        # Step 3: Wait for auth_ok
        msg = await self._ws.receive_json()
        if msg.get("type") != "auth_ok":
            logger.error("Auth failed: %s", msg)
            await self._cleanup_ws()
            return False

        # Step 4: Subscribe to state_changed events
        sub_id = self._next_id()
        await self._ws.send_json({
            "id": sub_id,
            "type": "subscribe_events",
            "event_type": "state_changed",
        })

        # Verify subscription acknowledgement
        msg = await self._ws.receive_json()
        if not msg.get("success"):
            logger.error("Failed to subscribe to events: %s", msg)
            await self._cleanup_ws()
            return False

        return True

    async def _cleanup_ws(self) -> None:
        """Close WebSocket and session."""
        if self._ws and not self._ws.closed:
            await self._ws.close()
        self._ws = None
        if self._session and not self._session.closed:
            await self._session.close()
        self._session = None

    async def disconnect(self) -> None:
        """Disconnect from Home Assistant."""
        self._running = False
        if self._listen_task:
            self._listen_task.cancel()
            try:
                await self._listen_task
            except asyncio.CancelledError:
                pass
            self._listen_task = None

        await self._cleanup_ws()
        print(f"[{self.name}] Disconnected")

    # ------------------------------------------------------------------
    # Event listener
    # ------------------------------------------------------------------

    async def _listen_loop(self) -> None:
        """Main event loop with automatic reconnection."""
        backoff_idx = 0

        while self._running:
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
            print(f"[{self.name}] Reconnecting in {delay}s...")
            await asyncio.sleep(delay)
            backoff_idx += 1

            try:
                await self._cleanup_ws()
                success = await self._ws_connect()
                if success:
                    backoff_idx = 0  # Reset on successful reconnect
                    print(f"[{self.name}] Reconnected")
            except Exception as e:
                logger.warning("[%s] Reconnection failed: %s", self.name, e)

    async def _read_events(self) -> None:
        """Read events from WebSocket until disconnected."""
        async for ws_msg in self._ws:
            if ws_msg.type == aiohttp.WSMsgType.TEXT:
                try:
                    data = json.loads(ws_msg.data)
                    if data.get("type") == "event":
                        await self._handle_ha_event(data.get("event", {}))
                except json.JSONDecodeError:
                    logger.debug("Invalid JSON from HA WS: %s", ws_msg.data[:200])
            elif ws_msg.type in (aiohttp.WSMsgType.CLOSED, aiohttp.WSMsgType.ERROR):
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

        # Apply domain/entity watch filters
        domain = entity_id.split(".")[0] if "." in entity_id else ""
        if self._watch_domains or self._watch_entities:
            domain_match = domain in self._watch_domains if self._watch_domains else False
            entity_match = entity_id in self._watch_entities if self._watch_entities else False
            if not domain_match and not entity_match:
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

        if domain in ("light", "switch", "fan"):
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
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    url,
                    headers=headers,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=10),
                ) as resp:
                    if resp.status < 300:
                        return SendResult(success=True, message_id=str(self._next_id()))
                    else:
                        body = await resp.text()
                        return SendResult(success=False, error=f"HTTP {resp.status}: {body}")

        except asyncio.TimeoutError:
            return SendResult(success=False, error="Timeout sending notification to HA")
        except Exception as e:
            return SendResult(success=False, error=str(e))

    async def send_typing(self, chat_id: str) -> None:
        """No typing indicator for Home Assistant."""
        pass

    async def get_chat_info(self, chat_id: str) -> Dict[str, Any]:
        """Return basic info about the HA event channel."""
        return {
            "name": "Home Assistant Events",
            "type": "channel",
            "url": self._hass_url,
        }
