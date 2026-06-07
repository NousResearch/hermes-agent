"""
MQTT platform adapter.

Connects to an MQTT broker, subscribes to a configurable topic allowlist, and
forwards each inbound message as a MessageEvent. Outbound messages publish to
the topic provided as chat_id. Built for Phoebe's existing MQTT mesh
(Atlas broker at 192.168.1.212:1883, auth required, optional TLS on 8883).

Requires:
- paho-mqtt (already in messaging extras)
- MQTT_USER / MQTT_PASSWORD env vars (or set in PlatformConfig.extra)
- MQTT_BROKER env var (default: 192.168.1.212)
- MQTT_CA_CERT env var (optional — flips port to 8883 + TLS when set)
"""

import asyncio
import logging
import os
import pathlib
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Set

try:
    import paho.mqtt.client as mqtt
    PAHO_AVAILABLE = True
except ImportError:
    PAHO_AVAILABLE = False
    mqtt = None  # type: ignore[assignment]

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import (
    BasePlatformAdapter,
    MessageEvent,
    MessageType,
    SendResult,
)

logger = logging.getLogger(__name__)


# Default topic allowlist if config doesn't specify one. Built from the
# pre-Helios integration audit (project_pre_helios_integration_audit.md) —
# the P0/P1 feeds Hermes-Phoebe needs to see to be on the mesh.
_DEFAULT_WATCH_TOPICS: List[str] = [
    "phoebe/atlas/speak",
    "phoebe/orion/think_signal",
    "phoebe/orion/alert",
    "phoebe/sentinel/speak",
    "phoebe/security/status",
    "phoebe/presence/state",
    "phoebe/banshee/desk_presence",
    "phoebe/luna/first_seen",
    "phoebe/luna/face_confirmed",
    "phoebe/castor/episode_tagged",
    "phoebe/castor/facts_extracted",
    "phoebe/wake",
]


def check_mqtt_requirements() -> bool:
    """Check if MQTT dependencies are available and configured."""
    if not PAHO_AVAILABLE:
        return False
    # Need at minimum a broker + credentials. Broker can default; user/pass must be set.
    if not os.getenv("MQTT_USER"):
        return False
    if not os.getenv("MQTT_PASSWORD"):
        return False
    return True


class MQTTAdapter(BasePlatformAdapter):
    """
    MQTT broker adapter.

    Subscribes to a configurable topic allowlist and forwards each message as a
    MessageEvent. Outbound `send(chat_id, content)` publishes `content` to the
    topic named by `chat_id`. Reconnection is handled by paho-mqtt's built-in
    `reconnect_delay_set` + `loop_start` background thread.
    """

    MAX_MESSAGE_LENGTH = 268435455  # MQTT 3.1.1 max payload (256 MB) — effectively unbounded

    _BACKOFF_MIN = 5
    _BACKOFF_MAX = 60

    def __init__(self, config: PlatformConfig):
        super().__init__(config, Platform.MQTT)

        extra = config.extra or {}

        # Broker connection details (env > extra > default)
        self._broker_host: str = (
            os.getenv("MQTT_BROKER")
            or extra.get("broker_host")
            or "192.168.1.212"
        )
        self._username: str = config.token or os.getenv("MQTT_USER") or extra.get("username", "")
        self._password: str = os.getenv("MQTT_PASSWORD") or extra.get("password", "")

        # TLS — port 8883 if CA cert set, plain 1883 otherwise
        self._ca_cert: str = os.getenv("MQTT_CA_CERT") or extra.get("ca_cert", "")
        if self._ca_cert and os.path.exists(self._ca_cert):
            self._broker_port: int = int(extra.get("broker_port", 8883))
            self._tls_enabled = True
        else:
            self._broker_port = int(extra.get("broker_port", 1883))
            self._tls_enabled = False

        # Subscribe list — default to audited P0/P1 topics, override via extra["watch_topics"]
        watch = extra.get("watch_topics")
        self._watch_topics: List[str] = (
            list(watch) if isinstance(watch, (list, tuple)) and watch else list(_DEFAULT_WATCH_TOPICS)
        )

        # Topics to never forward (e.g., own published topics to avoid loops)
        ignore = extra.get("ignore_topics", [])
        self._ignore_topics: Set[str] = set(ignore) if isinstance(ignore, (list, tuple, set)) else set()
        # Always ignore our own publish prefix so we don't loop on send()
        self._self_topic_prefix: str = extra.get("self_topic_prefix", "phoebe/hermes/")

        # Per-topic cooldown to prevent floods (chatty topics like presence/state
        # can fire many times per second; default 30s throttle is a reasonable
        # ceiling for inference cost without missing event semantics).
        self._cooldown_seconds: float = float(extra.get("cooldown_seconds", 30))
        self._last_event_time: Dict[str, float] = {}

        # Observational mode: append inbound events to a log file instead of
        # invoking the agent loop per event. MQTT events are observations, not
        # chat turns — auto-invoking inference for every banshee/presence tick
        # is wasted compute. With observational mode on, events accumulate in
        # the log file and the agent reads them via filesystem/MemPalace tools
        # when it actually wakes up. Toggle off if you want the legacy
        # agent-invoking behavior (with send()-suppression preventing the loop).
        self._observational: bool = bool(extra.get("observational", True))
        log_default = (
            pathlib.Path.home()
            / "PhoebeVault"
            / "Devlog"
            / f"mqtt-stream-{datetime.now().strftime('%Y-%m')}.md"
        )
        self._observe_log_path: Optional[pathlib.Path] = pathlib.Path(
            extra.get("observe_log_path", str(log_default))
        )

        # Client identity for the broker
        self._client_id: str = extra.get("client_id", "phoebe-hermes")
        self._keepalive: int = int(extra.get("keepalive", 60))

        # Connection state
        self._client: Optional["mqtt.Client"] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._connected = False

    # ------------------------------------------------------------------
    # Connection lifecycle
    # ------------------------------------------------------------------

    async def connect(self) -> bool:
        """Connect to broker, subscribe to topics, start background loop."""
        if not PAHO_AVAILABLE:
            logger.warning("[%s] paho-mqtt not installed. Run: pip install paho-mqtt", self.name)
            return False
        if not self._username or not self._password:
            logger.warning("[%s] MQTT_USER and MQTT_PASSWORD must be configured", self.name)
            return False

        # Capture the running event loop so MQTT-thread callbacks can dispatch back into async
        try:
            self._loop = asyncio.get_running_loop()
        except RuntimeError:
            logger.error("[%s] connect() must be called from an async context", self.name)
            return False

        try:
            self._client = mqtt.Client(
                client_id=self._client_id,
                clean_session=False,
                protocol=mqtt.MQTTv311,
            )
            self._client.username_pw_set(self._username, self._password)
            if self._tls_enabled:
                self._client.tls_set(ca_certs=self._ca_cert)
                logger.info("[%s] TLS enabled (CA: %s)", self.name, self._ca_cert)

            # paho's built-in exponential backoff
            self._client.reconnect_delay_set(min_delay=self._BACKOFF_MIN, max_delay=self._BACKOFF_MAX)

            self._client.on_connect = self._on_connect
            self._client.on_disconnect = self._on_disconnect
            self._client.on_message = self._on_message

            # Connect synchronously; loop_start spins the network thread
            self._client.connect(self._broker_host, self._broker_port, self._keepalive)
            self._client.loop_start()

            self._running = True
            logger.info(
                "[%s] Connecting to %s:%d as %s (%d topics)",
                self.name, self._broker_host, self._broker_port, self._username, len(self._watch_topics),
            )
            return True
        except Exception as e:
            logger.error("[%s] Failed to connect: %s", self.name, e)
            return False

    async def disconnect(self) -> None:
        """Stop the background loop and close the connection."""
        self._running = False
        if self._client is not None:
            try:
                self._client.loop_stop()
                self._client.disconnect()
            except Exception as e:
                logger.warning("[%s] Disconnect error: %s", self.name, e)
        self._connected = False
        logger.info("[%s] Disconnected", self.name)

    # ------------------------------------------------------------------
    # paho callbacks — run in the MQTT network thread
    # ------------------------------------------------------------------

    def _on_connect(self, client, userdata, flags, rc, properties=None):
        if rc != 0:
            logger.warning("[%s] Broker rejected connection, rc=%s", self.name, rc)
            return
        self._connected = True
        # Subscribe to each watch topic at QoS 1
        for topic in self._watch_topics:
            try:
                client.subscribe(topic, qos=1)
            except Exception as e:
                logger.warning("[%s] Subscribe failed for %s: %s", self.name, topic, e)
        logger.info(
            "[%s] Connected to %s:%d, subscribed to %d topics",
            self.name, self._broker_host, self._broker_port, len(self._watch_topics),
        )

    def _on_disconnect(self, client, userdata, *args):
        self._connected = False
        # paho will auto-reconnect via reconnect_delay_set; just log
        logger.info("[%s] Broker disconnected (will auto-reconnect)", self.name)

    def _on_message(self, client, userdata, msg):
        """Forward inbound MQTT message — observational by default."""
        try:
            topic = msg.topic or ""
            if not topic:
                return
            if topic in self._ignore_topics:
                return
            if self._self_topic_prefix and topic.startswith(self._self_topic_prefix):
                # Don't echo our own publishes back to ourselves
                return

            # Cooldown to prevent floods on chatty topics
            if self._cooldown_seconds > 0:
                now = time.time()
                last = self._last_event_time.get(topic, 0.0)
                if (now - last) < self._cooldown_seconds:
                    return
                self._last_event_time[topic] = now

            # Decode payload — MQTT carries bytes; assume UTF-8 text content for Phoebe topics
            try:
                payload_text = msg.payload.decode("utf-8", errors="replace")
            except Exception:
                payload_text = repr(msg.payload)

            if self._observational:
                # Observational mode: append to log file, don't invoke agent loop
                self._log_event(topic, payload_text)
                return

            # Legacy: build MessageEvent + dispatch into async agent loop
            source = self.build_source(
                chat_id=topic,
                chat_name=topic,
                chat_type="channel",
                user_id="mqtt",
                user_name=topic.split("/")[1] if "/" in topic else "mqtt",
            )
            event = MessageEvent(
                text=payload_text,
                message_type=MessageType.TEXT,
                source=source,
                message_id=f"mqtt_{topic}_{int(time.time() * 1000)}",
                timestamp=datetime.now(),
            )

            if self._loop is not None and self._loop.is_running():
                asyncio.run_coroutine_threadsafe(self.handle_message(event), self._loop)
        except Exception as e:
            logger.warning("[%s] _on_message error: %s", self.name, e)

    def _log_event(self, topic: str, payload: str) -> None:
        """Append an MQTT event to the observational log file.

        Format: one bullet per event with ISO timestamp + topic + payload.
        Long payloads are truncated to 500 chars to keep the log readable.
        The file is created on first write; parent directory must exist
        (PhoebeVault is scaffolded ahead of adapter use).
        """
        if not self._observe_log_path:
            return
        try:
            self._observe_log_path.parent.mkdir(parents=True, exist_ok=True)
            ts = datetime.now().isoformat(timespec="seconds")
            payload_short = payload[:500] + ("..." if len(payload) > 500 else "")
            # Strip newlines so each event is one bullet
            payload_one_line = payload_short.replace("\n", " ").replace("\r", " ")
            with self._observe_log_path.open("a", encoding="utf-8") as f:
                f.write(f"- `{ts}` **{topic}** {payload_one_line}\n")
        except Exception as e:
            logger.warning("[%s] log write failed: %s", self.name, e)

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
        """Read-mostly outbound. Most MQTT chat_id values are SOURCE topics
        (the topic an event came in on), and publishing back to them creates a
        feedback loop because the broker echoes the publish to all subscribers
        (including this adapter). The chat-turn semantics that Hermes assumes
        — "respond on the same channel" — are wrong for an event bus.

        Behavior: only publishes to topics under `self_topic_prefix`
        (default `phoebe/hermes/`). Everything else is suppressed silently with
        a successful SendResult so the agent loop doesn't see a fake failure.

        Intentional outbound (e.g., a future command-topic for orion control)
        should be done via a direct paho-mqtt client in a tool/plugin, NOT via
        this adapter's send(). This adapter is the INGEST surface.
        """
        topic = chat_id or ""
        if not topic:
            return SendResult(success=False, error="Empty topic (chat_id)")

        # Suppress publishes outside our designated outbound namespace.
        # This is the feedback-loop guard — MQTT events are observations,
        # not chat turns, so we should not auto-respond on the source topic.
        if not topic.startswith(self._self_topic_prefix):
            logger.debug(
                "[%s] Suppressed response publish to %s (read-mostly adapter)",
                self.name, topic,
            )
            return SendResult(success=True, message_id=f"suppressed_{topic}")

        # Allow phoebe/hermes/* publishes (intentional outbound only).
        if not self._client or not self._connected:
            return SendResult(success=False, error="MQTT client not connected")

        try:
            qos = int((metadata or {}).get("qos", 1))
            retain = bool((metadata or {}).get("retain", False))
            info = self._client.publish(topic, content, qos=qos, retain=retain)
            # paho returns immediately; rc != MQTT_ERR_SUCCESS means local-queue failure
            if info.rc != mqtt.MQTT_ERR_SUCCESS:
                return SendResult(success=False, error=f"publish rc={info.rc}")
            return SendResult(success=True, message_id=f"{topic}_{info.mid}")
        except Exception as e:
            return SendResult(success=False, error=str(e))

    async def send_typing(self, chat_id: str, metadata=None) -> None:
        """No typing indicator over MQTT."""

    async def get_chat_info(self, chat_id: str) -> Dict[str, Any]:
        """Return basic info about an MQTT topic 'chat'."""
        return {
            "name": chat_id,
            "type": "channel",
            "chat_id": chat_id,
            "broker": f"{self._broker_host}:{self._broker_port}",
        }
