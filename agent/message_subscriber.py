"""
Hermes Message Subscriber — MQTT listener for the orchestrator (Phoenix).

Subscribes to:
  hermes/+/+/phoenix     — events directed at Phoenix
  hermes/+/+/broadcast   — global broadcasts

When MQTT delivers a message, triage logic decides:
  - DROP: resolve immediately, no action needed
  - DELIVER: urgent, needs Phoenix attention in real-time
  - HOLD: store in pending queue for next Phoenix session

All messages are already durably stored in SQLite by the publisher.
"""
from __future__ import annotations

import json
import logging
import threading
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)

BROKER = "127.0.0.1"
PORT = 1883
KEEPALIVE = 15

# Topics Phoenix listens to
SUBSCRIBE_TOPICS = [
    ("hermes/+/+/phoenix", 1),
    ("hermes/+/+/broadcast", 1),
]

_HAS_PAHO: bool = False
try:
    import paho.mqtt.client as mqtt
    _HAS_PAHO = True
except Exception:
    logger.warning("paho-mqtt not available; message subscriber disabled")


# Threading state
_subscriber_thread: threading.Thread | None = None
_client: Any = None
_shutdown_event = threading.Event()
_handler: Optional[Callable[[Dict[str, Any]], None]] = None

# Pending escalations that need active Phoenix attention
_pending_lock = threading.Lock()
_pending_escalations: List[Dict[str, Any]] = []


def get_pending_escalations(clear: bool = True) -> List[Dict[str, Any]]:
    """
    Return pending escalations that need Phoenix attention.
    Called by the gateway or agent turn logic to surface unresolved events.
    If clear=True, empties the queue after returning.
    """
    with _pending_lock:
        copy = list(_pending_escalations)
        if clear:
            _pending_escalations.clear()
        return copy


def _resolve_event(event_id: int, status: str, resolution_time_ms: int = 0) -> None:
    """Resolve an event in the SQLite event log."""
    try:
        import hermes_event_bus as bus
        bus.resolve_event(event_id, status=status, resolution_time_ms=resolution_time_ms)
    except Exception as exc:
        logger.debug("Failed to resolve event %s: %s", event_id, exc)


def _triage_event(payload: Dict[str, Any]) -> str:
    """
    Triage an incoming event.

    Returns one of: DROP, DELIVER, HOLD
      DROP   — resolve immediately, no action needed (known noise, low priority)
      DELIVER — urgent, needs immediate attention (priority 1, auth failures)
      HOLD   — queue for next Phoenix session (priority 2-3 tool failures)
    """
    priority = payload.get("priority", 5)
    category = payload.get("_category", "system")
    event_type = payload.get("event_type", "unknown")
    event_id = payload.get("event_id", 0)
    source = payload.get("_source", "?")
    recurrence_hash = payload.get("recurrence_hash", "")

    # Priority 1: auth/permission/credential failures — always DELIVER
    if priority <= 1:
        return "DELIVER"

    # Category-specific rules
    if category == "guardrail" and event_type == "tool_failure":
        # Priority 2: programming errors (AttributeError, TypeError, etc.)
        # These are usually one-off bugs. Resolve and HOLD for batch review.
        _resolve_event(event_id, status="resolved", resolution_time_ms=0)
        return "HOLD"

    if category == "system" and event_type == "cron_output":
        # Cron outputs: resolve immediately unless they are failure alerts
        inner = payload.get("payload", {})
        content = inner.get("content", "")
        is_failure = content and content.lstrip().startswith("⚠")
        if is_failure:
            return "DELIVER"
        _resolve_event(event_id, status="resolved", resolution_time_ms=0)
        return "DROP"

    if category == "lifecycle":
        # Agent lifecycle events (start, stop, crash) — HOLD for review
        _resolve_event(event_id, status="resolved", resolution_time_ms=0)
        return "HOLD"

    # Default: medium priority, just log and resolve
    _resolve_event(event_id, status="resolved", resolution_time_ms=0)
    return "DROP"


def _phoenix_triage_handler(payload: Dict[str, Any]) -> None:
    """
    Main handler for Phoenix. Triage every incoming MQTT message.
    """
    category = payload.get("_category", "?")
    event_type = payload.get("event_type", "?")
    source = payload.get("_source", "?")
    target = payload.get("_target", "?")
    priority = payload.get("priority", "?")
    event_id = payload.get("event_id", 0)
    recurrence_hash = payload.get("recurrence_hash", "?")

    decision = _triage_event(payload)

    if decision == "DROP":
        logger.info(
            "[Bus DROP] %s/%s from %s (id=%s, hash=%s)",
            category, event_type, source, event_id, recurrence_hash,
        )
        return

    if decision == "DELIVER":
        logger.warning(
            "[Bus URGENT] %s/%s from %s (prio=%s, id=%s, hash=%s) — needs immediate attention",
            category, event_type, source, priority, event_id, recurrence_hash,
        )
        with _pending_lock:
            _pending_escalations.append(payload)
        return

    if decision == "HOLD":
        logger.info(
            "[Bus HOLD] %s/%s from %s (prio=%s, id=%s, hash=%s) — queued for review",
            category, event_type, source, priority, event_id, recurrence_hash,
        )
        with _pending_lock:
            _pending_escalations.append(payload)
        return


def _on_connect(client, userdata, flags, rc, properties=None):
    if rc == 0:
        logger.info("Message subscriber connected to MQTT broker")
        for topic, qos in SUBSCRIBE_TOPICS:
            client.subscribe(topic, qos=qos)
            logger.info("Subscribed to %s (qos=%s)", topic, qos)
    else:
        logger.error("Message subscriber connect failed, rc=%s", rc)


def _on_message(client, userdata, msg):
    try:
        payload = json.loads(msg.payload.decode("utf-8"))
        topic = msg.topic
        # Inject topic metadata for handler context
        payload["_topic"] = topic
        parts = topic.split("/")
        if len(parts) == 4:
            payload["_category"] = parts[1]
            payload["_source"] = parts[2]
            payload["_target"] = parts[3]

        handler = _handler or _phoenix_triage_handler
        try:
            handler(payload)
        except Exception as exc:
            logger.exception("Message handler failed for %s: %s", topic, exc)
    except Exception as exc:
        logger.exception("Failed to handle message on %s: %s", msg.topic, exc)


def _on_disconnect(client, userdata, disconnect_flags, rc, properties=None):
    logger.warning("Message subscriber disconnected, rc=%s", rc)


def start_message_subscriber(
    handler: Optional[Callable[[Dict[str, Any]], None]] = None,
) -> bool:
    """
    Start the MQTT message subscriber in a background daemon thread.

    Args:
        handler: Optional override callback. If None, uses the built-in
                 Phoenix triage handler.

    Returns:
        True if started, False if paho-mqtt is missing or broker unreachable.
    """
    global _subscriber_thread, _client, _handler, _shutdown_event

    if not _HAS_PAHO:
        logger.warning("Message subscriber: paho-mqtt not available")
        return False

    if _subscriber_thread is not None and _subscriber_thread.is_alive():
        logger.debug("Message subscriber already running")
        return True

    _handler = handler
    _shutdown_event.clear()

    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
    client.on_connect = _on_connect
    client.on_message = _on_message
    client.on_disconnect = _on_disconnect

    try:
        client.connect(BROKER, PORT, KEEPALIVE)
    except Exception as exc:
        logger.warning("Message subscriber: broker unreachable (%s)", exc)
        return False

    def _run():
        client.loop_start()
        logger.info("Message subscriber thread started")
        _shutdown_event.wait()
        client.loop_stop()
        client.disconnect()
        logger.info("Message subscriber thread stopped")

    _client = client
    _subscriber_thread = threading.Thread(
        target=_run, daemon=True, name="hermes-message-subscriber"
    )
    _subscriber_thread.start()
    return True


def stop_message_subscriber() -> None:
    """Signal the message subscriber to stop."""
    _shutdown_event.set()
