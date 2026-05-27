"""Periodic gateway runtime-status heartbeat updates.

Keeps ``gateway_state.json`` fresh during otherwise idle gateway periods so
cross-container health checks can distinguish "alive but idle" from "stale".
"""

from __future__ import annotations

import logging
import threading
from typing import Optional

from gateway.status import write_runtime_status

logger = logging.getLogger(__name__)

_heartbeat_thread: Optional[threading.Thread] = None
_stop_event: Optional[threading.Event] = None
_interval_seconds: float = 60.0
_lock = threading.Lock()


def heartbeat_once() -> None:
    """Refresh gateway_state.json without changing any semantic fields."""
    write_runtime_status()


def _heartbeat_loop(stop_event: threading.Event, interval: float) -> None:
    while not stop_event.wait(interval):
        try:
            heartbeat_once()
        except Exception as exc:
            logger.debug("Status heartbeat tick failed: %s", exc)


def start_status_heartbeat(interval_seconds: float = 60.0) -> bool:
    """Start a daemon thread that periodically refreshes runtime status."""
    global _heartbeat_thread, _stop_event, _interval_seconds

    with _lock:
        if _heartbeat_thread is not None and _heartbeat_thread.is_alive():
            return False

        _interval_seconds = float(interval_seconds)
        _stop_event = threading.Event()

        # Emit an immediate baseline tick so fresh gateways start with a
        # current timestamp instead of waiting one full interval.
        try:
            heartbeat_once()
        except Exception as exc:
            logger.debug("Status heartbeat baseline tick failed: %s", exc)

        _heartbeat_thread = threading.Thread(
            target=_heartbeat_loop,
            args=(_stop_event, _interval_seconds),
            name="gateway-status-heartbeat",
            daemon=True,
        )
        _heartbeat_thread.start()

        logger.info(
            "[STATUS] Periodic runtime heartbeat started (interval: %ds)",
            int(_interval_seconds),
        )
        return True


def stop_status_heartbeat(timeout: float = 2.0) -> None:
    """Stop the heartbeat thread if it is running."""
    global _heartbeat_thread, _stop_event

    with _lock:
        if _stop_event is None or _heartbeat_thread is None:
            return

        _stop_event.set()
        thread = _heartbeat_thread
        _heartbeat_thread = None
        _stop_event = None

    try:
        thread.join(timeout=timeout)
    except Exception:
        pass

    logger.info("[STATUS] Periodic runtime heartbeat stopped")


def is_running() -> bool:
    """Whether the heartbeat thread is currently alive."""
    return _heartbeat_thread is not None and _heartbeat_thread.is_alive()
