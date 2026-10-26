"""Policy helpers for Slack Socket Mode health checks."""

from __future__ import annotations

def normalize_connected_state(adapter, connected: bool, *, now: float):
    """Treat an initial no-session state as unknown until the SDK's connect grace expires.

    The SDK retains a session object after disconnect, so only a fresh client with no
    session can receive grace. Present-but-disconnected sessions fail immediately.
    """
    if connected:
        return True
    client = getattr(getattr(adapter, "_handler", None), "client", None)
    if (
        client is None
        or getattr(client, "closed", False)
        or getattr(client, "current_session", None) is not None
    ):
        return False
    started = getattr(adapter, "_socket_handler_started_monotonic", None)
    interval = getattr(client, "ping_interval", None)
    first_ping_grace = getattr(adapter, "_socket_first_ping_grace_s", None)
    if not isinstance(started, (int, float)):
        return False
    if not isinstance(interval, (int, float)) or interval <= 0:
        return False
    if not isinstance(first_ping_grace, (int, float)) or first_ping_grace < 0:
        return False
    grace = max(first_ping_grace, interval * 2)
    return None if now - started < grace else False


def ping_pong_is_stale(adapter, *, now_monotonic: float, now_wall: float) -> bool:
    """Check Socket Mode heartbeat freshness while preserving its first-ping grace."""
    client = getattr(getattr(adapter, "_handler", None), "client", None)
    if client is None:
        return False
    interval = getattr(client, "ping_interval", None)
    if not isinstance(interval, (int, float)) or interval <= 0:
        return False
    last = getattr(client, "last_ping_pong_time", None)
    if last is None:
        started = getattr(adapter, "_socket_handler_started_monotonic", None)
        grace = max(getattr(adapter, "_socket_first_ping_grace_s", 60.0), interval * 2)
        return started is not None and now_monotonic - started > grace
    if not isinstance(last, (int, float)):
        return False
    return now_wall - last > interval * getattr(adapter, "_socket_ping_stale_factor", 4)
