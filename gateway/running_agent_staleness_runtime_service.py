"""Helpers for evicting stale gateway running-agent entries."""

from __future__ import annotations

import os
import time
from typing import Any, MutableMapping


def _safe_float(value: Any, default: float) -> float:
    """Best-effort float conversion that never raises for mocks or bad values."""
    try:
        if value is None or isinstance(value, bool):
            raise TypeError("invalid numeric value")
        return float(value)
    except Exception:
        return default


def evict_stale_gateway_running_agent(
    *,
    session_key: str,
    running_agents: MutableMapping[str, Any],
    running_agents_ts: MutableMapping[str, float],
    pending_agent_sentinel: Any,
    logger: Any,
    stale_timeout_seconds: float | None = None,
    now: float | None = None,
) -> bool:
    """Evict a stale running-agent entry when it is idle or ancient enough.

    Active turns can legitimately run for a long time, so we prefer inactivity
    timeout over wall-clock age. A wall-clock escape hatch remains for
    pathological cases where the agent object stopped reporting activity.
    """

    stale_ts = running_agents_ts.get(session_key, 0)
    if session_key not in running_agents or not stale_ts:
        return False

    stale_timeout = stale_timeout_seconds
    if stale_timeout is None:
        stale_timeout = float(os.getenv("HERMES_AGENT_TIMEOUT", 1800))

    now_ts = time.time() if now is None else now
    stale_age = now_ts - stale_ts
    stale_agent = running_agents.get(session_key)

    stale_idle = 0.0
    stale_detail = ""
    if stale_agent and hasattr(stale_agent, "get_activity_summary"):
        try:
            activity = stale_agent.get_activity_summary()
            if isinstance(activity, dict):
                stale_idle = _safe_float(
                    activity.get("seconds_since_activity"),
                    float("inf"),
                )
                stale_detail = (
                    f" | last_activity={activity.get('last_activity_desc', 'unknown')} "
                    f"({stale_idle:.0f}s ago) "
                    f"| iteration={activity.get('api_call_count', 0)}/{activity.get('max_iterations', 0)}"
                )
        except Exception:
            pass

    wall_ttl = (
        max(stale_timeout * 10, 7200)
        if stale_timeout > 0
        else float("inf")
    )
    should_evict = (
        stale_agent is not pending_agent_sentinel
        and (
            (stale_timeout > 0 and stale_idle >= stale_timeout)
            or stale_age > wall_ttl
        )
    )
    if not should_evict:
        return False

    logger.warning(
        "Evicting stale _running_agents entry for %s "
        "(age: %.0fs, idle: %.0fs, timeout: %.0fs)%s",
        session_key[:30],
        stale_age,
        stale_idle,
        stale_timeout,
        stale_detail,
    )
    del running_agents[session_key]
    running_agents_ts.pop(session_key, None)
    return True
