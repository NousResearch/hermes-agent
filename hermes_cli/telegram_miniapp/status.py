"""Safe read-only status snapshot for the Telegram Mini App sidecar."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from gateway.status import (
    derive_gateway_busy,
    derive_gateway_drainable,
    get_running_pid,
    get_runtime_status_running_pid,
    parse_active_agents,
    read_runtime_status,
)


_ALLOWED_GATEWAY_STATES = {
    "starting",
    "running",
    "draining",
    "stopping",
    "stopped",
    "startup_failed",
    "degraded",
}


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _coarse_state(value: Any) -> str:
    if isinstance(value, str) and value in _ALLOWED_GATEWAY_STATES:
        return value
    return "unknown"


def build_status_snapshot(*, hermes_home_configured: bool | None = None) -> dict[str, Any]:
    """Build the allowlisted M2 status JSON shape.

    The return value deliberately avoids paths, PIDs, command lines, raw runtime
    records, logs, env values, and provider config.
    """
    runtime = read_runtime_status()
    if not isinstance(runtime, dict):
        runtime = {}

    live_pid = get_running_pid() or get_runtime_status_running_pid(runtime)
    gateway_running = bool(live_pid)
    gateway_state = _coarse_state(runtime.get("gateway_state"))
    active_agents = parse_active_agents(runtime.get("active_agents"))

    if hermes_home_configured is True:
        hermes_home = "configured"
    elif hermes_home_configured is False:
        hermes_home = "missing"
    else:
        hermes_home = "unknown"

    return {
        "ok": True,
        "updated_at": _now_iso(),
        "hermes_home": hermes_home,
        "gateway": {
            "running": gateway_running,
            "state": gateway_state,
            "busy": derive_gateway_busy(
                gateway_running=gateway_running,
                gateway_state=gateway_state,
                active_agents=active_agents,
            ),
            "drainable": derive_gateway_drainable(
                gateway_running=gateway_running,
                gateway_state=gateway_state,
            ),
            "active_agents": active_agents,
            "restart_requested": bool(runtime.get("restart_requested")),
        },
        "miniapp": {
            "mode": "local-read-only",
            "actions_enabled": False,
            "public_exposure": False,
        },
    }
