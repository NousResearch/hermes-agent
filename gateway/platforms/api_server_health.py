"""Detailed authenticated API gateway health response."""

from __future__ import annotations

import os
import time

from aiohttp import web


async def handle_health_detailed(adapter) -> web.Response:
    """Build health data from runtime state and live adapter metrics."""
    from gateway.platforms.api_server import _hermes_version

    if getattr(adapter.gateway_runner, "control_only", False):
        from gateway.status import read_runtime_status
        runtime = read_runtime_status() or {}
        return web.json_response({
            "status": "control_only", "mode": "control_only",
            "platform": "hermes-agent", "version": _hermes_version(),
            "gateway_state": "control_only", "readiness": {"status": "control_only"},
            "exit_reason": runtime.get("exit_reason"), "pid": os.getpid(),
        })
    from gateway.status import (
        derive_gateway_busy, derive_gateway_drainable, normalize_updated_at, parse_active_agents,
        read_runtime_status)
    runtime = read_runtime_status() or {}
    gw_state = runtime.get("gateway_state")
    gw_active = parse_active_agents(runtime.get("active_agents", 0))
    platforms = runtime.get("platforms", {})
    if not isinstance(platforms, dict):
        platforms = {}
    platforms = dict(platforms)
    api_status = adapter._api_server_status_payload(heartbeat_at=time.time())
    api_platform = dict(platforms.get("api_server", {}))
    api_platform.setdefault("state", "connected" if adapter.is_connected else "disconnected")
    api_platform["metrics"] = api_status
    platforms["api_server"] = api_platform
    active_api_runs, process_depth, active_delegations = adapter._readiness_work_counts()
    from gateway.run import _resolve_gateway_model
    from gateway.readiness import collect_runtime_readiness
    readiness = collect_runtime_readiness(
        configured_model=_resolve_gateway_model(), runtime_status=runtime,
        active_api_runs=active_api_runs, process_completion_queue_depth=process_depth,
        active_delegations=active_delegations)
    return web.json_response({
        "status": readiness["status"], "readiness": readiness, "platform": "hermes-agent",
        "version": _hermes_version(), "gateway_state": gw_state,
        "platforms": platforms,
        "api_server": api_status,
        "metrics_today": api_status["metrics_today"],
        "last_heartbeat": api_status["last_heartbeat"],
        "active_agents": gw_active,
        "gateway_busy": derive_gateway_busy(
            gateway_running=True, gateway_state=gw_state, active_agents=gw_active),
        "gateway_drainable": derive_gateway_drainable(
            gateway_running=True, gateway_state=gw_state),
        "exit_reason": runtime.get("exit_reason"),
        "updated_at": normalize_updated_at(runtime.get("updated_at")), "pid": os.getpid()})
